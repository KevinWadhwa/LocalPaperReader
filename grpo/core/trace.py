"""Exact accounting of what moves the model's tendency to abstain during GRPO training.

Each step's gradient is split by row group (target x outcome) plus the KL term: rows of a group are
backpropagated together, so the parts sum to the gradient the optimiser steps on. AdamW's momentum is linear
in the clipped gradients, so one momentum buffer per group splits the real update exactly, using the
optimiser's own second-moment denominator. A part's effect on the probe (log P of the abstain completion on
fixed dev prompts) is the integral of the probe gradient along the real update path, dotted with that part,
so the effects add up to the measured change; the integral uses 2-point Gauss-Legendre, and the measured
change minus the sum of effects is logged as the residual. The probe runs on an fp32 copy of the model,
because one step's change is far below bf16 resolution; the bf16 value after the step is logged alongside.
The fp32 copy waits on the CPU between probes, so it never adds to the rollout or backward peak.
"""
import copy
from contextlib import contextmanager
from statistics import mean

import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from core.grpo import grpo_loss
from core.logps import token_logprobs
from core.rollout import rollout
from core.train import prepare

GL2 = [(0.5 - 0.5 / 3 ** 0.5, 0.5), (0.5 + 0.5 / 3 ** 0.5, 0.5)]        # Gauss-Legendre nodes, weights on [0, 1]


def kind(outcome):
    return outcome if outcome in ("abstain", "answer") else "penalised"


def rel_err(xs, ys):
    num = sum(((x - y).float() ** 2).sum() for x, y in zip(xs, ys))
    den = sum((y.float() ** 2).sum() for y in ys)
    return (num / den).sqrt().item() if den > 0 else 0.0


def norm(xs):
    return sum((x.float() ** 2).sum() for x in xs).sqrt().item()


def group_grads(model, batch, adv, logp_old, logp_ref, labels, cfg, params):
    """Gradient of the GRPO loss as one part per row group plus the KL term. The parts sum to the full
    gradient: the loss is a sum over rows, and grpo_loss splits exactly into its beta=0 and adv=0 parts."""
    seqs, amask, cmask, P = batch["sequences"], batch["attention_mask"], batch["comp_mask"], batch["prompt_len"]
    N = seqs.shape[0]
    norm_len = getattr(cfg, "norm_len", None)
    if norm_len == "batch":
        norm_len = cmask.sum().item() / N
    groups = {}
    for i, (target, outcome) in enumerate(labels):
        if adv[i] != 0:                                                  # zero advantage: surrogate gradient is 0
            groups.setdefault(f"{target}-target/{kind(outcome)}", []).append(i)
    passes = [(name, rows, adv, 0.0) for name, rows in groups.items()]
    passes.append(("kl", list(range(N)), torch.zeros_like(adv), cfg.beta))

    grads, totals = {}, {"loss": 0.0, "clip_fraction": 0.0, "kl": 0.0, "ratio": 0.0, "entropy": 0.0}
    for name, rows, a, beta in passes:
        for j in range(0, len(rows), cfg.micro_batch):
            idx = torch.tensor(rows[j:j + cfg.micro_batch], device=seqs.device)
            logp_new, entropy = token_logprobs(model, seqs[idx], amask[idx], P, return_entropy=True,
                                               temperature=cfg.temperature,
                                               completion_logits=getattr(cfg, "completion_logits", False))
            loss, stats = grpo_loss(logp_new, logp_old[idx], logp_ref[idx], a[idx], cmask[idx],
                                    cfg.eps_clip, beta, norm_len)
            weight = len(idx) / N
            (loss * weight).backward()
            totals["loss"] += loss.item() * weight
            if name == "kl":                                             # every row once
                stats["entropy"] = ((entropy * cmask[idx]).sum() / cmask[idx].sum()).item()
                for key in ("kl", "ratio", "entropy"):
                    totals[key] += stats[key] * weight
            else:                                                        # only rows with adv != 0 can clip
                totals["clip_fraction"] += stats["clip_fraction"] * weight
        grads[name] = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
        for p in params:
            p.grad = None
    return grads, totals


class Tracer:
    def __init__(self, model, tok, model_name, probe_sets, completion, temperature, batch_size=4):
        torch.set_float32_matmul_precision("highest")
        self.model, self.temperature, self.device = model, temperature, model.device
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for p in model.parameters() if p.requires_grad]
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.probe_model = get_peft_model(base, copy.deepcopy(model.peft_config["default"])).eval()
        self.probe_model.gradient_checkpointing_enable()
        self.probe_model.enable_input_require_grads()
        named = dict(self.probe_model.named_parameters())
        self.probe_params = [named[n] for n in names]

        self.completion = tok(completion, add_special_tokens=False)["input_ids"]
        self.C = len(self.completion)
        self.sets = {}
        for name, prompts in probe_sets.items():
            batches = []
            for i in range(0, len(prompts), batch_size):
                ids = [tok(p)["input_ids"] + self.completion for p in prompts[i:i + batch_size]]
                T = max(map(len, ids))
                seq = [[tok.pad_token_id] * (T - len(x)) + x for x in ids]
                am = [[0] * (T - len(x)) + [1] * len(x) for x in ids]
                batches.append((torch.tensor(seq, device=model.device), torch.tensor(am, device=model.device)))
            self.sets[name] = batches
        self.m = {}
        self.f_prev = None

    @contextmanager
    def gpu(self):
        self.probe_model.to(self.device)                                 # same Parameter objects, bitwise copy
        try:
            yield
        finally:
            self.probe_model.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def logp(self, model, seq, am):
        """log P(abstain completion | prompt) per row, at the sampling temperature."""
        pos = (am.cumsum(dim=1) - 1).clamp(min=0)
        logits = model(input_ids=seq, attention_mask=am, position_ids=pos,
                       logits_to_keep=self.C + 1).logits[:, :-1].float() / self.temperature
        return torch.log_softmax(logits, dim=-1).gather(-1, seq[:, -self.C:].unsqueeze(-1)).squeeze(-1).sum(dim=1)

    @torch.no_grad()
    def probe(self, model):
        model.eval()
        return {name: [x for seq, am in batches for x in self.logp(model, seq, am).tolist()]
                for name, batches in self.sets.items()}

    @torch.no_grad()
    def load(self, theta):
        for q, t in zip(self.probe_params, theta):
            q.copy_(t)

    def probe_at(self, theta):
        self.load(theta)
        return self.probe(self.probe_model)

    def probe_grad(self, theta):
        """Gradient of each probe set's mean log P(abstain) with respect to the LoRA weights, at theta."""
        self.load(theta)
        self.probe_model.train()                                         # gradient checkpointing only runs in train mode
        out = {}
        for name, batches in self.sets.items():
            n = sum(seq.shape[0] for seq, _ in batches)
            for q in self.probe_params:
                q.grad = None
            for seq, am in batches:
                (self.logp(self.probe_model, seq, am).sum() / n).backward()
            out[name] = [q.grad.detach().clone() for q in self.probe_params]
        for q in self.probe_params:
            q.grad = None
        self.probe_model.eval()
        return out

    @torch.no_grad()
    def split_update(self, optimizer, grads, coef, before):
        """One part of the real AdamW update per group; returns the parts and how far their sum (and the sum
        of the momentum buffers) is from what the optimiser actually did."""
        group = optimizer.param_groups[0]
        assert group["weight_decay"] == 0 and not group["amsgrad"] and not group["maximize"]
        (b1, b2), lr, eps = group["betas"], group["lr"], group["eps"]
        t = float(optimizer.state[self.params[0]]["step"])
        step_size = lr / (1 - b1 ** t)
        bc2_sqrt = (1 - b2 ** t) ** 0.5
        denom = [optimizer.state[p]["exp_avg_sq"].sqrt() / bc2_sqrt + eps for p in self.params]
        for name in grads:
            self.m.setdefault(name, [torch.zeros_like(p) for p in self.params])
        for name, m in self.m.items():                                   # groups absent this step still decay
            g = grads.get(name)
            for i, mi in enumerate(m):
                mi.mul_(b1)
                if g is not None:
                    mi.add_(g[i] * coef, alpha=1 - b1)
        parts = {name: [-step_size * mi / d for mi, d in zip(m, denom)] for name, m in self.m.items()}
        actual = [p.detach() - b for p, b in zip(self.params, before)]
        m_err = rel_err([sum(ms) for ms in zip(*self.m.values())],
                        [optimizer.state[p]["exp_avg"] for p in self.params])
        d_err = rel_err([sum(ds) for ds in zip(*parts.values())], actual)
        return parts, actual, m_err, d_err, t

    def record(self, optimizer, grads, coef, before, labels, adv, batch):
        parts, actual, m_err, d_err, t = self.split_update(optimizer, grads, coef, before)
        after = [p.detach().clone() for p in self.params]
        f0 = self.f_prev
        effects = {name: {s: 0.0 for s in self.sets} for name in parts}
        with self.gpu():
            for node, w in GL2:                                          # ∫ ∇f(before + s·Δθ) · part ds
                g = self.probe_grad([b + node * a for b, a in zip(before, actual)])
                for name, d in parts.items():
                    for s in self.sets:
                        effects[name][s] += w * sum((gi * di).sum() for gi, di in zip(g[s], d)).item()
                del g
            f1 = self.probe_at(after)
        f1_bf16 = self.probe(self.model)
        self.f_prev = f1
        total = {s: mean(f1[s]) - mean(f0[s]) for s in f1}
        residual = {s: total[s] - sum(e[s] for e in effects.values()) for s in total}

        P, comp = batch["prompt_len"], batch["sequences"][:, batch["prompt_len"]:]
        lens = batch["comp_mask"].sum(dim=1).tolist()
        rows, noncanonical = {}, 0
        for i, ((target, outcome), a, r) in enumerate(zip(labels, adv.tolist(), batch["rewards"].view(-1).tolist())):
            s = rows.setdefault(f"{target}-target/{outcome}", {"n": 0, "n_adv": 0, "adv_sum": 0.0,
                                                               "reward_sum": 0.0, "len_sum": 0})
            s["n"] += 1
            s["n_adv"] += a != 0
            s["adv_sum"] += a
            s["reward_sum"] += r
            s["len_sum"] += lens[i]
            if outcome == "abstain":                                     # is the probe's token sequence the one sampled?
                noncanonical += comp[i, :lens[i]].tolist() != self.completion
        return {
            "adam_step": t, "clip_coef": coef.item(), "momentum_split_err": m_err, "update_split_err": d_err,
            "update_norm": norm(actual), "f_before": {s: mean(v) for s, v in f0.items()},
            "f_after": {s: mean(v) for s, v in f1.items()},
            "f_after_bf16": {s: mean(v) for s, v in f1_bf16.items()},
            "total": total, "effects": effects, "residual": residual,
            "grad_norms": {name: norm(g) * coef.item() for name, g in grads.items()},
            "update_norms": {name: norm(d) for name, d in parts.items()},
            "rows": rows, "abstain_noncanonical": noncanonical, "probe_after": f1,
        }


def traced_step(model, tok, optimizer, problems, format_prompt, reward_fn, label_fn, cfg, tracer):
    """grpo_step with the gradient split by group and the update accounted for by Tracer.
    label_fn(text, answer) -> (target, outcome)."""
    assert cfg.updates_per_batch == 1
    model.eval()
    batch = rollout(model, tok, problems, cfg.k, format_prompt, reward_fn,
                    cfg.temperature, cfg.max_new_tokens, getattr(cfg, "gen_batch", None))
    adv, logp_old, logp_ref = prepare(model, batch, cfg)
    labels = [label_fn(t, a) for t, a in zip(batch["texts"], [a for _, a in problems for _ in range(cfg.k)])]

    model.train()
    optimizer.zero_grad()
    grads, totals = group_grads(model, batch, adv, logp_old, logp_ref, labels, cfg, tracer.params)
    with torch.no_grad():
        for i, p in enumerate(tracer.params):
            p.grad = sum(g[i] for g in grads.values())                   # the gradient the optimiser steps on
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    coef = torch.clamp(cfg.max_grad_norm / (grad_norm + 1e-6), max=1.0)  # same factor clip_grad_norm_ applied
    before = [p.detach().clone() for p in tracer.params]
    if tracer.f_prev is None:
        with tracer.gpu():
            tracer.f_prev = tracer.probe_at(before)
    optimizer.step()
    model.eval()
    totals["grad_norm"] = grad_norm.item()
    return batch, totals, tracer.record(optimizer, grads, coef, before, labels, adv, batch)
