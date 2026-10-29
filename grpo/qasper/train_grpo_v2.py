import argparse
import json
import os
import random
import time
from collections import Counter

import torch

from core.train import grpo_step
from core.trace import Tracer, traced_step
from toy.train_toy import load_model, log_row
from qasper.taskdata import build_examples
from qasper.reward import reward, ABSTAIN
from qasper.sft_targets import build_target
from qasper.evaluate import run_eval, summarise
from qasper.abstain import twins

ABSTAIN_SHARE = 0.17                # abstain prompts in the training mix, as in train_sft_v2
UNANSWERABLE_SHARE = 0.40           # of the abstain prompts; the rest are twins


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/sft_v2_1.7b/merged")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--questions", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--out", default="runs/grpo_v2_1.7b")
    p.add_argument("--a", type=float, default=0.1) 
    p.add_argument("--lr", type=float, default=2e-5) 
    p.add_argument("--norm_len", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])  # fp32: LoRA's small changes aren't rounded away
    p.add_argument("--trace", action="store_true")              # log the exact split of each update (core/trace.py)
    p.add_argument("--trace_probe", type=int, default=32)       # dev prompts per probe set
    args = p.parse_args()
    if args.norm_len not in (None, "batch"):
        args.norm_len = float(args.norm_len)
    args.k, args.beta, args.eps_clip, args.lora_r = 8, 0.04, 0.2, 16
    args.max_new_tokens, args.micro_batch, args.updates_per_batch, args.max_grad_norm = 320, 4, 1, 1.0
    args.gen_batch = 2
    return args


def step_stats(batch, examples, k):
    rows = [e for e in examples for _ in range(k)]
    outs = [(e.target, reward(t, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps)[1])
            for t, e in zip(batch["texts"], rows)]
    ans = [o for target, o in outs if target == "answer"]
    abst = [o for target, o in outs if target == "abstain"]
    twin = [o for (target, o), e in zip(outs, rows) if target == "abstain" and e.question.label != "unanswerable"]
    return {"reward": batch["rewards"].mean().item(),
            "fabricated": sum(o == "fabricated" for _, o in outs) / len(outs),
            "abstain_on_answerable": ans.count("abstain") / max(len(ans), 1),
            "abstain_on_abstain": abst.count("abstain") / max(len(abst), 1),
            "abstain_on_twin": twin.count("abstain") / len(twin) if twin else None}


def evaluate(model, tok, dev):
    m = summarise(run_eval(model, tok, dev, batch_size=16, max_new_tokens=384))
    return {"eval_" + k: v for k, v in m.items() if isinstance(v, float)}


def build_mix(tok, rng):
    """Answer targets with a buildable target, unanswerable questions, and abstain twins of answer targets.
    Abstains are ABSTAIN_SHARE of the prompts, UNANSWERABLE_SHARE of them unanswerable; twins are drawn per
    question kind in proportion to its answers. A twin and its original form one unit, shuffled together."""
    train, _ = build_examples("train", tok)
    answers = [e for e in train if e.target == "answer" and build_target(e)[0] is not None]
    unans = [e for e in train if e.question.label == "unanswerable"]
    n_abstain = round(ABSTAIN_SHARE / (1 - ABSTAIN_SHARE) * len(answers))
    n_unans = min(round(UNANSWERABLE_SHARE * n_abstain), len(unans))
    pairs, kinds = twins(answers, "train", tok, drop_long=True), Counter(e.question.kind for e in answers)
    chosen = []
    for k, n in sorted(kinds.items()):
        ps = [p for p in pairs if p[0].question.kind == k]
        chosen += rng.sample(ps, min(round((n_abstain - n_unans) * n / len(answers)), len(ps)))
    paired = {id(e) for e, _ in chosen}
    units = [[e] for e in answers if id(e) not in paired] + [[e] for e in rng.sample(unans, n_unans)] \
        + [list(p) for p in chosen]
    rng.shuffle(units)
    print(f"  {len(answers)} answers, {n_unans} of {len(unans)} unanswerable, {len(chosen)} of {len(pairs)} twins "
          f"{dict(Counter(e.question.kind for e, _ in chosen))}: abstain share "
          f"{(n_unans + len(chosen)) / (len(answers) + n_unans + len(chosen)):.3f}")
    return units


def pack(units, size):
    """Units flattened in order, except a unit that would straddle two steps of `size` prompts waits for the
    next step. Only whole steps are kept, so every epoch lines up the same way."""
    out, held = [], []
    for u in units:
        if len(u) > size - len(out) % size:
            held.append(u)
            continue
        out += u
        while held and len(held[0]) <= size - len(out) % size:
            out += held.pop(0)
    return out[:len(out) - len(out) % size]


def main():
    args = parse_args()
    if args.dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = True                    # TF32 inputs, fp32 outputs: speed without rounding outputs
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    rng = random.Random(0)
    tok, model = load_model(args)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.0)

    dev_all = build_examples("dev", tok)[0]
    dev = rng.sample(dev_all, 200)                                       # drawn first: the same dev sample as train_grpo
    train = pack(build_mix(tok, rng), args.questions)
    fmt = lambda tok, e: e.prompt
    rew = lambda text, e: reward(text, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps, a=args.a)[0]
    lab = lambda text, e: (e.target, reward(text, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps, a=args.a)[1])

    os.makedirs(args.out, exist_ok=True)
    log = open(f"{args.out}/log.jsonl", "w")
    tracer = None
    if args.trace:
        prng = random.Random(1)
        probe_sets = {
            "answerable": [e.prompt for e in prng.sample([e for e in dev_all if e.target == "answer"], args.trace_probe)],
            "unanswerable": [e.prompt for e in prng.sample([e for e in dev_all if e.question.label == "unanswerable"],
                                                           args.trace_probe)],
        }
        tracer = Tracer(model, tok, args.model, probe_sets, f"<answer>{ABSTAIN}</answer><|im_end|>", args.temperature)
        trace = open(f"{args.out}/trace.jsonl", "w")
    log_row(log, {"step": 0, **evaluate(model, tok, dev)})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        start = (step - 1) * args.questions % len(train)
        examples = train[start:start + args.questions]
        if tracer:
            batch, stats, rec = traced_step(model, tok, optimizer, [(e, e) for e in examples], fmt, rew, lab, args, tracer)
            trace.write(json.dumps({"step": step, **rec}) + "\n")
            trace.flush()
            stats.update({f"probe_{s}": v for s, v in rec["f_after"].items()})
        else:
            batch, stats = grpo_step(model, tok, optimizer, [(e, e) for e in examples], fmt, rew, args)
        row = {"step": step, **step_stats(batch, examples, args.k), **stats,
               "seconds": time.time() - t0, "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        if step % args.eval_every == 0:
            row.update(evaluate(model, tok, dev))
            model.save_pretrained(f"{args.out}/step{step}")
        log_row(log, row)

    model.save_pretrained(f"{args.out}/final")


if __name__ == "__main__":
    main()
