import torch

from core.rollout import rollout
from core.logps import token_logprobs
from core.grpo import group_advantages, grpo_loss


def chunked_logprobs(model, seqs, amask, prompt_len, micro_batch, temperature=1.0):
    """token_logprobs over all rows, a few rows at a time. Call inside no_grad."""
    parts = []
    for i in range(0, seqs.shape[0], micro_batch):
        parts.append(token_logprobs(model, seqs[i:i + micro_batch], amask[i:i + micro_batch], prompt_len,
                                    temperature=temperature))
    return torch.cat(parts, dim=0)


def prepare(model, batch, cfg):
    """Steps 2-4: advantages, plus the fixed old (current weights) and ref (adapters off) log-probs."""
    seqs, amask, P = batch["sequences"], batch["attention_mask"], batch["prompt_len"]
    adv = group_advantages(batch["rewards"]).to(seqs.device)
    with torch.no_grad():
        logp_old = chunked_logprobs(model, seqs, amask, P, cfg.micro_batch, cfg.temperature)
        with model.disable_adapter():
            logp_ref = chunked_logprobs(model, seqs, amask, P, cfg.micro_batch, cfg.temperature)
    return adv, logp_old, logp_ref


def accumulate_grads(model, batch, adv, logp_old, logp_ref, cfg):
    """Backward over every row, a few rows at a time; gradients add up in .grad.
    Returns the loss stats averaged over rows. Does not zero grads or step."""
    seqs, amask = batch["sequences"], batch["attention_mask"]
    cmask, P = batch["comp_mask"], batch["prompt_len"]
    N = seqs.shape[0]
    norm_len = getattr(cfg, "norm_len", None)
    if norm_len == "batch": #for per batch
        norm_len = cmask.sum().item()/N #total tokens in batch/answers
    totals = {"loss": 0.0, "clip_fraction": 0.0, "kl": 0.0, "ratio": 0.0, "entropy": 0.0}
    for i in range(0, N, cfg.micro_batch):
        s = slice(i, i + cfg.micro_batch)
        logp_new, entropy = token_logprobs(model, seqs[s], amask[s], P, return_entropy=True,
                                           temperature=cfg.temperature)
        loss, stats = grpo_loss(logp_new, logp_old[s], logp_ref[s], adv[s], cmask[s],
                                cfg.eps_clip, cfg.beta, norm_len)
        stats["entropy"] = ((entropy * cmask[s]).sum() / cmask[s].sum()).item()  # mean over real answer tokens
        weight = logp_new.shape[0] / N
        (loss * weight).backward()

        totals["loss"] += loss.item() * weight
        for key, value in stats.items():
            totals[key] += value * weight
    return totals


def grpo_step(model, tok, optimizer, problems, format_prompt, reward_fn, cfg):
    # 1. Sample K answers per question with the current policy (eval: no dropout)
    model.eval()
    batch = rollout(model, tok, problems, cfg.k, format_prompt, reward_fn,
                    cfg.temperature, cfg.max_new_tokens, getattr(cfg, "gen_batch", None))

    # 2-4. One advantage per answer; fixed old and ref log-probs
    adv, logp_old, logp_ref = prepare(model, batch, cfg)

    # 5. Update (train: dropout on if configured)
    model.train()
    totals = {}
    for _ in range(cfg.updates_per_batch):
        optimizer.zero_grad()
        stats = accumulate_grads(model, batch, adv, logp_old, logp_ref, cfg)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value / cfg.updates_per_batch
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
    model.eval()                                                         # leave in eval for callers (evaluate, save)

    totals["grad_norm"] = grad_norm.item()
    return batch, totals
