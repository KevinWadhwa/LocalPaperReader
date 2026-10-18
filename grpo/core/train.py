import torch

from core.rollout import rollout
from core.logps import token_logprobs
from core.grpo import group_advantages, grpo_loss


def chunked_logprobs(model, seqs, amask, prompt_len, micro_batch):
    """token_logprobs over all rows, a few rows at a time. Call inside no_grad."""
    parts = []
    for i in range(0, seqs.shape[0], micro_batch):
        parts.append(token_logprobs(model, seqs[i:i + micro_batch], amask[i:i + micro_batch], prompt_len))
    return torch.cat(parts, dim=0)


def grpo_step(model, tok, optimizer, problems, format_prompt, reward_fn, cfg):
    # 1. Sample K answers per question with the current policy
    batch = rollout(model, tok, problems, cfg.k, format_prompt, reward_fn,
                    cfg.temperature, cfg.max_new_tokens)
    seqs, amask = batch["sequences"], batch["attention_mask"]
    cmask, P = batch["comp_mask"], batch["prompt_len"]

    # 2. One advantage per answer
    adv = group_advantages(batch["rewards"]).to(seqs.device)

    # 3-4. Fixed log-probs: old (current weights) and ref (adapters off)
    with torch.no_grad():
        logp_old = chunked_logprobs(model, seqs, amask, P, cfg.micro_batch)
        with model.disable_adapter():
            logp_ref = chunked_logprobs(model, seqs, amask, P, cfg.micro_batch)

    # 5. Update, a few rows at a time
    N = seqs.shape[0]
    totals = {"loss": 0.0, "clip_fraction": 0.0, "kl": 0.0, "ratio": 0.0}
    for _ in range(cfg.updates_per_batch):
        optimizer.zero_grad()
        for i in range(0, N, cfg.micro_batch):
            s = slice(i, i + cfg.micro_batch)
            logp_new = token_logprobs(model, seqs[s], amask[s], P)
            loss, stats = grpo_loss(logp_new, logp_old[s], logp_ref[s], adv[s], cmask[s],
                                    cfg.eps_clip, cfg.beta)
            weight = logp_new.shape[0] / N
            (loss * weight).backward()

            share = weight / cfg.updates_per_batch
            totals["loss"] += loss.item() * share
            for key, value in stats.items():
                totals[key] += value * share
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

    totals["grad_norm"] = grad_norm.item()
    return batch, totals