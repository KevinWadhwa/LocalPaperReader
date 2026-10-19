import torch


def group_advantages(rewards, eps=1e-8, scale=True):
    """rewards: [B, K] -> advantages: [B*K], in the same row order as the rollout."""
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean
    if scale:
        std = rewards.std(dim=1, keepdim=True)
        adv = adv / (std + eps)
    return adv.view(-1)

def grpo_loss(logp_new, logp_old, logp_ref, adv, mask, eps_clip=0.2, beta=0.04):
    ratio = torch.exp(logp_new - logp_old)                               # ρ for every token
    A = adv.unsqueeze(1)                                                 # same advantage for all tokens of an answer
    surrogate = torch.min(ratio * A, torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A)
    log_q = logp_ref - logp_new
    kl = torch.exp(log_q) - log_q - 1                                    # "k3" KL estimate per token
    # ×ratio: value unchanged (ratio = 1), but adds kl·∇log p, the term from tokens being sampled
    # from the policy, so the gradient is ∇KL(p‖q) rather than ∇KL(q‖p)
    per_token = surrogate - beta * kl * ratio
    per_answer = (per_token * mask).sum(1) / mask.sum(1)                 # average over each answer's tokens
    clipped = ((ratio > 1 + eps_clip) & (A > 0)) | ((ratio < 1 - eps_clip) & (A < 0))
    clip_fraction = (clipped.float() * mask).sum() / mask.sum()
    stats = {
        "clip_fraction": clip_fraction.item(),
        "kl": ((kl * mask).sum() / mask.sum()).item(),                   # mean KL over real answer tokens
        "ratio": ((ratio * mask).sum() / mask.sum()).item(),             # mean ratio over real answer tokens
    }
    return -per_answer.mean(), stats                                     # minus: optimisers minimise