import torch


def token_logprobs(model, sequences, attention_mask, prompt_len, return_entropy=False, temperature=1.0,
                   completion_logits=False):
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    T = sequences.shape[1] - prompt_len
    logits = model(input_ids=sequences, attention_mask=attention_mask, position_ids=position_ids,
                   logits_to_keep=T + 1 if completion_logits else 0).logits      # 0: every position, as before
    logits = logits[:, -T - 1:-1, :].float() / temperature               # same distribution the rollout sampled from
    targets = sequences[:, prompt_len:]
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if not return_entropy:
        return logp
    with torch.no_grad():                                                # logged only, no gradient
        entropy = -(logp_all.exp() * logp_all).sum(-1)                   # H(p) per token over the vocab
    return logp, entropy