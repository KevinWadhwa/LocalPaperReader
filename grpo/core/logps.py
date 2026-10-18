import torch


def token_logprobs(model, sequences, attention_mask, prompt_len):
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    logits = model(input_ids=sequences, attention_mask=attention_mask,
                   position_ids=position_ids).logits
    logits = logits[:, prompt_len - 1:-1, :].float()
    targets = sequences[:, prompt_len:]
    logp_all = torch.log_softmax(logits, dim=-1)
    return logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)