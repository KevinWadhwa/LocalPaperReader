import torch

def make_comp_mask(comp_ids, eos_ids):
    """
    1 1 for each token inlcuding eos, 0 for all that follow it.
    """

    is_eos = torch.isin(comp_ids, eos_ids) #creates grid of true false for eos tokens
    T = comp_ids.shape[1] #length of answer area
    has_eos = is_eos.any(dim=1) #is there any eos
    first_eos = is_eos.int().argmax(dim=1) #pos of first eos, argmax returns first 1 pos
    last_kept = torch.where(has_eos, first_eos, torch.full_like(first_eos,T)) #last token kept, fulllike makes tensor of same shape type and device as firsteos filled with T
    pos = torch.arange(T, device=comp_ids.device).unsqueeze(0) #makes list of size T
    return (pos <= last_kept.unsqueeze(1)).long()


@torch.no_grad()
def sample_comp(model, tok, prompts, k, temperature=1.0, max_new_tokens=512):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        do_sample = True, temperature = temperature, top_p = 1.0, top_k = 0,
        repetition_penalty = 1.0, num_return_sequences=k,
        max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
    )

    prompt_len = enc["input_ids"].shape[1]
    comp_ids = out[:, prompt_len:]

    eos = model.generation_config.eos_token_id
    eos_ids = torch.tensor(eos if isinstance(eos,list) else [eos], device=out.device)
    comp_mask = make_comp_mask(comp_ids, eos_ids)

    prompt_mask = enc["attention_mask"].repeat_interleave(k, dim=0)
    attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)

    texts = tok.batch_decode(comp_ids, skip_special_tokens=True)
    return {
        "sequences": out,
        "prompt_len": prompt_len,
        "completion_mask": completion_mask,
        "attention_mask": attention_mask,
        "texts": texts,
    }

def rollout(model, tok, problems, k, format_prompt, reward_fn,
            temperature=1.0, max_new_tokens=512):
    """Sample K answers per problem and score them.

    problems:      list of (question, answer)
    format_prompt: function (tok, question) -> prompt string
    reward_fn:     function (completion_text, answer) -> float
    """
    prompts = [format_prompt(tok, q) for q, _ in problems]
    batch = sample_comp(model, tok, prompts, k, temperature, max_new_tokens)

    answers = [a for _, a in problems for _ in range(k)]          # each answer repeated K times
    rewards = [reward_fn(t, a) for t, a in zip(batch["texts"], answers)]
    batch["rewards"] = torch.tensor(rewards, dtype=torch.float32).view(len(problems), k)
    return batch