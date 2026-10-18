import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from toy.task import make_dataset, format_prompt, system_prompt
from toy.reward import parse_answer, reward

def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda")
    model.eval()
    
    return tok, model

@torch.inference_mode()
def generate(tok, model, prompts, k, sample, temperature, max_new_tokens):
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    if sample:
        settings = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                        num_return_sequences=k)
    else:
        settings = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = model.generate(**enc, **settings, repetition_penalty=1.0,
                         max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)

    new = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(new, skip_special_tokens=True)

    eos = model.generation_config.eos_token_id
    eos = torch.tensor(eos if isinstance(eos, list) else [eos], device=new.device)
    is_eos = torch.isin(new, eos)
    finished = is_eos.any(dim=1)
    first_eos = is_eos.int().argmax(dim=1)
    lengths = torch.where(finished, first_eos + 1, torch.full_like(first_eos, new.shape[1]))
    return texts, lengths.tolist(), finished.tolist()

def run(tok, model, problems, k, sample, temperature, max_new_tokens, batch_size):
    texts, lengths, finished = [], [], []
    for i in range(0, len(problems), batch_size):
        chunk = problems[i:i + batch_size]
        prompts = [format_prompt(tok, q) for q, _ in chunk]
        t, l, f = generate(tok, model, prompts, k, sample, temperature, max_new_tokens)
        texts += t
        lengths += l
        finished += f
    return texts, lengths, finished

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    torch.manual_seed(0)
    tok, model = load_model(args.model)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
    problems = make_dataset(args.seed, args.n, args.difficulty)
    answers = [a for _, a in problems]

    # Greedy: 1 answer per problem
    g_texts, g_len, g_fin = run(tok, model, problems, 1, False, None,
                                args.max_new_tokens, args.batch_size)
    g_rewards = [reward(t, a) for t, a in zip(g_texts, answers)]

    # Sampled: K answers per problem (fewer problems per batch, so total sequences stay the same)
    s_texts, s_len, s_fin = run(tok, model, problems, args.k, True, args.temperature,
                                args.max_new_tokens, max(args.batch_size // args.k, 1))
    s_answers = [a for a in answers for _ in range(args.k)]
    s_rewards = [reward(t, a) for t, a in zip(s_texts, s_answers)]

    groups = [s_rewards[i:i + args.k] for i in range(0, len(s_rewards), args.k)]
    mixed = [0 < sum(g) < len(g) for g in groups]

    results = {
        "config": vars(args),
        "system_prompt": system_prompt,
        "greedy_accuracy": sum(g_rewards) / len(g_rewards),
        "greedy_format_rate": sum(parse_answer(t) is not None for t in g_texts) / len(g_texts),
        "sampled_accuracy": sum(s_rewards) / len(s_rewards),
        "sampled_format_rate": sum(parse_answer(t) is not None for t in s_texts) / len(s_texts),
        "mixed_group_fraction": sum(mixed) / len(mixed),
        "mean_length": sum(s_len) / len(s_len),
        "max_length": max(s_len),
        "truncated_rate": 1 - sum(s_fin) / len(s_fin),
    }
    for key, val in results.items():
        if isinstance(val, float):
            print(f"{key:22s} {val:.3f}")

    kinds = {
        "correct": lambda t, a: reward(t, a) == 1.0,
        "wrong": lambda t, a: parse_answer(t) is not None and reward(t, a) == 0.0,
        "bad_format": lambda t, a: parse_answer(t) is None,
    }
    for name, test in kinds.items():
        for t, a in zip(s_texts, s_answers):
            if test(t, a):
                print(f"\n===== {name} (true answer {a}) =====\n{t}")
                break

    os.makedirs("runs", exist_ok=True)
    tag = "_trained" if args.adapter else ""
    path = f"runs/eval_base_d{args.difficulty}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()