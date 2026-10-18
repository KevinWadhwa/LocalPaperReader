import argparse
import json
import os
import random
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from core.train import grpo_step
from core.rollout import rollout
from toy.task import make_problem, make_dataset, format_prompt
from toy.reward import reward, parse_answer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--difficulty", type=int, default=2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--questions", type=int, default=16)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--updates_per_batch", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--eval_n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/toy_grpo")
    return p.parse_args()

def load_model(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.eval()
    return tok, model

def summarise(batch, k):
    r = batch["rewards"]
    flat = r.view(-1)
    formatted = torch.tensor([parse_answer(t) is not None for t in batch["texts"]])
    sums = r.sum(dim=1)
    return {
        "reward": flat.mean().item(),
        "format_rate": formatted.float().mean().item(),
        "acc_when_formatted": flat[formatted].mean().item() if formatted.any() else 0.0,
        "mixed_groups": ((sums > 0) & (sums < k)).float().mean().item(),
        "mean_length": batch["comp_mask"].sum(dim=1).float().mean().item(),
    }

def evaluate(model, tok, problems, args):
    batch = rollout(model, tok, problems, 4, format_prompt, reward,
                    args.temperature, args.max_new_tokens)
    return {"eval_" + key: value for key, value in summarise(batch, 4).items()}

def log_row(log, row):
    log.write(json.dumps(row) + "\n")
    log.flush()
    print(" ".join(f"{name}={val:.3f}" if isinstance(val, float) else f"{name}={val}"
                   for name, val in row.items()), flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok, model = load_model(args)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    eval_problems = make_dataset(12345, args.eval_n, args.difficulty)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "w")

    log_row(log, {"step": 0, **evaluate(model, tok, eval_problems, args)})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        problems = [make_problem(rng, args.difficulty) for _ in range(args.questions)]
        batch, loss_stats = grpo_step(model, tok, optimizer, problems, format_prompt, reward, args)

        row = {"step": step, **summarise(batch, args.k), **loss_stats}
        row["seconds"] = time.time() - t0
        row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

        if step % args.eval_every == 0:
            row.update(evaluate(model, tok, eval_problems, args))
            model.save_pretrained(os.path.join(args.out, "lora"))

        log_row(log, row)

    model.save_pretrained(os.path.join(args.out, "lora"))
    log.close()


if __name__ == "__main__":
    main()