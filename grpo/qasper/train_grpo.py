import argparse
import os
import random
import time

import torch

from core.train import grpo_step
from toy.train_toy import load_model, log_row
from qasper.taskdata import build_examples
from qasper.reward import reward
from qasper.evaluate import run_eval, summarise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/sft_1.7b/merged")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--questions", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--out", default="runs/grpo_1.7b")
    p.add_argument("--a", type=float, default=0.1) 
    p.add_argument("--lr", type=float, default=2e-5) 
    p.add_argument("--norm_len", default=None)       
    args = p.parse_args()
    if args.norm_len not in (None, "batch"):
        args.norm_len = float(args.norm_len)
    args.k, args.beta, args.eps_clip, args.temperature, args.lora_r = 8, 0.04, 0.2, 1.0, 16
    args.max_new_tokens, args.micro_batch, args.updates_per_batch, args.max_grad_norm = 320, 4, 1, 1.0
    args.gen_batch = 2
    return args


def step_stats(batch, examples, k):
    rows = [e for e in examples for _ in range(k)]
    outs = [(e.target, reward(t, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps)[1])
            for t, e in zip(batch["texts"], rows)]
    ans = [o for target, o in outs if target == "answer"]
    abst = [o for target, o in outs if target == "abstain"]
    return {"reward": batch["rewards"].mean().item(),
            "fabricated": sum(o == "fabricated" for _, o in outs) / len(outs),
            "abstain_on_answerable": ans.count("abstain") / max(len(ans), 1),
            "abstain_on_abstain": abst.count("abstain") / max(len(abst), 1)}


def evaluate(model, tok, dev):
    m = summarise(run_eval(model, tok, dev, batch_size=16, max_new_tokens=384))
    return {"eval_" + k: v for k, v in m.items() if isinstance(v, float)}


def main():
    args = parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    tok, model = load_model(args)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.0)

    train, _ = build_examples("train", tok)
    dev = rng.sample(build_examples("dev", tok)[0], 200)
    rng.shuffle(train)
    fmt = lambda tok, e: e.prompt
    rew = lambda text, e: reward(text, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps, a=args.a)[0]

    os.makedirs(args.out, exist_ok=True)
    log = open(f"{args.out}/log.jsonl", "w")
    log_row(log, {"step": 0, **evaluate(model, tok, dev)})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        start = (step - 1) * args.questions % len(train)
        examples = train[start:start + args.questions]
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