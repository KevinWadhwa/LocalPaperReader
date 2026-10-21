import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qasper.taskdata import build_examples
from qasper.reward import reward, parse, answer_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--few_shot", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--out", default="runs/qasper_eval")
    return ap.parse_args()


def load_model(name, adapter):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens):
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens,
                         temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    return tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)


def summarise(records):
    def rate(rs, cond):
        return sum(map(cond, rs)) / max(len(rs), 1)
    ans = [r for r in records if r["target"] == "answer"]
    abst = [r for r in records if r["target"] == "abstain"]
    answered = [r for r in ans if r["outcome"] == "answer"]
    return {
        "n": len(records), "n_answer_target": len(ans), "n_abstain_target": len(abst),
        "mean_reward": sum(r["score"] for r in records) / max(len(records), 1),
        "outcomes": {k: v / len(records) for k, v in Counter(r["outcome"] for r in records).items()},
        "answerable_answered": rate(ans, lambda r: r["outcome"] == "answer"),
        "answerable_abstained": rate(ans, lambda r: r["outcome"] == "abstain"),
        "f1_when_answered": sum(r["f1"] for r in answered) / max(len(answered), 1),
        "f1_overall": sum(r.get("f1", 0.0) for r in ans) / max(len(ans), 1),
        "gold_citation_when_answered": rate(answered, lambda r: r["cited_gold"]),
        "abstain_target_abstained": rate(abst, lambda r: r["outcome"] == "abstain"),
    }


def main():
    args = parse_args()
    tok, model = load_model(args.model, args.adapter)
    examples, _ = build_examples(args.split, tok, few_shot=args.few_shot)
    if args.n:
        examples = examples[:args.n]
    order = sorted(range(len(examples)), key=lambda i: examples[i].n_tokens)
    records = [None] * len(examples)

    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        texts = generate(model, tok, [examples[i].prompt for i in idx], args.max_new_tokens)
        for i, text in zip(idx, texts):
            e = examples[i]
            score, outcome = reward(text, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps)
            rec = {"question_id": e.question.question_id, "target": e.target, "kind": e.question.kind,
                   "score": score, "outcome": outcome, "output": text}
            if e.target == "answer" and outcome == "answer":
                quotes, answer = parse(text)
                rec["f1"] = answer_score(answer, e.question)
                rec["cited_gold"] = any(p in e.gold_ps for p, _ in quotes)
            records[i] = rec
        print(f"{min(start + args.batch_size, len(order))}/{len(order)}", flush=True)

    tag = args.model.rstrip("/").split("/")[-1] + ("_fewshot" if args.few_shot else "")
    if args.adapter:
        tag += "_" + args.adapter.strip("/").replace("/", "_")
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{tag}_{args.split}_outputs.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    metrics = summarise(records)
    json.dump(metrics, open(f"{args.out}/{tag}_{args.split}_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()