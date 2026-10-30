import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qasper.taskdata import build_examples
from qasper.reward import reward, parse, answer_score, ANSWER_RE, ABSTAIN, norm_ws
from qasper.abstain import subtype, twins


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
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
    ap.add_argument("--no_twins", action="store_true")                  # skip the abstain-twin paired test
    return ap.parse_args()


def load_model(name, adapter, dtype="fp32"):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16).to("cuda")
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


@torch.no_grad()
def abstain_logps(model, tok, examples, batch_size=4):
    """log P(abstain completion | prompt) at temperature 1: ranks prompts by how much the model wants to abstain."""
    comp = tok(f"<answer>{ABSTAIN}</answer><|im_end|>", add_special_tokens=False,
               return_tensors="pt")["input_ids"].to(model.device)
    C = comp.shape[1]
    order = sorted(range(len(examples)), key=lambda i: examples[i].n_tokens)
    out = [None] * len(examples)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tok([examples[i].prompt for i in idx], return_tensors="pt", padding=True).to(model.device)
        seqs = torch.cat([batch["input_ids"], comp.expand(len(idx), -1)], dim=1)
        am = torch.cat([batch["attention_mask"], torch.ones_like(comp).expand(len(idx), -1)], dim=1)
        pos = (am.cumsum(dim=1) - 1).clamp(min=0)
        logits = model(input_ids=seqs, attention_mask=am, position_ids=pos, logits_to_keep=C + 1).logits[:, :-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, seqs[:, -C:].unsqueeze(-1)).squeeze(-1).sum(dim=1)
        for i, v in zip(idx, lp.tolist()):
            out[i] = v
    return out


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / max(len(pos) * len(neg), 1)


def summarise(records):
    def rate(rs, cond):
        return sum(map(cond, rs)) / max(len(rs), 1)
    twin_rs = [r for r in records if r.get("twin")]
    records = [r for r in records if not r.get("twin")]
    ans = [r for r in records if r["target"] == "answer"]
    abst = [r for r in records if r["target"] == "abstain"]
    answered = [r for r in ans if r["outcome"] == "answer"]
    out = {
        "n": len(records), "n_answer_target": len(ans), "n_abstain_target": len(abst),
        "mean_reward": sum(r["score"] for r in records) / max(len(records), 1),
        "outcomes": {k: v / len(records) for k, v in Counter(r["outcome"] for r in records).items()},
        "answerable_answered": rate(ans, lambda r: r["outcome"] == "answer"),
        "answerable_abstained": rate(ans, lambda r: r["outcome"] == "abstain"),
        "f1_when_answered": sum(r["f1"] for r in answered) / max(len(answered), 1),
        "f1_overall": sum(r.get("f1", 0.0) for r in ans) / max(len(ans), 1),
        "gold_citation_when_answered": rate(answered, lambda r: r["cited_gold"]),
        "abstain_target_abstained": rate(abst, lambda r: r["outcome"] == "abstain"),
        "f1_overall": sum(r.get("f1", 0.0) for r in ans) / max(len(ans), 1),
        "raw_f1_answerable": sum(r["raw_f1"] for r in ans) / max(len(ans), 1),
    }
    if records and "subtype" in records[0]:
        kinds = sorted({r["subtype"] for r in abst})
        out["abstain_target_abstained_clean"] = rate([r for r in abst if r["subtype"] != "contaminated"],
                                                     lambda r: r["outcome"] == "abstain")
        out["abstained_by_subtype"] = {s: rate([r for r in abst if r["subtype"] == s], lambda r: r["outcome"] == "abstain")
                                       for s in kinds}
        if "logp_abstain" in records[0]:                                 # ranking, separate from how often it abstains
            out["auc_vs_answerable"] = {s: auc([r["logp_abstain"] for r in abst if r["subtype"] == s],
                                               [r["logp_abstain"] for r in ans]) for s in kinds}
    if twin_rs:                                                          # same question, gold paragraphs removed
        orig = {r["question_id"]: r for r in ans}
        pairs = [(orig[t["question_id"]], t) for t in twin_rs]
        out["twin_abstained"] = rate(twin_rs, lambda r: r["outcome"] == "abstain")
        out["twin_pair_correct"] = rate(pairs, lambda p: p[0]["outcome"] != "abstain" and p[1]["outcome"] == "abstain")
        if "logp_abstain" in twin_rs[0]:
            out["twin_pair_auc"] = rate(pairs, lambda p: p[1]["logp_abstain"] > p[0]["logp_abstain"])
    return out

def run_eval(model, tok, examples, batch_size, max_new_tokens):
    order = sorted(range(len(examples)), key=lambda i: examples[i].n_tokens)
    records = [None] * len(examples)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        texts = generate(model, tok, [examples[i].prompt for i in idx], max_new_tokens)
        for i, text in zip(idx, texts):
            e = examples[i]
            score, outcome = reward(text, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps)
            rec = {"question_id": e.question.question_id, "target": e.target, "kind": e.question.kind,
                   "subtype": subtype(e), "score": score, "outcome": outcome, "output": text}
            m = ANSWER_RE.search(text)
            raw = norm_ws(m.group(1)) if m else None
            if e.target == "answer":
                rec["raw_f1"] = answer_score(raw, e.question) if raw and raw != ABSTAIN else 0.0
            if e.target == "answer" and outcome == "answer":
                quotes, answer = parse(text)
                rec["f1"] = answer_score(answer, e.question)
                rec["cited_gold"] = any(p in e.gold_ps for p, _ in quotes)
            records[i] = rec
        print(f"{min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return records

def main():
    args = parse_args()
    tok, model = load_model(args.model, args.adapter, args.dtype)
    examples, _ = build_examples(args.split, tok, few_shot=args.few_shot)
    if args.n:
        examples = examples[:args.n]
    twin_ex = [] if args.no_twins else [t for _, t in twins(examples, args.split, tok, few_shot=args.few_shot)]
    records = run_eval(model, tok, examples + twin_ex, args.batch_size, args.max_new_tokens)
    for r, lp in zip(records, abstain_logps(model, tok, examples + twin_ex)):
        r["logp_abstain"] = lp
    for r in records[len(examples):]:
        r["twin"] = True

    tag = args.model.rstrip("/").split("/")[-1] + ("_fewshot" if args.few_shot else "")
    if args.adapter:
        tag += "_" + args.adapter.strip("/").replace("/", "_")
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{tag}_{args.split}_outputs.jsonl", "w") as f:
        for r in records[:len(examples)]:
            f.write(json.dumps(r) + "\n")
    if twin_ex:                                                          # own file, so outputs.jsonl lines up with older runs
        with open(f"{args.out}/{tag}_{args.split}_twins.jsonl", "w") as f:
            for r in records[len(examples):]:
                f.write(json.dumps(r) + "\n")
    metrics = summarise(records)
    json.dump(metrics, open(f"{args.out}/{tag}_{args.split}_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()