"""Step 2 check: the Reader on the GGUF backend, end to end, over many dev rows.

Everything so far validated the two model calls IN ISOLATION (verifier gate: abstain scores;
generator gate: f1|ans). Reader.ask() on the GGUF path has never run: retrieval -> prompt ->
abstain_score -> generate -> rules -> tier. This runs it and checks three things:

 1. PROMPT IDENTITY. reader.ask() must build byte-identical prompts to build_examples(), else the
    gates and the tier calibration do not transfer (they were all measured on those prompts).
 2. GENERATION AGREEMENT. raw output vs the saved plain-Q4 gate generations
    (runs/gguf/gen_q4km_dev_full.jsonl). Same GGUF + greedy + same prompt => should be identical.
    Divergence here means the reader's call differs from the gate's (n_ctx, stop, tokenisation).
 3. TIER DISTRIBUTION. the realised HIGH/MEDIUM/LOW/ABSTAIN mix and per-tier correctness, which is
    the first test of the Q4 cutpoints THROUGH the generator and the rules rather than on verifier
    scores alone. recalibrate_tiers.py predicted HIGH ~75% right / ~6% wrong, coverage ~63%.

Scores f1 with the SAME strict scorer the fixed gate uses, so numbers are comparable to
parsescripts/rescore_gguf_gate.py.

    sbatch slurm/check_reader_gguf.sh                    # 300 rows
    sbatch slurm/check_reader_gguf.sh --n 0              # ALL answerable dev rows
"""
import argparse
import json

import numpy as np

from qasper.prompt import format_prompt
from qasper.reader import Reader
from qasper.reward import ABSTAIN, answer_score, format_ok, parse, quote_check
from qasper.taskdata import build_examples


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reader_4b_gguf.json")
    ap.add_argument("--q4_gen", default="runs/gguf/gen_q4km_dev_full.jsonl",
                    help="saved plain-Q4 gate generations, for output agreement")
    ap.add_argument("--n", type=int, default=300, help="rows to check (0 = all dev rows)")
    ap.add_argument("--save", default="runs/gguf/reader_gguf_check.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def score_strict(output, e):
    """Identical to the fixed gate's score_output: parse -> format rules -> quote grounding -> f1.

    Returns (f1_or_None, outcome). f1 is None on an UNANSWERABLE row: answer_score() is max() over
    the gold answers and an unanswerable question has none, so calling it raises ValueError. The
    gate never hit this because it sampled answerable rows only; evaluate.py guards both of its
    answer_score calls with `if e.target == "answer"` and this does the same. For those rows the
    rule OUTCOME is still meaningful (did it correctly abstain?), the f1 is not.
    """
    parsed = parse(output)
    if parsed is None:
        return 0.0, "format"
    quotes, answer = parsed
    if answer == ABSTAIN:
        return (0.0 if e.target == "answer" else None), "abstain"
    if not format_ok(quotes, answer, False, e.p_to_idx):
        return 0.0, "format"
    if quote_check(quotes, e.paper, e.p_to_idx) in ("fabricated", "miscited"):
        return 0.0, "quote"
    if e.target != "answer":
        return None, "answer"          # answered an unanswerable row: wrong, but f1 is undefined
    return answer_score(answer, e.question), "answer"


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    assert cfg.get("backend") == "gguf", f"{args.config} is not a gguf config"
    print(f"config {args.config}  tiers {cfg['tiers']}", flush=True)

    reader = Reader(cfg["sft_path"], cfg["adapter_path"], cfg["tiers"], cfg["k"],
                    cfg.get("device", "auto"), cfg["dtype"], backend="gguf",
                    verifier_gguf=cfg["verifier_gguf"], generator_gguf=cfg["generator_gguf"],
                    n_gpu_layers=cfg.get("n_gpu_layers", -1),
                    n_ctx=cfg.get("n_ctx", 8192), verifier_n_ctx=cfg.get("verifier_n_ctx", 4096))
    print(f"backend {reader.backend}  device {reader.device}  "
          f"gen n_ctx {reader.gguf.generator.n_ctx()}  ver n_ctx {reader.gguf.verifier.n_ctx()}", flush=True)

    examples, _ = build_examples("dev", reader.tok)
    saved = {}
    try:
        for l in open(args.q4_gen):
            r = json.loads(l)
            saved[r["question_id"]] = r
        print(f"loaded {len(saved)} saved Q4 gate generations for comparison", flush=True)
    except FileNotFoundError:
        print(f"NOTE: {args.q4_gen} not found - skipping output-agreement check", flush=True)

    idx = list(range(len(examples)))
    if args.n:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(idx, size=min(args.n, len(idx)), replace=False).tolist())
    print(f"checking {len(idx)} rows", flush=True)

    rows, cur_paper = [], None
    out_f = open(args.save, "w") if args.save else None
    for n, i in enumerate(idx):
        e = examples[i]
        if e.paper.paper_id != cur_paper:          # re-index only on a paper change (embedding is the cost)
            reader.index(e.paper)
            cur_paper = e.paper.paper_id
        r = reader.ask(e.question.question)
        d = r["debug"]
        qid = e.question.question_id

        prompt_ok = format_prompt(reader.tok, d["context"], e.question.question) == e.prompt
        f1, outcome = score_strict(d["raw_output"], e)
        rec = {"question_id": qid, "target": e.target, "kind": e.question.kind,
               "prompt_identical": bool(prompt_ok), "n_prompt_tokens": d["n_prompt_tokens"],
               "status": r["status"], "confidence": r.get("confidence"), "reason": d["reason"],
               "reader_outcome": d["outcome"], "strict_outcome": outcome,
               "f1": None if f1 is None else float(f1),
               "verifier_score": float(d["verifier_score"]), "raw_output": d["raw_output"]}
        if qid in saved:
            rec["gate_output_identical"] = d["raw_output"] == saved[qid]["output"]
            rec["gate_f1"] = saved[qid]["q4_f1"]
        rows.append(rec)
        if out_f:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()                          # per row: a long run must survive a cancellation
        if (n + 1) % 10 == 0:
            ans = [x for x in rows if x["status"] == "answer" and x["f1"] is not None]
            print(f"  {n+1}/{len(idx)}  answered {sum(x['status'] == 'answer' for x in rows)}  "
                  f"meanF1|shown-answerable {np.mean([x['f1'] for x in ans]) if ans else 0:.3f}", flush=True)
    if out_f:
        out_f.close()

    print("\n=== 1. prompt identity (gates + tiers only transfer if these match) ===")
    bad = [r for r in rows if not r["prompt_identical"]]
    print(f"identical {len(rows)-len(bad)}/{len(rows)}" + (f"   MISMATCHES: {[r['question_id'][:10] for r in bad[:5]]}" if bad else "   OK"))
    over = [r for r in rows if r["n_prompt_tokens"] > 4096]
    print(f"prompts over the verifier's 4096 ctx: {len(over)}"
          + (f"  (max {max(r['n_prompt_tokens'] for r in rows)})" if rows else ""))

    cmp_rows = [r for r in rows if "gate_output_identical" in r]
    if cmp_rows:
        same = sum(r["gate_output_identical"] for r in cmp_rows)
        print(f"\n=== 2. generation agreement vs saved Q4 gate ({len(cmp_rows)} overlapping rows) ===")
        print(f"byte-identical {same}/{len(cmp_rows)}")
        if same < len(cmp_rows):
            d = [r for r in cmp_rows if not r["gate_output_identical"]]
            print(f"  mean f1 reader {np.mean([r['f1'] for r in d]):.3f} vs gate {np.mean([r['gate_f1'] for r in d]):.3f} on the {len(d)} differing rows")
            print(f"  e.g. {d[0]['question_id'][:10]}\n    reader: {d[0]['raw_output'][:160]!r}")

    print("\n=== 3. tier distribution (first test of the Q4 cutpoints end to end) ===")
    ansrows = [r for r in rows if r["target"] == "answer"]
    print(f"{'tier':10s} {'n':>5s} {'%':>6s} {'P(right)':>9s} {'P(wrong)':>9s} {'meanF1':>7s}")
    for tier in ("high", "medium", "low"):
        t = [r for r in rows if r["confidence"] == tier]
        if not t:
            print(f"{tier:10s} {0:5d}")
            continue
        # P(right)/P(wrong)/meanF1 are over the ANSWERABLE rows in the tier (f1 is undefined for the
        # rest); the n and % columns are over every row that landed in the tier.
        ta = [r for r in t if r["f1"] is not None]
        if not ta:
            print(f"{tier:10s} {len(t):5d} {100*len(t)/len(rows):5.1f}%  (no answerable rows)")
            continue
        right = sum(r["f1"] >= 0.6 for r in ta) / len(ta)
        wrong = sum(r["f1"] <= 0.2 for r in ta) / len(ta)
        print(f"{tier:10s} {len(t):5d} {100*len(t)/len(rows):5.1f}% {right:9.2f} {wrong:9.2f} "
              f"{np.mean([r['f1'] for r in ta]):7.3f}   (n_ans={len(ta)})")
    ab = [r for r in rows if r["status"] == "abstain"]
    print(f"{'abstain':10s} {len(ab):5d} {100*len(ab)/max(len(rows),1):5.1f}%")
    shown = [r for r in rows if r["status"] == "answer"]
    print(f"\ncoverage (shown/all)        {len(shown)}/{len(rows)} = {100*len(shown)/max(len(rows),1):.1f}%   "
          f"(recalibrate_tiers predicted ~63%)")
    if ansrows:
        shown_ans = [r for r in ansrows if r["status"] == "answer"]
        print(f"coverage on answerable only {len(shown_ans)}/{len(ansrows)} = {100*len(shown_ans)/len(ansrows):.1f}%")
    shown_f1 = [r["f1"] for r in shown if r["f1"] is not None]
    print(f"mean f1 | shown answerable  {np.mean(shown_f1) if shown_f1 else 0:.4f}  (n={len(shown_f1)})")
    wrong_shown = [r for r in shown if r["target"] != "answer"]
    print(f"shown on UNANSWERABLE rows  {len(wrong_shown)}  (these should have abstained)")
    from collections import Counter
    print(f"abstain reasons             {dict(Counter(r['reason'] for r in ab))}")
    print(f"reader outcomes             {dict(Counter(r['reader_outcome'] for r in rows))}")
    if args.save:
        print(f"\nsaved {len(rows)} rows to {args.save}")


if __name__ == "__main__":
    main()
