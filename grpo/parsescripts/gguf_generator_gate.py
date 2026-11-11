"""GGUF generator-quality gate: does the quantised GENERATOR keep answer quality (f1 | answered)?

The verifier gate checked the abstain SCORE (teacher-forced, exact). This checks the GENERATOR,
which produces text, so the comparison is aggregate quality, not per-example identity: Q4 greedy
on Metal picks different tokens from fp32 greedy on CUDA on close calls, which is expected and
fine — what must hold is that mean f1|answered does not drop materially.

Generates answers with the GGUF generator (greedy, the same decoding as evaluate.generate's
do_sample=False) on a seeded STRATIFIED sample of answerable dev rows, scores each against the
full reader contract (parse -> format rules -> quote grounding -> answer_score), and compares
mean f1|ans to the fp32 generator RESCORED FROM ITS STORED TEXT BY THE SAME SCORER, with a
bootstrap CI. Both sides must go through score_output; see its docstring for why reading
evaluate.py's stored raw_f1 for the fp32 side is wrong.

PASS BUDGET (fixed before looking at any output):
    mean f1|ans(Q4) >= f1|ans(fp32) - 0.03             no material mean drop
    95% bootstrap CI upper bound on the drop <= 0.05    sampling noise can't hide a bigger drop
The thresholds predate the scorer fix below and are UNCHANGED; what changed is that both sides are
now scored by the same function, so the comparison measures quantisation rather than a difference
between two metrics. f1|ans(fp32) is the fp32 side's stored text rescored on the SAME sampled rows
(paired). The stored f1_when_answered is no longer used as the reference: it is rule-free, so it is
not comparable to these numbers.

    PYTHONPATH=. python parsescripts/gguf_generator_gate.py \
        --gguf runs/gguf/generator-q4km.gguf \
        --sft  runs/sft_v2_4b/merged \
        --fp32_outputs runs/qasper_eval/grpo_v2_4b/merged_runs_grpo_v2_4b_step80_dev_outputs.jsonl \
        --fp32_metrics runs/qasper_eval/grpo_v2_4b/merged_runs_grpo_v2_4b_step80_dev_metrics.json \
        --n 150 --n_ctx 4096 --n_gpu_layers -1
"""
import argparse
import json

import numpy as np
from llama_cpp import Llama
from transformers import AutoTokenizer

from qasper.context import build_context  # noqa: F401  (ensures the qasper import chain is importable)
from qasper.reward import ABSTAIN, answer_score, format_ok, parse, quote_check
from qasper.taskdata import build_examples

MEAN_DROP_MAX, CI_DROP_MAX = 0.03, 0.05


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the GGUF GENERATOR (SFT+GRPO merged, adapter on)")
    ap.add_argument("--sft", required=True, help="HF dir for the tokenizer (prompts built identically)")
    ap.add_argument("--fp32_outputs", required=True,
                    help="fp32 generator dev outputs jsonl (per-row 'output' text, rescored here)")
    ap.add_argument("--fp32_metrics", default=None,
                    help="optional: fp32 dev metrics json, printed as provenance only "
                         "(f1_when_answered is rule-free, so it is NOT the gate reference)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=150, help="stratified sample size of answerable rows; 0 = ALL answerable")
    ap.add_argument("--save", default=None, help="optional jsonl: persist per-row Q4 output+f1 (reusable, e.g. overnight full-dev run)")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def stratified_sample(examples, rows, n, seed):
    """Sample n answerable rows, stratified by question.kind so extractive/yes_no/abstractive are
    represented in proportion. Returns the sampled indices (into examples/rows, same order).
    n == 0 means ALL answerable rows (the full-dev overnight run), in order."""
    ans = [i for i, r in enumerate(rows) if r["target"] == "answer"]
    if n == 0:
        return ans
    rng = np.random.default_rng(seed)
    kinds = {}
    for i in ans:
        kinds.setdefault(examples[i].question.kind, []).append(i)
    picked = []
    for k, idxs in kinds.items():
        take = max(1, round(n * len(idxs) / len(ans)))
        picked += list(rng.choice(idxs, size=min(take, len(idxs)), replace=False))
    return sorted(picked)[:n]


def score_output(output, example, p_to_idx):
    """Full reader contract for one generated answer: parse, enforce the format rules, enforce
    quote grounding, then token/exact f1 against the golds. A row that abstains or fails any rule
    scores 0 (it did not produce usable output).

    This is NOT evaluate.py's raw_f1. raw_f1 is deliberately rule-free (a bare ANSWER_RE grab) and
    answers "when the model commits, how good is the content?". Scoring the quantised side here
    while reading raw_f1 for the fp32 side compared two different metrics and manufactured a
    ~0.04 'quantisation drop' that was really the fp32 side being paid for format failures the
    quantised side was charged for. Both sides must go through THIS function."""
    parsed = parse(output)
    if parsed is None:
        return 0.0, "format"
    quotes, answer = parsed
    if answer == ABSTAIN:
        return 0.0, "abstain"
    if not format_ok(quotes, answer, False, p_to_idx):
        return 0.0, "format"
    if quote_check(quotes, example.paper, p_to_idx) in ("fabricated", "miscited"):
        return 0.0, "quote"
    return answer_score(answer, example.question), "answer"


def main():
    args = parse_args()
    print("loading tokenizer + dev examples...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.sft)
    tok.padding_side = "left"
    examples, _ = build_examples(args.split, tok)
    rows = [json.loads(l) for l in open(args.fp32_outputs)]
    assert [e.question.question_id for e in examples] == [r["question_id"] for r in rows], \
        "example order doesn't match the fp32 outputs jsonl"
    # kept only as provenance in the printout; NOT the gate reference (rule-free, see below)
    fp32_fans = json.load(open(args.fp32_metrics))["f1_when_answered"] if args.fp32_metrics else None

    idx = stratified_sample(examples, rows, args.n, args.seed)
    print(f"sampled {len(idx)} answerable rows (seed {args.seed})", flush=True)

    print(f"loading GGUF generator {args.gguf} (gpu_layers={args.n_gpu_layers})...", flush=True)
    llm = Llama(model_path=args.gguf, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                logits_all=False, n_threads=args.threads, verbose=False)

    q4_f1 = np.zeros(len(idx))
    # SYMMETRY: rescore the fp32 side's STORED TEXT through the same score_output the quantised
    # side uses. Reading the stored raw_f1 here instead would compare strict scoring against
    # rule-free scoring (see score_output's docstring).
    fp32_f1 = np.array([score_output(rows[i]["output"], examples[i], examples[i].p_to_idx)[0]
                        for i in idx], dtype=np.float64)
    save_f = open(args.save, "w") if args.save else None
    for j, i in enumerate(idx):
        # rebuild p_to_idx the same way evaluate scored this row: it is stored per-row? no — recompute
        # from the example's context mapping. build_examples stored the prompt; the quote rules need
        # p_to_idx, which build_context produced. Re-derive it from the example.
        p_to_idx = examples[i].p_to_idx
        # tokenise the prompt OURSELVES with special=True: create_completion's default is special=False,
        # which would turn the chat-template tokens (<|im_start|> etc.) into literal text. This matches
        # the verifier gate's validated tokenisation (add_bos=True, special=True).
        p_ids = llm.tokenize(examples[i].prompt.encode(), add_bos=True, special=True)
        out = llm.create_completion(p_ids, max_tokens=args.max_new_tokens,
                                    temperature=0.0, stop=["<|im_end|>"])
        text = out["choices"][0]["text"]
        q4_f1[j], outc = score_output(text, examples[i], p_to_idx)
        if save_f:
            # write per row, not at the end: generation is hours of compute and a crash or a
            # cancelled job at row 700 must not lose the text (it is rescorable without a GPU).
            save_f.write(json.dumps(
                {"question_id": examples[i].question.question_id, "output": text,
                 "q4_f1": float(q4_f1[j]), "outcome": outc, "fp32_f1": float(fp32_f1[j])}) + "\n")
            save_f.flush()
        if (j + 1) % 10 == 0:
            print(f"  generated {j + 1}/{len(idx)}  (running mean f1 Q4 {q4_f1[:j+1].mean():.3f} "
                  f"vs fp32 {fp32_f1[:j+1].mean():.3f})", flush=True)

    drop = fp32_f1.mean() - q4_f1.mean()
    # bootstrap the paired drop
    rng = np.random.default_rng(args.seed)
    boots = np.array([(fp32_f1[b].mean() - q4_f1[b].mean())
                      for b in (rng.integers(0, len(idx), len(idx)) for _ in range(2000))])
    ci = np.percentile(boots, [2.5, 97.5])

    print("\n=== GGUF generator-quality gate ===")
    print(f"sampled n              {len(idx)}")
    # The stored f1_when_answered is rule-free, a different metric from the strict scoring both
    # sides use here, so it is shown as provenance only and is never the comparison reference.
    if fp32_fans is not None:
        print(f"fp32 f1|ans (stored)   {fp32_fans:.4f}   rule-free, NOT the reference")
    print(f"fp32 f1|ans (rescored) {fp32_f1.mean():.4f}   (same rows, same scorer)")
    print(f"Q4   f1|ans            {q4_f1.mean():.4f}")
    print(f"paired drop (fp32-Q4)  {drop:+.4f}   95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"budget: mean drop <= {MEAN_DROP_MAX}, CI upper <= {CI_DROP_MAX}")
    ok = drop <= MEAN_DROP_MAX and ci[1] <= CI_DROP_MAX
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")

    if save_f:
        save_f.close()
        print(f"saved {len(idx)} per-row generations to {args.save} "
              "(reusable: rescore with parsescripts/rescore_gguf_gate.py, no GPU)", flush=True)


if __name__ == "__main__":
    main()
