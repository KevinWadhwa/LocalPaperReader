"""Score the FULL dev (or test) split through a GGUF verifier and save per-row abstain scores.

This is the gate's scoring loop (parsescripts/gguf_verifier_gate.py) with the --n cap removed:
it recomputes s = -logP(abstain) through the quantised verifier for every row, in the SAME
order the fp32 npz was saved in, and writes an npz with the Q4 scores alongside the metadata
needed to recalibrate the confidence tiers (target, raw_f1, outcome, shown, twin, ids).

Scoring is byte-identical to the gate: one batched eval(prompt+comp) with logits_all=True,
raw logits -> log_softmax -> gather -> sum. (Verified against logits_all=False / get_logits:
same logits to 0.0; and the completion-position logits projection is ~0.1s, so there is no
faster faithful path — the cost is prompt prefill, which cannot be cached across distinct rows.)

    PYTHONPATH=. python parsescripts/gguf_score_dev.py \
        --gguf runs/gguf/verifier-q4km.gguf \
        --npz  runs/verifier/merged_runs_grpo_v2_4b_step80_dev.npz \
        --sft  runs/sft_v2_4b/merged \
        --split dev --n_ctx 4096 --n_gpu_layers -1 \
        --out  runs/gguf/verifier_q4km_dev_scores.npz
"""
import argparse
import time

import numpy as np
from llama_cpp import Llama
from transformers import AutoTokenizer

from qasper.abstain import twins
from qasper.reward import ABSTAIN
from qasper.taskdata import build_examples

COMPLETION = f"<answer>{ABSTAIN}</answer><|im_end|>"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--npz", required=True, help="fp32 reference npz (for row order + metadata)")
    ap.add_argument("--sft", required=True, help="HF model dir, for the reference tokenizer")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=None, help="cap rows (default: all); mainly for a quick test")
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n_gpu_layers", type=int, default=-1, help="-1 = all layers on GPU (Mac Metal)")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def quant_logp(llm, comp_ids, prompt_ids):
    """log P(comp | prompt): batched eval of prompt+completion with logits_all=True, then at each
    position predicting a completion token apply log_softmax over the raw logits and sum. Exactly
    evaluate.abstain_logps, and exactly gguf_verifier_gate.quant_logp."""
    seq = prompt_ids + comp_ids
    llm.reset()
    llm.eval(seq)
    total = 0.0
    for j, tok_id in enumerate(comp_ids):
        row = np.array(llm.scores[len(prompt_ids) + j - 1], dtype=np.float64)
        row = row - row.max()
        total += row[tok_id] - np.log(np.exp(row).sum())
    return total


def main():
    args = parse_args()
    print("loading tokenizer + examples...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.sft)
    tok.padding_side = "left"

    d = np.load(args.npz, allow_pickle=True)
    fp32_s = -d["logp_abstain"].astype(np.float64)

    # rebuild prompts in the SAME order the npz was saved in (examples then twins)
    examples, _ = build_examples(args.split, tok)
    pairs = twins(examples, args.split, tok)
    rows = examples + [t for _, t in pairs]
    assert len(rows) == len(fp32_s), f"row mismatch: {len(rows)} rebuilt vs {len(fp32_s)} in npz"

    print(f"loading GGUF {args.gguf} (n_ctx={args.n_ctx}, gpu_layers={args.n_gpu_layers})...", flush=True)
    llm = Llama(model_path=args.gguf, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                logits_all=True, n_threads=args.threads, verbose=False)
    hf_comp = tok(COMPLETION, add_special_tokens=False)["input_ids"]
    gg_comp = llm.tokenize(COMPLETION.encode(), add_bos=False, special=True)
    assert hf_comp == gg_comp, f"completion token ids differ!\n HF: {hf_comp}\n GGUF: {gg_comp}"
    print(f"completion tokenises identically ({len(hf_comp)} tokens, incl <|im_end|>={hf_comp[-1]})", flush=True)

    n = len(rows) if args.n is None else min(args.n, len(rows))
    quant_s = np.full(n, np.nan)
    skipped = 0
    t0 = time.time()
    for i in range(n):
        p_ids = llm.tokenize(rows[i].prompt.encode(), add_bos=True, special=True)
        if len(p_ids) + len(gg_comp) > args.n_ctx:
            skipped += 1
            continue
        quant_s[i] = -quant_logp(llm, gg_comp, p_ids)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            rate = el / (i + 1)
            eta = rate * (n - i - 1)
            print(f"  scored {i + 1}/{n}  ({rate:.1f}s/row, ETA {eta/60:.0f} min)", flush=True)
    print(f"skipped {skipped} rows longer than n_ctx={args.n_ctx}", flush=True)

    # save Q4 scores + the metadata the tier recalibration needs (same keys/order as the fp32 npz)
    np.savez(args.out,
             quant_s=quant_s,                               # s = -logP(abstain) through the GGUF
             fp32_s=fp32_s[:n],                             # the fp32 reference, same rows
             question_id=d["question_id"][:n], paper_id=d["paper_id"][:n], twin=d["twin"][:n],
             target=d["target"][:n], subtype=d["subtype"][:n], outcome=d["outcome"][:n],
             shown=d["shown"][:n], raw_f1=d["raw_f1"][:n].astype(np.float64),
             gguf=args.gguf, split=args.split, n_ctx=args.n_ctx)
    ok = ~np.isnan(quant_s)
    print(f"\nsaved {args.out}: {n} rows ({ok.sum()} scored, {skipped} skipped)", flush=True)
    print(f"total {(time.time() - t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
