"""GGUF verifier-score gate: does the quantised verifier reproduce the fp32 abstain score?

Recomputes s = -logP(abstain) through a GGUF verifier for the SAME dev prompts the fp32
run used (rebuilt via build_examples/twins, so prompts are identical by construction), and
compares to the stored fp32 logp_abstain in the dev .npz. Faithful port of evaluate.abstain_logps:
same completion string, raw logits -> log_softmax -> gather -> sum (NOT the logits_all path).

PASS THRESHOLDS (fixed before looking at any output):
    Spearman rho(fp32 s, quant s) >= 0.95       ranking survives quantisation
    paired AUC drop (fp32 - quant) <= 0.02       correct/wrong separation holds
Also asserts the GGUF tokenises the completion to the SAME ids as HF (tokenizer-drift guard).

    PYTHONPATH=. <localpaperreader python> parsescripts/gguf_verifier_gate.py \
        --gguf runs/gguf/verifier-q4km.gguf \
        --npz  runs/verifier/merged_runs_grpo_v2_4b_step80_dev.npz \
        --sft  runs/sft_v2_4b/merged
"""
import argparse

import numpy as np
from llama_cpp import Llama
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from qasper.abstain import twins
from qasper.reward import ABSTAIN
from qasper.taskdata import build_examples

RHO_MIN, AUC_DROP_MAX = 0.95, 0.02
COMPLETION = f"<answer>{ABSTAIN}</answer><|im_end|>"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--sft", required=True, help="HF model dir, for the reference tokenizer")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=400, help="score the first N rows (dev is ~1387)")
    ap.add_argument("--n_ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n_gpu_layers", type=int, default=0, help="-1 = all layers on GPU (Mac Metal)")
    return ap.parse_args()


def quant_logp(llm, comp_ids, prompt_ids):
    """log P(comp | prompt) through the GGUF: eval prompt+completion with logits_all=True,
    then at each position that PREDICTS a completion token apply log_softmax over the raw
    logits and sum (exactly as evaluate.abstain_logps does). logits_all is required here:
    with it off, 0.3.36 does not populate per-position logits (they come back all-zero)."""
    seq = prompt_ids + comp_ids
    llm.reset()
    llm.eval(seq)
    total = 0.0
    for j, tok_id in enumerate(comp_ids):
        row = np.array(llm.scores[len(prompt_ids) + j - 1], dtype=np.float64)   # predicts comp token j
        row = row - row.max()
        total += row[tok_id] - np.log(np.exp(row).sum())
    return total


def main():
    args = parse_args()
    print("loading tokenizer + dev examples...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.sft)
    tok.padding_side = "left"

    d = np.load(args.npz, allow_pickle=True)
    fp32_s = -d["logp_abstain"].astype(np.float64)          # s = -logP(abstain)
    target, raw_f1 = d["target"], d["raw_f1"].astype(np.float64)

    # rebuild prompts in the SAME order the npz was saved in (examples then twins)
    examples, _ = build_examples(args.split, tok)
    pairs = twins(examples, args.split, tok)
    rows = examples + [t for _, t in pairs]
    assert len(rows) == len(fp32_s), f"row mismatch: {len(rows)} rebuilt vs {len(fp32_s)} in npz"

    # tokenizer-drift guard: GGUF must tokenise the completion to the SAME ids as HF
    # logits_all=True is REQUIRED (0.3.36 returns all-zero logits otherwise). Keep n_ctx only as
    # large as the longest prompt+completion needs: the logits buffer is n_ctx*vocab*4 bytes, so
    # n_ctx=8192 alone is ~4.7GB and OOM-kills on a shared node. Size it to the data.
    print(f"loading GGUF {args.gguf} (n_ctx={args.n_ctx}, threads={args.threads}, "
          f"gpu_layers={args.n_gpu_layers})...", flush=True)
    llm = Llama(model_path=args.gguf, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                logits_all=True, n_threads=args.threads, verbose=False)
    hf_comp = tok(COMPLETION, add_special_tokens=False)["input_ids"]
    gg_comp = llm.tokenize(COMPLETION.encode(), add_bos=False, special=True)
    assert hf_comp == gg_comp, f"completion token ids differ!\n HF: {hf_comp}\n GGUF: {gg_comp}"
    print(f"completion tokenises identically ({len(hf_comp)} tokens, incl <|im_end|>={hf_comp[-1]})")

    n = min(args.n, len(rows))
    quant_s = np.full(n, np.nan)
    skipped = 0
    for i in range(n):
        p_ids = llm.tokenize(rows[i].prompt.encode(), add_bos=True, special=True)   # the already-built prompt
        if len(p_ids) + len(gg_comp) > args.n_ctx:                                  # won't fit the context
            skipped += 1
            continue
        quant_s[i] = -quant_logp(llm, gg_comp, p_ids)
        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}/{n}", flush=True)
    print(f"skipped {skipped} rows longer than n_ctx={args.n_ctx}", flush=True)

    fp = fp32_s[:n]
    ok_rows = ~np.isnan(quant_s)                            # drop rows skipped for length
    rho = spearmanr(fp[ok_rows], quant_s[ok_rows]).statistic
    # paired AUC on answerable rows: correct (raw_f1>=0.6) vs wrong (raw_f1<=0.2), s should rank correct higher
    ans = (target[:n] == "answer") & ok_rows
    corr = ans & (raw_f1[:n] >= 0.6)
    wrong = ans & (raw_f1[:n] <= 0.2)
    y = np.r_[np.ones(corr.sum()), np.zeros(wrong.sum())]
    auc_fp = roc_auc_score(y, np.r_[fp[corr], fp[wrong]])
    auc_q = roc_auc_score(y, np.r_[quant_s[corr], quant_s[wrong]])

    print("\n=== GGUF verifier-score gate ===")
    print(f"n scored             {n}  (correct {corr.sum()}, wrong {wrong.sum()})")
    print(f"Spearman rho         {rho:.4f}   (pass >= {RHO_MIN})")
    print(f"AUC fp32             {auc_fp:.4f}")
    print(f"AUC quant            {auc_q:.4f}")
    print(f"AUC drop             {auc_fp - auc_q:+.4f}   (pass <= {AUC_DROP_MAX})")
    dif = np.abs(fp[ok_rows] - quant_s[ok_rows])
    print(f"max |s diff|         {dif.max():.3f}   mean |diff| {dif.mean():.3f}")
    ok = rho >= RHO_MIN and (auc_fp - auc_q) <= AUC_DROP_MAX
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
