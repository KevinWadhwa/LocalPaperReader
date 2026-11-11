"""Recalibrate the 4-tier confidence cutpoints for the QUANTISED (GGUF) verifier.

The fp32 cutpoints (configs/reader_4b.json: high 3.477 / medium 2.598 / low 1.575) were picked
on the fp32 abstain score. Q4 shifts the score distribution (gate: mean |s diff| 0.244), so the
cutpoints must be re-derived on the Q4 scores using the SAME dev-pick rules, then APPLIED
UNCHANGED to the Q4 test scores (held-out validation, exactly as the fp32 tiers were).

THE RULES (reproduced verbatim from the fp32 calibration):
  population   natural (~twin) AND shown (outcome in {answer, ungrounded}); s = -logP(abstain),
               y = raw_f1 with NaN -> 0 (an answered-but-unanswerable row has no gold F1 and is
               a wrong answer, so it counts as F1=0).
  tau_high     sort s desc; cumulative P(y>=.6); deepest k with cum>=0.75; tau_high = s[k-1].
  tau_abstain  largest c with (s<c).sum()>=20 and P(y<=.2 | s<c) >= 0.50.   (majority wrong below)
  tau_low      smallest c<tau_abstain with LOW band [c,tau_abstain): n>=30 and P(y>=.6)>P(y<=.2).

tiers written as {"high": tau_high, "medium": tau_abstain, "low": tau_low} to match reader.py's
confidence(): HIGH s>=high, MEDIUM medium<=s<high, LOW low<=s<medium, ABSTAIN s<low.

This script FIRST reproduces the fp32 cutpoints from the fp32 npz as a self-check (the method is
only trusted if it lands on the known 3.477 / ~2.6 / 1.575), THEN derives the Q4 cutpoints from
the GGUF dev scores (parsescripts/gguf_score_dev.py output).

    PYTHONPATH=. python parsescripts/recalibrate_tiers.py \
        --fp32_npz   runs/verifier/merged_runs_grpo_v2_4b_step80_dev.npz \
        --quant_npz  runs/gguf/verifier_q4km_dev_scores.npz \
        --base_config configs/reader_4b.json \
        --out        configs/reader_4b_gguf.json
"""
import argparse
import json

import numpy as np


def load_fp32(npz):
    """Population + scores from the fp32 verifier_features npz (keys: twin, shown, logp_abstain, raw_f1)."""
    d = np.load(npz, allow_pickle=True)
    m = (~d["twin"]) & d["shown"]
    s = -d["logp_abstain"].astype(np.float64)[m]
    y = np.nan_to_num(d["raw_f1"].astype(np.float64)[m], nan=0.0)
    return s, y


def load_quant(npz):
    """Same population + scores from the GGUF dev-scores npz (keys: twin, shown, quant_s, raw_f1).
    Rows the GGUF skipped (prompt longer than n_ctx) are NaN in quant_s and dropped here."""
    d = np.load(npz, allow_pickle=True)
    m = (~d["twin"]) & d["shown"] & ~np.isnan(d["quant_s"])
    s = d["quant_s"].astype(np.float64)[m]
    y = np.nan_to_num(d["raw_f1"].astype(np.float64)[m], nan=0.0)
    return s, y


def cutpoints(s, y):
    """The three dev-pick rules. Returns (tau_high, tau_abstain, tau_low)."""
    order = np.argsort(-s)
    so, yo = s[order], y[order]
    run_hi = np.cumsum(yo >= 0.6) / np.arange(1, len(yo) + 1)          # cumulative P(right) top-down
    k_hi = np.max(np.where(run_hi >= 0.75)[0]) + 1 if (run_hi >= 0.75).any() else 0
    tau_high = float(so[k_hi - 1]) if k_hi > 0 else float(so[0] + 1)

    cand = np.unique(np.round(s, 3))
    tau_abstain = None
    for c in sorted(cand):                                            # largest c with majority-wrong below
        below = s < c
        if below.sum() >= 20 and np.mean(y[below] <= 0.2) >= 0.50:
            tau_abstain = float(c)

    tau_low = None
    if tau_abstain is not None:
        for c in sorted(cand):                                        # smallest c where LOW band beats a coin flip
            if c >= tau_abstain:
                break
            low = (s >= c) & (s < tau_abstain)
            if low.sum() >= 30 and np.mean(y[low] >= 0.6) > np.mean(y[low] <= 0.2):
                tau_low = float(c)
                break
    return tau_high, tau_abstain, tau_low


def report(s, y, taus, name):
    """Per-tier share / meanF1 / P(right) / P(wrong), as the fp32 calibration reported."""
    th, tm, tl = taus
    print(f"\n=== {name} (n={len(s)}) ===")
    bands = [("HIGH", s >= th), ("MEDIUM", (s >= tm) & (s < th)),
             ("LOW", (s >= tl) & (s < tm)), ("ABSTAIN", s < tl)]
    for lbl, mask in bands:
        n = int(mask.sum())
        if n == 0:
            print(f"  {lbl:8s} n=0")
            continue
        yy = y[mask]
        print(f"  {lbl:8s} n={n:4d} ({n/len(s):4.0%})  meanF1={yy.mean():.3f}  "
              f"P(F1>=.6)={np.mean(yy >= 0.6):4.0%}  P(F1<=.2)={np.mean(yy <= 0.2):4.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_npz", required=True, help="fp32 verifier dev npz, for the method self-check")
    ap.add_argument("--quant_npz", required=True, help="GGUF dev-scores npz from gguf_score_dev.py")
    ap.add_argument("--quant_test_npz", default=None, help="optional: GGUF test-scores npz, for held-out validation")
    ap.add_argument("--base_config", required=True, help="fp32 reader config to clone non-tier fields from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 1. self-check: reproduce the fp32 cutpoints with this exact method
    sf, yf = load_fp32(args.fp32_npz)
    tf = cutpoints(sf, yf)
    print(f"fp32 method self-check: high={tf[0]:.3f} medium={tf[1]:.3f} low={tf[2]:.3f}")
    print("  (fp32 config target: high=3.477 medium=2.598 low=1.575)")
    assert abs(tf[0] - 3.477) < 0.05, f"HIGH cut {tf[0]:.3f} drifted from 3.477 — method changed, STOP"
    assert abs(tf[2] - 1.575) < 0.15, f"LOW cut {tf[2]:.3f} drifted from 1.575 — method changed, STOP"
    report(sf, yf, tf, "fp32 DEV (self-check)")

    # 2. derive Q4 cutpoints on the GGUF dev scores
    sq, yq = load_quant(args.quant_npz)
    tq = cutpoints(sq, yq)
    print(f"\nQ4 DEV-picked cutpoints: high={tq[0]:.3f} medium={tq[1]:.3f} low={tq[2]:.3f}")
    assert tq[1] is not None and tq[2] is not None, "Q4 abstain/low cut not found — inspect the Q4 score distribution"
    report(sq, yq, tq, "Q4 DEV (where Q4 boundaries were picked)")

    # 3. optional held-out validation on Q4 test (boundaries applied UNCHANGED)
    if args.quant_test_npz:
        st, yt = load_quant(args.quant_test_npz)
        report(st, yt, tq, "Q4 TEST (held out, Q4 boundaries unchanged)")

    # 4. write the GGUF reader config: clone base, replace tiers, mark backend
    cfg = json.load(open(args.base_config))
    cfg["tiers"] = {"high": round(tq[0], 3), "medium": round(tq[1], 3), "low": round(tq[2], 3)}
    cfg["backend"] = "gguf"
    cfg["verifier_gguf"] = "runs/gguf/verifier-q4km.gguf"
    cfg["generator_gguf"] = "runs/gguf/generator-q4km.gguf"
    cfg["device"] = "mps"            # Apple Silicon embedder; set explicit so pick_device needs no torch probe
    cfg["n_gpu_layers"] = -1         # all GGUF layers on Metal
    cfg["calibrated_on"] = {"fp32_selfcheck": [round(x, 3) for x in tf],
                            "quant_dev_npz": args.quant_npz}
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nwrote {args.out}: tiers {cfg['tiers']}")


if __name__ == "__main__":
    main()
