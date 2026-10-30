import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PCA_DIM = 256
CS = np.logspace(-6, 1, 8)                  # probe L2 strengths, picked on the inner folds
COMBINER_C = 100.0                          # two inputs only: weak regularisation
COVERAGE = {"risk@10": 0.1, "risk@30": 0.3, "risk@50": 0.5}             # the win is decided at 30%
WIN, GUARD = 0.975, 0.95                    # 97.5% to claim a win (Bonferroni over 2, kept for 1 as a hedge for
                                            # reused dev questions); unadjusted 95% to flag harm
TWIN_MARGIN = -0.10                         # twin guard: non-inferiority on clear-correct originals. ~120 pairs give
                                            # SE ~0.032; -0.10 is the tightest margin an equal challenger passes ~90%
RISK_BAR, FALLBACK = 0.15, 0.30             # threshold rule: on dev, the highest coverage with risk <= 15% at every
GRID = [round(0.05 * i, 2) for i in range(2, 21)]   # grid point from 10% up; if none, 30%. Its tau is applied to test
N_BOOT, SEED = 2000, 0
NAMES = ("baseline", "prompt_probe", "abstain+answer_probe")
COLS = ("risk@10", "risk@30", "risk@50", "natural_auc", "unanswerable_auc", "twin_pair_auc", "twin_pair_auc_correct",
        "aurc")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="runs/verifier/merged_runs_grpo_v2_1.7b_final_dev.npz")
    ap.add_argument("--same_sft", action="store_true")   # generator on SFT v2 1.7B: baseline and prompt-probe features
                                                         # are the ones already tested, so only challenger 2 is re-tested
    ap.add_argument("--test", default=None)              # test features: scored once, at the end, at the dev tau
    return ap.parse_args()


def load(path):
    d = np.load(path)
    keep = d["subtype"] != "contaminated"                               # their abstain labels are wrong
    d = {k: d[k][keep] for k in d.files}
    d["y"] = np.where(d["target"] == "abstain", 0.0, d["raw_f1"])       # any answer to an abstain prompt is wrong
    assert not np.isnan(d["y"]).any()
    d["correct"], d["wrong"] = d["y"] >= 0.6, d["y"] <= 0.2              # yes/no F1 is an exact match: 0 or 1
    d["clear"] = d["correct"] | d["wrong"]
    return d


def inner(X, rows, d):
    """Probes on 4 inner folds (grouped by paper) of the training rows, fit on clear rows. C is picked by the mean
    inner-fold AUC on clear rows. Returns the out-of-fold log-odds at that C (NaN off `rows`), its 4 probes, and C."""
    oof, fits, auc = np.full((len(CS), len(X)), np.nan), [[] for _ in CS], np.zeros((len(CS), 4))
    for f, (a, b) in enumerate(GroupKFold(4).split(rows, groups=d["paper_id"][rows])):
        tr, te = rows[a], rows[b]
        tr = tr[d["clear"][tr]]
        pre = make_pipeline(StandardScaler(), PCA(PCA_DIM, svd_solver="full")).fit(X[tr])     # deterministic solver
        z_tr, z_te = pre.transform(X[tr]), pre.transform(X[te])
        for c, C in enumerate(CS):
            lr = LogisticRegression(C=C, max_iter=5000).fit(z_tr, d["correct"][tr])
            oof[c, te] = lr.decision_function(z_te)
            m = d["clear"][te]
            auc[c, f] = roc_auc_score(d["correct"][te][m], oof[c, te][m])
            fits[c].append((pre, lr))
    best = auc.mean(axis=1).argmax()
    return oof[best], fits[best], CS[best]


def probe_score(fits, X):
    """Mean log-odds of the inner probes: the same scale as the out-of-fold scores the combiner was fit on."""
    return np.mean([lr.decision_function(pre.transform(X)) for pre, lr in fits], axis=0)


def fold_score(name, part, d, rows):
    """One outer fold's fitted challenger scored on `rows`: prompt_probe's part is its inner probes, the stacked
    one's is (inner probes, combiner)."""
    if name == "prompt_probe":
        return probe_score(part, d["h_prompt"][rows])
    fits, comb = part
    return comb.decision_function(np.column_stack([d["logp_abstain"][rows], probe_score(fits, d["h_answer"][rows])]))


def fit_scores(d):
    """Out-of-fold scores (higher: more confident the answer is right) for every row that passes the rules,
    from 5 folds grouped by paper. Natural and twin rows are fit together; a twin shares its original's fold.
    Also returns each fold's fitted challengers, to score test with their mean."""
    shown = np.where(d["shown"])[0]
    assert not np.isnan(d["h_answer"][shown]).any()
    s = {n: np.full(len(d["y"]), np.nan) for n in NAMES}
    s["baseline"][shown] = -d["logp_abstain"][shown]                     # unfitted: the frozen SFT's own score
    fold, log, models = np.full(len(d["y"]), -1), [], []
    for k, (a, b) in enumerate(GroupKFold(5).split(shown, groups=d["paper_id"][shown])):
        tr, te = shown[a], shown[b]
        fold[te] = k
        _, fits_p, c1 = inner(d["h_prompt"], tr, d)
        oof, fits_a, c2 = inner(d["h_answer"], tr, d)                    # stage 1: out-of-fold probe scores
        trc = tr[d["clear"][tr]]
        comb = make_pipeline(StandardScaler(), LogisticRegression(C=COMBINER_C, max_iter=5000))
        comb.fit(np.column_stack([d["logp_abstain"][trc], oof[trc]]), d["correct"][trc])   # stage 2
        parts = {"prompt_probe": fits_p, "abstain+answer_probe": (fits_a, comb)}
        for n in NAMES[1:]:
            s[n][te] = fold_score(n, parts[n], d, te)
        models.append(parts)
        log.append({"fold": k, "C_prompt_probe": float(c1), "C_answer_probe": float(c2),
                    "combiner_coef_abstain_probe": [round(float(c), 3) for c in comb[-1].coef_[0]]})
    return s, fold, log, models


def curve(s, idx, d):
    """Risk (wrong / clear among shown) after showing the k most confident rows of `idx`, k = 1..len(idx)."""
    o = idx[np.argsort(-s[idx], kind="stable")]
    cc = np.cumsum(d["clear"][o])
    return np.where(cc > 0, np.cumsum(d["wrong"][o]) / np.maximum(cc, 1), np.nan)


def risk_at(r, n_nat, c):
    """Coverage over all natural questions: round(c * n_nat) answers shown; NaN if fewer passed the rules."""
    k = int(round(c * n_nat))
    return r[k - 1] if 0 < k <= len(r) else np.nan


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def measures(s, rows, pairs, d):
    """The tested measures on `rows` (natural and twin; repeats allowed) and the twin `pairs` among them. The
    cutoffs follow from the rows, so a resample recomputes them."""
    nat = rows[~d["twin"][rows]]
    shown = nat[d["shown"][nat]]
    r = curve(s, shown, d)
    good = shown[d["correct"][shown]]
    out = {m: risk_at(r, len(nat), c) for m, c in COVERAGE.items()}
    out["natural_auc"] = auc(s[good], s[shown[d["wrong"][shown]]])
    out["unanswerable_auc"] = auc(s[good], s[shown[d["subtype"][shown] == "unanswerable"]])
    o, t = pairs[:, 0], pairs[:, 1]
    out["twin_pair_auc"] = np.mean((s[o] > s[t]) + 0.5 * (s[o] == s[t]))
    o, t = o[d["correct"][o]], t[d["correct"][o]]                       # a wrong original should be hidden too
    out["twin_pair_auc_correct"] = np.mean((s[o] > s[t]) + 0.5 * (s[o] == s[t]))
    out["aurc"] = np.nanmean(r)                                          # up to the highest coverage the rules allow
    return out


def twin_pairs(d):
    """(original, twin) row indices where both pass the rules."""
    nat = {q: i for i, q in enumerate(d["question_id"]) if not d["twin"][i]}
    pairs = [(nat[q], i) for i, q in enumerate(d["question_id"]) if d["twin"][i]]
    return np.array([(o, t) for o, t in pairs if d["shown"][o] and d["shown"][t]])


def bootstrap(s, d, pairs):
    """Paired bootstrap over papers: every resample recomputes every measure for every verifier."""
    rng = np.random.default_rng(SEED)
    papers = np.unique(d["paper_id"])
    rows_of = {p: np.where(d["paper_id"] == p)[0] for p in papers}
    pairs_of = {p: pairs[d["paper_id"][pairs[:, 0]] == p] for p in papers}
    out = []
    for _ in range(N_BOOT):
        draw = rng.choice(papers, len(papers))
        rows, pr = np.concatenate([rows_of[p] for p in draw]), np.concatenate([pairs_of[p] for p in draw])
        out.append({n: measures(s[n], rows, pr, d) for n in NAMES})
    return out


def interval(x, level):
    x = np.asarray(x, dtype=float)
    a = (1 - level) / 2 * 100
    return np.nanpercentile(x, [a, 100 - a])


def diff(boots, a, b, m):
    return [x[a][m] - x[b][m] for x in boots]


def verdict(boots, n):
    """The win rule (v2, fixed 2026-10-30 before new data): risk@30 lower at WIN; twin-pair AUC on clear-correct
    originals non-inferior (lower end at GUARD above TWIN_MARGIN); no guard significantly worse at GUARD."""
    worse = {"natural_auc": interval(diff(boots, n, "baseline", "natural_auc"), GUARD)[1] < 0,
             "risk@10": interval(diff(boots, n, "baseline", "risk@10"), GUARD)[0] > 0,
             "risk@50": interval(diff(boots, n, "baseline", "risk@50"), GUARD)[0] > 0,
             "unanswerable_auc": interval(diff(boots, n, "baseline", "unanswerable_auc"), GUARD)[1] < 0}
    rule = {"risk@30_lower": bool(interval(diff(boots, n, "baseline", "risk@30"), WIN)[1] < 0),
            "twin_correct_noninferior": bool(interval(diff(boots, n, "baseline", "twin_pair_auc_correct"), GUARD)[0]
                                             > TWIN_MARGIN),
            "no_guard_worse": not any(worse.values())}
    return {"rule": rule, "guard_worse": {k: bool(v) for k, v in worse.items()}, "wins": all(rule.values())}


def choose(verdicts, boots):
    """One winner: it. Both: the stacked one only if its risk@30 is significantly lower (GUARD), else the probe.
    Neither: the abstain score with a threshold."""
    wins = [n for n in verdicts if verdicts[n]["wins"]]
    if len(wins) < 2:
        return wins[0] if wins else "baseline"
    return NAMES[2] if interval(diff(boots, NAMES[2], NAMES[1], "risk@30"), GUARD)[1] < 0 else NAMES[1]


def pick(s, d):
    """The threshold rule on dev: c = the highest GRID coverage with risk <= RISK_BAR at every grid point from 10% up
    to c, or FALLBACK if risk is above the bar already at 10%. tau = the score of the round(c * natural)-th most
    confident answer, so 'show if score >= tau' gives coverage c."""
    nat = np.where(~d["twin"])[0]
    shown = nat[d["shown"][nat]]
    r = curve(s, shown, d)
    risks = {c: risk_at(r, len(nat), c) for c in GRID if round(c * len(nat)) <= len(r)}    # coverages the rules allow
    c = None
    for g in sorted(risks):
        if risks[g] > RISK_BAR:
            break
        c = g
    fallback = c is None
    c = FALLBACK if fallback else c
    return c, float(np.sort(s[shown])[::-1][int(round(c * len(nat))) - 1]), fallback, risks


def apply(s, d, tau, rows):
    """Natural questions in `rows`: show rule-passing answers scoring >= tau. Returns coverage (over all natural
    questions) and risk (wrong / clear among shown)."""
    nat = rows[~d["twin"][rows]]
    show = nat[d["shown"][nat] & (s[nat] >= tau)]
    clear = d["clear"][show].sum()
    return len(show) / len(nat), d["wrong"][show].sum() / clear if clear else np.nan


def at_tau(s, d, tau):
    """Coverage and risk at a fixed tau, each with a paper-clustered 95% interval."""
    rng = np.random.default_rng(SEED)
    papers = np.unique(d["paper_id"])
    rows_of = {p: np.where(d["paper_id"] == p)[0] for p in papers}
    b = np.array([apply(s, d, tau, np.concatenate([rows_of[p] for p in rng.choice(papers, len(papers))]))
                  for _ in range(N_BOOT)])
    cov, risk = apply(s, d, tau, np.arange(len(d["y"])))
    return {"coverage": cov, "coverage_ci": interval(b[:, 0], GUARD).tolist(),
            "risk": risk, "risk_ci": interval(b[:, 1], GUARD).tolist()}


def test_scores(name, t, models):
    """Test rows that pass the rules, scored by the chosen verifier: the baseline as is, a challenger as the mean
    of its 5 fold models (the scale its dev scores, and so tau, are on)."""
    s, shown = np.full(len(t["y"]), np.nan), np.where(t["shown"])[0]
    s[shown] = -t["logp_abstain"][shown] if name == "baseline" else \
        np.mean([fold_score(name, m[name], t, shown) for m in models], axis=0)
    return s


def main():
    args = parse_args()
    d = load(args.features)
    base = args.features[:-len(".npz")]
    nat, sh = ~d["twin"], d["shown"]
    pairs = twin_pairs(d)
    print(f"{nat.sum()} natural questions ({(nat & sh).sum()} pass the rules), {(~nat).sum()} twins "
          f"({(~nat & sh).sum()}); shown: {(sh & d['correct']).sum()} correct, {(sh & d['wrong']).sum()} wrong, "
          f"{(sh & ~d['clear']).sum()} middle; {len(pairs)} twin pairs with both shown")

    s, fold, log, models = fit_scores(d)
    for f in log:
        edge = [k for k in ("C_prompt_probe", "C_answer_probe") if f[k] in (CS[0], CS[-1])]
        print(f"  fold {f['fold']}: C prompt probe {f['C_prompt_probe']:g}, answer probe {f['C_answer_probe']:g}, "
              f"combiner [abstain, probe] {f['combiner_coef_abstain_probe']}" + (f"  (grid edge: {edge})" if edge else ""))
    rows = np.arange(len(d["y"]))
    point = {n: measures(s[n], rows, pairs, d) for n in NAMES}
    boots = bootstrap(s, d, pairs)

    print(f"\nnatural questions; coverage = share of all {nat.sum()} shown; risk = wrong / clear among shown")
    print(f"  {'verifier':22s}" + "".join(f"{c:>22s}" for c in COLS))
    for n in NAMES:
        print(f"  {n:22s}" + "".join(f"{point[n][c]:22.3f}" for c in COLS))
    tested = NAMES[2:] if args.same_sft else NAMES[1:]
    verdicts, diffs = {}, {}
    for n in NAMES[1:]:
        print(f"\n{n} - baseline (risk@30 at {WIN:.1%}, the rest at {GUARD:.0%}; paired bootstrap over papers)")
        diffs[n] = {}
        for c in COLS:
            lvl = WIN if c == "risk@30" else GUARD
            lo, hi = interval(diff(boots, n, "baseline", c), lvl)
            diffs[n][c] = {"diff": float(point[n][c] - point["baseline"][c]), "level": lvl, "lo": float(lo), "hi": float(hi)}
            print(f"  {c:17s} {point[n][c] - point['baseline'][c]:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        if n not in tested:
            print("  reported, not tested: these features were already tested on this SFT")
            continue
        verdicts[n] = verdict(boots, n)
        print(f"  win rule {verdicts[n]['rule']}  guard worse {verdicts[n]['guard_worse']}  -> "
              f"{'WINS' if verdicts[n]['wins'] else 'does not win'}")
    chosen = choose(verdicts, boots)
    print(f"\nverifier: {chosen}")
    c, tau, fallback, risks = pick(s[chosen], d)
    threshold = {"coverage_picked": c, "fallback": fallback, "tau": tau, "dev": at_tau(s[chosen], d, tau)}
    print(f"threshold rule (risk <= {RISK_BAR} at every 5% step from 10% up): "
          + (f"none (risk at 10% is {risks[0.1]:.3f}), fallback {FALLBACK:.0%}" if fallback else f"{c:.0%} coverage")
          + f"; tau = {tau:.4f}; dev at tau: coverage {threshold['dev']['coverage']:.3f}, "
          f"risk {threshold['dev']['risk']:.3f} [{threshold['dev']['risk_ci'][0]:.3f}, {threshold['dev']['risk_ci'][1]:.3f}]")

    grid = [round(0.05 * i, 2) for i in range(1, 19)]
    print("\nrisk-coverage curve, natural questions (- : coverage the rules don't allow)")
    print(f"  {'coverage':22s}" + "".join(f"{int(c * 100):>6d}" for c in grid))
    shown_nat = np.where(nat & sh)[0]
    for n in NAMES:
        r = curve(s[n], shown_nat, d)
        print(f"  {n:22s}" + "".join(f"{risk_at(r, nat.sum(), c):6.3f}" if c * nat.sum() <= len(r) else f"{'-':>6s}"
                                     for c in grid))
    good = shown_nat[d["correct"][shown_nat]]
    types = sorted(set(d["subtype"][shown_nat][d["target"][shown_nat] == "abstain"]))
    groups = {t: shown_nat[d["subtype"][shown_nat] == t] for t in types} | {"twin": np.where(~nat & sh)[0]}
    print("\nAUC, clear-correct natural answers vs each abstain type (reported, not tested)")
    by_type = {n: {t: float(auc(s[n][good], s[n][g])) for t, g in groups.items()} for n in NAMES}
    for n in NAMES:
        print(f"  {n:22s}" + "  ".join(f"{t} {by_type[n][t]:.3f} (n={len(groups[t])})" for t in groups))
    r = curve(s["baseline"], shown_nat, d)
    print("\nbaseline, prototype's coverage (share of shown answers): risk@10/30/50 "
          + " ".join(f"{r[int(round(c * len(r))) - 1]:.3f}" for c in (0.1, 0.3, 0.5))
          + f", twin-pair AUC {point['baseline']['twin_pair_auc']:.3f} (prototype: 0.270 0.330 0.430, 0.833)")

    with open(f"{base}_scores.jsonl", "w") as f:                        # out-of-fold scores: thresholds need no refit
        for i in rows:
            label = ("correct" if d["correct"][i] else "wrong" if d["wrong"][i] else "middle") if sh[i] else None
            f.write(json.dumps({"question_id": str(d["question_id"][i]), "paper_id": str(d["paper_id"][i]),
                                "twin": bool(d["twin"][i]), "subtype": str(d["subtype"][i]),
                                "outcome": str(d["outcome"][i]), "shown": bool(sh[i]), "label": label,
                                "raw_f1": None if np.isnan(d["raw_f1"][i]) else float(d["raw_f1"][i]),
                                "fold": int(fold[i]), **{n: None if np.isnan(s[n][i]) else float(s[n][i]) for n in NAMES}})
                    + "\n")
    report = {"point": {n: {c: float(v) for c, v in point[n].items()} for n in NAMES}, "diff_vs_baseline": diffs,
              "verdicts": verdicts, "chosen": chosen, "threshold": threshold, "folds": log, "auc_by_type": by_type,
              "settings": {"pca_dim": PCA_DIM, "cs": CS.tolist(), "combiner_c": COMBINER_C, "win": WIN, "guard": GUARD,
                           "twin_margin": TWIN_MARGIN, "tested": list(tested), "n_boot": N_BOOT, "seed": SEED}}
    json.dump(report, open(f"{base}_report.json", "w"), indent=2)
    print(f"\nsaved {base}_scores.jsonl and {base}_report.json")

    if args.test:                                                        # once, at the end: the dev tau, unchanged
        t = load(args.test)
        result = {"verifier": chosen, "tau": tau, **at_tau(test_scores(chosen, t, models), t, tau)}
        print(f"\nTEST, {chosen} at the dev tau: coverage {result['coverage']:.3f} [{result['coverage_ci'][0]:.3f}, "
              f"{result['coverage_ci'][1]:.3f}], risk {result['risk']:.3f} [{result['risk_ci'][0]:.3f}, "
              f"{result['risk_ci'][1]:.3f}]")
        json.dump(result, open(f"{args.test[:-len('.npz')]}_test_result.json", "w"), indent=2)
        print(f"saved {args.test[:-len('.npz')]}_test_result.json")


if __name__ == "__main__":
    main()
