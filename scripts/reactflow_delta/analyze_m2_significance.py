#!/usr/bin/env python3
"""analyze_m2_significance — statistical significance of the residual-model
improvement over the per-position median prior on the M2 expanded pool.

Primary metric
--------------
improvement_d = mean_{changers in design d} ( baseline_wmae - resid_wmae )
where baseline_wmae is the wmed_spectrum prior (seed 0) and resid_wmae is the
wmae_resid_spectrum model averaged over seeds 0..4, both computed as weighted MAE
over eligible window positions on HELD-OUT changers.

The design (puzzle x method) is the exchangeable unit, so the PRIMARY permutation
test permutes at the DESIGN level:
    H0 : the residual model provides no net improvement over the prior,
         i.e. the distribution of per-design improvement is symmetric about 0.
    null : B times, flip the sign of each design's improvement with p=0.5,
           record the mean improvement.
    p    = P(null_mean >= observed_mean).

We additionally report:
  * a pair-level sign-flip permutation test (13,614 changers);
  * Wilcoxon signed-rank and a one-sample t-test on the per-design improvements;
  * bootstrap 95% CI of the mean per-design improvement.

Deterministic RNG seed; output JSON + printed tables.
"""
from __future__ import annotations

import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RNG_SEED = 20260812
B_DESIGN = 20000
B_PAIR = 20000


def row_wmae(y, w, pred):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = w.sum()
    if denom <= 0:
        return None
    return float((w * np.abs(y - pred)).sum() / denom)


def load_predictions(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B-design", type=int, default=B_DESIGN)
    ap.add_argument("--B-pair", type=int, default=B_PAIR)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    rows = load_predictions(args.pred)
    var_pairs = defaultdict(lambda: defaultdict(dict))   # variant -> pair -> seed -> wmae
    pair_fold = {}
    for r in rows:
        if r["coverage_status"] != "CALLED":
            continue
        wmae = row_wmae(r["y"], r["weight"], r["raw_prediction"])
        if wmae is None:
            continue
        var_pairs[r["model_variant"]][r["pair_id"]][int(r["seed"])] = wmae
        pair_fold[r["pair_id"]] = r["fold_id"]

    base = var_pairs["wmed_spectrum"]
    resid = var_pairs["wmae_resid_spectrum"]
    common = sorted(set(base) & set(resid))

    # per-pair improvement
    per_pair = {}
    for pid in common:
        b = float(np.mean(list(base[pid].values())))
        r = float(np.mean(list(resid[pid].values())))
        per_pair[pid] = (b - r, pair_fold[pid], b, r)

    # per-design mean improvement (exchangeable unit)
    by_design = defaultdict(list)
    for pid in common:
        imp, fd, b, r = per_pair[pid]
        by_design[fd].append(imp)
    design_ids = sorted(by_design)
    per_design = np.array([float(np.mean(by_design[d])) for d in design_ids])
    per_pair_arr = np.array([per_pair[p][0] for p in common])

    obs_design = float(per_design.mean())
    obs_pair = float(per_pair_arr.mean())

    rng = np.random.default_rng(args.seed)
    # ---- design-level sign-flip permutation ----
    null_design = np.empty(args.B_design)
    for b in range(args.B_design):
        signs = rng.choice([-1.0, 1.0], size=len(per_design))
        null_design[b] = (per_design * signs).mean()
    p_design = float((null_design >= obs_design).mean())

    # ---- pair-level sign-flip permutation ----
    null_pair = np.empty(args.B_pair)
    for b in range(args.B_pair):
        signs = rng.choice([-1.0, 1.0], size=len(per_pair_arr))
        null_pair[b] = (per_pair_arr * signs).mean()
    p_pair = float((null_pair >= obs_pair).mean())

    # ---- Wilcoxon signed-rank + one-sample t on per-design improvement ----
    w_stat, w_p = stats.wilcoxon(per_design)
    t_stat, t_p = stats.ttest_1samp(per_design, 0.0)
    # one-sided p (improvement > 0)
    t_p_one = t_p / 2.0 if t_stat > 0 else 1.0 - t_p / 2.0

    # ---- bootstrap 95% CI on mean per-design improvement ----
    boot = np.empty(5000)
    rng2 = np.random.default_rng(args.seed + 1)
    for i in range(5000):
        boot[i] = rng2.choice(per_design, size=len(per_design), replace=True).mean()
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # effect size (Cohen's d on per-design improvements)
    sd = per_design.std(ddof=1)
    cohens_d = float(obs_design / sd) if sd > 0 else float("nan")

    out = {
        "n_pairs": len(common),
        "n_designs": len(design_ids),
        "observed": {
            "mean_pair_improvement": obs_pair,
            "mean_design_improvement": obs_design,
            "pct_designs_improved": float((per_design > 0).mean()),
        },
        "permutation_design_level": {
            "n_perm": args.B_design, "mean_null": float(null_design.mean()),
            "sd_null": float(null_design.std(ddof=1)), "p_value": p_design,
            "z_score": float((obs_design - null_design.mean()) / null_design.std(ddof=1)),
        },
        "permutation_pair_level": {
            "n_perm": args.B_pair, "mean_null": float(null_pair.mean()),
            "sd_null": float(null_pair.std(ddof=1)), "p_value": p_pair,
        },
        "wilcoxon_design": {"statistic": float(w_stat), "p_value": float(w_p)},
        "ttest_design": {"statistic": float(t_stat), "p_value_two_sided": float(t_p),
                         "p_value_one_sided": t_p_one},
        "bootstrap": {"ci95_lo": ci_lo, "ci95_hi": ci_hi},
        "effect_size": {"cohens_d": cohens_d},
        "seed": args.seed,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print("=== SIGNIFICANCE ANALYSIS (improvement = baseline_wmae - resid_wmae) ===")
    print(f"n_pairs={len(common)}  n_designs={len(design_ids)}")
    print(f"observed mean pair improvement      = {obs_pair:.4f}")
    print(f"observed mean design improvement    = {obs_design:.4f}")
    print(f"pct designs improved                = {out['observed']['pct_designs_improved']:.3f}")
    print("\n--- DESIGN-LEVEL sign-flip permutation (primary, exchangeable unit) ---")
    print(f"  null mean={out['permutation_design_level']['mean_null']:.5f} "
          f"sd={out['permutation_design_level']['sd_null']:.5f} "
          f"z={out['permutation_design_level']['z_score']:.2f} "
          f"p={out['permutation_design_level']['p_value']:.5f}")
    print("--- PAIR-LEVEL sign-flip permutation ---")
    print(f"  null mean={out['permutation_pair_level']['mean_null']:.5f} "
          f"sd={out['permutation_pair_level']['sd_null']:.5f} "
          f"p={out['permutation_pair_level']['p_value']:.5f}")
    print(f"--- Wilcoxon signed-rank (designs) ---  W={w_stat:.1f} p={w_p:.6g}")
    print(f"--- one-sample t (designs) ---  t={t_stat:.3f} p_one={t_p_one:.6g}")
    print(f"--- bootstrap 95% CI of mean design improvement ---  "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"--- Cohen's d (per-design) ---  {cohens_d:.3f}")
    print(f"\nWROTE -> {args.out}")


if __name__ == "__main__":
    main()
