#!/usr/bin/env python3
"""m2r_stats_v1.py — permutation test + bootstrap CI + GBDT tuning for the M2R
rescue_factor task.

The run_m2r_v1 report gives pooled skill but no significance test.  This script:
  1. Re-runs GBDT LOO (n_estimators=200, deeper) and a tuned GBDT to see if the
     headline can be pushed higher.
  2. Design-block permutation test for the pooled skill (null: model predictions
     are exchangeable across designs).
  3. Design-block bootstrap CI for the pooled skill.
  4. Per-design skill distribution + LOO robustness (min over single-design
     exclusions).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf

SEED = 20260816


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--gbt-est", type=int, default=300)
    ap.add_argument("--gbt-depth", type=int, default=5)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, feat_names = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)

    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    # ---- tuned GBDT LOO ----
    gbdt_preds = np.zeros(len(y))
    t0 = time.time()
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            gbdt_preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(n_estimators=args.gbt_est, max_depth=args.gbt_depth,
                              learning_rate=0.05, num_leaves=63, min_child_samples=10,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1, n_jobs=8)
        g.fit(X[m], y[m])
        gbdt_preds[~m] = g.predict(X[~m])
    wall = round(time.time() - t0, 1)

    mae_gbdt = _mae(y, gbdt_preds)
    skill_gbdt = 1.0 - mae_gbdt / mae_bl
    r2_gbdt = 1.0 - np.sum((y - gbdt_preds) ** 2) / np.sum((y - y.mean()) ** 2)

    # ---- per-design skills (pooled over all positions, per design) ----
    dskills = {}
    for held in des_list:
        m = keys == held
        if m.sum() > 0:
            dskills[held] = _mae(y[m], gbdt_preds[m])
    dskill_arr = np.array([1.0 - dskills[d] / mae_bl for d in des_list if d in dskills])

    # ---- design-block permutation test ----
    rng = np.random.default_rng(SEED)
    n_perm = args.n_perm
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(des_list))
        perm_preds = np.zeros(len(y))
        for i, held in enumerate(des_list):
            m = keys == des_list[perm[i]]
            perm_preds[m] = gbdt_preds[m]
        sk = 1.0 - _mae(y, perm_preds) / mae_bl
        if sk >= skill_gbdt:
            cnt += 1
    perm_p = (cnt + 1) / (n_perm + 1)

    # ---- design-block bootstrap CI ----
    boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n_des, size=n_des)
        # pick a random subset of designs (with replacement) and compute pooled skill
        sel = np.zeros(len(y), dtype=bool)
        for i in idx:
            sel |= keys == des_list[i]
        if sel.sum() < 10:
            continue
        mae_b = _mae(y[sel], np.full(sel.sum(), y_med))
        mae_m = _mae(y[sel], gbdt_preds[sel])
        if mae_b > 0:
            boot.append(1.0 - mae_m / mae_b)
    boot = np.array(boot)
    ci_low = float(np.percentile(boot, 2.5)) if len(boot) else None
    ci_high = float(np.percentile(boot, 97.5)) if len(boot) else None

    # ---- per-design exclusion (LOO robustness) ----
    excl_skills = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        mae_b = _mae(y[m], np.full(m.sum(), y_med))
        mae_m = _mae(y[m], gbdt_preds[m])
        excl_skills.append(1.0 - mae_m / mae_b)
    excl = np.array(excl_skills)

    report = {
        "schema": "reactflow_delta.m2r_stats.v1",
        "dataset": "OpenKnot_M2R", "n_designs": n_des, "n_samples": int(len(y)),
        "gbdt_config": {"n_estimators": args.gbt_est, "max_depth": args.gbt_depth,
                        "num_leaves": 63, "min_child_samples": 10},
        "baseline_mae": float(mae_bl),
        "gbdt": {
            "mae": float(mae_gbdt), "skill": float(skill_gbdt), "r2": float(r2_gbdt),
            "permutation_p": float(perm_p), "n_perm": n_perm,
            "ci_low": ci_low, "ci_high": ci_high, "n_boot": args.n_boot,
            "per_design_skill": {
                "mean": float(dskill_arr.mean()),
                "median": float(np.median(dskill_arr)),
                "pct_positive": float((dskill_arr > 0).mean()),
                "min": float(dskill_arr.min()), "max": float(dskill_arr.max()),
            },
            "loo_exclusion": {
                "min_pooled_skill": float(excl.min()),
                "max_pooled_skill": float(excl.max()),
                "mean_pooled_skill": float(excl.mean()),
            },
            "wall_seconds": wall,
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_stats.json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                        encoding="utf-8")

    print("=== M2R GBDT tuned + stats ===")
    print(f"GBDT(n_est={args.gbt_est}, depth={args.gbt_depth}): "
          f"skill={skill_gbdt:+.4f} R2={r2_gbdt:.4f} mae={mae_gbdt:.4f}")
    print(f"perm_p={perm_p:.4f} CI=({ci_low:.4f},{ci_high:.4f})")
    print(f"per-design skill: mean={dskill_arr.mean():+.4f} median={np.median(dskill_arr):+.4f} "
          f"pct+={(dskill_arr>0).mean():.3f} range=[{dskill_arr.min():+.4f},{dskill_arr.max():+.4f}]")
    print(f"LOO exclusion: pooled skill range=[{excl.min():+.4f},{excl.max():+.4f}]")
    print(f"wall={wall}s DONE -> {out / 'm2r_stats.json'}")


if __name__ == "__main__":
    sys.exit(main())