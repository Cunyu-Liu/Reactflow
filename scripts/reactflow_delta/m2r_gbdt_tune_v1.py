#!/usr/bin/env python3
"""m2r_gbdt_tune_v1.py — GBDT hyperparameter search for the M2R task.

Tests whether the default config (100 trees, depth 3) is optimal, or a deeper
/wider config gives a real gain.  Runs a coarse grid on a SUBSET of designs
(30) for speed, then validates the best config on ALL designs (design-level
and puzzle-level LOO).

Server is heavily loaded, so the subset search uses small GBDTs; the final
validation uses the winner with enough trees.
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


def _loo_skill(X, y, keys, config, des_list, n_jobs=2):
    import lightgbm as lgb
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=n_jobs, **config)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    skill = 1.0 - _mae(y, preds) / mae_bl
    r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    return skill, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset-designs", type=int, default=30,
                    help="number of designs for the fast coarse grid")
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    subset = des_list[:args.subset_designs]
    sub_mask = np.array([k in subset for k in keys])
    Xs, ys, ks = X[sub_mask], y[sub_mask], keys[sub_mask]

    grid = {
        "default_100d3": {"n_estimators": 100, "max_depth": 3},
        "t100_d4_l31": {"n_estimators": 100, "max_depth": 4, "num_leaves": 31},
        "t100_d5_l31": {"n_estimators": 100, "max_depth": 5, "num_leaves": 31},
        "t100_d4_l63": {"n_estimators": 100, "max_depth": 4, "num_leaves": 63},
        "t200_d4_l31": {"n_estimators": 200, "max_depth": 4, "num_leaves": 31},
        "t200_d5_l63_lr01": {"n_estimators": 200, "max_depth": 5, "num_leaves": 63,
                             "learning_rate": 0.1, "min_child_samples": 20},
    }

    print(f"subset designs={len(subset)} samples={len(ys)}", flush=True)
    subset_results = {}
    for name, cfg in grid.items():
        t0 = time.time()
        sk, r2 = _loo_skill(Xs, ys, ks, cfg, subset)
        subset_results[name] = {"skill": float(sk), "r2": float(r2),
                                "wall": round(time.time() - t0, 1)}
        print(f"  {name:24s} skill={sk:+.4f} R2={r2:.4f} "
              f"wall={subset_results[name]['wall']}s", flush=True)

    # ---- validate best on ALL designs (design-level + puzzle-level) ----
    best_name = max(subset_results, key=lambda k: subset_results[k]["skill"])
    best_cfg = grid[best_name]
    print(f"\nbest subset config: {best_name}", flush=True)

    sk_d, r2_d = _loo_skill(X, y, keys, best_cfg, des_list)
    print(f"FULL design-level: {best_name} skill={sk_d:+.4f} R2={r2_d:.4f}", flush=True)

    # puzzle-level
    sample_puzzles = np.array([s.puzzle for s in samples])
    puzzles = sorted(set(sample_puzzles.tolist()))
    import lightgbm as lgb
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    preds = np.zeros(len(y))
    for held_p in puzzles:
        m = sample_puzzles != held_p
        if m.sum() <= 10:
            preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=2, **best_cfg)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    sk_p = 1.0 - _mae(y, preds) / mae_bl
    r2_p = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"FULL puzzle-level:  {best_name} skill={sk_p:+.4f} R2={r2_p:.4f}", flush=True)

    # ---- also report the DEFAULT config on full data for reference ----
    sk_dd, r2_dd = _loo_skill(X, y, keys, grid["default_100d3"], des_list)
    print(f"FULL design-level default: skill={sk_dd:+.4f} R2={r2_dd:.4f}", flush=True)

    report = {
        "schema": "reactflow_delta.m2r_gbdt_tune.v1",
        "dataset": "OpenKnot_M2R", "subset_designs": args.subset_designs,
        "subset_results": subset_results,
        "best_config": {"name": best_name, **best_cfg},
        "full_design_level_best": {"skill": float(sk_d), "r2": float(r2_d)},
        "full_puzzle_level_best": {"skill": float(sk_p), "r2": float(r2_p)},
        "full_design_level_default": {"skill": float(sk_dd), "r2": float(r2_dd)},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_gbdt_tune.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out / 'm2r_gbdt_tune.json'}")


if __name__ == "__main__":
    sys.exit(main())