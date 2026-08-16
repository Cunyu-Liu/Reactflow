#!/usr/bin/env python3
"""m2r_puzzle_loo_v1.py — puzzle-level leave-one-out (train 19 puzzles → test 1).
This is a MUCH harder generalization test than the design-level LOO.

If the GBDT signal survives puzzle-level LOO, it demonstrates that the model
learns a generalizable base-pairing mechanism, not dataset-specific patterns.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
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
    ap.add_argument("--m2-csv", default=None,
                    help="Optional M2 CSV to attach M2_structure features")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=30)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    if args.m2_csv:
        m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)

    # map each sample to its puzzle
    sample_puzzles = np.array([s.puzzle for s in samples])
    puzzles = sorted(set(sample_puzzles.tolist()))
    n_puzzles = len(puzzles)
    print(f"n_puzzles={n_puzzles} n_samples={len(y)}", flush=True)

    import lightgbm as lgb
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    # to compare with design-level LOO, also compute design-level here
    design_keys = np.array(keys)
    des_list = sorted(set(design_keys.tolist()))

    # ---- design-level LOO (for fair comparison, same GBDT config) ----
    design_preds = np.zeros(len(y))
    for held in des_list:
        m = design_keys != held
        if m.sum() <= 10:
            design_preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=3,
                              random_state=SEED, verbose=-1, n_jobs=2)
        g.fit(X[m], y[m])
        design_preds[~m] = g.predict(X[~m])
    mae_d = _mae(y, design_preds)
    skill_d = 1.0 - mae_d / mae_bl
    r2_d = 1.0 - np.sum((y - design_preds) ** 2) / np.sum((y - y.mean()) ** 2)
    # per-puzzle from design-level LOO
    d_by_puzzle = {}
    for p in puzzles:
        m = sample_puzzles == p
        if m.sum() > 0:
            mae_m = _mae(y[m], design_preds[m])
            mae_b = _mae(y[m], np.full(m.sum(), y_med))
            d_by_puzzle[p] = 1.0 - mae_m / mae_b
    print(f"design-level LOO: skill={skill_d:+.4f} R2={r2_d:.4f}", flush=True)

    # ---- puzzle-level LOO ----
    puzzle_preds = np.zeros(len(y))
    for held_p in puzzles:
        m = sample_puzzles != held_p
        if m.sum() <= 10:
            puzzle_preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=3,
                              random_state=SEED, verbose=-1, n_jobs=2)
        g.fit(X[m], y[m])
        puzzle_preds[~m] = g.predict(X[~m])
    mae_p = _mae(y, puzzle_preds)
    skill_p = 1.0 - mae_p / mae_bl
    r2_p = 1.0 - np.sum((y - puzzle_preds) ** 2) / np.sum((y - y.mean()) ** 2)
    # per-puzzle from puzzle-level LOO
    p_by_puzzle = {}
    for p in puzzles:
        m = sample_puzzles == p
        if m.sum() > 0:
            mae_m = _mae(y[m], puzzle_preds[m])
            mae_b = _mae(y[m], np.full(m.sum(), y_med))
            p_by_puzzle[p] = 1.0 - mae_m / mae_b
    print(f"puzzle-level LOO: skill={skill_p:+.4f} R2={r2_p:.4f}", flush=True)

    # ---- per-puzzle comparison ----
    print("\n--- per-puzzle skill comparison ---")
    print(f"{'puzzle':6s} {'design_LOO':>12s} {'puzzle_LOO':>12s} {'diff':>8s}")
    gains = []
    for p in puzzles:
        ds = d_by_puzzle.get(p, 0.0)
        ps = p_by_puzzle.get(p, 0.0)
        print(f"{p:6s} {ds:>+12.4f} {ps:>+12.4f} {ps-ds:>+8.4f}")
        gains.append(ps - ds)
    gains = np.array(gains)
    print(f"\n{'mean':6s} {np.mean(list(d_by_puzzle.values())):>+12.4f} "
          f"{np.mean(list(p_by_puzzle.values())):>+12.4f} {gains.mean():>+8.4f}")

    report = {
        "schema": "reactflow_delta.m2r_puzzle_loo.v1",
        "dataset": "OpenKnot_M2R", "n_puzzles": n_puzzles, "n_samples": int(len(y)),
        "trees": args.trees, "baseline_mae": float(mae_bl),
        "design_level_loo": {"skill": float(skill_d), "r2": float(r2_d),
                             "per_puzzle": {k: float(v) for k, v in d_by_puzzle.items()}},
        "puzzle_level_loo": {"skill": float(skill_p), "r2": float(r2_p),
                             "per_puzzle": {k: float(v) for k, v in p_by_puzzle.items()}},
        "pooled_skill_drop": {"absolute": float(skill_p - skill_d),
                              "relative": float((skill_p - skill_d) / max(abs(skill_d), 1e-9))},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_puzzle_loo.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out / 'm2r_puzzle_loo.json'}")


if __name__ == "__main__":
    sys.exit(main())