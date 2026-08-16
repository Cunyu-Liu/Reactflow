#!/usr/bin/env python3
"""m2r_m2structure_ablation_v1.py — M2_structure feature ablation for M2R.

The M2_structure (ShapeKnots, from M2 data) provides experimentally-inferred
secondary structure, which may differ from the designed target_structure.

This script:
  1. Loads M2R data + existing features (213 dims).
  2. Loads M2 data to get design-level M2_structure, M2_F1, M2_F1_crossed_pair.
  3. For each M2R pair, extracts M2_structure features at the pair sites (i,j):
     - m2_paired_i, m2_paired_j (is the site paired in inferred structure?)
     - m2_depth_i, m2_depth_j (bracket depth at sites)
     - m2_f1, m2_f1_crossed_pair (design-level quality)
  4. Runs GBDT LOO (design-block) with and without M2_structure features.
  5. Reports skill comparison.

LEGAL: M2_structure is derived from M2 experimental data (ShapeKnots on
single-mutant SHAPE), which is an INDEPENDENT data source from the M2R
rescue_factor target.  The M2 data for a given (puzzle,method) design was
collected in a separate experiment on the same RNA construct.

HONEST CAVEAT: M2_structure correlates with the same underlying pairing
biology as the rescue_factor target.  A large gain would indicate that the
experimentally-determined structure is a near-tautological predictor of
pairing support — a "data-level" rather than "method-level" improvement.
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf

SEED = 20260816
WINDOW = 7
BASES = "ACGU"
WC_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE = {("G", "U"), ("U", "G")}


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _dot_to_depth(structure):
    """Return (paired, depth) arrays over the structure string."""
    n = len(structure)
    paired = np.zeros(n, dtype=np.float64)
    depth = np.zeros(n, dtype=np.float64)
    stack = []
    openers = "([{"
    closers = ")]}"
    for i, ch in enumerate(structure):
        if ch in openers:
            stack.append(ch)
            paired[i] = 1.0
            depth[i] = len(stack)
        elif ch in closers:
            paired[i] = 1.0
            depth[i] = len(stack)
            if stack:
                stack.pop()
        else:
            depth[i] = len(stack)
    return paired, depth


def _read_m2_designs(m2_csv_path: str) -> dict:
    """Return dict: (puzzle, method) -> {m2_structure, m2_f1, m2_f1_cp, sub_start}."""
    info = {}
    with open(m2_csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("mutA") and not row.get("mutB"):
                key = (row["puzzle"], row["method"])
                info[key] = {
                    "m2_structure": row.get("M2_structure") or "",
                    "m2_f1": float(row["M2_F1"]) if row.get("M2_F1") else None,
                    "m2_f1_crossed_pair": float(row["M2_F1_crossed_pair"])
                    if row.get("M2_F1_crossed_pair") else None,
                    "sub_start": int(row["sub_start"]) if row.get("sub_start") else None,
                    "sub_end": int(row["sub_end"]) if row.get("sub_end") else None,
                }
    return info


def build_m2structure_features(
    samples: list,
    m2_designs: dict,
) -> np.ndarray:
    """Build M2_structure feature matrix for each M2R pair sample.

    For each sample, extract M2_structure at the pair sites (i,j) and
    design-level M2_F1/F1_crossed_pair.

    Returns (n_samples, 6) array: [m2_paired_i, m2_paired_j,
    m2_depth_i, m2_depth_j, m2_f1, m2_f1_cp].
    NaN/None entries are filled with 0.0.
    """
    n = len(samples)
    out = np.zeros((n, 6), dtype=np.float64)

    for idx, s in enumerate(samples):
        key = (s.puzzle, s.method)
        des = m2_designs.get(key)
        if des is None or not des["m2_structure"]:
            continue  # leave as zeros

        m2str = des["m2_structure"]
        sub_start = des["sub_start"]
        if sub_start is None:
            continue

        # M2_structure covers the design region starting at sub_start (1-indexed)
        # M2R pair sites are in full-sequence 0-indexed coordinates
        m2_idx_i = s.editA_seq_pos - (sub_start - 1)
        m2_idx_j = s.editB_seq_pos - (sub_start - 1)

        pa, dp = _dot_to_depth(m2str)

        if 0 <= m2_idx_i < len(pa):
            out[idx, 0] = pa[m2_idx_i]
            out[idx, 2] = dp[m2_idx_i]
        if 0 <= m2_idx_j < len(pa):
            out[idx, 1] = pa[m2_idx_j]
            out[idx, 3] = dp[m2_idx_j]

        if des["m2_f1"] is not None:
            out[idx, 4] = des["m2_f1"]
        if des["m2_f1_crossed_pair"] is not None:
            out[idx, 5] = des["m2_f1_crossed_pair"]

    return out


def _loo_gbdt(X, y, keys, n_estimators=100, max_depth=3):
    """Leave-one-design-out GBDT, return OOF predictions.

    Returns (preds, design_idx) where design_idx is the index of each sample
    within the sorted design list (for per-design aggregation).
    """
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    des2idx = {d: i for i, d in enumerate(des_list)}
    design_idx = np.array([des2idx[k] for k in keys])
    preds = np.zeros(len(y))
    for fi, held in enumerate(des_list):
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        try:
            import lightgbm as lgb
            g = lgb.LGBMRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=SEED, verbose=-1, n_jobs=8)
            g.fit(X[m], y[m])
            preds[~m] = g.predict(X[~m])
        except Exception:
            preds[~m] = np.median(y)
    return preds, design_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True,
                    help="Path to M2R CSV")
    ap.add_argument("--m2-csv", required=True,
                    help="Path to M2 CSV (for M2_structure)")
    ap.add_argument("--out", required=True,
                    help="Output directory")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load M2R data + existing features ----
    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = m2r.build_all_pair_samples(designs)
    samples = [s for s in samples if s.rescue_factor is not None]
    print(f"[m2r_str] n_samples={len(samples)} n_designs={len(designs)}", flush=True)

    X_existing, y, keys, feat_names = m2rf.build_all(samples)
    keys = np.array(keys)
    print(f"[m2r_str] X_existing={X_existing.shape} n_features={len(feat_names)}", flush=True)

    # ---- 2. Load M2 design info ----
    m2_designs = _read_m2_designs(args.m2_csv)
    print(f"[m2r_str] M2 designs loaded: {len(m2_designs)}", flush=True)

    # ---- 3. Build M2_structure features ----
    X_m2str = build_m2structure_features(samples, m2_designs)
    print(f"[m2r_str] X_m2str={X_m2str.shape}", flush=True)
    # check non-zero fraction
    nonzero_frac = (X_m2str[:, :4].sum(axis=1) > 0).mean()
    print(f"[m2r_str] M2_structure non-zero fraction (paired/depth): {nonzero_frac:.3f}", flush=True)
    f1_frac = (X_m2str[:, 4] > 0).mean()
    print(f"[m2r_str] M2_F1 non-zero fraction: {f1_frac:.3f}", flush=True)

    # ---- 4. Baseline ----
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[m2r_str] baseline MAE={mae_bl:.4f} (median={y_med:.4f})", flush=True)

    # ---- 5. GBDT LOO: existing features only ----
    t0 = time.time()
    pred_existing, _ = _loo_gbdt(X_existing, y, keys, args.trees, args.depth)
    wall_existing = time.time() - t0
    mae_existing = _mae(y, pred_existing)
    skill_existing = _skill(mae_existing, mae_bl)
    r2_existing = 1.0 - np.sum((y - pred_existing) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"[m2r_str] existing only: MAE={mae_existing:.4f} skill={skill_existing:+.4f} "
          f"R2={r2_existing:.4f} wall={wall_existing:.1f}s", flush=True)

    # ---- 6. GBDT LOO: existing + M2_structure features ----
    X_combined = np.concatenate([X_existing, X_m2str], axis=1)
    t0 = time.time()
    pred_combined, _ = _loo_gbdt(X_combined, y, keys, args.trees, args.depth)
    wall_combined = time.time() - t0
    mae_combined = _mae(y, pred_combined)
    skill_combined = _skill(mae_combined, mae_bl)
    r2_combined = 1.0 - np.sum((y - pred_combined) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"[m2r_str] existing+M2str: MAE={mae_combined:.4f} skill={skill_combined:+.4f} "
          f"R2={r2_combined:.4f} wall={wall_combined:.1f}s", flush=True)

    # ---- 7. GBDT LOO: M2_structure features ONLY ----
    t0 = time.time()
    pred_str_only, _ = _loo_gbdt(X_m2str, y, keys, args.trees, args.depth)
    wall_str_only = time.time() - t0
    mae_str_only = _mae(y, pred_str_only)
    skill_str_only = _skill(mae_str_only, mae_bl)
    r2_str_only = 1.0 - np.sum((y - pred_str_only) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"[m2r_str] M2str only: MAE={mae_str_only:.4f} skill={skill_str_only:+.4f} "
          f"R2={r2_str_only:.4f} wall={wall_str_only:.1f}s", flush=True)

    # ---- 8. LOO-exclusion robustness for the gain ----
    # Exclude each design and recompute the gain from M2_structure.
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            continue
        s_ex = _skill(_mae(y[m], pred_existing[m]), mae_bl)
        s_co = _skill(_mae(y[m], pred_combined[m]), mae_bl)
        gains.append(s_co - s_ex)
    gains = np.array(gains)

    # ---- 9. Report ----
    report = {
        "schema": "reactflow_delta.m2r_m2structure_ablation.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": len(y),
        "n_designs": len(set(keys.tolist())),
        "n_features_existing": X_existing.shape[1],
        "n_features_m2str": X_m2str.shape[1],
        "trees": args.trees,
        "depth": args.depth,
        "baseline_mae": mae_bl,
        "existing_only": {
            "mae": mae_existing,
            "skill": skill_existing,
            "r2": r2_existing,
            "wall_seconds": round(wall_existing, 1),
        },
        "existing_plus_m2str": {
            "mae": mae_combined,
            "skill": skill_combined,
            "r2": r2_combined,
            "wall_seconds": round(wall_combined, 1),
        },
        "m2str_only": {
            "mae": mae_str_only,
            "skill": skill_str_only,
            "r2": r2_str_only,
            "wall_seconds": round(wall_str_only, 1),
        },
        "gain_from_m2str": {
            "skill_delta": skill_combined - skill_existing,
            "r2_delta": r2_combined - r2_existing,
        },
        "loo_exclusion_gain": {
            "mean": float(gains.mean()),
            "std": float(gains.std()),
            "min": float(gains.min()),
            "max": float(gains.max()),
            "pct_positive": float((gains > 0).mean()),
            "n_folds": int(len(gains)),
        },
        "m2str_feature_stats": {
            "nonzero_frac_paired_depth": float(nonzero_frac),
            "nonzero_frac_f1": float(f1_frac),
        },
    }
    (out / "m2r_m2structure_ablation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[m2r_str] DONE -> {out / 'm2r_m2structure_ablation.json'}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()