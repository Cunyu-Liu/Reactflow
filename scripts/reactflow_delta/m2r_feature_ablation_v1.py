#!/usr/bin/env python3
"""m2r_feature_ablation_v1.py — which feature groups drive the M2R GBDT signal?

Trains a SMALL GBDT (30 trees, depth 3) LOO on:
  1. all features
  2. reactivity only (WT + singles)
  3. structure only (target depth/paired)
  4. sequence/pairing only
  5. disruption signals only
  6. no singles (WT-only, to measure the single-mutant information gain)

Server is heavily loaded, so this uses a tiny GBDT.  The point is RELATIVE
comparison (which feature group matters), not the final headline.
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
    ap.add_argument("--trees", type=int, default=30)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, names = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)

    import lightgbm as lgb
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    # feature groups by index prefix
    groups = {
        "react_wt": [i for i, n in enumerate(names) if n.startswith("wt_") or n in
                     ("react_i", "react_j", "err_i", "err_j")],
        "react_singles": [i for i, n in enumerate(names)
                          if n.startswith("A") or n.startswith("B") or n.startswith("chg")
                          or n.startswith("dA") or n.startswith("dB")],
        "structure": [i for i, n in enumerate(names) if n.startswith("str_")
                      or n == "edit_dist_norm" or n == "rel_i" or n == "rel_j"],
        "sequence": [i for i, n in enumerate(names) if n.startswith("oh_")
                     or n in ("wc_pair", "wobble")],
    }
    # "no_singles": all features EXCEPT single-mutant reactivity/error windows
    single_idx = set()
    for i, n in enumerate(names):
        if (n.startswith("A") or n.startswith("B")) and ("_" in n or n in
                ("A_i", "A_j", "Ae_i", "Ae_j", "Bi_0")):
            single_idx.add(i)
    no_singles = [i for i in range(len(names)) if i not in single_idx]

    configs = {
        "all": list(range(len(names))),
        "react_wt_only": groups["react_wt"],
        "react_singles_only": groups["react_singles"],
        "structure_only": groups["structure"],
        "sequence_only": groups["sequence"],
        "no_singles": no_singles,
    }

    results = {}
    for cfg_name, idx in configs.items():
        if not idx:
            continue
        Xs = X[:, idx]
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=3,
                                  random_state=SEED, verbose=-1, n_jobs=2)
            g.fit(Xs[m], y[m])
            preds[~m] = g.predict(Xs[~m])
        mae = _mae(y, preds)
        skill = 1.0 - mae / mae_bl
        r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        results[cfg_name] = {"mae": float(mae), "skill": float(skill), "r2": float(r2),
                             "n_feats": len(idx)}
        print(f"  {cfg_name:20s} nfeat={len(idx):3d} skill={skill:+.4f} R2={r2:.4f}",
              flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_feature_ablation.json").write_text(
        json.dumps({"schema": "reactflow_delta.m2r_feature_ablation.v1",
                    "baseline_mae": float(mae_bl), "trees": args.trees,
                    "results": results}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out / 'm2r_feature_ablation.json'}")


if __name__ == "__main__":
    sys.exit(main())