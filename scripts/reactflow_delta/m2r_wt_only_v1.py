#!/usr/bin/env python3
"""m2r_wt_only_v1.py — the strongest finding from ablation: removing single-mutant
spectra barely changes skill (+22.0% -> +21.7%).  Verify this with the FULL
100-tree GBDT LOO: if WT-only features match all-features, the task is solvable
from WT sequence + reactivity + structure alone (pre-experiment prediction),
which is a cleaner and more publishable formulation.
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
    ap.add_argument("--trees", type=int, default=100)
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

    # no-singles index: exclude single-mutant reactivity/error window features
    single_idx = set()
    for i, n in enumerate(names):
        # single-mutant window + site + disruption features
        if n.startswith("A") or n.startswith("B") or n.startswith("chg"):
            single_idx.add(i)
    no_singles = [i for i in range(len(names)) if i not in single_idx]
    Xns = X[:, no_singles]
    print(f"all feats={len(names)} no-singles={len(no_singles)}", flush=True)

    results = {}
    for label, Xs in (("all_features", X), ("wt_only", Xns)):
        preds = np.zeros(len(y))
        t0 = time.time()
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=3,
                                  random_state=SEED, verbose=-1, n_jobs=2)
            g.fit(Xs[m], y[m])
            preds[~m] = g.predict(Xs[~m])
        wall = round(time.time() - t0, 1)
        mae = _mae(y, preds)
        skill = 1.0 - mae / mae_bl
        r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        # per-design
        dskills = []
        for held in des_list:
            m = keys == held
            if m.sum() > 0:
                dskills.append(1.0 - _mae(y[m], preds[m]) / mae_bl)
        dskills = np.array(dskills)
        # exclusion robustness
        excl = []
        for held in des_list:
            m = keys != held
            if m.sum() < 10:
                continue
            excl.append(1.0 - _mae(y[m], preds[m]) / mae_bl)
        excl = np.array(excl)
        results[label] = {
            "mae": float(mae), "skill": float(skill), "r2": float(r2),
            "per_design_mean": float(dskills.mean()),
            "per_design_pct_positive": float((dskills > 0).mean()),
            "loo_exclusion_min": float(excl.min()),
            "loo_exclusion_max": float(excl.max()),
            "wall_seconds": wall,
        }
        print(f"  {label:12s} skill={skill:+.4f} R2={r2:.4f} "
              f"pct+={results[label]['per_design_pct_positive']:.3f} "
              f"excl=[{excl.min():+.4f},{excl.max():+.4f}] wall={wall}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_wt_only.json").write_text(
        json.dumps({"schema": "reactflow_delta.m2r_wt_only.v1",
                    "baseline_mae": float(mae_bl), "trees": args.trees,
                    "results": results}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out / 'm2r_wt_only.json'}")


if __name__ == "__main__":
    sys.exit(main())