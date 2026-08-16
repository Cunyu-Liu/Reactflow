#!/usr/bin/env python3
"""m2r_oof100_v1.py — save 100-tree GBDT OOF predictions + run calibration check.

run_m2r_v1 computed 100-tree GBDT LOO but did not save per-sample OOF preds.
This regenerates them (default 100-tree config) and saves npz for downstream
calibration / residual analysis on the FINAL headline model.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf

SEED = 20260816


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))

    import lightgbm as lgb
    y_med = np.median(y)
    preds = np.zeros(len(y))
    t0 = time.time()
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(n_estimators=100, max_depth=3,
                              random_state=SEED, verbose=-1, n_jobs=2)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    wall = round(time.time() - t0, 1)

    mae_bl = float(np.mean(np.abs(y - y_med)))
    skill = 1.0 - float(np.mean(np.abs(y - preds))) / mae_bl
    r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "m2r_gbdt_oof_t100.npz", preds=preds, y=y, keys=keys)
    print(f"skill={skill:+.4f} R2={r2:.4f} wall={wall}s "
          f"DONE -> {out / 'm2r_gbdt_oof_t100.npz'}")


if __name__ == "__main__":
    sys.exit(main())