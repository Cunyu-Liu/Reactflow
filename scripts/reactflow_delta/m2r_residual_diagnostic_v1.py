#!/usr/bin/env python3
"""m2r_residual_diagnostic_v1.py — decompose the M2R GBDT residual to decide if
there is remaining learnable headroom (junction-project r68/r69 style).

Uses the SAVED OOF predictions from m2r_permtest_v3 (30-tree GBDT).  Questions:
  1. Is residual variance concentrated in low-reactivity / high-uncertainty
     positions (i.e. measurement noise) or spread uniformly?
  2. Per-design residual spread: is there a stable per-design constant offset
     that a leak-free calibration could remove?
  3. Residual vs rescue magnitude: does the model systematically over/under
     predict for high/low rescue?
  4. Residual correlation with single-mutant disruption magnitude — is there a
     feature interaction the GBDT is missing?
"""
from __future__ import annotations

import argparse, json, sys
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
    ap.add_argument("--npz", required=True, help="saved OOF preds (preds,y,keys)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, names = m2rf.build_all(samples)

    d = np.load(args.npz)
    preds = d["preds"]; y = d["y"]; keys = d["keys"]
    print(f"loaded OOF: n={len(y)} preds={preds.shape}", flush=True)

    resid = y - preds
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    skill = 1.0 - _mae(y, preds) / mae_bl
    print(f"skill={skill:+.4f} resid_sd={resid.std():.4f} "
          f"y_sd={y.std():.4f}", flush=True)

    # ---- 1. residual by true rescue magnitude ----
    bins = np.quantile(y, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    by_rescue = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (y >= lo) & (y <= hi)
        if m.sum() > 0:
            by_rescue.append({
                "bin": f"[{lo:.3f},{hi:.3f}]", "n": int(m.sum()),
                "mean_resid": float(resid[m].mean()),
                "resid_sd": float(resid[m].std()),
                "pred_mean": float(preds[m].mean()),
                "y_mean": float(y[m].mean()),
            })
    print("\n-- residual by true rescue magnitude --")
    for r in by_rescue:
        print(f"  y bin {r['bin']:14s} n={r['n']:5d} mean_resid={r['mean_resid']:+.4f} "
              f"resid_sd={r['resid_sd']:.4f} pred={r['pred_mean']:.3f} y={r['y_mean']:.3f}")

    # ---- 2. per-design residual offset (leak-free vs in-sample) ----
    des_list = sorted(set(keys.tolist()))
    d_res = {}
    for d in des_list:
        m = keys == d
        if m.sum() > 0:
            d_res[d] = float(resid[m].mean())
    d_arr = np.array(list(d_res.values()))
    print(f"\nper-design mean-resid: std={d_arr.std():.4f} "
          f"abs_mean={np.abs(d_arr).mean():.4f} max_abs={np.abs(d_arr).max():.4f}")

    # ---- 3. residual by single-mutant disruption magnitude (chgAi) ----
    chgAi_idx = names.index("chgAi")
    chgAi = X[:, chgAi_idx]
    cb = np.quantile(chgAi, [0.0, 0.5, 0.9, 1.0])
    print("\n-- residual by single-mutant disruption at i (chgAi) --")
    for i in range(len(cb) - 1):
        lo, hi = cb[i], cb[i + 1]
        m = (chgAi >= lo) & (chgAi <= hi)
        if m.sum() > 0:
            print(f"  chgAi [{lo:.4f},{hi:.4f}] n={m.sum():5d} resid_sd={resid[m].std():.4f} "
                  f"mean_resid={resid[m].mean():+.4f}")

    # ---- 4. split-half per-design offset stability ----
    rng = np.random.default_rng(SEED)
    stable = []
    for d in des_list:
        m = keys == d
        r = resid[m]
        if len(r) >= 20:
            perm = rng.permutation(len(r))
            n2 = len(r) // 2
            h1 = r[perm[:n2]]; h2 = r[perm[n2:2*n2]]
            if h1.std() > 0 and h2.std() > 0:
                stable.append(float(np.corrcoef(h1, h2)[0, 1]))
    stable = np.array(stable)
    print(f"\nsplit-half per-design offset rho: median={np.median(stable):.4f} "
          f"(n={len(stable)} designs with >=20 obs)")

    # ---- 5. leak-free per-design shift calibration ----
    # estimate design offset from OTHER designs (leave-one-design-out)
    pred_shifted = preds.copy()
    for held in des_list:
        train_off = np.mean([v for k, v in d_res.items() if k != held])
        m = keys == held
        pred_shifted[m] = preds[m] + train_off
    skill_shift = 1.0 - _mae(y, pred_shifted) / mae_bl
    # in-sample per-design offset (leaky upper bound)
    pred_ins = preds + np.array([d_res[k] for k in keys])
    skill_ins = 1.0 - _mae(y, pred_ins) / mae_bl
    print(f"\nper-design shift calibration: leakfree_skill={skill_shift:+.4f} "
          f"in_sample_leaky={skill_ins:+.4f}")

    report = {
        "schema": "reactflow_delta.m2r_residual_diagnostic.v1",
        "n_samples": int(len(y)), "skill": float(skill),
        "resid_sd": float(resid.std()), "y_sd": float(y.std()),
        "residual_by_rescue_magnitude": by_rescue,
        "per_design_offset": {"std": float(d_arr.std()),
                              "abs_mean": float(np.abs(d_arr).mean()),
                              "max_abs": float(np.abs(d_arr).max()),
                              "split_half_median_rho": float(np.median(stable)),
                              "n_stable_designs": int(len(stable))},
        "per_design_shift_calibration": {
            "leakfree_skill": float(skill_shift),
            "in_sample_leaky_skill": float(skill_ins),
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_residual_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out / 'm2r_residual_diagnostic.json'}")


if __name__ == "__main__":
    sys.exit(main())