#!/usr/bin/env python3
"""m2r_multiseed_permtest_v1.py — significance of the multi-seed averaging gain.

Operates on the SAVED OOF npz from m2r_multiseed_v1.py
(blend_1, blend_K, y, keys).  Tests:
  1. multi-seed-vs-single-seed strong-3way gain significance (paired
     design-block permutation on per-design skill deltas)
  2. pooled skill of the multi-seed 3-way + bootstrap 95% CI
  3. LOO-exclusion 100%-positive check (already computed, re-verified here)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

SEED = 20260817


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def run_multiseed_permtest(npz_path: str, out_dir: str,
                           n_perm: int = 500, n_boot: int = 500) -> dict:
    z = np.load(npz_path)
    y = z["y"]; keys = z["keys"]
    b1 = z["blend_1"]; bK = z["blend_K"]
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)
    y_med = float(np.median(y))
    mae_bl = _mae(y, np.full_like(y, y_med))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d_masks = {d: keys == d for d in des_list}

    skill_1 = _skill(_mae(y, b1), mae_bl)
    skill_K = _skill(_mae(y, bK), mae_bl)
    r2_1 = 1.0 - float(np.sum((y - b1) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    r2_K = 1.0 - float(np.sum((y - bK) ** 2)) / float(np.sum((y - y.mean()) ** 2))

    # ---- per-design paired deltas ----
    d_1, d_K, dg = [], [], []
    for d in des_list:
        m = d_masks[d]
        if m.sum() == 0:
            continue
        s_1 = _skill(_mae(y[m], b1[m]), y_med)
        s_K = _skill(_mae(y[m], bK[m]), y_med)
        d_1.append(s_1); d_K.append(s_K); dg.append(s_K - s_1)
    d_1 = np.array(d_1); d_K = np.array(d_K); dg = np.array(dg)
    mean_delta = float(dg.mean())
    pct_pos = float((dg > 0).mean())

    # ---- paired design-block permutation ----
    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(n_perm):
        swap = rng.random(n_des) < 0.5
        perm_delta = np.where(swap, d_K - d_1, d_1 - d_K)
        if perm_delta.mean() >= mean_delta:
            cnt += 1
    perm_p = (cnt + 1) / (n_perm + 1)

    # ---- LOO-exclusion gain ----
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        gains.append(_skill(_mae(y[m], bK[m]), mae_bl) -
                     _skill(_mae(y[m], b1[m]), mae_bl))
    gains = np.array(gains)

    # ---- bootstrap CI for multi-seed pooled skill ----
    boot = []
    rng2 = np.random.default_rng(SEED + 1)
    for _ in range(n_boot):
        idx = rng2.integers(0, n_des, size=n_des)
        sel = np.zeros(len(y), dtype=bool)
        for i in idx:
            sel |= keys == des_list[i]
        if sel.sum() < 10:
            continue
        mb = _mae(y[sel], np.full(sel.sum(), y_med))
        mm = _mae(y[sel], bK[sel])
        if mb > 0:
            boot.append(1.0 - mm / mb)
    boot = np.array(boot)

    report = {
        "schema": "reactflow_delta.m2r_multiseed_permtest.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": int(len(y)), "n_designs": n_des,
        "baseline_mae": mae_bl,
        "models": {
            "single_seed_3way": {"skill": float(skill_1), "r2": float(r2_1)},
            "multiseed_3way": {
                "skill": float(skill_K), "r2": float(r2_K),
                "ci_low": float(np.percentile(boot, 2.5)) if len(boot) else None,
                "ci_high": float(np.percentile(boot, 97.5)) if len(boot) else None},
        },
        "multiseed_vs_single": {
            "pooled_gain_pp": float((skill_K - skill_1) * 100),
            "r2_gain": float(r2_K - r2_1),
            "per_design_mean_pp": float(mean_delta * 100),
            "per_design_pct_positive": pct_pos,
            "permutation_p": float(perm_p),
            "n_perm": n_perm,
            "loo_exclusion": {
                "gain_mean_pp": float(gains.mean() * 100),
                "gain_min_pp": float(gains.min() * 100),
                "gain_max_pp": float(gains.max() * 100),
                "pct_positive": float((gains > 0).mean()),
                "n_folds": int(len(gains)),
            },
        },
    }
    (out / "m2r_multiseed_permtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    ms = report["multiseed_vs_single"]
    print("=== M2R multi-seed averaging perm test ===")
    print(f"single-seed 3-way: skill={skill_1:+.4f} R2={r2_1:.4f}")
    print(f"multi-seed 3-way : skill={skill_K:+.4f} R2={r2_K:.4f} "
          f"CI=({report['models']['multiseed_3way']['ci_low']:.4f},"
          f"{report['models']['multiseed_3way']['ci_high']:.4f})")
    print(f"gain: pooled {ms['pooled_gain_pp']:+.2f}pp (R2 {ms['r2_gain']:+.4f}) | "
          f"per-design {ms['per_design_mean_pp']:+.2f}pp "
          f"pct_pos={ms['per_design_pct_positive']:.3f}")
    print(f"perm p = {ms['permutation_p']:.4f} (paired design-block, n={n_perm})")
    loo = ms["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"\nDONE -> {out / 'm2r_multiseed_permtest.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    run_multiseed_permtest(args.npz, args.out, args.n_perm, args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
