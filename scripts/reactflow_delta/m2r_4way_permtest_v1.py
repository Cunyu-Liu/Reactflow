#!/usr/bin/env python3
"""m2r_4way_permtest_v1.py — significance of the architecture-decorrelated
4-way ensemble gain over the strong 3-way.

Operates on the SAVED OOF npz from m2r_4way_ensemble_v1.py
(l1, l2, xg, ridge, blend, prev_3way, y, keys).
Questions:
  1. 4-way-vs-strong-3way gain significance (paired design-block perm on
     per-design skill deltas + LOO-exclusion 100%-positive check)
  2. pooled skill of the 4-way + bootstrap 95% CI
  3. per-design (unpooled) gain + pct positive
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


def run_4way_permtest(npz_path: str, out_dir: str,
                      n_perm: int = 500, n_boot: int = 500) -> dict:
    z = np.load(npz_path)
    y = z["y"]; keys = z["keys"]
    blend = z["blend"]; prev = z["prev_3way"]
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)
    y_med = float(np.median(y))
    mae_bl = _mae(y, np.full_like(y, y_med))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d_masks = {d: keys == d for d in des_list}

    # ---- pooled ----
    skill_4 = _skill(_mae(y, blend), mae_bl)
    skill_3 = _skill(_mae(y, prev), mae_bl)
    r2_4 = 1.0 - float(np.sum((y - blend) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    r2_3 = 1.0 - float(np.sum((y - prev) ** 2)) / float(np.sum((y - y.mean()) ** 2))

    # ---- per-design paired deltas ----
    d_3, d_4, dg = [], [], []
    for d in des_list:
        m = d_masks[d]
        if m.sum() == 0:
            continue
        s_3 = _skill(_mae(y[m], prev[m]), y_med)
        s_4 = _skill(_mae(y[m], blend[m]), y_med)
        d_3.append(s_3); d_4.append(s_4); dg.append(s_4 - s_3)
    d_3 = np.array(d_3); d_4 = np.array(d_4); dg = np.array(dg)
    mean_delta = float(dg.mean())
    pct_pos = float((dg > 0).mean())

    # ---- paired design-block permutation on per-design skill deltas ----
    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(n_perm):
        swap = rng.random(n_des) < 0.5
        perm_delta = np.where(swap, d_4 - d_3, d_3 - d_4)
        if perm_delta.mean() >= mean_delta:
            cnt += 1
    perm_p = (cnt + 1) / (n_perm + 1)

    # ---- LOO-exclusion gain (4-way vs strong 3-way) ----
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        gains.append(_skill(_mae(y[m], blend[m]), mae_bl) -
                     _skill(_mae(y[m], prev[m]), mae_bl))
    gains = np.array(gains)

    # ---- bootstrap CI for 4-way pooled skill ----
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
        mm = _mae(y[sel], blend[sel])
        if mb > 0:
            boot.append(1.0 - mm / mb)
    boot = np.array(boot)

    report = {
        "schema": "reactflow_delta.m2r_4way_permtest.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": int(len(y)), "n_designs": n_des,
        "baseline_mae": mae_bl,
        "models": {
            "strong_3way": {"skill": float(skill_3), "r2": float(r2_3)},
            "fourway_a_priori": {
                "skill": float(skill_4), "r2": float(r2_4),
                "ci_low": float(np.percentile(boot, 2.5)) if len(boot) else None,
                "ci_high": float(np.percentile(boot, 97.5)) if len(boot) else None},
        },
        "fourway_vs_strong_3way": {
            "pooled_gain_pp": float((skill_4 - skill_3) * 100),
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
    (out / "m2r_4way_permtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    sv = report["fourway_vs_strong_3way"]
    print("=== M2R architecture-decorrelated 4-way perm test ===")
    print(f"strong 3-way: skill={skill_3:+.4f} R2={r2_3:.4f}")
    print(f"4-way a-priori: skill={skill_4:+.4f} R2={r2_4:.4f} "
          f"CI=({report['models']['fourway_a_priori']['ci_low']:.4f},"
          f"{report['models']['fourway_a_priori']['ci_high']:.4f})")
    print(f"gain: pooled {sv['pooled_gain_pp']:+.2f}pp | per-design "
          f"{sv['per_design_mean_pp']:+.2f}pp pct_pos={sv['per_design_pct_positive']:.3f}")
    print(f"perm p = {sv['permutation_p']:.4f} (paired design-block, n={n_perm})")
    loo = sv["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"\nDONE -> {out / 'm2r_4way_permtest.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    run_4way_permtest(args.npz, args.out, args.n_perm, args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
