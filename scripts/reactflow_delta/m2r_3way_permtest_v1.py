#!/usr/bin/env python3
"""m2r_3way_permtest_v1.py — statistical significance of the M2R 3-way ensemble.

Operates on the SAVED OOF npz from m2r_3way_ensemble_v1.py.  Questions:
  1. Is the 3-way blend skill significantly > 0 (design-block permutation)?
  2. Is the 3-way gain over the previous headline (L1+Ridge a=0.80) robust
     to leave-one-design-out (paired LOO-exclusion, 100%-positive check)?
  3. Bootstrap 95% CI for the 3-way blend pooled skill.
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


def run_3way_permtest(npz_path: str, out_dir: str,
                      n_perm: int = 500, n_boot: int = 500) -> dict:
    z = np.load(npz_path)
    y = z["y"]; keys = z["keys"]
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)
    y_med = float(np.median(y))
    mae_bl = _mae(y, np.full_like(y, y_med))

    models = {
        "l1_gbdt": z["l1"],
        "l2_gbdt": z["l2"],
        "ridge": z["ridge"],
        "prev_headline_l1_ridge_a80": z["prev_blend"],
        "threeway_blend": z["blend"],
        "threeway_blend_clipped": z["blend_clipped"],
    }

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "reactflow_delta.m2r_3way_permtest.v1",
              "dataset": "OpenKnot_M2R", "n_samples": int(len(y)),
              "n_designs": n_des, "baseline_mae": mae_bl, "models": {}}

    d_masks = {d: keys == d for d in des_list}
    dam = np.array([y[d_masks[d]].mean() for d in des_list])
    for name, pred in models.items():
        skill = _skill(_mae(y, pred), mae_bl)
        r2 = 1.0 - float(np.sum((y - pred) ** 2)) / float(np.sum((y - y.mean()) ** 2))
        # design-block perm on design-level mean skill
        dpm = np.array([pred[d_masks[d]].mean() for d in des_list])
        mae_bl_d = float(np.mean(np.abs(dam - np.median(dam))))
        skill_d = 1.0 - float(np.mean(np.abs(dam - dpm))) / mae_bl_d if mae_bl_d > 0 else 0.0
        rng = np.random.default_rng(SEED)
        cnt = 0
        for _ in range(n_perm):
            sk = 1.0 - float(np.mean(np.abs(dam - dpm[rng.permutation(n_des)]))) / mae_bl_d if mae_bl_d > 0 else 0.0
            if sk >= skill_d:
                cnt += 1
        perm_p = (cnt + 1) / (n_perm + 1)
        # bootstrap CI (pooled skill)
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
            mm = _mae(y[sel], pred[sel])
            if mb > 0:
                boot.append(1.0 - mm / mb)
        boot = np.array(boot)
        ci_low = float(np.percentile(boot, 2.5)) if len(boot) else None
        ci_high = float(np.percentile(boot, 97.5)) if len(boot) else None
        dsk = np.array([1.0 - _mae(y[d_masks[d]], pred[d_masks[d]]) /
                        _mae(y[d_masks[d]], np.full(d_masks[d].sum(), y_med))
                        for d in des_list if d_masks[d].sum() > 0])
        report["models"][name] = {
            "skill": float(skill), "r2": float(r2),
            "permutation_p": float(perm_p),
            "ci_low": ci_low, "ci_high": ci_high,
            "per_design_mean": float(dsk.mean()),
            "per_design_pct_positive": float((dsk > 0).mean()),
        }

    # ---- paired 3-way vs previous headline, leave-one-design-out ----
    p3 = models["threeway_blend"]; pp = models["prev_headline_l1_ridge_a80"]
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        gains.append(_skill(_mae(y[m], p3[m]), mae_bl) -
                     _skill(_mae(y[m], pp[m]), mae_bl))
    gains = np.array(gains)
    report["threeway_vs_prev_loo"] = {
        "gain_mean_pp": float(gains.mean() * 100),
        "gain_min_pp": float(gains.min() * 100),
        "gain_max_pp": float(gains.max() * 100),
        "pct_positive": float((gains > 0).mean()),
        "n_folds": int(len(gains)),
    }
    # per-design paired gain
    pg = []
    for d in des_list:
        m = d_masks[d]
        if m.sum() == 0:
            continue
        pg.append(_skill(_mae(y[m], p3[m]), y_med) -
                  _skill(_mae(y[m], pp[m]), y_med))
    pg = np.array(pg)
    report["threeway_vs_prev_per_design"] = {
        "gain_mean_pp": float(pg.mean() * 100),
        "pct_positive": float((pg > 0).mean()),
        "n_designs": int(len(pg)),
    }

    (out / "m2r_3way_permtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("=== M2R 3-way ensemble perm test ===")
    for k, v in report["models"].items():
        print(f"  {k:28s} skill={v['skill']:+.4f} perm_p={v['permutation_p']:.4f} "
              f"CI=({v['ci_low']:.4f},{v['ci_high']:.4f}) "
              f"pz+={(v['per_design_pct_positive']):.3f}")
    g = report["threeway_vs_prev_loo"]
    print(f"  3way-vs-prev LOO-exclusion: mean={g['gain_mean_pp']:+.2f}pp "
          f"range=[{g['gain_min_pp']:+.2f},{g['gain_max_pp']:+.2f}]pp "
          f"pct_pos={g['pct_positive']:.3f}")
    pg = report["threeway_vs_prev_per_design"]
    print(f"  per-design: mean={pg['gain_mean_pp']:+.2f}pp pct_pos={pg['pct_positive']:.3f}")
    print(f"\n  DONE -> {out / 'm2r_3way_permtest.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    run_3way_permtest(args.npz, args.out, args.n_perm, args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
