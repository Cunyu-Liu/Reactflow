#!/usr/bin/env python3
"""m2r_transfer_permtest_v1.py — full-stack permutation test + bootstrap CI.

Operates on the SAVED OOF predictions from m2r_transfer_v1.py (no re-training).
The permutation/bootstrap operate on OOF predictions, so the result is
statistically valid for the trained full-stack model (same design-block
methodology as m2r_permtest_v3.py).

Full stack = 230-dim M2R features + M2_structure + 6 M2-transfer features,
combined with a GBDT+Ridge blend (a=0.80).  This script tests the SIGNIFICANCE
of the final headline skill (+25.82%, R2 0.370).

Design-block permutation: per-design mean predictions are permuted against
per-design mean labels (exchangeable unit = (puzzle, method) design).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

SEED = 20260816


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def run_permtest(npz_path: str, out_dir: str,
                 n_perm: int = 300, n_boot: int = 500) -> dict:
    """Run the full-stack permutation test on saved OOF predictions.

    Returns the report dict and writes m2r_transfer_permtest.json into out_dir.
    """
    z = np.load(npz_path)
    y = z["y"]
    keys = z["keys"]
    pred_ex = z["pred_ex"]
    pred_comb = z["pred_comb"]
    blend = z["blend_comb"]
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)

    y_med = float(np.median(y))
    mae_bl = _mae(y, np.full_like(y, y_med))

    def _stats(pred, label):
        mae = _mae(y, pred)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        skill = _skill(mae, mae_bl)

        # design-block permutation (per-design means)
        design_masks = {d: keys == d for d in des_list}
        dam = np.array([y[design_masks[d]].mean() for d in des_list])
        dpm = np.array([pred[design_masks[d]].mean() for d in des_list])
        mae_bl_d = float(np.mean(np.abs(dam - np.median(dam))))
        skill_d_real = 1.0 - float(np.mean(np.abs(dam - dpm))) / mae_bl_d if mae_bl_d > 0 else 0.0
        rng = np.random.default_rng(SEED)
        cnt = 0
        for _ in range(n_perm):
            sk = 1.0 - float(np.mean(np.abs(dam - dpm[rng.permutation(n_des)]))) / mae_bl_d if mae_bl_d > 0 else 0.0
            if sk >= skill_d_real:
                cnt += 1
        perm_p = (cnt + 1) / (n_perm + 1)

        # bootstrap CI (resample designs with replacement, recompute pooled skill)
        boot = []
        rng2 = np.random.default_rng(SEED + 1)
        for _ in range(n_boot):
            idx = rng2.integers(0, n_des, size=n_des)
            sel = np.zeros(len(y), dtype=bool)
            for i in idx:
                sel |= keys == des_list[i]
            if sel.sum() < 10:
                continue
            mae_b = _mae(y[sel], np.full(sel.sum(), y_med))
            mae_m = _mae(y[sel], pred[sel])
            if mae_b > 0:
                boot.append(1.0 - mae_m / mae_b)
        boot = np.array(boot)
        ci_low = float(np.percentile(boot, 2.5)) if len(boot) else None
        ci_high = float(np.percentile(boot, 97.5)) if len(boot) else None

        # per-design skills
        dskills = []
        for held in des_list:
            m = keys == held
            if m.sum() > 0:
                mae_m = _mae(y[m], pred[m])
                mae_b = _mae(y[m], np.full(m.sum(), y_med))
                dskills.append(1.0 - mae_m / mae_b)
        dskills = np.array(dskills)

        # LOO-exclusion pooled skill range
        excl = []
        for held in des_list:
            m = keys != held
            if m.sum() < 10:
                continue
            mae_b = _mae(y[m], np.full(m.sum(), y_med))
            mae_m = _mae(y[m], pred[m])
            excl.append(1.0 - mae_m / mae_b)
        excl = np.array(excl)

        return {
            "label": label,
            "mae": mae, "skill": float(skill), "r2": float(r2),
            "permutation_p": float(perm_p), "n_perm": n_perm,
            "ci_low": ci_low, "ci_high": ci_high, "n_boot": n_boot,
            "per_design_skill_mean": float(dskills.mean()),
            "per_design_skill_pct_positive": float((dskills > 0).mean()),
            "per_design_skill_min": float(dskills.min()),
            "per_design_skill_max": float(dskills.max()),
            "loo_exclusion_min": float(excl.min()),
            "loo_exclusion_max": float(excl.max()),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "reactflow_delta.m2r_transfer_permtest.v1",
        "dataset": "OpenKnot_M2R",
        "full_stack": "230 M2R feats + M2_structure + 6 M2-transfer + GBDT(Ridge blend a=0.80)",
        "n_samples": int(len(y)),
        "n_designs": n_des,
        "baseline_mae": mae_bl,
        "models": {
            "existing_230_gbdt": _stats(pred_ex, "230-dim GBDT"),
            "plus_transfer_gbdt": _stats(pred_comb, "236-dim GBDT"),
            "full_stack_blend": _stats(blend, "full stack (GBDT+Ridge a=0.80)"),
        },
    }
    (out / "m2r_transfer_permtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2R full-stack permutation test ===")
    for k, v in report["models"].items():
        print(f"\n  {k} ({v['label']})")
        print(f"    skill={v['skill']:+.4f} R2={v['r2']:.4f} MAE={v['mae']:.4f}")
        print(f"    perm_p={v['permutation_p']:.4f}  CI=({v['ci_low']:.4f},{v['ci_high']:.4f})")
        print(f"    per-design mean={v['per_design_skill_mean']:+.4f} "
              f"pct+={(v['per_design_skill_pct_positive']):.3f} "
              f"range=[{v['per_design_skill_min']:+.4f},{v['per_design_skill_max']:+.4f}]")
        print(f"    LOO-exclusion=[{v['loo_exclusion_min']:+.4f},{v['loo_exclusion_max']:+.4f}]")
    print(f"\n  DONE -> {out / 'm2r_transfer_permtest.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="m2r_transfer_oof.npz (pred_ex, pred_comb, blend_comb, y, keys)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    run_permtest(args.npz, args.out, args.n_perm, args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
