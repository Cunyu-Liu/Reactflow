#!/usr/bin/env python3
"""m2r_transfer_puzzle_permtest_v1.py — puzzle-block permutation test for the
puzzle-level M2R full stack (existing + puzzle-level transfer + Ridge blend).

Operates on SAVED OOF predictions (no re-training).  Block permutation over the
20 puzzles: per-puzzle mean predictions are permuted against per-puzzle mean
labels (exchangeable unit = puzzle).  This is the correct significance test for
the puzzle-level generalization claim.
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


def run_puzzle_permtest(npz_path: str, out_dir: str,
                        n_perm: int = 2000, n_boot: int = 1000) -> dict:
    z = np.load(npz_path)
    y = z["y"]
    sp = z["sample_puzzles"]
    puzzles = sorted(set(sp.tolist()))
    n_pz = len(puzzles)
    y_med = float(np.median(y))
    mae_bl = _mae(y, np.full_like(y, y_med))

    models = {"existing_230": z["pred_ex"], "puzzle_transfer": z["pred_pz"],
              "full_stack_blend": z["blend_pz"]}
    if "pred_dl" in z:
        models["design_transfer_LEAKY"] = z["pred_dl"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "reactflow_delta.m2r_transfer_puzzle_permtest.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": int(len(y)), "n_puzzles": n_pz,
        "baseline_mae": mae_bl,
        "models": {},
    }

    for name, pred in models.items():
        pm = {p: sp == p for p in puzzles}
        dam = np.array([y[pm[p]].mean() for p in puzzles])
        dpm = np.array([pred[pm[p]].mean() for p in puzzles])
        mae_bl_d = float(np.mean(np.abs(dam - np.median(dam))))
        skill_d_real = 1.0 - float(np.mean(np.abs(dam - dpm))) / mae_bl_d if mae_bl_d > 0 else 0.0

        rng = np.random.default_rng(SEED)
        cnt = 0
        for _ in range(n_perm):
            sk = 1.0 - float(np.mean(np.abs(dam - dpm[rng.permutation(n_pz)]))) / mae_bl_d if mae_bl_d > 0 else 0.0
            if sk >= skill_d_real:
                cnt += 1
        perm_p = (cnt + 1) / (n_perm + 1)

        # pooled skill (consistent with the bootstrap CI below)
        mae_m_pool = _mae(y, pred)
        skill_pool = _skill(mae_m_pool, mae_bl)

        # bootstrap CI over puzzles (resample puzzles with replacement)
        boot = []
        rng2 = np.random.default_rng(SEED + 1)
        for _ in range(n_boot):
            idx = rng2.integers(0, n_pz, size=n_pz)
            sel = np.zeros(len(y), dtype=bool)
            for i in idx:
                sel |= sp == puzzles[i]
            if sel.sum() < 10:
                continue
            mae_b = _mae(y[sel], np.full(sel.sum(), y_med))
            mae_m = _mae(y[sel], pred[sel])
            if mae_b > 0:
                boot.append(1.0 - mae_m / mae_b)
        boot = np.array(boot)
        ci_low = float(np.percentile(boot, 2.5)) if len(boot) else None
        ci_high = float(np.percentile(boot, 97.5)) if len(boot) else None

        # per-puzzle skills
        dskills = [1.0 - _mae(y[pm[p]], pred[pm[p]]) / _mae(y[pm[p]], np.full(pm[p].sum(), y_med))
                   for p in puzzles if pm[p].sum() > 0]
        dskills = np.array(dskills)

        report["models"][name] = {
            "skill": float(skill_pool),
            "per_puzzle_mean_skill": float(skill_d_real),
            "permutation_p": float(perm_p), "n_perm": n_perm,
            "ci_low": ci_low, "ci_high": ci_high, "n_boot": n_boot,
            "per_puzzle_skill_mean": float(dskills.mean()),
            "per_puzzle_skill_pct_positive": float((dskills > 0).mean()),
            "per_puzzle_skill_min": float(dskills.min()),
            "per_puzzle_skill_max": float(dskills.max()),
        }

    (out / "m2r_transfer_puzzle_permtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2R puzzle-level full-stack perm test ===")
    for k, v in report["models"].items():
        print(f"  {k:24s} skill={v['skill']:+.4f} perm_p={v['permutation_p']:.4f} "
              f"CI=({v['ci_low']:.4f},{v['ci_high']:.4f}) "
              f"pz_mean={v['per_puzzle_skill_mean']:+.4f} "
              f"pct+={v['per_puzzle_skill_pct_positive']:.3f}")
    print(f"\n  DONE -> {out / 'm2r_transfer_puzzle_permtest.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    run_puzzle_permtest(args.npz, args.out, args.n_perm, args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
