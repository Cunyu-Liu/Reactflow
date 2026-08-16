#!/usr/bin/env python3
"""m2r_noise_floor_v1.py — empirically verify the rescue_factor measurement-noise
floor and the learnability ceiling of the M2R task.

The chapter previously claimed "noise/signal = 0.30 -> ~91% learnable" without a
backing script.  This script derives the exact rescue_factor formula from the
data (design-region RMSD, corr=1.0000/MAE=0.0001), then propagates the
per-position reactivity errors (Monte Carlo) to get the honest per-sample
measurement-noise sigma, and reports:

  * rescue_factor = 1 - RMSD(double,wt) / sqrt(RMSD(singleA,wt)^2 + RMSD(singleB,wt)^2)
    over the design region [sub_start, sub_end)
  * per-sample sigma_noise from error propagation
  * median sigma_noise, total std, noise/signal ratio
  * learnable variance fraction = 1 - (sigma_noise/total_std)^2  (R2 ceiling)
  * comparison with the achieved R2 (0.37) -> honest headroom statement
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r

RNG_SEED = 20260817
N_MC = 400          # Monte Carlo draws per sample for sigma_noise


def _prof(p):
    return np.array([x if x is not None and np.isfinite(x) else np.nan for x in p],
                    dtype=np.float64)


def _design_mask(n, sub_start, sub_end):
    m = np.zeros(n, dtype=bool)
    lo = max(sub_start - 1, 0) if sub_start is not None else 0
    hi = sub_end if sub_end is not None else n
    m[lo:hi] = True
    return m


def rmsd_region(a, b, mask):
    m = np.isfinite(a) & np.isfinite(b) & mask
    if m.sum() < 3:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def rescue_from_profiles(wt, ra, rb, rd, mask):
    rA = rmsd_region(wt, ra, mask)
    rB = rmsd_region(wt, rb, mask)
    rD = rmsd_region(wt, rd, mask)
    if not all(np.isfinite(x) for x in (rA, rB, rD)) or rA + rB <= 0:
        return np.nan
    return 1.0 - rD / np.sqrt(rA ** 2 + rB ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mc", type=int, default=N_MC)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]

    # ---- verify formula + compute noise floor ----
    rng = np.random.default_rng(RNG_SEED)
    sigma_noise = []
    residuals = []
    n_verified = 0
    for s in samples:
        n = len(s.wt_reactivity)
        mask = _design_mask(n, s.sub_start, s.sub_end)
        wt = _prof(s.wt_reactivity)
        ra = _prof(s.singleA_reactivity)
        rb = _prof(s.singleB_reactivity)
        rd = _prof(s.double_reactivity)
        we = _prof(s.wt_error)
        ae = _prof(s.singleA_error)
        be = _prof(s.singleB_error)
        de = _prof(s.double_error)

        pred = rescue_from_profiles(wt, ra, rb, rd, mask)
        if not np.isfinite(pred):
            continue
        residuals.append(s.rescue_factor - pred)
        if abs(s.rescue_factor - pred) < 0.01:
            n_verified += 1

        # Monte Carlo error propagation: perturb each profile by its error
        draws = []
        for _ in range(args.n_mc):
            wtp = np.where(np.isfinite(wt), wt + rng.normal(0, 1, n) * np.where(np.isfinite(we), we, 0.0), np.nan)
            rap = np.where(np.isfinite(ra), ra + rng.normal(0, 1, n) * np.where(np.isfinite(ae), ae, 0.0), np.nan)
            rbp = np.where(np.isfinite(rb), rb + rng.normal(0, 1, n) * np.where(np.isfinite(be), be, 0.0), np.nan)
            rdp = np.where(np.isfinite(rd), rd + rng.normal(0, 1, n) * np.where(np.isfinite(de), de, 0.0), np.nan)
            v = rescue_from_profiles(wtp, rap, rbp, rdp, mask)
            if np.isfinite(v):
                draws.append(v)
        if len(draws) >= 30:
            sigma_noise.append(float(np.std(draws)))

    sigma_noise = np.array(sigma_noise)
    residuals = np.array(residuals)
    rf = np.array([s.rescue_factor for s in samples if np.isfinite(s.rescue_factor)])

    total_std = float(np.std(rf))
    med_noise = float(np.median(sigma_noise))
    mean_noise = float(np.mean(sigma_noise))
    noise_ratio = med_noise / total_std
    learnable_frac = 1.0 - (med_noise / total_std) ** 2

    # also: R2 ceiling via noise variance / total variance (using mean noise)
    r2_ceil_mean = 1.0 - (mean_noise ** 2) / (total_std ** 2)

    report = {
        "schema": "reactflow_delta.m2r_noise_floor.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": int(len(rf)),
        "n_noise_floor_samples": int(len(sigma_noise)),
        "n_mc": args.n_mc,
        "formula": "rescue = 1 - RMSD(double,wt) / sqrt(RMSD(sA,wt)^2 + RMSD(sB,wt)^2) over design region",
        "formula_verification": {
            "corr": float(np.corrcoef(rf[:len(residuals)], rf[:len(residuals)] - residuals)[0, 1]),
            "mae": float(np.mean(np.abs(residuals))),
            "n_within_0.01": int(n_verified),
        },
        "rescue_total_std": total_std,
        "sigma_noise": {
            "median": med_noise, "mean": mean_noise,
            "pct25": float(np.percentile(sigma_noise, 25)),
            "pct75": float(np.percentile(sigma_noise, 75)),
        },
        "noise_signal_ratio": noise_ratio,
        "learnable_variance_fraction_median": learnable_frac,
        "r2_ceiling_mean_noise": r2_ceil_mean,
        "achieved_r2": 0.370,
        "honest_headroom_statement": (
            "R2 ceiling (from measurement-noise-only) is roughly "
            f"{100*learnable_frac:.0f}% of variance, achieved R2 is 0.37; "
            "however the ceiling is a NECESSARY condition for a better model, "
            "not proof that a better model exists (feature representation may "
            "cap the achievable R2 well below the measurement-noise ceiling)."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_noise_floor.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2R noise floor ===")
    print(f"formula verification: corr={report['formula_verification']['corr']:.4f} "
          f"mae={report['formula_verification']['mae']:.4f} "
          f"n<0.01={n_verified}/{len(residuals)}")
    print(f"rescue total std={total_std:.4f}")
    print(f"sigma_noise: median={med_noise:.4f} mean={mean_noise:.4f} "
          f"[p25={report['sigma_noise']['pct25']:.4f}, p75={report['sigma_noise']['pct75']:.4f}]")
    print(f"noise/signal = {noise_ratio:.3f}")
    print(f"learnable variance fraction (median) = {learnable_frac:.4f}  "
          f"({100*learnable_frac:.1f}%)")
    print(f"R2 ceiling (mean-noise) = {r2_ceil_mean:.4f}")
    print(f"achieved R2 = 0.370")
    print(f"\nDONE -> {out / 'm2r_noise_floor.json'}")


if __name__ == "__main__":
    sys.exit(main())
