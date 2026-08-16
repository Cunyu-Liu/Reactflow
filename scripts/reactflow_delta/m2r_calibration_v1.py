#!/usr/bin/env python3
"""m2r_calibration_v1.py — is the M2R GBDT residual a fixable regression-to-the-mean?

Diagnostic showed: low-rescue predicted too high, high-rescue predicted too low
(mean resid -0.26 to +0.20 across y bins).  This is classic shrinkage.  We test
whether a monotone post-hoc calibration (fit on TRAIN half, applied to TEST half)
recovers real skill — if the shrink is stable, calibration is a legal lever.

Setup (leak-free): split samples by DESIGN into two halves; fit the calibration
curve (binned affine / isotonic / logistic-of-pred) on half-A OOF preds, apply to
half-B OOF preds, evaluate.  Repeat swapped, average.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260816


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _binned_affine(y, p, n_bins=8):
    """Fit per-bin additive+slope calibration, return a predictor."""
    order = np.argsort(p)
    p_s = p[order]; y_s = y[order]
    n = len(p_s)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    means = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            means[i] = float(y_s[lo:hi].mean())
    def apply(preds):
        out = np.zeros_like(preds)
        q = np.searchsorted(p_s, preds, side="right")
        for j, v in enumerate(preds):
            # map pred to a bin via percentile position
            frac = q[j] / n
            bin_i = min(int(frac * n_bins), n_bins - 1)
            out[j] = means[bin_i]
        return out
    return apply


def _logistic_cal(y, p):
    """Map pred -> calibrated prob via sigmoid(m + s*p), fit by OLS on logit."""
    eps = 1e-3
    yc = np.clip(y, eps, 1 - eps)
    logit_y = np.log(yc / (1 - yc))
    A = np.stack([np.ones_like(p), p], axis=1)
    beta, *_ = np.linalg.lstsq(A, logit_y, rcond=None)
    def apply(preds):
        lin = beta[0] + beta[1] * preds
        return 1.0 / (1.0 + np.exp(-lin))
    return apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    preds = d["preds"]; y = d["y"]; keys = d["keys"]
    n = len(y)
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    skill_raw = 1.0 - _mae(y, preds) / mae_bl
    print(f"n={n} raw_skill={skill_raw:+.4f}", flush=True)

    # split by design: each design fully in A or B
    des_list = sorted(set(keys.tolist()))
    rng = np.random.default_rng(SEED)
    perm_d = rng.permutation(len(des_list))
    half = len(des_list) // 2
    desA = set(des_list[i] for i in perm_d[:half])
    mA = np.array([k in desA for k in keys])
    mB = ~mA
    print(f"designsA={sum(mA)} samplesA={int(mA.sum())} samplesB={int(mB.sum())}", flush=True)

    results = {}
    for name, fit_fn in (("binned_affine", _binned_affine),
                         ("logistic", _logistic_cal)):
        # fit on A, apply to B; and fit on B, apply to A
        cAB = fit_fn(y[mA], preds[mA])
        cBA = fit_fn(y[mB], preds[mB])
        pB = cAB(preds[mB])
        pA = cBA(preds[mA])
        p_cal = np.empty(n)
        p_cal[mA] = pA; p_cal[mB] = pB
        skillA = 1.0 - _mae(y[mB], pB) / mae_bl
        skillB = 1.0 - _mae(y[mA], pA) / mae_bl
        skill_cal = 1.0 - _mae(y, p_cal) / mae_bl
        results[name] = {"skill_Afit_Beval": float(skillA),
                         "skill_Bfit_Aeval": float(skillB),
                         "skill_pooled": float(skill_cal),
                         "gain": float(skill_cal - skill_raw)}
        print(f"  {name:14s} skill_pooled={skill_cal:+.4f} gain={skill_cal-skill_raw:+.4f} "
              f"(A->B {skillA:+.4f} B->A {skillB:+.4f})", flush=True)

    # raw skills within each half for reference
    print(f"  raw_skill A={1 - _mae(y[mA], preds[mA]) / mae_bl:+.4f} "
          f"B={1 - _mae(y[mB], preds[mB]) / mae_bl:+.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_calibration.json").write_text(
        json.dumps({"schema": "reactflow_delta.m2r_calibration.v1",
                    "n": n, "raw_skill": float(skill_raw), "results": results},
                   indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out / 'm2r_calibration.json'}")


if __name__ == "__main__":
    sys.exit(main())