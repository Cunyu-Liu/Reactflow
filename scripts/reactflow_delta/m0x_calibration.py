#!/usr/bin/env python3
"""EPRO_DEV_05 / M0-X: proper probability calibration for changer detection.

Diagnosis of the DEV_04 / DEV_03 degradation:
  * The old "multi-layer" cascade (Platt -> temperature -> isotonic PAVA) was
    mathematically redundant: temperature scaling is itself a logistic
    transform applied to the logit of an already-logistic (Platt) fit, so it
    re-fits the identical family on the same data.
  * The temperature solver used a fixed lr=1.0 gradient step on a mostly-flat
    loss, which collapsed T to the 1e-3 floor (numerical degeneration), and the
    isotonic step then extrapolated on that corrupted output -> worst of all.

Fix (this module): choose ONE calibration family, fit it on a held-out
calibration split, and report Brier / log-loss / expected calibration error
(ECE) with a reliability histogram on a genuinely held-out evaluation split.
No stacking of redundant logistic transforms.  isotonic PAVA is the robust
default for binary probability calibration; temperature scaling is offered as
a proper alternative fit via a bounded scalar optimizer (not the old lr=1.0
gradient descent).

All fitted functions are order-preserving monotone maps, so the study-macro
AUPRC (the primary, calibration-invariant dev metric) is unaffected by which
calibration is chosen.
"""

from __future__ import annotations

import numpy as np


def brier_logloss(y, p):
    """Brier score and binary log-loss for probabilities p vs labels y."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    brier = float(np.mean((y - p) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return brier, ll


def ece(y, p, n_bins: int = 10):
    """Expected calibration error over equal-width bins of p in [0,1]."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        cnt[b] = m.sum()
        conf[b] = p[m].mean()
        acc[b] = y[m].mean()
    tot = cnt.sum()
    if tot == 0:
        return float("nan")
    return float(np.sum(cnt / tot * np.abs(conf - acc)))


def reliability_histogram(y, p, n_bins: int = 10):
    """Return per-bin (bin_center, mean_conf, mean_acc, count) for plotting."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append([float((bins[b] + bins[b + 1]) / 2), float("nan"),
                         float("nan"), 0])
            continue
        rows.append([float((bins[b] + bins[b + 1]) / 2),
                     float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return rows


def _isotonic_pava_fit(y, x):
    """Isotonic regression (PAVA) of y on x: returns (xs_sorted, ys_fit_sorted).

    Robust block-average PAVA (unit weights) that never divides by zero.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order].copy()
    n = len(ys)
    wts = np.ones(n)
    # Pool adjacent violators.
    i = 0
    while i < n - 1:
        if wts[i] > 0 and wts[i + 1] > 0 and ys[i] > ys[i + 1]:
            w = wts[i] + wts[i + 1]
            merged = (wts[i] * ys[i] + wts[i + 1] * ys[i + 1]) / w
            # Merge into block i, mark i+1 dead, then step back to re-check.
            ys[i] = merged
            wts[i] = w
            wts[i + 1] = 0.0
            i = max(i - 1, 0)
        else:
            i += 1
    keep = wts > 0
    return xs[keep], ys[keep]


def _iso_apply(xs, ys, x):
    x = np.asarray(x, dtype=np.float64)
    if len(xs) < 2:
        return np.full_like(x, ys[0] if len(ys) else 0.0)
    # Monotone non-decreasing interpolation (np.interp on sorted xs is monotone).
    return np.interp(x, xs, ys)


def isotonic_calibrate(train_score, train_y):
    """Fit isotonic PAVA on train; return a callable score->probability."""
    xs, ys = _isotonic_pava_fit(train_y, train_score)
    return lambda s: _iso_apply(xs, ys, s)


def temperature_calibrate(train_logits, train_y, bounds=(0.01, 10.0)):
    """Fit temperature scaling T on logits via bounded scalar minimization.

    Uses a coarse grid + fine golden-section refinement so it cannot collapse
    to the floor the way the old lr=1.0 gradient descent did.
    """
    logits = np.asarray(train_logits, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)

    def nll(T):
        T = max(float(T), 1e-3)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits / T, -30, 30)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    lo, hi = bounds
    best_T, best_nll = 1.0, nll(1.0)
    for T in np.linspace(lo, hi, 201):
        v = nll(T)
        if v < best_nll:
            best_nll, best_T = v, T
    # Golden-section refinement around the grid optimum.
    lo, hi = max(lo, best_T * 0.5), min(hi, best_T * 2.0)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
    best_T = (lo + hi) / 2.0
    return best_T, nll(best_T)


def platt_calibrate(train_score, train_y, iters=3000, lr=0.1):
    """Platt (logistic) calibration on standardized score; returns callable."""
    score = np.asarray(train_score, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mu, sd = float(score.mean()), float(score.std()) + 1e-6
    x = (score - mu) / sd
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    n = X.shape[0]
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (p - y)) / n

    def apply(s):
        s = np.asarray(s, dtype=np.float64)
        x = (s - mu) / sd
        return 1.0 / (1.0 + np.exp(-np.clip(w[0] + w[1] * x, -30, 30)))

    return apply


def fit_and_report(train_score, train_y, val_score, val_y, n_bins: int = 10):
    """Fit each single calibration method on train; report Brier/Logloss/ECE on val.

    Returns (report, calibrated_val_probs, method).  method is the one with the
    lowest validation Brier score (a principled single-method selection).
    """
    tr = np.asarray(train_score, dtype=np.float64)
    ty = np.asarray(train_y, dtype=np.float64)
    vs = np.asarray(val_score, dtype=np.float64)
    vy = np.asarray(val_y, dtype=np.float64)

    cals = {}
    cals["raw"] = (lambda p: p, vs.copy())
    cals["platt"] = (platt_calibrate(tr, ty), None)
    cals["isotonic_pava"] = (isotonic_calibrate(tr, ty), None)

    # Temperature operates on raw logits (not on a Platt output).
    logits_tr = np.log(np.clip(tr, 1e-9, 1 - 1e-9) / (1 - np.clip(tr, 1e-9, 1 - 1e-9)))
    T, _ = temperature_calibrate(logits_tr, ty)
    cals["temperature_scaled"] = (
        lambda p, _T=T: 1.0 / (1.0 + np.exp(-np.clip(
            np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))) / _T,
            -30, 30))),
        None)

    report = {}
    best_brier = float("inf")
    best_method = None
    calibrated = None
    for name, (fn, _fitted) in cals.items():
        pv = np.clip(np.asarray(fn(vs), dtype=np.float64), 1e-9, 1 - 1e-9)
        b, ll = brier_logloss(vy, pv)
        ec = ece(vy, pv, n_bins=n_bins)
        hist = reliability_histogram(vy, pv, n_bins=n_bins)
        report[name] = {"brier": b, "log_loss": ll, "ece": ec,
                        "reliability": hist}
        if name == "temperature_scaled":
            report[name]["temperature"] = T
        if b < best_brier:
            best_brier = b
            best_method = name
            calibrated = pv
    report["selected_method"] = best_method
    return report, calibrated, best_method


if __name__ == "__main__":
    # Smoke test on synthetic data.
    rng = np.random.default_rng(0)
    s_tr = rng.normal(size=2000)
    p_tr = 1.0 / (1.0 + np.exp(-s_tr))
    y_tr = rng.binomial(1, p_tr).astype(float)
    s_va = rng.normal(size=2000)
    p_va = 1.0 / (1.0 + np.exp(-s_va))
    y_va = rng.binomial(1, p_va).astype(float)
    rep, cal, sel = fit_and_report(s_tr, y_tr, s_va, y_va)
    for k, v in rep.items():
        if isinstance(v, dict):
            print(k, {kk: round(vv, 4) for kk, vv in v.items()
                      if kk != "reliability"})
    print("selected:", sel)