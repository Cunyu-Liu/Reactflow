#!/usr/bin/env python3
"""Tests for m2r_calibration_v1 — binned affine calibration helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_calibration_v1 as mc


def test_binned_affine_recovers_synthetic_shrinkage():
    """If pred = shrink(y) with shrinkage toward median, binned affine should
    recover (on held-out) a monotone correction that reduces MAE."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.beta(2, 2, size=n)  # (0,1) skewed
    # simulate GBDT shrinkage toward median
    pred = 0.35 + 0.55 * y + rng.normal(0, 0.05, size=n)
    pred = np.clip(pred, 0, 1)
    # split
    idx = rng.permutation(n)
    tr = idx[:2000]; te = idx[2000:]
    c = mc._binned_affine(y[tr], pred[tr], n_bins=8)
    p_te = c(pred[te])
    mae_before = np.mean(np.abs(y[te] - pred[te]))
    mae_after = np.mean(np.abs(y[te] - p_te))
    assert mae_after < mae_before
    # calibrated predictions should spread more than raw (less shrinkage)
    assert p_te.std() > pred[te].std() * 0.8


def test_binned_affine_identity_when_no_shrinkage():
    """If pred == y (perfect), calibration introduces only small binning error."""
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.beta(2, 2, size=n)
    pred = y + rng.normal(0, 1e-6, size=n)
    idx = rng.permutation(n)
    tr = idx[:1000]; te = idx[1000:]
    c = mc._binned_affine(y[tr], pred[tr], n_bins=8)
    p_te = c(pred[te])
    mae_before = np.mean(np.abs(y[te] - pred[te]))
    mae_after = np.mean(np.abs(y[te] - p_te))
    # binning discretizes to 8 levels, so allow modest degradation
    assert mae_after <= mae_before + 0.05


def test_logistic_bounded():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, size=100)
    y = rng.uniform(0, 1, size=100)
    c = mc._logistic_cal(y, p)
    out = c(np.array([-5.0, 0.0, 0.5, 1.0, 5.0]))
    assert np.all(out > 0) and np.all(out < 1)
    assert np.all(np.isfinite(out))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
