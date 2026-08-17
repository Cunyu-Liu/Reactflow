#!/usr/bin/env python3
"""Tests for m2r_4way_puzzle_v1 — puzzle-level 4-way ensemble helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_4way_puzzle_v1 as fwz


def test_helpers():
    y = np.array([1.0, 2.0, 3.0])
    assert abs(fwz._mae(y, np.array([1.0, 2.0, 3.0]))) < 1e-12
    assert abs(fwz._mae(y, np.array([2.0, 3.0, 4.0])) - 1.0) < 1e-12
    assert abs(fwz._skill(fwz._mae(y, np.array([2.0, 2.0, 2.0])),
                          fwz._mae(y, np.full(3, np.median(y))))) < 1e-12
    assert abs(fwz._r2(y, np.array([1.0, 2.0, 3.0])) - 1.0) < 1e-12


def test_blend_weights():
    rng = np.random.default_rng(0)
    y = rng.normal(0.5, 0.3, 50)
    l1 = 0.6 * y + rng.normal(0, 0.2, 50)
    l2 = 0.6 * y + rng.normal(0, 0.2, 50)
    xg = 0.6 * y + rng.normal(0, 0.2, 50)
    rg = 0.5 * y + rng.normal(0, 0.3, 50)
    b4 = fwz.W1 * l1 + fwz.W2 * l2 + fwz.WX * xg + fwz.WR * rg
    assert abs(1.0 - fwz.W1 - fwz.W2 - fwz.WX - fwz.WR) < 1e-12
    assert b4.shape == (50,)
    assert np.all(np.isfinite(b4))


def test_loo_xgb_shapes():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pytest.skip("xgboost not available")
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 6))
    y = X[:, 0] + rng.normal(0, 0.2, n)
    pz = np.concatenate([np.full(20, "P1"), np.full(20, "P2"),
                         np.full(20, "P3"), np.full(20, "P4")])
    puzzles = sorted(set(pz.tolist()))
    px = fwz._loo_xgb(X, y, pz, puzzles)
    assert px.shape == (n,)
    assert np.all(np.isfinite(px))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
