#!/usr/bin/env python3
"""Tests for m2r_3way_strong_puzzle_v1 — puzzle-level strong-GBDT 3-way."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_3way_strong_puzzle_v1 as stpz


def test_helpers():
    y = np.array([1.0, 2.0, 3.0])
    assert abs(stpz._mae(y, np.array([1.0, 2.0, 3.0]))) < 1e-12
    assert abs(stpz._mae(y, np.array([2.0, 3.0, 4.0])) - 1.0) < 1e-12
    assert abs(stpz._skill(stpz._mae(y, np.array([2.0, 2.0, 2.0])),
                           stpz._mae(y, np.full(3, np.median(y))))) < 1e-12
    assert abs(stpz._r2(y, np.array([1.0, 2.0, 3.0])) - 1.0) < 1e-12


def test_loo_pz_shapes():
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm not available")
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 6))
    y = X[:, 0] + rng.normal(0, 0.2, n)
    pz = np.concatenate([np.full(20, "P1"), np.full(20, "P2"),
                         np.full(20, "P3"), np.full(20, "P4")])
    puzzles = sorted(set(pz.tolist()))
    pd_ = stpz._loo_pz(X, y, pz, puzzles, "regression", False)
    ps = stpz._loo_pz(X, y, pz, puzzles, "regression", True)
    assert pd_.shape == (n,)
    assert ps.shape == (n,)
    assert np.all(np.isfinite(pd_)) and np.all(np.isfinite(ps))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
