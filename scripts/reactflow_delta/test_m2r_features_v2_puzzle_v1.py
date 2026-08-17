#!/usr/bin/env python3
"""Tests for m2r_features_v2_puzzle_v1 — puzzle-level v1+v2 strong 3-way."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_features_v2_puzzle_v1 as v2z


def test_helpers():
    y = np.array([1.0, 2.0, 3.0])
    assert abs(v2z._mae(y, np.array([1.0, 2.0, 3.0]))) < 1e-12
    assert abs(v2z._r2(y, np.array([1.0, 2.0, 3.0])) - 1.0) < 1e-12
    assert abs(1.0 - v2z.W1 - v2z.W2 - v2z.W3) < 1e-12


def test_loo_ridge_shapes():
    from sklearn.linear_model import Ridge  # noqa: F401
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 6))
    y = X[:, 0] + rng.normal(0, 0.2, n)
    pz = np.concatenate([np.full(20, "P1"), np.full(20, "P2"),
                         np.full(20, "P3"), np.full(20, "P4")])
    puzzles = sorted(set(pz.tolist()))
    pr = v2z._loo_ridge(X, y, pz, puzzles)
    assert pr.shape == (n,)
    assert np.all(np.isfinite(pr))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
