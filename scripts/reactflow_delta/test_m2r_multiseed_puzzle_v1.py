#!/usr/bin/env python3
"""Tests for m2r_multiseed_puzzle_v1 — puzzle-level multi-seed averaging."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_multiseed_puzzle_v1 as msp


def test_run_puzzle_multiseed_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_pz, per = 4, 25
    X = rng.normal(size=(n_pz * per, 12))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.2, n_pz * per)
    pz = np.concatenate([np.full(per, f"P{i}") for i in range(n_pz)]).astype("U16")

    orig = msp.SEEDS
    msp.SEEDS = [1, 2, 3]

    def fake_loo(X, y, pz, puzzles, obj, seed):
        return 0.5 * y + rng.normal(0, 0.02, len(y))

    def fake_ridge(X, y, pz, puzzles):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    monkeypatch.setattr(msp, "_loo_lgb_seed", fake_loo)
    monkeypatch.setattr(msp, "_loo_ridge", fake_ridge)

    class A:
        pass
    A.out = str(tmp_path / "out")
    A.n_perm = 50
    rep = msp.run_puzzle_multiseed(X, y, pz, A)
    msp.SEEDS = orig

    assert "results" in rep and "single_seed_3way" in rep["results"]
    assert "multiseed_3way" in rep["results"]
    assert "multiseed_gain" in rep and "per_puzzle_pct_positive" in rep["multiseed_gain"]
    assert "permutation_p" in rep and 0.0 <= rep["permutation_p"] <= 1.0
    assert rep["k_seeds"] == 3
    assert (tmp_path / "out" / "m2r_multiseed_puzzle_report.json").exists()


def test_seed_list():
    assert len(msp.SEEDS) == 5
    assert len(set(msp.SEEDS)) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
