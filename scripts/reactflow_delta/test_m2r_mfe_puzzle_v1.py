#!/usr/bin/env python3
"""Tests for m2r_mfe_puzzle_v1 — puzzle-level multi-seed 3-way + MFE."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_mfe_puzzle_v1 as mp


def test_run_puzzle_mfe_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n_pz, per = 4, 25
    X = rng.normal(size=(n_pz * per, 12))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.2, n_pz * per)
    pz = np.concatenate([np.full(per, f"P{i}") for i in range(n_pz)]).astype("U16")

    orig = mp.SEEDS
    mp.SEEDS = [1, 2, 3]

    def fake_loo(X, y, pz, puzzles, obj, seed):
        return 0.5 * y + rng.normal(0, 0.02, len(y))

    def fake_ridge(X, y, pz, puzzles):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    monkeypatch.setattr(mp, "_loo_lgb_seed", fake_loo)
    monkeypatch.setattr(mp, "_loo_ridge", fake_ridge)

    base_blend = 0.45 * y + rng.normal(0, 0.04, len(y))
    np.savez(tmp_path / "base.npz", blend_K=base_blend, y=y, puzzles=pz)

    class A:
        pass
    A.out = str(tmp_path / "out")
    A.base_npz = str(tmp_path / "base.npz")
    A.n_perm = 50
    rep = mp.run_puzzle_mfe(X, y, pz, A)
    mp.SEEDS = orig

    assert "results" in rep and "mfe_multiseed_3way" in rep["results"]
    assert "mfe_gain_vs_nonmfe" in rep and "per_puzzle_pct_positive" in rep["mfe_gain_vs_nonmfe"]
    assert 0.0 <= rep["permutation_p"] <= 1.0
    assert rep["k_seeds"] == 3
    assert (tmp_path / "out" / "m2r_mfe_puzzle_report.json").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
