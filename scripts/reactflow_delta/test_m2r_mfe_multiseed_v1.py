#!/usr/bin/env python3
"""Tests for m2r_mfe_multiseed_v1 — multi-seed 3-way + MFE features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_mfe_multiseed_v1 as ms


def test_run_mfe_multiseed_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_des, per = 4, 25
    X = rng.normal(size=(n_des * per, 12))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.2, n_des * per)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")

    orig = ms.SEEDS
    ms.SEEDS = [1, 2, 3]

    def fake_loo(X, y, keys, des_list, obj, seed):
        return 0.5 * y + rng.normal(0, 0.02, len(y))

    def fake_ridge(X, y, keys, des_list):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    monkeypatch.setattr(ms, "_loo_lgb_seed", fake_loo)
    monkeypatch.setattr(ms, "_loo_ridge", fake_ridge)

    # baseline npz (non-MFE multi-seed)
    base_blend = 0.45 * y + rng.normal(0, 0.04, len(y))
    np.savez(tmp_path / "base.npz", blend_K=base_blend, y=y, keys=keys)

    class A:
        pass
    A.out = str(tmp_path / "out")
    A.base_npz = str(tmp_path / "base.npz")
    A.n_perm = 50
    rep = ms.run_mfe_multiseed(X, y, keys, A)
    ms.SEEDS = orig

    assert "results" in rep and "mfe_multiseed_3way" in rep["results"]
    assert "mfe_gain_vs_nonmfe" in rep and "loo_exclusion" in rep["mfe_gain_vs_nonmfe"]
    assert rep["mfe_gain_vs_nonmfe"]["permutation_p"] <= 1.0
    assert rep["k_seeds"] == 3
    assert (tmp_path / "out" / "m2r_mfe_multiseed_report.json").exists()


def test_seeds():
    assert len(ms.SEEDS) == 5
    assert len(set(ms.SEEDS)) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
