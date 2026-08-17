#!/usr/bin/env python3
"""Tests for m2r_multiseed_permtest_v1 — multi-seed averaging significance."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_multiseed_permtest_v1 as mp


def test_run_multiseed_permtest_structure(tmp_path):
    rng = np.random.default_rng(3)
    n_des, per = 4, 25
    y = rng.uniform(0, 1, n_des * per)
    b1 = y + rng.normal(0, 0.3, n_des * per)
    bK = b1 - rng.normal(0, 0.05, n_des * per)  # slightly better + lower var
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")

    npz = tmp_path / "oof.npz"
    np.savez(npz, blend_1=b1, blend_K=bK, y=y, keys=keys)

    rep = mp.run_multiseed_permtest(str(npz), str(tmp_path / "out"),
                                    n_perm=100, n_boot=100)
    assert "models" in rep and "multiseed_3way" in rep["models"]
    assert "multiseed_vs_single" in rep and "loo_exclusion" in rep["multiseed_vs_single"]
    assert rep["multiseed_vs_single"]["loo_exclusion"]["n_folds"] >= 3
    assert 0.0 <= rep["multiseed_vs_single"]["permutation_p"] <= 1.0
    assert (tmp_path / "out" / "m2r_multiseed_permtest.json").exists()


def test_metrics_consistent():
    y = np.array([0.1, 0.4, 0.9, 0.2, 0.7, 0.3])
    p = np.array([0.15, 0.35, 0.85, 0.25, 0.75, 0.35])
    mae = mp._mae(y, p)
    bl = mp._mae(y, np.full_like(y, np.median(y)))
    assert mp._skill(mae, bl) == pytest.approx(1.0 - mae / bl)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
