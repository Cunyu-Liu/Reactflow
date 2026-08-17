#!/usr/bin/env python3
"""Tests for m2r_3way_strong_permtest_v1 — strong-GBDT 3-way significance."""
from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_3way_strong_permtest_v1 as stp


def _make_npz(tmp, n_des=4, per=20, seed=0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(0.5, 0.2, per) for _ in range(n_des)])
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")
    blend_d = 0.3 * y + rng.normal(0, 0.2, len(y))
    blend_s = 0.5 * y + rng.normal(0, 0.2, len(y))
    p = tmp / "oof.npz"
    np.savez(p, blend_d=blend_d, blend_s=blend_s, y=y, keys=keys)
    return p


def test_run_strong_permtest_structure(tmp_path):
    npz = _make_npz(tmp_path)
    out = tmp_path / "out"
    rep = stp.run_strong_permtest(str(npz), str(out), n_perm=50, n_boot=50)
    assert "models" in rep and "default_3way" in rep["models"]
    assert "strong_3way" in rep["models"]
    sv = rep["strong_vs_default"]
    assert {"pooled_gain_pp", "per_design_mean_pp", "per_design_pct_positive",
            "permutation_p", "loo_exclusion"} <= set(sv)
    assert sv["loo_exclusion"]["n_folds"] >= 3
    # a strictly-better-on-average strong model should be positive
    assert rep["models"]["strong_3way"]["skill"] > rep["models"]["default_3way"]["skill"]
    assert (out / "m2r_3way_strong_permtest.json").exists()


def test_perm_p_in_range(tmp_path):
    npz = _make_npz(tmp_path, seed=1)
    out = tmp_path / "out2"
    rep = stp.run_strong_permtest(str(npz), str(out), n_perm=100, n_boot=50)
    p = rep["strong_vs_default"]["permutation_p"]
    assert 0.0 <= p <= 1.0
    # strong clearly better -> small p
    assert p < 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
