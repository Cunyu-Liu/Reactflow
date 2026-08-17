#!/usr/bin/env python3
"""Tests for m2r_features_v2_permtest_v1 — v2 feature-group significance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_features_v2_permtest_v1 as v2p


def _make_npz(tmp, n_des=4, per=20, seed=0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(0.5, 0.2, per) for _ in range(n_des)])
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")
    blend1 = 0.3 * y + rng.normal(0, 0.25, len(y))
    blend2 = 0.55 * y + rng.normal(0, 0.22, len(y))
    p = tmp / "oof.npz"
    np.savez(p, blend1=blend1, blend2=blend2, y=y, keys=keys)
    return p


def test_run_v2_permtest_structure(tmp_path):
    npz = _make_npz(tmp_path)
    out = tmp_path / "out"
    rep = v2p.run_v2_permtest(str(npz), str(out), n_perm=50, n_boot=50)
    assert "models" in rep and "v1_3way" in rep["models"]
    assert "v1_v2_3way" in rep["models"]
    sv = rep["v2_vs_v1"]
    assert {"pooled_gain_pp", "per_design_mean_pp", "per_design_pct_positive",
            "permutation_p", "loo_exclusion"} <= set(sv)
    assert sv["loo_exclusion"]["n_folds"] >= 3
    assert rep["models"]["v1_v2_3way"]["skill"] > rep["models"]["v1_3way"]["skill"]
    assert (out / "m2r_features_v2_permtest.json").exists()


def test_perm_p_in_range(tmp_path):
    npz = _make_npz(tmp_path, seed=1)
    out = tmp_path / "out2"
    rep = v2p.run_v2_permtest(str(npz), str(out), n_perm=100, n_boot=50)
    p = rep["v2_vs_v1"]["permutation_p"]
    assert 0.0 <= p <= 1.0
    assert p < 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
