#!/usr/bin/env python3
"""Tests for m2r_4way_permtest_v1 — 4-way vs strong-3way significance."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_4way_permtest_v1 as fwp


def _make_npz(tmp, n_des=4, per=20, seed=0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(0.5, 0.2, per) for _ in range(n_des)])
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")
    prev = 0.3 * y + rng.normal(0, 0.25, len(y))
    blend = 0.55 * y + rng.normal(0, 0.22, len(y))
    p = tmp / "oof.npz"
    np.savez(p, l1=blend, l2=blend, xg=blend, ridge=blend,
             blend=blend, prev_3way=prev, y=y, keys=keys)
    return p


def test_run_4way_permtest_structure(tmp_path):
    npz = _make_npz(tmp_path)
    out = tmp_path / "out"
    rep = fwp.run_4way_permtest(str(npz), str(out), n_perm=50, n_boot=50)
    assert "models" in rep and "strong_3way" in rep["models"]
    assert "fourway_a_priori" in rep["models"]
    sv = rep["fourway_vs_strong_3way"]
    assert {"pooled_gain_pp", "per_design_mean_pp", "per_design_pct_positive",
            "permutation_p", "loo_exclusion"} <= set(sv)
    assert sv["loo_exclusion"]["n_folds"] >= 3
    # a strictly-better-on-average 4-way blend should be positive
    assert rep["models"]["fourway_a_priori"]["skill"] > rep["models"]["strong_3way"]["skill"]
    assert (out / "m2r_4way_permtest.json").exists()


def test_perm_p_in_range(tmp_path):
    npz = _make_npz(tmp_path, seed=1)
    out = tmp_path / "out2"
    rep = fwp.run_4way_permtest(str(npz), str(out), n_perm=100, n_boot=50)
    p = rep["fourway_vs_strong_3way"]["permutation_p"]
    assert 0.0 <= p <= 1.0
    assert p < 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
