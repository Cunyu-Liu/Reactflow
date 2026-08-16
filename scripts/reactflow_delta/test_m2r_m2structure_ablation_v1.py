#!/usr/bin/env python3
"""Tests for m2r_m2structure_ablation_v1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_m2structure_ablation_v1 as m2r_str


M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
M2_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"


@pytest.fixture(scope="module")
def m2_designs():
    return m2r_str._read_m2_designs(M2_CSV)


@pytest.fixture(scope="module")
def samples():
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    return m2r.build_all_pair_samples(designs)


def test_read_m2_designs_count(m2_designs):
    """All 160 M2 designs loaded."""
    assert len(m2_designs) == 160


def test_read_m2_designs_nonempty_structure(m2_designs):
    """All M2 designs have non-empty M2_structure."""
    empty = [k for k, v in m2_designs.items() if not v["m2_structure"]]
    assert len(empty) == 0, f"empty M2_structure: {empty}"


def test_read_m2_designs_f1(m2_designs):
    """All M2 designs have finite M2_F1."""
    f1s = [v["m2_f1"] for v in m2_designs.values()]
    assert all(f is not None for f in f1s)
    assert all(0 <= f <= 1 for f in f1s)


def test_m2structure_features_dim(samples, m2_designs):
    """Feature matrix has (n_samples, 6) shape."""
    X = m2r_str.build_m2structure_features(samples, m2_designs)
    assert X.shape == (len(samples), 6)
    assert np.all(np.isfinite(X))


def test_m2structure_features_nonzero_overlap(samples, m2_designs):
    """Most pairs have at least one site in the M2_structure region."""
    X = m2r_str.build_m2structure_features(samples, m2_designs)
    frac = (X[:, :4].sum(axis=1) > 0).mean()
    assert frac > 0.5, f"only {frac:.3f} of pairs have M2_structure coverage"


def test_m2structure_features_zero_for_unknown_design(samples, m2_designs):
    """Unknown design keys produce all-zero features."""
    # Find a sample and modify its puzzle/method to a non-existent key
    s = samples[0]
    s.puzzle = "NONEXISTENT"
    s.method = "NONEXISTENT"
    X = m2r_str.build_m2structure_features([s], m2_designs)
    assert X.shape == (1, 6)
    assert X.sum() == 0.0


def test_build_all_consistent(samples, m2_designs):
    """build_all produces consistent arrays."""
    samples_f = [s for s in samples if s.rescue_factor is not None]
    X_existing, y, keys, _ = m2rf.build_all(samples_f)
    X_m2str = m2r_str.build_m2structure_features(samples_f, m2_designs)
    assert len(samples_f) > 100
    assert X_existing.shape[0] == len(samples_f)
    assert X_m2str.shape[0] == len(samples_f)
    assert y.shape[0] == len(samples_f)
    assert len(keys) == len(samples_f)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))