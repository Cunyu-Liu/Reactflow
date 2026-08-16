#!/usr/bin/env python3
"""Tests for m2r_features_v1 — feature engineering."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf


@pytest.fixture(scope="module")
def samples():
    path = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
    designs, meta = m2r.parse_m2r_csv(path)
    return m2r.build_all_pair_samples(designs)


def test_feature_dim(samples):
    s = samples[0]
    x = m2rf.build_pair_features(s)
    names = m2rf.feature_names()
    assert len(x) == len(names)
    assert len(x) > 50


def test_no_double_profile_in_features(samples):
    """CRITICAL: the feature vector must not contain double-mutant reactivity."""
    s = samples[0]
    x = m2rf.build_pair_features(s)
    # double_reactivity values must not appear; verify by checking that the
    # feature vector is unchanged if we blank the double profile.
    saved = s.double_reactivity
    s.double_reactivity = [None] * len(s.double_reactivity)
    x2 = m2rf.build_pair_features(s)
    np.testing.assert_array_equal(x, x2)
    s.double_reactivity = saved


def test_feature_values_finite(samples):
    for s in samples[:50]:
        x = m2rf.build_pair_features(s)
        assert np.all(np.isfinite(x))


def test_wc_pair_feature():
    # construct a minimal sample where i,j form a WC pair
    class Fake:
        pass
    s = Fake()
    s.sequence = "A" * 40
    s.editA_seq_pos = 10
    s.editB_seq_pos = 20
    s.wt_reactivity = [0.5] * 40
    s.wt_error = [0.1] * 40
    s.singleA_reactivity = [0.6] * 40
    s.singleA_error = [0.1] * 40
    s.singleB_reactivity = [0.6] * 40
    s.singleB_error = [0.1] * 40
    s.double_reactivity = [0.5] * 40
    s.double_error = [0.1] * 40
    s.target_structure = "." * 40
    # base A at i and base T... use A-U: set j base U
    s.sequence = list(s.sequence)
    s.sequence[20] = "U"
    s.sequence = "".join(s.sequence)
    x = m2rf.build_pair_features(s)
    names = m2rf.feature_names()
    wc_idx = names.index("wc_pair")
    wob_idx = names.index("wobble")
    assert x[wc_idx] == 1.0
    assert x[wob_idx] == 0.0


def test_build_all_consistent(samples):
    X, y, keys, names = m2rf.build_all(samples)
    assert X.shape[0] == len(samples)
    assert y.shape[0] == len(samples)
    assert len(keys) == len(samples)
    assert X.shape[1] == len(names)
    assert np.all(np.isfinite(X))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
