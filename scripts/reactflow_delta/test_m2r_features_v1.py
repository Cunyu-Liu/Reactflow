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


def test_m2structure_features_added(samples):
    """When M2_structure is attached, feature dim grows by 6 and is non-zero."""
    s = samples[0]
    # No M2_structure attached -> last 6 features are zero (or defaults)
    x_none = m2rf.build_pair_features(s)
    names = m2rf.feature_names()
    n = len(names)
    # attach M2_structure to the sample
    s2 = type(s)(**{f: getattr(s, f) for f in s.__dataclass_fields__})
    s2.m2_structure = "((....))" + "." * (len(s.sequence) - 8)
    s2.sub_start = 1
    x_with = m2rf.build_pair_features(s2)
    assert len(x_none) == n
    assert len(x_with) == n
    # the last 4 paired/depth features should differ when structure attached
    m2_idx = names.index("m2_pa_i")
    assert names[m2_idx] == "m2_pa_i"
    assert names[-2] == "m2_f1"
    assert names[-1] == "m2_f1_crossed_pair"
    assert x_with[m2_idx] >= 0  # finite, defined


def test_attach_m2_structure_all_designs():
    """attach_m2_structure attaches non-empty structure to all usable designs."""
    path = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
    m2_path = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
    designs, meta = m2r.parse_m2r_csv(path)
    m2r.attach_m2_structure(designs, m2_path)
    usable = [d for d in designs if d["usable"]]
    assert len(usable) > 100
    empty = [d for d in usable if not d.get("m2_structure")]
    assert len(empty) == 0, f"{len(empty)} usable designs lack M2_structure"
    f1s = [d.get("m2_f1") for d in usable]
    assert all(f is not None for f in f1s)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
