#!/usr/bin/env python3
"""Tests for m2r_transfer_v1."""
from __future__ import annotations

import sys, json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_transfer_v1 as tr

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
M2_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
M2_PRED = "/mnt/cunyuliu/m2_response_spectrum_attn_v5_deep_20260815/keyed_predictions_m2_attn.jsonl"


@pytest.fixture(scope="module")
def samples():
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    m2r.attach_m2_structure(designs, M2_CSV)
    return [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]


@pytest.fixture(scope="module")
def m2_oof():
    return tr.load_m2_oof(M2_PRED)


def test_load_m2_oof_structure(m2_oof):
    assert len(m2_oof) > 100
    # each design -> {mutA: 21-array}
    did = next(iter(m2_oof))
    ma = next(iter(m2_oof[did]))
    arr = m2_oof[did][ma]
    assert len(arr) == 21
    assert np.all(np.isfinite(arr))


def test_design_key_mapping(m2_oof, samples):
    # build mapping like main()
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    # most samples should map
    matched = sum(1 for s in samples if (s.puzzle, s.method) in m2_design_key)
    assert matched / len(samples) > 0.9, f"matched {matched}/{len(samples)}"


def test_build_transfer_features_shape(m2_oof, samples):
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    assert X_tr.shape == (len(samples), 6)
    assert np.all(np.isfinite(X_tr))
    nz = (np.abs(X_tr).sum(axis=1) > 0).mean()
    assert nz > 0.8, f"nonzero frac {nz:.3f}"


def test_transfer_features_center_consistent(m2_oof, samples):
    """The center feature should equal the M2 prediction at the edit site."""
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    checked = 0
    for idx, s in enumerate(samples):
        m2id = m2_design_key.get((s.puzzle, s.method))
        if m2id is None:
            continue
        mm = m2_oof.get(m2id)
        if mm is None or s.mutA not in mm:
            continue
        assert abs(X_tr[idx, 0] - mm[s.mutA][tr.CENTER]) < 1e-9
        checked += 1
    assert checked > 100


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))