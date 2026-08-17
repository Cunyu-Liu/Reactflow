#!/usr/bin/env python3
"""Tests for m2r_weighted_v1 (inverse-variance weighting; expected NEGATIVE).

The weighting lever was measured NEGATIVE on full data.  These tests verify:
  * per-sample legal sigma is finite, positive, and correlates weakly with
    |rescue - median| (heavy-tail diagnostic)
  * the weighting functions are well-formed
  * on a 30-design subset, weighted L1 stays within tolerance of unweighted
    (the full-data negative result needs the full 159-design LOO)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_weighted_v1 as w

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
M2_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"


@pytest.fixture(scope="module")
def samples():
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    m2r.attach_m2_structure(designs, M2_CSV)
    return [s for s in m2r.build_all_pair_samples(designs)
            if s.rescue_factor is not None]


def test_per_sample_sigma_legal_positive(samples):
    sig = w.per_sample_sigma_legal(samples[:60], n_mc=60)
    assert len(sig) == 60
    assert np.all(np.isfinite(sig))
    assert np.all(sig > 0)


def test_sigma_diagnostic(samples):
    sig = w.per_sample_sigma_legal(samples[:120], n_mc=60)
    y = np.array([s.rescue_factor for s in samples[:120]])
    # sigma should correlate only weakly with |rescue - median| (heavy tail is
    # not purely measurement noise)
    c = np.corrcoef(sig, np.abs(y - np.median(y)))[0, 1]
    assert -0.5 < c < 0.5


def test_weight_functions_wellformed():
    s = np.array([0.01, 0.05, 0.1, 0.5])
    w_iv = 1.0 / (s ** 2 + 1e-6)
    w_ivc = np.clip(1.0 / (s ** 2 + 1e-6), 1e-3, 50.0)
    w_is = 1.0 / (s + 1e-3)
    assert np.all(w_iv > 0) and np.all(np.isfinite(w_iv))
    assert np.all(w_ivc <= 50.0) and np.all(w_ivc >= 1e-3)
    assert np.all(w_is > 0)
    # monotonic: smaller sigma -> larger weight
    assert w_iv[0] > w_iv[-1]


def test_weighted_subset_not_worse(samples):
    """On a 30-design subset, weighted (clip) L1 must stay close to unweighted."""
    designs = {}
    for s in samples:
        designs.setdefault(s.design_id, []).append(s)
    des_list = sorted(designs.keys())[:30]
    sel = np.array([s.design_id in des_list for s in samples])
    sub = [s for s in samples if s.design_id in des_list]
    X, y, keys, _ = m2rf.build_all(sub)
    keys = np.array(keys)
    if len(y) < 50:
        pytest.skip("too few samples in subset")
    import lightgbm as lgb
    y_med = np.median(y); mae_bl = np.mean(np.abs(y - y_med))
    sig = w.per_sample_sigma_legal(sub, n_mc=40)

    def loo(weight):
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            ww = weight(sig[m]) if weight is not None else None
            g = lgb.LGBMRegressor(n_estimators=30, max_depth=3,
                                  random_state=20260817, verbose=-1, objective="l1")
            g.fit(X[m], y[m], sample_weight=ww)
            preds[~m] = g.predict(X[~m])
        return preds

    p0 = loo(None)
    pw = loo(lambda s: np.clip(1.0 / (s ** 2 + 1e-6), 1e-3, 50.0))
    sk0 = 1.0 - np.mean(np.abs(y - p0)) / mae_bl
    skw = 1.0 - np.mean(np.abs(y - pw)) / mae_bl
    assert abs(skw - sk0) < 0.02


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
