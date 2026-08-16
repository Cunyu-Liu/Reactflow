#!/usr/bin/env python3
"""Tests for m2r_ensemble_v1."""
from __future__ import annotations

import sys, json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_ensemble_v1 as m2r_ens

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
M2_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"


@pytest.fixture(scope="module")
def design_data():
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    m2r.attach_m2_structure(designs, M2_CSV)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    return X, y, np.array(keys), designs


def test_seed_bagging_not_worse(design_data):
    """Seed-bagging (5 seeds) should not be worse than single seed.
    Uses a subset of 10 designs for speed under heavy server load.
    """
    X, y, keys, _ = design_data
    des_list = sorted(set(keys.tolist()))[:30]  # subset for speed
    sel = np.isin(keys, des_list)
    if sel.sum() < 50:
        pytest.skip("too few samples in subset")
    y_med = np.median(y)
    mae_bl = np.mean(np.abs(y - y_med))

    import lightgbm as lgb
    preds_single = np.zeros(len(y))
    preds_mean = np.zeros(len(y))
    n_seeds = 5
    preds_seeds = np.zeros((len(y), n_seeds))

    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds_single[~m] = y_med
            preds_mean[~m] = y_med
            continue
        Xtr, ytr = X[m], y[m]
        Xte = X[~m]
        gs = np.zeros((Xte.shape[0], n_seeds))
        for s in range(n_seeds):
            g = lgb.LGBMRegressor(
                n_estimators=30, max_depth=3,
                random_state=20260816 + s, verbose=-1, n_jobs=2)
            g.fit(Xtr, ytr)
            gs[:, s] = g.predict(Xte)
        preds_seeds[~m] = gs
        preds_single[~m] = gs[:, 0]
        preds_mean[~m] = gs.mean(axis=1)

    skill_single = 1.0 - np.mean(np.abs(y[sel] - preds_single[sel])) / mae_bl
    skill_mean = 1.0 - np.mean(np.abs(y[sel] - preds_mean[sel])) / mae_bl
    # seed-bagging should not be worse (within noise)
    assert skill_mean >= skill_single - 0.005


def test_ridge_blend_in_range(design_data):
    """Ridge has lower skill than GBDT, so blend should be between them.
    Uses a subset of 10 designs for speed under heavy server load.
    """
    X, y, keys, _ = design_data
    des_list = sorted(set(keys.tolist()))[:30]  # subset for speed
    sel = np.isin(keys, des_list)
    if sel.sum() < 50:
        pytest.skip("too few samples in subset")
    y_med = np.median(y)
    mae_bl = np.mean(np.abs(y - y_med))

    from sklearn.linear_model import Ridge
    import lightgbm as lgb

    gb_preds = np.zeros(len(y))
    rg_preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            gb_preds[~m] = y_med
            rg_preds[~m] = y_med
            continue
        Xtr, ytr = X[m], y[m]
        Xte = X[~m]
        g = lgb.LGBMRegressor(
            n_estimators=30, max_depth=3,
            random_state=20260816, verbose=-1, n_jobs=2)
        g.fit(Xtr, ytr)
        gb_preds[~m] = g.predict(Xte)
        r = Ridge(alpha=1.0).fit(Xtr, ytr)
        rg_preds[~m] = r.predict(Xte)

    skill_gb = 1.0 - np.mean(np.abs(y[sel] - gb_preds[sel])) / mae_bl
    skill_rg = 1.0 - np.mean(np.abs(y[sel] - rg_preds[sel])) / mae_bl
    # A convex blend of predictions is never worse than the worse component
    # (triangle inequality): skill_blend >= min(skill_gb, skill_rg) - eps.
    for alpha in [0.5, 0.8]:
        pred = alpha * gb_preds + (1 - alpha) * rg_preds
        skill = 1.0 - np.mean(np.abs(y[sel] - pred[sel])) / mae_bl
        assert skill >= min(skill_gb, skill_rg) - 0.01, \
            f"blend a={alpha}: {skill:.4f} below min({skill_gb:.4f},{skill_rg:.4f})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))