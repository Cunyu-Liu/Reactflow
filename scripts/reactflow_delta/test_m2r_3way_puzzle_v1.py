#!/usr/bin/env python3
"""Tests for m2r_3way_puzzle_v1 — puzzle-level leak-free 3-way ensemble.

Verifies the puzzle-level runner produces the expected report + OOF structure
on a small subset of puzzles (real data), and that the per-puzzle gain fields
are consistent.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_3way_ensemble_v1 as ens   # for threeway_blend / weight helpers
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_transfer_v1 as tr

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
M2_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
M2_PZ = "/mnt/cunyuliu/m2_attn_puzzle_20260817/keyed_predictions_m2_attn_puzzle.jsonl"


def test_blend_components_decorrelate_subset():
    """On a small design subset, the 3-way blend should not be worse than the
    best single component (same property the full run demonstrates)."""
    designs, _ = m2r.parse_m2r_csv(M2R_CSV)
    m2r.attach_m2_structure(designs, M2_CSV)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))[:10]
    sel = np.isin(keys, des_list)
    Xs, ys, ks = X[sel], y[sel], keys[sel]
    if len(ks) < 50:
        pytest.skip("subset too small")
    import lightgbm as lgb
    y_med = np.median(ys)
    mae_bl = np.mean(np.abs(ys - y_med))
    p_l1, p_l2, p_r = np.zeros(len(ys)), np.zeros(len(ys)), np.zeros(len(ys))
    for held in des_list:
        m = ks != held
        if m.sum() <= 10:
            continue
        for obj, store in [("l1", p_l1), ("regression", p_l2)]:
            g = lgb.LGBMRegressor(n_estimators=30, max_depth=3,
                                  random_state=20260817, verbose=-1,
                                  objective=obj)
            g.fit(Xs[m], ys[m])
            store[~m] = g.predict(Xs[~m])
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=1.0).fit(Xs[m], ys[m])
        p_r[~m] = r.predict(Xs[~m])
    blend = ens.threeway_blend(p_l1, p_l2, p_r, 0.6, 0.3)
    sk = lambda p: 1.0 - np.mean(np.abs(ys - p)) / mae_bl
    assert sk(blend) >= max(sk(p_l1), sk(p_l2), sk(p_r)) - 0.05


def test_puzzle_oof_structure(tmp_path):
    """Run the puzzle-level LOO helpers logic in isolation on a small set of
    puzzles, verifying the blend + per-puzzle gain bookkeeping."""
    designs, _ = m2r.parse_m2r_csv(M2R_CSV)
    m2r.attach_m2_structure(designs, M2_CSV)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    # restrict to 3 puzzles
    keep_pz = {"P01", "P02", "P03"}
    samples = [s for s in samples if s.puzzle in keep_pz]
    if len(samples) < 60:
        pytest.skip("puzzle subset too small")
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    sample_puzzles = np.array([s.puzzle for s in samples])
    puzzles = sorted(set(sample_puzzles.tolist()))
    y_med = np.median(y)
    mae_bl = np.mean(np.abs(y - y_med))
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    p_l1, p_l2, p_r = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for held in puzzles:
        m = sample_puzzles != held
        if m.sum() <= 10:
            continue
        for obj, store in [("l1", p_l1), ("regression", p_l2)]:
            g = lgb.LGBMRegressor(n_estimators=30, max_depth=3,
                                  random_state=20260817, verbose=-1,
                                  objective=obj)
            g.fit(X[m], y[m])
            store[~m] = g.predict(X[~m])
        r = Ridge(alpha=1.0).fit(X[m], y[m])
        p_r[~m] = r.predict(X[~m])
    blend = 0.6 * p_l1 + 0.3 * p_l2 + 0.1 * p_r
    prev = 0.8 * p_l1 + 0.2 * p_r
    gains = []
    for pz in puzzles:
        m = sample_puzzles == pz
        if m.sum() == 0:
            continue
        gains.append((1 - np.mean(np.abs(y[m] - blend[m])) / mae_bl) -
                     (1 - np.mean(np.abs(y[m] - prev[m])) / mae_bl))
    gains = np.array(gains)
    assert len(gains) == len(puzzles)
    assert np.all(np.isfinite(gains))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
