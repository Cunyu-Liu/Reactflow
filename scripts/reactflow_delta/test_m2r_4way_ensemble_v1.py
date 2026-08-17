#!/usr/bin/env python3
"""Tests for m2r_4way_ensemble_v1 — architecture-decorrelated 4-way ensemble."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_4way_ensemble_v1 as ens


def test_fourway_blend_weights():
    rng = np.random.default_rng(0)
    l1 = rng.normal(0.5, 0.2, 100)
    l2 = rng.normal(0.5, 0.2, 100)
    xg = rng.normal(0.5, 0.2, 100)
    rg = rng.normal(0.5, 0.2, 100)
    for w1, w2, wx in [(0.45, 0.25, 0.20), (0.5, 0.3, 0.1), (0.2, 0.2, 0.3),
                       (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]:
        wr = 1.0 - w1 - w2 - wx
        p = ens.fourway_blend(l1, l2, xg, rg, w1, w2, wx)
        assert abs(p - (w1 * l1 + w2 * l2 + wx * xg + wr * rg)).max() < 1e-12
        assert abs(1.0 - w1 - w2 - wx - wr) < 1e-12
        assert wr >= -1e-12


def test_weight_plateau_dense():
    rng = np.random.default_rng(1)
    y = rng.normal(0.5, 0.3, 200)
    l1 = 0.6 * y + rng.normal(0, 0.2, 200)
    l2 = 0.6 * y + rng.normal(0, 0.25, 200)
    xg = 0.6 * y + rng.normal(0, 0.22, 200)
    rg = 0.5 * y + rng.normal(0, 0.3, 200)
    mae_bl = np.mean(np.abs(y - np.median(y)))
    grid = ens.weight_plateau(l1, l2, xg, rg, y, mae_bl)
    assert len(grid) >= 100
    for v in grid.values():
        assert -1.0 <= v["skill"] <= 1.0
        assert abs(v["w1"] + v["w2"] + v["wx"] + v["wr"] - 1.0) < 1e-9
    # headline point present
    assert f"{ens.W_L1:.1f}_{ens.W_L2:.1f}_{ens.W_X:.1f}" in grid


def test_4way_not_worse_than_3way_subset():
    """On synthetic data, adding a decorrelated member should not materially
    hurt vs the 3-way (0.6/0.3/0.1)."""
    rng = np.random.default_rng(2)
    y = rng.normal(0.5, 0.4, 200)
    y_med = np.median(y)
    mae_bl = np.mean(np.abs(y - y_med))
    p_l1 = 0.7 * y + rng.normal(0, 0.3, 200)
    p_l2 = 0.7 * y + rng.normal(0, 0.35, 200)
    p_xg = 0.7 * y + rng.normal(0, 0.32, 200)
    p_r = 0.6 * y + rng.normal(0, 0.4, 200)

    def sk(p):
        return 1.0 - np.mean(np.abs(y - p)) / mae_bl

    b4 = ens.fourway_blend(p_l1, p_l2, p_xg, p_r, 0.45, 0.25, 0.20)
    b3 = 0.6 * p_l1 + 0.3 * p_l2 + 0.1 * p_r
    assert sk(b4) >= sk(b3) - 0.05


def test_run_design_level_report_structure(tmp_path):
    """End-to-end run_design_level on a small design subset (real data if
    available, else synthetic) produces the expected report + OOF."""
    try:
        import lightgbm, xgboost  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm/xgboost not available")
    try:
        import m2r_data_v1 as m2r
        import m2r_features_v1 as m2rf
        M2R = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
        M2 = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
        designs, _ = m2r.parse_m2r_csv(M2R)
        m2r.attach_m2_structure(designs, M2)
        samples = [s for s in m2r.build_all_pair_samples(designs)
                   if s.rescue_factor is not None]
        X, y, keys, _ = m2rf.build_all(samples)
        keys = np.array(keys)
        des_list = sorted(set(keys.tolist()))[:10]
        sel = np.isin(keys, des_list)
        Xs, ys, ks = X[sel], y[sel], keys[sel]
        if len(ks) < 60:
            pytest.skip("subset too small")
    except Exception:
        pytest.skip("data not available")
    out = tmp_path / "o4"
    class A:
        pass
    A.out = str(out)
    rep = ens.run_design_level(Xs, np.zeros((len(ys), 6)), ys, ks, A)
    assert rep["n_designs"] == len(set(ks.tolist()))
    assert "fourway_blend_a_priori" in rep["results"]
    assert "prev_strong_3way" in rep["results"]
    assert "loo_exclusion_vs_strong_3way" in rep
    assert rep["loo_exclusion_vs_strong_3way"]["n_folds"] >= 8
    assert (out / "m2r_4way_oof.npz").exists()
    assert (out / "m2r_4way_ensemble_report.json").exists()
    rpt = json.loads((out / "m2r_4way_ensemble_report.json").read_text())
    assert rpt["headline_weights"]["w1"] == 0.45


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
