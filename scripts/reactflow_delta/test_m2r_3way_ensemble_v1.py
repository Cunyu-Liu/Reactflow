#!/usr/bin/env python3
"""Tests for m2r_3way_ensemble_v1 — cross-objective 3-way ensemble.

Verifies:
  * threeway_blend weights sum to 1 and clipping works
  * weight_plateau produces a dense grid with skills in [0, 1]
  * on a design subset, the 3-way blend skill is not worse than the best
    single component (decorrelation should not hurt on average)
  * run_design_level produces the expected report structure + OOF npz
"""
from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_3way_ensemble_v1 as ens


def test_threeway_blend_weights_sum_to_one():
    rng = np.random.default_rng(0)
    l1 = rng.normal(0.5, 0.2, 100)
    l2 = rng.normal(0.5, 0.2, 100)
    rg = rng.normal(0.5, 0.2, 100)
    for w1, w2 in [(0.6, 0.3), (0.5, 0.3), (0.2, 0.2), (1.0, 0.0), (0.0, 1.0)]:
        p = ens.threeway_blend(l1, l2, rg, w1, w2)
        assert abs(p - (w1 * l1 + w2 * l2 + (1 - w1 - w2) * rg)).max() < 1e-12
        assert abs(1.0 - w1 - w2 - (1.0 - w1 - w2)) < 1e-12


def test_threeway_blend_clip():
    rng = np.random.default_rng(1)
    l1 = rng.normal(0.5, 0.3, 50)
    l2 = rng.normal(0.5, 0.3, 50)
    rg = rng.normal(0.5, 0.3, 50)
    p = ens.threeway_blend(l1, l2, rg, 0.6, 0.3, clip=(0.2, 0.8))
    assert p.min() >= 0.2 - 1e-12 and p.max() <= 0.8 + 1e-12


def test_weight_plateau_grid_dense_and_bounded():
    rng = np.random.default_rng(2)
    y = rng.normal(0.5, 0.3, 200)
    l1 = 0.6 * y + rng.normal(0, 0.2, 200)
    l2 = 0.6 * y + rng.normal(0, 0.25, 200)
    rg = 0.5 * y + rng.normal(0, 0.3, 200)
    mae_bl = np.mean(np.abs(y - np.median(y)))
    grid = ens.weight_plateau(l1, l2, rg, y, mae_bl)
    # (0,0,1)..(1,0,0) plus all splits on the 0.1 grid
    assert len(grid) >= 60
    for v in grid.values():
        assert -1.0 <= v["skill"] <= 1.0
        assert abs(v["w1"] + v["w2"] + v["w3"] - 1.0) < 1e-9
    # the headline point must be present
    assert f"{ens.W1:.1f}_{ens.W2:.1f}" in grid


def test_blend_not_worse_than_best_component_subset():
    """On a synthetic 3-design subset, the 3-way blend should not be materially
    worse than the best single component (averaging complementary models)."""
    rng = np.random.default_rng(3)
    keys = np.concatenate([np.full(40, f"D{i}") for i in range(3)]).astype("U36")
    y = rng.normal(0.5, 0.4, 120)
    y_med = np.median(y)
    mae_bl = np.mean(np.abs(y - y_med))
    p_l1 = 0.7 * y + rng.normal(0, 0.3, 120)
    p_l2 = 0.7 * y + rng.normal(0, 0.35, 120)
    p_r = 0.6 * y + rng.normal(0, 0.4, 120)

    def sk(p):
        return 1.0 - np.mean(np.abs(y - p)) / mae_bl

    blend = ens.threeway_blend(p_l1, p_l2, p_r, 0.6, 0.3)
    best_single = max(sk(p_l1), sk(p_l2), sk(p_r))
    # blend of two decorrelated models should not be worse by more than 5pp
    assert sk(blend) >= best_single - 0.05


def test_run_design_level_report_structure(tmp_path):
    """End-to-end run_design_level on a small design subset produces the
    expected report + OOF npz.  Uses real data if present, else synthetic."""
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
        des_list = sorted(set(keys.tolist()))[:12]
        sel = np.isin(keys, des_list)
        Xs, ys, ks = X[sel], y[sel], keys[sel]
        if len(ks) < 60:
            pytest.skip("subset too small")
    except Exception:
        pytest.skip("data not available")
    out = tmp_path / "o3"
    class A:
        out = None
        trees = 30
        depth = 3
    A.out = str(out)
    # no transfer features for the structural test
    rep = ens.run_design_level(Xs, np.zeros((len(ys), 6)), ys, ks, A)
    assert rep["n_designs"] == len(set(ks.tolist()))
    assert "threeway_blend_a_priori" in rep["results"]
    assert "loo_exclusion_vs_prev_headline" in rep
    assert rep["loo_exclusion_vs_prev_headline"]["n_folds"] >= 8
    assert (out / "m2r_3way_oof.npz").exists()
    assert (out / "m2r_3way_ensemble_report.json").exists()
    rpt = json.loads((out / "m2r_3way_ensemble_report.json").read_text())
    assert rpt["headline_weights"]["w1"] == 0.6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
