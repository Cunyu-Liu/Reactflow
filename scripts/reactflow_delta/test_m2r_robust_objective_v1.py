#!/usr/bin/env python3
"""Tests for m2r_robust_objective_v1 + m2r_robust_permtest_v1.

Uses a subset of designs for speed (like the ensemble test).  Verifies:
  * robust objectives (l1/huber/fair) run end-to-end via the script's internals
  * on the subset, l1 skill >= l2 skill - eps (MAE objective should not hurt)
  * permtest report structure on a synthetic npz
"""
from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_robust_objective_v1 as rob
import m2r_robust_permtest_v1 as robp

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
M2_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"


@pytest.fixture(scope="module")
def subset_data():
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    m2r.attach_m2_structure(designs, M2_CSV)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))[:30]
    sel = np.isin(keys, des_list)
    return X[sel], y[sel], keys[sel], des_list


def test_robust_objectives_not_worse(subset_data):
    """On a 30-design subset, l1 objective should not be worse than l2 (MAE
    objective directly targets the metric)."""
    X, y, keys, des_list = subset_data
    if len(keys) < 50:
        pytest.skip("too few samples in subset")
    import lightgbm as lgb
    y_med = np.median(y)
    mae_bl = np.mean(np.abs(y - y_med))

    def loo(obj, **kw):
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = lgb.LGBMRegressor(n_estimators=30, max_depth=3,
                                  random_state=20260817, verbose=-1,
                                  objective=obj, **kw)
            g.fit(X[m], y[m])
            preds[~m] = g.predict(X[~m])
        return preds

    p_l2 = loo("regression")
    p_l1 = loo("l1")
    p_fair = loo("fair", c=1.0)
    sk_l2 = 1.0 - np.mean(np.abs(y - p_l2)) / mae_bl
    sk_l1 = 1.0 - np.mean(np.abs(y - p_l1)) / mae_bl
    sk_fair = 1.0 - np.mean(np.abs(y - p_fair)) / mae_bl
    # MAE-family objectives must be close to L2 on the small subset (the +0.56pp
    # gain needs the full 159-design data; verified by full-run LOO-exclusion
    # being 100% positive).  Assert closeness, not strict superiority.
    assert abs(sk_l1 - sk_l2) < 0.02
    assert abs(sk_fair - sk_l2) < 0.02


def test_permtest_report_structure(tmp_path):
    rng = np.random.default_rng(20260817)
    keys = np.concatenate([np.full(40, f"D{i}") for i in range(4)]).astype("U36")
    y = rng.normal(0.5, 1.0, 160)
    pred = 0.6 * y + rng.normal(0, 0.4, 160)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "o.npz"
        np.savez(p, l2=0.9 * pred, l1=pred, l2_blend=pred, best_robust_blend=pred,
                 y=y, keys=keys)
        out = tmp_path / "out"
        rep = robp.run_robust_permtest(str(p), str(out), n_perm=100, n_boot=100)
        assert set(rep["models"]) == {"l2_230", "l1_230",
                                      "l2_fullstack_blend", "l1_fullstack_blend"}
        for v in rep["models"].values():
            assert 0 < v["permutation_p"] <= 1.0
            assert v["ci_low"] <= v["skill"] <= v["ci_high"]
        assert "l1_vs_l2_fullstack_loo" in rep
        assert "l1_vs_l2_per_design" in rep


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
