#!/usr/bin/env python3
"""Tests for m2r_ceiling_audit_v1 — capacity/oracle/legal-feature audit."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_ceiling_audit_v1 as audit
import m2r_design_region_features_v1 as drf


def test_fit_predict_default_and_strong():
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm not available")
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(200, 10))
    ytr = Xtr[:, 0] + 0.5 * Xtr[:, 1] + rng.normal(0, 0.1, 200)
    Xte = rng.normal(size=(20, 10))
    pd_ = audit._fit_predict("gbdt_default", Xtr, ytr, Xte)
    ps = audit._fit_predict("gbdt_strong", Xtr, ytr, Xte)
    assert pd_.shape == (20,)
    assert ps.shape == (20,)
    assert np.all(np.isfinite(pd_)) and np.all(np.isfinite(ps))


def test_unknown_model_raises():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        audit._fit_predict("nope", rng.normal(size=(10, 3)),
                           rng.normal(size=10), rng.normal(size=(4, 3)))


def test_loo_factory_shape():
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm not available")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 6))
    y = X[:, 0] + rng.normal(0, 0.2, 120)
    keys = np.concatenate([np.full(40, f"D{i}") for i in range(3)]).astype("U36")
    des_list = sorted(set(keys.tolist()))
    p = audit._loo_factory(X, y, keys, des_list, "gbdt_default")
    assert p.shape == (120,)
    assert np.all(np.isfinite(p))


def test_run_cells_report_structure(tmp_path):
    """run_cells on synthetic data produces the expected report + OOF."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm not available")
    rng = np.random.default_rng(3)
    n = 150
    X_legal = rng.normal(size=(n, 8))
    X_dr = rng.normal(size=(n, 8))
    X_oracle = rng.normal(size=(n, 2))
    y = X_legal[:, 0] * 0.3 + X_dr[:, 0] * 0.2 + rng.normal(0, 0.2, n)
    keys = np.concatenate([np.full(30, f"D{i}") for i in range(5)]).astype("U36")

    class A:
        out = str(tmp_path)
        xgb = False
    rep = audit.run_cells(X_legal, X_dr, X_oracle, y, keys, A)
    assert "oracle_default" in rep["cells"]
    assert "oracle_strong" in rep["cells"]
    assert "legal_dr_strong" in rep["cells"]
    for name in ("legal_default", "legal_dr_default", "legal_strong",
                 "legal_dr_strong", "oracle_default", "oracle_strong",
                 "oracle_dr_strong"):
        assert name in rep["cells"]
        assert {"mae", "skill", "r2", "n_features", "model"} <= set(rep["cells"][name])
    assert (tmp_path / "m2r_ceiling_audit_report.json").exists()
    assert (tmp_path / "m2r_ceiling_audit_oof.npz").exists()
    z = np.load(tmp_path / "m2r_ceiling_audit_oof.npz")
    assert z["y"].shape == (n,)


def test_design_region_features_legal_dim():
    """The legal design-region builder must be 8-dim and oracle 2-dim."""
    assert len(drf.design_region_feature_names()) == 8
    assert len(drf.oracle_feature_names()) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
