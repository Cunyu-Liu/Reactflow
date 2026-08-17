#!/usr/bin/env python3
"""Tests for m2r_features_v2_ablation_v1 — v2 feature-group ablation."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_features_v2_ablation_v1 as abla


def test_group_ranges_cover_all_v2():
    assert sum(abla.V2_GROUPS.values()) == 22
    # A=7 (3 design overlap + 4 window overlap), B=5 (magnitudes),
    # C=8 (stem context), D=2 (M2 cross context)
    assert abla.V2_GROUPS == {"A": 7, "B": 5, "C": 8, "D": 2}


def test_run_ablation_structure(tmp_path, monkeypatch):
    # build synthetic data
    rng = np.random.default_rng(0)
    n_des = 4
    per = 30
    n1, n2 = 6, 22
    X1 = rng.normal(size=(n_des * per, n1))
    X2 = rng.normal(size=(n_des * per, n2))
    X_tr = rng.normal(size=(n_des * per, 3))
    y = X1[:, 0] * 0.5 + rng.normal(0, 0.3, n_des * per)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")

    # monkeypatch LOO funcs to cheap linear fns
    def fake_loo(X, y, keys, des_list, obj):
        return 0.5 * y + rng.normal(0, 0.05, len(y))

    def fake_ridge(X, y, keys, des_list):
        return 0.4 * y + rng.normal(0, 0.1, len(y))

    monkeypatch.setattr(abla, "_loo_lgb", fake_loo)
    monkeypatch.setattr(abla, "_loo_ridge", fake_ridge)

    class A:
        pass
    A.out = str(tmp_path / "out")
    rep = abla.run_ablation(X1, X2, X_tr, y, keys, A)
    assert "v1_3way" in rep["results"] and "v1_v2_3way" in rep["results"]
    assert set(rep["per_group_attribution"]) == set(abla.V2_GROUPS)
    assert "loo_exclusion_v2_gain" in rep
    assert rep["loo_exclusion_v2_gain"]["n_folds"] >= 3
    assert (tmp_path / "out" / "m2r_features_v2_ablation_report.json").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
