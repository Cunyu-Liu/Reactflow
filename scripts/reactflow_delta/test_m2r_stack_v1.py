#!/usr/bin/env python3
"""Tests for m2r_stack_v1 — stacking / residual-boost / config-soup levers."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_stack_v1 as st


def test_run_stack_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_des, per = 4, 25
    X = rng.normal(size=(n_des * per, 12))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.2, n_des * per)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")

    def fake_loo_cfg(X, y, keys, des_list, obj, cfg, seed):
        return 0.5 * y + rng.normal(0, 0.02, len(y))

    def fake_ridge(X, y, keys, des_list):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    def fake_meta(M, y, keys, des_list, kind):
        return 0.45 * y + rng.normal(0, 0.03, len(y))

    monkeypatch.setattr(st, "_loo_lgb_cfg", fake_loo_cfg)
    monkeypatch.setattr(st, "_loo_ridge", fake_ridge)
    monkeypatch.setattr(st, "_loo_meta", fake_meta)
    orig = st.SEEDS
    st.SEEDS = [1, 2, 3]

    class A:
        pass
    A.out = str(tmp_path / "out")
    A.n_perm = 50
    A.soup = True
    rep = st.run_stack(X, y, keys, A)
    st.SEEDS = orig

    assert "results" in rep and "fixed_3way" in rep["results"]
    assert "stacking" in rep["results"] and "nnls" in rep["results"]["stacking"]
    assert "residual_boost" in rep["results"]
    assert "candidate_gains" in rep and "loo_exclusion" in rep["candidate_gains"]["nnls"]
    assert rep["candidate_gains"]["nnls"]["permutation_p"] <= 1.0
    assert (tmp_path / "out" / "m2r_stack_report.json").exists()


def test_quad_shapes():
    M = np.ones((5, 3))
    Q = st._quad(M)
    # 3 linear + C(4,2)=6 quadratic = 9 cols
    assert Q.shape == (5, 9)


def test_metrics():
    y = np.array([0.1, 0.4, 0.9, 0.2, 0.7])
    p = np.array([0.15, 0.35, 0.85, 0.25, 0.75])
    assert st._skill(st._mae(y, p), st._mae(y, np.full_like(y, np.median(y)))) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
