#!/usr/bin/env python3
"""Tests for m2r_shape_eval_v1 — SHAPE-guided feature screening harness."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_shape_eval_v1 as se


def test_run_shape_screen_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_des, per = 4, 25
    y = rng.normal(0.4, 0.25, n_des * per)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")

    def fake_loo(X, y, keys, des_list, obj):
        return 0.5 * y + rng.normal(0, 0.02, len(y))

    def fake_ridge(X, y, keys, des_list):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    monkeypatch.setattr(se, "_loo_lgb", fake_loo)
    monkeypatch.setattr(se, "_loo_ridge", fake_ridge)

    Xb = rng.normal(size=(n_des * per, 5))
    Xm = rng.normal(size=(n_des * per, 8))
    Xs = rng.normal(size=(n_des * per, 10))
    mae_bl = float(np.mean(np.abs(y - np.median(y))))
    rep = se.run_shape_screen(Xb, Xm, Xs, y, keys, tmp_path,
                              {"base": 5, "mfe": 8, "mfe_shape": 10},
                              {"m": 1.8, "b": -0.6, "clip": [0.0, 8.0]},
                              mae_bl)

    assert "results" in rep
    for k in ("base_3way", "mfe_3way", "mfe_shape_3way"):
        assert "skill" in rep["results"][k] and "r2" in rep["results"][k]
    g = rep["shape_marginal_gain"]
    assert "pooled_gain_pp" in g and "loo_exclusion" in g
    assert g["loo_exclusion"]["n_folds"] >= 3
    assert rep["n_features"]["mfe_shape"] == 10
    assert rep["deigan"]["m"] == 1.8
    assert (tmp_path / "m2r_shape_eval_report.json").exists()
    assert (tmp_path / "m2r_shape_eval_oof.npz").exists()
    d = json.loads((tmp_path / "m2r_shape_eval_report.json").read_text())
    assert d["schema"] == "reactflow_delta.m2r_shape_eval.v1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
