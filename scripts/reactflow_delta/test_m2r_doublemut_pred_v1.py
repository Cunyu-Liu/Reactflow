#!/usr/bin/env python3
"""Tests for m2r_doublemut_pred_v1 — rD auxiliary-predictor lever."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_doublemut_pred_v1 as rdp
from m2r_data_v1 import M2RPair


def _sample(seed=0):
    rng = np.random.default_rng(seed)
    n = 40
    wt = rng.normal(0.5, 0.2, n)
    rd = wt + rng.normal(0, 0.3, n)
    return M2RPair(
        design_id="D", puzzle="P", method="M",
        mutA=6, mutB=16, editA_seq_pos=5, editB_seq_pos=15,
        sequence="A" * n, wt_reactivity=list(wt), wt_error=list(rng.normal(0.05, 0.01, n)),
        singleA_reactivity=list(wt + rng.normal(0, 0.2, n)),
        singleA_error=list(rng.normal(0.05, 0.01, n)),
        singleB_reactivity=list(wt + rng.normal(0, 0.2, n)),
        singleB_error=list(rng.normal(0.05, 0.01, n)),
        double_reactivity=list(rd), double_error=list(rng.normal(0.05, 0.01, n)),
        rescue_factor=0.5, eligibility_mask=[1] * n,
        target_structure="." * n, sub_start=1, sub_end=n,
        mutA_seq="A" * n, mutB_seq="A" * n)


def test_rmsd_double_wt():
    s = _sample()
    r = rdp.rmsd_double_wt(s)
    assert np.isfinite(r) and r > 0
    # perfect match -> 0
    s2 = _sample(1)
    s2.double_reactivity = list(s2.wt_reactivity)
    assert abs(rdp.rmsd_double_wt(s2)) < 1e-9


def test_region_mask():
    m = rdp._region_mask(10, 3, 7)
    assert m[:2].sum() == 0 and m[2:7].sum() == 5 and m[7:].sum() == 0


def test_run_lever_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_des, per = 4, 25
    n1, n2, nt = 6, 10, 3
    X1 = rng.normal(size=(n_des * per, n1))
    X2 = rng.normal(size=(n_des * per, n2))
    X_tr = rng.normal(size=(n_des * per, nt))
    y = X1[:, 0] * 0.5 + rng.normal(0, 0.2, n_des * per)
    rD = np.abs(X1[:, 0]) * 0.3 + rng.normal(0, 0.05, n_des * per)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")

    def fake_loo(X, y, keys, des_list, obj, seed=0):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    def fake_ridge(X, y, keys, des_list):
        return 0.3 * y + rng.normal(0, 0.1, len(y))

    monkeypatch.setattr(rdp, "_loo_lgb", fake_loo)
    monkeypatch.setattr(rdp, "_loo_ridge", fake_ridge)

    class A:
        pass
    A.out = str(tmp_path / "out")
    rep = rdp.run_lever(X1, X2, X_tr, rD, y, keys, A)
    assert "rd_predictability" in rep
    assert "results" in rep and "base_3way" in rep["results"]
    assert "aug_rD_3way" in rep["results"]
    assert "rD_gain" in rep and "loo_exclusion" in rep["rD_gain"]
    assert rep["rD_gain"]["loo_exclusion"]["n_folds"] >= 3
    assert (tmp_path / "out" / "m2r_doublemut_pred_report.json").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
