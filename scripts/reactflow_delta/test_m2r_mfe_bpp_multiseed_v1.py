#!/usr/bin/env python3
"""Tests for m2r_mfe_bpp_multiseed_v1 — BPP+MFE multi-seed audit harness."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_mfe_bpp_multiseed_v1 as bppms


def test_run_mfe_bpp_multiseed_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_des, per = 4, 25
    y = rng.normal(0.4, 0.25, n_des * per)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")

    def fake_loo(X, y, keys, des_list, obj, seed):
        return 0.5 * y + rng.normal(0, 0.02, len(y))

    def fake_ridge(X, y, keys, des_list):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    monkeypatch.setattr(bppms, "_loo_lgb_seed", fake_loo)
    monkeypatch.setattr(bppms, "_loo_ridge", fake_ridge)

    class A:
        pass
    A.out = str(tmp_path / "out")
    A.base_npz = tmp_path / "base.npz"
    np.savez(A.base_npz, blend_K=rng.normal(0.4, 0.1, len(y)), y=y)
    A.n_perm = 100
    Xb = rng.normal(size=(n_des * per, 10))
    rep = bppms.run_mfe_bpp_multiseed(Xb, y, keys, A)

    assert "results" in rep
    for k in ("mfe_bpp_multiseed_3way", "mfe_multiseed_3way"):
        assert "skill" in rep["results"][k] and "r2" in rep["results"][k]
    g = rep["bpp_gain_vs_mfe"]
    assert "pooled_gain_pp" in g and "loo_exclusion" in g
    assert g["loo_exclusion"]["n_folds"] >= 3
    assert (tmp_path / "out" / "m2r_mfe_bpp_multiseed_report.json").exists()
    assert (tmp_path / "out" / "m2r_mfe_bpp_multiseed_oof.npz").exists()
    d = json.loads((tmp_path / "out" / "m2r_mfe_bpp_multiseed_report.json").read_text())
    assert d["schema"] == "reactflow_delta.m2r_mfe_bpp_multiseed.v1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
