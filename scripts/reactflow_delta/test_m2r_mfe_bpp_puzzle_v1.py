#!/usr/bin/env python3
"""Tests for m2r_mfe_bpp_puzzle_v1 — BPP+MFE puzzle-level audit."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_mfe_bpp_puzzle_v1 as bpppz


def test_run_puzzle_bpp_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_pz, per = 4, 25
    y = rng.normal(0.4, 0.25, n_pz * per)
    pz = np.concatenate([np.full(per, f"P{i}") for i in range(n_pz)])

    def fake_loo(X, y, pz, puzzles, obj, seed):
        return 0.5 * y + rng.normal(0, 0.02, len(y))

    def fake_ridge(X, y, pz, puzzles):
        return 0.4 * y + rng.normal(0, 0.05, len(y))

    monkeypatch.setattr(bpppz, "_loo_lgb_seed", fake_loo)
    monkeypatch.setattr(bpppz, "_loo_ridge", fake_ridge)

    class A:
        pass
    A.out = str(tmp_path / "out")
    A.base_npz = tmp_path / "base.npz"
    np.savez(A.base_npz, blend_K=rng.normal(0.4, 0.1, len(y)), y=y)
    A.n_perm = 100
    Xb = rng.normal(size=(n_pz * per, 10))
    rep = bpppz.run_puzzle_bpp(Xb, y, pz, A)

    assert "results" in rep
    for k in ("mfe_bpp_multiseed_3way", "mfe_multiseed_3way"):
        assert "skill" in rep["results"][k]
    g = rep["bpp_gain_vs_mfe"]
    assert "pooled_gain_pp" in g
    assert g["n_puzzles"] >= 3
    assert (tmp_path / "out" / "m2r_mfe_bpp_puzzle_report.json").exists()
    d = json.loads((tmp_path / "out" / "m2r_mfe_bpp_puzzle_report.json").read_text())
    assert d["schema"] == "reactflow_delta.m2r_mfe_bpp_puzzle.v1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))