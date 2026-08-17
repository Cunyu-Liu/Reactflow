#!/usr/bin/env python3
"""Tests for m2r_formula_blend_v1 — physics-constrained 4th blend member."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_formula_blend_v1 as fb


def _pair(rng, n=40, i=12, j=27, sub=(5, 35)):
    wt = rng.uniform(0, 1, n)
    ra = wt + rng.normal(0, 0.2, n)
    rb = wt + rng.normal(0, 0.25, n)
    rd = wt + rng.normal(0, 0.1, n)  # double mostly restored
    return m2r.M2RPair(
        design_id="OK7a_M2R_P1_m", puzzle="P1", method="m",
        mutA=6, mutB=23, editA_seq_pos=i, editB_seq_pos=j,
        sequence="A" * n,
        wt_reactivity=wt.tolist(), wt_error=[0.05] * n,
        singleA_reactivity=ra.tolist(), singleA_error=[0.05] * n,
        singleB_reactivity=rb.tolist(), singleB_error=[0.05] * n,
        double_reactivity=rd.tolist(), double_error=[0.05] * n,
        rescue_factor=None, eligibility_mask=[1] * n,
        target_structure="." * n, sub_start=sub[0], sub_end=sub[1],
        mutA_seq="A" * n, mutB_seq="A" * n,
        m2_structure="", m2_f1=None, m2_f1_crossed_pair=None)


def test_per_sample_geometry_matches_formula():
    rng = np.random.default_rng(1)
    s = _pair(rng)
    g = fb.per_sample_geometry(s)
    assert np.isfinite(g["rA"]) and np.isfinite(g["rB"]) and np.isfinite(g["rD"])
    assert np.isfinite(g["rnorm"]) and g["rnorm"] > 0
    expected = 1.0 - g["rD"] / g["rnorm"]
    assert g["rescue_exact"] == pytest.approx(expected)


def test_run_formula_blend_structure(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    n_des, per = 4, 25
    n = 40
    samples = []
    ys = []
    for d in range(n_des):
        for _ in range(per):
            s = _pair(rng)
            g = fb.per_sample_geometry(s)
            s.rescue_factor = g["rescue_exact"]
            samples.append(s)
            ys.append(g["rescue_exact"])
    y = np.array(ys)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U36")
    rD_pred = np.array([fb.per_sample_geometry(s)["rD"] for s in samples]) + rng.normal(0, 0.02, len(y))
    blend_base = y + rng.normal(0, 0.05, len(y))

    npz_path = tmp_path / "oof.npz"
    np.savez(npz_path, blend_base=blend_base, rD_pred=rD_pred, y=y, keys=keys)

    fake_designs = [{"puzzle": "P1", "method": "m"}]
    monkeypatch.setattr(m2r, "parse_m2r_csv", lambda p: (fake_designs, {"n_designs": 1}))
    monkeypatch.setattr(m2r, "build_all_pair_samples", lambda d: samples)

    class A:
        pass
    A.out = str(tmp_path / "out")
    rep = fb.run_formula_blend(str(npz_path), "dummy.csv", A)

    assert rep["align_corr_exact_vs_npz"] >= 0.999
    assert "results" in rep and "blend_base" in rep["results"]
    assert "formula_blend" in rep["results"]
    assert "formula_blend_gain" in rep and "loo_exclusion" in rep["formula_blend_gain"]
    assert rep["formula_blend_gain"]["loo_exclusion"]["n_folds"] >= 3
    assert "decorrelation_corr" in rep
    assert (tmp_path / "out" / "m2r_formula_blend_report.json").exists()


def test_metrics():
    y = np.array([0.0, 0.5, 1.0, 0.2, 0.8])
    p = np.array([0.1, 0.4, 0.9, 0.3, 0.7])
    mae = fb._mae(y, p)
    bl = fb._mae(y, np.full_like(y, np.median(y)))
    assert fb._skill(mae, bl) == pytest.approx(1.0 - mae / bl)
    assert 0.0 <= fb._r2(y, p) <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
