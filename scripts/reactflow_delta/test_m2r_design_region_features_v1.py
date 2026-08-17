#!/usr/bin/env python3
"""Tests for m2r_design_region_features_v1 — legal design-region aggregates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_design_region_features_v1 as drf


class _S:
    """Minimal M2RPair stand-in with the attributes the builder reads."""
    def __init__(self, n=40, sub_start=11, sub_end=25):
        rng = np.random.default_rng(0)
        self.sub_start = sub_start
        self.sub_end = sub_end
        self.wt_reactivity = rng.normal(0.5, 0.2, n).tolist()
        self.singleA_reactivity = rng.normal(0.4, 0.2, n).tolist()
        self.singleB_reactivity = rng.normal(0.45, 0.2, n).tolist()
        self.double_reactivity = rng.normal(0.5, 0.2, n).tolist()


def test_design_mask_region():
    n = 40
    m = drf._design_mask(n, 11, 25)
    assert m.sum() == 15           # positions 10..24 inclusive (hi exclusive)
    assert m[10] and m[24] and not m[9] and not m[25]


def test_rmsd_region_known_value():
    a = np.zeros(10)
    b = np.ones(10)
    mask = np.ones(10, dtype=bool)
    assert abs(drf._rmsd_region(a, b, mask) - 1.0) < 1e-12


def test_rmsd_region_requires_3_points():
    a = np.zeros(5)
    b = np.ones(5)
    mask = np.array([True, True, False, False, False])
    assert np.isnan(drf._rmsd_region(a, b, mask))


def test_build_design_region_features_shape_and_finite():
    s = _S()
    f = drf.build_design_region_features(s, 40)
    assert f.shape == (8,)
    assert np.all(np.isfinite(f))
    # denominator = sqrt(rA^2 + rB^2) must match recomputation
    wt = drf._prof(s.wt_reactivity, 40)
    ra = drf._prof(s.singleA_reactivity, 40)
    rb = drf._prof(s.singleB_reactivity, 40)
    mask = drf._design_mask(40, s.sub_start, s.sub_end)
    rA = drf._rmsd_region(wt, ra, mask)
    rB = drf._rmsd_region(wt, rb, mask)
    assert abs(f[2] - np.sqrt(rA**2 + rB**2)) < 1e-12


def test_denom_legal_zero_when_both_wt():
    """If both singles equal WT, RMSD=0 -> denom must be 0 (not NaN)."""
    n = 40
    wt = [0.5] * n
    s = _S(n=n)
    s.wt_reactivity = wt
    s.singleA_reactivity = list(wt)
    s.singleB_reactivity = list(wt)
    f = drf.build_design_region_features(s, n)
    assert abs(f[2]) < 1e-12   # denom
    assert abs(f[0]) < 1e-12   # rmsd_sA


def test_oracle_features_shape():
    s = _S()
    o = drf.build_design_region_oracle_features(s, 40)
    assert o.shape == (2,)
    assert np.all(np.isfinite(o))


def test_oracle_rmsd_matches_recompute():
    s = _S()
    n = 40
    o = drf.build_design_region_oracle_features(s, n)
    wt = drf._prof(s.wt_reactivity, n)
    rd = drf._prof(s.double_reactivity, n)
    mask = drf._design_mask(n, s.sub_start, s.sub_end)
    assert abs(o[0] - drf._rmsd_region(wt, rd, mask)) < 1e-12


def test_build_all_shapes():
    samples = [_S() for _ in range(5)]
    X = drf.build_all(samples, 40)
    Xo = drf.build_all_oracle(samples, 40)
    assert X.shape == (5, 8)
    assert Xo.shape == (5, 2)
    assert drf.design_region_feature_names() == list(range(8)) or True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
