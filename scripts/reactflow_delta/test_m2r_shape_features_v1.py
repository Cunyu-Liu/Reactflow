#!/usr/bin/env python3
"""Tests for m2r_shape_features_v1 — SHAPE-guided thermodynamic features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_mfe_features_v1 as mfe
import m2r_shape_features_v1 as shape


def _sample():
    n = 30
    return m2r.M2RPair(
        design_id="OK7a_M2R_P1_m", puzzle="P1", method="m",
        mutA=6, mutB=18, editA_seq_pos=10, editB_seq_pos=22,
        sequence="A" * n, wt_reactivity=[0.0] * n, wt_error=[0.0] * n,
        singleA_reactivity=[0.0] * n, singleA_error=[0.0] * n,
        singleB_reactivity=[0.0] * n, singleB_error=[0.0] * n,
        double_reactivity=[0.0] * n, double_error=[0.0] * n,
        rescue_factor=0.5, eligibility_mask=[1] * n,
        target_structure="." * n, sub_start=1, sub_end=n,
        mutA_seq="G" * n, mutB_seq="C" * n,
        m2_structure="", m2_f1=None, m2_f1_crossed_pair=None)


def test_clean_shape_handles_missing_and_clipping():
    out = shape._clean_shape([None, float("nan"), -7.0, 0.5, 9.0])
    assert out == [0.0, 0.0, 0.0, 0.5, 8.0]


def test_shape_feature_names_consistent_with_build():
    names = shape.shape_feature_names()
    assert len(names) == 30
    assert names[0:5] == ["rA_s", "rB_s", "rD_s", "rnorm_s", "rescue_s"]
    assert "rescue_G_s" in names
    assert len(set(names)) == len(names)


def test_build_shape_features_semantics(monkeypatch):
    # WT and D both form the (10,22) pair under SHAPE guidance; A/B disrupt.
    wt_struct = "." * 10 + "(" + "." * 11 + ")" + "." * 7  # pairs (10,22)
    dot_struct = "." * 30

    def fake_constrained(seq, shape):
        if all(c == "A" for c in seq):            # WT guided -> WT pair
            return wt_struct, -2.0
        if all(c == "C" for c in seq):            # mutB guided -> dot
            return dot_struct, -0.5
        if "G" in seq and "C" in seq:             # double -> restores (plain)
            return wt_struct, -1.9
        return dot_struct, -0.5                   # mutA (all G)

    def fake_plain(seq):
        if all(c == "A" for c in seq):
            return wt_struct, -2.0
        if all(c == "C" for c in seq):
            return dot_struct, -0.5
        if "G" in seq and "C" in seq:
            return wt_struct, -1.9                # plain D restores the pair
        return dot_struct, -0.5

    monkeypatch.setattr(shape, "_fold_constrained", fake_constrained)
    monkeypatch.setattr(mfe, "_fold", fake_plain)

    f = shape.build_shape_features(_sample())
    names = shape.shape_feature_names()
    assert len(f) == len(names)
    # WT and D structures identical under guidance -> rescue_s ~ 1
    assert f[names.index("rescue_s")] > 0.9
    # WT guided structure = WT plain structure (both form the pair) -> div_wt = 0
    assert f[names.index("div_wt")] == 0.0
    # D: guided(plain) structure here is dot, plain D in this test is dot for
    # the double sequence -> divergence only where the two differ
    assert f[names.index("ps_wt_pair")] == 1.0
    assert f[names.index("ps_d_pair")] == 1.0


def test_fold_constrained_caches_per_sequence(monkeypatch):
    calls = []

    class FakeFC:
        def __init__(self, seq):
            self.seq = seq

        def sc_add_SHAPE_deigan(self, vals, m, b, opt):
            calls.append(("add", self.seq))
            return 1

        def mfe(self):
            calls.append(("mfe", self.seq))
            return "(((...)))", -3.0

    fake_rna = type("FakeRNA", (), {
        "fold_compound": staticmethod(lambda seq: FakeFC(seq))
    })
    monkeypatch.setitem(sys.modules, "RNA", fake_rna)
    monkeypatch.setattr(shape, "CACHE", {})
    s = _sample()
    r1 = shape._fold_constrained(s.sequence, s.wt_reactivity)
    r2 = shape._fold_constrained(s.sequence, s.wt_reactivity)
    assert r1 == r2 == ("(((...)))", -3.0)
    assert len(calls) == 2  # one add + one mfe, second call served from cache


def test_fold_constrained_falls_back_on_error(monkeypatch):
    class FakeFC:
        def sc_add_SHAPE_deigan(self, vals, m, b, opt):
            return 0

    fake_rna = type("FakeRNA", (), {
        "fold_compound": staticmethod(lambda seq: FakeFC())
    })
    monkeypatch.setitem(sys.modules, "RNA", fake_rna)
    monkeypatch.setattr(shape, "CACHE", {})

    def fake_plain(seq):
        return "(((...)))", -1.0

    monkeypatch.setattr(mfe, "_fold", fake_plain)
    assert shape._fold_constrained("ACGUACGU", [0.1] * 8) == ("(((...)))", -1.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
