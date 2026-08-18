#!/usr/bin/env python3
"""Tests for m2r_mfe_features_v1 — ViennaRNA MFE legal feature extraction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_mfe_features_v1 as mfe


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


def test_pair_set_simple_and_pseudoknot():
    assert mfe.pair_set("((..))") == {(0, 5), (1, 4)}
    # nested [({...})]: open 0,1,2 close 6,7,8
    assert mfe.pair_set("([{...}])") == {(0, 8), (1, 7), (2, 6)}
    # pseudoknot: [.(.).]  -> [ at 0 closes ] at 6, ( at 2 closes ) at 4
    assert mfe.pair_set("[.(.).]") == {(0, 6), (2, 4)}


def test_pair_distance():
    assert mfe.pair_distance("((..))", "((..))") == 0
    assert mfe.pair_distance("((..))", "(....)") == 1
    assert mfe.pair_distance("((..))", "......") == 2


def test_pairing_status():
    i, j = 0, 5
    pi, pj, both = mfe.pairing_status("((..))", i, j)
    assert (pi, pj, both) == (1, 1, 1)
    # "(.....)" pairs (0,6); position 5 is unpaired
    pi, pj, both = mfe.pairing_status("(.....)", 0, 5)
    assert (pi, pj, both) == (1, 0, 0)


def test_build_mfe_features(monkeypatch):
    # WT and D both form the (10,22) pair; A and B fully disrupt.
    wt_struct = "." * 10 + "(" + "." * 11 + ")" + "." * 7  # pairs (10,22)
    dot_struct = "." * 30

    def fake_fold(seq):
        if all(c == "A" for c in seq):            # WT
            return wt_struct, -2.0
        if all(c == "C" for c in seq):            # mutB
            return dot_struct, -0.5
        if "G" in seq and "C" in seq:             # double (G..C..) restores
            return wt_struct, -1.9
        return dot_struct, -0.5                   # mutA (all G)

    def fake_centroid(seq):
        return fake_fold(seq)[0]

    monkeypatch.setattr(mfe, "_fold", fake_fold)
    monkeypatch.setattr(mfe, "_centroid", fake_centroid)

    f = mfe.build_mfe_features(_sample())
    names = mfe.mfe_feature_names()
    assert len(f) == len(names)
    # WT and D structures identical -> mfe_rescue ~ 1
    idx = names.index("mfe_rescue")
    assert f[idx] > 0.9
    # energy rescue also ~1 (dG_D ~ dG_WT, singles destabilize)
    idx_g = names.index("mfe_rescue_G")
    assert f[idx_g] > 0.9
    # WT pair at (i,j) formed; D pair formed too
    assert f[names.index("pa_wt_pair")] == 1.0
    assert f[names.index("pa_d_pair")] == 1.0
    assert f[names.index("rescue_cue_1")] >= 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
