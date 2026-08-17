#!/usr/bin/env python3
"""Tests for m2r_features_v2 — cross-mutant overlap + structural context."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_features_v2 as m2rf2
from m2r_data_v1 import M2RPair


def _sample(n=40, i=5, j=15, sub_start=3, sub_end=30):
    seq = "A" * n
    rng = np.random.default_rng(0)
    wt = rng.normal(0.5, 0.2, n)
    sA = wt + rng.normal(0, 0.1, n)
    sB = wt + rng.normal(0, 0.1, n)
    # target structure: i and j paired inside a stem, rest dots
    st = ["."] * n
    st[i] = "("; st[j] = ")"
    # a small nested pair to give depth>1
    st[i + 1] = "("; st[j - 1] = ")"
    return M2RPair(
        design_id="D", puzzle="P", method="M",
        mutA=i + 1, mutB=j + 1, editA_seq_pos=i, editB_seq_pos=j,
        sequence=seq, wt_reactivity=list(wt), wt_error=list(rng.normal(0.05, 0.01, n)),
        singleA_reactivity=list(sA), singleA_error=list(rng.normal(0.05, 0.01, n)),
        singleB_reactivity=list(sB), singleB_error=list(rng.normal(0.05, 0.01, n)),
        double_reactivity=list(wt), double_error=list(rng.normal(0.05, 0.01, n)),
        rescue_factor=0.5, eligibility_mask=[1] * n,
        target_structure="".join(st), sub_start=sub_start, sub_end=sub_end,
        mutA_seq=seq, mutB_seq=seq, m2_structure="", m2_f1=0.9,
        m2_f1_crossed_pair=0.8)


def test_dot_to_depth_partner():
    pa, dp, partner = m2rf2.dot_to_depth("(.)")
    assert pa[0] == 1 and pa[1] == 0 and pa[2] == 1
    assert partner[0] == 2 and partner[2] == 0 and partner[1] == -1


def test_dot_to_depth_pseudoknot():
    pa, dp, partner = m2rf2.dot_to_depth("([)]")
    assert pa.sum() == 4
    # first-match-per-bracket-type pairing: ( pairs with ), [ pairs with ]
    assert set((partner[0], partner[2])) == {0, 2}
    assert set((partner[1], partner[3])) == {1, 3}
    assert dp.max() == 2


def test_stem_lengths():
    pa, dp, partner = m2rf2.dot_to_depth("((..))")
    stem = m2rf2._stem_lengths(pa, partner)
    # two contiguous paired runs (left stem [0,2) and right stem [4,6))
    assert stem[0] == 2 and stem[1] == 2 and stem[2] == 0 and stem[5] == 2


def test_build_v2_shapes_and_finite():
    s = _sample()
    f = m2rf2.build_v2_features(s)
    names = m2rf2.v2_feature_names()
    assert f.shape == (len(names),)
    assert len(names) == 22
    assert np.all(np.isfinite(f))


def test_overlap_detects_shared_disruption():
    """If both single mutants disrupt the SAME region, xcorr + spatial overlap
    (xmin_design) are high, and the feature set separates it from disjoint
    disruption."""
    rng = np.random.default_rng(1)
    n = 60

    def _build(shared: bool):
        wt = rng.normal(0.5, 0.2, n)
        sA = wt.copy(); sB = wt.copy()
        if shared:
            for k in range(10, 16):
                sA[k] += 2.0; sB[k] += 2.0
        else:
            for k in range(10, 16):
                sA[k] += 2.0
            for k in range(30, 36):
                sB[k] += 2.0
        st = ["."] * n
        return M2RPair(
            design_id="D", puzzle="P", method="M",
            mutA=11, mutB=21, editA_seq_pos=10, editB_seq_pos=20,
            sequence="A" * n, wt_reactivity=list(wt), wt_error=list(np.full(n, 0.05)),
            singleA_reactivity=list(sA), singleA_error=list(np.full(n, 0.05)),
            singleB_reactivity=list(sB), singleB_error=list(np.full(n, 0.05)),
            double_reactivity=list(wt), double_error=list(np.full(n, 0.05)),
            rescue_factor=0.8, eligibility_mask=[1] * n,
            target_structure="".join(st), sub_start=1, sub_end=n,
            mutA_seq="A" * n, mutB_seq="A" * n)

    fs = m2rf2.build_v2_features(_build(shared=True))
    fd = m2rf2.build_v2_features(_build(shared=False))
    names = m2rf2.v2_feature_names()
    assert fs[names.index("xcorr_design")] > 0.7
    assert fs[names.index("xcorr_design")] > fd[names.index("xcorr_design")]
    assert fs[names.index("xmin_design")] > fd[names.index("xmin_design")]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
