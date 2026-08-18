#!/usr/bin/env python3
"""Tests for m2_gbdt_features_v1 — per-position legal features for M2 GBDT."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_gbdt_features_v1 as g


def _sample():
    n = 60
    return SimpleNamespace(
        design_id="OK7a_M2_P01_MPNN", puzzle="P01", method="MPNN",
        mutA=5, edit_seq_pos=30,
        sequence="A" * n,
        wt_reactivity=[0.1] * n, mut_reactivity=[0.2] * n,
        wt_error=[0.01] * n, mut_error=[0.02] * n,
        eligibility_mask=[1] * n,
        wt_rec={}, pair={"ref_allele": "A", "alt_allele": "G"},
    )


def test_base_oh():
    assert g._base_oh("A").tolist() == [1.0, 0, 0, 0, 0]
    assert g._base_oh("T").tolist() == [0, 0, 0, 1.0, 0]
    assert g._base_oh("N").tolist() == [0, 0, 0, 0, 1.0]
    assert g._base_oh("").tolist() == [0, 0, 0, 0, 1.0]


def test_mutant_sequence_replaces_edit_base():
    s = _sample()
    seq = g.mutant_sequence(s)
    assert seq[30] == "G"          # alt allele at edit site
    assert seq[29] == "A" and seq[31] == "A"


def test_structure_at(monkeypatch):
    # WT: all unpaired; the pair-change logic only needs deterministic output
    def fake_fold(seq):
        return "(" + "." * (len(seq) - 2) + ")", -1.0

    def fake_bpp(seq):
        n = len(seq)
        m = np.eye(n) * 0.1
        m[0][n - 1] = m[n - 1][0] = 0.9
        return m

    monkeypatch.setattr(g.mfe, "_fold", fake_fold)
    monkeypatch.setattr(g.mfe, "_bpp", fake_bpp)
    monkeypatch.setattr(g, "CACHE", {})
    paired, depth, bpp_p = g._structure_at("AC" * 30)
    assert paired[0] == 1.0 and paired[-1] == 1.0
    assert depth[0] == 1 and depth[1] == 1
    # bpp_paired[0] = 1 - bpp[0][0] = 1 - 0.1 = 0.9
    assert abs(bpp_p[0] - 0.9) < 1e-9


def test_feature_names_consistent():
    names = g.feature_names()
    assert len(names) == 31
    assert len(set(names)) == len(names)
    assert names[0:2] == ["pos_rel", "edit_dist"]


def test_features_are_leak_free():
    """The target is the mutant response; the mutant reactivity profile must
    NEVER appear as a feature (that would be circular)."""
    names = g.feature_names()
    joined = " ".join(names)
    assert "mut_" not in joined or joined.count("mut_") == 3  # only mut_paired/depth/bpp
    assert "d_m2" not in joined and "d_0" not in joined and "mut_err" not in joined
    # the only mut* features are the (legal) mutant-FOLD structural features
    for n in names:
        if n.startswith("mut_"):
            assert n in ("mut_paired", "mut_depth", "mut_bpp"), n


def test_build_position_features_semantics(monkeypatch):
    def fake_structure(seq):
        n = len(seq)
        return (np.zeros(n), np.zeros(n), np.zeros(n))

    monkeypatch.setattr(g, "_structure_at", fake_structure)
    s = _sample()
    f = g.build_position_features(s, g.HALF)   # central position
    names = g.feature_names()
    assert len(f) == len(names)
    # central position: pos_rel ~ 0, edit_dist ~ 0
    assert abs(f[names.index("pos_rel")]) < 1e-9
    assert f[names.index("edit_dist")] < 1e-9
    # base at edit site is A (WT)
    assert f[names.index("base_A")] == 1.0
    # wt_0 = 0.1
    assert abs(f[names.index("wt_0")] - 0.1) < 1e-9
    # ref A, alt G
    assert f[names.index("ref_A")] == 1.0
    assert f[names.index("alt_G")] == 1.0


def test_build_all_keeps_only_eligible_positions():
    def fake_structure(seq):
        n = len(seq)
        return (np.zeros(n), np.zeros(n), np.zeros(n))

    s1 = _sample()
    s2 = _sample()
    s2.design_id = "OK7a_M2_P02_MPNN"
    s2.mutA = 6
    # weight vector: only positions 0..2 eligible for both
    spectra = {
        f"{s1.design_id}:{s1.mutA}": {"y": [1.0] * 21, "w": [1, 1, 1] + [0] * 18,
                                      "design_id": s1.design_id},
        f"{s2.design_id}:{s2.mutA}": {"y": [2.0] * 21, "w": [1, 1, 1] + [0] * 18,
                                      "design_id": s2.design_id},
    }
    # patch CACHE and structure_at so no ViennaRNA is needed
    g.CACHE["__never__"] = True
    orig = g._structure_at
    g._structure_at = fake_structure
    try:
        X, y, w, keys, pids = g.build_all([s1, s2], spectra)
    finally:
        g._structure_at = orig
    assert len(y) == 6  # 2 samples x 3 eligible positions
    assert w.sum() == 6
    assert set(keys.tolist()) == {s1.design_id, s2.design_id}
    assert X.shape[1] == 31
    assert len(pids) == 6 and all(":0" in p or ":1" in p or ":2" in p for p in pids)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
