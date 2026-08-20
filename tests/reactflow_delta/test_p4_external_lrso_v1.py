#!/usr/bin/env python3
"""test_p4_external_lrso_v1: unit fixtures for the FINAL LRSO external cluster
validation (run_p4_external_lrso_v1).

Covers:
  - seqpos_offset / window_bounds: seqpos-correct alignment (reactivity array
    element k <-> sequence index seqpos[k]-1); regression guard for audit
    finding P4-M1 (legacy ridge run used index-0 alignment, offset ~26).
  - build_external_ctx: context covers exactly the observed window, sequence
    sliced from seqpos offset, WT-observed mask derived from reactivity.
  - _score_component: with a zero-delta stub model the LRSO CRPS must equal the
    ZeroResponse baseline exactly (D_vs_zero == 0), which validates window
    indices, wt_filled, mixture CRPS and baseline all share one coordinate
    system; positions outside the window are non-observed; attrition rules
    1/2/3 behave.
  - _load_frozen_graph: fails closed unless exactly 24 components.
  - aggregate_clusters: study-level (K_joint=2) aggregation, LOSO, CI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import scripts.reactflow_delta.run_p4_external_lrso_v1 as P


# --------------------------------------------------------------------------- #
# alignment helpers (audit finding P4-M1 regression guard)
# --------------------------------------------------------------------------- #
class TestSeqposOffset:
    def test_x27_gives_26(self):
        assert P.seqpos_offset(["X27", "X28", "X29"]) == 26

    def test_lowercase_x(self):
        assert P.seqpos_offset(["x7", "x8"]) == 6

    def test_plain_integer_label(self):
        assert P.seqpos_offset(["5", "6"]) == 4

    def test_empty(self):
        assert P.seqpos_offset([]) == 0

    def test_window_bounds(self):
        prof = {"reactivity": [0.1] * 139, "profile_sequence": "A" * 206}
        off, end = P.window_bounds(prof, ["X27", "X28"])
        assert off == 26
        assert end == 26 + 139


# --------------------------------------------------------------------------- #
# build_external_ctx
# --------------------------------------------------------------------------- #
class TestBuildExternalCtx:
    def test_window_alignment(self):
        # full sequence length 206, reactivity covers window [26, 165)
        seq = "A" * 26 + "C" * 139 + "U" * 41
        react = [0.5] * 139
        err = [0.1] * 139
        prof = {"profile_sequence": seq, "reactivity": react,
                "reactivity_error": err}
        ctx = P.build_external_ctx(prof, ["X27", "X28"], "cpu")
        seq_t, react_t, prec_t, obs_t, pos_t, region_t = ctx
        # context length == observed window length
        assert seq_t.shape == (139, 4)
        assert react_t.shape == (139,)
        # window sequence = seq[26:165] = all C's -> one-hot index 1
        assert (seq_t[:, 1] == 1.0).all()
        assert (seq_t[:, 0] == 0.0).all()
        # all observed, positions 0..138
        assert (obs_t == 1.0).all()
        assert pos_t[0].item() == 0.0 and pos_t[-1].item() == 138.0
        # region = design_region
        assert (region_t[:, 0] == 1.0).all() and (region_t[:, 1] == 0.0).all()
        # precision = -log(err)
        np.testing.assert_allclose(prec_t.numpy(),
                                   -np.log(0.1) * np.ones(139), rtol=1e-5)

    def test_nan_reactivity_mean_filled_with_obs_token(self):
        seq = "A" * 10 + "C" * 20
        react = [0.2] * 10 + [np.nan] * 20
        prof = {"profile_sequence": seq, "reactivity": react,
                "reactivity_error": [0.1] * 30}
        ctx = P.build_external_ctx(prof, ["X1"], "cpu")
        _s, react_t, _p, obs_t, _pos, _r = ctx
        assert (obs_t[:10] == 1.0).all() and (obs_t[10:] == 0.0).all()
        # missing filled with observed mean (0.2), NOT 0
        assert (react_t[10:] == 0.2).all()


# --------------------------------------------------------------------------- #
# _score_component mechanics (zero-delta stub => LRSO == zero baseline)
# --------------------------------------------------------------------------- #
class ZeroDeltaModel:
    """Stub deployment model: delta=0, scale=FIXED_SCALE. A model that predicts
    the WT profile exactly must tie the ZeroResponse baseline."""
    def __init__(self, scale: float = P.FIXED_SCALE) -> None:
        self._scale = scale

    def eval(self):
        return self

    def encode(self, ctx):
        return None

    def forward_op(self, H, edit_idx, dists, refs, alts, masks):
        B, L = masks.shape
        delta = torch.zeros(B, L)
        scale = torch.full((L,), self._scale, dtype=torch.float32)
        return delta, scale


def _profiles(seq_len: int, wlen: int, off: int, wt_val: float, mut_val: float,
              n_mutants: int = 25, edit_p: int | None = None,
              shared_region: list[int] | None = None):
    """Build WT + n mutant profiles sharing one reactivity window."""
    seq = "A" * seq_len
    react = [wt_val] * wlen
    profs = {"WT": {"profile_name": "WT", "profile_sequence": seq,
                    "reactivity": react, "reactivity_error": [0.1] * wlen}}
    if shared_region is None:
        shared_region = list(range(seq_len))
    if edit_p is None:
        edit_p = off + 2
    mutants = [{"name": f"C{k}A-C", "edit_pos": edit_p, "shared_region": shared_region}
               for k in range(n_mutants)]
    for k in range(n_mutants):
        profs[f"C{k}A-C"] = {"profile_name": f"C{k}A-C", "profile_sequence": seq,
                             "reactivity": [mut_val] * wlen,
                             "reactivity_error": [0.1] * wlen}
    comp = {"wt_name": "WT", "seq_len": seq_len, "n_snv_mutants": n_mutants,
            "mutants": mutants}
    return comp, profs


class TestScoreComponent:
    def test_zero_delta_ties_zero_baseline(self):
        # window [10, 35) of a 50-nt construct; shared region covers it
        comp, profs = _profiles(50, 25, 10, wt_val=0.9, mut_val=0.3)
        # seqpos labels X11..X35 => window [10, 35)
        seqpos = ["X11"] + [f"X{i}" for i in range(12, 35 + 1)]
        row, drop = P._score_component([ZeroDeltaModel()], comp, profs, seqpos, "cpu")
        assert drop is None
        assert row is not None
        # zero-delta model == WT-anchor prediction -> CRPS identical to baseline
        # (float32 wt_filled vs float64 baseline gives ~1e-8 rounding)
        assert row["crps_lrso"] == pytest.approx(row["crps_zero"], abs=1e-6)
        assert row["D_vs_zero"] == pytest.approx(0.0, abs=1e-6)
        # 25 mutants each with 25 window positions scored
        assert row["n_scored"] == 25
        assert row["n_positions"] == 25 * 25

    def test_out_of_window_shared_positions_are_non_observed(self):
        # window [26, 165) of a 206-nt construct; shared region 0..205
        seqpos = ["X27"] + [f"X{i}" for i in range(28, 165 + 1)]
        comp, profs = _profiles(206, 139, 26, 0.5, 0.5)
        # recompute shared_region over the full sequence
        comp["mutants"] = [dict(m, shared_region=list(range(206))) for m in comp["mutants"]]
        row, drop = P._score_component([ZeroDeltaModel()], comp, profs, seqpos, "cpu")
        assert drop is None
        # only window positions are scored: positions 26..164 (139), not 0..205
        assert row["n_positions"] == 25 * 139

    def test_rule3_insufficient_positions_drops_mutant(self):
        seqpos = ["X1"]
        wlen = 5
        seq = "A" * 30
        profs = {"WT": {"profile_name": "WT", "profile_sequence": seq,
                        "reactivity": [0.5] * wlen, "reactivity_error": [0.1] * wlen}}
        comp = {"wt_name": "WT", "seq_len": 30, "n_snv_mutants": 25,
                "mutants": [{"name": f"C{k}A-C", "edit_pos": 2,
                             "shared_region": [0, 1, 2, 3, 4]} for k in range(25)]}
        for k in range(25):
            profs[f"C{k}A-C"] = {"profile_name": f"C{k}A-C", "profile_sequence": seq,
                                 "reactivity": [0.5] * wlen, "reactivity_error": [0.1] * wlen}
        row, drop = P._score_component([ZeroDeltaModel()], comp, profs, seqpos, "cpu")
        # 5 shared positions < MIN_SHARED_NONMISSING (20) -> every mutant dropped
        assert row is None
        assert drop is not None and drop["rule"] == 2 and drop["n_scored"] == 0

    def test_rule2_insufficient_mutants(self):
        seqpos = ["X1"]
        comp, profs = _profiles(30, 25, 0, 0.5, 0.5, n_mutants=5)
        row, drop = P._score_component([ZeroDeltaModel()], comp, profs, seqpos, "cpu")
        assert row is None and drop["rule"] == 2 and drop["n_scored"] == 5

    def test_rule1_missing_wt(self):
        row, drop = P._score_component([ZeroDeltaModel()], {"wt_name": "NOPE"}, {}, ["X1"], "cpu")
        assert row is None and drop["rule"] == 1

    def test_nan_mutant_positions_skipped(self):
        seqpos = ["X1"]
        seq = "A" * 30
        profs = {"WT": {"profile_name": "WT", "profile_sequence": seq,
                        "reactivity": [0.5] * 25, "reactivity_error": [0.1] * 25}}
        comp = {"wt_name": "WT", "seq_len": 30, "n_snv_mutants": 25,
                "mutants": [{"name": f"C{k}A-C", "edit_pos": 5,
                             "shared_region": list(range(25))} for k in range(25)]}
        for k in range(25):
            r = [0.5] * 25
            r[1] = np.nan
            profs[f"C{k}A-C"] = {"profile_name": f"C{k}A-C", "profile_sequence": seq,
                                 "reactivity": r, "reactivity_error": [0.1] * 25}
        row, drop = P._score_component([ZeroDeltaModel()], comp, profs, seqpos, "cpu")
        assert drop is None
        # position 1 excluded for every mutant
        assert row["n_positions"] == 24 * 25


# --------------------------------------------------------------------------- #
# frozen graph
# --------------------------------------------------------------------------- #
class TestLoadFrozenGraph:
    def test_fails_closed_on_wrong_count(self, tmp_path):
        doc = {"direct_external": {"components": [{"n_snv_mutants": 1}] * 23}}
        p = tmp_path / "graph.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(RuntimeError, match="frozen graph mismatch"):
            P._load_frozen_graph(p)

    def test_matches_expected(self, tmp_path):
        doc = {"direct_external": {"components":
                                   [{"n_snv_mutants": 3} for _ in range(P.K_PREACCESS_EXPECTED)]}}
        p = tmp_path / "graph.json"
        p.write_text(json.dumps(doc))
        comps = P._load_frozen_graph(p)
        assert len(comps) == P.K_PREACCESS_EXPECTED


# --------------------------------------------------------------------------- #
# cluster aggregation (K_joint = 2)
# --------------------------------------------------------------------------- #
class TestAggregateClusters:
    def test_study_level_split(self):
        rows = [
            {"wt_name": "a", "dataset": "M2SL5_2A3_0000", "D_vs_zero": 0.01},
            {"wt_name": "b", "dataset": "M2SL5_2A3_0000", "D_vs_zero": 0.03},
            {"wt_name": "c", "dataset": "M3SARS_2A3_0000", "D_vs_zero": 0.02},
            {"wt_name": "d", "dataset": "15KLIB_2A3_0000", "D_vs_zero": 0.06},
        ]
        res = P.aggregate_clusters(rows)
        assert res["K_joint"] == 2
        assert res["N_study"] == 2
        assert res["cluster_macro_D_vs_zero"]["study_sl5"]["mean_D_vs_zero"] == pytest.approx(0.02)
        assert res["cluster_macro_D_vs_zero"]["study_ribonanza"]["mean_D_vs_zero"] == pytest.approx(0.04)
        assert res["cluster_macro_D_vs_zero"]["study_ribonanza"]["n_components"] == 2
        # LOSO with K=2 leaves the single other cluster
        assert res["loso"]["study_sl5"]["leave_out_mean_D_vs_zero"] == pytest.approx(0.04)
        assert res["loso"]["study_ribonanza"]["leave_out_mean_D_vs_zero"] == pytest.approx(0.02)
        assert res["cluster_level_ci"]["n"] == 2
        assert res["unknown_study_components"] == []

    def test_unknown_study_fail_closed(self):
        rows = [{"wt_name": "x", "dataset": "UNKNOWN", "D_vs_zero": 0.1}]
        res = P.aggregate_clusters(rows)
        assert res["unknown_study_components"] == ["x"]
        assert res["cluster_level_ci"]["ci_low"] is None

    def test_ref_alt_formats(self):
        assert P._ref_alt("SL5_SARS_CoV_2_0G-A_5pad6_w53barcode") == ("G", "A")
        assert P._ref_alt("miniTTR6:6DVK_0A-C_2pad3_libraryready") == ("A", "C")
        assert P._ref_alt("xxx_123C-G_yyy") == ("C", "G")
