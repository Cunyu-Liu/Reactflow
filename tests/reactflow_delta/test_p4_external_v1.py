#!/usr/bin/env python3
"""Unit fixtures for run_p4_external_v1 locked external protocol.

Covers:
  - _ref_alt: single-SNV token parsing for all three external name formats.
  - _load_frozen_graph: fails closed unless the frozen graph has exactly 24
    components (no outcome access on mismatch).
  - _score_component: bounded shared-region scoring (reactivity arrays are
    shorter than sequences: 3' pads/barcodes are non-observed), per-mutant
    rule-3 filter (>= 20 non-missing shared positions), per-component rule-2
    filter (>= 20 scored mutants), and the exact IndexError regression that
    killed the first execution.
  - _ci_one_sided: one-sided 95% t-CI used for the component-macro estimand.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.reactflow_delta.run_p4_external_v1 as P


def _wt_profile(seq: str, reactivity: list[float | None], name: str = "WT") -> dict:
    return {"profile_name": name, "profile_sequence": seq, "reactivity": reactivity}


def _mutant_profile(name: str, seq: str, reactivity: list[float | None]) -> dict:
    return {"profile_name": name, "profile_sequence": seq, "reactivity": reactivity}


class TestRefAlt:
    def test_m2sl5_format(self):
        # SNV token is positionless "0G-A" style in M2SL5 names
        assert P._ref_alt("SL5_SARS_CoV_2_0G-A_5pad6_w53barcode") == ("G", "A")

    def test_m3sars_format(self):
        assert P._ref_alt("fse__NC_045512.2_13470-13569_libraryready_4G-C_0pad27_librar") == ("G", "C")

    def test_15klib_format(self):
        assert P._ref_alt("miniTTR6:6DVK_0A-C_2pad3_libraryready") == ("A", "C")
        assert P._ref_alt("8000_construct_12U-A_0pad0_libraryready") == ("U", "A")

    def test_fallback_regex(self):
        # multi-digit token: reversed-split heuristic fails, regex fallback wins
        assert P._ref_alt("xxx_123C-G_yyy") == ("C", "G")


class TestLoadFrozenGraph:
    def test_matches_expected(self, tmp_path):
        doc = {
            "direct_external": {
                "components": [
                    {"n_snv_mutants": 3} for _ in range(P.K_PREACCESS_EXPECTED)
                ]
            }
        }
        p = tmp_path / "graph.json"
        p.write_text(json.dumps(doc))
        comps, total = P._load_frozen_graph(p)
        assert len(comps) == P.K_PREACCESS_EXPECTED
        assert total == 3 * P.K_PREACCESS_EXPECTED

    def test_fails_closed_on_wrong_count(self, tmp_path):
        doc = {"direct_external": {"components": [{"n_snv_mutants": 1}] * 23}}
        p = tmp_path / "graph.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(RuntimeError, match="frozen graph mismatch"):
            P._load_frozen_graph(p)


class TestScoreComponent:
    def _coef(self) -> np.ndarray:
        return np.zeros(13)

    def _make_component(self, seq_len: int, shared: list[int], edit_pos: int,
                        n_mutants: int = 25) -> dict:
        # all mutants share the same mask; reactivity arrays shorter than seq
        mutants = [{"name": f"C1_{k}A-C", "edit_pos": edit_pos,
                    "shared_region": shared} for k in range(n_mutants)]
        return {
            "wt_name": "WT", "seq_len": seq_len, "n_snv_mutants": n_mutants,
            "mutants": mutants,
        }

    def test_out_of_bounds_shared_indices_are_non_observed(self):
        # seq_len 30, reactivity only covers 25 positions -> indices 25..29 are
        # non-observed and MUST NOT raise IndexError (regression of first run)
        seq = "A" * 30
        wt = _wt_profile(seq, [0.5] * 25)  # reactivity shorter than sequence
        profs = {"WT": wt}
        shared = list(range(30))  # includes out-of-coverage indices
        comp = self._make_component(30, shared, edit_pos=10, n_mutants=25)
        for k in range(25):
            profs[f"C1_{k}A-C"] = _mutant_profile(
                f"C1_{k}A-C", seq, [0.5] * 25)
        row, drop = P._score_component(self._coef(), comp, profs)
        assert drop is None
        assert row is not None
        # only the 25 covered positions are scored, not the 30 sequence positions
        assert row["n_positions"] == 25 * 25
        assert row["n_scored"] == 25

    def test_edit_pos_beyond_coverage_sets_we_to_zero(self):
        seq = "A" * 30
        wt = _wt_profile(seq, [0.5] * 20)
        profs = {"WT": wt}
        # edit at position 27, which is beyond reactivity coverage (20)
        comp = self._make_component(30, list(range(20)), edit_pos=27, n_mutants=25)
        for k in range(25):
            profs[f"C1_{k}A-C"] = _mutant_profile(f"C1_{k}A-C", seq, [0.5] * 20)
        row, drop = P._score_component(self._coef(), comp, profs)
        assert drop is None
        assert row is not None
        assert row["n_positions"] == 20 * 25

    def test_nan_positions_skipped(self):
        # 25 shared positions, position 1 is NaN in both WT and mutant profiles
        wt_react = [0.5] * 25
        wt_react[1] = np.nan
        wt = _wt_profile("A" * 30, wt_react)
        profs = {"WT": wt}
        shared = list(range(25))
        comp = self._make_component(30, shared, edit_pos=2, n_mutants=25)
        for k in range(25):
            mut = [0.5] * 25
            mut[1] = np.nan
            profs[f"C1_{k}A-C"] = _mutant_profile(f"C1_{k}A-C", "A" * 30, mut)
        row, drop = P._score_component(self._coef(), comp, profs)
        assert drop is None
        assert row is not None
        # position 1 excluded for every mutant (24 positions x 25 mutants)
        assert row["n_positions"] == 24 * 25
        assert row["n_scored"] == 25

    def test_rule3_insufficient_positions_drops_mutant(self):
        seq = "A" * 30
        wt = _wt_profile(seq, [0.5] * 5)
        profs = {"WT": wt}
        # shared region has only 5 positions < MIN_SHARED_NONMISSING (20)
        comp = self._make_component(5, [0, 1, 2, 3, 4], edit_pos=2, n_mutants=25)
        for k in range(25):
            profs[f"C1_{k}A-C"] = _mutant_profile(f"C1_{k}A-C", seq[:5], [0.5] * 5)
        row, drop = P._score_component(self._coef(), comp, profs)
        # every mutant fails rule 3 -> component dropped by rule 2
        assert row is None
        assert drop is not None
        assert drop["rule"] == 2
        assert drop["n_scored"] == 0

    def test_rule2_insufficient_scored_mutants_drops_component(self):
        seq = "A" * 40
        wt = _wt_profile(seq, [0.5] * 25)
        profs = {"WT": wt}
        shared = list(range(25))
        # only 5 mutants (not >= MIN_SCORED_MUTANTS=20)
        comp = self._make_component(40, shared, edit_pos=5, n_mutants=5)
        for k in range(5):
            profs[f"C1_{k}A-C"] = _mutant_profile(f"C1_{k}A-C", seq, [0.5] * 25)
        row, drop = P._score_component(self._coef(), comp, profs)
        assert row is None
        assert drop is not None
        assert drop["rule"] == 2
        assert drop["n_scored"] == 5
        assert drop["n_matched"] == 5

    def test_missing_wt_profile_drops_rule1(self):
        comp = self._make_component(30, list(range(25)), edit_pos=5)
        row, drop = P._score_component(self._coef(), comp, {})
        assert row is None
        assert drop is not None
        assert drop["rule"] == 1

    def test_direct_vs_zero_positive_when_direct_is_exact(self):
        # WT profile flat 0.9; mutant targets flat 0.3; coef = intercept 0.3
        # makes the direct model predict every target exactly, while the
        # WT-anchor baseline predicts 0.9 (wrong) -> D_vs_zero > 0.
        # Feat template: [intercept, we, wt_r, dist, tanh(dist), ref(4), alt(4)].
        coef = np.zeros(13)
        coef[0] = 0.3  # intercept -> prediction == 0.3 == target everywhere
        wt_react = [0.9] * 25
        wt = _wt_profile("A" * 30, wt_react)
        profs = {"WT": wt}
        shared = list(range(25))
        comp = self._make_component(30, shared, edit_pos=5, n_mutants=25)
        for k in range(25):
            profs[f"C1_{k}A-C"] = _mutant_profile(f"C1_{k}A-C", "A" * 30, [0.3] * 25)
        row, drop = P._score_component(coef, comp, profs)
        assert drop is None
        assert row is not None
        # direct predicts exactly -> its CRPS is far below the WT-anchor baseline
        assert row["crps_direct"] < row["crps_zero"]
        assert row["D_vs_zero"] > 0.0


class TestCiOneSided:
    def test_ci_low_positive(self):
        ci = P._ci_one_sided([0.1] * 10)
        assert ci["ci_low"] > 0.0
        assert ci["n"] == 10

    def test_too_few(self):
        ci = P._ci_one_sided([0.1])
        assert ci["ci_low"] is None
