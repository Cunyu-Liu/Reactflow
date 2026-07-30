"""Tests for T-D1.5 probe eligibility masks (v3 §6.6 step 7)."""

from __future__ import annotations

import pytest

from reactflow.delta.data import (
    PROBE_ELIGIBLE_BASES,
    build_position_masks,
    build_probe_eligibility_mask,
    build_probe_eligibility_unchanged_mask,
    normalize_probe,
    verify_substitution,
)


class TestProbeEligibleBases:
    """T-D1.5: PROBE_ELIGIBLE_BASES frozen mapping (v3 §6.6 step 7 chemistry)."""

    def test_dms_modifies_a_and_c(self):
        # DMS methylates N1-A and N3-C (classic 2-base DMS-MaP reagent).
        assert PROBE_ELIGIBLE_BASES["DMS"] == frozenset({"A", "C"})

    def test_cmct_modifies_g_and_u(self):
        # CMCT carbodiimide-modifies N1-G and N3-U.
        assert PROBE_ELIGIBLE_BASES["CMCT"] == frozenset({"G", "U"})

    def test_shape_class_modifies_all_four_bases(self):
        # SHAPE-class acylates 2'-OH universally → all four RNA bases.
        assert PROBE_ELIGIBLE_BASES["SHAPE"] == frozenset({"A", "C", "G", "U"})

    def test_control_probes_have_empty_eligible_set(self):
        # nomod / none are no-probe controls → no eligible bases (mask all 0).
        assert PROBE_ELIGIBLE_BASES["NOMOD"] == frozenset()
        assert PROBE_ELIGIBLE_BASES["NONE"] == frozenset()

    def test_values_are_frozensets(self):
        # Frozen so the chemistry table cannot be mutated at runtime.
        for value in PROBE_ELIGIBLE_BASES.values():
            assert isinstance(value, frozenset)

    def test_all_keys_are_canonical(self):
        # Every key must also appear as a canonical alias target.
        assert set(PROBE_ELIGIBLE_BASES) == {"DMS", "CMCT", "SHAPE", "NOMOD", "NONE"}


class TestNormalizeProbe:
    """T-D1.5: normalize_probe alias collapsing (SHAPE-class → "SHAPE")."""

    @pytest.mark.parametrize(
        "alias",
        ["1M7", "NMIA", "SHAPE", "2A3"],
    )
    def test_shape_class_aliases_collapse_to_shape(self, alias):
        # 1M7, NMIA, SHAPE, 2A3 all acylate 2'-OH → single canonical "SHAPE".
        assert normalize_probe(alias) == "SHAPE"

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("DMS", "DMS"),
            ("CMCT", "CMCT"),
            ("NOMOD", "NOMOD"),
            ("NONE", "NONE"),
        ],
    )
    def test_canonical_probes_unchanged(self, alias, expected):
        assert normalize_probe(alias) == expected

    def test_case_insensitive_and_whitespace_stripped(self):
        assert normalize_probe(" dms ") == "DMS"
        assert normalize_probe("shape") == "SHAPE"
        assert normalize_probe("  2a3  ") == "SHAPE"

    @pytest.mark.parametrize(
        "probe",
        [None, "", "   ", "kethoxal", "glyoxal", "UNKNOWN", "DEPC"],
    )
    def test_unknown_probes_return_unknown(self, probe):
        # Unrecognized probes collapse to "UNKNOWN" without raising.
        assert normalize_probe(probe) == "UNKNOWN"

    def test_normalized_result_is_always_a_known_key_or_unknown(self):
        # The contract: normalize_probe returns a key of PROBE_ELIGIBLE_BASES
        # or the literal "UNKNOWN".
        for probe in ["DMS", "1M7", "2A3", "CMCT", "NOMOD", "NONE", "bogus", None]:
            result = normalize_probe(probe)
            assert result in set(PROBE_ELIGIBLE_BASES) or result == "UNKNOWN"


class TestBuildProbeEligibilityMask:
    """T-D1.5: construct-level probe-eligibility mask (v3 §6.6 step 7)."""

    def test_dms_mask_acgu(self):
        result = build_probe_eligibility_mask("ACGU", "DMS")
        assert result["mask"] == [1, 1, 0, 0]
        assert result["normalized_probe"] == "DMS"
        assert result["probe_known"] is True
        assert result["eligible_base_count"] == 2

    def test_shape_class_mask_all_ones(self):
        # SHAPE-class modifies every base → all positions eligible.
        for probe in ["SHAPE", "1M7", "NMIA", "2A3"]:
            result = build_probe_eligibility_mask("ACGU", probe)
            assert result["mask"] == [1, 1, 1, 1]
            assert result["normalized_probe"] == "SHAPE"
            assert result["probe_known"] is True
            assert result["eligible_base_count"] == 4

    def test_cmct_mask_acgu(self):
        result = build_probe_eligibility_mask("ACGU", "CMCT")
        assert result["mask"] == [0, 0, 1, 1]
        assert result["normalized_probe"] == "CMCT"
        assert result["eligible_base_count"] == 2

    def test_nomod_mask_all_zeros(self):
        result = build_probe_eligibility_mask("ACGU", "nomod")
        assert result["mask"] == [0, 0, 0, 0]
        assert result["normalized_probe"] == "NOMOD"
        assert result["probe_known"] is True
        assert result["eligible_base_count"] == 0

    def test_none_probe_mask_all_zeros(self):
        result = build_probe_eligibility_mask("ACGU", "none")
        assert result["mask"] == [0, 0, 0, 0]
        assert result["normalized_probe"] == "NONE"
        assert result["eligible_base_count"] == 0

    def test_unknown_probe_returns_none_mask(self):
        result = build_probe_eligibility_mask("ACGU", "kethoxal")
        assert result["mask"] is None
        assert result["normalized_probe"] == "UNKNOWN"
        assert result["probe_known"] is False
        assert result["eligible_base_count"] is None

    def test_none_probe_argument_returns_unknown(self):
        result = build_probe_eligibility_mask("ACGU", None)
        assert result["mask"] is None
        assert result["probe_known"] is False

    def test_t_to_u_normalization_applied(self):
        # DNA-typed "ACGT" must be T→U normalized to "ACGU" before masking.
        # Under CMCT {G,U}, position 3 (T→U) becomes eligible: [0,0,1,1].
        result = build_probe_eligibility_mask("ACGT", "CMCT")
        assert result["mask"] == [0, 0, 1, 1]
        assert result["eligible_base_count"] == 2

    def test_lowercase_sequence_handled(self):
        # _normalize_rna lowercases T→u only; bases are matched uppercase, so
        # lowercase A/C/G/U inputs are NOT eligible (only T is normalized).
        # This documents the defensive contract: callers should pass uppercase.
        result = build_probe_eligibility_mask("acgu", "DMS")
        assert result["mask"] == [0, 0, 0, 0]

    def test_none_sequence_returns_none_mask(self):
        result = build_probe_eligibility_mask(None, "DMS")
        assert result["mask"] is None
        assert result["normalized_probe"] == "DMS"
        assert result["probe_known"] is True
        assert result["eligible_base_count"] is None

    def test_empty_sequence(self):
        result = build_probe_eligibility_mask("", "DMS")
        assert result["mask"] == []
        assert result["eligible_base_count"] == 0

    def test_mask_length_matches_sequence(self):
        seq = "AUGCCAUGG"
        result = build_probe_eligibility_mask(seq, "DMS")
        assert len(result["mask"]) == len(seq)


class TestBuildProbeEligibilityUnchangedMask:
    """T-D1.5: pair-level probe-eligibility-unchanged mask (v3 §6.6 step 7)."""

    def test_dms_a_to_c_is_unchanged(self):
        # DMS {A,C}: A eligible, C eligible → same eligibility → unchanged.
        result = build_probe_eligibility_unchanged_mask("A", "C", "DMS")
        assert result["mask"] == [1]
        assert result["eligibility_changed_count"] == 0
        assert result["eligibility_changed_positions"] == []
        assert result["probe_known"] is True
        assert result["normalized_probe"] == "DMS"

    def test_dms_a_to_g_is_changed(self):
        # DMS {A,C}: A eligible, G ineligible → eligibility toggled → changed.
        result = build_probe_eligibility_unchanged_mask("A", "G", "DMS")
        assert result["mask"] == [0]
        assert result["eligibility_changed_count"] == 1
        assert result["eligibility_changed_positions"] == [0]

    def test_single_substitution_at_edit_position_changed(self):
        # wt="ACGU" mut="GCGU": edit at pos 0 (A→G) under DMS toggles
        # eligibility (A eligible, G not); positions 1-3 unchanged bases.
        result = build_probe_eligibility_unchanged_mask("ACGU", "GCGU", "DMS")
        assert result["mask"] == [0, 1, 1, 1]
        assert result["eligibility_changed_count"] == 1
        assert result["eligibility_changed_positions"] == [0]

    def test_unedited_positions_all_ones(self):
        # No edits → every position carries the same base in WT and mutant →
        # eligibility identical everywhere → mask all 1.
        result = build_probe_eligibility_unchanged_mask("ACGUACGU", "ACGUACGU", "DMS")
        assert result["mask"] == [1, 1, 1, 1, 1, 1, 1, 1]
        assert result["eligibility_changed_count"] == 0
        assert result["eligibility_changed_positions"] == []

    def test_shape_class_all_ones_because_all_eligible(self):
        # 2A3 (SHAPE-class) makes every base eligible, so any edit keeps
        # eligibility identical → unchanged mask all 1.
        for probe in ["2A3", "1M7", "SHAPE", "NMIA"]:
            result = build_probe_eligibility_unchanged_mask("ACGU", "GGGG", probe)
            assert result["mask"] == [1, 1, 1, 1]
            assert result["eligibility_changed_count"] == 0
            assert result["normalized_probe"] == "SHAPE"

    def test_nomod_all_ones_because_none_eligible(self):
        # nomod makes no base eligible, so eligibility is uniformly False in
        # both WT and mutant → unchanged mask all 1 (no eligibility toggle).
        result = build_probe_eligibility_unchanged_mask("ACGU", "GGGG", "nomod")
        assert result["mask"] == [1, 1, 1, 1]
        assert result["eligibility_changed_count"] == 0
        assert result["normalized_probe"] == "NOMOD"

    def test_changed_positions_are_zero_indexed(self):
        # wt="GGGGG" mut="GAGGG": edit at pos 1 (G→A) under DMS toggles
        # eligibility (G ineligible, A eligible) → changed position is index 1.
        result = build_probe_eligibility_unchanged_mask("GGGGG", "GAGGG", "DMS")
        assert result["mask"] == [1, 0, 1, 1, 1]
        assert result["eligibility_changed_positions"] == [1]
        assert result["eligibility_changed_count"] == 1

    def test_multiple_changed_positions_collected_sorted(self):
        # wt="GGGG" mut="AAGG" under DMS: pos0 G→A (changed), pos1 G→A
        # (changed), pos2-3 unchanged.
        result = build_probe_eligibility_unchanged_mask("GGGG", "AAGG", "DMS")
        assert result["mask"] == [0, 0, 1, 1]
        assert result["eligibility_changed_positions"] == [0, 1]
        assert result["eligibility_changed_count"] == 2

    def test_unknown_probe_returns_none_mask(self):
        result = build_probe_eligibility_unchanged_mask("ACGU", "GCGU", "kethoxal")
        assert result["mask"] is None
        assert result["normalized_probe"] == "UNKNOWN"
        assert result["probe_known"] is False
        assert result["eligibility_changed_count"] is None
        assert result["eligibility_changed_positions"] is None

    def test_none_probe_argument_returns_none_mask(self):
        result = build_probe_eligibility_unchanged_mask("ACGU", "GCGU", None)
        assert result["mask"] is None
        assert result["probe_known"] is False

    def test_none_wt_sequence_returns_none_mask(self):
        result = build_probe_eligibility_unchanged_mask(None, "ACGU", "DMS")
        assert result["mask"] is None
        assert result["probe_known"] is True
        assert result["eligibility_changed_count"] is None

    def test_none_mut_sequence_returns_none_mask(self):
        result = build_probe_eligibility_unchanged_mask("ACGU", None, "DMS")
        assert result["mask"] is None
        assert result["probe_known"] is True

    def test_t_to_u_normalization_applied_to_both(self):
        # DNA-typed sequences are normalized before comparison. wt="ACGT"
        # mut="GCGT" → "ACGU"/"GCGU": pos0 A→G under DMS toggles eligibility.
        result = build_probe_eligibility_unchanged_mask("ACGT", "GCGT", "DMS")
        assert result["mask"] == [0, 1, 1, 1]
        assert result["eligibility_changed_positions"] == [0]

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            build_probe_eligibility_unchanged_mask("ACGU", "ACG", "DMS")


class TestProbeEligibilityIntegration:
    """T-D1.5: integration verify_substitution → build_position_masks →
    build_probe_eligibility_unchanged_mask (v3 §6.6 steps 4→6→7)."""

    def test_edit_preserving_eligibility(self):
        # wt="GACGU" mut="GCCGU": single sub A→C at pos 1 under DMS.
        # A and C are both DMS-eligible → eligibility unchanged at edit site.
        verify = verify_substitution("GACGU", "GCCGU")
        assert verify["edit_positions"] == [1]
        masks = build_position_masks(verify["edit_positions"], len("GACGU"))
        assert masks["unchanged_position_mask"] == [1, 0, 1, 1, 1]
        elig = build_probe_eligibility_unchanged_mask("GACGU", "GCCGU", "DMS")
        # No eligibility toggle anywhere (A→C both eligible under DMS).
        assert elig["mask"] == [1, 1, 1, 1, 1]
        assert elig["eligibility_changed_count"] == 0

    def test_edit_toggling_eligibility_aligns_with_changed_mask(self):
        # wt="GACGU" mut="GGCGU": single sub A→G at pos 1 under DMS.
        # A eligible, G not → eligibility toggled at the edit position only.
        verify = verify_substitution("GACGU", "GGCGU")
        assert verify["edit_positions"] == [1]
        masks = build_position_masks(verify["edit_positions"], len("GACGU"))
        assert masks["changed_position_mask"] == [0, 1, 0, 0, 0]
        elig = build_probe_eligibility_unchanged_mask("GACGU", "GGCGU", "DMS")
        # Eligibility-unchanged mask is 0 exactly at the edited position.
        assert elig["mask"] == [1, 0, 1, 1, 1]
        assert elig["eligibility_changed_positions"] == [1]
        assert elig["eligibility_changed_count"] == 1
        # The eligibility-changed position is a subset of the edit positions.
        assert set(elig["eligibility_changed_positions"]).issubset(
            set(verify["edit_positions"])
        )

    def test_shape_class_edit_never_toggles_eligibility(self):
        # Under SHAPE-class every base is eligible, so no edit can toggle
        # eligibility → unchanged mask is all 1 regardless of the edit.
        verify = verify_substitution("GACGU", "GGCGU")
        masks = build_position_masks(verify["edit_positions"], len("GACGU"))
        elig = build_probe_eligibility_unchanged_mask("GACGU", "GGCGU", "2A3")
        assert elig["mask"] == [1, 1, 1, 1, 1]
        assert elig["eligibility_changed_count"] == 0

    def test_main_endpoint_position_gate_logic(self):
        # §12.1 main endpoint restricts to unedited + aligned +
        # eligibility-unchanged + valid positions. With an edit that toggles
        # eligibility, the regression-eligible positions are exactly the
        # positions that are unchanged AND eligibility-unchanged.
        wt = "GACGU"
        mut = "GGCGU"
        verify = verify_substitution(wt, mut)
        masks = build_position_masks(verify["edit_positions"], len(wt))
        elig = build_probe_eligibility_unchanged_mask(wt, mut, "DMS")
        unchanged = masks["unchanged_position_mask"]
        elig_unchanged = elig["mask"]
        # Regression-eligible positions (assuming all valid): unchanged AND
        # eligibility-unchanged.
        regression_eligible = [
            i for i in range(len(wt)) if unchanged[i] == 1 and elig_unchanged[i] == 1
        ]
        # pos 1 is edited (excluded); all other positions qualify.
        assert regression_eligible == [0, 2, 3, 4]
