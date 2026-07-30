"""Tests for D1 cleaning pipeline functions (T-D1.2+)."""

from __future__ import annotations

import pytest

from reactflow.delta.data import (
    COMPARABLE_MIN_FRACTION,
    CONDITION_MATCH_FIELDS,
    build_position_masks,
    check_comparable_positions,
    match_conditions,
    verify_annotation_ref,
    verify_substitution,
)
from reactflow.delta.schema import EXCLUSION_REASONS


def _wt_construct(**overrides) -> dict:
    """Build a WT construct with default conditions for testing."""
    base = {
        "probe": "DMS",
        "probe_protocol": None,
        "temperature": None,
        "ligand": None,
        "ligand_concentration": None,
        "buffer": None,
        "in_vivo_in_vitro": "in_vitro",
    }
    base.update(overrides)
    return base


def _mut_construct(**overrides) -> dict:
    """Build a mutant construct with default conditions matching WT."""
    base = {
        "probe": "DMS",
        "probe_protocol": None,
        "temperature": None,
        "ligand": None,
        "ligand_concentration": None,
        "buffer": None,
        "in_vivo_in_vitro": "in_vitro",
    }
    base.update(overrides)
    return base


class TestConditionMatching:
    """T-D1.2: condition exact matching (v3 §6.5, §6.6 step 5)."""

    def test_all_conditions_match(self):
        wt = _wt_construct()
        mut = _mut_construct()
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "exact_match"
        assert result["mismatched_fields"] == []
        assert set(result["condition_match_fields"]) == set(CONDITION_MATCH_FIELDS)

    def test_probe_mismatch(self):
        wt = _wt_construct(probe="DMS")
        mut = _mut_construct(probe="2A3")
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "mismatch"
        assert "probe" in result["mismatched_fields"]
        assert "probe" not in result["condition_match_fields"]

    def test_temperature_mismatch(self):
        wt = _wt_construct(temperature=25.0)
        mut = _mut_construct(temperature=37.0)
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "mismatch"
        assert "temperature" in result["mismatched_fields"]

    def test_both_null_matches(self):
        wt = _wt_construct(temperature=None)
        mut = _mut_construct(temperature=None)
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "exact_match"
        assert "temperature" in result["condition_match_fields"]

    def test_one_null_one_not_mismatches(self):
        wt = _wt_construct(temperature=None)
        mut = _mut_construct(temperature=25.0)
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "mismatch"
        assert "temperature" in result["mismatched_fields"]

    def test_in_vivo_in_vitro_mismatch(self):
        wt = _wt_construct(in_vivo_in_vitro="in_vivo")
        mut = _mut_construct(in_vivo_in_vitro="in_vitro")
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "mismatch"
        assert "in_vivo_in_vitro" in result["mismatched_fields"]

    def test_multiple_mismatches(self):
        wt = _wt_construct(probe="DMS", temperature=25.0, ligand="MG2")
        mut = _mut_construct(probe="2A3", temperature=37.0, ligand=None)
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "mismatch"
        assert set(result["mismatched_fields"]) == {"probe", "temperature", "ligand"}

    def test_empty_dicts_match(self):
        """Constructs with no condition fields: all None == None."""
        result = match_conditions({}, {})
        assert result["condition_match_status"] == "exact_match"
        assert len(result["condition_match_fields"]) == len(CONDITION_MATCH_FIELDS)

    def test_ligand_concentration_string_mismatch(self):
        wt = _wt_construct(ligand_concentration="10 mM")
        mut = _mut_construct(ligand_concentration="10mM")
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "mismatch"
        assert "ligand_concentration" in result["mismatched_fields"]

    def test_int_float_temperature_equal(self):
        """25 (int) and 25.0 (float) should match."""
        wt = _wt_construct(temperature=25)
        mut = _mut_construct(temperature=25.0)
        result = match_conditions(wt, mut)
        assert result["condition_match_status"] == "exact_match"

    def test_all_seven_fields_checked(self):
        """Verify all 7 condition fields are in the match set."""
        assert len(CONDITION_MATCH_FIELDS) == 7
        assert set(CONDITION_MATCH_FIELDS) == {
            "probe", "probe_protocol", "temperature",
            "ligand", "ligand_concentration", "buffer",
            "in_vivo_in_vitro",
        }

    def test_condition_match_fields_only_contains_matched(self):
        """condition_match_fields should only contain fields that matched."""
        wt = _wt_construct(probe="DMS", buffer="HEPES")
        mut = _mut_construct(probe="2A3", buffer="HEPES")
        result = match_conditions(wt, mut)
        assert "buffer" in result["condition_match_fields"]
        assert "probe" not in result["condition_match_fields"]
        assert "probe" in result["mismatched_fields"]


class TestVerifySubstitution:
    """T-D1.3: sequence-based substitution verification (v3.1 §3.1)."""

    def test_single_substitution_passes(self):
        wt = "ACGUGCAC"
        mut = "ACGAGCAC"  # position 3 (0-indexed) U->A
        r = verify_substitution(wt, mut)
        assert r["is_substitution_single"] is True
        assert r["edit_type"] == "substitution"
        assert r["edit_count"] == 1
        assert r["edit_positions"] == [3]
        assert r["wt_alleles"] == ["U"]
        assert r["mut_alleles"] == ["A"]
        assert r["alignment_cigar"] == "8M"
        assert r["exclusion_reason"] is None

    def test_identical_sequences_zero_edits(self):
        """No-edit pair (replicate/no-edit control, T-D1.6): not a single sub."""
        wt = "ACGUGCAC"
        r = verify_substitution(wt, wt)
        assert r["is_substitution_single"] is False
        assert r["edit_count"] == 0
        assert r["edit_positions"] == []
        assert r["exclusion_reason"] == "edit_count_not_one"

    def test_two_substitutions_excluded(self):
        wt = "ACGUGCAC"
        mut = "ACGAGCUC"  # positions 3 (U->A) and 6 (A->U) differ
        r = verify_substitution(wt, mut)
        assert r["is_substitution_single"] is False
        assert r["edit_count"] == 2
        assert r["edit_positions"] == [3, 6]
        assert r["exclusion_reason"] == "edit_count_not_one"

    def test_missing_wt_unverifiable(self):
        r = verify_substitution(None, "ACGU")
        assert r["is_substitution_single"] is False
        assert r["exclusion_reason"] == "substitution_not_verifiable"
        assert r["edit_count"] == 0
        assert r["alignment_cigar"]  # non-empty per schema

    def test_missing_mut_unverifiable(self):
        r = verify_substitution("ACGU", "")
        assert r["is_substitution_single"] is False
        assert r["exclusion_reason"] == "substitution_not_verifiable"

    def test_dna_t_normalized_to_rna_u(self):
        """v3.1 §3.1: verify after DNA->RNA T->U normalization."""
        # WT has T (DNA), mut has U at same position -> should be 0 edits.
        wt_dna = "ACGTGCAC"
        mut_rna = "ACGUGCAC"
        r = verify_substitution(wt_dna, mut_rna)
        assert r["edit_count"] == 0
        assert r["exclusion_reason"] == "edit_count_not_one"

    def test_dna_t_normalized_single_sub(self):
        # Both DNA-encoded; single real substitution at pos 3 (T->A after norm
        # the WT T becomes U, mut keeps A).
        wt_dna = "ACGTGCAC"  # -> ACGUGCAC
        mut_dna = "ACGAGCAC"  # -> ACGAGCAC, pos 3 U->A
        r = verify_substitution(wt_dna, mut_dna)
        assert r["is_substitution_single"] is True
        assert r["edit_positions"] == [3]
        assert r["wt_alleles"] == ["U"]
        assert r["mut_alleles"] == ["A"]

    def test_insertion_excluded(self):
        wt = "ACGU"
        mut = "ACGUA"  # longer -> insertion
        r = verify_substitution(wt, mut)
        assert r["is_substitution_single"] is False
        assert r["edit_type"] == "insertion"
        assert r["exclusion_reason"] == "indel_not_substitution"
        assert r["alignment_cigar"]  # non-empty best-effort

    def test_deletion_excluded(self):
        wt = "ACGUA"
        mut = "ACGU"  # shorter -> deletion
        r = verify_substitution(wt, mut)
        assert r["is_substitution_single"] is False
        assert r["edit_type"] == "deletion"
        assert r["exclusion_reason"] == "indel_not_substitution"

    def test_edit_positions_zero_indexed(self):
        """edit_positions must be 0-indexed to align with mask arrays."""
        wt = "GACGU"
        mut = "GUCGU"  # pos 1 A->U
        r = verify_substitution(wt, mut)
        assert r["edit_positions"] == [1]

    def test_exclusion_reasons_in_vocab(self):
        """All returned exclusion reasons must be in the frozen vocabulary."""
        for wt, mut in [(None, "A"), ("A", "AC"), ("AC", "AC"), ("AC", "AG")]:
            r = verify_substitution(wt, mut)
            if r["exclusion_reason"] is not None:
                assert r["exclusion_reason"] in EXCLUSION_REASONS

    def test_edit_count_consistency(self):
        """edit_count == len(edit_positions) == len(wt_alleles) == len(mut_alleles)."""
        wt = "ACGUGCACGG"
        mut = "AAGUCCACGA"  # 3 diffs at 1,4,9
        r = verify_substitution(wt, mut)
        assert r["edit_count"] == len(r["edit_positions"]) == len(r["wt_alleles"]) == len(r["mut_alleles"])

    def test_single_substitution_at_first_position(self):
        wt = "ACGU"
        mut = "UCGU"
        r = verify_substitution(wt, mut)
        assert r["is_substitution_single"] is True
        assert r["edit_positions"] == [0]

    def test_single_substitution_at_last_position(self):
        wt = "ACGU"
        mut = "ACGA"
        r = verify_substitution(wt, mut)
        assert r["is_substitution_single"] is True
        assert r["edit_positions"] == [3]


class TestVerifyAnnotationRefHIV3PROffset:
    """T-D1.3: HIV3PR genome-numbering offset fix (v3.1 §3.3).

    HIV3PR_DMS_* files (8 files) use HIV genome numbering in their mutation
    annotations (e.g. G8932X) with OFFSET 8931; the header SEQUENCE is the
    261-nt construct. Construct position 1 (index 0) = genome position 8932.
    """

    # Real HIV3PR_DMS_0001 header (first 20 nt) + OFFSET 8931.
    HIV3PR_SEQ_PREFIX = "GGAGUACUUCAAGAACUGCUGACAUCGAGCUUGCUACAAGGGACUUUCCGCUGGGGACUUUCCAGGGAGGCGUGGCCUGGGCGGGACUGGGGAGUGGCGAGCCCUCAGAUGCUGCAUAUAAGCAGCUGCUUUUUGCCUGUACUGGGUCUCUCUGGUUAGACCAGAUCUGAGCCUGGGAGCUCUCUGGCUAACUAGGGAACCCACUGCUUAAGCCUCAAUAAAGCUUGCCUUGAGUGCUUCAAAAGAAACAACAACAACAAC"
    HIV3PR_OFFSET = 8931

    def test_genome_numbering_offset_ref_verified(self):
        """G8932X with OFFSET 8931 -> construct position 1 (index 0) = 'G'."""
        r = verify_annotation_ref(8932, "G", self.HIV3PR_SEQ_PREFIX, self.HIV3PR_OFFSET)
        assert r["ref_verified"] is True
        assert r["ref_match_index"] == "genome_numbering_offset"
        assert r["construct_local_position_1indexed"] == 1
        assert r["actual_base"] == "G"
        assert r["exclusion_reason"] is None

    def test_offset_second_position(self):
        """G8933X -> construct position 2 (index 1); seq[1] = 'G'."""
        r = verify_annotation_ref(8933, "G", self.HIV3PR_SEQ_PREFIX, self.HIV3PR_OFFSET)
        assert r["ref_verified"] is True
        assert r["ref_match_index"] == "genome_numbering_offset"
        assert r["construct_local_position_1indexed"] == 2
        assert r["actual_base"] == "G"

    def test_offset_third_position_a(self):
        """A8934X -> construct position 3 (index 2); seq[2] = 'A'."""
        r = verify_annotation_ref(8934, "A", self.HIV3PR_SEQ_PREFIX, self.HIV3PR_OFFSET)
        assert r["ref_verified"] is True
        assert r["construct_local_position_1indexed"] == 3

    def test_offset_real_ref_mismatch_still_excluded(self):
        """If the encoded ref truly doesn't match, exclude even with offset."""
        # genome 8932 -> construct pos 1 = 'G', but encode ref='A'.
        r = verify_annotation_ref(8932, "A", self.HIV3PR_SEQ_PREFIX, self.HIV3PR_OFFSET)
        assert r["ref_verified"] is False
        assert r["exclusion_reason"] == "annotation_ref_mismatch"
        # actual base at the construct-local index is reported for diagnosis
        assert r["actual_base"] == "G"

    def test_construct_local_no_offset(self):
        """Without offset, annotation positions are construct-local 1-indexed."""
        seq = "ACGUGCAC"
        r = verify_annotation_ref(3, "G", seq, offset=0)
        assert r["ref_verified"] is True
        assert r["ref_match_index"] == "construct_local_1indexed"
        assert r["construct_local_position_1indexed"] == 3
        assert r["actual_base"] == "G"

    def test_construct_local_preferred_over_offset(self):
        """When construct-local matches, it takes precedence (first match wins)."""
        # seq[0]='A', so pos 1 construct-local matches 'A' even with offset set.
        seq = "ACGU"
        r = verify_annotation_ref(1, "A", seq, offset=8931)
        assert r["ref_verified"] is True
        assert r["ref_match_index"] == "construct_local_1indexed"

    def test_no_offset_genome_position_out_of_range(self):
        """Genome-numbered position without offset -> construct-local index OOR."""
        seq = "ACGU"
        r = verify_annotation_ref(8932, "A", seq, offset=0)
        assert r["ref_verified"] is False
        assert r["exclusion_reason"] == "annotation_ref_mismatch"
        # idx_local = 8931 -> out of range, actual_base None
        assert r["actual_base"] is None

    def test_dna_ref_t_normalized_to_u(self):
        """Annotation ref T (DNA) normalized to U before comparison."""
        seq = "ACGU"  # seq[1] = 'C', seq[3] = 'U'
        # Encode ref as 'T' (DNA) at pos 4 where seq has 'U'.
        r = verify_annotation_ref(4, "T", seq, offset=0)
        assert r["ref_verified"] is True
        assert r["actual_base"] == "U"

    def test_missing_sequence_unverifiable(self):
        r = verify_annotation_ref(8932, "G", None, self.HIV3PR_OFFSET)
        assert r["ref_verified"] is False
        assert r["exclusion_reason"] == "annotation_ref_mismatch"

    def test_offset_zero_does_not_subtract(self):
        """offset=0 must not attempt the genome-numbering branch."""
        seq = "ACGU"
        # pos 5 construct-local is OOR; offset 0 -> no genome branch -> mismatch.
        r = verify_annotation_ref(5, "A", seq, offset=0)
        assert r["ref_verified"] is False

    @pytest.mark.parametrize("genome_pos,ref", [
        (8932, "G"), (8933, "G"), (8934, "A"), (8935, "G"),
        (8936, "U"), (8937, "A"), (8938, "C"), (8939, "U"), (8940, "U"),
    ])
    def test_hiv3pr_first_nine_annotations_all_verified(self, genome_pos, ref):
        """All 9 annotations from the real HIV3PR_DMS_0001 file verify after offset."""
        r = verify_annotation_ref(genome_pos, ref, self.HIV3PR_SEQ_PREFIX, self.HIV3PR_OFFSET)
        assert r["ref_verified"] is True
        assert r["ref_match_index"] == "genome_numbering_offset"


class TestPositionMasks:
    """T-D1.4: unchanged/changed position masks (v3 §6.6 step 6, §6.4)."""

    def test_single_edit_middle(self):
        r = build_position_masks([2], 5)
        assert r["unchanged_position_mask"] == [1, 1, 0, 1, 1]
        assert r["changed_position_mask"] == [0, 0, 1, 0, 0]
        assert r["unchanged_position_count"] == 4
        assert r["changed_position_count"] == 1

    def test_no_edits(self):
        r = build_position_masks([], 4)
        assert r["unchanged_position_mask"] == [1, 1, 1, 1]
        assert r["changed_position_mask"] == [0, 0, 0, 0]
        assert r["unchanged_position_count"] == 4
        assert r["changed_position_count"] == 0

    def test_all_edits(self):
        r = build_position_masks([0, 1, 2], 3)
        assert r["unchanged_position_mask"] == [0, 0, 0]
        assert r["changed_position_mask"] == [1, 1, 1]
        assert r["unchanged_position_count"] == 0
        assert r["changed_position_count"] == 3

    def test_multiple_edits(self):
        r = build_position_masks([1, 4], 6)
        assert r["unchanged_position_mask"] == [1, 0, 1, 1, 0, 1]
        assert r["changed_position_mask"] == [0, 1, 0, 0, 1, 0]
        assert r["unchanged_position_count"] == 4
        assert r["changed_position_count"] == 2

    def test_edit_at_boundaries(self):
        r = build_position_masks([0, 4], 5)
        assert r["unchanged_position_mask"] == [0, 1, 1, 1, 0]
        assert r["changed_position_mask"] == [1, 0, 0, 0, 1]

    def test_zero_length(self):
        r = build_position_masks([], 0)
        assert r["unchanged_position_mask"] == []
        assert r["changed_position_mask"] == []
        assert r["unchanged_position_count"] == 0
        assert r["changed_position_count"] == 0

    def test_masks_are_complements(self):
        """changed mask must be the exact complement of unchanged mask."""
        r = build_position_masks([1, 3, 5], 8)
        for u, c in zip(r["unchanged_position_mask"], r["changed_position_mask"]):
            assert u + c == 1

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            build_position_masks([3], 3)

    def test_negative_position_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            build_position_masks([-1], 3)

    def test_bool_position_rejected(self):
        """bool is an int subclass; True/False must not silently become 1/0."""
        with pytest.raises(ValueError, match="plain int"):
            build_position_masks([True], 3)

    def test_non_int_position_rejected(self):
        with pytest.raises(ValueError, match="plain int"):
            build_position_masks([1.0], 3)

    def test_negative_length_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            build_position_masks([], -1)

    def test_duplicate_positions_collapse(self):
        """Duplicate edit positions are idempotent (set semantics)."""
        r = build_position_masks([2, 2], 5)
        assert r["changed_position_mask"] == [0, 0, 1, 0, 0]
        assert r["changed_position_count"] == 1

    def test_from_verify_substitution_output(self):
        """Integration: verify_substitution edit_positions feed build_position_masks."""
        wt = "ACGUGCAC"
        mut = "ACGAGCAC"  # single sub at index 3 (U->A)
        v = verify_substitution(wt, mut)
        assert v["is_substitution_single"] is True
        assert v["edit_positions"] == [3]
        m = build_position_masks(v["edit_positions"], len(wt))
        assert m["unchanged_position_mask"] == [1, 1, 1, 0, 1, 1, 1, 1]
        assert m["changed_position_mask"] == [0, 0, 0, 1, 0, 0, 0, 0]
        assert m["unchanged_position_count"] == 7


class TestComparablePositions:
    """T-D1.4: >=60% comparable-positions check (v3 §6.5, v3.1 §4)."""

    def test_all_valid_no_masks_passes(self):
        """Default (no missingness info) -> all unedited comparable -> passes."""
        unchanged = [1, 1, 0, 1, 1]
        r = check_comparable_positions(unchanged)
        assert r["unchanged_position_count"] == 4
        assert r["comparable_position_count"] == 4
        assert r["comparable_fraction"] == 1.0
        assert r["is_comparable"] is True
        assert r["exclusion_reason"] is None
        assert r["comparable_position_mask"] == [1, 1, 0, 1, 1]

    def test_explicit_all_valid_passes(self):
        unchanged = [1, 1, 0, 1, 1]
        r = check_comparable_positions(unchanged, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
        assert r["is_comparable"] is True
        assert r["comparable_fraction"] == 1.0

    def test_exactly_60pct_passes(self):
        """Boundary: 3/5 = 0.60 -> is_comparable True (>=)."""
        unchanged = [1, 1, 1, 1, 1, 0]  # 5 unedited
        wt = [1, 1, 1, 0, 0, 1]  # 3 of the 5 unedited valid
        mut = [1, 1, 1, 1, 1, 1]
        r = check_comparable_positions(unchanged, wt, mut)
        assert r["comparable_position_count"] == 3
        assert r["unchanged_position_count"] == 5
        assert r["comparable_fraction"] == 0.6
        assert r["is_comparable"] is True
        assert r["exclusion_reason"] is None

    def test_just_below_60pct_fails(self):
        """2/5 = 0.40 < 0.60 -> exclusion."""
        unchanged = [1, 1, 1, 1, 1, 0]  # 5 unedited
        wt = [1, 1, 0, 0, 0, 1]  # 2 of the 5 unedited valid
        mut = [1, 1, 1, 1, 1, 1]
        r = check_comparable_positions(unchanged, wt, mut)
        assert r["comparable_position_count"] == 2
        assert r["comparable_fraction"] == 0.4
        assert r["is_comparable"] is False
        assert r["exclusion_reason"] == "comparable_positions_below_60pct"
        assert r["exclusion_reason"] in EXCLUSION_REASONS

    def test_high_fraction_passes(self):
        """4/5 = 0.80 -> passes."""
        unchanged = [1, 1, 1, 1, 1, 0]
        wt = [1, 1, 1, 1, 0, 1]
        mut = [1, 1, 1, 1, 1, 1]
        r = check_comparable_positions(unchanged, wt, mut)
        assert r["comparable_fraction"] == 0.8
        assert r["is_comparable"] is True

    def test_no_unchanged_positions_fails(self):
        """All edited -> 0 unedited -> fraction 0.0 -> not comparable."""
        unchanged = [0, 0, 0]
        r = check_comparable_positions(unchanged)
        assert r["unchanged_position_count"] == 0
        assert r["comparable_fraction"] == 0.0
        assert r["is_comparable"] is False
        assert r["exclusion_reason"] == "comparable_positions_below_60pct"

    def test_edited_position_never_comparable(self):
        """Even if valid in both profiles, an edited position is not comparable."""
        unchanged = [1, 0, 1, 1, 1]  # 4 unedited, 1 edited at idx 1
        wt = [1, 1, 1, 1, 1]
        mut = [1, 1, 1, 1, 1]
        r = check_comparable_positions(unchanged, wt, mut)
        # idx 1 is edited -> not comparable even though valid
        assert r["comparable_position_mask"] == [1, 0, 1, 1, 1]
        assert r["comparable_position_count"] == 4
        assert r["is_comparable"] is True

    def test_wt_missing_excludes_position(self):
        unchanged = [1, 1, 1, 1, 1]  # 5 unedited
        wt = [1, 1, 1, 0, 0]  # 3 valid -> 3/5 = 0.6 boundary
        mut = [1, 1, 1, 1, 1]
        r = check_comparable_positions(unchanged, wt, mut)
        assert r["comparable_position_count"] == 3
        assert r["comparable_fraction"] == 0.6
        assert r["is_comparable"] is True

    def test_mut_missing_excludes_position(self):
        unchanged = [1, 1, 1, 1, 1]
        wt = [1, 1, 1, 1, 1]
        mut = [1, 1, 0, 0, 0]  # 2 valid -> 2/5 = 0.4
        r = check_comparable_positions(unchanged, wt, mut)
        assert r["comparable_position_count"] == 2
        assert r["is_comparable"] is False

    def test_both_missing_excludes_position(self):
        unchanged = [1, 1, 1, 1]
        wt = [1, 0, 1, 1]
        mut = [1, 1, 0, 1]  # idx 1 wt-missing, idx 2 mut-missing -> 2 comparable
        r = check_comparable_positions(unchanged, wt, mut)
        assert r["comparable_position_count"] == 2
        assert r["comparable_fraction"] == 0.5
        assert r["is_comparable"] is False

    def test_wt_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="wt_valid_mask length"):
            check_comparable_positions([1, 1, 0, 1, 1], [1, 1, 1, 1], None)

    def test_mut_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mut_valid_mask length"):
            check_comparable_positions([1, 1, 0, 1, 1], None, [1, 1, 1])

    def test_invalid_unchanged_value_raises(self):
        with pytest.raises(ValueError, match="unchanged_position_mask entries"):
            check_comparable_positions([1, 2, 0, 1])

    def test_invalid_valid_mask_value_raises(self):
        with pytest.raises(ValueError, match="valid mask entries"):
            check_comparable_positions([1, 1, 0, 1], [1, 5, 1, 1], None)

    def test_custom_min_fraction(self):
        """A higher threshold (0.9) rejects a 0.8 fraction."""
        unchanged = [1, 1, 1, 1, 1, 0]  # 5 unedited
        wt = [1, 1, 1, 1, 0, 1]  # 4 valid -> 4/5 = 0.8
        mut = [1, 1, 1, 1, 1, 1]
        r = check_comparable_positions(unchanged, wt, mut, min_fraction=0.9)
        assert r["comparable_fraction"] == 0.8
        assert r["is_comparable"] is False
        assert r["exclusion_reason"] == "comparable_positions_below_60pct"

    def test_default_min_fraction_constant(self):
        """The default min_fraction equals the frozen COMPARABLE_MIN_FRACTION."""
        assert COMPARABLE_MIN_FRACTION == 0.60

    def test_integration_substitution_masks_comparable(self):
        """End-to-end: verify_substitution -> build_position_masks -> check."""
        wt_seq = "ACGUGCACGT"  # 10 nt
        mut_seq = "ACGAGCACGT"  # sub at index 3 U->A
        v = verify_substitution(wt_seq, mut_seq)
        masks = build_position_masks(v["edit_positions"], len(wt_seq))
        # 9 unedited, all valid (no missingness) -> 9/9 = 1.0
        chk = check_comparable_positions(masks["unchanged_position_mask"])
        assert chk["unchanged_position_count"] == 9
        assert chk["comparable_fraction"] == 1.0
        assert chk["is_comparable"] is True
        # Now simulate missingness: 4 of 9 unedited invalid -> 5/9 ~ 0.556 < 0.6
        wt_valid = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  # idx 4-7 invalid
        mut_valid = [1] * 10
        chk2 = check_comparable_positions(
            masks["unchanged_position_mask"], wt_valid, mut_valid
        )
        assert chk2["comparable_position_count"] == 5
        assert chk2["is_comparable"] is False
        assert chk2["exclusion_reason"] == "comparable_positions_below_60pct"
