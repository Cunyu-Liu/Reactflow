"""Tests for the frozen D1 construct/pair schemas (T-D1.1, v3 §6.3/§6.4)."""

from __future__ import annotations

from copy import deepcopy
import pytest

from reactflow.delta.schema import (
    CONSTRUCT_SCHEMA_VERSION,
    PAIR_SCHEMA_VERSION,
    EDIT_TYPES,
    CONDITION_MATCH_STATUSES,
    IN_VIVO_IN_VITRO_STATUSES,
    EXCLUSION_REASONS,
    REQUIRED_CONSTRUCT_FIELDS,
    REQUIRED_PAIR_FIELDS,
    NULLABLE_CONSTRUCT_FIELDS,
    NULLABLE_PAIR_FIELDS,
    ConstructValidationError,
    PairValidationError,
    construct_json_schema,
    pair_json_schema,
    validate_construct_record,
    validate_pair_record,
)


# --- fixture builders ---

def _valid_construct() -> dict:
    """Build a minimal construct record that passes validation."""
    return {
        "schema_version": CONSTRUCT_SCHEMA_VERSION,
        "construct_id": "M2SL5_DMS_0000:0",
        "source_entry_id": "rmdb:M2SL5_DMS_0000:abc123",
        "study_id": "diegelhalter_2024",
        "publication_id": None,
        "laboratory_id": None,
        "parent_id": "M2SL5_DMS",
        "design_lineage_id": None,
        "sequence_raw": "ACGTACGT",
        "sequence_normalized": "ACGUACGU",
        "length": 8,
        "probe": "DMS",
        "probe_protocol": None,
        "temperature": None,
        "ligand": None,
        "ligand_concentration": None,
        "buffer": None,
        "in_vivo_in_vitro": "in_vitro",
        "batch_id": None,
        "replicate_id": None,
        "reactivity_raw": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "reactivity_upstream": None,
        "reactivity_error": None,
        "coverage": None,
        "snr": None,
        "valid_mask": [1, 1, 1, 1, 1, 1, 1, 1],
        "probe_eligibility_mask": [1, 1, 0, 0, 1, 1, 0, 0],
        "normalization_method": None,
        "quality_flags": {},
        "missing_reasons": {
            "publication_id": "not asserted by RMDB metadata",
            "laboratory_id": "not asserted by RMDB metadata",
            "design_lineage_id": "single-parent construct, no lineage",
            "probe_protocol": "not encoded in RDAT annotation",
            "temperature": "not encoded in RDAT annotation",
            "ligand": "apo condition",
            "ligand_concentration": "apo condition",
            "buffer": "not encoded in RDAT annotation",
            "batch_id": "single batch in file",
            "replicate_id": "no replicate identified",
            "reactivity_upstream": "upstream normalization not yet computed",
            "reactivity_error": "error bar not provided by RDAT",
            "coverage": "per-position coverage not provided",
            "snr": "SNR not computed",
            "normalization_method": "not yet frozen",
        },
    }


def _valid_pair() -> dict:
    """Build a minimal pair record that passes validation (eligible substitution)."""
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "pair_id": "M2SL5_DMS_0000:0:5",
        "wt_construct_id": "M2SL5_DMS_0000:0",
        "mut_construct_id": "M2SL5_DMS_0000:5",
        "parent_id": "M2SL5_DMS",
        "study_id": "diegelhalter_2024",
        "design_lineage_id": None,
        "edit_type": "substitution",
        "edit_positions": [4],
        "wt_alleles": ["A"],
        "mut_alleles": ["G"],
        "edit_count": 1,
        "alignment_cigar": "8M",
        "condition_match_fields": ["probe", "temperature", "ligand", "buffer"],
        "condition_match_status": "exact_match",
        "delta_reactivity_raw": [0.0, 0.1, -0.2, 0.5, -0.3, 0.0, 0.1, -0.1],
        "delta_reactivity_normalized": None,
        "unchanged_position_mask": [1, 1, 1, 1, 0, 1, 1, 1],
        "changed_position_mask": [0, 0, 0, 0, 1, 0, 0, 0],
        "probe_eligibility_unchanged_mask": [1, 1, 0, 0, 0, 1, 0, 0],
        "local_mask": [0, 0, 1, 1, 1, 0, 0, 0],
        "mid_mask": [1, 1, 0, 0, 0, 0, 1, 1],
        "remote_mask": [0, 0, 0, 0, 0, 1, 0, 0],
        "replicate_noise_estimate": None,
        "measurement_variance": None,
        "pair_quality_weight": 1.0,
        "primary_eligible": True,
        "exclusion_reasons": [],
        "missing_reasons": {
            "design_lineage_id": "single-parent pair",
            "delta_reactivity_normalized": "normalization not yet frozen",
            "replicate_noise_estimate": "no replicate available",
            "measurement_variance": "no replicate available",
        },
    }


# --- construct schema tests ---

class TestConstructSchema:
    def test_valid_construct_passes(self):
        rec = _valid_construct()
        out = validate_construct_record(rec)
        assert out["construct_id"] == rec["construct_id"]

    def test_missing_required_field(self):
        rec = _valid_construct()
        del rec["construct_id"]
        with pytest.raises(ConstructValidationError, match="missing required fields"):
            validate_construct_record(rec)

    def test_unexpected_field(self):
        rec = _valid_construct()
        rec["extra_field"] = "bad"
        with pytest.raises(ConstructValidationError, match="unexpected fields"):
            validate_construct_record(rec)

    def test_wrong_schema_version(self):
        rec = _valid_construct()
        rec["schema_version"] = "wrong-version"
        with pytest.raises(ConstructValidationError, match="schema_version"):
            validate_construct_record(rec)

    def test_T_in_sequence_normalized_rejected(self):
        rec = _valid_construct()
        rec["sequence_normalized"] = "ACGTACGT"
        with pytest.raises(ConstructValidationError, match="T.*U normalization"):
            validate_construct_record(rec)

    def test_length_mismatch_rejected(self):
        rec = _valid_construct()
        rec["length"] = 99
        with pytest.raises(ConstructValidationError, match="length.*sequence_normalized"):
            validate_construct_record(rec)

    def test_mask_length_mismatch_rejected(self):
        rec = _valid_construct()
        rec["valid_mask"] = [1, 1, 1]
        with pytest.raises(ConstructValidationError, match="valid_mask length"):
            validate_construct_record(rec)

    def test_mask_invalid_value_rejected(self):
        rec = _valid_construct()
        rec["valid_mask"] = [1, 1, 1, 1, 2, 1, 1, 1]
        with pytest.raises(ConstructValidationError, match="0/1"):
            validate_construct_record(rec)

    def test_invalid_in_vivo_in_vitro(self):
        rec = _valid_construct()
        rec["in_vivo_in_vitro"] = "ex_vivo"
        with pytest.raises(ConstructValidationError, match="in_vivo_in_vitro"):
            validate_construct_record(rec)

    def test_reactivity_array_length_mismatch(self):
        rec = _valid_construct()
        rec["reactivity_raw"] = [0.1, 0.2]
        with pytest.raises(ConstructValidationError, match="reactivity_raw length"):
            validate_construct_record(rec)

    def test_null_without_missing_reason(self):
        rec = _valid_construct()
        rec["snr"] = None
        # snr is already None in fixture and has a missing_reason; remove the reason
        rec["missing_reasons"].pop("snr")
        with pytest.raises(ConstructValidationError, match="missing_reasons keys"):
            validate_construct_record(rec)

    def test_missing_reason_for_non_null_field(self):
        rec = _valid_construct()
        # 'probe' is non-null and non-nullable; adding it to missing_reasons is invalid
        rec["missing_reasons"]["probe"] = "should not be here"
        with pytest.raises(ConstructValidationError, match="missing_reasons keys"):
            validate_construct_record(rec)

    def test_quality_flags_must_be_dict(self):
        rec = _valid_construct()
        rec["quality_flags"] = ["flag1"]
        with pytest.raises(ConstructValidationError, match="quality_flags"):
            validate_construct_record(rec)

    def test_non_mapping_rejected(self):
        with pytest.raises(ConstructValidationError, match="must be a mapping"):
            validate_construct_record("not a dict")

    def test_all_nullable_fields_can_be_null(self):
        """Every nullable field can be null if it has a missing_reason."""
        rec = _valid_construct()
        for f in NULLABLE_CONSTRUCT_FIELDS:
            rec[f] = None
        rec["missing_reasons"] = {f: "test" for f in NULLABLE_CONSTRUCT_FIELDS}
        validate_construct_record(rec)


# --- pair schema tests ---

class TestPairSchema:
    def test_valid_pair_passes(self):
        rec = _valid_pair()
        out = validate_pair_record(rec)
        assert out["pair_id"] == rec["pair_id"]

    def test_missing_required_field(self):
        rec = _valid_pair()
        del rec["pair_id"]
        with pytest.raises(PairValidationError, match="missing required fields"):
            validate_pair_record(rec)

    def test_unexpected_field(self):
        rec = _valid_pair()
        rec["extra"] = "bad"
        with pytest.raises(PairValidationError, match="unexpected fields"):
            validate_pair_record(rec)

    def test_edit_count_mismatch(self):
        rec = _valid_pair()
        rec["edit_count"] = 2
        with pytest.raises(PairValidationError, match="edit_count"):
            validate_pair_record(rec)

    def test_allele_length_mismatch(self):
        rec = _valid_pair()
        rec["wt_alleles"] = ["A", "G"]
        with pytest.raises(PairValidationError, match="edit_count"):
            validate_pair_record(rec)

    def test_substitution_wt_eq_mut_rejected(self):
        rec = _valid_pair()
        rec["wt_alleles"] = ["A"]
        rec["mut_alleles"] = ["A"]
        with pytest.raises(PairValidationError, match="wt_allele != mut_allele"):
            validate_pair_record(rec)

    def test_invalid_exclusion_reason(self):
        rec = _valid_pair()
        rec["primary_eligible"] = False
        rec["exclusion_reasons"] = ["not_a_real_reason"]
        with pytest.raises(PairValidationError, match="invalid reason"):
            validate_pair_record(rec)

    def test_primary_ineligible_without_reasons(self):
        rec = _valid_pair()
        rec["primary_eligible"] = False
        rec["exclusion_reasons"] = []
        with pytest.raises(PairValidationError, match="primary_eligible=False"):
            validate_pair_record(rec)

    def test_indel_without_exclusion(self):
        rec = _valid_pair()
        rec["edit_type"] = "insertion"
        with pytest.raises(PairValidationError, match="indel_not_substitution"):
            validate_pair_record(rec)

    def test_indel_with_exclusion_ok(self):
        rec = _valid_pair()
        rec["edit_type"] = "insertion"
        rec["edit_positions"] = [4]
        rec["wt_alleles"] = ["A"]
        rec["mut_alleles"] = ["AG"]
        rec["primary_eligible"] = False
        rec["exclusion_reasons"] = ["indel_not_substitution"]
        validate_pair_record(rec)

    def test_condition_mismatch_without_reason(self):
        rec = _valid_pair()
        rec["condition_match_status"] = "mismatch"
        with pytest.raises(PairValidationError, match="condition_mismatch"):
            validate_pair_record(rec)

    def test_condition_mismatch_with_reason_ok(self):
        rec = _valid_pair()
        rec["condition_match_status"] = "mismatch"
        rec["primary_eligible"] = False
        rec["exclusion_reasons"] = ["condition_mismatch"]
        validate_pair_record(rec)

    def test_invalid_edit_type(self):
        rec = _valid_pair()
        rec["edit_type"] = "translocation"
        with pytest.raises(PairValidationError, match="edit_type"):
            validate_pair_record(rec)

    def test_null_without_missing_reason(self):
        rec = _valid_pair()
        rec["missing_reasons"].pop("design_lineage_id")
        with pytest.raises(PairValidationError, match="missing_reasons keys"):
            validate_pair_record(rec)

    def test_non_mapping_rejected(self):
        with pytest.raises(PairValidationError, match="must be a mapping"):
            validate_pair_record(42)

    def test_mask_invalid_value(self):
        rec = _valid_pair()
        rec["local_mask"] = [0, 0, 1, 1, 5, 0, 0, 0]
        with pytest.raises(PairValidationError, match="0/1"):
            validate_pair_record(rec)


# --- frozen vocabulary tests ---

class TestSchemaVocabulary:
    def test_exclusion_reasons_contains_required_set(self):
        # v3.1 §4 D1 Gate required exclusion reasons
        required = {
            "annotation_only_alt_not_verifiable",
            "sequence_based_no_independent_corroboration",
            "substitution_not_verifiable",
            "annotation_ref_mismatch",
            "condition_mismatch",
            "probe_mismatch",
            "comparable_positions_below_60pct",
            "edit_count_not_one",
            "indel_not_substitution",
            "no_wt_anchor",
            "normalization_domain_unknown",
            "parent_lineage_unverified",
            "in_vivo_in_vitro_mixed",
        }
        assert required <= EXCLUSION_REASONS

    def test_edit_types_frozen(self):
        assert EDIT_TYPES == frozenset({"substitution", "insertion", "deletion"})

    def test_condition_statuses_frozen(self):
        assert CONDITION_MATCH_STATUSES == frozenset({"exact_match", "mismatch"})

    def test_in_vivo_in_vitro_frozen(self):
        assert IN_VIVO_IN_VITRO_STATUSES == frozenset({"in_vivo", "in_vitro"})

    def test_construct_has_28_contract_fields(self):
        # v3 §6.3 lists 28 fields + schema_version + missing_reasons = 30
        contract_fields = REQUIRED_CONSTRUCT_FIELDS - {"schema_version", "missing_reasons"}
        assert len(contract_fields) == 28

    def test_pair_has_27_contract_fields(self):
        # v3 §6.4 lists 27 fields + schema_version + missing_reasons = 29
        contract_fields = REQUIRED_PAIR_FIELDS - {"schema_version", "missing_reasons"}
        assert len(contract_fields) == 27


# --- JSON schema generator tests ---

class TestJsonSchemaGenerators:
    def test_construct_json_schema_id(self):
        js = construct_json_schema()
        assert js["$id"] == CONSTRUCT_SCHEMA_VERSION
        assert js["type"] == "object"
        assert js["additionalProperties"] is False

    def test_pair_json_schema_id(self):
        js = pair_json_schema()
        assert js["$id"] == PAIR_SCHEMA_VERSION
        assert js["type"] == "object"
        assert js["additionalProperties"] is False

    def test_construct_json_schema_required_matches(self):
        js = construct_json_schema()
        assert set(js["required"]) == set(REQUIRED_CONSTRUCT_FIELDS)

    def test_pair_json_schema_required_matches(self):
        js = pair_json_schema()
        assert set(js["required"]) == set(REQUIRED_PAIR_FIELDS)
