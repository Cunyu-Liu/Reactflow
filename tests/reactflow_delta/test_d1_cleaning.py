"""Tests for D1 cleaning pipeline functions (T-D1.2+)."""

from __future__ import annotations

import pytest

from reactflow.delta.data import (
    CONDITION_MATCH_FIELDS,
    match_conditions,
)


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
