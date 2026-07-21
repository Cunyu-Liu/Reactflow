"""Validate the frozen evaluator contract (configs/evaluation/static_v1.yaml).

This test ensures the contract file is well-formed YAML and contains all
required sections.  It is a structural check, not a semantic check — the
gold-fixture tests in test_evaluator_gold_fixtures.py verify the actual
metric computations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ImportError:
    pytest.skip("PyYAML not installed", allow_module_level=True)


CONTRACT_PATH = Path(__file__).resolve().parent.parent / "configs" / "evaluation" / "static_v1.yaml"

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_name",
    "frozen_date",
    "pair_types",
    "constraints",
    "indexing",
    "scoring",
    "threshold",
    "aggregation",
    "distance_bins",
    "empty_structure",
    "decoder",
    "pipeline",
    "test_data",
    "gate",
}

REQUIRED_PAIR_TYPES = {"canonical", "wobble", "illegal"}
REQUIRED_SCORING_KEYS = {"confusion", "f1", "mcc", "precision", "recall", "shifted_f1"}
REQUIRED_DISTANCE_BIN_NAMES = {"short", "medium", "long"}
REQUIRED_AGGREGATION_KEYS = {"primary", "secondary", "micro_f1", "macro_f1", "per_tier"}


@pytest.fixture
def contract():
    """Load and parse the contract YAML."""
    if not CONTRACT_PATH.exists():
        pytest.skip(f"Contract file not found at {CONTRACT_PATH}")
    with CONTRACT_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_contract_file_exists():
    """The contract file must exist at the expected path."""
    assert CONTRACT_PATH.exists(), f"Contract file missing: {CONTRACT_PATH}"


def test_contract_has_all_top_level_keys(contract):
    """All required top-level keys must be present."""
    missing = REQUIRED_TOP_LEVEL_KEYS - set(contract.keys())
    assert not missing, f"Missing top-level keys: {missing}"


def test_contract_schema_version(contract):
    """Schema version must be '1.0'."""
    assert contract["schema_version"] == "1.0"


def test_contract_name(contract):
    """Contract name must be 'static_v1'."""
    assert contract["contract_name"] == "static_v1"


def test_pair_types_complete(contract):
    """Pair types must include canonical, wobble, and illegal definitions."""
    pair_types = contract["pair_types"]
    missing = REQUIRED_PAIR_TYPES - set(pair_types.keys())
    assert not missing, f"Missing pair types: {missing}"

    canonical_pairs = pair_types["canonical"]["pairs"]
    assert len(canonical_pairs) == 4, "Canonical pairs must have 4 entries (AU, UA, GC, CG)"

    wobble_pairs = pair_types["wobble"]["pairs"]
    assert len(wobble_pairs) == 2, "Wobble pairs must have 2 entries (GU, UG)"


def test_constraints(contract):
    """Constraints must specify allow_wobble, allow_pseudoknot, and min_loop."""
    constraints = contract["constraints"]
    assert constraints["allow_wobble"] is True
    assert constraints["allow_pseudoknot"] is False
    assert constraints["min_loop"] == 3


def test_indexing_zero_based(contract):
    """Indexing must be 0-based with upper-triangle storage."""
    indexing = contract["indexing"]
    assert indexing["base"] == 0
    assert indexing["storage"] == "upper_triangle"
    assert indexing["self_pair_rejection"] is True
    assert indexing["out_of_range_rejection"] is True


def test_scoring_formulas(contract):
    """Scoring must define confusion, f1, mcc, precision, recall, shifted_f1."""
    scoring = contract["scoring"]
    missing = REQUIRED_SCORING_KEYS - set(scoring.keys())
    assert not missing, f"Missing scoring keys: {missing}"

    assert "2*TP" in scoring["f1"]["formula"]
    assert scoring["f1"]["undefined_value"] == 0.0
    assert scoring["shifted_f1"]["tolerance"] == 1


def test_threshold(contract):
    """Threshold must specify matrix_cell (0.5) and decoder_min_score (0.0)."""
    threshold = contract["threshold"]
    assert threshold["matrix_cell"] == 0.5
    assert threshold["decoder_min_score"] == 0.0


def test_aggregation(contract):
    """Aggregation must specify micro_f1 as primary and macro_f1 as secondary."""
    aggregation = contract["aggregation"]
    assert aggregation["primary"] == "micro_f1"
    assert aggregation["secondary"] == "macro_f1"
    missing = REQUIRED_AGGREGATION_KEYS - set(aggregation.keys())
    assert not missing, f"Missing aggregation keys: {missing}"


def test_distance_bins(contract):
    """Distance bins must be short (1-11), medium (12-23), long (24+)."""
    bins = contract["distance_bins"]
    names = {b["name"] for b in bins}
    missing = REQUIRED_DISTANCE_BIN_NAMES - names
    assert not missing, f"Missing distance bins: {missing}"

    for b in bins:
        if b["name"] == "short":
            assert b["min_distance"] == 1
            assert b["max_distance"] == 11
        elif b["name"] == "medium":
            assert b["min_distance"] == 12
            assert b["max_distance"] == 23
        elif b["name"] == "long":
            assert b["min_distance"] == 24
            assert b["max_distance"] is None


def test_empty_structure_convention(contract):
    """Empty-structure convention must adopt ReactFlow (F1=0.0 for empty-vs-empty)."""
    empty = contract["empty_structure"]
    assert empty["reactflow_convention"]["empty_vs_empty"] == 0.0
    assert empty["efold_convention"]["empty_vs_empty"] == 1.0
    assert empty["contract_adopts"] == "reactflow_convention"


def test_decoder_defaults(contract):
    """Decoder must default to calibrated_marginal with nested_dp."""
    decoder = contract["decoder"]
    assert decoder["default_mode"] == "calibrated_marginal"
    assert decoder["default_policy"] == "nested_dp"
    assert decoder["min_loop"] == 3


def test_test_data_split_preference(contract):
    """Test data must prefer mmseqs split and exclude human_mRNA."""
    test_data = contract["test_data"]
    assert test_data["split_preference"] == "mmseqs"
    assert "human_mRNA" in test_data["excluded_tiers"]


def test_gate_criteria(contract):
    """Gate must list all 4 criteria."""
    gate = contract["gate"]
    for i in range(1, 5):
        key = f"criterion_{i}"
        assert key in gate, f"Missing {key}"
        assert isinstance(gate[key], str) and len(gate[key]) > 10
