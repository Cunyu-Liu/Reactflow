"""D1 pipeline executor smoke tests (T-D1.13).

Tests the pure helper functions in scripts/reactflow_delta/d1_pipeline_executor.py
with small synthetic inputs — no real RDAT files, no network. Verifies the
wiring of T-D1.1~10 building blocks into ``_evaluate_one`` produces the
expected exclusion-reason vector and Tier-judgment basis.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "reactflow_delta"
        / "d1_pipeline_executor.py"
    )
    spec = importlib.util.spec_from_file_location("d1_pipeline_executor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# ---------------------------------------------------------------------------
# _comparable_stats
# ---------------------------------------------------------------------------

def test_comparable_stats_all_present():
    s = mod._comparable_stats([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert s == {
        "aligned_length": 3,
        "both_nonmissing": 3,
        "comparable_fraction": 1.0,
        "missing_fraction": 0.0,
    }


def test_comparable_stats_with_missing():
    # Position 1 missing in WT, position 2 missing in mut → 1/3 both present.
    s = mod._comparable_stats([0.1, None, 0.3], [0.4, 0.5, None])
    assert s["aligned_length"] == 3
    assert s["both_nonmissing"] == 1
    assert s["comparable_fraction"] == pytest.approx(1 / 3)
    assert s["missing_fraction"] == pytest.approx(2 / 3)


def test_comparable_stats_empty():
    s = mod._comparable_stats([], [0.4])
    assert s["aligned_length"] == 0
    assert s["comparable_fraction"] is None
    assert s["missing_fraction"] is None


# ---------------------------------------------------------------------------
# _evaluate_one — annotation-only candidate with unverified lineage
# (mirrors the D0-R v2 pool: parent_lineage_unverified for all)
# ---------------------------------------------------------------------------

def _make_profile(reactivity, reactivity_error=None):
    return {
        "index": 1,
        "reactivity": reactivity,
        "reactivity_error": reactivity_error,
        "missing_reactivity_count": sum(v is None for v in reactivity),
        "profile_sequence": None,
        "profile_name": None,
    }


def _make_relation(*, alt_not_verified=True):
    return {
        "rdat_path": "/synthetic/test.rdat",
        "rmdb_id": "TEST",
        "owner": "Test Owner",
        "parent_prefix": "TEST-parent",
        "citation_doi": "10.0000/test",
        "modifier": "DMS",
        "wt_profile_index": 1,
        "mutant_profile_index": 2,
        "audit_method": "annotation_only_mutation_ref_verified_against_header",
        "annotation_mutation_count": 1,
        "lineage_status": "candidate_only_pending_parent_lineage_and_functional_region_validation",
        "matched_mutation": {
            "alt_not_verified": alt_not_verified,
            "encoded_alt": "X",
            "encoded_position_1indexed": 1,
            "encoded_ref": "G",
            "encoding_source": "annotation",
        },
    }


def test_evaluate_one_annotation_only_unverified_lineage_blocks_upgrade():
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation(alt_not_verified=True)
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None)

    # Annotation-only + alt not verified + lineage pending → two reasons.
    assert res["exclusion_reasons"] == [
        "annotation_only_alt_not_verifiable",
        "parent_lineage_unverified",
    ]
    assert res["primary_eligible"] is False
    assert res["true_pair"] is False
    assert res["is_annotation_only"] is True
    assert res["is_sequence_based"] is False
    assert res["substitution_verified"] is False
    assert res["parent_lineage_verified"] is False
    assert res["has_wt_anchor"] is True
    assert res["condition_match_status"] == "match"
    assert res["caller_status"] == "no_replicate_continuous_only"
    assert res["comparable_fraction"] == 1.0
    assert res["missing_fraction"] == 0.0
    assert res["aligned_length"] == 4
    # quality weight is a clamped product in [0, 1]
    assert 0.0 <= res["pair_quality_weight"] <= 1.0
    assert set(res["quality_factors"]) == {
        "comparable", "snr", "coverage", "replicate", "missing"
    }


def test_evaluate_one_verified_substitution_still_blocked_by_lineage():
    # alt_not_verified=False → substitution_verified=True, but lineage still
    # pending → only parent_lineage_unverified remains (mirrors the 285 cases).
    wt = _make_profile([0.1, 0.2, 0.3])
    mut = _make_profile([0.5, 0.6, 0.7])
    rel = _make_relation(alt_not_verified=False)
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None)

    assert res["exclusion_reasons"] == ["parent_lineage_unverified"]
    assert res["substitution_verified"] is True
    assert res["primary_eligible"] is False
    assert res["true_pair"] is False


def test_evaluate_one_missing_profile_records_no_wt_anchor():
    rel = _make_relation()
    res = mod._evaluate_one(rel, {}, None)  # empty profile map
    assert res["has_wt_anchor"] is False
    assert res["profile_lookup_ok"] is False
    assert res["caller_status"] == "no_profile"
    # no_wt_anchor reason added on top of lineage + annotation-only reasons
    assert "no_wt_anchor" in res["exclusion_reasons"]
    assert "parent_lineage_unverified" in res["exclusion_reasons"]
    assert res["true_pair"] is False


def test_evaluate_one_low_comparable_fraction_adds_reason():
    # Only 1 of 4 positions present in both → 0.25 < 0.60 threshold.
    wt = _make_profile([0.1, None, None, None])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation(alt_not_verified=True)
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None)

    assert res["comparable_fraction"] == 0.25
    assert "comparable_positions_below_60pct" in res["exclusion_reasons"]
    assert "annotation_only_alt_not_verifiable" in res["exclusion_reasons"]
    assert "parent_lineage_unverified" in res["exclusion_reasons"]


# ---------------------------------------------------------------------------
# _evaluate_one — v3.1 §必须输出: three-layer reactivity + Δreactivity
# arrays + noise estimate fields persisted in the registry entry
# ---------------------------------------------------------------------------

# Fields required by v3.1 §必须输出 / pair schema (schema.py REQUIRED_PAIR_FIELDS).
_THREE_LAYER_FIELDS = (
    "wt_reactivity_raw",
    "wt_reactivity_upstream",
    "wt_reactivity_project",
    "wt_normalization_method",
    "mut_reactivity_raw",
    "mut_reactivity_upstream",
    "mut_reactivity_project",
    "mut_normalization_method",
)
_DELTA_ARRAY_FIELDS = ("delta_reactivity_raw", "delta_reactivity_normalized")
_NOISE_FIELDS = (
    "replicate_noise_estimate",
    "measurement_variance",
    "noise_wt_variance",
    "noise_mut_variance",
    "noise_source",
)


def test_evaluate_one_persists_three_layer_reactivity_when_profile_ok():
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation(alt_not_verified=True)
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None)

    # All three-layer keys present and non-null (profile lookup succeeded).
    for field in _THREE_LAYER_FIELDS:
        assert field in res, f"missing field: {field}"
        assert res[field] is not None, f"field should be non-null: {field}"

    # WT three-layer values match the input reactivity (raw == upstream for
    # normalization_method=None; project == z-score of upstream).
    assert res["wt_reactivity_raw"] == [0.1, 0.2, 0.3, 0.4]
    assert res["wt_reactivity_upstream"] == [0.1, 0.2, 0.3, 0.4]
    assert res["mut_reactivity_raw"] == [0.5, 0.6, 0.7, 0.8]
    assert res["wt_normalization_method"] == "unknown"
    assert res["mut_normalization_method"] == "unknown"
    # project layer length matches raw length (z-score applied per-element).
    assert len(res["wt_reactivity_project"]) == 4
    assert len(res["mut_reactivity_project"]) == 4


def test_evaluate_one_persists_delta_reactivity_arrays_when_profile_ok():
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation(alt_not_verified=True)
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None)

    for field in _DELTA_ARRAY_FIELDS:
        assert field in res, f"missing field: {field}"
        assert res[field] is not None, f"field should be non-null: {field}"
    # Δr = r_mut − r_wt = 0.4 at every position (float-approx: 0.6−0.2 etc).
    assert res["delta_reactivity_raw"] == pytest.approx([0.4, 0.4, 0.4, 0.4])
    assert len(res["delta_reactivity_normalized"]) == 4


def test_evaluate_one_persists_noise_fields_when_profile_ok():
    # reactivity_error provided → upstream_error path (v3 §7.3).
    wt = _make_profile([0.1, 0.2, 0.3, 0.4], reactivity_error=[0.01, 0.01, 0.01, 0.01])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8], reactivity_error=[0.02, 0.02, 0.02, 0.02])
    rel = _make_relation(alt_not_verified=True)
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None)

    for field in _NOISE_FIELDS:
        assert field in res, f"missing field: {field}"
    # No replicates → replicate_noise_estimate is None, but measurement_variance
    # comes from upstream REACTIVITY_ERROR (wt_var + mut_var > 0).
    assert res["replicate_noise_estimate"] is None
    assert res["measurement_variance"] is not None
    assert res["measurement_variance"] > 0
    assert res["noise_source"] == "upstream_error"
    assert res["noise_wt_variance"] is not None
    assert res["noise_mut_variance"] is not None


def test_evaluate_one_noise_fields_null_when_no_error_and_no_replicate():
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])  # no reactivity_error
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation(alt_not_verified=True)
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None)

    # No replicates and no upstream error → all noise fields null, source "none".
    assert res["replicate_noise_estimate"] is None
    assert res["measurement_variance"] is None
    assert res["noise_wt_variance"] is None
    assert res["noise_mut_variance"] is None
    assert res["noise_source"] == "none"
    # But three-layer + delta arrays still populated (profile lookup ok).
    assert res["wt_reactivity_raw"] is not None
    assert res["delta_reactivity_raw"] is not None


def test_evaluate_one_all_new_fields_null_when_profile_lookup_fails():
    # No profile map → profile_lookup_ok=False → all new fields null.
    rel = _make_relation()
    res = mod._evaluate_one(rel, {}, None)

    for field in _THREE_LAYER_FIELDS + _DELTA_ARRAY_FIELDS:
        assert res[field] is None, f"field should be null on no-profile: {field}"
    assert res["replicate_noise_estimate"] is None
    assert res["measurement_variance"] is None
    assert res["noise_source"] is None
    assert res["wt_normalization_method"] is None
    assert res["mut_normalization_method"] is None


# ---------------------------------------------------------------------------
# _tier_judgment — true_pair=0 fails all tiers; thresholds not lowered
# ---------------------------------------------------------------------------

def test_tier_judgment_zero_true_pairs_fails_all_tiers():
    registry = [
        {"true_pair": False, "citation_doi": "d1", "parent_prefix": "p1", "owner": "o1"},
        {"true_pair": False, "citation_doi": "d2", "parent_prefix": "p2", "owner": "o2"},
    ]
    tj = mod._tier_judgment(registry)
    assert tj["tier_a"]["pass"] is False
    assert tj["tier_b"]["pass"] is False
    assert tj["tier_c"]["pass"] is False
    assert tj["tier_b"]["true_pairs"] == 0
    assert tj["tier_b"]["threshold"] == mod.TIER_B_MIN_TRUE_PAIRS == 1000
    # candidate-level reference still reports the raw candidate counts
    assert tj["candidate_level_reference"]["candidate_total"] == 2


def test_tier_judgment_thresholds_not_lowered():
    # The Tier B threshold must remain 1000 (v3.1 §7: no lowering).
    assert mod.TIER_A_MIN_STUDIES == 5
    assert mod.TIER_A_MIN_PARENTS == 20
    assert mod.TIER_A_MIN_PAIRS == 5000
    assert mod.TIER_B_MIN_TRUE_PAIRS == 1000
