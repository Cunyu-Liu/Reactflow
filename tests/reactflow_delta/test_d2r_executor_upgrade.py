"""D2-R executor upgrade tests (v3.1 §3.2: same-parent replicate corroboration).

Verifies that ``_evaluate_one`` correctly wires the D2-R evidence manifest
into ``evaluate_pair_upgrade`` so that annotation-only candidates with
same-parent replicate corroboration are upgraded (§3.2: replicate substitutes
for alt resolution), while:

  - candidates with no evidence keep the annotation_only_alt_not_verifiable
    blocker (§3.2 default),
  - quality gates (comparable fraction) still apply even when corroborated
    (§3.2 suppresses ONLY the annotation-only blocker, not other reasons),
  - substitution_verified stays False (replicate substitutes; alt is not
    resolved to a concrete base),
  - evidence_source is recorded for every candidate.
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


def _make_profile(reactivity, reactivity_error=None):
    return {
        "index": 1,
        "reactivity": reactivity,
        "reactivity_error": reactivity_error,
        "missing_reactivity_count": sum(v is None for v in reactivity),
        "profile_sequence": None,
        "profile_name": None,
    }


def _make_relation(*, alt_not_verified=True, rdat_sha256="sha-test-1"):
    return {
        "rdat_path": "/synthetic/test.rdat",
        "rmdb_id": "TEST",
        "owner": "Test Owner",
        "parent_prefix": "TEST-parent",
        "citation_doi": "10.0000/test",
        "modifier": "DMS",
        "rdat_sha256": rdat_sha256,
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


def _evidence_lookup_found():
    return {
        ("sha-test-1", 1, 2): {
            "status": "evidence_found",
            "evidence_type": "same_parent_replicate",
        }
    }


def _evidence_lookup_not_found():
    return {
        ("sha-test-1", 1, 2): {
            "status": "no_evidence_in_scope",
            "evidence_type": "none",
        }
    }


# ---------------------------------------------------------------------------
# §3.2: replicate corroboration clears annotation_only_alt_not_verifiable
# ---------------------------------------------------------------------------

def test_replicate_corroboration_clears_annotation_only_blocker():
    # annotation-only + D2 lineage verified + replicate corroboration →
    # no annotation_only_alt_not_verifiable, true_pair=True.
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation()
    lineage_lookup = {("sha-test-1", 1, 2): True}
    res = mod._evaluate_one(
        rel, {1: wt, 2: mut}, None, lineage_lookup, _evidence_lookup_found()
    )

    assert "annotation_only_alt_not_verifiable" not in res["exclusion_reasons"]
    assert res["exclusion_reasons"] == []
    assert res["primary_eligible"] is True
    assert res["true_pair"] is True
    assert res["has_replicate_corroboration"] is True
    assert res["evidence_source"] == "d2r_same_parent_replicate"


def test_replicate_corroboration_does_not_set_substitution_verified():
    # §3.2: replicate substitutes for alt resolution; the alt is NOT resolved
    # to a concrete base. substitution_verified stays False.
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation()
    lineage_lookup = {("sha-test-1", 1, 2): True}
    res = mod._evaluate_one(
        rel, {1: wt, 2: mut}, None, lineage_lookup, _evidence_lookup_found()
    )

    assert res["substitution_verified"] is False
    assert res["is_annotation_only"] is True
    assert res["true_pair"] is True  # upgraded via replicate path, not seq


# ---------------------------------------------------------------------------
# §3.2 default: no evidence → annotation_only_alt_not_verifiable remains
# ---------------------------------------------------------------------------

def test_no_evidence_keeps_annotation_only_blocker():
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation()
    lineage_lookup = {("sha-test-1", 1, 2): True}
    res = mod._evaluate_one(
        rel, {1: wt, 2: mut}, None, lineage_lookup, _evidence_lookup_not_found()
    )

    assert "annotation_only_alt_not_verifiable" in res["exclusion_reasons"]
    assert res["true_pair"] is False
    assert res["has_replicate_corroboration"] is False
    assert res["evidence_source"] == "d2r_manifest_no_evidence"


def test_no_evidence_lookup_preserves_default_behavior():
    # evidence_lookup=None → pre-D2-R behavior: annotation-only blocker present.
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation()
    lineage_lookup = {("sha-test-1", 1, 2): True}
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None, lineage_lookup, None)

    assert "annotation_only_alt_not_verifiable" in res["exclusion_reasons"]
    assert res["true_pair"] is False
    assert res["has_replicate_corroboration"] is False
    assert res["evidence_source"] == "none"


# ---------------------------------------------------------------------------
# §3.2 does NOT override other quality gates
# ---------------------------------------------------------------------------

def test_replicate_does_not_override_comparable_fraction_gate():
    # Corroborated BUT only 1/4 positions comparable (0.25 < 0.60 threshold).
    # §3.2 suppresses ONLY annotation_only_alt_not_verifiable; the quality
    # gate still blocks true_pair.
    wt = _make_profile([0.1, None, None, None])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation()
    lineage_lookup = {("sha-test-1", 1, 2): True}
    res = mod._evaluate_one(
        rel, {1: wt, 2: mut}, None, lineage_lookup, _evidence_lookup_found()
    )

    assert res["has_replicate_corroboration"] is True
    assert "annotation_only_alt_not_verifiable" not in res["exclusion_reasons"]
    assert "comparable_positions_below_60pct" in res["exclusion_reasons"]
    assert res["true_pair"] is False
    assert res["primary_eligible"] is False  # comparable reason is a hard blocker


def test_replicate_with_unverified_lineage_keeps_lineage_blocker():
    # Corroborated BUT lineage NOT verified (no D2 lookup) → lineage blocker
    # remains. §3.2 only clears the annotation-only blocker.
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation()  # lineage_status starts with candidate_only_pending
    res = mod._evaluate_one(
        rel, {1: wt, 2: mut}, None, None, _evidence_lookup_found()
    )

    assert res["has_replicate_corroboration"] is True
    assert "annotation_only_alt_not_verifiable" not in res["exclusion_reasons"]
    assert "parent_lineage_unverified" in res["exclusion_reasons"]
    assert res["true_pair"] is False


# ---------------------------------------------------------------------------
# evidence_lookup key matching
# ---------------------------------------------------------------------------

def test_evidence_lookup_sha_mismatch_no_corroboration():
    # The candidate's rdat_sha256 does not match any evidence_lookup key →
    # no corroboration (defensive: lookup is per-candidate).
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation(rdat_sha256="sha-different")
    lineage_lookup = {("sha-different", 1, 2): True}
    res = mod._evaluate_one(
        rel, {1: wt, 2: mut}, None, lineage_lookup, _evidence_lookup_found()
    )

    assert res["has_replicate_corroboration"] is False
    assert "annotation_only_alt_not_verifiable" in res["exclusion_reasons"]
    assert res["evidence_source"] == "none"


def test_evidence_lookup_wrong_evidence_type_no_corroboration():
    # status=evidence_found but evidence_type=per_profile_sequence → does NOT
    # trigger same_parent_replicate corroboration (only same_parent_replicate
    # substitutes for alt resolution per §3.2).
    wt = _make_profile([0.1, 0.2, 0.3, 0.4])
    mut = _make_profile([0.5, 0.6, 0.7, 0.8])
    rel = _make_relation()
    lineage_lookup = {("sha-test-1", 1, 2): True}
    bad_lookup = {
        ("sha-test-1", 1, 2): {
            "status": "evidence_found",
            "evidence_type": "per_profile_sequence",
        }
    }
    res = mod._evaluate_one(rel, {1: wt, 2: mut}, None, lineage_lookup, bad_lookup)

    assert res["has_replicate_corroboration"] is False
    assert "annotation_only_alt_not_verifiable" in res["exclusion_reasons"]
    # evidence_entry exists but wrong type → d2r_manifest_no_evidence
    assert res["evidence_source"] == "d2r_manifest_no_evidence"
