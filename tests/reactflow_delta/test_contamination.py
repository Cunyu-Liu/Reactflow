"""D2 contamination & lineage-graph audit tests (T-D2.1 / T-D2.2 / T-D2.10).

Tests the pure helpers in ``src/reactflow/delta/contamination.py`` with
small synthetic D0-R v2 relation documents. No real RDAT files, no network,
no training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure ``src`` is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reactflow.delta.contamination import (  # noqa: E402
    SPLIT_LEVELS,
    TIER_A_MIN_PAIRS,
    TIER_A_MIN_PARENTS,
    TIER_A_MIN_STUDIES,
    TIER_B_MIN_TRUE_PAIRS,
    audit_split_overlap,
    build_lineage_graph,
    compute_overlap_report,
    compute_tier_judgment,
    verify_parent_lineage,
)


def _make_relation(
    *,
    rdat_sha256: str = "sha-A",
    wt_idx: int = 1,
    mut_idx: int = 2,
    ref_verified_against: str = "header_SEQUENCE",
    parent_prefix: str = "P1",
    rmdb_id: str = "RMDB-1",
    doi: str = "10.0000/test-a",
    owner: str = "Owner A",
    modifier: str = "DMS",
):
    return {
        "rdat_sha256": rdat_sha256,
        "rdat_path": f"/synthetic/{rdat_sha256}.rdat",
        "wt_profile_index": wt_idx,
        "mutant_profile_index": mut_idx,
        "parent_prefix": parent_prefix,
        "rmdb_id": rmdb_id,
        "citation_doi": doi,
        "owner": owner,
        "modifier": modifier,
        "matched_mutation": {
            "encoding_source": "annotation",
            "ref_verified_against": ref_verified_against,
            "alt_not_verified": True,
            "encoded_alt": "X",
            "encoded_ref": "G",
            "encoded_position_1indexed": 1,
        },
    }


# ---------------------------------------------------------------------------
# verify_parent_lineage (T-D2.1)
# ---------------------------------------------------------------------------

def test_verify_parent_lineage_same_rdat_header_ref_verified():
    rel = _make_relation()
    v = verify_parent_lineage(rel)
    assert v["parent_lineage_verified"] is True
    assert v["evidence"]["same_rdat"] is True
    assert v["evidence"]["ref_verified_against"] == "header_SEQUENCE"


def test_verify_parent_lineage_ref_not_verified_against_header():
    # ref not verified against header SEQUENCE → lineage unverified.
    rel = _make_relation(ref_verified_against="name_encoding")
    v = verify_parent_lineage(rel)
    assert v["parent_lineage_verified"] is False
    assert v["evidence"]["same_rdat"] is True
    assert v["evidence"]["ref_verified_against"] == "name_encoding"


def test_verify_parent_lineage_missing_rdat_sha():
    rel = _make_relation(rdat_sha256="")
    v = verify_parent_lineage(rel)
    assert v["parent_lineage_verified"] is False
    assert v["evidence"]["same_rdat"] is False


def test_verify_parent_lineage_keyed_by_pair_indices():
    # The verification record carries the pair indices so the D1 executor
    # can build a (rdat_sha256, wt_idx, mut_idx) lookup.
    rel = _make_relation(wt_idx=3, mut_idx=7)
    v = verify_parent_lineage(rel)
    assert v["wt_profile_index"] == 3
    assert v["mutant_profile_index"] == 7
    assert v["rdat_sha256"] == "sha-A"


# ---------------------------------------------------------------------------
# build_lineage_graph (T-D2.1)
# ---------------------------------------------------------------------------

def test_build_lineage_graph_counts_and_groupings():
    rels = [
        _make_relation(rdat_sha256="sha-A", parent_prefix="P1", rmdb_id="R1", doi="d1", owner="O1"),
        _make_relation(rdat_sha256="sha-A", wt_idx=1, mut_idx=3, parent_prefix="P1", rmdb_id="R1", doi="d1", owner="O1"),
        _make_relation(rdat_sha256="sha-B", parent_prefix="P2", rmdb_id="R2", doi="d1", owner="O1"),
        _make_relation(rdat_sha256="sha-C", parent_prefix="P3", rmdb_id="R3", doi="d2", owner="O2"),
    ]
    g = build_lineage_graph(rels)
    assert g["construct_count"] == 4
    assert g["unique_counts"]["rdat_sha256"] == 3
    assert g["unique_counts"]["parent_prefix"] == 3
    assert g["unique_counts"]["citation_doi"] == 2
    assert g["unique_counts"]["owner"] == 2
    # by_parent grouping: P1 has 2, P2/P3 have 1 each.
    assert g["groupings"]["by_parent"]["P1"] == 2
    assert g["groupings"]["by_parent"]["P2"] == 1
    assert g["groupings"]["by_parent"]["P3"] == 1
    # by_study grouping: d1 has 3, d2 has 1.
    assert g["groupings"]["by_study"]["d1"] == 3
    assert g["groupings"]["by_study"]["d2"] == 1


def test_build_lineage_graph_split_levels_present():
    g = build_lineage_graph([_make_relation()])
    assert g["split_levels"] == list(SPLIT_LEVELS)
    # Family and structure levels are recorded as unknown (no Rfam / no
    # structure computation at D2 — model forward forbidden).
    assert g["groupings"]["by_family"]["status"] == "unknown"
    assert g["groupings"]["by_structure"]["status"] == "unknown"


# ---------------------------------------------------------------------------
# audit_split_overlap (T-D2.2)
# ---------------------------------------------------------------------------

def test_audit_split_overlap_zero_when_disjoint():
    assignment = {
        "train": [("sha-A", 1, 2)],
        "validation": [("sha-B", 1, 2)],
        "test": [("sha-C", 1, 2)],
    }
    report = audit_split_overlap(assignment)
    assert report["max_overlap"] == 0
    assert report["overlap_count"] == 0
    assert report["gate_pass"] is True


def test_audit_split_overlap_detects_shared_construct():
    # sha-A#1-2 appears in both train and test → overlap = 1.
    assignment = {
        "train": [("sha-A", 1, 2), ("sha-B", 1, 2)],
        "test": [("sha-A", 1, 2), ("sha-C", 1, 2)],
    }
    report = audit_split_overlap(assignment)
    assert report["max_overlap"] == 1
    assert report["overlap_count"] == 1
    assert report["gate_pass"] is False
    assert len(report["overlapping_constructs"]) == 1


def test_audit_split_overlap_empty_assignment_passes():
    # No splits frozen yet (true_pair = 0) → trivially gate_pass.
    report = audit_split_overlap({})
    assert report["max_overlap"] == 0
    assert report["gate_pass"] is True
    assert report["construct_count"] == 0


# ---------------------------------------------------------------------------
# compute_overlap_report (within-pool concentration, no splits frozen)
# ---------------------------------------------------------------------------

def test_compute_overlap_report_no_splits_frozen_status():
    rels = [
        _make_relation(rdat_sha256="sha-A", parent_prefix="P1", doi="d1", owner="O1"),
        _make_relation(rdat_sha256="sha-B", parent_prefix="P2", doi="d1", owner="O1"),
    ]
    rep = compute_overlap_report(rels)
    assert rep["status"] == "no_splits_frozen"
    assert rep["candidate_total"] == 2
    assert rep["unique_counts"]["citation_doi"] == 1
    assert rep["unique_counts"]["parent_prefix"] == 2
    # Top-per-level entries are lists of {group, count}.
    for level in ("parent", "design_lineage", "study", "owner"):
        assert isinstance(rep["top_per_level"][level], list)


# ---------------------------------------------------------------------------
# compute_tier_judgment (T-D2.10)
# ---------------------------------------------------------------------------

def test_tier_judgment_zero_true_pairs_below_tier_b():
    tj = compute_tier_judgment(
        true_pair_count=0,
        true_pair_studies=0,
        true_pair_parents=0,
        candidate_total=7761,
        candidate_studies=8,
        candidate_parents=31,
        binding_blocker="annotation_only_alt_not_verifiable",
    )
    assert tj["tier_a"]["pass"] is False
    assert tj["tier_b"]["pass"] is False
    assert tj["tier_c"]["pass"] is False
    assert tj["outcome"] == "below_tier_b_data_audit"
    assert tj["true_pairs"] == 0
    # Candidate counts are reference only — NOT the gate basis.
    assert tj["candidate_level_reference"]["candidate_total"] == 7761


def test_tier_judgment_thresholds_frozen():
    # D2 must not lower thresholds (v3.1 §6.2).
    assert TIER_A_MIN_STUDIES == 5
    assert TIER_A_MIN_PARENTS == 20
    assert TIER_A_MIN_PAIRS == 5000
    assert TIER_B_MIN_TRUE_PAIRS == 1000


def test_tier_judgment_uses_true_pair_counts_not_candidate_counts():
    # A pool with many candidates but zero true_pairs must fail Tier B.
    tj = compute_tier_judgment(
        true_pair_count=0,
        true_pair_studies=0,
        true_pair_parents=0,
        candidate_total=7761,
        candidate_studies=8,
        candidate_parents=31,
    )
    assert tj["tier_b"]["pass"] is False
    assert tj["tier_a"]["pass"] is False


def test_tier_judgment_tier_b_pass_at_threshold():
    tj = compute_tier_judgment(
        true_pair_count=TIER_B_MIN_TRUE_PAIRS,
        true_pair_studies=2,
        true_pair_parents=5,
        candidate_total=8000,
        candidate_studies=8,
        candidate_parents=31,
    )
    assert tj["tier_b"]["pass"] is True
    # Tier A still needs ≥5 studies, ≥20 parents, ≥5000 pairs.
    assert tj["tier_a"]["pass"] is False
