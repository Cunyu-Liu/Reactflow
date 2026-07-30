"""D2 RSIB-v1 split / lineage-graph build tests (T-D2.1 / T-D2.2 / T-D2.3-5).

Tests ``scripts/reactflow_delta/build_rsib.py`` end-to-end on a small
synthetic D0-R v2 relations document: lineage graph construction, parent
lineage verification, overlap audit, and Tier judgment. Verifies the D2
Gate machinery (split group overlap = 0) and the forward-only artifact
shape that the D1 executor consumes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_build_rsib():
    path = _REPO_ROOT / "scripts" / "reactflow_delta" / "build_rsib.py"
    spec = importlib.util.spec_from_file_location("build_rsib", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_build_rsib()


def _make_relations_doc():
    """Small synthetic D0-R v2 relations document (4 candidates, 2 studies)."""
    relations = [
        {
            "rdat_sha256": "sha-A", "rdat_path": "/syn/A.rdat",
            "wt_profile_index": 1, "mutant_profile_index": 2,
            "parent_prefix": "P1", "rmdb_id": "R1",
            "citation_doi": "10.0000/a", "owner": "O1", "modifier": "DMS",
            "matched_mutation": {
                "encoding_source": "annotation",
                "ref_verified_against": "header_SEQUENCE",
                "alt_not_verified": True, "encoded_alt": "X",
                "encoded_ref": "G", "encoded_position_1indexed": 1,
            },
        },
        {
            "rdat_sha256": "sha-A", "rdat_path": "/syn/A.rdat",
            "wt_profile_index": 1, "mutant_profile_index": 3,
            "parent_prefix": "P1", "rmdb_id": "R1",
            "citation_doi": "10.0000/a", "owner": "O1", "modifier": "DMS",
            "matched_mutation": {
                "encoding_source": "annotation",
                "ref_verified_against": "header_SEQUENCE",
                "alt_not_verified": False, "encoded_alt": "A",
                "encoded_ref": "G", "encoded_position_1indexed": 5,
            },
        },
        {
            "rdat_sha256": "sha-B", "rdat_path": "/syn/B.rdat",
            "wt_profile_index": 1, "mutant_profile_index": 2,
            "parent_prefix": "P2", "rmdb_id": "R2",
            "citation_doi": "10.0000/a", "owner": "O1", "modifier": "DMS",
            "matched_mutation": {
                "encoding_source": "annotation",
                "ref_verified_against": "header_SEQUENCE",
                "alt_not_verified": True, "encoded_alt": "X",
                "encoded_ref": "C", "encoded_position_1indexed": 3,
            },
        },
        {
            "rdat_sha256": "sha-C", "rdat_path": "/syn/C.rdat",
            "wt_profile_index": 1, "mutant_profile_index": 2,
            "parent_prefix": "P3", "rmdb_id": "R3",
            "citation_doi": "10.0000/b", "owner": "O2", "modifier": "SHAPE",
            "matched_mutation": {
                "encoding_source": "annotation",
                "ref_verified_against": "header_SEQUENCE",
                "alt_not_verified": True, "encoded_alt": "X",
                "encoded_ref": "U", "encoded_position_1indexed": 10,
            },
        },
    ]
    return {
        "generated_at": "2026-07-31T00:00:00+08:00",
        "lineage_status_all": "candidate_only_pending_parent_lineage_and_functional_region_validation",
        "relations": relations,
        "schema_version": "reactflow-delta-d0r-reaudit-tierA-relations-v1",
        "stage": "D0-R v2 re-audit",
        "total_candidate_relations": len(relations),
        "true_pair_all": False,
    }


def test_build_rsib_main_writes_four_artifacts(tmp_path):
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_make_relations_doc()))
    out_dir = tmp_path / "d2"

    rc = mod.main(["--relations", str(rel_path), "--out-dir", str(out_dir)])
    assert rc == 0

    assert (out_dir / "d2_lineage_graph.json").exists()
    assert (out_dir / "d2_lineage_verification.json").exists()
    assert (out_dir / "d2_overlap_audit.json").exists()
    assert (out_dir / "d2_tier_judgment.json").exists()


def test_d2_lineage_graph_artifact_shape(tmp_path):
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_make_relations_doc()))
    out_dir = tmp_path / "d2"
    mod.main(["--relations", str(rel_path), "--out-dir", str(out_dir)])

    graph = json.loads((out_dir / "d2_lineage_graph.json").read_text())
    assert graph["schema_version"] == "reactflow-delta-d2-lineage-graph-v1"
    assert graph["construct_count"] == 4
    assert graph["unique_counts"]["rdat_sha256"] == 3
    assert graph["unique_counts"]["parent_prefix"] == 3
    assert graph["unique_counts"]["citation_doi"] == 2
    assert graph["unique_counts"]["owner"] == 2
    # by_parent: P1 has 2, P2/P3 have 1 each.
    assert graph["groupings"]["by_parent"]["P1"] == 2
    assert graph["groupings"]["by_parent"]["P2"] == 1
    assert graph["groupings"]["by_parent"]["P3"] == 1
    # Family / structure levels recorded as unknown (forward-only).
    assert graph["groupings"]["by_family"]["status"] == "unknown"
    assert graph["groupings"]["by_structure"]["status"] == "unknown"


def test_d2_lineage_verification_all_verified_for_header_ref(tmp_path):
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_make_relations_doc()))
    out_dir = tmp_path / "d2"
    mod.main(["--relations", str(rel_path), "--out-dir", str(out_dir)])

    ver = json.loads((out_dir / "d2_lineage_verification.json").read_text())
    assert ver["schema_version"] == "reactflow-delta-d2-lineage-verification-v1"
    assert ver["total_candidates"] == 4
    # All 4 carry ref_verified_against="header_SEQUENCE" → all verified.
    assert ver["parent_lineage_verified_count"] == 4
    assert ver["parent_lineage_unverified_count"] == 0
    # The verification rule is recorded for auditability.
    assert "header_SEQUENCE" in ver["verification_rule"]
    # Each verification entry carries the pair key for D1 executor lookup.
    for v in ver["verifications"]:
        assert v["parent_lineage_verified"] is True
        assert "rdat_sha256" in v
        assert "wt_profile_index" in v
        assert "mutant_profile_index" in v


def test_d2_overlap_audit_no_splits_frozen(tmp_path):
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_make_relations_doc()))
    out_dir = tmp_path / "d2"
    mod.main(["--relations", str(rel_path), "--out-dir", str(out_dir)])

    audit = json.loads((out_dir / "d2_overlap_audit.json").read_text())
    assert audit["schema_version"] == "reactflow-delta-d2-overlap-audit-v1"
    # No true_pairs → split freezing deferred, empty-overlap gate passes.
    assert audit["split_freezing_status"]["status"] == "deferred"
    assert audit["split_freezing_status"]["empty_split_overlap_gate"]["gate_pass"] is True
    assert audit["split_freezing_status"]["empty_split_overlap_gate"]["max_overlap"] == 0
    assert audit["status"] == "no_splits_frozen"
    assert audit["candidate_total"] == 4


def test_d2_tier_judgment_below_tier_b_when_no_d1_summary(tmp_path):
    # No --d1-summary supplied (file absent) → true_pair_count defaults to 0.
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_make_relations_doc()))
    out_dir = tmp_path / "d2"
    mod.main(
        [
            "--relations", str(rel_path),
            "--out-dir", str(out_dir),
            "--d1-summary", str(tmp_path / "nonexistent_summary.json"),
        ]
    )

    tj = json.loads((out_dir / "d2_tier_judgment.json").read_text())
    assert tj["schema_version"] == "reactflow-delta-d2-tier-judgment-v1"
    assert tj["true_pairs"] == 0
    assert tj["tier_a"]["pass"] is False
    assert tj["tier_b"]["pass"] is False
    assert tj["outcome"] == "below_tier_b_data_audit"


def test_d2_tier_judgment_reads_d1_summary_true_pair_count(tmp_path):
    # Supply a synthetic D1 summary with true_pair_count=0 and the
    # annotation-only reason distribution → binding blocker reported.
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_make_relations_doc()))
    d1_summary = {
        "candidate_total": 4,
        "true_pair_count": 0,
        "reason_distribution_per_reason": {
            "annotation_only_alt_not_verifiable": 4,
        },
        "tier_judgment": {
            "study_distribution_true_pair": {},
            "tier_a": {"studies_true_pair": 0, "parents_true_pair": 0},
            "candidate_level_reference": {
                "candidate_total": 4, "candidate_studies": 2, "candidate_parents": 3,
            },
        },
    }
    d1_summary_path = tmp_path / "d1_summary.json"
    d1_summary_path.write_text(json.dumps(d1_summary))
    out_dir = tmp_path / "d2"
    mod.main(
        [
            "--relations", str(rel_path),
            "--out-dir", str(out_dir),
            "--d1-summary", str(d1_summary_path),
        ]
    )

    tj = json.loads((out_dir / "d2_tier_judgment.json").read_text())
    assert tj["true_pairs"] == 0
    assert tj["tier_b"]["pass"] is False
    assert "annotation_only_alt_not_verifiable" in tj["binding_blocker"]


def test_d2_tier_judgment_pre_d2_summary_reports_lineage_not_cleared(tmp_path):
    # A pre-D2 summary (parent_lineage_unverified still present,
    # _d2_lineage_applied False) → binding_blocker tells the user to re-run
    # the executor with --lineage-verification.
    rel_path = tmp_path / "relations.json"
    rel_path.write_text(json.dumps(_make_relations_doc()))
    d1_summary = {
        "candidate_total": 4,
        "true_pair_count": 0,
        "_d2_lineage_applied": False,
        "reason_distribution_per_reason": {
            "parent_lineage_unverified": 4,
            "annotation_only_alt_not_verifiable": 3,
        },
        "tier_judgment": {
            "study_distribution_true_pair": {},
            "tier_a": {"studies_true_pair": 0, "parents_true_pair": 0},
            "candidate_level_reference": {
                "candidate_total": 4, "candidate_studies": 2, "candidate_parents": 3,
            },
        },
    }
    d1_summary_path = tmp_path / "d1_summary.json"
    d1_summary_path.write_text(json.dumps(d1_summary))
    out_dir = tmp_path / "d2"
    mod.main(
        [
            "--relations", str(rel_path),
            "--out-dir", str(out_dir),
            "--d1-summary", str(d1_summary_path),
        ]
    )

    tj = json.loads((out_dir / "d2_tier_judgment.json").read_text())
    assert "parent_lineage_unverified" in tj["binding_blocker"]
    assert "re-run" in tj["binding_blocker"]
