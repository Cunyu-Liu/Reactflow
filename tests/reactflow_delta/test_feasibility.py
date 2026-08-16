from __future__ import annotations

import json
from pathlib import Path

from reactflow.delta.feasibility import build_d0_feasibility_summary, render_d0_feasibility_report


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_feasibility_summary_blocks_d1_when_pair_truth_is_absent(tmp_path: Path) -> None:
    construct = {
        "schema_version": "reactflow-delta-rmdb-construct-audit-v1",
        "construct_records": [{"probe": "DMS", "study_id": None, "parent_id": None, "condition_key": "condition", "missing_reactivity_count": 0}],
        "summary": {"unique_profile_record_count": 1, "explicit_wt_profile_count": 1, "confirmed_single_mutant_profile_count": 0, "single_site_endpoint_unknown_profile_count": 0, "confirmed_double_mutant_profile_count": 0, "same_sequence_replicate_group_count": 0, "same_sequence_replicate_profile_count": 0, "explicit_no_edit_profile_count": 1},
    }
    candidate = {"schema_version": "reactflow-delta-candidate-pair-registry-v1", "summary": {"rmdb_candidate_pair_count": 0, "construct_exclusion_reason_counts": {"missing": 1}}}
    matrix = {"schema_version": "reactflow-delta-source-pair-matrix-v1", "summary": {"matrix_has_unknown_rows": True}}
    relations = {"schema_version": "reactflow-delta-pair-relation-audit-v1", "summary": {"true_pair": 0, "designed_neighbour": 0, "synthetic_pair": 0, "unresolved_candidate": 0}}
    availability = {"schema_version": "reactflow-delta-ribonanza-availability-v1", "same_condition_single_edit_pair_count": None, "pair_count_missing_reason": "no raw table"}
    candidates = {"schema_version": "reactflow-delta-rmdb-candidate-manifest-v1", "categories": [{"candidate_category": "m2_named_candidate", "candidate_count": 2}, {"candidate_category": "m2r_named_unconfirmed", "candidate_count": 1}]}
    parsed = {"schema_version": "reactflow-delta-rdat-construct-parse-manifest-v1", "fixtures": [{"name": "fixture", "sha256": "a" * 64, "seqpos_count": 2, "profiles": [{"missing_reactivity_count": 0}]}]}
    summary, parser = build_d0_feasibility_summary(
        construct_audit_path=_write(tmp_path / "construct.json", construct), candidate_registry_path=_write(tmp_path / "candidate.json", candidate),
        matrix_path=_write(tmp_path / "matrix.json", matrix), relation_audit_path=_write(tmp_path / "relations.json", relations),
        ribonanza_availability_path=_write(tmp_path / "availability.json", availability), filename_candidate_manifest_path=_write(tmp_path / "filename.json", candidates),
        rdat_parse_manifest_path=_write(tmp_path / "parsed.json", parsed),
    )

    assert summary["d1_allowed"] is False
    assert summary["counts"]["confirmed_true_pair_count"] == 0
    assert summary["pair_state"]["ribonanza_same_condition_single_edit_pair_count"] is None
    assert parser["fixture_count"] == 1
    report = render_d0_feasibility_report(summary)
    assert "Allow D1: False" in report
    assert "unknown" in report
