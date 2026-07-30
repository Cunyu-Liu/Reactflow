from __future__ import annotations

import json
from pathlib import Path

from reactflow.delta.matrix import build_source_pair_matrix


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_matrix_preserves_observed_zero_and_unknown_as_different_states(tmp_path: Path) -> None:
    audit = {
        "schema_version": "reactflow-delta-rmdb-construct-audit-v1",
        "construct_records": [
            {"source": "RMDB", "entry_id": "entry-a", "study_id": None, "parent_id": None, "probe": "DMS", "condition_key": "condition-a"},
            {"source": "RMDB", "entry_id": "entry-a", "study_id": None, "parent_id": None, "probe": "DMS", "condition_key": "condition-a"},
        ],
    }
    registry = {"schema_version": "reactflow-delta-candidate-pair-registry-v1", "candidate_pairs": []}
    availability = {"schema_version": "reactflow-delta-ribonanza-availability-v1", "same_condition_single_edit_pair_count": None, "pair_count_missing_reason": "no raw table"}

    matrix = build_source_pair_matrix(
        _write(tmp_path / "audit.json", audit),
        _write(tmp_path / "registry.json", registry),
        _write(tmp_path / "availability.json", availability),
    )

    assert matrix["summary"]["observed_rmdb_matrix_row_count"] == 1
    assert matrix["summary"]["rmdb_candidate_pair_count"] == 0
    assert matrix["summary"]["ribonanza_pair_count"] is None
    assert matrix["matrix_rows"][0]["pair_count_status"] == "observed_zero_or_more"
    assert matrix["matrix_rows"][1]["pair_count_status"] == "unknown_not_acquired"


def test_matrix_rejects_pair_not_present_in_construct_stratum(tmp_path: Path) -> None:
    audit = {"schema_version": "reactflow-delta-rmdb-construct-audit-v1", "construct_records": []}
    registry = {
        "schema_version": "reactflow-delta-candidate-pair-registry-v1",
        "candidate_pairs": [{"source": "RMDB", "entry_id": "entry", "parent_id": "parent", "probe": "DMS", "condition_key": "condition"}],
    }
    availability = {"schema_version": "reactflow-delta-ribonanza-availability-v1", "same_condition_single_edit_pair_count": None, "pair_count_missing_reason": "no raw table"}

    try:
        build_source_pair_matrix(
            _write(tmp_path / "audit.json", audit),
            _write(tmp_path / "registry.json", registry),
            _write(tmp_path / "availability.json", availability),
        )
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("matrix must reject a pair without a construct stratum")
