from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from reactflow.delta.registry import (
    CONSTRUCT_CANDIDATE_COLUMNS,
    PAIR_CANDIDATE_COLUMNS,
    build_candidate_pair_registry,
    construct_candidate_rows,
    write_parquet_rows,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def _availability() -> dict:
    return {
        "schema_version": "reactflow-delta-ribonanza-availability-v1",
        "same_condition_single_edit_pair_count": None,
        "pair_count_missing_reason": "raw data unavailable",
    }


def _record(record_id: str, mutation_class: str, *, eligible: bool) -> dict:
    return {
        "construct_record_id": record_id,
        "source": "RMDB",
        "entry_id": "entry",
        "entry_sha256": "a" * 64,
        "profile_index": 1 if mutation_class == "explicit_wt" else 2,
        "study_id": None,
        "parent_id": "parent-1" if eligible else None,
        "probe": "DMS",
        "condition_key": "condition-1",
        "experiment_type": "MutateAndMap",
        "mutation_labels": ["WT"] if mutation_class == "explicit_wt" else ["A1G"],
        "mutation_class": mutation_class,
        "edit_count": 0 if mutation_class == "explicit_wt" else 1,
        "endpoint_identity": "explicit",
        "profile_sequence": "ACGU",
        "missing_reactivity_count": 0,
        "reactivity_error_present": False,
        "pair_eligibility": eligible,
        "pair_ineligibility_reason": "not eligible in this fixture" if not eligible else "",
    }


def test_candidate_registry_keeps_unknown_ribonanza_total_unknown(tmp_path: Path) -> None:
    audit = {"schema_version": "reactflow-delta-rmdb-construct-audit-v1", "construct_records": [_record("wt", "explicit_wt", eligible=True), _record("mut", "single_exact_endpoint", eligible=True)]}
    registry = build_candidate_pair_registry(_write(tmp_path / "audit.json", audit), _write(tmp_path / "availability.json", _availability()))

    assert registry["summary"]["rmdb_candidate_pair_count"] == 1
    assert registry["summary"]["total_candidate_pair_count"] is None
    assert registry["candidate_pairs"][0]["normalization_status"] == "not_attempted_in_D0"


def test_ineligible_constructs_do_not_become_pairs(tmp_path: Path) -> None:
    audit = {"schema_version": "reactflow-delta-rmdb-construct-audit-v1", "construct_records": [_record("wt", "explicit_wt", eligible=False), _record("mut", "single_exact_endpoint", eligible=False)]}
    registry = build_candidate_pair_registry(_write(tmp_path / "audit.json", audit), _write(tmp_path / "availability.json", _availability()))
    assert registry["candidate_pairs"] == []
    assert registry["summary"]["construct_exclusion_reason_counts"] == {"not eligible in this fixture": 2}


def test_parquet_writer_preserves_empty_pair_schema_and_refuses_overwrite(tmp_path: Path) -> None:
    construct_path = tmp_path / "construct.parquet"
    pair_path = tmp_path / "pairs.parquet"
    construct_rows = construct_candidate_rows({"construct_records": [_record("wt", "explicit_wt", eligible=False)]})
    write_parquet_rows(construct_path, construct_rows, CONSTRUCT_CANDIDATE_COLUMNS)
    write_parquet_rows(pair_path, [], PAIR_CANDIDATE_COLUMNS)

    assert pq.read_table(construct_path).num_rows == 1
    table = pq.read_table(pair_path)
    assert table.num_rows == 0
    assert table.column_names == list(PAIR_CANDIDATE_COLUMNS)
    try:
        write_parquet_rows(pair_path, [], PAIR_CANDIDATE_COLUMNS)
    except FileExistsError:
        pass
    else:
        raise AssertionError("writer must reject overwrite")
