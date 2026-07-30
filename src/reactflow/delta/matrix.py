"""Source/study/parent/probe/condition matrix for D0 candidate evidence."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


SOURCE_PAIR_MATRIX_SCHEMA_VERSION = "reactflow-delta-source-pair-matrix-v1"


def build_source_pair_matrix(
    construct_audit_path: str | Path,
    candidate_registry_path: str | Path,
    ribonanza_availability_path: str | Path,
) -> dict[str, Any]:
    """Build a matrix that preserves zero and unknown pair counts distinctly."""

    audit = _load_object(construct_audit_path, "construct audit")
    registry = _load_object(candidate_registry_path, "candidate registry")
    availability = _load_object(ribonanza_availability_path, "Ribonanza availability")
    if audit.get("schema_version") != "reactflow-delta-rmdb-construct-audit-v1":
        raise ValueError("unexpected construct audit schema version")
    if registry.get("schema_version") != "reactflow-delta-candidate-pair-registry-v1":
        raise ValueError("unexpected candidate registry schema version")
    if availability.get("schema_version") != "reactflow-delta-ribonanza-availability-v1":
        raise ValueError("unexpected Ribonanza availability schema version")

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in _require_list(audit, "construct_records"):
        key = _construct_key(record)
        row = grouped.setdefault(
            key,
            {
                "source": key[0], "study_id": key[1], "parent_id": key[2], "probe": key[3], "condition_key": key[4],
                "entry_ids": set(), "construct_profile_count": 0, "candidate_pair_count": 0,
                "pair_count_status": "observed_zero_or_more", "pair_count_missing_reason": None,
            },
        )
        row["entry_ids"].add(_required_string(record, "entry_id"))
        row["construct_profile_count"] += 1

    for pair in _require_list(registry, "candidate_pairs"):
        key = _pair_key(pair)
        if key not in grouped:
            raise ValueError("candidate pair is absent from its construct matrix stratum")
        grouped[key]["candidate_pair_count"] += 1

    rows = []
    for key, row in sorted(grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        rows.append({**row, "entry_ids": sorted(row["entry_ids"])})

    ribonanza_count = availability.get("same_condition_single_edit_pair_count")
    if ribonanza_count is None:
        rows.append(
            {
                "source": "Ribonanza", "study_id": None, "parent_id": None, "probe": None, "condition_key": None,
                "entry_ids": [], "construct_profile_count": 0, "candidate_pair_count": None,
                "pair_count_status": "unknown_not_acquired", "pair_count_missing_reason": availability.get("pair_count_missing_reason"),
            }
        )
    else:
        raise ValueError("D0.9 requires raw Ribonanza matrix construction when a non-null Ribonanza count is supplied")

    return {
        "schema_version": SOURCE_PAIR_MATRIX_SCHEMA_VERSION,
        "stage": "D0",
        "input_construct_audit": _fingerprint(construct_audit_path),
        "input_candidate_registry": _fingerprint(candidate_registry_path),
        "input_ribonanza_availability": _fingerprint(ribonanza_availability_path),
        "matrix_rows": rows,
        "summary": {
            "observed_rmdb_matrix_row_count": len(grouped),
            "rmdb_candidate_pair_count": sum(row["candidate_pair_count"] for row in grouped.values()),
            "ribonanza_pair_count": ribonanza_count,
            "matrix_has_unknown_rows": any(row["pair_count_status"] == "unknown_not_acquired" for row in rows),
        },
        "scientific_boundary": "Rows with zero observed candidates and rows with unknown candidates are distinct. This matrix does not normalize reactivity, construct labels, or training data.",
    }


def _construct_key(record: Any) -> tuple[Any, ...]:
    if not isinstance(record, dict):
        raise ValueError("construct record must be an object")
    return (_required_string(record, "source"), record.get("study_id"), record.get("parent_id"), record.get("probe"), _required_string(record, "condition_key"))


def _pair_key(pair: Any) -> tuple[Any, ...]:
    if not isinstance(pair, dict):
        raise ValueError("candidate pair must be an object")
    return (_required_string(pair, "source"), None, _required_string(pair, "parent_id"), pair.get("probe"), _required_string(pair, "condition_key"))


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _require_list(document: dict[str, Any], key: str) -> list[Any]:
    value = document.get(key)
    if not isinstance(value, list):
        raise ValueError(f"document lacks list {key}")
    return value


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record lacks non-empty {key}")
    return value


def _fingerprint(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    return {"path": str(candidate.resolve()), "sha256": sha256(candidate.read_bytes()).hexdigest()}
