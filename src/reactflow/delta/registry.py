"""Candidate registry construction for D0; never normalizes or trains."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


CANDIDATE_PAIR_REGISTRY_SCHEMA_VERSION = "reactflow-delta-candidate-pair-registry-v1"
CONSTRUCT_CANDIDATE_COLUMNS = (
    "construct_record_id", "source", "entry_id", "entry_sha256", "profile_index", "study_id", "parent_id", "probe", "condition_key", "experiment_type", "mutation_labels_json", "mutation_class", "edit_count", "endpoint_identity", "profile_sequence", "missing_reactivity_count", "reactivity_error_present", "pair_eligibility", "pair_ineligibility_reason",
)
PAIR_CANDIDATE_COLUMNS = (
    "candidate_pair_id", "source", "entry_id", "condition_key", "parent_id", "probe", "wt_construct_record_id", "mutant_construct_record_id", "mutation_labels_json", "pair_relation", "normalization_status", "scientific_status",
)


def build_candidate_pair_registry(construct_audit_path: str | Path, ribonanza_availability_path: str | Path) -> dict[str, Any]:
    """Build only explicit eligible WT--single candidates from audited records."""

    audit = _load_object(construct_audit_path, "construct audit")
    availability = _load_object(ribonanza_availability_path, "Ribonanza availability report")
    if audit.get("schema_version") != "reactflow-delta-rmdb-construct-audit-v1":
        raise ValueError("unexpected construct audit schema version")
    if availability.get("schema_version") != "reactflow-delta-ribonanza-availability-v1":
        raise ValueError("unexpected Ribonanza availability schema version")
    records = audit.get("construct_records")
    if not isinstance(records, list):
        raise ValueError("construct audit lacks construct_records list")

    exclusions: Counter[str] = Counter()
    groups: dict[tuple[str, str, str, str, str | None], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"wt": [], "mutant": []})
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("construct audit contains a non-object record")
        if record.get("pair_eligibility") is not True:
            exclusions[_required_string(record, "pair_ineligibility_reason")] += 1
            continue
        mutation_class = _required_string(record, "mutation_class")
        if mutation_class not in {"explicit_wt", "single_exact_endpoint"}:
            exclusions[f"eligible flag incompatible with {mutation_class}"] += 1
            continue
        group_key = (
            _required_string(record, "source"),
            _required_string(record, "entry_id"),
            _required_string(record, "condition_key"),
            _required_string(record, "parent_id"),
            record.get("probe"),
        )
        groups[group_key]["wt" if mutation_class == "explicit_wt" else "mutant"].append(record)

    pairs: list[dict[str, Any]] = []
    for (source, entry_id, condition_key, parent_id, probe), endpoints in sorted(groups.items()):
        for wt in sorted(endpoints["wt"], key=lambda item: _required_string(item, "construct_record_id")):
            for mutant in sorted(endpoints["mutant"], key=lambda item: _required_string(item, "construct_record_id")):
                pair_id = _stable_digest([wt["construct_record_id"], mutant["construct_record_id"]])
                pairs.append(
                    {
                        "candidate_pair_id": f"candidate:{pair_id}",
                        "source": source,
                        "entry_id": entry_id,
                        "condition_key": condition_key,
                        "parent_id": parent_id,
                        "probe": probe,
                        "wt_construct_record_id": wt["construct_record_id"],
                        "mutant_construct_record_id": mutant["construct_record_id"],
                        "mutation_labels_json": json.dumps(mutant.get("mutation_labels", []), sort_keys=True),
                        "pair_relation": "explicit_wt_to_single_exact_endpoint_candidate",
                        "normalization_status": "not_attempted_in_D0",
                        "scientific_status": "candidate_only_pending_pair_identity_and_source_matrix_audit",
                    }
                )
    pair_ids = [pair["candidate_pair_id"] for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate candidate pair IDs")

    ribonanza_count = availability.get("same_condition_single_edit_pair_count")
    if ribonanza_count is not None and (isinstance(ribonanza_count, bool) or not isinstance(ribonanza_count, int) or ribonanza_count < 0):
        raise ValueError("Ribonanza pair count must be non-negative integer or null")
    return {
        "schema_version": CANDIDATE_PAIR_REGISTRY_SCHEMA_VERSION,
        "stage": "D0",
        "input_construct_audit": {"path": str(Path(construct_audit_path).resolve()), "sha256": _sha256_file(construct_audit_path)},
        "input_ribonanza_availability": {"path": str(Path(ribonanza_availability_path).resolve()), "sha256": _sha256_file(ribonanza_availability_path)},
        "candidate_pairs": pairs,
        "summary": {
            "rmdb_candidate_pair_count": len(pairs),
            "ribonanza_same_condition_single_edit_pair_count": ribonanza_count,
            "total_candidate_pair_count": None if ribonanza_count is None else len(pairs) + ribonanza_count,
            "total_count_missing_reason": availability.get("pair_count_missing_reason") if ribonanza_count is None else None,
            "construct_exclusion_reason_counts": dict(sorted(exclusions.items())),
        },
        "scientific_boundary": "Candidate registry only. No final normalization, label construction, pair weighting, split assignment, or learned training is performed in D0.",
    }


def construct_candidate_rows(construct_audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten audited construct records into a fixed Parquet schema."""

    rows = []
    for record in construct_audit.get("construct_records", []):
        if not isinstance(record, dict):
            raise ValueError("construct audit contains a non-object record")
        row = {column: None for column in CONSTRUCT_CANDIDATE_COLUMNS}
        for key in row:
            if key == "mutation_labels_json":
                row[key] = json.dumps(record.get("mutation_labels", []), sort_keys=True)
            else:
                row[key] = record.get(key)
        rows.append(row)
    return rows


def write_parquet_rows(path: str | Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    """Write an immutable Parquet artifact with explicit columns, including zero rows."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Parquet artifact: {output}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write D0 Parquet artifacts") from exc
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    table = pa.Table.from_pylist(normalized, schema=pa.schema([(column, _arrow_type(column, pa)) for column in columns]))
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")


def _arrow_type(column: str, pa: Any) -> Any:
    if column in {"profile_index", "edit_count", "missing_reactivity_count"}:
        return pa.int64()
    if column in {"reactivity_error_present", "pair_eligibility"}:
        return pa.bool_()
    return pa.string()


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record lacks non-empty {key}")
    return value


def _stable_digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
