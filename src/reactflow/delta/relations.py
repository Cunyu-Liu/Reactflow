"""Explicit D0 relation taxonomy: true candidates, design neighbours, synthetic."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PAIR_RELATION_AUDIT_SCHEMA_VERSION = "reactflow-delta-pair-relation-audit-v1"
_RELATION_MAP = {
    "explicit_wt_to_single_exact_endpoint_candidate": "unresolved_candidate",
    "designed_neighbour": "designed_neighbour",
    "synthetic_pair": "synthetic_pair",
}


def classify_pair_relations(candidate_registry_path: str | Path, construct_audit_path: str | Path) -> dict[str, Any]:
    """Classify pair rows without creating any relation from construct proximity."""

    registry = _load_object(candidate_registry_path, "candidate registry")
    construct_audit = _load_object(construct_audit_path, "construct audit")
    if registry.get("schema_version") != "reactflow-delta-candidate-pair-registry-v1":
        raise ValueError("unexpected candidate registry schema version")
    if construct_audit.get("schema_version") != "reactflow-delta-rmdb-construct-audit-v1":
        raise ValueError("unexpected construct audit schema version")
    rows = registry.get("candidate_pairs")
    if not isinstance(rows, list):
        raise ValueError("candidate registry lacks candidate_pairs list")
    classified = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate pair must be an object")
        relation = row.get("pair_relation")
        if relation not in _RELATION_MAP:
            raise ValueError(f"unknown candidate pair relation: {relation!r}")
        category = _RELATION_MAP[relation]
        classified.append(
            {
                "candidate_pair_id": _required_string(row, "candidate_pair_id"),
                "input_pair_relation": relation,
                "category": category,
                "classification_reason": _classification_reason(category),
                "eligible_as_primary_truth": False,
            }
        )
    ids = [row["candidate_pair_id"] for row in classified]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate pair IDs in relation audit")
    summary = {category: sum(row["category"] == category for row in classified) for category in ("true_pair", "unresolved_candidate", "designed_neighbour", "synthetic_pair")}
    return {
        "schema_version": PAIR_RELATION_AUDIT_SCHEMA_VERSION,
        "stage": "D0",
        "input_candidate_registry": _fingerprint(candidate_registry_path),
        "input_construct_audit": _fingerprint(construct_audit_path),
        "pair_relation_rows": classified,
        "summary": {
            **summary,
            "classified_pair_count": len(classified),
            "same_sequence_replicate_group_count": len(construct_audit.get("replicate_groups", [])),
            "constructs_not_promoted_to_pairs": True,
        },
        "scientific_boundary": "A construct, a same-sequence replicate, or a sequence-neighbour is never promoted to a pair. This D0 registry has no true pair unless a raw candidate row independently carries the required provenance.",
    }


def _classification_reason(category: str) -> str:
    if category == "unresolved_candidate":
        return "explicit candidate relation exists but D0 has not established final true-pair provenance"
    if category == "designed_neighbour":
        return "design adjacency is not a matched experimental WT-mutant pair"
    if category == "synthetic_pair":
        return "synthetic relation cannot be primary intervention truth"
    raise ValueError(f"unsupported category: {category}")


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"row lacks non-empty {key}")
    return value


def _fingerprint(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    return {"path": str(candidate.resolve()), "sha256": sha256(candidate.read_bytes()).hexdigest()}
