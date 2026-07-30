from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactflow.delta.relations import classify_pair_relations


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def _audit() -> dict:
    return {"schema_version": "reactflow-delta-rmdb-construct-audit-v1", "replicate_groups": [{"replicate_group_id": "same-sequence:x"}]}


def test_relation_audit_separates_unresolved_design_and_synthetic(tmp_path: Path) -> None:
    registry = {
        "schema_version": "reactflow-delta-candidate-pair-registry-v1",
        "candidate_pairs": [
            {"candidate_pair_id": "p1", "pair_relation": "explicit_wt_to_single_exact_endpoint_candidate"},
            {"candidate_pair_id": "p2", "pair_relation": "designed_neighbour"},
            {"candidate_pair_id": "p3", "pair_relation": "synthetic_pair"},
        ],
    }
    document = classify_pair_relations(_write(tmp_path / "registry.json", registry), _write(tmp_path / "audit.json", _audit()))

    assert document["summary"]["true_pair"] == 0
    assert document["summary"]["unresolved_candidate"] == 1
    assert document["summary"]["designed_neighbour"] == 1
    assert document["summary"]["synthetic_pair"] == 1
    assert document["summary"]["constructs_not_promoted_to_pairs"] is True
    assert all(not row["eligible_as_primary_truth"] for row in document["pair_relation_rows"])


def test_relation_audit_rejects_unknown_relation(tmp_path: Path) -> None:
    registry = {"schema_version": "reactflow-delta-candidate-pair-registry-v1", "candidate_pairs": [{"candidate_pair_id": "p1", "pair_relation": "made_up"}]}
    with pytest.raises(ValueError, match="unknown"):
        classify_pair_relations(_write(tmp_path / "registry.json", registry), _write(tmp_path / "audit.json", _audit()))
