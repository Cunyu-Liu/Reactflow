from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reactflow_delta.project_independent_rnet_distill_source import inspect_shard


def _write_shard(root: Path, *, bad_length: bool = False, outcome_key: bool = False) -> tuple[Path, dict]:
    shard = root / "shard_00000"
    shard.mkdir()
    np.savez(
        shard / "features.npz",
        **{"000000.single": np.zeros((3, 384), dtype=np.float32)},
    )
    provenance = {
        "model_name": "RibonanzaNet2",
        "model_version": "alpha-v1",
        "weights_sha256": "weights",
        "content_sha256": "content",
        "record_count": 1,
        "schema": {"single": {"axes": ["L", 384], "dtype": "<f4"}},
    }
    (shard / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    entry = {
        "arrays": {"single": {"shape": [3 if not bad_length else 2, 384], "dtype": "<f4"}},
        "family": "fixture",
        "length": 3,
        "record_id": "fixture-0",
        "row": 0,
        "sequence": "ACG",
    }
    if outcome_key:
        entry["target"] = [0.0]
    (shard / "index.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return shard, {
        "path": shard.name,
        "record_count": 1,
        "content_sha256": "content",
        "weights_sha256": "weights",
    }


def test_inspect_shard_accepts_exact_single_only_schema(tmp_path: Path) -> None:
    shard, root_entry = _write_shard(tmp_path)
    count, record_ids, content_binding_matches = inspect_shard(
        shard, root_entry, expected_weights="weights", expected_width=384
    )
    assert count == 1
    assert record_ids == ["fixture-0"]
    assert content_binding_matches is True


def test_inspect_shard_rejects_silent_length_resize(tmp_path: Path) -> None:
    shard, root_entry = _write_shard(tmp_path, bad_length=True)
    with pytest.raises(RuntimeError, match="teacher shape mismatch"):
        inspect_shard(shard, root_entry, expected_weights="weights", expected_width=384)


def test_inspect_shard_rejects_outcome_field(tmp_path: Path) -> None:
    shard, root_entry = _write_shard(tmp_path, outcome_key=True)
    with pytest.raises(RuntimeError, match="index schema changed"):
        inspect_shard(shard, root_entry, expected_weights="weights", expected_width=384)
