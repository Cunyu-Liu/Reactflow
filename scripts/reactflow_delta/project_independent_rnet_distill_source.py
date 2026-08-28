#!/usr/bin/env python3
"""Project a strict, outcome-free source manifest for RNet2 teacher singles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    CONTRACT_PATH,
    assert_run_authority,
    _load_yaml,
)


SCHEMA = "reactflow_delta.independent_rnet_distill_teacher_source.v1"
PASS = "RNET2_TEACHER_STRUCTURAL_SOURCE_BINDING_EXACT_PASS"
EXPECTED_INDEX_KEYS = {"arrays", "family", "length", "record_id", "row", "sequence"}
FORBIDDEN_INDEX_TOKENS = ("target", "outcome", "reactivity", "error", "mask", "score", "loss")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def inspect_shard(
    shard_dir: Path,
    root_entry: dict[str, Any],
    *,
    expected_weights: str,
    expected_width: int,
) -> tuple[int, list[str], bool]:
    for name in ("provenance.json", "features.npz", "index.jsonl"):
        path = shard_dir / name
        _require(path.is_file() and path.stat().st_size > 0, f"missing {path}")
    provenance = json.loads((shard_dir / "provenance.json").read_text(encoding="utf-8"))
    count = int(root_entry["record_count"])
    _require(provenance["record_count"] == count, f"record count mismatch in {shard_dir.name}")
    _require(provenance["model_name"] == "RibonanzaNet2", "teacher model changed")
    _require(provenance["model_version"] == "alpha-v1", "teacher version changed")
    _require(provenance["weights_sha256"] == expected_weights, "teacher weights changed")
    legacy_parent_content_binding_matches = (
        provenance["content_sha256"] == root_entry["content_sha256"]
    )
    single_schema = provenance["schema"]["single"]
    _require(single_schema["axes"] == ["L", expected_width], f"single axes changed in {shard_dir.name}")
    _require(single_schema["dtype"] == "<f4", f"single dtype changed in {shard_dir.name}")

    record_ids: list[str] = []
    lines = (shard_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    _require(len(entries) == count, f"index count mismatch in {shard_dir.name}")
    for expected_row, entry in enumerate(entries):
        _require(set(entry) == EXPECTED_INDEX_KEYS, f"index schema changed in {shard_dir.name}")
        _require(not any(token in key.lower() for key in entry for token in FORBIDDEN_INDEX_TOKENS), "outcome-like index field found")
        length = int(entry["length"])
        sequence = str(entry["sequence"])
        _require(entry["row"] == expected_row, f"row order changed in {shard_dir.name}")
        _require(length == len(sequence) and length > 0, f"sequence length mismatch in {shard_dir.name}")
        single = entry["arrays"]["single"]
        _require(single["shape"] == [length, expected_width], f"teacher shape mismatch in {shard_dir.name}")
        _require(single["dtype"] == "<f4", f"teacher array dtype mismatch in {shard_dir.name}")
        record_ids.append(str(entry["record_id"]))
    _require(len(set(record_ids)) == len(record_ids), f"duplicate record id inside {shard_dir.name}")
    expected_members = {f"{row:06d}.single" for row in range(count)}
    with np.load(shard_dir / "features.npz", allow_pickle=False) as archive:
        _require(
            set(archive.files) == expected_members
            and len(archive.files) == len(expected_members),
            f"NPZ member universe changed in {shard_dir.name}",
        )
    return count, record_ids, legacy_parent_content_binding_matches


def project_source(repo_root: Path, cache_root: Path, output: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    assert_run_authority(repo_root, "RND0")
    contract = _load_yaml(repo_root / CONTRACT_PATH)
    source = contract["source_binding"]
    _require(cache_root.resolve() == Path(source["source_cache"]), "cache path is not contract-bound")
    _require(output.resolve() == Path(source["source_manifest"]), "output path is not contract-bound")
    _require(not output.exists(), f"refusing to overwrite {output}")

    root = json.loads((cache_root / "sharded_manifest.json").read_text(encoding="utf-8"))
    expected_weights = source["expected_weights_sha256"]
    _require(root["layout"] == source["expected_layout"], "root layout changed")
    _require(root["model_name"] == source["expected_model_name"], "root model changed")
    _require(root["model_version"] == source["expected_model_version"], "root version changed")
    _require(root["weights_sha256"] == expected_weights, "root weights changed")
    _require(root["record_count"] == source["expected_record_count"], "root record count changed")
    _require(root["shard_count"] == source["expected_shard_count"], "root shard count changed")
    _require(root["shard_size"] == source["expected_full_shard_size"], "root shard size changed")
    shards = root["shards"]
    _require(len(shards) == source["expected_shard_count"], "root shard list changed")
    _require(
        [entry["path"] for entry in shards]
        == [f"shard_{index:05d}" for index in range(source["expected_shard_count"])],
        "shard path universe changed",
    )
    _require(shards[-1]["record_count"] == source["expected_last_shard_records"], "last shard count changed")
    last_provenance = json.loads(
        (cache_root / shards[-1]["path"] / "provenance.json").read_text(
            encoding="utf-8"
        )
    )
    _require(
        last_provenance["content_sha256"]
        == source["source_integrity_repair"]["verified_child_content_sha256"],
        "verified child provenance for the last shard is absent",
    )

    all_ids: set[str] = set()
    counted = 0
    legacy_parent_content_binding_mismatches = 0
    for entry in shards:
        shard_count, record_ids, content_binding_matches = inspect_shard(
            cache_root / entry["path"],
            entry,
            expected_weights=expected_weights,
            expected_width=source["expected_single_feature_dim"],
        )
        counted += shard_count
        legacy_parent_content_binding_mismatches += int(not content_binding_matches)
        overlap = all_ids.intersection(record_ids)
        _require(not overlap, f"duplicate record ids across shards: {sorted(overlap)[:3]}")
        all_ids.update(record_ids)
    _require(counted == source["expected_record_count"], "physical record count changed")
    _require(len(all_ids) == counted, "record-id universe is not unique")
    _require(
        legacy_parent_content_binding_mismatches
        == source["source_integrity_repair"][
            "expected_parent_child_content_binding_mismatches"
        ],
        "legacy parent/child content-binding mismatch count changed",
    )

    document = {
        "schema_version": SCHEMA,
        "status": PASS,
        "project_task_id": contract["project_task_id"],
        "source_cache": str(cache_root),
        "layout": root["layout"],
        "model_name": root["model_name"],
        "model_version": root["model_version"],
        "weights_sha256": root["weights_sha256"],
        "record_count": counted,
        "shard_count": len(shards),
        "single_feature_dim": source["expected_single_feature_dim"],
        "legacy_parent_content_hashes_authoritative": False,
        "legacy_parent_content_binding_mismatches": legacy_parent_content_binding_mismatches,
        "verified_recovered_last_shard_content_sha256": last_provenance[
            "content_sha256"
        ],
        "payload_hash_verification_scope": "SHARD_00408_ONLY",
        "full_cache_rehash_performed": False,
        "structural_binding_basis": source["source_integrity_repair"]["binding_basis"],
        "index_contains_outcome_fields": False,
        "teacher_pair_features_used": False,
        "live_teacher_used": False,
        "openknot_mutant_outcome_accessed": False,
        "new_external_outcome_accessed": False,
        "exact_sequence_overlap_prior_audit": {
            "observed_overlap": source["exact_sequence_overlap_with_registered_openknot_mutants"],
            "registered_sequence_count": source["registered_openknot_mutant_sequence_count"],
            "interpretation": source["overlap_audit_interpretation"],
        },
        "scientific_evidence_ceiling": contract["scope"]["result_class"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(project_source(args.repo_root, args.cache_root, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
