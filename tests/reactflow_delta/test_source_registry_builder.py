from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactflow.delta.data import (
    RAW_MANIFEST_SCHEMA_VERSION,
    RMDB_METADATA_SPECS,
    build_rmdb_metadata_registry,
    write_json_document,
    write_jsonl_records,
)


def _write_snapshot(directory: Path) -> None:
    directory.mkdir()
    (directory / "github_main_commit.json").write_text(json.dumps({"sha": "b" * 40}))
    (directory / "github_releases.json").write_text(
        json.dumps([{"tag_name": "data-general", "published_at": "2026-01-01T00:00:00Z", "html_url": "https://example.org/release", "assets": [{"size": 12}, {"size": 34}]}])
    )
    (directory / "github_root_contents.json").write_text("[]")
    (directory / "rmdb_index.html").write_text("<html>index</html>")
    (directory / "rdat_specification.html").write_text("<html>rdat</html>")
    (directory / "rmdb_about.html").write_text("<html>license</html>")


def test_rmdb_metadata_registry_is_checksumed_and_does_not_claim_pairs(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    _write_snapshot(metadata_dir)

    records, manifest = build_rmdb_metadata_registry(
        metadata_dir,
        retrieved_at="2026-07-30T12:00:00+08:00",
    )

    assert len(records) == len(RMDB_METADATA_SPECS)
    assert {record["upstream_id"] for record in records} == {spec[1] for spec in RMDB_METADATA_SPECS}
    assert all(record["download_status"] == "downloaded" for record in records)
    assert all(len(record["sha256"]) == 64 for record in records)
    assert manifest["schema_version"] == RAW_MANIFEST_SCHEMA_VERSION
    assert manifest["release_summary"] == [
        {
            "tag_name": "data-general",
            "published_at": "2026-01-01T00:00:00Z",
            "asset_count": 2,
            "asset_bytes": 46,
            "html_url": "https://example.org/release",
        }
    ]
    assert "No pair count" in manifest["scientific_boundary"]


def test_metadata_registry_fails_when_required_snapshot_file_is_missing(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    _write_snapshot(metadata_dir)
    (metadata_dir / "rmdb_about.html").unlink()

    with pytest.raises(FileNotFoundError, match="rmdb_about.html"):
        build_rmdb_metadata_registry(metadata_dir, retrieved_at="2026-07-30T12:00:00+08:00")


def test_registry_writers_refuse_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    _write_snapshot(metadata_dir)
    records, manifest = build_rmdb_metadata_registry(metadata_dir, retrieved_at="2026-07-30T12:00:00+08:00")
    registry = tmp_path / "source_registry.jsonl"
    raw_manifest = tmp_path / "raw_manifest.json"

    write_jsonl_records(registry, records)
    write_json_document(raw_manifest, manifest)
    with pytest.raises(FileExistsError):
        write_jsonl_records(registry, records)
    with pytest.raises(FileExistsError):
        write_json_document(raw_manifest, manifest)


def test_invalid_timestamp_fails_before_emitting_registry(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    _write_snapshot(metadata_dir)

    with pytest.raises(ValueError, match="timezone"):
        build_rmdb_metadata_registry(metadata_dir, retrieved_at="2026-07-30")
