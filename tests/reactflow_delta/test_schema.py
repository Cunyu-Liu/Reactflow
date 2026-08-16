from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from reactflow.delta.schema import (
    SOURCE_REGISTRY_SCHEMA_VERSION,
    SourceRegistryValidationError,
    source_registry_json_schema,
    validate_source_registry_record,
)


def _downloaded_record() -> dict[str, object]:
    return {
        "schema_version": SOURCE_REGISTRY_SCHEMA_VERSION,
        "record_id": "rmdb:example:release-1",
        "source": "RMDB",
        "source_version": "release-1",
        "source_url": "https://rmdb.stanford.edu/example.rdat",
        "publication_doi": "10.0000/example",
        "publication_pmid": "12345678",
        "license": "CC-BY-4.0",
        "retrieved_at": "2026-07-30T12:00:00+08:00",
        "sha256": "a" * 64,
        "bytes": 1024,
        "upstream_id": "RMDB:EXAMPLE",
        "raw_path": "/mnt/cunyuliu/reactflow_delta_raw/rmdb/example.rdat",
        "parser_version": "unparsed",
        "download_status": "downloaded",
        "source_tier": "A",
        "source_type": "rmdb",
        "missing_reasons": {},
    }


def test_downloaded_record_requires_complete_artifact_provenance() -> None:
    record = _downloaded_record()
    assert validate_source_registry_record(record) == record


def test_metadata_only_record_requires_field_specific_missing_reasons() -> None:
    record = _downloaded_record()
    record.update(
        {
            "source_version": None,
            "publication_doi": None,
            "publication_pmid": None,
            "license": None,
            "sha256": None,
            "bytes": None,
            "raw_path": None,
            "parser_version": None,
            "download_status": "metadata_only",
            "missing_reasons": {
                "source_version": "upstream release does not expose a version",
                "publication_doi": "not indexed by source metadata",
                "publication_pmid": "not indexed by source metadata",
                "license": "license not stated upstream",
                "sha256": "no raw artifact retrieved",
                "bytes": "no raw artifact retrieved",
                "raw_path": "no raw artifact retrieved",
                "parser_version": "no raw artifact available to parse",
            },
        }
    )
    assert validate_source_registry_record(record)["download_status"] == "metadata_only"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda record: record.pop("license"), "missing required fields"),
        (lambda record: record.__setitem__("sha256", "not-a-digest"), "sha256"),
        (lambda record: record.__setitem__("bytes", -1), "bytes"),
        (lambda record: record.__setitem__("retrieved_at", "2026-07-30"), "timezone"),
        (lambda record: record.__setitem__("source_url", "ftp://example.org/data"), "source_url"),
        (lambda record: record.__setitem__("source_tier", "D"), "source_tier"),
        (lambda record: record.__setitem__("unexpected", "value"), "unexpected fields"),
    ],
)
def test_invalid_source_registry_records_fail_closed(mutator, message: str) -> None:
    record = deepcopy(_downloaded_record())
    mutator(record)
    with pytest.raises(SourceRegistryValidationError, match=message):
        validate_source_registry_record(record)


def test_null_field_without_exact_missing_reason_fails_closed() -> None:
    record = _downloaded_record()
    record["license"] = None
    with pytest.raises(SourceRegistryValidationError, match="missing_reasons"):
        validate_source_registry_record(record)


def test_downloaded_record_cannot_omit_raw_artifact_even_with_reason() -> None:
    record = _downloaded_record()
    record["raw_path"] = None
    record["missing_reasons"] = {"raw_path": "artifact unavailable"}
    with pytest.raises(SourceRegistryValidationError, match="downloaded record lacks artifact fields"):
        validate_source_registry_record(record)


def test_json_schema_is_versioned_and_closed_to_unknown_fields() -> None:
    schema = source_registry_json_schema()
    assert schema["$id"] == SOURCE_REGISTRY_SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {"source_url", "sha256", "raw_path", "missing_reasons"}


def test_checked_in_json_schema_matches_runtime_schema() -> None:
    schema_path = Path(__file__).parents[2] / "data_registry" / "source_registry.schema.json"
    assert json.loads(schema_path.read_text()) == source_registry_json_schema()
