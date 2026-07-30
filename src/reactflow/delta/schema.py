"""Fail-closed schemas for the ReactFlow-Delta public-source audit."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import re
from typing import Any
from urllib.parse import urlparse


SOURCE_REGISTRY_SCHEMA_VERSION = "reactflow-delta-source-registry-v1"
DOWNLOAD_STATUSES = frozenset({"planned", "downloaded", "metadata_only", "unavailable", "failed"})
SOURCE_TIERS = frozenset({"A", "B", "C"})
SOURCE_TYPES = frozenset({"rmdb", "ribonanza", "other_public"})

REQUIRED_SOURCE_REGISTRY_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "source",
        "source_version",
        "source_url",
        "publication_doi",
        "publication_pmid",
        "license",
        "retrieved_at",
        "sha256",
        "bytes",
        "upstream_id",
        "raw_path",
        "parser_version",
        "download_status",
        "source_tier",
        "source_type",
        "missing_reasons",
    }
)

NULLABLE_AUDIT_FIELDS = frozenset(
    {
        "source_version",
        "source_url",
        "publication_doi",
        "publication_pmid",
        "license",
        "sha256",
        "bytes",
        "raw_path",
        "parser_version",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PMID_RE = re.compile(r"^[0-9]+$")


class SourceRegistryValidationError(ValueError):
    """Raised when a source-registry entry lacks auditable provenance."""


def source_registry_json_schema() -> dict[str, Any]:
    """Return a dependency-free JSON-Schema representation of registry v1."""

    nullable_string = {"type": ["string", "null"]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "reactflow-delta-source-registry-v1",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_SOURCE_REGISTRY_FIELDS),
        "properties": {
            "schema_version": {"const": SOURCE_REGISTRY_SCHEMA_VERSION},
            "record_id": {"type": "string", "minLength": 1},
            "source": {"type": "string", "minLength": 1},
            "source_version": nullable_string,
            "source_url": nullable_string,
            "publication_doi": nullable_string,
            "publication_pmid": nullable_string,
            "license": nullable_string,
            "retrieved_at": {"type": "string", "format": "date-time"},
            "sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
            "bytes": {"type": ["integer", "null"], "minimum": 0},
            "upstream_id": {"type": "string", "minLength": 1},
            "raw_path": nullable_string,
            "parser_version": nullable_string,
            "download_status": {"enum": sorted(DOWNLOAD_STATUSES)},
            "source_tier": {"enum": sorted(SOURCE_TIERS)},
            "source_type": {"enum": sorted(SOURCE_TYPES)},
            "missing_reasons": {
                "type": "object",
                "additionalProperties": {"type": "string", "minLength": 1},
            },
        },
    }


def validate_source_registry_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one auditable source record without inferring missing metadata.

    `None` is permitted only for explicitly nullable audit fields and every such
    value must have a non-empty field-specific explanation in `missing_reasons`.
    A record marked `downloaded` must have a physical raw artifact with bytes and
    a lowercase SHA-256 digest; metadata-only and failed records remain explicit.
    """

    if not isinstance(record, Mapping):
        raise SourceRegistryValidationError("source registry record must be a mapping")

    keys = set(record)
    missing = REQUIRED_SOURCE_REGISTRY_FIELDS - keys
    unexpected = keys - REQUIRED_SOURCE_REGISTRY_FIELDS
    if missing:
        raise SourceRegistryValidationError(f"missing required fields: {sorted(missing)}")
    if unexpected:
        raise SourceRegistryValidationError(f"unexpected fields: {sorted(unexpected)}")

    normalized = dict(record)
    if normalized["schema_version"] != SOURCE_REGISTRY_SCHEMA_VERSION:
        raise SourceRegistryValidationError("unsupported source registry schema_version")

    for field in ("record_id", "source", "upstream_id"):
        _require_nonempty_string(field, normalized[field])
    for field in ("source_version", "source_url", "publication_doi", "publication_pmid", "license", "raw_path", "parser_version"):
        _require_nullable_string(field, normalized[field])

    _validate_timestamp(normalized["retrieved_at"])
    _validate_optional_url(normalized["source_url"])
    _validate_optional_doi(normalized["publication_doi"])
    _validate_optional_pmid(normalized["publication_pmid"])
    _validate_optional_sha256(normalized["sha256"])
    _validate_optional_bytes(normalized["bytes"])

    if normalized["download_status"] not in DOWNLOAD_STATUSES:
        raise SourceRegistryValidationError("invalid download_status")
    if normalized["source_tier"] not in SOURCE_TIERS:
        raise SourceRegistryValidationError("invalid source_tier")
    if normalized["source_type"] not in SOURCE_TYPES:
        raise SourceRegistryValidationError("invalid source_type")

    reasons = normalized["missing_reasons"]
    if not isinstance(reasons, Mapping):
        raise SourceRegistryValidationError("missing_reasons must be a mapping")
    null_fields = {field for field in NULLABLE_AUDIT_FIELDS if normalized[field] is None}
    if set(reasons) != null_fields:
        raise SourceRegistryValidationError("missing_reasons keys must exactly match nullable null fields")
    for field, reason in reasons.items():
        _require_nonempty_string(f"missing_reasons[{field}]", reason)

    if normalized["download_status"] == "downloaded":
        absent = [field for field in ("sha256", "bytes", "raw_path") if normalized[field] is None]
        if absent:
            raise SourceRegistryValidationError(f"downloaded record lacks artifact fields: {absent}")
        if not normalized["raw_path"].startswith("/"):
            raise SourceRegistryValidationError("downloaded raw_path must be absolute")

    return deepcopy(normalized)


def _require_nonempty_string(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SourceRegistryValidationError(f"{field} must be a non-empty string")


def _require_nullable_string(field: str, value: Any) -> None:
    if value is not None:
        _require_nonempty_string(field, value)


def _validate_timestamp(value: Any) -> None:
    _require_nonempty_string("retrieved_at", value)
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SourceRegistryValidationError("retrieved_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise SourceRegistryValidationError("retrieved_at must include a timezone")


def _validate_optional_url(value: str | None) -> None:
    if value is not None and urlparse(value).scheme not in {"http", "https"}:
        raise SourceRegistryValidationError("source_url must use http or https")


def _validate_optional_doi(value: str | None) -> None:
    if value is not None and (not value.startswith("10.") or any(char.isspace() for char in value)):
        raise SourceRegistryValidationError("publication_doi must be a compact DOI")


def _validate_optional_pmid(value: str | None) -> None:
    if value is not None and not _PMID_RE.fullmatch(value):
        raise SourceRegistryValidationError("publication_pmid must contain only digits")


def _validate_optional_sha256(value: str | None) -> None:
    if value is not None and not _SHA256_RE.fullmatch(value):
        raise SourceRegistryValidationError("sha256 must be 64 lowercase hexadecimal characters")


def _validate_optional_bytes(value: Any) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise SourceRegistryValidationError("bytes must be a non-negative integer")
