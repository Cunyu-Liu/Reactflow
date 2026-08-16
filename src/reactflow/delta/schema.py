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

# ============================================================================
# D1 Construct schema (v3 §6.3) and Pair schema (v3 §6.4)
#
# Frozen by T-D1.1. These schemas define the field set, types, nullability,
# enums, and key invariants for construct and pair records produced by the D1
# cleaning pipeline. All fields listed in v3 §6.3/§6.4 are present; nullable
# fields may be null but every null MUST have a matching missing_reason entry
# (no guessing, per v3 §6.3).
# ============================================================================

CONSTRUCT_SCHEMA_VERSION = "reactflow-delta-construct-v1"
PAIR_SCHEMA_VERSION = "reactflow-delta-pair-v1"

EDIT_TYPES = frozenset({"substitution", "insertion", "deletion"})
CONDITION_MATCH_STATUSES = frozenset({"exact_match", "mismatch"})
IN_VIVO_IN_VITRO_STATUSES = frozenset({"in_vivo", "in_vitro"})

# Exclusion reasons controlled vocabulary (v3.1 §4 D1 Gate).
# "取值至少包含" — these are the minimum frozen set for D1.
EXCLUSION_REASONS = frozenset({
    "annotation_only_alt_not_verifiable",
    "sequence_based_no_independent_corroboration",
    "substitution_not_verifiable",
    "annotation_ref_mismatch",
    "condition_mismatch",
    "probe_mismatch",
    "comparable_positions_below_60pct",
    "edit_count_not_one",
    "indel_not_substitution",
    "no_wt_anchor",
    "normalization_domain_unknown",
    "parent_lineage_unverified",
    "in_vivo_in_vitro_mixed",
})

# --- Construct schema (v3 §6.3): 28 contract fields + schema_version + missing_reasons ---

REQUIRED_CONSTRUCT_FIELDS = frozenset({
    "schema_version",
    "construct_id",
    "source_entry_id",
    "study_id",
    "publication_id",
    "laboratory_id",
    "parent_id",
    "design_lineage_id",
    "sequence_raw",
    "sequence_normalized",
    "length",
    "probe",
    "probe_protocol",
    "temperature",
    "ligand",
    "ligand_concentration",
    "buffer",
    "in_vivo_in_vitro",
    "batch_id",
    "replicate_id",
    "reactivity_raw",
    "reactivity_upstream",
    "reactivity_error",
    "coverage",
    "snr",
    "valid_mask",
    "probe_eligibility_mask",
    "normalization_method",
    "quality_flags",
    "missing_reasons",
})

# Nullable construct fields: may be null, but each null MUST appear in missing_reasons.
NULLABLE_CONSTRUCT_FIELDS = frozenset({
    "publication_id",
    "laboratory_id",
    "design_lineage_id",
    "probe_protocol",
    "temperature",
    "ligand",
    "ligand_concentration",
    "buffer",
    "batch_id",
    "replicate_id",
    "reactivity_raw",
    "reactivity_upstream",
    "reactivity_error",
    "coverage",
    "snr",
    "normalization_method",
})

# --- Pair schema (v3 §6.4): 27 contract fields + schema_version + missing_reasons ---

REQUIRED_PAIR_FIELDS = frozenset({
    "schema_version",
    "pair_id",
    "wt_construct_id",
    "mut_construct_id",
    "parent_id",
    "study_id",
    "design_lineage_id",
    "edit_type",
    "edit_positions",
    "wt_alleles",
    "mut_alleles",
    "edit_count",
    "alignment_cigar",
    "condition_match_fields",
    "condition_match_status",
    "delta_reactivity_raw",
    "delta_reactivity_normalized",
    "unchanged_position_mask",
    "changed_position_mask",
    "probe_eligibility_unchanged_mask",
    "local_mask",
    "mid_mask",
    "remote_mask",
    "replicate_noise_estimate",
    "measurement_variance",
    "pair_quality_weight",
    "primary_eligible",
    "exclusion_reasons",
    "missing_reasons",
})

NULLABLE_PAIR_FIELDS = frozenset({
    "design_lineage_id",
    "delta_reactivity_raw",
    "delta_reactivity_normalized",
    "replicate_noise_estimate",
    "measurement_variance",
})

# Fields that must be non-empty lists of 0/1 integers with length == construct length.
_CONSTRUCT_MASK_FIELDS = ("valid_mask", "probe_eligibility_mask")
# Reactivity-like arrays that, when non-null, must have length == construct length.
_CONSTRUCT_ARRAY_FIELDS = ("reactivity_raw", "reactivity_upstream", "reactivity_error", "coverage")
# Pair mask fields: lists of 0/1 (length validated by alignment, not by schema freeze).
_PAIR_MASK_FIELDS = (
    "unchanged_position_mask",
    "changed_position_mask",
    "probe_eligibility_unchanged_mask",
    "local_mask",
    "mid_mask",
    "remote_mask",
)


class ConstructValidationError(ValueError):
    """Raised when a construct record violates the frozen D1 schema (v3 §6.3)."""


class PairValidationError(ValueError):
    """Raised when a pair record violates the frozen D1 schema (v3 §6.4)."""


def construct_json_schema() -> dict[str, Any]:
    """Return a dependency-free JSON-Schema representation of construct v1."""

    def ns() -> dict[str, Any]:
        return {"type": ["string", "null"]}

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CONSTRUCT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_CONSTRUCT_FIELDS),
        "properties": {
            "schema_version": {"const": CONSTRUCT_SCHEMA_VERSION},
            "construct_id": {"type": "string", "minLength": 1},
            "source_entry_id": {"type": "string", "minLength": 1},
            "study_id": {"type": "string", "minLength": 1},
            "publication_id": ns(),
            "laboratory_id": ns(),
            "parent_id": {"type": "string", "minLength": 1},
            "design_lineage_id": ns(),
            "sequence_raw": {"type": "string", "minLength": 1},
            "sequence_normalized": {"type": "string", "minLength": 1},
            "length": {"type": "integer", "minimum": 0},
            "probe": {"type": "string", "minLength": 1},
            "probe_protocol": ns(),
            "temperature": {"type": ["number", "null"]},
            "ligand": ns(),
            "ligand_concentration": ns(),
            "buffer": ns(),
            "in_vivo_in_vitro": {"enum": sorted(IN_VIVO_IN_VITRO_STATUSES)},
            "batch_id": ns(),
            "replicate_id": ns(),
            "reactivity_raw": {"type": ["array", "null"], "items": {"type": "number"}},
            "reactivity_upstream": {"type": ["array", "null"], "items": {"type": "number"}},
            "reactivity_error": {"type": ["array", "null"], "items": {"type": "number"}},
            "coverage": {"type": ["array", "null"], "items": {"type": "number"}},
            "snr": {"type": ["number", "null"]},
            "valid_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "probe_eligibility_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "normalization_method": ns(),
            "quality_flags": {"type": "object"},
            "missing_reasons": {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}},
        },
    }


def pair_json_schema() -> dict[str, Any]:
    """Return a dependency-free JSON-Schema representation of pair v1."""

    def ns() -> dict[str, Any]:
        return {"type": ["string", "null"]}

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PAIR_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_PAIR_FIELDS),
        "properties": {
            "schema_version": {"const": PAIR_SCHEMA_VERSION},
            "pair_id": {"type": "string", "minLength": 1},
            "wt_construct_id": {"type": "string", "minLength": 1},
            "mut_construct_id": {"type": "string", "minLength": 1},
            "parent_id": {"type": "string", "minLength": 1},
            "study_id": {"type": "string", "minLength": 1},
            "design_lineage_id": ns(),
            "edit_type": {"enum": sorted(EDIT_TYPES)},
            "edit_positions": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "wt_alleles": {"type": "array", "items": {"type": "string"}},
            "mut_alleles": {"type": "array", "items": {"type": "string"}},
            "edit_count": {"type": "integer", "minimum": 0},
            "alignment_cigar": {"type": "string", "minLength": 1},
            "condition_match_fields": {"type": "array", "items": {"type": "string"}},
            "condition_match_status": {"enum": sorted(CONDITION_MATCH_STATUSES)},
            "delta_reactivity_raw": {"type": ["array", "null"], "items": {"type": "number"}},
            "delta_reactivity_normalized": {"type": ["array", "null"], "items": {"type": "number"}},
            "unchanged_position_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "changed_position_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "probe_eligibility_unchanged_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "local_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "mid_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "remote_mask": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
            "replicate_noise_estimate": {"type": ["number", "null"]},
            "measurement_variance": {"type": ["number", "null"]},
            "pair_quality_weight": {"type": "number", "minimum": 0},
            "primary_eligible": {"type": "boolean"},
            "exclusion_reasons": {"type": "array", "items": {"enum": sorted(EXCLUSION_REASONS)}},
            "missing_reasons": {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}},
        },
    }


def validate_construct_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a construct record against the frozen D1 schema (v3 §6.3).

    All REQUIRED_CONSTRUCT_FIELDS must be present. Nullable fields may be null
    but every null must have a matching entry in missing_reasons. No guessing.
    Key invariants: length == len(sequence_normalized), no T in
    sequence_normalized, mask/array lengths match length.
    """

    if not isinstance(record, Mapping):
        raise ConstructValidationError("construct record must be a mapping")

    keys = set(record)
    missing = REQUIRED_CONSTRUCT_FIELDS - keys
    unexpected = keys - REQUIRED_CONSTRUCT_FIELDS
    if missing:
        raise ConstructValidationError(f"missing required fields: {sorted(missing)}")
    if unexpected:
        raise ConstructValidationError(f"unexpected fields: {sorted(unexpected)}")

    normalized = dict(record)
    if normalized["schema_version"] != CONSTRUCT_SCHEMA_VERSION:
        raise ConstructValidationError("unsupported construct schema_version")

    for field in ("construct_id", "source_entry_id", "study_id", "parent_id",
                  "sequence_raw", "sequence_normalized", "probe"):
        _d1_require_nonempty_string(field, normalized[field], ConstructValidationError)

    # Nullable string-typed fields
    for field in ("publication_id", "laboratory_id", "design_lineage_id",
                  "probe_protocol", "ligand", "ligand_concentration", "buffer",
                  "batch_id", "replicate_id", "normalization_method"):
        val = normalized[field]
        if val is not None and not isinstance(val, str):
            raise ConstructValidationError(f"{field} must be a string or null")

    # length
    length = normalized["length"]
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ConstructValidationError("length must be a non-negative integer")

    # in_vivo_in_vitro enum
    if normalized["in_vivo_in_vitro"] not in IN_VIVO_IN_VITRO_STATUSES:
        raise ConstructValidationError("in_vivo_in_vitro must be 'in_vivo' or 'in_vitro'")

    # sequence_normalized: T->U normalization (no T allowed)
    seq_norm = normalized["sequence_normalized"]
    if "T" in seq_norm:
        raise ConstructValidationError("sequence_normalized must not contain 'T' (T->U normalization)")

    # length == len(sequence_normalized)
    if length != len(seq_norm):
        raise ConstructValidationError(
            f"length ({length}) must equal len(sequence_normalized) ({len(seq_norm)})"
        )

    # mask length + 0/1 membership
    for mask_field in _CONSTRUCT_MASK_FIELDS:
        mask = normalized[mask_field]
        if not isinstance(mask, list):
            raise ConstructValidationError(f"{mask_field} must be a list")
        if len(mask) != length:
            raise ConstructValidationError(
                f"{mask_field} length ({len(mask)}) must equal length ({length})"
            )
        for v in mask:
            if v not in (0, 1):
                raise ConstructValidationError(f"{mask_field} must contain only 0/1")

    # reactivity-like arrays: when non-null, length must match
    for arr_field in _CONSTRUCT_ARRAY_FIELDS:
        arr = normalized[arr_field]
        if arr is not None:
            if not isinstance(arr, list):
                raise ConstructValidationError(f"{arr_field} must be a list or null")
            if len(arr) != length:
                raise ConstructValidationError(
                    f"{arr_field} length ({len(arr)}) must equal length ({length})"
                )

    # temperature
    temp = normalized["temperature"]
    if temp is not None and (isinstance(temp, bool) or not isinstance(temp, (int, float))):
        raise ConstructValidationError("temperature must be a number or null")

    # snr
    snr = normalized["snr"]
    if snr is not None and (isinstance(snr, bool) or not isinstance(snr, (int, float))):
        raise ConstructValidationError("snr must be a number or null")

    # quality_flags must be a dict
    if not isinstance(normalized["quality_flags"], dict):
        raise ConstructValidationError("quality_flags must be a dict")

    _d1_validate_missing_reasons(normalized, NULLABLE_CONSTRUCT_FIELDS, ConstructValidationError)

    return deepcopy(normalized)


def validate_pair_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a pair record against the frozen D1 schema (v3 §6.4).

    All REQUIRED_PAIR_FIELDS must be present. Nullable fields may be null but
    every null must have a matching entry in missing_reasons. Key invariants:
    edit_count == len(edit_positions) == len(wt_alleles) == len(mut_alleles);
    substitution positions must have wt_allele != mut_allele; exclusion_reasons
    values must be in the frozen EXCLUSION_REASONS vocabulary; D1 eligibility
    invariants link primary_eligible, edit_type, and condition_match_status to
    exclusion_reasons.
    """

    if not isinstance(record, Mapping):
        raise PairValidationError("pair record must be a mapping")

    keys = set(record)
    missing = REQUIRED_PAIR_FIELDS - keys
    unexpected = keys - REQUIRED_PAIR_FIELDS
    if missing:
        raise PairValidationError(f"missing required fields: {sorted(missing)}")
    if unexpected:
        raise PairValidationError(f"unexpected fields: {sorted(unexpected)}")

    normalized = dict(record)
    if normalized["schema_version"] != PAIR_SCHEMA_VERSION:
        raise PairValidationError("unsupported pair schema_version")

    for field in ("pair_id", "wt_construct_id", "mut_construct_id", "parent_id",
                  "study_id", "alignment_cigar"):
        _d1_require_nonempty_string(field, normalized[field], PairValidationError)

    # design_lineage_id: string or null
    dli = normalized["design_lineage_id"]
    if dli is not None and not isinstance(dli, str):
        raise PairValidationError("design_lineage_id must be a string or null")

    # edit_type enum
    if normalized["edit_type"] not in EDIT_TYPES:
        raise PairValidationError("edit_type must be 'substitution', 'insertion', or 'deletion'")

    # condition_match_status enum
    if normalized["condition_match_status"] not in CONDITION_MATCH_STATUSES:
        raise PairValidationError("condition_match_status must be 'exact_match' or 'mismatch'")

    # edit_count consistency
    ec = normalized["edit_count"]
    if isinstance(ec, bool) or not isinstance(ec, int) or ec < 0:
        raise PairValidationError("edit_count must be a non-negative integer")
    ep = normalized["edit_positions"]
    wa = normalized["wt_alleles"]
    ma = normalized["mut_alleles"]
    if not isinstance(ep, list) or not isinstance(wa, list) or not isinstance(ma, list):
        raise PairValidationError("edit_positions, wt_alleles, mut_alleles must be lists")
    if len(ep) != ec or len(wa) != ec or len(ma) != ec:
        raise PairValidationError(
            f"edit_count ({ec}) must equal len(edit_positions)={len(ep)}, "
            f"len(wt_alleles)={len(wa)}, len(mut_alleles)={len(ma)}"
        )

    # substitution: wt != mut at each position
    if normalized["edit_type"] == "substitution":
        for i, (w, m) in enumerate(zip(wa, ma)):
            if w == m:
                raise PairValidationError(
                    f"substitution requires wt_allele != mut_allele at index {i} (got {w!r})"
                )

    # condition_match_fields must be a list of strings
    cmf = normalized["condition_match_fields"]
    if not isinstance(cmf, list) or not all(isinstance(x, str) for x in cmf):
        raise PairValidationError("condition_match_fields must be a list of strings")

    # delta_reactivity arrays: list or null
    for arr_field in ("delta_reactivity_raw", "delta_reactivity_normalized"):
        arr = normalized[arr_field]
        if arr is not None and not isinstance(arr, list):
            raise PairValidationError(f"{arr_field} must be a list or null")

    # mask lists: 0/1 membership
    for mask_field in _PAIR_MASK_FIELDS:
        mask = normalized[mask_field]
        if not isinstance(mask, list):
            raise PairValidationError(f"{mask_field} must be a list")
        for v in mask:
            if v not in (0, 1):
                raise PairValidationError(f"{mask_field} must contain only 0/1")

    # replicate_noise_estimate, measurement_variance: number or null
    for num_field in ("replicate_noise_estimate", "measurement_variance"):
        val = normalized[num_field]
        if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float))):
            raise PairValidationError(f"{num_field} must be a number or null")

    # pair_quality_weight
    pqw = normalized["pair_quality_weight"]
    if isinstance(pqw, bool) or not isinstance(pqw, (int, float)) or pqw < 0:
        raise PairValidationError("pair_quality_weight must be a non-negative number")

    # primary_eligible
    if not isinstance(normalized["primary_eligible"], bool):
        raise PairValidationError("primary_eligible must be a boolean")

    # exclusion_reasons: must be valid reason strings
    reasons = normalized["exclusion_reasons"]
    if not isinstance(reasons, list):
        raise PairValidationError("exclusion_reasons must be a list")
    for r in reasons:
        if not isinstance(r, str) or r not in EXCLUSION_REASONS:
            raise PairValidationError(f"exclusion_reasons contains invalid reason: {r!r}")

    # D1 invariants linking eligibility and exclusion reasons
    if not normalized["primary_eligible"] and not reasons:
        raise PairValidationError(
            "primary_eligible=False requires non-empty exclusion_reasons"
        )
    if normalized["edit_type"] != "substitution" and "indel_not_substitution" not in reasons:
        raise PairValidationError(
            "non-substitution edit_type requires 'indel_not_substitution' in exclusion_reasons"
        )
    if normalized["condition_match_status"] == "mismatch" and "condition_mismatch" not in reasons:
        raise PairValidationError(
            "condition_match_status='mismatch' requires 'condition_mismatch' in exclusion_reasons"
        )

    _d1_validate_missing_reasons(normalized, NULLABLE_PAIR_FIELDS, PairValidationError)

    return deepcopy(normalized)


# --- shared D1 helpers ---

def _d1_require_nonempty_string(field: str, value: Any, error_cls: type[ValueError]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"{field} must be a non-empty string")


def _d1_validate_missing_reasons(
    normalized: dict[str, Any],
    nullable: frozenset[str],
    error_cls: type[ValueError],
) -> None:
    reasons = normalized["missing_reasons"]
    if not isinstance(reasons, dict):
        raise error_cls("missing_reasons must be a dict")
    null_fields = {f for f in nullable if normalized[f] is None}
    if set(reasons) != null_fields:
        raise error_cls(
            f"missing_reasons keys {sorted(set(reasons))} must exactly match "
            f"nullable null fields {sorted(null_fields)}"
        )
    for f, r in reasons.items():
        if not isinstance(r, str) or not r.strip():
            raise error_cls(f"missing_reasons[{f}] must be a non-empty string")
