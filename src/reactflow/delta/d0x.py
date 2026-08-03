"""Strict D0-X RDAT audit helpers.

This module deliberately produces a candidate inventory, not a canonical Delta
dataset.  It layers loss accounting, raw-token retention, profile-override
tracing, and exact-ref/alt checks on the historical RDAT parser without changing
that parser's backward-compatible behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .rdat import RdatParseError, parse_rdat


D0X_PROFILE_SCHEMA = "reactflow_delta.d0x_profile_inventory.v1"
_INDEXED_KIND = re.compile(
    r"^(ANNOTATION_DATA|SEQUENCE|REACTIVITY|REACTIVITY_ERROR):(.*)$"
)
_SEQPOS = re.compile(r"^[ACGUNX]?([1-9][0-9]*)$")
_SUBSTITUTION = re.compile(r"^([ACGUT])([1-9][0-9]*)([ACGUTX])$", re.IGNORECASE)
_MULTI_SEPARATOR = re.compile(r"[,;/+]\s*")


class D0XContractError(ValueError):
    """A fail-closed violation of the D0-X parsing contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_seqpos_to_indices(tokens: Iterable[str]) -> list[int]:
    """Parse every SEQPOS token, rejecting malformed or duplicate coordinates."""

    result: list[int] = []
    for ordinal, raw in enumerate(tokens, 1):
        match = _SEQPOS.fullmatch(raw.strip())
        if match is None:
            raise D0XContractError(
                f"malformed SEQPOS token at ordinal {ordinal}: {raw!r}"
            )
        result.append(int(match.group(1)))
    if len(result) != len(set(result)):
        raise D0XContractError("duplicate SEQPOS coordinate")
    return result


def _split_fields(raw_line: str) -> list[str]:
    return raw_line.split("\t") if "\t" in raw_line else raw_line.split()


def scan_indexed_and_annotation_lines(path: str | Path) -> dict[str, Any]:
    """Inventory indexed RDAT lines and raw annotation tokens with line numbers."""

    indexed: dict[str, dict[int, int]] = {
        "ANNOTATION_DATA": {},
        "SEQUENCE": {},
        "REACTIVITY": {},
        "REACTIVITY_ERROR": {},
    }
    annotations: dict[str, Any] = {"construct": [], "profiles": {}}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        fields = _split_fields(raw_line)
        key = fields[0].strip()
        values = [value.strip() for value in fields[1:]]
        if key == "ANNOTATION":
            target = annotations["construct"]
            for token in values:
                target.append(_raw_annotation_token(token, line_number, None))
            continue
        indexed_match = _INDEXED_KIND.fullmatch(key)
        if indexed_match is None:
            if any(key.startswith(prefix + ":") for prefix in indexed):
                raise D0XContractError(
                    f"malformed indexed key at line {line_number}: {key!r}"
                )
            continue
        kind, raw_index = indexed_match.groups()
        if not raw_index.isdigit() or int(raw_index) < 1:
            raise D0XContractError(
                f"invalid positive profile index at line {line_number}: {key!r}"
            )
        index = int(raw_index)
        if index in indexed[kind]:
            raise D0XContractError(
                f"duplicate {kind} index {index} at lines "
                f"{indexed[kind][index]} and {line_number}"
            )
        indexed[kind][index] = line_number
        if kind == "ANNOTATION_DATA":
            target = annotations["profiles"].setdefault(index, [])
            for token in values:
                target.append(_raw_annotation_token(token, line_number, index))

    reactivity_indices = set(indexed["REACTIVITY"])
    all_indices: set[int] = set()
    for members in indexed.values():
        all_indices.update(members)
    orphan_by_kind = {
        kind: sorted(set(members) - reactivity_indices)
        for kind, members in indexed.items()
        if kind != "REACTIVITY"
    }
    missing_annotation = sorted(reactivity_indices - set(indexed["ANNOTATION_DATA"]))
    explicitly_accounted = reactivity_indices | {
        index for values in orphan_by_kind.values() for index in values
    }
    if all_indices != explicitly_accounted:
        missing = sorted(all_indices - explicitly_accounted)
        raise D0XContractError(f"silent index-accounting loss: {missing}")
    return {
        "indexed_line_numbers": {
            kind: {str(index): line for index, line in sorted(members.items())}
            for kind, members in indexed.items()
        },
        "annotations": annotations,
        "all_indexed_profile_indices": sorted(all_indices),
        "reactivity_profile_indices": sorted(reactivity_indices),
        "orphan_indices_by_kind": orphan_by_kind,
        "missing_profile_annotation_indices": missing_annotation,
        "accounting_equation": {
            "all_unique_index_count": len(all_indices),
            "reactivity_profile_count": len(reactivity_indices),
            "explicit_orphan_union_count": len(
                {index for values in orphan_by_kind.values() for index in values}
            ),
            "silent_drop_count": 0,
        },
    }


def _raw_annotation_token(
    token: str, line_number: int, profile_index: int | None
) -> dict[str, Any]:
    if not token or ":" not in token:
        raise D0XContractError(
            f"annotation token lacks key/value separator at line {line_number}: {token!r}"
        )
    key, value = token.split(":", 1)
    if not key or not value:
        raise D0XContractError(
            f"annotation token has empty key/value at line {line_number}: {token!r}"
        )
    return {
        "raw_token": token,
        "key": key,
        "value": value,
        "line_number": line_number,
        "profile_index": profile_index,
    }


def resolve_annotations(
    construct_blocks: list[dict[str, list[str]]],
    profile_block: dict[str, list[str]],
    raw_inventory: dict[str, Any],
    profile_index: int,
) -> dict[str, Any]:
    """Resolve construct annotations with a fully traceable profile override."""

    construct_values: dict[str, list[str]] = {}
    for block in construct_blocks:
        for key, values in block.items():
            construct_values.setdefault(key, []).extend(values)
    keys = sorted(set(construct_values) | set(profile_block))
    construct_raw = raw_inventory["annotations"]["construct"]
    profile_raw = raw_inventory["annotations"]["profiles"].get(profile_index, [])
    resolved: dict[str, Any] = {}
    for key in keys:
        profile_values = list(profile_block.get(key, []))
        inherited_values = list(construct_values.get(key, []))
        use_profile = bool(profile_values)
        resolved[key] = {
            "construct_values": inherited_values,
            "profile_values": profile_values,
            "resolved_values": profile_values if use_profile else inherited_values,
            "resolution": "PROFILE_OVERRIDE" if use_profile else "CONSTRUCT_INHERITED",
            "construct_raw_tokens": [
                item for item in construct_raw if item["key"] == key
            ],
            "profile_raw_tokens": [item for item in profile_raw if item["key"] == key],
        }
    return resolved


def _normalize_base(base: str) -> str:
    return base.upper().replace("T", "U")


def parse_mutation_value(
    raw_value: str,
    *,
    header_sequence: str,
    offset: int,
    profile_sequence: str | None,
) -> dict[str, Any]:
    """Parse one mutation annotation without dropping invalid or latent alleles."""

    value = raw_value.strip()
    if value.upper() == "WT":
        return {"raw_value": raw_value, "kind": "WT", "edits": [], "issues": []}
    pieces = _MULTI_SEPARATOR.split(value)
    edits: list[dict[str, Any]] = []
    issues: list[str] = []
    for piece in pieces:
        match = _SUBSTITUTION.fullmatch(piece)
        if match is None:
            return {
                "raw_value": raw_value,
                "kind": "INVALID_MUTATION_TOKEN",
                "edits": [],
                "issues": [f"unparsed mutation component: {piece!r}"],
            }
        ref_raw, position_text, alt_raw = match.groups()
        position = int(position_text)
        sequence_index = position - offset - 1
        ref = _normalize_base(ref_raw)
        alt = alt_raw.upper()
        alt_normalized = None if alt == "X" else _normalize_base(alt)
        header_ref = None
        ref_match = False
        if 0 <= sequence_index < len(header_sequence):
            header_ref = _normalize_base(header_sequence[sequence_index])
            ref_match = header_ref == ref
        else:
            issues.append("coordinate_outside_header_sequence")
        if header_ref is not None and not ref_match:
            issues.append("reference_allele_mismatch")
        profile_alt = None
        alt_match = None
        if profile_sequence is not None and 0 <= sequence_index < len(profile_sequence):
            profile_alt = _normalize_base(profile_sequence[sequence_index])
            alt_match = alt_normalized is not None and profile_alt == alt_normalized
            if alt_normalized is not None and not alt_match:
                issues.append("alternate_allele_profile_sequence_mismatch")
        edits.append(
            {
                "raw_component": piece,
                "ref_allele_raw": ref_raw.upper(),
                "ref_allele": ref,
                "alt_allele_raw": alt,
                "alt_allele": alt_normalized,
                "source_coordinate_1_based": position,
                "source_offset": offset,
                "sequence_index_0_based": sequence_index,
                "header_ref_allele": header_ref,
                "ref_matches_header": ref_match,
                "profile_alt_allele": profile_alt,
                "alt_matches_profile_sequence": alt_match,
            }
        )
    if len(edits) > 1:
        kind = "MULTI_EDIT"
    elif edits[0]["alt_allele"] is None:
        kind = "LATENT_ALT_SINGLE_SUBSTITUTION"
    else:
        kind = "EXACT_SINGLE_SUBSTITUTION"
    return {"raw_value": raw_value, "kind": kind, "edits": edits, "issues": issues}


def _profile_role_and_evidence(parsed: list[dict[str, Any]]) -> tuple[str | None, str]:
    if not parsed:
        return None, "MISSING_MUTATION_ANNOTATION"
    if len(parsed) != 1:
        return "RESCUE_MULTI_EDIT", "MULTIPLE_MUTATION_ANNOTATION_VALUES"
    item = parsed[0]
    if item["kind"] == "WT":
        return None, "WT_CONTROL_CANDIDATE"
    if item["kind"] == "INVALID_MUTATION_TOKEN":
        return None, "INVALID_MUTATION_TOKEN"
    if item["kind"] == "MULTI_EDIT":
        return "RESCUE_MULTI_EDIT", "MULTI_EDIT_TOKEN_PARSED"
    if item["kind"] == "LATENT_ALT_SINGLE_SUBSTITUTION":
        return "AUXILIARY_LATENT_ALT", "LATENT_ALT_X_REF_CHECKED"
    edit = item["edits"][0]
    if not edit["ref_matches_header"]:
        return None, "EXACT_TOKEN_REFERENCE_MISMATCH"
    if edit["alt_matches_profile_sequence"] is False:
        return None, "EXACT_TOKEN_ALTERNATE_MISMATCH"
    if edit["alt_matches_profile_sequence"] is True:
        return "PRIMARY_EXACT_DELTA", "EXACT_REF_ALT_PROFILE_SEQUENCE_VERIFIED"
    return (
        "PRIMARY_EXACT_DELTA",
        "EXACT_REF_ALT_TOKEN_REF_VERIFIED_PROFILE_SEQUENCE_UNAVAILABLE",
    )


def audit_rdat_candidate_profiles(
    path: str | Path, *, source_accession: str
) -> dict[str, Any]:
    """Build an auditable D0-X candidate profile inventory for one RDAT file."""

    source = Path(path)
    raw_inventory = scan_indexed_and_annotation_lines(source)
    try:
        document = parse_rdat(source)
    except RdatParseError as exc:
        raise D0XContractError(str(exc)) from exc
    coordinates = strict_seqpos_to_indices(document["seqpos"])
    offset = int(document["headers"]["OFFSET"])
    profiles: list[dict[str, Any]] = []
    for profile in document["profiles"]:
        resolved = resolve_annotations(
            document["global_annotations"],
            profile["annotation"],
            raw_inventory,
            profile["index"],
        )
        mutation_values = resolved.get("mutation", {}).get("resolved_values", [])
        parsed_mutations = [
            parse_mutation_value(
                value,
                header_sequence=document["headers"]["SEQUENCE"],
                offset=offset,
                profile_sequence=profile.get("profile_sequence"),
            )
            for value in mutation_values
        ]
        provisional_role, exact_status = _profile_role_and_evidence(parsed_mutations)
        first_edit = (
            parsed_mutations[0]["edits"][0]
            if len(parsed_mutations) == 1 and len(parsed_mutations[0]["edits"]) == 1
            else None
        )
        profiles.append(
            {
                "schema_version": D0X_PROFILE_SCHEMA,
                "source_accession": source_accession,
                "source_profile_index": profile["index"],
                "source_file_sha256": document["sha256"],
                "raw_mutation_token": mutation_values,
                "ref_allele": first_edit["ref_allele"] if first_edit else None,
                "alt_allele": first_edit["alt_allele"] if first_edit else None,
                "mutation_coordinate_system": {
                    "source_convention": "RDAT_POSITION_WITH_OFFSET",
                    "source_coordinate_1_based": (
                        first_edit["source_coordinate_1_based"] if first_edit else None
                    ),
                    "source_offset": offset,
                    "sequence_index_0_based": (
                        first_edit["sequence_index_0_based"] if first_edit else None
                    ),
                    "canonicalization_status": "NOT_RUN_D0X_INVENTORY_ONLY",
                },
                "exact_mutation_evidence_status": exact_status,
                "source_to_canonical_retention_status": (
                    "RAW_TOKEN_COORDINATE_AND_OVERRIDE_TRACE_RETAINED_"
                    "CANONICALIZATION_NOT_RUN"
                ),
                "parent_lineage_evidence": "NOT_ASSESSED_D0X",
                "condition_match_evidence": "NOT_ASSESSED_D0X",
                "noise_source": "NOT_ASSESSED_D0X",
                "replicate_block_id": None,
                "measurement_variance": {
                    "reactivity_error_present": profile["reactivity_error"] is not None,
                    "status": "RAW_ERROR_VECTOR_NOT_AGGREGATED_D0X",
                },
                "data_role": None,
                "provisional_data_role": provisional_role,
                "data_role_status": "D0X_CANDIDATE_ONLY_REQUIRES_D1X",
                "exclusion_reason": (
                    None if provisional_role is not None else exact_status
                ),
                "resolved_annotations": resolved,
                "parsed_mutations": parsed_mutations,
                "seqpos_count": len(coordinates),
                "missing_reactivity_count": profile["missing_reactivity_count"],
            }
        )
    return {
        "schema_version": "reactflow_delta.d0x_file_audit.v1",
        "source_accession": source_accession,
        "source_path": str(source.resolve()),
        "source_sha256": document["sha256"],
        "profile_records": profiles,
        "profile_accounting": raw_inventory,
        "scientific_boundary": (
            "D0-X candidate inventory only; no exact pair count, eligibility, "
            "Tier, split, training, or scientific claim."
        ),
    }


def write_create_once_json(path: str | Path, payload: Any) -> None:
    """Write deterministic JSON once; refuse overwrite even with identical bytes."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
