"""Small fail-closed RDAT 0.34 parser for D0 provenance and construct audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .manifests import sha256_file


RDAT_CONSTRUCT_PARSE_MANIFEST_SCHEMA_VERSION = "reactflow-delta-rdat-construct-parse-manifest-v1"


class RdatParseError(ValueError):
    """Raised when an RDAT record cannot be structurally audited."""


def parse_rdat(path: str | Path) -> dict[str, Any]:
    """Parse RDAT text without imputing missing values or experiment labels."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    headers: dict[str, str] = {}
    comments: list[str] = []
    global_annotations: list[dict[str, list[str]]] = []
    data_annotations: dict[int, dict[str, list[str]]] = {}
    seqpos: list[str] | None = None
    reactivity: dict[int, list[float | None]] = {}
    reactivity_error: dict[int, list[float | None]] = {}

    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        key = fields[0].strip()
        values = [value.strip() for value in fields[1:]]
        if key == "COMMENT":
            comments.append("\t".join(values).strip())
        elif key == "ANNOTATION":
            global_annotations.append(_annotation_map(values))
        elif key.startswith("ANNOTATION_DATA:"):
            index = _parse_index(key, line_number)
            if index in data_annotations:
                raise RdatParseError(f"duplicate ANNOTATION_DATA index {index}")
            data_annotations[index] = _annotation_map(values)
        elif key == "SEQPOS":
            if seqpos is not None:
                raise RdatParseError("duplicate SEQPOS")
            if not values:
                raise RdatParseError("SEQPOS is empty")
            seqpos = values
        elif key.startswith("REACTIVITY_ERROR:"):
            index = _parse_index(key, line_number)
            if index in reactivity_error:
                raise RdatParseError(f"duplicate REACTIVITY_ERROR index {index}")
            reactivity_error[index] = _numeric_values(values, key)
        elif key.startswith("REACTIVITY:"):
            index = _parse_index(key, line_number)
            if index in reactivity:
                raise RdatParseError(f"duplicate REACTIVITY index {index}")
            reactivity[index] = _numeric_values(values, key)
        elif key in {"RDAT_VERSION", "NAME", "SEQUENCE", "STRUCTURE", "OFFSET"}:
            if key in headers:
                raise RdatParseError(f"duplicate header {key}")
            if len(values) != 1 or not values[0]:
                raise RdatParseError(f"header {key} requires one non-empty value")
            headers[key] = values[0]
        elif key.startswith(("TRACE:", "READS:")):
            continue
        else:
            headers.setdefault(f"unknown:{key}", "\t".join(values))

    if headers.get("RDAT_VERSION") != "0.34":
        raise RdatParseError("only RDAT_VERSION 0.34 is accepted in D0")
    for required in ("NAME", "SEQUENCE", "OFFSET"):
        if required not in headers:
            raise RdatParseError(f"missing required header {required}")
    if seqpos is None:
        raise RdatParseError("missing SEQPOS")
    if not reactivity:
        raise RdatParseError("missing REACTIVITY rows")

    profiles = []
    for index in sorted(reactivity):
        values = reactivity[index]
        if len(values) != len(seqpos):
            raise RdatParseError(f"REACTIVITY:{index} length does not match SEQPOS")
        errors = reactivity_error.get(index)
        if errors is not None and len(errors) != len(seqpos):
            raise RdatParseError(f"REACTIVITY_ERROR:{index} length does not match SEQPOS")
        profiles.append(
            {
                "index": index,
                "annotation": data_annotations.get(index, {}),
                "reactivity": values,
                "reactivity_error": errors,
                "missing_reactivity_count": sum(value is None for value in values),
            }
        )
    orphan_annotation_indices = sorted(set(data_annotations) - set(reactivity))
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "headers": headers,
        "comments": comments,
        "global_annotations": global_annotations,
        "seqpos": seqpos,
        "profiles": profiles,
        "orphan_annotation_indices": orphan_annotation_indices,
    }


def _parse_index(key: str, line_number: int) -> int:
    try:
        index = int(key.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RdatParseError(f"invalid indexed key at line {line_number}: {key}") from exc
    if index < 1:
        raise RdatParseError(f"indexed key must be positive: {key}")
    return index


def _annotation_map(values: list[str]) -> dict[str, list[str]]:
    """Preserve every repeated RDAT annotation value in source order."""

    parsed: dict[str, list[str]] = {}
    for token in values:
        if not token:
            continue
        if ":" not in token:
            raise RdatParseError(f"annotation token lacks colon: {token!r}")
        key, value = token.split(":", 1)
        if not key or not value:
            raise RdatParseError(f"annotation token has empty key or value: {token!r}")
        parsed.setdefault(key, []).append(value)
    return parsed


def _numeric_values(values: list[str], key: str) -> list[float | None]:
    result: list[float | None] = []
    for value in values:
        if value.lower() == "nan":
            result.append(None)
            continue
        try:
            number = float(value)
        except ValueError as exc:
            raise RdatParseError(f"non-numeric value in {key}: {value!r}") from exc
        if not math.isfinite(number):
            raise RdatParseError(f"non-finite non-NaN value in {key}: {value!r}")
        result.append(number)
    return result


def build_rdat_construct_parse_manifest(fixture_manifest_path: str | Path) -> dict[str, Any]:
    """Create a construct-level audit view while retaining reactivity only in raw RDAT.

    Each fixture must remain byte-identical to the checked-in fixture manifest.
    Profile annotations are retained exactly, including repeated annotation values;
    numeric reactivities are deliberately not copied or imputed in this D0 artifact.
    """

    path = Path(fixture_manifest_path)
    with path.open(encoding="utf-8") as handle:
        fixture_manifest = json.load(handle)
    if not isinstance(fixture_manifest, dict):
        raise RdatParseError("fixture manifest must be a JSON object")
    if fixture_manifest.get("schema_version") != "reactflow-delta-rdat-fixture-manifest-v1":
        raise RdatParseError("unexpected fixture manifest schema version")
    fixtures = fixture_manifest.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise RdatParseError("fixture manifest must contain a non-empty fixtures list")

    parsed_fixtures = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise RdatParseError("fixture manifest contains a non-object fixture")
        for required in ("name", "path", "sha256", "candidate_category", "status"):
            if not isinstance(fixture.get(required), str) or not fixture[required]:
                raise RdatParseError(f"fixture lacks non-empty {required}")
        if fixture["status"] != "verified_against_release_index":
            raise RdatParseError(f"fixture is not byte-verified: {fixture['name']}")
        document = parse_rdat(fixture["path"])
        if document["sha256"] != fixture["sha256"]:
            raise RdatParseError(f"fixture checksum no longer matches manifest: {fixture['name']}")
        parsed_fixtures.append(
            {
                "name": fixture["name"],
                "path": fixture["path"],
                "sha256": fixture["sha256"],
                "candidate_category": fixture["candidate_category"],
                "rdat_version": document["headers"]["RDAT_VERSION"],
                "rdat_name": document["headers"]["NAME"],
                "sequence_length": len(document["headers"]["SEQUENCE"]),
                "seqpos_count": len(document["seqpos"]),
                "global_annotations": document["global_annotations"],
                "profiles": [
                    {
                        "index": profile["index"],
                        "annotation": profile["annotation"],
                        "missing_reactivity_count": profile["missing_reactivity_count"],
                        "reactivity_error_present": profile["reactivity_error"] is not None,
                    }
                    for profile in document["profiles"]
                ],
                "orphan_annotation_indices": document["orphan_annotation_indices"],
            }
        )

    return {
        "schema_version": RDAT_CONSTRUCT_PARSE_MANIFEST_SCHEMA_VERSION,
        "stage": "D0",
        "input_fixture_manifest": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "fixtures": parsed_fixtures,
        "fixture_count": len(parsed_fixtures),
        "scientific_boundary": (
            "This is a byte-verified RDAT structural parse only. Candidate categories remain unconfirmed, "
            "numeric reactivities remain only in immutable raw RDAT, and no construct, pair, tier, or model claim is made."
        ),
    }
