"""Fail-closed construct auditing before any ReactFlow-Delta pair registry."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from .manifests import sha256_file
from .rdat import RdatParseError, parse_rdat


RMDB_CONSTRUCT_AUDIT_SCHEMA_VERSION = "reactflow-delta-rmdb-construct-audit-v1"
_EXACT_MUTATION = re.compile(r"^[ACGU][1-9][0-9]*[ACGU]$")
_UNSPECIFIED_ENDPOINT_MUTATION = re.compile(r"^[ACGU][1-9][0-9]*X$")
_RNA_SEQUENCE = re.compile(r"^[ACGU]+$")


def classify_mutation_labels(values: list[str]) -> dict[str, Any]:
    """Classify only explicit, machine-readable mutation labels.

    An ``X`` endpoint identifies an edited site but does not identify a mutant
    sequence, so it cannot serve as an endpoint for a true WT--mutant pair.
    """

    if not values:
        return {"mutation_class": "mutation_annotation_missing", "edit_count": None, "endpoint_identity": "unknown"}
    if len(values) != 1:
        return {"mutation_class": "mutation_annotation_ambiguous", "edit_count": None, "endpoint_identity": "unknown"}
    label = values[0].strip()
    if label == "WT":
        return {"mutation_class": "explicit_wt", "edit_count": 0, "endpoint_identity": "not_applicable"}
    tokens = [token.strip() for token in re.split(r"[,;+]", label) if token.strip()]
    if not tokens:
        return {"mutation_class": "mutation_annotation_unparseable", "edit_count": None, "endpoint_identity": "unknown"}
    if all(_EXACT_MUTATION.fullmatch(token) for token in tokens):
        if len(tokens) == 1:
            return {"mutation_class": "single_exact_endpoint", "edit_count": 1, "endpoint_identity": "explicit"}
        if len(tokens) == 2:
            return {"mutation_class": "double_exact_endpoint", "edit_count": 2, "endpoint_identity": "explicit"}
        return {"mutation_class": "multi_exact_endpoint", "edit_count": len(tokens), "endpoint_identity": "explicit"}
    if all(_UNSPECIFIED_ENDPOINT_MUTATION.fullmatch(token) for token in tokens):
        if len(tokens) == 1:
            return {"mutation_class": "single_site_endpoint_unknown", "edit_count": 1, "endpoint_identity": "unknown"}
        if len(tokens) == 2:
            return {"mutation_class": "double_site_endpoint_unknown", "edit_count": 2, "endpoint_identity": "unknown"}
        return {"mutation_class": "multi_site_endpoint_unknown", "edit_count": len(tokens), "endpoint_identity": "unknown"}
    return {"mutation_class": "mutation_annotation_unparseable", "edit_count": None, "endpoint_identity": "unknown"}


def build_rmdb_construct_audit(
    fixture_manifest_path: str | Path,
    construct_parse_manifest_path: str | Path,
) -> dict[str, Any]:
    """Audit parsed RMDB profiles without treating constructs as pairs."""

    fixture_manifest = _load_object(fixture_manifest_path, "fixture manifest")
    construct_manifest = _load_object(construct_parse_manifest_path, "construct parse manifest")
    if fixture_manifest.get("schema_version") != "reactflow-delta-rdat-fixture-manifest-v1":
        raise RdatParseError("unexpected fixture manifest schema version")
    if construct_manifest.get("schema_version") != "reactflow-delta-rdat-construct-parse-manifest-v1":
        raise RdatParseError("unexpected construct parse manifest schema version")
    fixtures = fixture_manifest.get("fixtures")
    parsed_fixtures = construct_manifest.get("fixtures")
    if not isinstance(fixtures, list) or not isinstance(parsed_fixtures, list):
        raise RdatParseError("fixture and construct parse manifests must contain fixtures lists")

    parsed_by_identity = _index_parsed_fixtures(parsed_fixtures)
    records: list[dict[str, Any]] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise RdatParseError("fixture manifest contains a non-object fixture")
        name = _required_string(fixture, "name", "fixture")
        fixture_sha = _required_string(fixture, "sha256", name)
        raw_path = _required_string(fixture, "path", name)
        parsed = parsed_by_identity.get((name, fixture_sha))
        if parsed is None:
            raise RdatParseError(f"no matching parsed fixture for {name}")
        raw_document = parse_rdat(raw_path)
        if raw_document["sha256"] != fixture_sha:
            raise RdatParseError(f"fixture checksum no longer matches manifest: {name}")
        header_sequence = raw_document["headers"]["SEQUENCE"]
        parent_status = "rdat_header_sequence_available_but_lineage_unprovenanced" if _RNA_SEQUENCE.fullmatch(header_sequence) else "rdat_header_sequence_masked_or_noncanonical"
        global_annotations = parsed.get("global_annotations")
        if not isinstance(global_annotations, list):
            raise RdatParseError(f"parsed fixture lacks global annotations: {name}")
        condition_key = _stable_digest(global_annotations)
        probe_values = _annotation_values_from_lines(global_annotations, "modifier")
        experiment_values = _annotation_values_from_lines(global_annotations, "experimentType")
        probe = probe_values[0] if len(probe_values) == 1 else None
        experiment_type = experiment_values[0] if len(experiment_values) == 1 else None
        profiles = parsed.get("profiles")
        if not isinstance(profiles, list):
            raise RdatParseError(f"parsed fixture lacks profiles: {name}")
        seen_indices: set[int] = set()
        for profile in profiles:
            if not isinstance(profile, dict):
                raise RdatParseError(f"non-object profile in {name}")
            index = profile.get("index")
            if not isinstance(index, int) or index < 1 or index in seen_indices:
                raise RdatParseError(f"missing, invalid, or duplicate profile index in {name}")
            seen_indices.add(index)
            annotation = profile.get("annotation")
            if not isinstance(annotation, dict):
                raise RdatParseError(f"profile annotation is not an object in {name}:{index}")
            mutation_values = _annotation_values(annotation, "mutation")
            sequence_values = _annotation_values(annotation, "sequence")
            mutation = classify_mutation_labels(mutation_values)
            profile_sequence = sequence_values[0] if len(sequence_values) == 1 and _RNA_SEQUENCE.fullmatch(sequence_values[0]) else None
            records.append(
                {
                    "construct_record_id": f"rmdb:{fixture_sha}:{index}",
                    "source": "RMDB",
                    "entry_id": name,
                    "entry_sha256": fixture_sha,
                    "profile_index": index,
                    "study_id": None,
                    "study_id_missing_reason": "fixture metadata does not supply a study identifier",
                    "parent_id": None,
                    "parent_id_missing_reason": parent_status,
                    "probe": probe,
                    "probe_missing_reason": None if probe is not None else "global modifier annotation is absent or ambiguous",
                    "condition_key": condition_key,
                    "condition_annotations": global_annotations,
                    "experiment_type": experiment_type,
                    "experiment_type_missing_reason": None if experiment_type is not None else "global experimentType annotation is absent or ambiguous",
                    "mutation_labels": mutation_values,
                    **mutation,
                    "profile_sequence": profile_sequence,
                    "profile_sequence_missing_reason": None if profile_sequence is not None else "profile sequence annotation is absent, ambiguous, or noncanonical",
                    "missing_reactivity_count": profile.get("missing_reactivity_count"),
                    "reactivity_error_present": profile.get("reactivity_error_present"),
                    "pair_eligibility": False,
                    "pair_ineligibility_reason": _pair_ineligibility_reason(mutation, parent_status),
                }
            )

    record_ids = [record["construct_record_id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise RdatParseError("construct audit would double-count a profile")
    replicate_groups = _same_sequence_replicate_groups(records)
    mutation_counts = Counter(record["mutation_class"] for record in records)
    return {
        "schema_version": RMDB_CONSTRUCT_AUDIT_SCHEMA_VERSION,
        "stage": "D0",
        "input_fixture_manifest": {"path": str(Path(fixture_manifest_path).resolve()), "sha256": sha256_file(fixture_manifest_path)},
        "input_construct_parse_manifest": {"path": str(Path(construct_parse_manifest_path).resolve()), "sha256": sha256_file(construct_parse_manifest_path)},
        "construct_records": records,
        "replicate_groups": replicate_groups,
        "summary": {
            "profile_record_count": len(records),
            "unique_profile_record_count": len(set(record_ids)),
            "mutation_class_counts": dict(sorted(mutation_counts.items())),
            "explicit_wt_profile_count": mutation_counts["explicit_wt"],
            "confirmed_single_mutant_profile_count": mutation_counts["single_exact_endpoint"],
            "single_site_endpoint_unknown_profile_count": mutation_counts["single_site_endpoint_unknown"],
            "confirmed_double_mutant_profile_count": mutation_counts["double_exact_endpoint"],
            "explicit_no_edit_profile_count": mutation_counts["explicit_wt"],
            "same_sequence_replicate_group_count": len(replicate_groups),
            "same_sequence_replicate_profile_count": sum(group["profile_count"] for group in replicate_groups),
            "pair_eligible_profile_count": sum(record["pair_eligibility"] for record in records),
        },
        "scientific_boundary": (
            "Construct/profile auditing only. A masked or unprovenanced parent, an unspecified mutation endpoint, "
            "or a missing mutation label makes a record ineligible for a true WT--mutant pair. Same-sequence repeats "
            "are replicate evidence, not mutation pairs."
        ),
    }


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise RdatParseError(f"{label} must be a JSON object")
    return document


def _index_parsed_fixtures(parsed_fixtures: list[Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for fixture in parsed_fixtures:
        if not isinstance(fixture, dict):
            raise RdatParseError("construct parse manifest contains a non-object fixture")
        name = _required_string(fixture, "name", "parsed fixture")
        digest = _required_string(fixture, "sha256", name)
        identity = (name, digest)
        if identity in result:
            raise RdatParseError(f"duplicate parsed fixture identity: {name}")
        result[identity] = fixture
    return result


def _required_string(document: dict[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise RdatParseError(f"missing non-empty {key} in {label}")
    return value


def _annotation_values(annotation: dict[str, Any], key: str) -> list[str]:
    values = annotation.get(key, [])
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise RdatParseError(f"annotation {key} must be a list of non-empty strings")
    return values


def _annotation_values_from_lines(lines: list[Any], key: str) -> list[str]:
    values: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            raise RdatParseError("global annotation line is not an object")
        values.extend(_annotation_values(line, key))
    return values


def _stable_digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _pair_ineligibility_reason(mutation: dict[str, Any], parent_status: str) -> str:
    if parent_status != "rdat_header_sequence_available_but_lineage_unprovenanced":
        return "parent sequence is masked or noncanonical"
    if mutation["mutation_class"] != "single_exact_endpoint":
        return "no explicit single-mutant endpoint"
    return "parent lineage is not independently established"


def _same_sequence_replicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        sequence = record["profile_sequence"]
        if sequence is not None:
            grouped[(record["entry_sha256"], sequence)].append(record["construct_record_id"])
    groups = []
    for (entry_sha256, sequence), record_ids in sorted(grouped.items()):
        if len(record_ids) > 1:
            groups.append(
                {
                    "replicate_group_id": f"same-sequence:{entry_sha256}:{sha256(sequence.encode()).hexdigest()}",
                    "entry_sha256": entry_sha256,
                    "profile_record_ids": sorted(record_ids),
                    "profile_count": len(record_ids),
                    "relation": "same annotated sequence within one RDAT fixture and its global condition",
                    "pair_status": "replicate_only_not_mutation_pair",
                }
            )
    return groups
