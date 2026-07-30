"""D0 feasibility synthesis with strict evidence and training boundaries."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


DATA_FEASIBILITY_SUMMARY_SCHEMA_VERSION = "reactflow-delta-data-feasibility-summary-v1"
PARSER_FIXTURE_RESULTS_SCHEMA_VERSION = "reactflow-delta-parser-fixture-results-v1"


def build_d0_feasibility_summary(
    *,
    construct_audit_path: str | Path,
    candidate_registry_path: str | Path,
    matrix_path: str | Path,
    relation_audit_path: str | Path,
    ribonanza_availability_path: str | Path,
    filename_candidate_manifest_path: str | Path,
    rdat_parse_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synthesize D0 evidence without transforming unknown or construct into pair."""

    construct = _load_object(construct_audit_path, "construct audit", "reactflow-delta-rmdb-construct-audit-v1")
    candidate = _load_object(candidate_registry_path, "candidate registry", "reactflow-delta-candidate-pair-registry-v1")
    matrix = _load_object(matrix_path, "source pair matrix", "reactflow-delta-source-pair-matrix-v1")
    relations = _load_object(relation_audit_path, "relation audit", "reactflow-delta-pair-relation-audit-v1")
    ribonanza = _load_object(ribonanza_availability_path, "Ribonanza availability", "reactflow-delta-ribonanza-availability-v1")
    filename_candidates = _load_object(filename_candidate_manifest_path, "filename candidate manifest", "reactflow-delta-rmdb-candidate-manifest-v1")
    parsed = _load_object(rdat_parse_manifest_path, "RDAT parse manifest", "reactflow-delta-rdat-construct-parse-manifest-v1")

    records = _require_list(construct, "construct_records")
    construct_summary = _require_object(construct, "summary")
    candidate_summary = _require_object(candidate, "summary")
    relation_summary = _require_object(relations, "summary")
    matrix_summary = _require_object(matrix, "summary")
    probe_counts = Counter(record.get("probe") if record.get("probe") is not None else "unknown_probe" for record in records)
    known_studies = sorted({record.get("study_id") for record in records if record.get("study_id") is not None})
    known_parents = sorted({record.get("parent_id") for record in records if record.get("parent_id") is not None})
    condition_counts = Counter(record.get("condition_key") for record in records)
    candidate_category_counts = {
        item["candidate_category"]: item["candidate_count"]
        for item in _require_list(filename_candidates, "categories")
        if isinstance(item, dict) and isinstance(item.get("candidate_category"), str) and isinstance(item.get("candidate_count"), int)
    }
    true_pair_count = _require_int(relation_summary, "true_pair")
    if true_pair_count != 0:
        raise ValueError("D0 summary implementation currently requires a separate D1 identity audit for nonzero true pairs")
    d1_allowed = False
    top_uncertainties = [
        "Ribonanza raw data was not acquired in this environment, so same-batch/same-condition single-edit pair count is unknown rather than zero.",
        "The six frozen RMDB fixtures are a narrow audited sample; their RDAT header parent sequences are masked/noncanonical or their mutation endpoints are unspecified.",
        "Study IDs and independently established parent lineage are absent from all 5,175 audited construct records, preventing true-pair provenance.",
    ]
    summary = {
        "schema_version": DATA_FEASIBILITY_SUMMARY_SCHEMA_VERSION,
        "stage": "D0",
        "inputs": {
            "construct_audit": _fingerprint(construct_audit_path), "candidate_registry": _fingerprint(candidate_registry_path),
            "matrix": _fingerprint(matrix_path), "relation_audit": _fingerprint(relation_audit_path),
            "ribonanza_availability": _fingerprint(ribonanza_availability_path), "filename_candidate_manifest": _fingerprint(filename_candidate_manifest_path),
            "rdat_parse_manifest": _fingerprint(rdat_parse_manifest_path),
        },
        "counts": {
            "audited_entry_count": len(_require_list(parsed, "fixtures")),
            "audited_profile_construct_record_count": _require_int(construct_summary, "unique_profile_record_count"),
            "candidate_mutational_filename_entry_count": candidate_category_counts.get("m2_named_candidate", 0),
            "candidate_m2r_filename_entry_count": candidate_category_counts.get("m2r_named_unconfirmed", 0),
            "explicit_wt_profile_count": _require_int(construct_summary, "explicit_wt_profile_count"),
            "confirmed_single_mutant_profile_count": _require_int(construct_summary, "confirmed_single_mutant_profile_count"),
            "single_site_endpoint_unknown_profile_count": _require_int(construct_summary, "single_site_endpoint_unknown_profile_count"),
            "confirmed_true_pair_count": true_pair_count,
            "candidate_pair_count_in_fixture_scope": _require_int(candidate_summary, "rmdb_candidate_pair_count"),
            "confirmed_double_mutant_profile_count": _require_int(construct_summary, "confirmed_double_mutant_profile_count"),
            "same_sequence_replicate_group_count": _require_int(construct_summary, "same_sequence_replicate_group_count"),
            "same_sequence_replicate_profile_count": _require_int(construct_summary, "same_sequence_replicate_profile_count"),
            "explicit_no_edit_profile_count": _require_int(construct_summary, "explicit_no_edit_profile_count"),
            "known_study_count": len(known_studies), "known_parent_count": len(known_parents),
            "observed_condition_stratum_count": len(condition_counts),
        },
        "distributions": {
            "probe_profile_counts": dict(sorted(probe_counts.items())),
            "condition_profile_counts": dict(sorted((key, value) for key, value in condition_counts.items() if isinstance(key, str))),
            "in_vitro_in_vivo": {"unknown_metadata": len(records)},
            "construct_exclusion_reason_counts": candidate_summary.get("construct_exclusion_reason_counts"),
        },
        "pair_state": {
            "ribonanza_same_condition_single_edit_pair_count": ribonanza.get("same_condition_single_edit_pair_count"),
            "ribonanza_pair_count_missing_reason": ribonanza.get("pair_count_missing_reason"),
            "matrix_has_unknown_rows": matrix_summary.get("matrix_has_unknown_rows"),
            "relation_summary": relation_summary,
        },
        "tier_preassessment": {
            "tier_A": "not_supported: zero confirmed true pairs in the six-fixture audit scope",
            "tier_B": "not_assessable: Ribonanza raw table unavailable in current environment",
            "tier_C": "audit_only: parsed public construct/profile observations are not primary intervention truth",
            "highest_currently_supported": "below_Tier_B_audit_only",
        },
        "largest_uncertainties": top_uncertainties,
        "d1_allowed": d1_allowed,
        "d1_block_reasons": ["No confirmed true WT--single-mutant pairs", "Ribonanza eligible-pair count remains unknown", "No study/parent provenance for frozen RMDB fixture profiles"],
        "learned_training_started": False,
        "scientific_boundary": "D0 is an availability audit. These counts are limited to parsed artifacts and do not constitute biological validation, a benchmark result, or a training authorization.",
    }
    parser_results = build_parser_fixture_results(parsed, rdat_parse_manifest_path)
    return summary, parser_results


def build_parser_fixture_results(parsed: dict[str, Any], path: str | Path) -> dict[str, Any]:
    fixtures = _require_list(parsed, "fixtures")
    rows = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("parsed fixture is not an object")
        profiles = _require_list(fixture, "profiles")
        rows.append({"name": _required_string(fixture, "name"), "sha256": _required_string(fixture, "sha256"), "profile_count": len(profiles), "seqpos_count": _require_int(fixture, "seqpos_count"), "missing_reactivity_count": sum(_require_int(profile, "missing_reactivity_count") for profile in profiles)})
    return {
        "schema_version": PARSER_FIXTURE_RESULTS_SCHEMA_VERSION,
        "stage": "D0",
        "input_rdat_parse_manifest": _fingerprint(path),
        "fixture_results": rows,
        "fixture_count": len(rows),
        "scientific_boundary": "Structural parser fixture results only; numeric reactivities were not imputed and fixtures are not pair claims.",
    }


def render_d0_feasibility_report(summary: dict[str, Any]) -> str:
    """Render the required readable D0 report from the machine summary."""

    counts = _require_object(summary, "counts")
    distributions = _require_object(summary, "distributions")
    tier = _require_object(summary, "tier_preassessment")
    lines = [
        "# ReactFlow-Delta D0 data feasibility audit", "",
        "## Scope and scientific boundary", "",
        "This is a public-data feasibility audit only. It reports parsed-artifact evidence; it does not report a trained model, a benchmark result, or biological validation.", "",
        "## Audited counts", "",
        f"- Audited entries: {counts['audited_entry_count']}", f"- Audited profile/construct records: {counts['audited_profile_construct_record_count']}",
        f"- Filename-only mutational candidates: {counts['candidate_mutational_filename_entry_count']} M2-named; {counts['candidate_m2r_filename_entry_count']} M2R-named-unconfirmed.",
        f"- Explicit WT profiles: {counts['explicit_wt_profile_count']}", f"- Confirmed single-mutant profiles: {counts['confirmed_single_mutant_profile_count']}",
        f"- Single-site labels with unknown endpoint: {counts['single_site_endpoint_unknown_profile_count']}", f"- Confirmed true WT--single-mutant pairs: {counts['confirmed_true_pair_count']}",
        f"- Confirmed double/rescue profiles: {counts['confirmed_double_mutant_profile_count']}", f"- Same-sequence replicate groups/profiles: {counts['same_sequence_replicate_group_count']}/{counts['same_sequence_replicate_profile_count']}",
        f"- Explicit no-edit profiles: {counts['explicit_no_edit_profile_count']}", f"- Known studies/parents: {counts['known_study_count']}/{counts['known_parent_count']}",
        "", "## Probe, condition, and observation metadata", "",
        f"- Probe profile counts: {json.dumps(distributions['probe_profile_counts'], sort_keys=True)}", f"- Condition strata: {counts['observed_condition_stratum_count']}",
        f"- In-vitro/in-vivo metadata: {json.dumps(distributions['in_vitro_in_vivo'], sort_keys=True)}", f"- Exclusion reasons: {json.dumps(distributions['construct_exclusion_reason_counts'], sort_keys=True)}", "",
        "## Pair and tier decision", "",
        f"- RMDB fixture-scope candidate pairs: {counts['candidate_pair_count_in_fixture_scope']}",
        "- Ribonanza same-condition single-edit pairs: unknown (raw data was not acquired in this environment).",
        f"- Tier A: {tier['tier_A']}", f"- Tier B: {tier['tier_B']}", f"- Tier C: {tier['tier_C']}", f"- Highest currently supported status: {tier['highest_currently_supported']}",
        f"- Allow D1: {summary['d1_allowed']}", "",
        "## Largest uncertainties", "",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(summary["largest_uncertainties"], 1))
    lines.extend(["", "## Stop rule", "", "D1 and any learned training are blocked. No metric threshold is lowered; resolving the documented data/provenance gaps is required before a new gate decision.", ""])
    return "\n".join(lines)


def write_text_once(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _load_object(path: str | Path, label: str, schema_version: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("schema_version") != schema_version:
        raise ValueError(f"unexpected {label} schema version")
    return document


def _fingerprint(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    return {"path": str(candidate.resolve()), "sha256": sha256(candidate.read_bytes()).hexdigest()}


def _require_list(document: dict[str, Any], key: str) -> list[Any]:
    value = document.get(key)
    if not isinstance(value, list):
        raise ValueError(f"document lacks list {key}")
    return value


def _require_object(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"document lacks object {key}")
    return value


def _required_string(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"document lacks non-empty {key}")
    return value


def _require_int(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"document lacks integer {key}")
    return value
