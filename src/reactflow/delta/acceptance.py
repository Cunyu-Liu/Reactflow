"""Fail-closed D0 acceptance certificate construction."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


D0_ACCEPTANCE_SCHEMA_VERSION = "reactflow-delta-d0-acceptance-v1"


def build_d0_acceptance_certificate(
    *,
    summary_path: str | Path,
    parser_fixture_results_path: str | Path,
    report_path: str | Path,
    contract_path: str | Path,
    evidence_generating_commit: str,
    branch: str,
    push_status: str,
) -> dict[str, Any]:
    """Build a non-advancing D0 certificate from immutable evidence files."""

    summary = _load_object(summary_path, "reactflow-delta-data-feasibility-summary-v1")
    parser_results = _load_object(parser_fixture_results_path, "reactflow-delta-parser-fixture-results-v1")
    if summary.get("stage") != "D0" or parser_results.get("stage") != "D0":
        raise ValueError("acceptance inputs do not identify D0")
    if summary.get("d1_allowed") is not False:
        raise ValueError("D0 acceptance refuses a non-false D1 authorization")
    if summary.get("learned_training_started") is not False:
        raise ValueError("D0 acceptance refuses an already-started learned-training claim")
    tier = _require_object(summary, "tier_preassessment")
    if tier.get("highest_currently_supported") != "below_Tier_B_audit_only":
        raise ValueError("D0 certificate requires the observed fail-closed tier result")
    if not _is_commit_sha(evidence_generating_commit):
        raise ValueError("evidence-generating commit must be a 40-character lowercase hexadecimal SHA")
    if not branch or push_status != "verified_pushed":
        raise ValueError("acceptance requires a named branch and verified push status")
    counts = _require_object(summary, "counts")
    return {
        "schema_version": D0_ACCEPTANCE_SCHEMA_VERSION,
        "stage": "D0",
        "acceptance_status": "complete_gate_not_passed_no_advance",
        "decision": {
            "d0_execution_complete": True,
            "d1_allowed": False,
            "learned_training_started": False,
            "scientific_result": "not_a_final_scientific_conclusion",
            "next_stage": "blocked_pending_new_data_and_provenance_evidence",
        },
        "observed_counts": {
            "audited_entry_count": _require_int(counts, "audited_entry_count"),
            "audited_profile_construct_record_count": _require_int(counts, "audited_profile_construct_record_count"),
            "confirmed_true_pair_count": _require_int(counts, "confirmed_true_pair_count"),
            "candidate_pair_count_in_fixture_scope": _require_int(counts, "candidate_pair_count_in_fixture_scope"),
        },
        "inputs": {
            "summary": _fingerprint(summary_path),
            "parser_fixture_results": _fingerprint(parser_fixture_results_path),
            "report": _fingerprint(report_path),
            "contract": _fingerprint(contract_path),
        },
        "version_control": {
            "evidence_generating_commit": evidence_generating_commit,
            "branch": branch,
            "push_status": push_status,
        },
        "scientific_boundary": "D0 is an availability and provenance audit. It neither trains a model nor establishes a benchmark or biological result.",
    }


def _load_object(path: str | Path, schema_version: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("schema_version") != schema_version:
        raise ValueError("unexpected acceptance input schema version")
    return document


def _fingerprint(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    return {"path": str(candidate.resolve()), "sha256": sha256(candidate.read_bytes()).hexdigest()}


def _require_object(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"document lacks object {key}")
    return value


def _require_int(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"document lacks integer {key}")
    return value


def _is_commit_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)
