from __future__ import annotations

import json
from pathlib import Path

import pytest

from reactflow.delta.acceptance import build_d0_acceptance_certificate


def _write(path: Path, value: dict | str) -> Path:
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    return path


def _inputs(tmp_path: Path, *, d1_allowed: bool = False) -> dict[str, Path]:
    return {
        "summary": _write(tmp_path / "summary.json", {"schema_version": "reactflow-delta-data-feasibility-summary-v1", "stage": "D0", "d1_allowed": d1_allowed, "learned_training_started": False, "tier_preassessment": {"highest_currently_supported": "below_Tier_B_audit_only"}, "counts": {"audited_entry_count": 6, "audited_profile_construct_record_count": 5175, "confirmed_true_pair_count": 0, "candidate_pair_count_in_fixture_scope": 0}}),
        "parser": _write(tmp_path / "parser.json", {"schema_version": "reactflow-delta-parser-fixture-results-v1", "stage": "D0"}),
        "report": _write(tmp_path / "report.md", "# report\n"),
        "contract": _write(tmp_path / "contract.md", "# contract\n"),
    }


def test_d0_acceptance_is_explicitly_nonadvancing(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    certificate = build_d0_acceptance_certificate(
        summary_path=paths["summary"], parser_fixture_results_path=paths["parser"], report_path=paths["report"], contract_path=paths["contract"],
        evidence_generating_commit="a" * 40, branch="codex/reactflow-delta-r0", push_status="verified_pushed",
    )
    assert certificate["acceptance_status"] == "complete_gate_not_passed_no_advance"
    assert certificate["decision"]["d1_allowed"] is False
    assert certificate["observed_counts"]["confirmed_true_pair_count"] == 0


def test_d0_acceptance_rejects_attempted_advance(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, d1_allowed=True)
    with pytest.raises(ValueError, match="D1 authorization"):
        build_d0_acceptance_certificate(
            summary_path=paths["summary"], parser_fixture_results_path=paths["parser"], report_path=paths["report"], contract_path=paths["contract"],
            evidence_generating_commit="a" * 40, branch="codex/reactflow-delta-r0", push_status="verified_pushed",
        )
