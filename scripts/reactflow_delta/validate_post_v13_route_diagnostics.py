#!/usr/bin/env python3
"""Validate the narrow post-V13 route-diagnostic authority."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


EXPECTED_PARENT_HEAD = "90a8c6372d354126b49c4406a88c44a59b0427ff"
EXPECTED_BRANCH = "codex/reactflow-delta-post-v13-diagnostics-20260827"


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def validate_contract(repo_root: Path) -> dict[str, Any]:
    active = _yaml(repo_root / "configs/reactflow_delta/active_contract.yaml")
    machine = _yaml(
        repo_root / "configs/reactflow_delta/post_v13_route_diagnostics.yaml"
    )
    ledger = _yaml(
        repo_root / "docs/prospective_v2/post_v13_route_diagnostics_ledger.yaml"
    )
    v13 = _yaml(repo_root / "configs/reactflow_delta/model_rescue_v13_amendment.yaml")
    v13_ledger = _yaml(
        repo_root / "docs/prospective_v2/model_rescue_v13_decision_ledger.yaml"
    )

    if v13.get("contract_status") != "TERMINAL_V13M3_TOP_JOURNAL_SCREEN_FAIL":
        raise RuntimeError("post-V13 diagnostics changed the V13 terminal contract")
    if v13_ledger.get("current_status") != (
        "TERMINAL_V13M3_TOP_JOURNAL_SCREEN_FAIL_EXACT_MUTANT_REENCODING_CLOSED"
    ):
        raise RuntimeError("post-V13 diagnostics changed the V13 terminal ledger")
    if machine.get("parent", {}).get("parent_head") != EXPECTED_PARENT_HEAD:
        raise RuntimeError("post-V13 diagnostics are not anchored to the terminal V13 head")
    if active.get("authority", {}).get("branch") != EXPECTED_BRANCH:
        raise RuntimeError("post-V13 diagnostics active branch is not the frozen branch")
    if active.get("parent_state", {}).get("v13_head") != EXPECTED_PARENT_HEAD:
        raise RuntimeError("active authority parent head differs from the machine contract")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial diagnostic score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("external outcomes must remain closed")
    if active.get("v13_terminal_verdict_change_allowed") is not False:
        raise RuntimeError("V13 terminal verdict mutation must remain prohibited")

    phase = active.get("authority", {}).get("current_phase")
    held = active.get("held_score_read_allowed")
    if held is not False:
        if (
            phase != "PV13D3"
            or held != "PV13D_COMPLETE_MERGE_SCORE_ONCE_ONLY"
            or active.get("training_allowed") is not False
            or active.get("candidate_model_training_allowed") is not False
        ):
            raise RuntimeError("held-score authority exceeds the one complete diagnostic score")

    arms = machine.get("diagnostic_arms", {})
    noise = arms.get("noise_aware", {})
    coherent = arms.get("coherent_sign_magnitude", {})
    if noise.get("features") != "IDENTICAL_FEATURE41":
        raise RuntimeError("noise-aware diagnostic changed the feature universe")
    if noise.get("observation_variance") != (
        "MUTANT_ERROR_SQUARED_PLUS_WT_ERROR_SQUARED_PLUS_0_05_SQUARED"
    ):
        raise RuntimeError("noise-aware diagnostic changed the frozen variance")
    if noise.get("normalization") != (
        "MEAN_ONE_WITHIN_EACH_MUTANT_THEN_EXISTING_CELL_MUTANT_WEIGHT"
    ):
        raise RuntimeError("noise-aware diagnostic changed its weighting estimand")
    if coherent.get("equation") != (
        "SIGNED_POINT_EQUALS_SIGN_SIGNED_TIMES_MAX_ABSOLUTE_ZERO"
    ):
        raise RuntimeError("coherent diagnostic changed its frozen reconstruction")

    gates = machine.get("route_gates", {})
    if gates.get("noise_aware_supported") != {
        "signed_relative_gain_min": 0.005,
        "point_absolute_relative_gain_min": 0.005,
        "signed_and_point_ci_lower_gt": 0.0,
        "signed_and_point_positive_puzzles_min": 14,
    }:
        raise RuntimeError("noise-aware diagnostic Gate changed")
    if gates.get("coherent_factorization_supported") != {
        "signed_relative_gain_min": 0.005,
        "point_absolute_relative_gain_min": 0.01,
        "signed_and_point_ci_lower_gt": 0.0,
        "signed_and_point_positive_puzzles_min": 14,
    }:
        raise RuntimeError("coherent diagnostic Gate changed")
    if gates.get("both_pass_margin_terms") != [
        "SIGNED_RELATIVE_GAIN_DIVIDED_BY_ITS_MINIMUM",
        "POINT_ABSOLUTE_RELATIVE_GAIN_DIVIDED_BY_ITS_MINIMUM",
        "SIGNED_POSITIVE_PUZZLES_DIVIDED_BY_14",
        "POINT_ABSOLUTE_POSITIVE_PUZZLES_DIVIDED_BY_14",
    ] or gates.get("ci_terms_in_both_pass_margin") != (
        "BINARY_PASS_PREREQUISITE_NOT_NUMERIC_MARGIN"
    ):
        raise RuntimeError("post-V13 deterministic both-pass rule changed")

    if ledger.get("immutable_parent_verdicts", {}).get("v13") != (
        "TERMINAL_V13M3_TOP_JOURNAL_SCREEN_FAIL"
    ):
        raise RuntimeError("diagnostic ledger does not preserve V13")

    return {
        "status": "POST_V13_ROUTE_DIAGNOSTIC_CONTRACT_VALIDATION_PASS",
        "phase": phase,
        "training_allowed": active.get("training_allowed"),
        "held_score_read_allowed": held,
        "external_outcome_access_allowed": active.get(
            "new_external_outcome_access_allowed"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = validate_contract(args.repo_root.resolve())
    for key in sorted(result):
        print(f"{key}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
