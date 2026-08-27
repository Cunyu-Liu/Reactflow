#!/usr/bin/env python3
"""Apply the pre-score post-V14 first-matching contingency mechanically."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from scripts.reactflow_delta.qualify_model_rescue_v10 import paired_summary
from scripts.reactflow_delta.qualify_model_rescue_v14 import (
    SCHEMA as QUALIFICATION_SCHEMA,
    qualify,
)
from scripts.reactflow_delta.score_model_rescue_v14 import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.post_v14_first_matching_router.v1"
STATUS = "POST_V14_FIRST_MATCHING_ROUTE_SELECTED"
AUDIT_FAILURE_STATUS = "POST_V14_TERMINAL_INPUT_AUDIT_FAILURE"
AUTHORITY_TOKEN = "POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY"
AUTHORITY_ACTION = "RUN_SINGLE_POST_V14_FIRST_MATCHING_ROUTER"
ACTIVE_CONTRACT_SCHEMA = "reactflow_delta.active_contract.v14"
PROJECT_TASK_ID = "reactflow_delta_model_rescue_v14"
FIRST_MATCHING_ORDER = ("1", "2", "3", "4", "5", "6", "P3")
ROUTER_PLAN = "docs/plans/2026-08-27-post-v14-model-contingency.md"
ROUTER_PLAN_FROZEN_COMMIT = "98d8eb519fb7c69d8a489a44ff72380204d6599c"
MACHINE_BINDING = "docs/plans/2026-08-27-post-v14-router-machine-binding.md"
AUTHORITY_MAPPING = "post_v14_router_authority"
CANONICAL_ACTIVE_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "configs/reactflow_delta/active_contract.yaml"
).resolve()


COMPARISON_RULES: dict[str, dict[str, Any]] = {
    "signed_vs_feature41": {
        "baseline_field": "feature41_signed_delta_mae",
        "candidate_field": "candidate_signed_delta_mae",
        "margin": 0.12,
        "positive_puzzles": 16,
    },
    "signed_vs_terminal_v12": {
        "baseline_field": "terminal_v12_signed_delta_mae",
        "candidate_field": "candidate_signed_delta_mae",
        "margin": 0.02,
        "positive_puzzles": 14,
    },
    "signed_vs_nested_null": {
        "baseline_field": "null_signed_delta_mae",
        "candidate_field": "candidate_signed_delta_mae",
        "margin": 0.015,
        "positive_puzzles": 14,
    },
    "point_absolute_vs_feature41": {
        "baseline_field": "feature41_absolute_delta_mae",
        "candidate_field": "candidate_point_absolute_delta_mae",
        "margin": 0.07,
        "positive_puzzles": 16,
    },
    "point_absolute_vs_terminal_v11": {
        "baseline_field": "terminal_v11_point_absolute_delta_mae",
        "candidate_field": "candidate_point_absolute_delta_mae",
        "margin": 0.02,
        "positive_puzzles": 14,
    },
    "point_absolute_vs_nested_null": {
        "baseline_field": "null_point_absolute_delta_mae",
        "candidate_field": "candidate_point_absolute_delta_mae",
        "margin": 0.01,
        "positive_puzzles": 14,
    },
    "task_crps_vs_feature41": {
        "baseline_field": "feature41_crps",
        "candidate_field": "candidate_crps",
        "margin": 0.05,
        "positive_puzzles": 16,
    },
    "task_crps_vs_terminal_v12": {
        "baseline_field": "terminal_v12_crps",
        "candidate_field": "candidate_crps",
        "margin": 0.02,
        "positive_puzzles": 14,
    },
    "task_crps_vs_nested_null": {
        "baseline_field": "null_crps",
        "candidate_field": "candidate_crps",
        "margin": 0.015,
        "positive_puzzles": 14,
    },
    "distribution_absolute_vs_feature41": {
        "baseline_field": "feature41_absolute_delta_mae",
        "candidate_field": "candidate_distribution_absolute_delta_mae",
        "margin": 0.15,
        "positive_puzzles": 16,
    },
    "distribution_absolute_vs_terminal_v10": {
        "baseline_field": "terminal_v10_distribution_absolute_delta_mae",
        "candidate_field": "candidate_distribution_absolute_delta_mae",
        "margin": 0.02,
        "positive_puzzles": 14,
    },
    "distribution_absolute_vs_nested_null": {
        "baseline_field": "null_distribution_absolute_delta_mae",
        "candidate_field": "candidate_distribution_absolute_delta_mae",
        "margin": 0.01,
        "positive_puzzles": 14,
    },
}

EXPECTED_GATE_KEYS = {
    "prediction_integrity",
    "signed_gain_vs_feature41_ge_12pct",
    "signed_gain_vs_terminal_v12_ge_2pct",
    "signed_gain_vs_nested_null_ge_1_5pct",
    "signed_ci_lower_each_gt_zero",
    "signed_positive_puzzles_ge_16_14_14",
    "point_absolute_gain_vs_feature41_ge_7pct",
    "point_absolute_gain_vs_terminal_v11_ge_2pct",
    "point_absolute_gain_vs_nested_null_ge_1pct",
    "point_absolute_ci_lower_each_gt_zero",
    "point_absolute_positive_puzzles_ge_16_14_14",
    "task_crps_gain_vs_feature41_ge_5pct",
    "task_crps_gain_vs_terminal_v12_ge_2pct",
    "task_crps_gain_vs_nested_null_ge_1_5pct",
    "task_crps_ci_lower_each_gt_zero",
    "task_crps_positive_puzzles_ge_16_14_14",
    "distribution_absolute_gain_vs_feature41_ge_15pct",
    "distribution_absolute_gain_vs_terminal_v10_ge_2pct",
    "distribution_absolute_gain_vs_nested_null_ge_1pct",
    "distribution_absolute_ci_lower_each_gt_zero",
    "distribution_absolute_positive_puzzles_ge_16_14_14",
    "leave_one_puzzle_out_all_headline_metrics_positive",
    "max_single_puzzle_effect_fraction_le_0_20",
    "coverage_error_guardrail",
}

SCORE_FIELDS = {
    rule[field]
    for rule in COMPARISON_RULES.values()
    for field in ("baseline_field", "candidate_field")
} | {
    "registered_prediction_coverage",
    "failure_rate",
    "n_unexpected_prediction_keys",
    "feature41_coverage68",
    "candidate_coverage68",
    "feature41_coverage95",
    "candidate_coverage95",
    "n_qualified_positions",
    "n_registered_expected",
    "n_registered_observed",
}

COUNT_FIELDS = {
    "n_qualified_positions",
    "n_registered_expected",
    "n_registered_observed",
    "n_unexpected_prediction_keys",
}
COVERAGE_FIELDS = {
    "registered_prediction_coverage",
    "feature41_coverage68",
    "candidate_coverage68",
    "feature41_coverage95",
    "candidate_coverage95",
}
NONNEGATIVE_METRIC_FIELDS = SCORE_FIELDS - COUNT_FIELDS - COVERAGE_FIELDS - {
    "failure_rate"
}
EXPECTED_SCORE_ROW_FIELDS = SCORE_FIELDS | {"outer_fold", "held_puzzle"}

HISTORICAL_COMPARISONS = (
    "signed_vs_feature41",
    "signed_vs_terminal_v12",
    "point_absolute_vs_feature41",
    "point_absolute_vs_terminal_v11",
    "task_crps_vs_feature41",
    "task_crps_vs_terminal_v12",
    "distribution_absolute_vs_feature41",
    "distribution_absolute_vs_terminal_v10",
)
NULL_COMPARISONS = (
    "signed_vs_nested_null",
    "point_absolute_vs_nested_null",
    "task_crps_vs_nested_null",
    "distribution_absolute_vs_nested_null",
)
HISTORICAL_POINT_COMPARISONS = (
    "signed_vs_feature41",
    "signed_vs_terminal_v12",
    "point_absolute_vs_feature41",
    "point_absolute_vs_terminal_v11",
)
POINT_COMPARISONS = (
    "signed_vs_feature41",
    "signed_vs_terminal_v12",
    "signed_vs_nested_null",
    "point_absolute_vs_feature41",
    "point_absolute_vs_terminal_v11",
    "point_absolute_vs_nested_null",
)
CRPS_COMPARISONS = (
    "task_crps_vs_feature41",
    "task_crps_vs_terminal_v12",
    "task_crps_vs_nested_null",
)
DISTRIBUTION_COMPARISONS = (
    "distribution_absolute_vs_feature41",
    "distribution_absolute_vs_terminal_v10",
    "distribution_absolute_vs_nested_null",
)

NULL_HISTORICAL_RULES = {
    "signed_vs_feature41": (
        "feature41_signed_delta_mae",
        "null_signed_delta_mae",
        0.12,
    ),
    "signed_vs_terminal_v12": (
        "terminal_v12_signed_delta_mae",
        "null_signed_delta_mae",
        0.02,
    ),
    "point_absolute_vs_feature41": (
        "feature41_absolute_delta_mae",
        "null_point_absolute_delta_mae",
        0.07,
    ),
    "point_absolute_vs_terminal_v11": (
        "terminal_v11_point_absolute_delta_mae",
        "null_point_absolute_delta_mae",
        0.02,
    ),
}


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _assert_bound_paths(
    binding: Any,
    *,
    mapping_name: str,
    authority_token: str,
    actual_paths: dict[str, Path],
) -> None:
    expected_fields = {"runtime_authority_token", *actual_paths}
    if not isinstance(binding, dict) or set(binding) != expected_fields:
        raise RuntimeError(f"{mapping_name} field universe changed")
    if binding.get("runtime_authority_token") != authority_token:
        raise RuntimeError(f"{mapping_name} runtime authority token is not issued")
    for field, actual in actual_paths.items():
        declared = binding.get(field)
        if type(declared) is not str or not Path(declared).is_absolute():
            raise RuntimeError(f"{mapping_name} {field} is not an absolute path")
        declared_resolved = Path(declared).resolve()
        if str(declared_resolved) != declared:
            raise RuntimeError(f"{mapping_name} {field} is not canonical")
        if declared_resolved != actual:
            raise RuntimeError(f"{mapping_name} {field} differs from the CLI path")


def _load_canonical_active_contract(active_contract: Path) -> dict[str, Any]:
    if active_contract != CANONICAL_ACTIVE_CONTRACT:
        raise RuntimeError(
            "authority path is not this repository's canonical active contract"
        )
    active = yaml.safe_load(active_contract.read_text(encoding="utf-8"))
    if not isinstance(active, dict):
        raise RuntimeError("active contract root must be a mapping")
    if active.get("schema_version") != ACTIVE_CONTRACT_SCHEMA:
        raise RuntimeError("active contract schema is not V14")
    if active.get("project_task_id") != PROJECT_TASK_ID:
        raise RuntimeError("active contract project task identity changed")
    if active.get("runnable_phases") != ["V14M3"]:
        raise RuntimeError("active contract must expose only V14M3")
    return active


def assert_router_authority(
    active_contract: Path,
    *,
    score_path: Path,
    qualification_path: Path,
    output_path: Path,
) -> None:
    active = _load_canonical_active_contract(active_contract)
    if active.get("authority", {}).get("current_phase") != "V14M3":
        raise RuntimeError("post-V14 router requires terminal V14M3 authority")
    if active.get("training_allowed") is not False or active.get(
        "candidate_model_training_allowed"
    ) is not False:
        raise RuntimeError("post-V14 router requires training closed")
    if active.get("held_score_read_allowed") != AUTHORITY_TOKEN:
        raise RuntimeError("post-V14 router score-read token is not issued")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("post-V14 router partial score access remains prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("post-V14 router external outcome access remains prohibited")
    if active.get("next_allowed_action") != AUTHORITY_ACTION:
        raise RuntimeError("post-V14 router action is not bound")
    _assert_bound_paths(
        active.get(AUTHORITY_MAPPING),
        mapping_name=AUTHORITY_MAPPING,
        authority_token=AUTHORITY_TOKEN,
        actual_paths={
            "complete_score_path": score_path,
            "qualification_path": qualification_path,
            "router_output_path": output_path,
        },
    )


def _validate_score(score: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(score, dict):
        raise ValueError("complete score root must be a mapping")
    if score.get("schema_version") != SCORE_SCHEMA:
        raise ValueError("complete score schema is invalid")
    if score.get("phase") != "V14M3" or score.get("status") != (
        "V14M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("router requires the complete V14M3 score")
    required_top = {
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "terminal_parent_metrics_from_frozen_complete_v12_score": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }
    for name, expected in required_top.items():
        if type(expected) is bool:
            valid = score.get(name) is expected
        else:
            valid = score.get(name) == expected
        if not valid:
            raise ValueError(f"complete score violates {name}")
    rows = score.get("scores")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("complete score must contain twenty puzzle rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("complete score rows must be mappings")
    if any(type(row.get("outer_fold")) is not int for row in rows):
        raise ValueError("complete score outer_fold values must be integers")
    rows = sorted(rows, key=lambda row: row["outer_fold"])
    if [row["outer_fold"] for row in rows] != list(range(20)):
        raise ValueError("complete score must contain unique folds0-19")
    for fold, row in enumerate(rows):
        if set(row) != EXPECTED_SCORE_ROW_FIELDS:
            raise ValueError("complete score row field universe changed")
        if type(row.get("held_puzzle")) is not str or row.get(
            "held_puzzle"
        ) != f"P{fold + 1:02d}":
            raise ValueError("complete score held-puzzle mapping is invalid")
        for name in SCORE_FIELDS:
            if name not in row or not _finite(row[name]):
                raise ValueError(f"complete score has invalid field {name}")
        for name in COUNT_FIELDS:
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError(f"complete score has invalid count {name}")
        if row["n_registered_expected"] <= 0:
            raise ValueError("complete score registered expected count is not positive")
        if row["n_registered_observed"] != row["n_registered_expected"]:
            raise ValueError("complete score registered observed count differs")
        if not 0 < row["n_qualified_positions"] <= row["n_registered_observed"]:
            raise ValueError("complete score qualified position count is invalid")
        for name in NONNEGATIVE_METRIC_FIELDS:
            if float(row[name]) < 0.0:
                raise ValueError(f"complete score has negative metric {name}")
        for name in COVERAGE_FIELDS:
            if not 0.0 <= float(row[name]) <= 1.0:
                raise ValueError(f"complete score has invalid coverage {name}")
        if not 0.0 <= float(row["failure_rate"]) <= 1.0:
            raise ValueError("complete score failure rate is outside zero to one")
        if float(row["registered_prediction_coverage"]) != 1.0:
            raise ValueError("complete score coverage is not one")
        if float(row["failure_rate"]) != 0.0:
            raise ValueError("complete score contains failures")
        if int(row["n_unexpected_prediction_keys"]) != 0:
            raise ValueError("complete score contains unexpected keys")
    return rows


def _validate_summary(
    name: str, summary: dict[str, Any], rule: dict[str, Any]
) -> None:
    if summary.get("baseline_field") != rule["baseline_field"]:
        raise ValueError(f"qualification comparison {name} baseline changed")
    if summary.get("candidate_field") != rule["candidate_field"]:
        raise ValueError(f"qualification comparison {name} candidate changed")
    for field in ("baseline_mean", "candidate_mean", "mean_gain", "relative_gain"):
        if not _finite(summary.get(field)):
            raise ValueError(f"qualification comparison {name} has invalid {field}")
    ci95 = summary.get("ci95")
    per_puzzle = summary.get("per_puzzle")
    leave_one_out = summary.get("leave_one_puzzle_out")
    if not isinstance(ci95, list) or len(ci95) != 2 or not all(map(_finite, ci95)):
        raise ValueError(f"qualification comparison {name} has invalid CI")
    if (
        not isinstance(per_puzzle, list)
        or len(per_puzzle) != 20
        or not all(map(_finite, per_puzzle))
    ):
        raise ValueError(f"qualification comparison {name} has invalid puzzle effects")
    if (
        not isinstance(leave_one_out, list)
        or len(leave_one_out) != 20
        or not all(map(_finite, leave_one_out))
    ):
        raise ValueError(f"qualification comparison {name} has invalid LOO effects")
    positive = summary.get("positive_puzzles")
    if type(positive) is not int or positive < 0 or positive > 20:
        raise ValueError(f"qualification comparison {name} has invalid direction count")
    influence = summary.get("max_single_puzzle_effect_fraction")
    if type(influence) not in (int, float) or math.isnan(float(influence)):
        raise ValueError(f"qualification comparison {name} has invalid influence")
    if type(summary.get("leave_one_puzzle_out_all_positive")) is not bool:
        raise ValueError(f"qualification comparison {name} has invalid LOO status")


def _validate_qualification(
    score: dict[str, Any], qualification: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(qualification, dict):
        raise ValueError("qualification root must be a mapping")
    if qualification.get("schema_version") != QUALIFICATION_SCHEMA:
        raise ValueError("qualification schema is invalid")
    if qualification.get("phase") != "V14M3":
        raise ValueError("router requires a V14M3 qualification")
    gates = qualification.get("gates")
    if not isinstance(gates, dict) or set(gates) != EXPECTED_GATE_KEYS:
        raise ValueError("qualification Gate universe changed")
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError("qualification contains a non-boolean Gate")
    gate_passed = all(value is True for value in gates.values())
    if qualification.get("gate_passed") is not gate_passed:
        raise ValueError("qualification status is inconsistent with its Gates")
    expected_status = (
        "V14M3_TOP_JOURNAL_SCREEN_PASS"
        if gate_passed
        else "V14M3_TOP_JOURNAL_SCREEN_FAIL"
    )
    if qualification.get("status") != expected_status:
        raise ValueError("qualification exact verdict is inconsistent")
    if qualification.get("v14m4_authorized") is not gate_passed:
        raise ValueError("qualification V14M4 authorization is inconsistent")
    if qualification.get("target_profile_identity_exact") is not True:
        raise ValueError("qualification target identity is not exact")
    if qualification.get("model_or_threshold_selection_performed") is not False:
        raise ValueError("qualification reports model or threshold selection")
    comparisons = qualification.get("comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != set(COMPARISON_RULES):
        raise ValueError("qualification comparison universe changed")
    for name, rule in COMPARISON_RULES.items():
        summary = comparisons.get(name)
        if not isinstance(summary, dict):
            raise ValueError(f"qualification comparison {name} is missing")
        _validate_summary(name, summary, rule)
    recomputed = qualify(score)
    if qualification != recomputed:
        raise ValueError(
            "qualification differs from the complete structure recomputed from score"
        )
    return recomputed


def _comparison_stable(name: str, summary: dict[str, Any]) -> bool:
    rule = COMPARISON_RULES[name]
    return bool(
        float(summary["ci95"][0]) > 0.0
        and int(summary["positive_puzzles"]) >= int(rule["positive_puzzles"])
        and summary["leave_one_puzzle_out_all_positive"] is True
        and float(summary["max_single_puzzle_effect_fraction"]) <= 0.20
    )


def _comparison_full(name: str, summary: dict[str, Any]) -> bool:
    return bool(
        float(summary["relative_gain"]) >= float(COMPARISON_RULES[name]["margin"])
        and _comparison_stable(name, summary)
    )


def _empty_primitives() -> dict[str, bool]:
    return {
        "candidate_historical_all_full": False,
        "candidate_null_all_full": False,
        "candidate_historical_point_margins_all": False,
        "null_historical_point_margins_all": False,
        "candidate_point_stability_all": False,
        "candidate_point_all_full": False,
        "candidate_crps_all_full": False,
        "candidate_distribution_absolute_all_full": False,
    }


def _route_metadata(branch: str) -> tuple[str, str, dict[str, str]]:
    rows = {
        "1": (
            "V14M3_TOP_JOURNAL_SCREEN_PASS",
            "OPEN_ONLY_V14M4_FIXED_FIVE_SEED_FORMAL",
            {"requirement": "NOT_APPLICABLE", "status": "NOT_APPLICABLE"},
        ),
        "2": (
            "V14_TERMINAL_INPUT_AUDIT_FAILURE",
            "REPAIR_ONLY_IDENTIFIED_ENGINEERING_FAULT_IN_SAME_FROZEN_V14_UNIVERSE",
            {"requirement": "NOT_APPLICABLE", "status": "NOT_APPLICABLE"},
        ),
        "3": (
            "CAPACITY_WITHOUT_PRETRAINING_INCREMENT",
            "OPEN_P1_SOURCE_PROJECTION_AUTHORITY",
            {"requirement": "NOT_APPLICABLE", "status": "NOT_APPLICABLE"},
        ),
        "4": (
            "PRETRAINING_SIGNAL_INSUFFICIENT_FOR_TRANSFER",
            "OPEN_P1_SOURCE_PROJECTION_AUTHORITY",
            {"requirement": "NOT_APPLICABLE", "status": "NOT_APPLICABLE"},
        ),
        "5": (
            "INDEPENDENT_CONSTRUCT_TRANSFER_LIMITED",
            "OPEN_B5RP0_SOURCE_PROJECTION_ONLY_AUTHORITY",
            {"requirement": "REQUIRED", "status": "NOT_RUN"},
        ),
        "6": (
            "DISTRIBUTION_ONLY_FAILURE",
            "RUN_FROZEN_BRANCH6_TAIL_DIAGNOSTIC",
            {"requirement": "NOT_APPLICABLE", "status": "NOT_APPLICABLE"},
        ),
        "P3": (
            "STOP_MODEL_RESCUE",
            "P3_STOP_MODEL_RESCUE",
            {"requirement": "NOT_APPLICABLE", "status": "NOT_APPLICABLE"},
        ),
    }
    return rows[branch]


def route(
    score: dict[str, Any],
    qualification: dict[str, Any],
    *,
    score_path: str,
    qualification_path: str,
) -> dict[str, Any]:
    audit_errors: list[str] = []
    rows: list[dict[str, Any]] = []
    recomputed_qualification: dict[str, Any] = {}
    try:
        rows = _validate_score(score)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        audit_errors.append(str(error))
    if not audit_errors:
        try:
            recomputed_qualification = _validate_qualification(score, qualification)
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            ZeroDivisionError,
        ) as error:
            audit_errors.append(str(error))

    audit_valid = not audit_errors
    primitives = _empty_primitives()
    null_historical: dict[str, dict[str, Any]] = {}
    valid_pass = False
    valid_fail = False
    if audit_valid:
        comparisons = recomputed_qualification["comparisons"]
        null_historical = {
            name: paired_summary(rows, baseline, candidate)
            for name, (baseline, candidate, _margin) in NULL_HISTORICAL_RULES.items()
        }
        primitives = {
            "candidate_historical_all_full": all(
                _comparison_full(name, comparisons[name])
                for name in HISTORICAL_COMPARISONS
            ),
            "candidate_null_all_full": all(
                _comparison_full(name, comparisons[name]) for name in NULL_COMPARISONS
            ),
            "candidate_historical_point_margins_all": all(
                float(comparisons[name]["relative_gain"])
                >= float(COMPARISON_RULES[name]["margin"])
                for name in HISTORICAL_POINT_COMPARISONS
            ),
            "null_historical_point_margins_all": all(
                float(null_historical[name]["relative_gain"]) >= float(rule[2])
                for name, rule in NULL_HISTORICAL_RULES.items()
            ),
            "candidate_point_stability_all": all(
                _comparison_stable(name, comparisons[name])
                for name in POINT_COMPARISONS
            ),
            "candidate_point_all_full": all(
                _comparison_full(name, comparisons[name])
                for name in POINT_COMPARISONS
            ),
            "candidate_crps_all_full": all(
                _comparison_full(name, comparisons[name]) for name in CRPS_COMPARISONS
            ),
            "candidate_distribution_absolute_all_full": all(
                _comparison_full(name, comparisons[name])
                for name in DISTRIBUTION_COMPARISONS
            ),
        }
        valid_pass = (
            recomputed_qualification["status"] == "V14M3_TOP_JOURNAL_SCREEN_PASS"
        )
        valid_fail = (
            recomputed_qualification["status"] == "V14M3_TOP_JOURNAL_SCREEN_FAIL"
        )

    branch_predicates = {
        "1": audit_valid and valid_pass,
        "2": not audit_valid,
        "3": (
            audit_valid
            and valid_fail
            and primitives["candidate_historical_all_full"]
            and not primitives["candidate_null_all_full"]
        ),
        "4": (
            audit_valid
            and valid_fail
            and primitives["candidate_null_all_full"]
            and not primitives["candidate_historical_point_margins_all"]
        ),
        "5": (
            audit_valid
            and valid_fail
            and (
                (
                    not primitives["candidate_historical_point_margins_all"]
                    and not primitives["null_historical_point_margins_all"]
                )
                or not primitives["candidate_point_stability_all"]
            )
        ),
        "6": (
            audit_valid
            and valid_fail
            and primitives["candidate_point_all_full"]
            and (
                not primitives["candidate_crps_all_full"]
                or not primitives["candidate_distribution_absolute_all_full"]
            )
        ),
        "P3": audit_valid and valid_fail,
    }
    selected = next(
        branch for branch in FIRST_MATCHING_ORDER if branch_predicates[branch]
    )
    classification, next_action, route_probe = _route_metadata(selected)
    score_metadata = score if isinstance(score, dict) else {}
    qualification_metadata = qualification if isinstance(qualification, dict) else {}
    return {
        "schema_version": SCHEMA,
        "phase": "POST_V14R0",
        "status": AUDIT_FAILURE_STATUS if selected == "2" else STATUS,
        "artifact_class": (
            "ENGINEERING_EVIDENCE_AUDIT"
            if selected == "2"
            else "SCIENTIFIC_CONTINGENCY_ROUTE"
        ),
        "scientific_route_selected": selected != "2",
        "router_plan": {
            "path": ROUTER_PLAN,
            "frozen_commit": ROUTER_PLAN_FROZEN_COMMIT,
            "machine_binding_path": MACHINE_BINDING,
        },
        "source_artifacts": {
            "complete_score": {
                "path": score_path,
                "schema_version": score_metadata.get("schema_version"),
                "status": score_metadata.get("status"),
            },
            "qualification": {
                "path": qualification_path,
                "schema_version": qualification_metadata.get("schema_version"),
                "status": qualification_metadata.get("status"),
            },
        },
        "complete_terminal_input": audit_valid,
        "audit_valid": audit_valid,
        "audit_errors": audit_errors,
        "primitives": primitives,
        "null_historical_point_comparisons": null_historical,
        "first_matching_order": list(FIRST_MATCHING_ORDER),
        "branch_predicates": branch_predicates,
        "selected_router_branch_id": selected,
        "route_classification": classification,
        "next_action": next_action,
        "route_probe": route_probe,
        "branch6_residual_diagnostic": {
            "predeclared_before_v14_score": True,
            "primary_statistic": "LOWER_MINUS_UPPER_TAIL_MISS90",
            "source_implementation_commit": (
                "7468f1e066c7e1f80aae326bcadc41f0349f172a"
            ),
            "independent_unit": "HELD_PUZZLE",
            "n_independent_units_required": 20,
            "ci": "TWO_SIDED_STUDENT_T_95_DF19",
            "same_direction_puzzles_required": 14,
            "point_update_allowed": False,
        },
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("post-V14 router refuses to overwrite its selection")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-contract", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--qualification-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    active_contract = args.active_contract.resolve()
    score_json = args.score_json.resolve()
    qualification_json = args.qualification_json.resolve()
    out_json = args.out_json.resolve()
    assert_router_authority(
        active_contract,
        score_path=score_json,
        qualification_path=qualification_json,
        output_path=out_json,
    )
    if out_json.exists():
        raise FileExistsError("post-V14 router refuses to overwrite its selection")
    result = route(
        json.loads(score_json.read_text(encoding="utf-8")),
        json.loads(qualification_json.read_text(encoding="utf-8")),
        score_path=str(score_json),
        qualification_path=str(qualification_json),
    )
    _atomic_write_json(out_json, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_router_branch_id": result["selected_router_branch_id"],
                "route_classification": result["route_classification"],
                "result": str(out_json),
            }
        )
    )
    return 2 if result["selected_router_branch_id"] == "2" else 0


if __name__ == "__main__":
    raise SystemExit(main())
