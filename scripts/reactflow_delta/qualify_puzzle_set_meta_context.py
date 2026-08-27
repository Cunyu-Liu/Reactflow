#!/usr/bin/env python3
"""Apply the predeclared top-journal Gate to a complete puzzle-set score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.qualify_model_rescue_v10 import paired_summary
from scripts.reactflow_delta.score_puzzle_set_meta_context import (
    EXPECTED_PHASE,
    EXPECTED_SCORE_TOKEN,
    SCHEMA as SCORE_SCHEMA,
)
from scripts.reactflow_delta.puzzle_set_score_chain import (
    assert_active_phase,
    assert_authority_paths,
    validate_p1_score_rows,
    validate_retention_score_protocol,
)


SCHEMA = "reactflow_delta.puzzle_set_meta_context_qualification.proposed.v2"
EXPECTED_SCORE_TOP_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "scores",
    "context_retention_summary",
    "target_profile_identity",
    "target_join_after_complete_merge",
    "v13_parent_and_feature41_replay_at_5e_7",
    "v13_historical_bundle_protocol_validated",
    "tic2a_registry_cross_linked_to_merged_provenance",
    "partial_fold_scores_inspected",
    "external_outcome_accessed",
    "model_or_threshold_selection_performed",
}
EXPECTED_GATE_FIELDS = {
    "prediction_integrity",
    "candidate_pretraining_established_all_runs",
    "candidate_context_retention_positive_all_runs",
    "retention_protocol_selection_free_and_outcome_blind",
    "signed_gain_vs_feature41_ge_12pct",
    "signed_gain_vs_terminal_v12_ge_2pct",
    "signed_gain_vs_v13_parent_ge_2pct",
    "signed_gain_vs_matched_null_ge_1_5pct",
    "signed_ci_lower_each_gt_zero",
    "signed_positive_puzzles_ge_16_14_14_14",
    "point_absolute_gain_vs_feature41_ge_7pct",
    "point_absolute_gain_vs_terminal_v11_ge_2pct",
    "point_absolute_gain_vs_v13_parent_ge_2pct",
    "point_absolute_gain_vs_matched_null_ge_1pct",
    "point_absolute_ci_lower_each_gt_zero",
    "point_absolute_positive_puzzles_ge_16_14_14_14",
    "task_crps_gain_vs_feature41_ge_5pct",
    "task_crps_gain_vs_terminal_v12_ge_2pct",
    "task_crps_gain_vs_matched_null_ge_1_5pct",
    "task_crps_ci_lower_each_gt_zero",
    "task_crps_positive_puzzles_ge_16_14_14",
    "distribution_absolute_gain_vs_feature41_ge_15pct",
    "distribution_absolute_gain_vs_terminal_v10_ge_2pct",
    "distribution_absolute_gain_vs_matched_null_ge_1pct",
    "distribution_absolute_ci_lower_each_gt_zero",
    "distribution_absolute_positive_puzzles_ge_16_14_14",
    "leave_one_puzzle_out_all_headline_metrics_positive",
    "max_single_puzzle_effect_fraction_le_0_20",
    "coverage_error_guardrail",
}


def _validate_complete_score_protocol(scores: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        set(scores) != EXPECTED_SCORE_TOP_FIELDS
        or scores.get("schema_version") != SCORE_SCHEMA
        or scores.get("phase") != EXPECTED_PHASE
        or scores.get("status") != "PUZZLE_SET_M3_COMPLETE_SCORE_PASS"
        or scores.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION"
        or scores.get("target_join_after_complete_merge") is not True
        or scores.get("v13_parent_and_feature41_replay_at_5e_7") is not True
        or scores.get("v13_historical_bundle_protocol_validated") is not True
        or scores.get("tic2a_registry_cross_linked_to_merged_provenance") is not True
        or scores.get("partial_fold_scores_inspected") is not False
        or scores.get("external_outcome_accessed") is not False
        or scores.get("model_or_threshold_selection_performed") is not False
    ):
        raise ValueError("puzzle-set qualifier requires one exact complete score")
    rows = validate_p1_score_rows(
        scores.get("scores"), source="Puzzle-Set P1M3 complete score"
    )
    validate_retention_score_protocol(
        scores.get("context_retention_summary"), expected_run_count=20
    )
    return rows


def qualify(
    scores: dict[str, Any], *, validate_protocol: bool = True
) -> dict[str, Any]:
    rows = (
        _validate_complete_score_protocol(scores)
        if validate_protocol
        else sorted(scores.get("scores", []), key=lambda row: int(row["outer_fold"]))
    )
    if not validate_protocol and (
        len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20))
    ):
        raise ValueError("puzzle-set qualifier requires unique folds0-19")
    integrity = all(
        float(row["registered_prediction_coverage"]) == 1.0
        and float(row["failure_rate"]) == 0.0
        and int(row["n_unexpected_prediction_keys"]) == 0
        for row in rows
    )
    comparisons = {
        "signed_vs_feature41": paired_summary(
            rows, "feature41_signed_delta_mae", "candidate_signed_delta_mae"
        ),
        "signed_vs_v13_parent": paired_summary(
            rows, "parent_signed_delta_mae", "candidate_signed_delta_mae"
        ),
        "signed_vs_terminal_v12": paired_summary(
            rows, "terminal_v12_signed_delta_mae", "candidate_signed_delta_mae"
        ),
        "signed_vs_matched_null": paired_summary(
            rows, "null_signed_delta_mae", "candidate_signed_delta_mae"
        ),
        "point_absolute_vs_feature41": paired_summary(
            rows,
            "feature41_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "point_absolute_vs_v13_parent": paired_summary(
            rows,
            "parent_point_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "point_absolute_vs_terminal_v11": paired_summary(
            rows,
            "terminal_v11_point_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "point_absolute_vs_matched_null": paired_summary(
            rows,
            "null_point_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "task_crps_vs_feature41": paired_summary(
            rows, "feature41_crps", "candidate_crps"
        ),
        "task_crps_vs_terminal_v12": paired_summary(
            rows, "terminal_v12_crps", "candidate_crps"
        ),
        "task_crps_vs_matched_null": paired_summary(
            rows, "null_crps", "candidate_crps"
        ),
        "distribution_absolute_vs_feature41": paired_summary(
            rows,
            "feature41_absolute_delta_mae",
            "candidate_distribution_absolute_delta_mae",
        ),
        "distribution_absolute_vs_terminal_v10": paired_summary(
            rows,
            "terminal_v10_distribution_absolute_delta_mae",
            "candidate_distribution_absolute_delta_mae",
        ),
        "distribution_absolute_vs_matched_null": paired_summary(
            rows,
            "null_distribution_absolute_delta_mae",
            "candidate_distribution_absolute_delta_mae",
        ),
    }
    calibration = {}
    calibration_gate = True
    for level in (68, 95):
        nominal = level / 100.0
        baseline = float(np.mean([row[f"feature41_coverage{level}"] for row in rows]))
        candidate = float(np.mean([row[f"candidate_coverage{level}"] for row in rows]))
        baseline_error = abs(baseline - nominal)
        candidate_error = abs(candidate - nominal)
        passed = candidate_error <= baseline_error + 0.01
        calibration[f"coverage{level}"] = {
            "nominal": nominal,
            "baseline": baseline,
            "candidate": candidate,
            "baseline_absolute_error": baseline_error,
            "candidate_absolute_error": candidate_error,
            "candidate_not_worse_by_more_than_0_01": passed,
        }
        calibration_gate = calibration_gate and passed

    retention = scores.get("context_retention_summary", {})
    retention_protocol = bool(
        retention.get("selection_performed") is False
        and retention.get("mutant_outcome_used") is False
        and retention.get("held_puzzle_accessed") is False
    )

    signed = (
        comparisons["signed_vs_feature41"],
        comparisons["signed_vs_terminal_v12"],
        comparisons["signed_vs_v13_parent"],
        comparisons["signed_vs_matched_null"],
    )
    point = (
        comparisons["point_absolute_vs_feature41"],
        comparisons["point_absolute_vs_terminal_v11"],
        comparisons["point_absolute_vs_v13_parent"],
        comparisons["point_absolute_vs_matched_null"],
    )
    crps = (
        comparisons["task_crps_vs_feature41"],
        comparisons["task_crps_vs_terminal_v12"],
        comparisons["task_crps_vs_matched_null"],
    )
    distribution = (
        comparisons["distribution_absolute_vs_feature41"],
        comparisons["distribution_absolute_vs_terminal_v10"],
        comparisons["distribution_absolute_vs_matched_null"],
    )
    headline = signed + point + crps + distribution
    gates = {
        "prediction_integrity": integrity,
        "candidate_pretraining_established_all_runs": (
            retention.get("candidate_pretraining_established_all_runs") is True
        ),
        "candidate_context_retention_positive_all_runs": (
            retention.get("candidate_retention_positive_all_runs") is True
        ),
        "retention_protocol_selection_free_and_outcome_blind": retention_protocol,
        "signed_gain_vs_feature41_ge_12pct": signed[0]["relative_gain"] >= 0.12,
        "signed_gain_vs_terminal_v12_ge_2pct": signed[1]["relative_gain"] >= 0.02,
        "signed_gain_vs_v13_parent_ge_2pct": signed[2]["relative_gain"] >= 0.02,
        "signed_gain_vs_matched_null_ge_1_5pct": signed[3]["relative_gain"] >= 0.015,
        "signed_ci_lower_each_gt_zero": all(item["ci95"][0] > 0.0 for item in signed),
        "signed_positive_puzzles_ge_16_14_14_14": (
            signed[0]["positive_puzzles"] >= 16
            and signed[1]["positive_puzzles"] >= 14
            and signed[2]["positive_puzzles"] >= 14
            and signed[3]["positive_puzzles"] >= 14
        ),
        "point_absolute_gain_vs_feature41_ge_7pct": point[0]["relative_gain"] >= 0.07,
        "point_absolute_gain_vs_terminal_v11_ge_2pct": point[1]["relative_gain"]
        >= 0.02,
        "point_absolute_gain_vs_v13_parent_ge_2pct": point[2]["relative_gain"] >= 0.02,
        "point_absolute_gain_vs_matched_null_ge_1pct": point[3]["relative_gain"]
        >= 0.01,
        "point_absolute_ci_lower_each_gt_zero": all(
            item["ci95"][0] > 0.0 for item in point
        ),
        "point_absolute_positive_puzzles_ge_16_14_14_14": (
            point[0]["positive_puzzles"] >= 16
            and point[1]["positive_puzzles"] >= 14
            and point[2]["positive_puzzles"] >= 14
            and point[3]["positive_puzzles"] >= 14
        ),
        "task_crps_gain_vs_feature41_ge_5pct": crps[0]["relative_gain"] >= 0.05,
        "task_crps_gain_vs_terminal_v12_ge_2pct": crps[1]["relative_gain"] >= 0.02,
        "task_crps_gain_vs_matched_null_ge_1_5pct": crps[2]["relative_gain"] >= 0.015,
        "task_crps_ci_lower_each_gt_zero": all(item["ci95"][0] > 0.0 for item in crps),
        "task_crps_positive_puzzles_ge_16_14_14": (
            crps[0]["positive_puzzles"] >= 16
            and crps[1]["positive_puzzles"] >= 14
            and crps[2]["positive_puzzles"] >= 14
        ),
        "distribution_absolute_gain_vs_feature41_ge_15pct": (
            distribution[0]["relative_gain"] >= 0.15
        ),
        "distribution_absolute_gain_vs_terminal_v10_ge_2pct": (
            distribution[1]["relative_gain"] >= 0.02
        ),
        "distribution_absolute_gain_vs_matched_null_ge_1pct": (
            distribution[2]["relative_gain"] >= 0.01
        ),
        "distribution_absolute_ci_lower_each_gt_zero": all(
            item["ci95"][0] > 0.0 for item in distribution
        ),
        "distribution_absolute_positive_puzzles_ge_16_14_14": (
            distribution[0]["positive_puzzles"] >= 16
            and distribution[1]["positive_puzzles"] >= 14
            and distribution[2]["positive_puzzles"] >= 14
        ),
        "leave_one_puzzle_out_all_headline_metrics_positive": all(
            item["leave_one_puzzle_out_all_positive"] for item in headline
        ),
        "max_single_puzzle_effect_fraction_le_0_20": all(
            item["max_single_puzzle_effect_fraction"] <= 0.20 for item in headline
        ),
        "coverage_error_guardrail": calibration_gate,
    }
    if set(gates) != EXPECTED_GATE_FIELDS:
        raise AssertionError("Puzzle-Set frozen Gate field universe changed")
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "P1M3",
        "status": (
            "PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_PASS"
            if passed
            else "PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": comparisons,
        "calibration": calibration,
        "context_retention_summary": retention,
        "target_profile_identity_exact": True,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "POST_HOC_DEVELOPMENT_SCREEN_ONLY",
        "puzzle_set_m4_authorized": passed,
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def assert_qualifier_authority(
    repo_root: Path, *, score_json: Path, out_json: Path
) -> dict[str, Any]:
    active = assert_active_phase(
        repo_root,
        phase=EXPECTED_PHASE,
        score_token=EXPECTED_SCORE_TOKEN,
        training_must_be_closed=True,
    )
    assert_authority_paths(
        active,
        {"complete_score_path": score_json, "qualification_path": out_json},
    )
    return active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    score_json = args.score_json.resolve()
    out_json = args.out_json.resolve()
    assert_qualifier_authority(
        args.repo_root.resolve(), score_json=score_json, out_json=out_json
    )
    if out_json.exists():
        raise FileExistsError("puzzle-set refuses to overwrite its qualification")
    result = qualify(json.loads(score_json.read_text(encoding="utf-8")))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_json.with_name(f"{out_json.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, out_json)
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
