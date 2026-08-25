#!/usr/bin/env python3
"""Apply every frozen Model Rescue v12 top-journal screen Gate mechanically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.qualify_model_rescue_v10 import paired_summary
from scripts.reactflow_delta.score_model_rescue_v12 import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v12_qualification.v1"


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V12M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("V12 qualifier requires one complete V12M3 score artifact")
    rows = sorted(scores.get("scores", []), key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("V12 qualifier requires unique folds0-19")
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
        "signed_vs_parent_v11": paired_summary(
            rows, "parent_v11_signed_delta_mae", "candidate_signed_delta_mae"
        ),
        "point_absolute_vs_feature41": paired_summary(
            rows,
            "feature41_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "point_absolute_vs_parent_v11": paired_summary(
            rows,
            "parent_v11_point_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "task_crps_vs_feature41": paired_summary(
            rows, "feature41_crps", "candidate_crps"
        ),
        "task_crps_vs_parent_v11": paired_summary(
            rows, "parent_v11_crps", "candidate_crps"
        ),
        "task_crps_vs_terminal_v10": paired_summary(
            rows, "historical_v10_crps", "candidate_crps"
        ),
        "distribution_absolute_vs_feature41": paired_summary(
            rows,
            "feature41_absolute_delta_mae",
            "candidate_distribution_absolute_delta_mae",
        ),
        "distribution_absolute_vs_terminal_v10": paired_summary(
            rows,
            "historical_v10_distribution_absolute_delta_mae",
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
        passed = candidate_error <= baseline_error + 0.02
        calibration[f"coverage{level}"] = {
            "nominal": nominal,
            "baseline": baseline,
            "candidate": candidate,
            "baseline_absolute_error": baseline_error,
            "candidate_absolute_error": candidate_error,
            "candidate_not_worse_by_more_than_0_02": passed,
        }
        calibration_gate = calibration_gate and passed

    signed_f = comparisons["signed_vs_feature41"]
    signed_parent = comparisons["signed_vs_parent_v11"]
    point_f = comparisons["point_absolute_vs_feature41"]
    point_parent = comparisons["point_absolute_vs_parent_v11"]
    crps_f = comparisons["task_crps_vs_feature41"]
    crps_parent = comparisons["task_crps_vs_parent_v11"]
    crps_v10 = comparisons["task_crps_vs_terminal_v10"]
    dist_f = comparisons["distribution_absolute_vs_feature41"]
    dist_v10 = comparisons["distribution_absolute_vs_terminal_v10"]
    headline = tuple(comparisons.values())
    gates = {
        "prediction_integrity": integrity,
        "signed_gain_vs_feature41_ge_10pct": signed_f["relative_gain"] >= 0.10,
        "signed_gain_vs_parent_v11_ge_1pct": signed_parent["relative_gain"] >= 0.01,
        "signed_ci_lower_each_gt_zero": signed_f["ci95"][0] > 0.0
        and signed_parent["ci95"][0] > 0.0,
        "signed_positive_puzzles_vs_feature41_ge_16": signed_f["positive_puzzles"] >= 16,
        "signed_positive_puzzles_vs_parent_v11_ge_14": signed_parent[
            "positive_puzzles"
        ]
        >= 14,
        "point_absolute_gain_vs_feature41_ge_5pct": point_f["relative_gain"] >= 0.05,
        "point_absolute_gain_vs_parent_v11_ge_1pct": point_parent[
            "relative_gain"
        ]
        >= 0.01,
        "point_absolute_ci_lower_each_gt_zero": point_f["ci95"][0] > 0.0
        and point_parent["ci95"][0] > 0.0,
        "point_absolute_positive_puzzles_vs_feature41_ge_16": point_f[
            "positive_puzzles"
        ]
        >= 16,
        "point_absolute_positive_puzzles_vs_parent_v11_ge_14": point_parent[
            "positive_puzzles"
        ]
        >= 14,
        "task_crps_gain_vs_feature41_ge_5pct": crps_f["relative_gain"] >= 0.05,
        "task_crps_gain_vs_parent_v11_ge_1_5pct": crps_parent["relative_gain"] >= 0.015,
        "task_crps_gain_vs_terminal_v10_ge_1_5pct": crps_v10["relative_gain"] >= 0.015,
        "task_crps_ci_lower_each_gt_zero": crps_f["ci95"][0] > 0.0
        and crps_parent["ci95"][0] > 0.0
        and crps_v10["ci95"][0] > 0.0,
        "task_crps_positive_puzzles_vs_feature41_ge_16": crps_f["positive_puzzles"] >= 16,
        "task_crps_positive_puzzles_vs_parent_v11_ge_14": crps_parent[
            "positive_puzzles"
        ]
        >= 14,
        "task_crps_positive_puzzles_vs_terminal_v10_ge_14": crps_v10[
            "positive_puzzles"
        ]
        >= 14,
        "distribution_absolute_gain_vs_feature41_ge_15pct": dist_f[
            "relative_gain"
        ]
        >= 0.15,
        "distribution_absolute_gain_vs_terminal_v10_ge_1pct": dist_v10[
            "relative_gain"
        ]
        >= 0.01,
        "distribution_absolute_ci_lower_each_gt_zero": dist_f["ci95"][0] > 0.0
        and dist_v10["ci95"][0] > 0.0,
        "distribution_absolute_positive_puzzles_vs_feature41_ge_16": dist_f[
            "positive_puzzles"
        ]
        >= 16,
        "distribution_absolute_positive_puzzles_vs_terminal_v10_ge_14": dist_v10[
            "positive_puzzles"
        ]
        >= 14,
        "leave_one_puzzle_out_all_headline_metrics_positive": all(
            item["leave_one_puzzle_out_all_positive"] for item in headline
        ),
        "max_single_puzzle_effect_fraction_le_0_20": all(
            item["max_single_puzzle_effect_fraction"] <= 0.20 for item in headline
        ),
        "coverage_error_guardrail": calibration_gate,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V12M3",
        "status": (
            "V12M3_TOP_JOURNAL_SCREEN_PASS"
            if passed
            else "V12M3_TOP_JOURNAL_SCREEN_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": comparisons,
        "calibration": calibration,
        "target_profile_identity_exact": True,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "POST_HOC_DEVELOPMENT_SCREEN_ONLY",
        "v12m4_authorized": passed,
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(json.loads(args.score_json.read_text()))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
