#!/usr/bin/env python3
"""Apply the frozen V13 top-journal seed-0 Gate mechanically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.qualify_model_rescue_v10 import paired_summary
from scripts.reactflow_delta.score_model_rescue_v13 import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v13_qualification.v1"


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V13M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("V13 qualifier requires one complete V13 score artifact")
    rows = sorted(scores.get("scores", []), key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("V13 qualifier requires unique folds0-19")
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
        "signed_vs_terminal_v12": paired_summary(
            rows, "terminal_v12_signed_delta_mae", "candidate_signed_delta_mae"
        ),
        "signed_vs_nested_null": paired_summary(
            rows, "null_signed_delta_mae", "candidate_signed_delta_mae"
        ),
        "point_absolute_vs_feature41": paired_summary(
            rows,
            "feature41_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "point_absolute_vs_terminal_v11": paired_summary(
            rows,
            "terminal_v11_point_absolute_delta_mae",
            "candidate_point_absolute_delta_mae",
        ),
        "point_absolute_vs_nested_null": paired_summary(
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
        "task_crps_vs_nested_null": paired_summary(
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
        "distribution_absolute_vs_nested_null": paired_summary(
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

    signed_f = comparisons["signed_vs_feature41"]
    signed_parent = comparisons["signed_vs_terminal_v12"]
    signed_null = comparisons["signed_vs_nested_null"]
    point_f = comparisons["point_absolute_vs_feature41"]
    point_parent = comparisons["point_absolute_vs_terminal_v11"]
    point_null = comparisons["point_absolute_vs_nested_null"]
    crps_f = comparisons["task_crps_vs_feature41"]
    crps_parent = comparisons["task_crps_vs_terminal_v12"]
    crps_null = comparisons["task_crps_vs_nested_null"]
    dist_f = comparisons["distribution_absolute_vs_feature41"]
    dist_parent = comparisons["distribution_absolute_vs_terminal_v10"]
    dist_null = comparisons["distribution_absolute_vs_nested_null"]
    groups = {
        "signed": (signed_f, signed_parent, signed_null),
        "point_absolute": (point_f, point_parent, point_null),
        "task_crps": (crps_f, crps_parent, crps_null),
        "distribution_absolute": (dist_f, dist_parent, dist_null),
    }
    headline = tuple(item for group in groups.values() for item in group)
    gates = {
        "prediction_integrity": integrity,
        "signed_gain_vs_feature41_ge_12pct": signed_f["relative_gain"] >= 0.12,
        "signed_gain_vs_terminal_v12_ge_2pct": signed_parent["relative_gain"] >= 0.02,
        "signed_gain_vs_nested_null_ge_1_5pct": signed_null["relative_gain"] >= 0.015,
        "signed_ci_lower_each_gt_zero": all(item["ci95"][0] > 0.0 for item in groups["signed"]),
        "signed_positive_puzzles_ge_16_14_14": (
            signed_f["positive_puzzles"] >= 16
            and signed_parent["positive_puzzles"] >= 14
            and signed_null["positive_puzzles"] >= 14
        ),
        "point_absolute_gain_vs_feature41_ge_7pct": point_f["relative_gain"] >= 0.07,
        "point_absolute_gain_vs_terminal_v11_ge_2pct": point_parent["relative_gain"] >= 0.02,
        "point_absolute_gain_vs_nested_null_ge_1pct": point_null["relative_gain"] >= 0.01,
        "point_absolute_ci_lower_each_gt_zero": all(
            item["ci95"][0] > 0.0 for item in groups["point_absolute"]
        ),
        "point_absolute_positive_puzzles_ge_16_14_14": (
            point_f["positive_puzzles"] >= 16
            and point_parent["positive_puzzles"] >= 14
            and point_null["positive_puzzles"] >= 14
        ),
        "task_crps_gain_vs_feature41_ge_5pct": crps_f["relative_gain"] >= 0.05,
        "task_crps_gain_vs_terminal_v12_ge_2pct": crps_parent["relative_gain"] >= 0.02,
        "task_crps_gain_vs_nested_null_ge_1_5pct": crps_null["relative_gain"] >= 0.015,
        "task_crps_ci_lower_each_gt_zero": all(
            item["ci95"][0] > 0.0 for item in groups["task_crps"]
        ),
        "task_crps_positive_puzzles_ge_16_14_14": (
            crps_f["positive_puzzles"] >= 16
            and crps_parent["positive_puzzles"] >= 14
            and crps_null["positive_puzzles"] >= 14
        ),
        "distribution_absolute_gain_vs_feature41_ge_15pct": dist_f["relative_gain"] >= 0.15,
        "distribution_absolute_gain_vs_terminal_v10_ge_2pct": dist_parent["relative_gain"] >= 0.02,
        "distribution_absolute_gain_vs_nested_null_ge_1pct": dist_null["relative_gain"] >= 0.01,
        "distribution_absolute_ci_lower_each_gt_zero": all(
            item["ci95"][0] > 0.0 for item in groups["distribution_absolute"]
        ),
        "distribution_absolute_positive_puzzles_ge_16_14_14": (
            dist_f["positive_puzzles"] >= 16
            and dist_parent["positive_puzzles"] >= 14
            and dist_null["positive_puzzles"] >= 14
        ),
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
        "phase": "V13M3",
        "status": (
            "V13M3_TOP_JOURNAL_SCREEN_PASS"
            if passed
            else "V13M3_TOP_JOURNAL_SCREEN_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": comparisons,
        "calibration": calibration,
        "target_profile_identity_exact": True,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "POST_HOC_DEVELOPMENT_SCREEN_ONLY",
        "v13m4_authorized": passed,
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(json.loads(args.score_json.read_text(encoding="utf-8")))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
