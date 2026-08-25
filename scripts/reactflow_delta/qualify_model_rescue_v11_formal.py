#!/usr/bin/env python3
"""Apply the pre-frozen V11 five-seed formal Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.qualify_model_rescue_v10 import paired_summary
from scripts.reactflow_delta.qualify_model_rescue_v11 import SCHEMA as SCREEN_QUAL_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v11_formal import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v11_formal_qualification.v1"


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "signed_vs_feature41": paired_summary(
            rows, "feature41_signed_delta_mae", "anchored_signed_delta_mae"
        ),
        "signed_vs_v8": paired_summary(
            rows, "v8_signed_delta_mae", "anchored_signed_delta_mae"
        ),
        "signed_vs_unanchored": paired_summary(
            rows, "unanchored_signed_delta_mae", "anchored_signed_delta_mae"
        ),
        "point_absolute_vs_feature41": paired_summary(
            rows,
            "feature41_absolute_delta_mae",
            "anchored_point_absolute_delta_mae",
        ),
        "task_crps_vs_feature41": paired_summary(
            rows, "feature41_crps", "anchored_crps"
        ),
        "task_crps_vs_unanchored": paired_summary(
            rows, "unanchored_crps", "anchored_crps"
        ),
        "distribution_absolute_vs_feature41": paired_summary(
            rows,
            "feature41_absolute_delta_mae",
            "anchored_distribution_absolute_delta_mae",
        ),
    }


def qualify(scores: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
    if screen.get("schema_version") != SCREEN_QUAL_SCHEMA or screen.get("status") != (
        "V11M3_TOP_JOURNAL_SCREEN_PASS"
    ) or screen.get("gate_passed") is not True:
        raise ValueError("V11 formal qualifier requires exact screen PASS")
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V11M4_COMPLETE_FORMAL_SCORE_PASS"
    ):
        raise ValueError("V11 formal qualifier requires complete formal scores")
    if not (
        scores.get("equal_seed_mixture") is True
        and scores.get("partial_fold_scores_inspected") is False
        and scores.get("external_outcome_accessed") is False
        and scores.get("model_or_threshold_selection_performed") is False
    ):
        raise ValueError("V11 formal score violates the frozen protocol")
    rows = sorted(scores.get("mixture_scores", []), key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("V11 formal qualifier requires mixture folds0-19")
    comparisons = _comparisons(rows)
    signed_f = comparisons["signed_vs_feature41"]
    signed_v8 = comparisons["signed_vs_v8"]
    signed_null = comparisons["signed_vs_unanchored"]
    point_absolute = comparisons["point_absolute_vs_feature41"]
    crps_f = comparisons["task_crps_vs_feature41"]
    crps_null = comparisons["task_crps_vs_unanchored"]
    distribution_absolute = comparisons["distribution_absolute_vs_feature41"]
    headline = tuple(comparisons.values())
    integrity = all(
        float(row["registered_prediction_coverage"]) == 1.0
        and float(row["failure_rate"]) == 0.0
        and int(row["n_unexpected_prediction_keys"]) == 0
        for row in rows
    )
    calibration = {}
    calibration_gate = True
    for level in (68, 95):
        nominal = level / 100.0
        baseline = float(np.mean([row[f"feature41_coverage{level}"] for row in rows]))
        candidate = float(np.mean([row[f"anchored_coverage{level}"] for row in rows]))
        baseline_error = abs(baseline - nominal)
        candidate_error = abs(candidate - nominal)
        passed = candidate_error <= baseline_error + 0.02
        calibration[f"coverage{level}"] = {
            "nominal": nominal,
            "baseline": baseline,
            "candidate": candidate,
            "candidate_not_worse_by_more_than_0_02": passed,
        }
        calibration_gate = calibration_gate and passed

    individual = scores.get("individual_seed_scores", {})
    if sorted(map(int, individual)) != list(range(5)):
        raise ValueError("V11 formal qualifier requires individual seeds0-4")
    seed_directions = {}
    signed_positive_seeds = 0
    crps_positive_seeds = 0
    for seed in range(5):
        seed_rows = sorted(
            individual[str(seed)], key=lambda row: int(row["outer_fold"])
        )
        if len(seed_rows) != 20 or [
            int(row["outer_fold"]) for row in seed_rows
        ] != list(range(20)):
            raise ValueError(f"V11 formal seed{seed} lacks twenty folds")
        signed = paired_summary(
            seed_rows, "feature41_signed_delta_mae", "anchored_signed_delta_mae"
        )
        crps = paired_summary(seed_rows, "feature41_crps", "anchored_crps")
        signed_positive = signed["mean_gain"] > 0.0
        crps_positive = crps["mean_gain"] > 0.0
        signed_positive_seeds += int(signed_positive)
        crps_positive_seeds += int(crps_positive)
        seed_directions[str(seed)] = {
            "signed_mean_gain": signed["mean_gain"],
            "signed_positive": signed_positive,
            "task_crps_mean_gain": crps["mean_gain"],
            "task_crps_positive": crps_positive,
        }
    gates = {
        "screen_prerequisite_exact_pass": True,
        "prediction_integrity": integrity,
        "signed_gain_vs_feature41_ge_10pct": signed_f["relative_gain"] >= 0.10,
        "signed_gain_vs_v8_ge_2pct": signed_v8["relative_gain"] >= 0.02,
        "signed_gain_vs_unanchored_ge_1pct": signed_null["relative_gain"] >= 0.01,
        "signed_ci_lower_each_gt_zero": all(
            value["ci95"][0] > 0.0 for value in (signed_f, signed_v8, signed_null)
        ),
        "signed_positive_puzzles_vs_feature41_ge_16": signed_f["positive_puzzles"] >= 16,
        "signed_positive_puzzles_vs_v8_ge_14": signed_v8["positive_puzzles"] >= 14,
        "signed_positive_puzzles_vs_unanchored_ge_14": signed_null["positive_puzzles"] >= 14,
        "point_absolute_gain_vs_feature41_ge_1pct": point_absolute["relative_gain"] >= 0.01,
        "point_absolute_ci_lower_gt_zero": point_absolute["ci95"][0] > 0.0,
        "point_absolute_positive_puzzles_ge_14": point_absolute["positive_puzzles"] >= 14,
        "task_crps_gain_vs_feature41_ge_5pct": crps_f["relative_gain"] >= 0.05,
        "task_crps_gain_vs_unanchored_ge_1pct": crps_null["relative_gain"] >= 0.01,
        "task_crps_ci_lower_each_gt_zero": crps_f["ci95"][0] > 0.0
        and crps_null["ci95"][0] > 0.0,
        "task_crps_positive_puzzles_vs_feature41_ge_16": crps_f["positive_puzzles"] >= 16,
        "task_crps_positive_puzzles_vs_unanchored_ge_14": crps_null["positive_puzzles"] >= 14,
        "distribution_absolute_gain_vs_feature41_ge_12pct": distribution_absolute["relative_gain"] >= 0.12,
        "leave_one_puzzle_out_all_headline_metrics_positive": all(
            value["leave_one_puzzle_out_all_positive"] for value in headline
        ),
        "max_single_puzzle_effect_fraction_le_0_20": all(
            value["max_single_puzzle_effect_fraction"] <= 0.20 for value in headline
        ),
        "coverage_error_guardrail": calibration_gate,
        "signed_positive_individual_seeds_ge_4": signed_positive_seeds >= 4,
        "task_crps_positive_individual_seeds_ge_4": crps_positive_seeds >= 4,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V11M4",
        "status": (
            "V11M4_TOP_JOURNAL_FORMAL_PASS"
            if passed
            else "V11M4_TOP_JOURNAL_FORMAL_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": comparisons,
        "calibration": calibration,
        "individual_seed_directions": seed_directions,
        "evidence_status": (
            "POST_HOC_DEVELOPMENT_PASS" if passed else "DEVELOPMENT_NEGATIVE"
        ),
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--screen-qualification-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(
        json.loads(args.score_json.read_text(encoding="utf-8")),
        json.loads(args.screen_qualification_json.read_text(encoding="utf-8")),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
