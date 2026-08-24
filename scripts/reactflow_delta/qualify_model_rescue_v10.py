#!/usr/bin/env python3
"""Apply the frozen V10 top-journal Gate to one complete score artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from scripts.reactflow_delta.score_model_rescue_v10 import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v10_qualification.v1"


def paired_summary(
    rows: list[dict[str, Any]], baseline_field: str, candidate_field: str
) -> dict[str, Any]:
    baseline = np.asarray([float(row[baseline_field]) for row in rows])
    candidate = np.asarray([float(row[candidate_field]) for row in rows])
    if baseline.shape != (20,) or candidate.shape != (20,):
        raise ValueError("V10 paired summary requires exactly 20 puzzles")
    effects = baseline - candidate
    mean_gain = float(effects.mean())
    half = float(student_t.ppf(0.975, 19) * effects.std(ddof=1) / math.sqrt(20))
    baseline_mean = float(baseline.mean())
    leave_one_out = [float(np.delete(effects, index).mean()) for index in range(20)]
    effect_sum = float(effects.sum())
    max_fraction = (
        float(np.max(np.abs(effects)) / abs(effect_sum))
        if effect_sum != 0.0
        else float("inf")
    )
    return {
        "baseline_field": baseline_field,
        "candidate_field": candidate_field,
        "baseline_mean": baseline_mean,
        "candidate_mean": float(candidate.mean()),
        "mean_gain": mean_gain,
        "relative_gain": mean_gain / baseline_mean,
        "ci95": [mean_gain - half, mean_gain + half],
        "positive_puzzles": int((effects > 0.0).sum()),
        "per_puzzle": effects.tolist(),
        "leave_one_puzzle_out": leave_one_out,
        "leave_one_puzzle_out_all_positive": all(value > 0.0 for value in leave_one_out),
        "max_single_puzzle_effect_fraction": max_fraction,
    }


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V10M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("V10 qualifier requires one complete V10 score artifact")
    rows = sorted(scores.get("scores", []), key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("V10 qualifier requires unique folds0-19")
    integrity = all(
        float(row["registered_prediction_coverage"]) == 1.0
        and float(row["failure_rate"]) == 0.0
        and int(row["n_unexpected_prediction_keys"]) == 0
        for row in rows
    )
    signed = paired_summary(
        rows, "feature41_signed_delta_mae", "meanaligned_signed_delta_mae"
    )
    absolute = paired_summary(
        rows,
        "feature41_absolute_delta_mae",
        "meanaligned_asymmetric_absolute_delta_mae",
    )
    task_crps = paired_summary(
        rows, "feature41_asymmetric_crps", "meanaligned_asymmetric_crps"
    )
    historical = paired_summary(
        rows, "historical_v9_crps", "meanaligned_asymmetric_crps"
    )
    asymmetry = paired_summary(
        rows, "meanaligned_symmetric_crps", "meanaligned_asymmetric_crps"
    )
    capacity = paired_summary(
        rows, "historical_v9_crps", "meanaligned_symmetric_crps"
    )
    calibration = {}
    calibration_gate = True
    for level in (68, 95):
        nominal = level / 100.0
        baseline = float(
            np.mean([row[f"feature41_asymmetric_coverage{level}"] for row in rows])
        )
        candidate = float(
            np.mean([row[f"meanaligned_asymmetric_coverage{level}"] for row in rows])
        )
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
    headline = (signed, absolute, task_crps, historical, asymmetry)
    gates = {
        "prediction_integrity": integrity,
        "signed_relative_gain_ge_5pct": signed["relative_gain"] >= 0.05,
        "signed_ci_lower_gt_zero": signed["ci95"][0] > 0.0,
        "signed_positive_puzzles_ge_16": signed["positive_puzzles"] >= 16,
        "absolute_relative_gain_ge_5pct": absolute["relative_gain"] >= 0.05,
        "absolute_ci_lower_gt_zero": absolute["ci95"][0] > 0.0,
        "absolute_positive_puzzles_ge_16": absolute["positive_puzzles"] >= 16,
        "task_crps_relative_gain_ge_5pct": task_crps["relative_gain"] >= 0.05,
        "task_crps_ci_lower_gt_zero": task_crps["ci95"][0] > 0.0,
        "task_crps_positive_puzzles_ge_16": task_crps["positive_puzzles"] >= 16,
        "crps_vs_historical_v9_relative_gain_ge_1pct": (
            historical["relative_gain"] >= 0.01
        ),
        "crps_vs_historical_v9_ci_lower_gt_zero": historical["ci95"][0] > 0.0,
        "asymmetric_vs_symmetric_relative_gain_ge_1pct": (
            asymmetry["relative_gain"] >= 0.01
        ),
        "asymmetric_vs_symmetric_ci_lower_gt_zero": asymmetry["ci95"][0] > 0.0,
        "asymmetric_vs_symmetric_positive_puzzles_ge_14": (
            asymmetry["positive_puzzles"] >= 14
        ),
        "leave_one_puzzle_out_all_headline_metrics_positive": all(
            result["leave_one_puzzle_out_all_positive"] for result in headline
        ),
        "max_single_puzzle_effect_fraction_le_0_20": all(
            result["max_single_puzzle_effect_fraction"] <= 0.20
            for result in headline
        ),
        "coverage_error_guardrail": calibration_gate,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V10M3",
        "status": (
            "V10M3_TOP_JOURNAL_SCREEN_PASS"
            if passed
            else "V10M3_TOP_JOURNAL_SCREEN_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": {
            "signed_delta": signed,
            "absolute_delta": absolute,
            "task_crps": task_crps,
            "crps_vs_historical_v9": historical,
            "asymmetric_vs_symmetric": asymmetry,
            "capacity_symmetric_vs_historical_v9": capacity,
        },
        "calibration": calibration,
        "target_profile_identity_exact": True,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "POST_HOC_DEVELOPMENT_SCREEN_ONLY",
        "v10m4_authorized": passed,
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
