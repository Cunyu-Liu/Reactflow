#!/usr/bin/env python3
"""Apply the frozen post-V13 route-support Gates mechanically."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import t as student_t

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.score_post_v13_route_diagnostics import (
    SCHEMA as SCORE_SCHEMA,
)


SCHEMA = "reactflow_delta.post_v13_route_diagnostic_qualification.v1"
NOISE_SIGNED_GAIN_MIN = 0.005
NOISE_POINT_ABSOLUTE_GAIN_MIN = 0.005
COHERENT_SIGNED_GAIN_MIN = 0.005
COHERENT_POINT_ABSOLUTE_GAIN_MIN = 0.01
POSITIVE_PUZZLES_MIN = 14


def paired_summary(
    rows: list[dict[str, Any]], baseline_field: str, candidate_field: str
) -> dict[str, Any]:
    if len(rows) != 20:
        raise ValueError("post-V13 route qualification requires 20 puzzles")
    baseline = np.asarray([float(row[baseline_field]) for row in rows])
    candidate = np.asarray([float(row[candidate_field]) for row in rows])
    effects = baseline - candidate
    mean = float(effects.mean())
    half = float(student_t.ppf(0.975, 19) * effects.std(ddof=1) / math.sqrt(20))
    baseline_mean = float(baseline.mean())
    if baseline_mean <= 0:
        raise ValueError("post-V13 route baseline metric must be positive")
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": float(candidate.mean()),
        "mean_gain": mean,
        "relative_gain": mean / baseline_mean,
        "ci95": [mean - half, mean + half],
        "positive_puzzles": int((effects > 0).sum()),
        "per_puzzle": effects.tolist(),
    }


def _route_margin(
    signed: dict[str, Any],
    point_absolute: dict[str, Any],
    *,
    signed_min: float,
    point_min: float,
) -> float:
    """Minimum normalized non-binary Gate margin for deterministic routing."""

    return min(
        float(signed["relative_gain"]) / signed_min,
        float(point_absolute["relative_gain"]) / point_min,
        float(signed["positive_puzzles"]) / POSITIVE_PUZZLES_MIN,
        float(point_absolute["positive_puzzles"]) / POSITIVE_PUZZLES_MIN,
    )


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "PV13D3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("post-V13 qualifier requires one complete score artifact")
    if (
        scores.get("target_join_after_complete_merge") is not True
        or scores.get("corrected_feature41_replay_all_folds") is not True
        or scores.get("partial_fold_scores_inspected") is not False
        or scores.get("model_or_threshold_selection_performed") is not False
        or scores.get("external_outcome_accessed") is not False
    ):
        raise ValueError("post-V13 qualifier rejects contaminated score provenance")
    rows = sorted(scores.get("scores", []), key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("post-V13 qualifier requires unique folds0-19")
    integrity = all(
        float(row["registered_prediction_coverage"]) == 1.0
        and float(row["failure_rate"]) == 0.0
        and int(row["n_unexpected_prediction_keys"]) == 0
        for row in rows
    )
    comparisons = {
        "noise_signed": paired_summary(
            rows,
            "baseline_signed_delta_mae",
            "noise_aware_signed_delta_mae",
        ),
        "noise_point_absolute": paired_summary(
            rows,
            "baseline_point_absolute_delta_mae",
            "noise_aware_point_absolute_delta_mae",
        ),
        "coherent_signed": paired_summary(
            rows,
            "baseline_signed_delta_mae",
            "coherent_signed_delta_mae",
        ),
        "coherent_point_absolute": paired_summary(
            rows,
            "baseline_point_absolute_delta_mae",
            "coherent_point_absolute_delta_mae",
        ),
    }
    noise_signed = comparisons["noise_signed"]
    noise_absolute = comparisons["noise_point_absolute"]
    coherent_signed = comparisons["coherent_signed"]
    coherent_absolute = comparisons["coherent_point_absolute"]
    noise_checks = {
        "prediction_integrity": integrity,
        "signed_relative_gain_ge_0_005": noise_signed["relative_gain"]
        >= NOISE_SIGNED_GAIN_MIN,
        "point_absolute_relative_gain_ge_0_005": noise_absolute["relative_gain"]
        >= NOISE_POINT_ABSOLUTE_GAIN_MIN,
        "signed_ci_lower_gt_zero": noise_signed["ci95"][0] > 0.0,
        "point_absolute_ci_lower_gt_zero": noise_absolute["ci95"][0] > 0.0,
        "signed_positive_puzzles_ge_14": noise_signed["positive_puzzles"]
        >= POSITIVE_PUZZLES_MIN,
        "point_absolute_positive_puzzles_ge_14": noise_absolute["positive_puzzles"]
        >= POSITIVE_PUZZLES_MIN,
    }
    coherent_checks = {
        "prediction_integrity": integrity,
        "signed_relative_gain_ge_0_005": coherent_signed["relative_gain"]
        >= COHERENT_SIGNED_GAIN_MIN,
        "point_absolute_relative_gain_ge_0_01": coherent_absolute["relative_gain"]
        >= COHERENT_POINT_ABSOLUTE_GAIN_MIN,
        "signed_ci_lower_gt_zero": coherent_signed["ci95"][0] > 0.0,
        "point_absolute_ci_lower_gt_zero": coherent_absolute["ci95"][0] > 0.0,
        "signed_positive_puzzles_ge_14": coherent_signed["positive_puzzles"]
        >= POSITIVE_PUZZLES_MIN,
        "point_absolute_positive_puzzles_ge_14": coherent_absolute[
            "positive_puzzles"
        ]
        >= POSITIVE_PUZZLES_MIN,
    }
    noise_pass = all(noise_checks.values())
    coherent_pass = all(coherent_checks.values())
    margins = {
        "noise_aware": _route_margin(
            noise_signed,
            noise_absolute,
            signed_min=NOISE_SIGNED_GAIN_MIN,
            point_min=NOISE_POINT_ABSOLUTE_GAIN_MIN,
        ),
        "coherent_factorization": _route_margin(
            coherent_signed,
            coherent_absolute,
            signed_min=COHERENT_SIGNED_GAIN_MIN,
            point_min=COHERENT_POINT_ABSOLUTE_GAIN_MIN,
        ),
    }
    if noise_pass and coherent_pass:
        selected = (
            "NOISE_AWARE_POINT_TRAINING"
            if margins["noise_aware"] >= margins["coherent_factorization"]
            else "SIGNED_MAGNITUDE_FACTORIZATION"
        )
        status = f"PV13D3_BOTH_SUPPORTED_{selected}_SELECTED"
    elif noise_pass:
        selected = "NOISE_AWARE_POINT_TRAINING"
        status = "PV13D3_NOISE_AWARE_ROUTE_SELECTED"
    elif coherent_pass:
        selected = "SIGNED_MAGNITUDE_FACTORIZATION"
        status = "PV13D3_COHERENT_FACTORIZATION_ROUTE_SELECTED"
    else:
        selected = "WT_PROFILE_SELF_SUPERVISED_PRETRAINING_ONLY"
        status = "PV13D3_A_AND_C_CLOSED_WT_PROFILE_PRETRAINING_ONLY"
    return {
        "schema_version": SCHEMA,
        "phase": "PV13D3",
        "status": status,
        "selected_next_route": selected,
        "route_support": {
            "noise_aware": noise_pass,
            "coherent_factorization": coherent_pass,
        },
        "checks": {
            "noise_aware": noise_checks,
            "coherent_factorization": coherent_checks,
        },
        "comparisons": comparisons,
        "normalized_route_margins": margins,
        "deterministic_tie_breaker": "NOISE_AWARE_POINT_TRAINING",
        "evidence_status": "POST_HOC_DEVELOPMENT_ROUTE_DIAGNOSTIC_ONLY",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
