#!/usr/bin/env python3
"""Mechanically apply the frozen corrected V7M2 eligibility Gate."""

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

from scripts.reactflow_delta.score_model_rescue_v7_probe import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v7_probe_qualification.v1"
SIGNED_RELATIVE_GAIN_MIN = 0.01
SIGNED_POSITIVE_PUZZLES_MIN = 14
ABSOLUTE_RELATIVE_GAIN_MIN = -0.005


def paired_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    baseline_field = f"baseline_{metric}_mae"
    candidate_field = f"candidate_{metric}_mae"
    baseline = np.asarray([float(row[baseline_field]) for row in rows], dtype=np.float64)
    candidate = np.asarray([float(row[candidate_field]) for row in rows], dtype=np.float64)
    effects = baseline - candidate
    if effects.shape != (20,):
        raise ValueError("V7M2 paired summary requires 20 puzzle effects")
    mean = float(effects.mean())
    half = float(student_t.ppf(0.975, 19) * effects.std(ddof=1) / math.sqrt(20))
    baseline_mean = float(baseline.mean())
    return {
        "metric": metric,
        "baseline_mean": baseline_mean,
        "candidate_mean": float(candidate.mean()),
        "mean_gain": mean,
        "relative_gain": mean / baseline_mean,
        "ci95": [mean - half, mean + half],
        "positive_puzzles": int((effects > 0).sum()),
        "per_puzzle": effects.tolist(),
    }


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    rows = scores.get("scores", [])
    if scores.get("schema_version") != SCORE_SCHEMA:
        raise ValueError("V7M2 qualifier requires the frozen score schema")
    if scores.get("status") != "V7M2_COMPLETE_CORRECTED_SCORE_PASS":
        raise ValueError("V7M2 qualifier requires one complete score artifact")
    if scores.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise ValueError("V7M2 qualifier requires corrected target identity")
    if scores.get("target_join_after_complete_merge") is not True:
        raise ValueError("V7M2 qualifier requires target join after complete merge")
    if scores.get("partial_fold_scores_inspected") is not False:
        raise ValueError("V7M2 qualifier rejects partial score inspection")
    if scores.get("model_selection_performed") is not False:
        raise ValueError("V7M2 qualifier rejects model selection")
    if scores.get("legacy_target_dependent_prediction_reused") is not False:
        raise ValueError("V7M2 qualifier rejects legacy target-dependent predictions")
    if scores.get("external_outcome_accessed") is not False:
        raise ValueError("V7M2 qualifier rejects external outcome access")
    if len(rows) != 20:
        raise ValueError("V7M2 qualifier requires exactly 20 folds")
    rows = sorted(rows, key=lambda row: int(row["outer_fold"]))
    if [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("V7M2 qualifier requires unique folds 0 through 19")

    coverage = all(float(row["registered_prediction_coverage"]) == 1.0 for row in rows)
    failures = all(float(row["failure_rate"]) == 0.0 for row in rows)
    unexpected = all(int(row["n_unexpected_prediction_keys"]) == 0 for row in rows)
    signed = paired_summary(rows, "signed_delta")
    absolute = paired_summary(rows, "absolute_delta")
    checks = {
        "complete_twenty_fold_universe": True,
        "registered_prediction_coverage_100_percent": coverage,
        "failure_rate_zero": failures,
        "unexpected_keys_zero": unexpected,
        "signed_delta_relative_gain_at_least_one_percent": (
            signed["relative_gain"] >= SIGNED_RELATIVE_GAIN_MIN
        ),
        "signed_delta_ci_lower_above_zero": signed["ci95"][0] > 0.0,
        "signed_delta_positive_puzzles_at_least_fourteen": (
            signed["positive_puzzles"] >= SIGNED_POSITIVE_PUZZLES_MIN
        ),
        "absolute_delta_relative_gain_guardrail": (
            absolute["relative_gain"] >= ABSOLUTE_RELATIVE_GAIN_MIN
        ),
    }
    eligible = all(checks.values())
    status = (
        "V7M2_RINALMO_DEPENDENCY_SIGNAL_ELIGIBLE"
        if eligible
        else "V7M2_RINALMO_DEPENDENCY_SIGNAL_NOT_ELIGIBLE"
    )
    return {
        "schema_version": SCHEMA,
        "phase": "V7M2",
        "status": status,
        "checks": checks,
        "signed_delta": signed,
        "absolute_delta": absolute,
        "target_identity_exact": True,
        "corrected_feature41_comparator": True,
        "candidate_addition": "RINALMO_DEPENDENCY6_ONLY",
        "ridge_alpha": 1.0,
        "model_selection_performed": False,
        "legacy_target_dependent_prediction_reused": False,
        "evidence_status": "CORRECTED_DEVELOPMENT_CONSUMED_ELIGIBILITY_PROBE",
        "candidate_model_trained": False,
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    scores = json.loads(args.score_json.read_text(encoding="utf-8"))
    result = qualify(scores)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}, indent=2))
    return (
        0
        if result["status"] == "V7M2_RINALMO_DEPENDENCY_SIGNAL_ELIGIBLE"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
