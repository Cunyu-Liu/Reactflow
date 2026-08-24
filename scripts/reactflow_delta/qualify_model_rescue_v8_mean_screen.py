#!/usr/bin/env python3
"""Mechanically apply the frozen V8M2 mean-signal Gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from scripts.reactflow_delta.score_model_rescue_v8_mean_screen import (
    SCHEMA as SCORE_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v8_mean_screen_qualification.v1"


def paired_summary(
    rows: list[dict[str, Any]], baseline: str, candidate: str, metric: str
) -> dict[str, Any]:
    baseline_values = np.asarray(
        [float(row[f"{baseline}_{metric}_mae"]) for row in rows], dtype=np.float64
    )
    candidate_values = np.asarray(
        [float(row[f"{candidate}_{metric}_mae"]) for row in rows], dtype=np.float64
    )
    if baseline_values.shape != (20,) or candidate_values.shape != (20,):
        raise ValueError("V8M2 paired summary requires 20 puzzle effects")
    effects = baseline_values - candidate_values
    mean_gain = float(effects.mean())
    half = float(student_t.ppf(0.975, 19) * effects.std(ddof=1) / math.sqrt(20))
    baseline_mean = float(baseline_values.mean())
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "baseline_mean": baseline_mean,
        "candidate_mean": float(candidate_values.mean()),
        "mean_gain": mean_gain,
        "relative_gain": mean_gain / baseline_mean,
        "ci95": [mean_gain - half, mean_gain + half],
        "positive_puzzles": int((effects > 0).sum()),
        "per_puzzle": effects.tolist(),
    }


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    if scores.get("schema_version") != SCORE_SCHEMA:
        raise ValueError("V8M2 qualifier requires the V8M2 score schema")
    if scores.get("status") != "V8M2_COMPLETE_MEAN_SCREEN_SCORE_PASS":
        raise ValueError("V8M2 qualifier requires one complete score artifact")
    if scores.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise ValueError("V8M2 qualifier requires exact target identity")
    rows = sorted(scores.get("scores", []), key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("V8M2 qualifier requires unique folds 0 through 19")
    integrity = all(
        float(row["registered_prediction_coverage"]) == 1.0
        and float(row["failure_rate"]) == 0.0
        and int(row["n_unexpected_prediction_keys"]) == 0
        for row in rows
    )
    signed_vs_feature41 = paired_summary(
        rows, "feature41", "meanaligned", "signed_delta"
    )
    signed_vs_b1 = paired_summary(rows, "b1", "meanaligned", "signed_delta")
    absolute_vs_feature41 = paired_summary(
        rows, "feature41", "meanaligned", "absolute_delta"
    )
    diagnostics = {
        "signed_feature41_to_b1": paired_summary(
            rows, "feature41", "b1", "signed_delta"
        ),
        "absolute_feature41_to_b1": paired_summary(
            rows, "feature41", "b1", "absolute_delta"
        ),
    }
    gates = {
        "prediction_integrity": integrity,
        "signed_relative_gain_vs_feature41_ge_1pct": (
            signed_vs_feature41["relative_gain"] >= 0.01
        ),
        "signed_relative_gain_vs_b1_ge_1pct": signed_vs_b1["relative_gain"] >= 0.01,
        "signed_ci_lower_vs_feature41_gt_zero": signed_vs_feature41["ci95"][0]
        > 0.0,
        "signed_ci_lower_vs_b1_gt_zero": signed_vs_b1["ci95"][0] > 0.0,
        "signed_positive_puzzles_vs_feature41_ge_14": (
            signed_vs_feature41["positive_puzzles"] >= 14
        ),
        "absolute_relative_gain_vs_feature41_ge_minus_0_5pct": (
            absolute_vs_feature41["relative_gain"] >= -0.005
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V8M2",
        "status": (
            "V8M2_MEAN_SIGNAL_ELIGIBLE"
            if passed
            else "V8M2_MEAN_SIGNAL_NOT_ELIGIBLE"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": {
            "signed_feature41_to_meanaligned": signed_vs_feature41,
            "signed_b1_to_meanaligned": signed_vs_b1,
            "absolute_feature41_to_meanaligned": absolute_vs_feature41,
        },
        "diagnostics": diagnostics,
        "target_profile_identity_exact": True,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "POST_HOC_DEVELOPMENT_SCREEN_ONLY",
        "v8m3_authorized": passed,
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
