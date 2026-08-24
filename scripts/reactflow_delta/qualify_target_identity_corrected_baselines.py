#!/usr/bin/env python3
"""Qualify the complete corrected comparator rebuild without model selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from scripts.reactflow_delta.score_target_identity_corrected_baselines import (
    SCHEMA as SCORE_SCHEMA,
)


SCHEMA = "reactflow_delta.target_identity_corrected_baseline_qualification.v1"


def paired_summary(
    rows: list[dict[str, Any]], baseline: str, candidate: str, metric: str
) -> dict[str, Any]:
    baseline_field = f"{baseline}_{metric}_mae"
    candidate_field = f"{candidate}_{metric}_mae"
    baseline_values = np.asarray(
        [float(row[baseline_field]) for row in rows], dtype=np.float64
    )
    candidate_values = np.asarray(
        [float(row[candidate_field]) for row in rows], dtype=np.float64
    )
    effects = baseline_values - candidate_values
    if effects.shape != (20,):
        raise ValueError("corrected baseline summary requires 20 puzzle effects")
    mean = float(effects.mean())
    half = float(student_t.ppf(0.975, 19) * effects.std(ddof=1) / math.sqrt(20))
    baseline_mean = float(baseline_values.mean())
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "baseline_mean": baseline_mean,
        "candidate_mean": float(candidate_values.mean()),
        "mean_gain": mean,
        "relative_gain": mean / baseline_mean,
        "ci95": [mean - half, mean + half],
        "positive_puzzles": int((effects > 0).sum()),
        "per_puzzle": effects.tolist(),
    }


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    rows = scores.get("scores", [])
    if scores.get("schema_version") != SCORE_SCHEMA:
        raise ValueError("corrected qualifier requires the corrected score schema")
    if scores.get("status") != "TIC2A_COMPLETE_CORRECTED_SCORE_PASS":
        raise ValueError("corrected qualifier requires one complete score artifact")
    if scores.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise ValueError("corrected qualifier requires exact target identity")
    if len(rows) != 20:
        raise ValueError("corrected qualifier requires exactly 20 folds")
    rows = sorted(rows, key=lambda row: int(row["outer_fold"]))
    if [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("corrected qualifier requires unique folds 0 through 19")
    integrity = all(
        float(row["registered_prediction_coverage"]) == 1.0
        and float(row["failure_rate"]) == 0.0
        and int(row["n_unexpected_prediction_keys"]) == 0
        for row in rows
    )
    comparisons = {}
    for comparison, baseline, candidate in (
        ("direct18_to_v5_feature30", "direct18", "v5_feature30"),
        ("v5_feature30_to_v6_feature41", "v5_feature30", "v6_feature41"),
        ("direct18_to_v6_feature41", "direct18", "v6_feature41"),
    ):
        comparisons[comparison] = {
            metric: paired_summary(rows, baseline, candidate, metric)
            for metric in ("signed_delta", "absolute_delta")
        }
    status = (
        "TIC2A_CORRECTED_BASELINE_REBUILD_PASS"
        if integrity
        else "TIC2A_CORRECTED_BASELINE_REBUILD_FAIL"
    )
    return {
        "schema_version": SCHEMA,
        "phase": "TIC2A",
        "status": status,
        "target_identity_exact": True,
        "prediction_integrity": integrity,
        "comparisons": comparisons,
        "model_selection_performed": False,
        "legacy_prediction_reused": False,
        "evidence_status": "CORRECTED_DEVELOPMENT_CONSUMED_COMPARATOR_REBUILD",
        "v7_candidate_evaluated": False,
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
    return 0 if result["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
