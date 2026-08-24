#!/usr/bin/env python3
"""Apply the pre-frozen V6M2 incremental constrained-signal Gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


SCHEMA = "reactflow_delta.model_rescue_v6_probe_qualification.v1"


def paired_summary(effects: list[float]) -> dict[str, Any]:
    values = np.asarray(effects, dtype=np.float64)
    if values.shape != (20,):
        raise ValueError("V6M2 qualification requires exactly 20 puzzle effects")
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, 19) * values.std(ddof=1) / math.sqrt(20))
    return {
        "mean_gain": mean,
        "ci95": [mean - half, mean + half],
        "positive_puzzles": int((values > 0).sum()),
        "per_puzzle": values.tolist(),
    }


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    rows = scores.get("scores", [])
    if scores.get("status") != "V6M2_COMPLETE_SCORE_PASS" or len(rows) != 20:
        raise ValueError("v6 qualifier requires one complete V6M2 score artifact")
    if scores.get("v5_baseline_replay_all_folds") is not True:
        raise ValueError("v6 qualifier requires v5 baseline replay on every fold")
    rows = sorted(rows, key=lambda row: int(row["outer_fold"]))
    if [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("v6 qualifier requires unique outer folds 0 through 19")
    signed = paired_summary(
        [
            float(row["baseline_signed_delta_mae"])
            - float(row["candidate_signed_delta_mae"])
            for row in rows
        ]
    )
    absolute = paired_summary(
        [
            float(row["baseline_absolute_delta_mae"])
            - float(row["candidate_absolute_delta_mae"])
            for row in rows
        ]
    )
    signed_baseline = float(
        np.mean([float(row["baseline_signed_delta_mae"]) for row in rows])
    )
    absolute_baseline = float(
        np.mean([float(row["baseline_absolute_delta_mae"]) for row in rows])
    )
    signed["baseline_mean"] = signed_baseline
    signed["candidate_mean"] = signed_baseline - signed["mean_gain"]
    signed["relative_gain"] = signed["mean_gain"] / signed_baseline
    absolute["baseline_mean"] = absolute_baseline
    absolute["candidate_mean"] = absolute_baseline - absolute["mean_gain"]
    absolute["relative_gain"] = absolute["mean_gain"] / absolute_baseline
    integrity = all(
        float(row["registered_prediction_coverage"]) == 1.0
        and float(row["failure_rate"]) == 0.0
        and int(row["n_unexpected_prediction_keys"]) == 0
        for row in rows
    )
    gates = {
        "signed_delta_relative_gain": signed["relative_gain"] >= 0.01,
        "signed_delta_ci_lower": signed["ci95"][0] > 0.0,
        "signed_delta_positive_puzzles": signed["positive_puzzles"] >= 14,
        "absolute_delta_guardrail": absolute["relative_gain"] >= -0.005,
        "prediction_integrity": integrity,
        "v5_baseline_replay": True,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V6M2",
        "overall_status": (
            "V6M2_CONSTRAINED_SIGNAL_ELIGIBLE"
            if passed
            else "MODEL_RESCUE_V6_FAIL"
        ),
        "v6m3_authorized": passed,
        "evidence_status": "DEVELOPMENT_CONSUMED_ELIGIBILITY_ONLY",
        "signed_delta": signed,
        "absolute_delta": absolute,
        "gates": gates,
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "mechanism": "NOT_ESTABLISHED",
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
    print(
        json.dumps(
            {"status": result["overall_status"], "result": str(args.out_json)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
