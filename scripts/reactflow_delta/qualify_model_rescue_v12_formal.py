#!/usr/bin/env python3
"""Apply the frozen V12 five-seed formal Gate without seed selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.qualify_model_rescue_v10 import paired_summary
from scripts.reactflow_delta.qualify_model_rescue_v12 import (
    SCHEMA as SCREEN_QUAL_SCHEMA,
    qualify as qualify_screen,
)
from scripts.reactflow_delta.score_model_rescue_v12 import SCHEMA as SCREEN_SCORE_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v12_formal import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v12_formal_qualification.v1"


def qualify(scores: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
    if screen.get("schema_version") != SCREEN_QUAL_SCHEMA or screen.get("status") != (
        "V12M3_TOP_JOURNAL_SCREEN_PASS"
    ) or screen.get("gate_passed") is not True:
        raise ValueError("V12 formal qualifier requires exact screen PASS")
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V12M4_COMPLETE_FORMAL_SCORE_PASS"
    ):
        raise ValueError("V12 formal qualifier requires complete formal scores")
    if not (
        scores.get("equal_seed_mixture") is True
        and scores.get("partial_fold_scores_inspected") is False
        and scores.get("external_outcome_accessed") is False
        and scores.get("model_or_threshold_selection_performed") is False
    ):
        raise ValueError("V12 formal score violates the frozen protocol")
    rows = sorted(scores.get("mixture_scores", []), key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("V12 formal qualifier requires mixture folds0-19")
    repeated = qualify_screen(
        {
            "schema_version": SCREEN_SCORE_SCHEMA,
            "status": "V12M3_COMPLETE_SCORE_PASS",
            "scores": rows,
        }
    )
    individual = scores.get("individual_seed_scores", {})
    if sorted(map(int, individual)) != list(range(5)):
        raise ValueError("V12 formal qualifier requires individual seeds0-4")
    seed_directions = {}
    positive_counts = {
        "signed_delta": 0,
        "point_absolute_delta": 0,
        "task_crps": 0,
        "distribution_absolute_delta": 0,
    }
    for seed in range(5):
        seed_rows = sorted(individual[str(seed)], key=lambda row: int(row["outer_fold"]))
        if len(seed_rows) != 20 or [
            int(row["outer_fold"]) for row in seed_rows
        ] != list(range(20)):
            raise ValueError(f"V12 formal seed{seed} lacks twenty folds")
        summaries = {
            "signed_delta": paired_summary(
                seed_rows, "feature41_signed_delta_mae", "candidate_signed_delta_mae"
            ),
            "point_absolute_delta": paired_summary(
                seed_rows,
                "feature41_absolute_delta_mae",
                "candidate_point_absolute_delta_mae",
            ),
            "task_crps": paired_summary(seed_rows, "feature41_crps", "candidate_crps"),
            "distribution_absolute_delta": paired_summary(
                seed_rows,
                "feature41_absolute_delta_mae",
                "candidate_distribution_absolute_delta_mae",
            ),
        }
        seed_directions[str(seed)] = {}
        for name, summary in summaries.items():
            positive = summary["mean_gain"] > 0.0
            positive_counts[name] += int(positive)
            seed_directions[str(seed)][f"{name}_mean_gain"] = summary["mean_gain"]
            seed_directions[str(seed)][f"{name}_positive"] = positive
    gates = {
        "screen_prerequisite_exact_pass": True,
        "mixture_repeats_every_screen_gate": repeated["gate_passed"] is True,
        **{f"mixture_{name}": passed for name, passed in repeated["gates"].items()},
        "signed_positive_individual_seeds_ge_4": positive_counts["signed_delta"] >= 4,
        "point_absolute_positive_individual_seeds_ge_4": positive_counts[
            "point_absolute_delta"
        ]
        >= 4,
        "task_crps_positive_individual_seeds_ge_4": positive_counts["task_crps"] >= 4,
        "distribution_absolute_positive_individual_seeds_ge_4": positive_counts[
            "distribution_absolute_delta"
        ]
        >= 4,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V12M4",
        "status": (
            "V12M4_TOP_JOURNAL_FORMAL_PASS"
            if passed
            else "V12M4_TOP_JOURNAL_FORMAL_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": repeated["comparisons"],
        "calibration": repeated["calibration"],
        "individual_seed_directions": seed_directions,
        "positive_seed_counts": positive_counts,
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
        json.loads(args.score_json.read_text()),
        json.loads(args.screen_qualification_json.read_text()),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
