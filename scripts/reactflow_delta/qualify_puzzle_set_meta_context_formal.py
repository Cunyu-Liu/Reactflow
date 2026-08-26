#!/usr/bin/env python3
"""Apply the predeclared Puzzle-Set five-seed top-journal Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.qualify_model_rescue_v10 import paired_summary
from scripts.reactflow_delta.qualify_puzzle_set_meta_context import (
    SCHEMA as SCREEN_QUALIFICATION_SCHEMA,
    qualify as qualify_screen,
)
from scripts.reactflow_delta.score_puzzle_set_meta_context import (
    SCHEMA as SCREEN_SCORE_SCHEMA,
)
from scripts.reactflow_delta.score_puzzle_set_meta_context_formal import (
    SCHEMA as FORMAL_SCORE_SCHEMA,
)


SCHEMA = "reactflow_delta.puzzle_set_meta_context_formal_qualification.proposed.v2"


def _complete_seed_rows(individual: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = sorted(individual[str(seed)], key=lambda row: int(row["outer_fold"]))
    if len(rows) != 20 or [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError(f"puzzle-set formal seed{seed} lacks unique folds0-19")
    return rows


def qualify(scores: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
    if (
        screen.get("schema_version") != SCREEN_QUALIFICATION_SCHEMA
        or screen.get("status") != "PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_PASS"
        or screen.get("gate_passed") is not True
        or screen.get("puzzle_set_m4_authorized") is not True
    ):
        raise ValueError("puzzle-set formal qualifier requires exact P1M3 PASS")
    if (
        scores.get("schema_version") != FORMAL_SCORE_SCHEMA
        or scores.get("status") != "PUZZLE_SET_M4_COMPLETE_FORMAL_SCORE_PASS"
    ):
        raise ValueError("puzzle-set formal qualifier requires complete formal score")
    if not (
        scores.get("equal_seed_mixture") is True
        and scores.get("best_seed_selection_performed") is False
        and scores.get("target_profile_identity") == "EXACT_PUZZLE_METHOD_MUTATION"
        and scores.get("v13_parent_and_feature41_replay_at_5e_7") is True
        and scores.get("feature41_reference_fixed_across_seeds") is True
        and scores.get(
            "formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources"
        )
        is True
        and scores.get("partial_fold_scores_inspected") is False
        and scores.get("external_outcome_accessed") is False
        and scores.get("model_or_threshold_selection_performed") is False
    ):
        raise ValueError("puzzle-set formal score violates the frozen protocol")

    mixture_rows = sorted(
        scores.get("mixture_scores", []), key=lambda row: int(row["outer_fold"])
    )
    if len(mixture_rows) != 20 or [
        int(row["outer_fold"]) for row in mixture_rows
    ] != list(range(20)):
        raise ValueError("puzzle-set formal qualifier requires mixture folds0-19")
    screen_equivalent = {
        "schema_version": SCREEN_SCORE_SCHEMA,
        "status": "PUZZLE_SET_M3_COMPLETE_SCORE_PASS",
        "scores": mixture_rows,
        "context_retention_summary": scores.get("context_retention_summary", {}),
    }
    mixture_result = qualify_screen(screen_equivalent)

    individual = scores.get("individual_seed_scores", {})
    if set(individual) != {str(seed) for seed in range(5)}:
        raise ValueError("puzzle-set formal qualifier requires exactly seeds0-4")
    seed_directions: dict[str, dict[str, Any]] = {}
    signed_positive_seeds = 0
    crps_positive_seeds = 0
    individual_integrity = True
    for seed in range(5):
        rows = _complete_seed_rows(individual, seed)
        individual_integrity = individual_integrity and all(
            float(row["registered_prediction_coverage"]) == 1.0
            and float(row["failure_rate"]) == 0.0
            and int(row["n_unexpected_prediction_keys"]) == 0
            for row in rows
        )
        signed = paired_summary(
            rows, "feature41_signed_delta_mae", "candidate_signed_delta_mae"
        )
        crps = paired_summary(rows, "feature41_crps", "candidate_crps")
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
        **mixture_result["gates"],
        "screen_prerequisite_exact_pass": True,
        "individual_seed_prediction_integrity": individual_integrity,
        "formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources": (
            True
        ),
        "signed_positive_individual_seeds_ge_4": signed_positive_seeds >= 4,
        "task_crps_positive_individual_seeds_ge_4": crps_positive_seeds >= 4,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "P1M4",
        "status": (
            "PUZZLE_SET_M4_TOP_JOURNAL_FORMAL_PASS"
            if passed
            else "PUZZLE_SET_M4_TOP_JOURNAL_FORMAL_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "comparisons": mixture_result["comparisons"],
        "calibration": mixture_result["calibration"],
        "individual_seed_directions": seed_directions,
        "target_profile_identity_exact": True,
        "model_or_threshold_selection_performed": False,
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
    if args.out_json.exists():
        raise FileExistsError(
            "puzzle-set refuses to overwrite its formal qualification"
        )
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
