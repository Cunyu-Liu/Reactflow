#!/usr/bin/env python3
"""Merge complete score-blind V13 fold universes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.model_rescue_v13 import (
    EXPECTED_POINT_PARAMETERS,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.run_model_rescue_v13 import FOLD_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v13_merged.v1"
FORBIDDEN_PREDICTION_FIELDS = {
    "target",
    "target_error",
    "qualified_target_mask",
    "qualified_mask",
    "loss",
    "score",
}


def _expected_universe(phase: str) -> tuple[list[int], list[int], int, int, str]:
    if phase == "V13M2":
        return [0, 1], [0], 3, 3, "V13M2_COMPLETE_UNSCORED_SMOKE_MERGE_PASS"
    if phase == "V13M3":
        return list(range(20)), [0], 40, 40, "V13M3_COMPLETE_UNSCORED_MERGE_PASS"
    if phase == "V13M4":
        return list(range(20)), list(range(5)), 40, 40, "V13M4_COMPLETE_UNSCORED_MERGE_PASS"
    raise ValueError("unsupported V13 merge phase")


def prediction_checks(
    path: Path, *, fold: int, seed: int, expected_rows: int
) -> dict[str, bool]:
    with np.load(path, allow_pickle=True) as handle:
        names = set(handle.files)
        required = {
            "schema_version",
            "keys",
            "biological_scoring_key",
            "outer_fold",
            "seed",
            "registered_status",
            "feature41_point",
            "candidate_point",
            "null_point",
            "candidate_weights",
            "candidate_locations",
            "candidate_scales",
            "candidate_expected_absolute_delta",
            "null_weights",
            "null_locations",
            "null_scales",
            "null_expected_absolute_delta",
        }
        keys = list(map(str, handle["keys"])) if "keys" in names else []
        rows = len(keys)
        checks = {
            "schema": "schema_version" in names
            and str(handle["schema_version"].item()) == PREDICTION_SCHEMA,
            "required_fields": required <= names,
            "target_free": not bool(names & FORBIDDEN_PREDICTION_FIELDS),
            "expected_rows": rows == expected_rows,
            "unique_keys": len(keys) == len(set(keys)),
            "biological_key_match": "biological_scoring_key" in names
            and keys == list(map(str, handle["biological_scoring_key"])),
            "fold": "outer_fold" in names
            and set(map(int, handle["outer_fold"])) == {fold},
            "seed": "seed" in names and set(map(int, handle["seed"])) == {seed},
            "covered": "registered_status" in names
            and set(map(str, handle["registered_status"])) == {"covered"},
            "finite": all(
                np.isfinite(handle[name]).all()
                for name in names
                if handle[name].dtype.kind in "fiu"
            ),
            "null_delta": "null_hidden_delta_max_abs" in names
            and float(np.max(handle["null_hidden_delta_max_abs"])) <= 1e-7,
        }
    return checks


def recorded_invariants_pass(invariants: dict[str, Any]) -> bool:
    required_true = (
        "target_profile_identity_exact",
        "exact_point_parameter_and_initial_state_match",
        "second_pass_sequence_is_only_candidate_null_difference",
        "candidate_exact_mutant_null_wt_replay",
        "null_hidden_delta_at_most_1e_7",
        "same_point_training_order_and_dropout_seed",
        "paired_encoder_dropout_mask_shared",
        "point_frozen_during_calibration",
        "v10_residual_family_reused",
        "feature41_replay_at_1e_7",
        "median_constraint_all_held_rows",
    )
    required_false = (
        "held_score_computed",
        "prediction_contains_target_fields",
        "external_outcome_accessed",
    )
    return all(invariants.get(name) is True for name in required_true) and all(
        invariants.get(name) is False for name in required_false
    )


def merge_folds(input_dir: Path, phase: str) -> dict[str, Any]:
    expected_folds, expected_seeds, point_epochs, calibration_epochs, status = (
        _expected_universe(phase)
    )
    rows = []
    seen: set[tuple[int, int]] = set()
    point_counts: set[int] = set()
    for path in sorted(input_dir.glob("v13_fold_result_fold*_seed*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        fold = int(row.get("outer_fold", -1))
        seed = int(row.get("seed", -1))
        pair = (fold, seed)
        if pair in seen:
            raise ValueError(f"duplicate V13 fold-seed {pair}")
        seen.add(pair)
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != phase:
            raise ValueError(f"invalid V13 {phase} fold result in {path}")
        if int(row.get("point_epochs", -1)) != point_epochs or int(
            row.get("calibration_epochs", -1)
        ) != calibration_epochs:
            raise ValueError(f"V13 fold-seed {pair} violates epoch freeze")
        if not recorded_invariants_pass(row.get("invariants", {})):
            raise ValueError(f"V13 fold-seed {pair} lacks required invariants")
        counts = row.get("point_parameter_counts", {})
        if len(counts) != 2 or len(set(map(int, counts.values()))) != 1:
            raise ValueError(f"V13 fold-seed {pair} point parameters are unmatched")
        if set(map(int, counts.values())) != {EXPECTED_POINT_PARAMETERS}:
            raise ValueError(f"V13 fold-seed {pair} point parameter count changed")
        point_counts.update(map(int, counts.values()))
        if set(map(int, row.get("residual_parameter_counts", {}).values())) != {63748}:
            raise ValueError(f"V13 fold-seed {pair} residual family changed")
        checks = prediction_checks(
            Path(row["prediction_artifact"]),
            fold=fold,
            seed=seed,
            expected_rows=int(row["n_registered_prediction_rows"]),
        )
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(f"V13 fold-seed {pair} prediction checks failed: {failed}")
        checkpoints = {
            **row.get("point_checkpoints", {}),
            **row.get("residual_checkpoints", {}),
        }
        if not checkpoints or not all(Path(value).is_file() for value in checkpoints.values()):
            raise FileNotFoundError(f"V13 fold-seed {pair} lacks a checkpoint")
        rows.append(row)
    expected = {(fold, seed) for seed in expected_seeds for fold in expected_folds}
    if seen != expected or len(rows) != len(expected):
        raise ValueError(
            "V13 fold-seed universe is incomplete: "
            f"found={sorted(seen)} expected={sorted(expected)}"
        )
    if len(point_counts) != 1:
        raise ValueError("V13 point parameter count changed across runs")
    rows.sort(key=lambda row: (int(row["seed"]), int(row["outer_fold"])))
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "status": status,
        "folds": rows,
        "exact_point_parameter_count": next(iter(point_counts)),
        "merge_integrity": {
            "complete_fold_seed_universe": True,
            "unique_fold_seed_pairs": True,
            "prediction_only_schema": True,
            "target_identity_exact": True,
            "exact_point_parameter_and_initial_state_match_all_runs": True,
            "second_pass_only_difference_all_runs": True,
            "exact_mutant_vs_wt_replay_all_runs": True,
            "null_hidden_delta_at_most_1e_7_all_runs": True,
            "paired_encoder_dropout_mask_shared_all_runs": True,
            "point_frozen_during_calibration_all_runs": True,
            "v10_residual_family_all_runs": True,
            "feature41_replay_all_runs": True,
            "median_constraint_all_runs": True,
            "partial_scores_inspected": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("V13M2", "V13M3", "V13M4"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_folds(args.input_dir, args.phase)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
