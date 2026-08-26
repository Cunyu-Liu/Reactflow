#!/usr/bin/env python3
"""Merge only a complete target-free puzzle-set fold universe."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr

from scripts.reactflow_delta.puzzle_set_meta_context import (
    BLOCK_DIAGONAL_NULL,
    FULL_CROSS_CONSTRUCT,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    EXPECTED_RESIDUAL_PARAMETERS,
)
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import FOLD_SCHEMA


MERGED_SCHEMA = "reactflow_delta.puzzle_set_meta_context_merged.proposed.v2"
FOLD_FILENAME = re.compile(
    r"puzzle_set_fold_result_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.json"
)
PREDICTION_FIELDS = {
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


def prediction_checks(
    path: Path, *, fold: int, seed: int, expected_rows: int
) -> tuple[dict[str, bool], list[str]]:
    with np.load(path, allow_pickle=True) as handle:
        names = set(handle.files)
        keys = list(map(str, handle["keys"])) if "keys" in names else []
        aligned = names >= PREDICTION_FIELDS and all(
            len(handle[name]) == len(keys)
            for name in PREDICTION_FIELDS - {"schema_version"}
        )
        distribution_shapes = PREDICTION_FIELDS <= names and all(
            handle[f"{name}_{suffix}"].shape == (len(keys), 2)
            for name in ("candidate", "null")
            for suffix in ("weights", "locations", "scales")
        ) and all(
            handle[f"{name}_expected_absolute_delta"].shape == (len(keys),)
            for name in ("candidate", "null")
        )
        distribution_valid = distribution_shapes and all(
            np.all(handle[f"{name}_scales"] > 0.0)
            and np.allclose(
                handle[f"{name}_weights"].sum(axis=1),
                1.0,
                atol=1e-7,
                rtol=0.0,
            )
            for name in ("candidate", "null")
        )
        median_preserved = distribution_valid and all(
            np.allclose(
                np.sum(
                    handle[f"{name}_weights"]
                    * ndtr(
                        (
                            handle[f"{name}_point"][:, None]
                            - handle[f"{name}_locations"]
                        )
                        / handle[f"{name}_scales"]
                    ),
                    axis=1,
                ),
                0.5,
                atol=3e-6,
                rtol=0.0,
            )
            for name in ("candidate", "null")
        )
        checks = {
            "schema": "schema_version" in names
            and str(handle["schema_version"].item()) == PREDICTION_SCHEMA,
            "exact_fields": names == PREDICTION_FIELDS,
            "target_free": not bool(names & FORBIDDEN_PREDICTION_FIELDS),
            "expected_rows": int(expected_rows) > 0
            and len(keys) == int(expected_rows),
            "aligned_rows": aligned,
            "unique_keys": len(keys) == len(set(keys)),
            "biological_key_match": "biological_scoring_key" in names
            and keys == list(map(str, handle["biological_scoring_key"])),
            "fold": "outer_fold" in names
            and set(map(int, handle["outer_fold"])) == {int(fold)},
            "seed": "seed" in names
            and set(map(int, handle["seed"])) == {int(seed)},
            "covered": "registered_status" in names
            and set(map(str, handle["registered_status"])) == {"covered"},
            "finite": all(
                np.isfinite(handle[name]).all()
                for name in names
                if handle[name].dtype.kind in "fiu"
            ),
            "distribution_shapes": distribution_shapes,
            "distribution_valid": distribution_valid,
            "median_preserved": median_preserved,
        }
    return checks, keys


def recorded_invariants_pass(invariants: dict[str, Any]) -> bool:
    required_true = (
        "outcome_blind_puzzle_set_inputs",
        "exact_parameter_and_initialization_match",
        "candidate_full_cross_construct_attention",
        "null_block_diagonal_attention",
        "puzzle_balanced_training",
        "point_frozen_during_calibration",
        "v10_residual_family_reused",
        "puzzle_balanced_residual_calibration",
        "median_constraint_all_held_rows",
        "prediction_target_free",
    )
    required_false = ("held_score_computed", "external_outcome_accessed")
    return all(invariants.get(name) is True for name in required_true) and all(
        invariants.get(name) is False for name in required_false
    )


def merge_complete_universe(
    input_dir: Path,
    *,
    expected_folds: list[int],
    expected_seeds: list[int],
    expected_point_epochs: int,
    expected_calibration_epochs: int,
    expected_parameter_count: int,
) -> dict[str, Any]:
    if len(set(expected_folds)) != len(expected_folds) or len(
        set(expected_seeds)
    ) != len(expected_seeds):
        raise ValueError("expected puzzle-set fold or seed universe is duplicated")
    expected = {
        (int(fold), int(seed))
        for seed in expected_seeds
        for fold in expected_folds
    }
    if not expected:
        raise ValueError("expected puzzle-set universe cannot be empty")
    rows = []
    seen: set[tuple[int, int]] = set()
    keys_by_seed: dict[int, set[str]] = {
        int(seed): set() for seed in expected_seeds
    }
    for path in sorted(input_dir.glob("puzzle_set_fold_result_fold*_seed*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        pair = (int(row.get("outer_fold", -1)), int(row.get("seed", -1)))
        filename_match = FOLD_FILENAME.fullmatch(path.name)
        if filename_match is None or pair != (
            int(filename_match.group("fold")),
            int(filename_match.group("seed")),
        ):
            raise ValueError(f"puzzle-set fold identity disagrees with {path.name}")
        if pair not in expected:
            raise ValueError(f"unexpected puzzle-set fold-seed pair {pair}")
        if pair in seen:
            raise ValueError(f"duplicate puzzle-set fold-seed pair {pair}")
        seen.add(pair)
        if row.get("schema_version") != FOLD_SCHEMA:
            raise ValueError(f"invalid puzzle-set fold schema in {path}")
        if int(row.get("point_epochs", -1)) != int(
            expected_point_epochs
        ) or int(row.get("calibration_epochs", -1)) != int(
            expected_calibration_epochs
        ):
            raise ValueError(f"puzzle-set fold {pair} violates the epoch freeze")
        if row.get("candidate_connectivity") != FULL_CROSS_CONSTRUCT or row.get(
            "null_connectivity"
        ) != BLOCK_DIAGONAL_NULL:
            raise ValueError(f"puzzle-set fold {pair} changed connectivity")
        if int(row.get("candidate_parameter_count", -1)) != int(
            expected_parameter_count
        ) or int(row.get("null_parameter_count", -1)) != int(
            expected_parameter_count
        ):
            raise ValueError(f"puzzle-set fold {pair} changed parameter count")
        histories = row.get("training_histories", {})
        expected_history_lengths = {
            "candidate_point": expected_point_epochs,
            "null_point": expected_point_epochs,
            "candidate_residual": expected_calibration_epochs,
            "null_residual": expected_calibration_epochs,
        }
        for history_name, history_length in expected_history_lengths.items():
            history = np.asarray(histories.get(history_name, []), dtype=float)
            if len(history) != history_length or not np.isfinite(history).all():
                raise ValueError(f"puzzle-set fold {pair} has invalid {history_name}")
        if set(map(int, row.get("residual_parameter_counts", {}).values())) != {
            EXPECTED_RESIDUAL_PARAMETERS
        }:
            raise ValueError(f"puzzle-set fold {pair} changed residual family")
        if not recorded_invariants_pass(row.get("invariants", {})):
            raise ValueError(f"puzzle-set fold {pair} lacks required invariants")
        checks, prediction_keys = prediction_checks(
            Path(row["prediction_artifact"]),
            fold=pair[0],
            seed=pair[1],
            expected_rows=int(row["n_registered_prediction_rows"]),
        )
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(
                f"puzzle-set fold {pair} prediction checks failed: {failed}"
            )
        overlap = keys_by_seed[pair[1]] & set(prediction_keys)
        if overlap:
            raise ValueError(
                f"puzzle-set seed {pair[1]} repeats biological keys across folds"
            )
        keys_by_seed[pair[1]].update(prediction_keys)
        point_checkpoints = row.get("point_checkpoints", {})
        residual_checkpoints = row.get("residual_checkpoints", {})
        checkpoint_paths = [
            *point_checkpoints.values(),
            *residual_checkpoints.values(),
        ]
        if (
            set(point_checkpoints) != {"candidate", "null"}
            or set(residual_checkpoints) != {"candidate", "null"}
            or len(checkpoint_paths) != 4
            or not all(Path(value).is_file() for value in checkpoint_paths)
        ):
            raise FileNotFoundError(f"puzzle-set fold {pair} lacks checkpoints")
        rows.append(row)
    if seen != expected or len(rows) != len(expected):
        raise ValueError(
            "puzzle-set fold universe is incomplete or unexpected: "
            f"found={sorted(seen)} expected={sorted(expected)}"
        )
    rows.sort(key=lambda row: (int(row["seed"]), int(row["outer_fold"])))
    return {
        "schema_version": MERGED_SCHEMA,
        "status": "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS",
        "expected_folds": sorted(map(int, expected_folds)),
        "expected_seeds": sorted(map(int, expected_seeds)),
        "expected_point_epochs": int(expected_point_epochs),
        "expected_calibration_epochs": int(expected_calibration_epochs),
        "expected_parameter_count_each": int(expected_parameter_count),
        "expected_residual_parameter_count_each": EXPECTED_RESIDUAL_PARAMETERS,
        "folds": rows,
        "merge_integrity": {
            "complete_fold_seed_universe": True,
            "unique_fold_seed_pairs": True,
            "prediction_only_schema": True,
            "outcome_blind_puzzle_set_inputs_all_runs": True,
            "exact_parameter_and_initialization_match_all_runs": True,
            "candidate_full_cross_construct_attention_all_runs": True,
            "null_block_diagonal_attention_all_runs": True,
            "puzzle_balanced_training_all_runs": True,
            "point_frozen_during_calibration_all_runs": True,
            "v10_residual_family_all_runs": True,
            "puzzle_balanced_residual_calibration_all_runs": True,
            "median_constraint_all_runs": True,
            "partial_scores_inspected": False,
            "external_outcome_accessed": False,
        },
    }


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--point-epochs", type=int, required=True)
    parser.add_argument("--calibration-epochs", type=int, required=True)
    parser.add_argument("--parameter-count", type=int, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError("refusing to overwrite puzzle-set complete merge")
    result = merge_complete_universe(
        args.input_dir,
        expected_folds=_csv_ints(args.folds),
        expected_seeds=_csv_ints(args.seeds),
        expected_point_epochs=args.point_epochs,
        expected_calibration_epochs=args.calibration_epochs,
        expected_parameter_count=args.parameter_count,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
