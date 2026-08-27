#!/usr/bin/env python3
"""Merge complete score-blind V14 fold universes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.model_rescue_v14 import (
    EXPECTED_DOWNSTREAM_PARAMETERS,
    EXPECTED_TOTAL_PARAMETERS,
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v14_merged.v1"
FORBIDDEN_PREDICTION_FIELDS = {
    "target",
    "target_error",
    "qualified_target_mask",
    "qualified_mask",
    "loss",
    "score",
}


def _expected_universe(
    phase: str,
) -> tuple[list[int], list[int], int, int, int, str]:
    if phase == "V14M2":
        return [0, 1], [0], 3, 3, 3, "V14M2_COMPLETE_UNSCORED_SMOKE_MERGE_PASS"
    if phase == "V14M3":
        return list(range(20)), [0], 200, 40, 40, "V14M3_COMPLETE_UNSCORED_MERGE_PASS"
    if phase == "V14M4":
        return list(range(20)), list(range(5)), 200, 40, 40, "V14M4_COMPLETE_UNSCORED_MERGE_PASS"
    raise ValueError("unsupported V14 merge phase")


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
            "feature41_weights",
            "feature41_locations",
            "feature41_scales",
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
        checks = {
            "schema": "schema_version" in names
            and str(handle["schema_version"].item()) == PREDICTION_SCHEMA,
            "required_fields": required <= names,
            "target_free": not bool(names & FORBIDDEN_PREDICTION_FIELDS),
            "expected_rows": len(keys) == expected_rows,
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
        }
    return checks


def recorded_invariants_pass(invariants: dict[str, Any]) -> bool:
    required_true = (
        "target_profile_identity_exact",
        "outer_train_wt_only_pretraining",
        "zero_observed_constructs_excluded_from_pretraining",
        "held_puzzle_wt_excluded_from_pretraining",
        "mutant_outcome_excluded_from_pretraining",
        "exact_initial_parameter_match",
        "exact_total_and_downstream_parameter_match",
        "candidate_encoder_changed_during_pretraining",
        "null_state_unchanged_before_supervised_training",
        "residual_heads_identical_before_supervised_step_one",
        "pretraining_decoder_frozen_downstream",
        "same_point_training_order_and_dropout_stream",
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
    (
        expected_folds,
        expected_seeds,
        pretraining_epochs,
        point_epochs,
        calibration_epochs,
        status,
    ) = _expected_universe(phase)
    rows = []
    seen: set[tuple[int, int]] = set()
    for path in sorted(input_dir.glob("v14_fold_result_fold*_seed*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        fold = int(row.get("outer_fold", -1))
        seed = int(row.get("seed", -1))
        pair = (fold, seed)
        if pair in seen:
            raise ValueError(f"duplicate V14 fold-seed {pair}")
        seen.add(pair)
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != phase:
            raise ValueError(f"invalid V14 {phase} fold result in {path}")
        if (
            int(row.get("pretraining_epochs", -1)) != pretraining_epochs
            or int(row.get("point_epochs", -1)) != point_epochs
            or int(row.get("calibration_epochs", -1)) != calibration_epochs
        ):
            raise ValueError(f"V14 fold-seed {pair} violates epoch freeze")
        if int(row.get("n_registered_outer_train_wt_constructs", -1)) != 152:
            raise ValueError(f"V14 fold-seed {pair} registered WT universe changed")
        eligible = int(row.get("n_pretraining_constructs", -1))
        exclusions = row.get("zero_observed_pretraining_exclusions", [])
        if eligible not in (151, 152) or eligible + len(exclusions) != 152:
            raise ValueError(f"V14 fold-seed {pair} eligible WT universe changed")
        if exclusions not in ([], ["P20_Eterna"]):
            raise ValueError(f"V14 fold-seed {pair} zero-observed exclusion changed")
        expected_exclusions = [] if str(row.get("held_puzzle")) == "P20" else ["P20_Eterna"]
        expected_eligible = 152 - len(expected_exclusions)
        if exclusions != expected_exclusions or eligible != expected_eligible:
            raise ValueError(
                f"V14 fold-seed {pair} zero-observed exclusion is assigned to the wrong held puzzle"
            )
        if not recorded_invariants_pass(row.get("invariants", {})):
            raise ValueError(f"V14 fold-seed {pair} lacks required invariants")
        total = set(map(int, row.get("total_parameter_counts", {}).values()))
        downstream = set(map(int, row.get("point_parameter_counts", {}).values()))
        if total != {EXPECTED_TOTAL_PARAMETERS} or downstream != {
            EXPECTED_DOWNSTREAM_PARAMETERS
        }:
            raise ValueError(f"V14 fold-seed {pair} parameter contract changed")
        if set(map(int, row.get("residual_parameter_counts", {}).values())) != {63748}:
            raise ValueError(f"V14 fold-seed {pair} residual family changed")
        checks = prediction_checks(
            Path(row["prediction_artifact"]),
            fold=fold,
            seed=seed,
            expected_rows=int(row["n_registered_prediction_rows"]),
        )
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(f"V14 fold-seed {pair} prediction checks failed: {failed}")
        checkpoints = {
            **row.get("point_checkpoints", {}),
            **row.get("residual_checkpoints", {}),
        }
        if not checkpoints or not all(Path(value).is_file() for value in checkpoints.values()):
            raise FileNotFoundError(f"V14 fold-seed {pair} lacks a checkpoint")
        rows.append(row)
    expected = {(fold, seed) for seed in expected_seeds for fold in expected_folds}
    if seen != expected or len(rows) != len(expected):
        raise ValueError(
            "V14 fold-seed universe is incomplete: "
            f"found={sorted(seen)} expected={sorted(expected)}"
        )
    rows.sort(key=lambda row: (int(row["seed"]), int(row["outer_fold"])))
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "status": status,
        "folds": rows,
        "exact_total_parameter_count_each": EXPECTED_TOTAL_PARAMETERS,
        "exact_downstream_parameter_count_each": EXPECTED_DOWNSTREAM_PARAMETERS,
        "merge_integrity": {
            "complete_fold_seed_universe": True,
            "unique_fold_seed_pairs": True,
            "prediction_only_schema": True,
            "target_identity_exact": True,
            "outer_train_wt_only_pretraining_all_runs": True,
            "zero_observed_constructs_excluded_all_runs": True,
            "held_puzzle_wt_excluded_all_runs": True,
            "mutant_outcome_excluded_from_pretraining_all_runs": True,
            "exact_initial_and_parameter_match_all_runs": True,
            "candidate_encoder_changed_all_runs": True,
            "null_unchanged_before_supervision_all_runs": True,
            "residual_head_equal_before_supervision_all_runs": True,
            "decoder_frozen_downstream_all_runs": True,
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
    parser.add_argument("--phase", choices=("V14M2", "V14M3", "V14M4"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError(
            f"refusing to overwrite existing V14 merge artifact: {args.out_json}"
        )
    result = merge_folds(args.input_dir, args.phase)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
