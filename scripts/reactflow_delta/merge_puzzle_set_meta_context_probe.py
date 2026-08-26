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
    FULL_CROSS_CONSTRUCT,
    POINT_CONTEXT_LR,
    POINT_GRADIENT_CLIP,
    POINT_HEAD_LR,
    POINT_HEAD_WARMUP_EPOCHS,
    POSITION_ALIGNED_OPERATOR,
    POSITION_DERANGEMENT_SHIFT,
    POSITION_DERANGED_NULL,
)
from scripts.reactflow_delta.puzzle_set_meta_context_retention import (
    RETENTION_DIAGNOSTIC_EPOCH,
    RETENTION_SCHEMA,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    EXPECTED_RESIDUAL_PARAMETERS,
)
from scripts.reactflow_delta.puzzle_set_meta_context_pretraining import (
    EXPECTED_DECODER_PARAMETERS,
    EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS,
    PRETRAINING_MASK_FRACTION,
)
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import FOLD_SCHEMA


MERGED_SCHEMA = "reactflow_delta.puzzle_set_meta_context_merged.proposed.v8"
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
    "parent_point",
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
        distribution_shapes = (
            PREDICTION_FIELDS <= names
            and all(
                handle[f"{name}_{suffix}"].shape == (len(keys), 2)
                for name in ("candidate", "null")
                for suffix in ("weights", "locations", "scales")
            )
            and all(
                handle[f"{name}_expected_absolute_delta"].shape == (len(keys),)
                for name in ("candidate", "null")
            )
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
                        (handle[f"{name}_point"][:, None] - handle[f"{name}_locations"])
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
            "expected_rows": int(expected_rows) > 0 and len(keys) == int(expected_rows),
            "aligned_rows": aligned,
            "unique_keys": len(keys) == len(set(keys)),
            "biological_key_match": "biological_scoring_key" in names
            and keys == list(map(str, handle["biological_scoring_key"])),
            "fold": "outer_fold" in names
            and set(map(int, handle["outer_fold"])) == {int(fold)},
            "seed": "seed" in names and set(map(int, handle["seed"])) == {int(seed)},
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
        "candidate_nonfocal_only_cross_attention",
        "null_position_deranged_nonfocal_cross_attention",
        "candidate_null_equal_attention_support",
        "attention_weight_dropout_disabled",
        "puzzle_balanced_training",
        "position_aligned_nonfocal_cross_values",
        "nonfocal_summary_alignment_statistics",
        "matched_null_position_deranged_summary_statistics",
        "nonfocal_only_cross_values",
        "focal_excluded_from_cross_kv",
        "eight_token_cross_support",
        "paired_cross_block_reference_cancellation",
        "zero_nonfocal_exact_cross_replay",
        "paired_point_head_reference_cancellation",
        "zero_cross_exact_parent_replay",
        "fixed_position_derangement_shift_17",
        "outer_train_wt_only_puzzle_set_pretraining",
        "held_puzzle_excluded_from_pretraining",
        "mutant_outcome_excluded_from_pretraining",
        "candidate_null_equal_pretraining_budget",
        "pretraining_decoder_frozen_downstream",
        "encoder_and_point_unchanged_during_pretraining",
        "puzzle_coordinate_frames_validated",
        "frozen_v13_point_parent",
        "frozen_v14_context_encoder",
        "zero_initialized_parent_replay_at_1e_7",
        "point_head_only_warmup",
        "point_discriminative_learning_rates",
        "pretraining_capability_retention_diagnostic_complete",
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


def _validate_retention_diagnostics(
    row: dict[str, Any],
    *,
    pair: tuple[int, int],
    n_train_puzzles: int,
    outer_train_puzzle_ids: list[str],
    expected_pretraining_epochs: int,
) -> dict[str, dict[str, Any]]:
    diagnostics = row.get("context_retention_diagnostics", {})
    if set(diagnostics) != {"candidate", "null"}:
        raise ValueError(f"puzzle-set fold {pair} lacks retention diagnostics")
    validated: dict[str, dict[str, Any]] = {}
    for arm in ("candidate", "null"):
        diagnostic = diagnostics[arm]
        per_puzzle = diagnostic.get("per_puzzle", [])
        means = diagnostic.get("mean", {})
        if (
            diagnostic.get("schema_version") != RETENTION_SCHEMA
            or diagnostic.get("arm") != arm
            or diagnostic.get("evidence_status")
            != "OUTER_TRAIN_WT_RETENTION_DIAGNOSTIC_ONLY"
            or int(diagnostic.get("diagnostic_epoch", -1)) != RETENTION_DIAGNOSTIC_EPOCH
            or diagnostic.get("training_mask_epochs")
            != [0, int(expected_pretraining_epochs) - 1]
            or str(diagnostic.get("held_puzzle")) != str(row.get("held_puzzle"))
            or list(map(str, diagnostic.get("outer_train_puzzle_ids", [])))
            != sorted(outer_train_puzzle_ids)
            or not isinstance(per_puzzle, list)
            or len(per_puzzle) != n_train_puzzles
        ):
            raise ValueError(
                f"puzzle-set fold {pair} has malformed {arm} retention identity"
            )
        per_puzzle_ids = [str(item.get("puzzle")) for item in per_puzzle]
        metric_names = (
            "initial_context_l1",
            "post_pretraining_l1",
            "post_point_l1",
        )
        if (
            per_puzzle_ids != sorted(outer_train_puzzle_ids)
            or any(
                int(item.get("eligible_constructs", -1)) not in {7, 8}
                or any(
                    not np.isfinite(float(item.get(metric, np.nan)))
                    or float(item.get(metric, np.nan)) < 0.0
                    for metric in metric_names
                )
                for item in per_puzzle
            )
            or set(means) != set(metric_names)
            or any(not np.isfinite(float(means[metric])) for metric in metric_names)
            or any(
                not np.isclose(
                    float(means[metric]),
                    float(np.mean([float(item[metric]) for item in per_puzzle])),
                    atol=1e-12,
                    rtol=0.0,
                )
                for metric in metric_names
            )
        ):
            raise ValueError(
                f"puzzle-set fold {pair} has malformed {arm} retention values"
            )
        pretraining_gain = float(means["initial_context_l1"]) - float(
            means["post_pretraining_l1"]
        )
        expected_fraction = (
            None
            if abs(pretraining_gain) <= 1.0e-12
            else (float(means["initial_context_l1"]) - float(means["post_point_l1"]))
            / pretraining_gain
        )
        observed_fraction = diagnostic.get("retained_fraction")
        fraction_matches = (
            expected_fraction is None and observed_fraction is None
        ) or (
            expected_fraction is not None
            and observed_fraction is not None
            and np.isclose(
                float(observed_fraction), expected_fraction, atol=1e-12, rtol=0.0
            )
        )
        pretraining_established = pretraining_gain > 0.0
        retention_positive = bool(
            pretraining_established
            and expected_fraction is not None
            and expected_fraction > 0.0
        )
        if (
            not fraction_matches
            or diagnostic.get("pretraining_established") is not pretraining_established
            or diagnostic.get("retention_positive") is not retention_positive
            or diagnostic.get("same_final_frozen_decoder") is not True
            or diagnostic.get("diagnostic_mask_disjoint_from_training") is not True
            or diagnostic.get("mutant_outcome_used") is not False
            or diagnostic.get("held_puzzle_accessed") is not False
            or diagnostic.get("checkpoint_selection_performed") is not False
            or diagnostic.get("learning_rate_selection_performed") is not False
        ):
            raise ValueError(
                f"puzzle-set fold {pair} has inconsistent {arm} retention result"
            )
        validated[arm] = {
            "pretraining_established": pretraining_established,
            "retention_positive": retention_positive,
            "retained_fraction": expected_fraction,
        }
    return validated


def merge_complete_universe(
    input_dir: Path,
    *,
    expected_phase: str,
    expected_folds: list[int],
    expected_seeds: list[int],
    expected_pretraining_epochs: int,
    expected_point_epochs: int,
    expected_calibration_epochs: int,
    expected_parameter_count: int,
    expected_trainable_parameter_count: int,
) -> dict[str, Any]:
    if len(set(expected_folds)) != len(expected_folds) or len(
        set(expected_seeds)
    ) != len(expected_seeds):
        raise ValueError("expected puzzle-set fold or seed universe is duplicated")
    expected = {
        (int(fold), int(seed)) for seed in expected_seeds for fold in expected_folds
    }
    if not expected:
        raise ValueError("expected puzzle-set universe cannot be empty")
    rows = []
    retention_rows = []
    seen: set[tuple[int, int]] = set()
    keys_by_seed: dict[int, set[str]] = {int(seed): set() for seed in expected_seeds}
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
        if row.get("phase") != expected_phase:
            raise ValueError(f"puzzle-set fold {pair} changed phase")
        if (
            int(row.get("pretraining_epochs", -1)) != int(expected_pretraining_epochs)
            or int(row.get("point_epochs", -1)) != int(expected_point_epochs)
            or int(row.get("calibration_epochs", -1))
            != int(expected_calibration_epochs)
        ):
            raise ValueError(f"puzzle-set fold {pair} violates the epoch freeze")
        if (
            row.get("candidate_connectivity") != FULL_CROSS_CONSTRUCT
            or row.get("null_connectivity") != POSITION_DERANGED_NULL
            or int(row.get("position_derangement_shift", -1))
            != POSITION_DERANGEMENT_SHIFT
        ):
            raise ValueError(f"puzzle-set fold {pair} changed connectivity")
        if row.get("cross_construct_operator") != POSITION_ALIGNED_OPERATOR:
            raise ValueError(f"puzzle-set fold {pair} changed aligned operator")
        if int(row.get("candidate_parameter_count", -1)) != int(
            expected_parameter_count
        ) or int(row.get("null_parameter_count", -1)) != int(expected_parameter_count):
            raise ValueError(f"puzzle-set fold {pair} changed parameter count")
        trainable_counts = {
            int(row.get("candidate_trainable_parameter_count", -1)),
            int(row.get("null_trainable_parameter_count", -1)),
        }
        if trainable_counts != {int(expected_trainable_parameter_count)}:
            raise ValueError(f"puzzle-set fold {pair} changed trainable count")
        for replay_name in (
            "initial_parent_replay_max_abs_difference",
            "post_pretraining_parent_replay_max_abs_difference",
        ):
            replay = row.get(replay_name, {})
            if (
                set(replay) != {"candidate", "null"}
                or max(map(float, replay.values())) > 1e-7
            ):
                raise ValueError(
                    f"puzzle-set fold {pair} does not replay its parent at "
                    f"{replay_name}"
                )
        parents = row.get("frozen_parent_checkpoints", {})
        if set(parents) != {"v13_point", "v14_encoder"} or not all(
            Path(value).is_file() for value in parents.values()
        ):
            raise FileNotFoundError(f"puzzle-set fold {pair} lacks frozen parents")
        if int(row.get("n_validated_puzzle_coordinate_frames", 0)) <= 0:
            raise ValueError(f"puzzle-set fold {pair} lacks coordinate validation")
        n_train_puzzles = int(row.get("n_outer_train_puzzles", 0))
        n_pretraining_puzzles = int(row.get("n_pretraining_puzzles", 0))
        outer_train_puzzle_ids = list(map(str, row.get("outer_train_puzzle_ids", [])))
        pretraining_puzzle_ids = list(map(str, row.get("pretraining_puzzle_ids", [])))
        expected_eligible = {
            int(value)
            for value in row.get("expected_pretraining_eligible_construct_counts", [])
        }
        if (
            n_train_puzzles <= 0
            or n_pretraining_puzzles != n_train_puzzles
            or len(outer_train_puzzle_ids) != n_train_puzzles
            or len(set(outer_train_puzzle_ids)) != n_train_puzzles
            or sorted(pretraining_puzzle_ids) != sorted(outer_train_puzzle_ids)
            or str(row.get("held_puzzle")) in set(pretraining_puzzle_ids)
            or not expected_eligible
            or not expected_eligible <= {7, 8}
            or int(row.get("pretraining_optimizer_steps_each", -1))
            != int(expected_pretraining_epochs) * n_pretraining_puzzles
            or int(row.get("point_optimizer_steps_each", -1))
            != (int(expected_point_epochs) * n_train_puzzles)
            or int(row.get("residual_optimizer_steps_each", -1))
            != (int(expected_calibration_epochs) * n_train_puzzles)
        ):
            raise ValueError(
                f"puzzle-set fold {pair} changed optimizer-step accounting"
            )
        histories = row.get("training_histories", {})
        expected_history_lengths = {
            "candidate_pretraining": expected_pretraining_epochs,
            "null_pretraining": expected_pretraining_epochs,
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
        decoder_counts = row.get("pretraining_decoder_parameter_counts", {})
        if set(decoder_counts) != {"candidate", "null"} or set(
            map(int, decoder_counts.values())
        ) != {EXPECTED_DECODER_PARAMETERS}:
            raise ValueError(f"puzzle-set fold {pair} changed pretraining decoder")
        summaries = row.get("pretraining_summaries", {})
        if set(summaries) != {"candidate", "null"}:
            raise ValueError(f"puzzle-set fold {pair} lacks pretraining summaries")
        for arm, summary in summaries.items():
            eligible = {
                int(value) for value in summary.get("eligible_construct_counts", [])
            }
            if (
                int(summary.get("optimizer_steps", -1))
                != int(expected_pretraining_epochs) * n_pretraining_puzzles
                or int(summary.get("trainable_parameter_count", -1))
                != EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS
                or eligible != expected_eligible
                or not np.isclose(
                    float(summary.get("mask_fraction", np.nan)),
                    PRETRAINING_MASK_FRACTION,
                    atol=0.0,
                    rtol=0.0,
                )
                or summary.get("context_layers_changed") is not True
                or summary.get("encoder_changed") is not False
                or summary.get("point_head_changed") is not False
                or summary.get("decoder_frozen_downstream") is not True
                or summary.get("mutant_outcome_used") is not False
            ):
                raise ValueError(
                    f"puzzle-set fold {pair} has invalid {arm} pretraining summary"
                )
        point_summaries = row.get("point_training_summaries", {})
        if set(point_summaries) != {"candidate", "null"}:
            raise ValueError(f"puzzle-set fold {pair} lacks point-training summaries")
        expected_point_summary = {
            "optimizer_steps": int(expected_point_epochs) * n_train_puzzles,
            "head_update_steps": int(expected_point_epochs) * n_train_puzzles,
            "context_update_steps": max(
                int(expected_point_epochs) - POINT_HEAD_WARMUP_EPOCHS, 0
            )
            * n_train_puzzles,
            "target_exposures_per_available_cell": int(expected_point_epochs),
            "head_only_warmup_epochs": POINT_HEAD_WARMUP_EPOCHS,
            "head_learning_rate": POINT_HEAD_LR,
            "context_learning_rate": POINT_CONTEXT_LR,
            "gradient_clip": POINT_GRADIENT_CLIP,
            "warmup_context_unchanged": True,
            "best_epoch_selection_performed": False,
        }
        if any(
            point_summaries[arm] != expected_point_summary
            for arm in ("candidate", "null")
        ):
            raise ValueError(
                f"puzzle-set fold {pair} changed point warmup or optimizer schedule"
            )
        retention_rows.append(
            {
                "outer_fold": pair[0],
                "seed": pair[1],
                **_validate_retention_diagnostics(
                    row,
                    pair=pair,
                    n_train_puzzles=n_train_puzzles,
                    outer_train_puzzle_ids=outer_train_puzzle_ids,
                    expected_pretraining_epochs=expected_pretraining_epochs,
                ),
            }
        )
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
        decoder_checkpoints = row.get("pretraining_decoder_checkpoints", {})
        residual_checkpoints = row.get("residual_checkpoints", {})
        checkpoint_paths = [
            *point_checkpoints.values(),
            *decoder_checkpoints.values(),
            *residual_checkpoints.values(),
        ]
        if (
            set(point_checkpoints) != {"candidate", "null"}
            or set(decoder_checkpoints) != {"candidate", "null"}
            or set(residual_checkpoints) != {"candidate", "null"}
            or len(checkpoint_paths) != 6
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
    retention_rows.sort(key=lambda row: (int(row["seed"]), int(row["outer_fold"])))
    context_retention_summary = {
        "candidate_pretraining_established_all_runs": all(
            item["candidate"]["pretraining_established"] for item in retention_rows
        ),
        "candidate_retention_positive_all_runs": all(
            item["candidate"]["retention_positive"] for item in retention_rows
        ),
        "null_pretraining_established_all_runs": all(
            item["null"]["pretraining_established"] for item in retention_rows
        ),
        "null_retention_positive_all_runs": all(
            item["null"]["retention_positive"] for item in retention_rows
        ),
        "fold_seed_diagnostics": retention_rows,
        "selection_performed": False,
        "mutant_outcome_used": False,
        "held_puzzle_accessed": False,
    }
    context_retention_gate_passed = bool(
        context_retention_summary["candidate_pretraining_established_all_runs"]
        and context_retention_summary["candidate_retention_positive_all_runs"]
    )
    retention_gate_required = expected_phase in {"P1M3", "P1M4"}
    merge_status = (
        "PUZZLE_SET_TRAIN_ONLY_RETENTION_GATE_FAIL"
        if retention_gate_required and not context_retention_gate_passed
        else "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
    )
    return {
        "schema_version": MERGED_SCHEMA,
        "status": merge_status,
        "phase": expected_phase,
        "expected_folds": sorted(map(int, expected_folds)),
        "expected_seeds": sorted(map(int, expected_seeds)),
        "expected_pretraining_epochs": int(expected_pretraining_epochs),
        "expected_point_epochs": int(expected_point_epochs),
        "expected_calibration_epochs": int(expected_calibration_epochs),
        "expected_parameter_count_each": int(expected_parameter_count),
        "expected_trainable_parameter_count_each": int(
            expected_trainable_parameter_count
        ),
        "expected_residual_parameter_count_each": EXPECTED_RESIDUAL_PARAMETERS,
        "expected_pretraining_decoder_parameter_count_each": (
            EXPECTED_DECODER_PARAMETERS
        ),
        "expected_pretraining_trainable_parameter_count_each": (
            EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS
        ),
        "folds": rows,
        "context_retention_gate_required": retention_gate_required,
        "context_retention_gate_passed": context_retention_gate_passed,
        "context_retention_summary": context_retention_summary,
        "merge_integrity": {
            "complete_fold_seed_universe": True,
            "unique_fold_seed_pairs": True,
            "prediction_only_schema": True,
            "outcome_blind_puzzle_set_inputs_all_runs": True,
            "exact_parameter_and_initialization_match_all_runs": True,
            "candidate_nonfocal_only_cross_attention_all_runs": True,
            "null_position_deranged_nonfocal_cross_attention_all_runs": True,
            "candidate_null_equal_attention_support_all_runs": True,
            "attention_weight_dropout_disabled_all_runs": True,
            "puzzle_balanced_training_all_runs": True,
            "position_aligned_nonfocal_cross_values_all_runs": True,
            "nonfocal_summary_alignment_statistics_all_runs": True,
            "matched_null_position_deranged_summary_statistics_all_runs": True,
            "nonfocal_only_cross_values_all_runs": True,
            "focal_excluded_from_cross_kv_all_runs": True,
            "eight_token_cross_support_all_runs": True,
            "paired_cross_block_reference_cancellation_all_runs": True,
            "zero_nonfocal_exact_cross_replay_all_runs": True,
            "paired_point_head_reference_cancellation_all_runs": True,
            "zero_cross_exact_parent_replay_all_runs": True,
            "fixed_position_derangement_shift_17_all_runs": True,
            "outer_train_wt_only_puzzle_set_pretraining_all_runs": True,
            "held_puzzle_excluded_from_pretraining_all_runs": True,
            "mutant_outcome_excluded_from_pretraining_all_runs": True,
            "candidate_null_equal_pretraining_budget_all_runs": True,
            "pretraining_decoder_frozen_downstream_all_runs": True,
            "encoder_and_point_unchanged_during_pretraining_all_runs": True,
            "masked_wt_pretraining_protocol_all_runs": True,
            "puzzle_coordinate_frames_validated_all_runs": True,
            "frozen_v13_point_parent_all_runs": True,
            "frozen_v14_context_encoder_all_runs": True,
            "parent_replay_before_and_after_pretraining_all_runs": True,
            "point_head_only_warmup_all_runs": True,
            "point_discriminative_learning_rates_all_runs": True,
            "pretraining_capability_retention_diagnostic_complete_all_runs": True,
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
    parser.add_argument("--phase", choices=("P1M2", "P1M3", "P1M4"), required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--pretraining-epochs", type=int, required=True)
    parser.add_argument("--point-epochs", type=int, required=True)
    parser.add_argument("--calibration-epochs", type=int, required=True)
    parser.add_argument("--parameter-count", type=int, required=True)
    parser.add_argument("--trainable-parameter-count", type=int, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError("refusing to overwrite puzzle-set complete merge")
    result = merge_complete_universe(
        args.input_dir,
        expected_phase=args.phase,
        expected_folds=_csv_ints(args.folds),
        expected_seeds=_csv_ints(args.seeds),
        expected_pretraining_epochs=args.pretraining_epochs,
        expected_point_epochs=args.point_epochs,
        expected_calibration_epochs=args.calibration_epochs,
        expected_parameter_count=args.parameter_count,
        expected_trainable_parameter_count=args.trainable_parameter_count,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["status"] == "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
