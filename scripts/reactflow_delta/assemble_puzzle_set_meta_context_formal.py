#!/usr/bin/env python3
"""Assemble the frozen puzzle-set five-seed prediction mixture without scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    MERGED_SCHEMA,
    prediction_checks,
)
from scripts.reactflow_delta.model_rescue_v9 import expected_absolute_delta


SCHEMA = "reactflow_delta.puzzle_set_meta_context_formal_assembly.proposed.v2"
FORMAL_PREDICTION_SCHEMA = (
    "reactflow_delta.puzzle_set_meta_context_formal_prediction.proposed.v1"
)
ASSEMBLY_STATUS = "PUZZLE_SET_M4_FIVE_SEED_PREDICTION_ONLY_ASSEMBLY_PASS"
SOURCE_MERGE_STATUS = "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
EXPECTED_PHASE = "P1M4"
EXPECTED_FOLDS = tuple(range(20))
EXPECTED_SEEDS = tuple(range(5))
EXPECTED_PRETRAINING_EPOCHS = 200
EXPECTED_POINT_EPOCHS = 40
EXPECTED_CALIBRATION_EPOCHS = 40
DISTRIBUTION_NAMES = ("candidate", "null")
FORMAL_PREDICTION_FIELDS = {
    "schema_version",
    "keys",
    "biological_scoring_key",
    "outer_fold",
    "seed",
    "assembled_seed_count",
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

_REQUIRED_MERGE_INTEGRITY_TRUE = {
    "complete_fold_seed_universe",
    "unique_fold_seed_pairs",
    "prediction_only_schema",
    "outcome_blind_puzzle_set_inputs_all_runs",
    "exact_parameter_and_initialization_match_all_runs",
    "candidate_nonfocal_only_cross_attention_all_runs",
    "null_position_deranged_nonfocal_cross_attention_all_runs",
    "candidate_null_equal_attention_support_all_runs",
    "attention_weight_dropout_disabled_all_runs",
    "puzzle_balanced_training_all_runs",
    "position_aligned_nonfocal_cross_values_all_runs",
    "nonfocal_summary_alignment_statistics_all_runs",
    "matched_null_position_deranged_summary_statistics_all_runs",
    "nonfocal_only_cross_values_all_runs",
    "focal_excluded_from_cross_kv_all_runs",
    "eight_token_cross_support_all_runs",
    "paired_point_head_reference_cancellation_all_runs",
    "zero_cross_exact_parent_replay_all_runs",
    "paired_cross_block_reference_cancellation_all_runs",
    "zero_nonfocal_exact_cross_replay_all_runs",
    "fixed_position_derangement_shift_17_all_runs",
    "outer_train_wt_only_puzzle_set_pretraining_all_runs",
    "held_puzzle_excluded_from_pretraining_all_runs",
    "mutant_outcome_excluded_from_pretraining_all_runs",
    "candidate_null_equal_pretraining_budget_all_runs",
    "pretraining_decoder_frozen_downstream_all_runs",
    "encoder_and_point_unchanged_during_pretraining_all_runs",
    "masked_wt_pretraining_protocol_all_runs",
    "puzzle_coordinate_frames_validated_all_runs",
    "frozen_v13_point_parent_all_runs",
    "frozen_v14_context_encoder_all_runs",
    "complete_frozen_input_provenance_all_runs",
    "parent_replay_before_and_after_pretraining_all_runs",
    "point_head_only_warmup_all_runs",
    "point_discriminative_learning_rates_all_runs",
    "pretraining_capability_retention_diagnostic_complete_all_runs",
    "point_frozen_during_calibration_all_runs",
    "v10_residual_family_all_runs",
    "puzzle_balanced_residual_calibration_all_runs",
    "median_constraint_all_runs",
}


def _expected_absolute(
    weights: np.ndarray, locations: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        result = expected_absolute_delta(
            torch.as_tensor(weights, dtype=torch.float64),
            torch.as_tensor(locations, dtype=torch.float64),
            torch.as_tensor(scales, dtype=torch.float64),
        )
    return result.cpu().numpy()


def _validate_complete_merge(merged: dict[str, Any]) -> None:
    if (
        merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("status") != SOURCE_MERGE_STATUS
        or merged.get("phase") != EXPECTED_PHASE
    ):
        raise ValueError("formal puzzle-set assembly requires a complete P1M4 merge")
    freezes = {
        "expected_folds": list(EXPECTED_FOLDS),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_pretraining_epochs": EXPECTED_PRETRAINING_EPOCHS,
        "expected_point_epochs": EXPECTED_POINT_EPOCHS,
        "expected_calibration_epochs": EXPECTED_CALIBRATION_EPOCHS,
    }
    for field, expected in freezes.items():
        if merged.get(field) != expected:
            raise ValueError(f"formal puzzle-set merge changed frozen {field}")

    integrity = merged.get("merge_integrity")
    if not isinstance(integrity, dict) or not _REQUIRED_MERGE_INTEGRITY_TRUE <= set(
        integrity
    ):
        raise ValueError("formal puzzle-set merge lacks required integrity evidence")
    if any(integrity[field] is not True for field in _REQUIRED_MERGE_INTEGRITY_TRUE):
        raise ValueError("formal puzzle-set merge contains a failed integrity check")
    if (
        integrity.get("partial_scores_inspected") is not False
        or integrity.get("external_outcome_accessed") is not False
    ):
        raise ValueError(
            "formal puzzle-set merge is not score-blind and outcome-isolated"
        )

    rows = merged.get("folds")
    if not isinstance(rows, list):
        raise ValueError("formal puzzle-set merge has no fold rows")
    pairs: list[tuple[int, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("formal puzzle-set fold row is malformed")
        pairs.append((int(row.get("outer_fold", -1)), int(row.get("seed", -1))))
    expected_pairs = {
        (fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS
    }
    if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
        raise ValueError(
            "formal puzzle-set assembly requires the unique 20-fold x five-seed universe"
        )


def _load_prediction(
    row: dict[str, Any], *, fold: int, seed: int
) -> dict[str, np.ndarray]:
    if int(row.get("outer_fold", -1)) != fold or int(row.get("seed", -1)) != seed:
        raise ValueError(f"formal puzzle-set fold {fold} source identity changed")
    expected_rows = int(row.get("n_registered_prediction_rows", -1))
    path = Path(row["prediction_artifact"])
    checks, _ = prediction_checks(
        path,
        fold=fold,
        seed=seed,
        expected_rows=expected_rows,
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(
            f"formal puzzle-set fold {fold} seed{seed} prediction failed {failed}"
        )
    with np.load(path, allow_pickle=True) as handle:
        return {name: np.asarray(handle[name]).copy() for name in handle.files}


def _ordered_sources(
    rows: list[dict[str, Any]], *, fold: int
) -> list[dict[str, np.ndarray]]:
    seeds = [int(row.get("seed", -1)) for row in rows]
    if len(seeds) != len(EXPECTED_SEEDS) or sorted(seeds) != list(EXPECTED_SEEDS):
        raise ValueError(f"formal puzzle-set fold {fold} requires unique seeds0-4")
    by_seed = {int(row["seed"]): row for row in rows}
    return [
        _load_prediction(by_seed[seed], fold=fold, seed=seed) for seed in EXPECTED_SEEDS
    ]


def _validate_cross_seed_alignment(
    sources: list[dict[str, np.ndarray]], *, fold: int
) -> list[str]:
    keys = list(map(str, sources[0]["keys"]))
    for seed, prediction in enumerate(sources[1:], start=1):
        if list(map(str, prediction["keys"])) != keys:
            raise ValueError(
                f"formal puzzle-set fold {fold} seed{seed} key order differs"
            )
        for name in ("feature41_point", "parent_point"):
            if not np.allclose(
                np.asarray(prediction[name], dtype=np.float64),
                np.asarray(sources[0][name], dtype=np.float64),
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    f"formal puzzle-set fold {fold} {name} differs by seed"
                )
    return keys


def assemble_fold_prediction_arrays(
    sources: list[dict[str, np.ndarray]], *, fold: int
) -> dict[str, np.ndarray]:
    """Purely assemble one formal fold from ordered seed prediction arrays."""

    if len(sources) != len(EXPECTED_SEEDS):
        raise ValueError(f"formal puzzle-set fold {fold} requires five source arrays")
    keys = _validate_cross_seed_alignment(sources, fold=fold)
    n_rows = len(keys)
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FORMAL_PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(n_rows, fold, dtype=np.int64),
        "seed": np.full(n_rows, -1, dtype=np.int64),
        "assembled_seed_count": np.full(n_rows, len(EXPECTED_SEEDS), dtype=np.int64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
        "feature41_point": np.asarray(sources[0]["feature41_point"], dtype=np.float64),
        # V13 parent is a fixed seed-0 comparator, not an ensemble member.
        "parent_point": np.asarray(sources[0]["parent_point"], dtype=np.float64),
        "candidate_point": np.mean(
            [
                np.asarray(source["candidate_point"], dtype=np.float64)
                for source in sources
            ],
            axis=0,
        ),
        "null_point": np.mean(
            [np.asarray(source["null_point"], dtype=np.float64) for source in sources],
            axis=0,
        ),
    }

    for name in DISTRIBUTION_NAMES:
        weights = np.concatenate(
            [
                np.asarray(source[f"{name}_weights"], dtype=np.float64)
                / len(EXPECTED_SEEDS)
                for source in sources
            ],
            axis=1,
        )
        locations = np.concatenate(
            [
                np.asarray(source[f"{name}_locations"], dtype=np.float64)
                for source in sources
            ],
            axis=1,
        )
        scales = np.concatenate(
            [
                np.asarray(source[f"{name}_scales"], dtype=np.float64)
                for source in sources
            ],
            axis=1,
        )
        expected_shape = (n_rows, 2 * len(EXPECTED_SEEDS))
        if not all(
            array.shape == expected_shape for array in (weights, locations, scales)
        ):
            raise ValueError(
                f"formal puzzle-set {name} distribution must contain ten components"
            )
        if (
            not np.isfinite(weights).all()
            or not np.isfinite(locations).all()
            or not np.isfinite(scales).all()
            or np.any(weights < 0.0)
            or np.any(scales <= 0.0)
        ):
            raise ValueError(f"formal puzzle-set {name} mixture is invalid")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
            raise ValueError(f"formal puzzle-set {name} weights do not sum to one")
        for seed in EXPECTED_SEEDS:
            seed_slice = slice(2 * seed, 2 * seed + 2)
            if not np.allclose(
                weights[:, seed_slice].sum(axis=1),
                1.0 / len(EXPECTED_SEEDS),
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    f"formal puzzle-set {name} seed{seed} does not have equal mass"
                )
        output[f"{name}_weights"] = weights
        output[f"{name}_locations"] = locations
        output[f"{name}_scales"] = scales
        output[f"{name}_expected_absolute_delta"] = _expected_absolute(
            weights, locations, scales
        )

    if set(output) != FORMAL_PREDICTION_FIELDS:
        raise RuntimeError(
            "formal puzzle-set array assembly changed its field universe"
        )
    return output


def assemble_fold(
    rows: list[dict[str, Any]], *, fold: int, out_dir: Path
) -> dict[str, Any]:
    """Assemble one fold from all five seeds, preserving the frozen parents."""

    sources = _ordered_sources(rows, fold=fold)
    output = assemble_fold_prediction_arrays(sources, fold=fold)
    n_rows = len(output["keys"])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"puzzle_set_formal_predictions_fold{fold}_seeds0_4.npz"
    np.savez_compressed(path, **output)
    return {
        "outer_fold": fold,
        "seeds": list(EXPECTED_SEEDS),
        "prediction_artifact": str(path),
        "n_registered_prediction_rows": n_rows,
        "components_per_seed": 2,
        "components_per_distribution": 10,
        "equal_seed_weight": 0.2,
        "parent_point_policy": "fixed_seed0_parent_not_averaged",
        "candidate_null_point_policy": "five_seed_arithmetic_mean",
    }


def assemble(merged: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    _validate_complete_merge(merged)
    rows_by_fold: dict[int, list[dict[str, Any]]] = {
        fold: [] for fold in EXPECTED_FOLDS
    }
    for row in merged["folds"]:
        rows_by_fold[int(row["outer_fold"])].append(row)

    folds = [
        assemble_fold(rows_by_fold[fold], fold=fold, out_dir=out_dir)
        for fold in EXPECTED_FOLDS
    ]
    return {
        "schema_version": SCHEMA,
        "phase": EXPECTED_PHASE,
        "status": ASSEMBLY_STATUS,
        "folds": folds,
        "source_run_count": len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS),
        "equal_seed_mixture": True,
        "parent_point_policy": "fixed_seed0_parent_not_averaged",
        "candidate_null_point_policy": "five_seed_arithmetic_mean",
        "best_seed_selection_performed": False,
        "score_computed": False,
        "partial_scores_inspected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError("refusing to overwrite puzzle-set formal assembly")
    result = assemble(
        json.loads(args.merged_json.read_text(encoding="utf-8")), args.out_dir
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
