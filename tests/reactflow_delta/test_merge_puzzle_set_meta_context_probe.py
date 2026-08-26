from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    MERGED_SCHEMA,
    merge_complete_universe,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    BLOCK_DIAGONAL_NULL,
    FULL_CROSS_CONSTRUCT,
    POSITION_ALIGNED_OPERATOR,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import FOLD_SCHEMA


def _write_fold(
    directory: Path,
    *,
    fold: int,
    seed: int = 0,
    epochs: int = 1,
    target_field: bool = False,
    short_candidate: bool = False,
) -> None:
    keys = np.asarray([f"k{fold}-0", f"k{fold}-1"], dtype=object)
    prediction = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.full(2, fold, dtype=np.int64),
        "seed": np.full(2, seed, dtype=np.int64),
        "registered_status": np.full(2, "covered", dtype=object),
        "feature41_point": np.zeros(2),
        "parent_point": np.zeros(2),
        "candidate_point": np.zeros(2),
        "null_point": np.zeros(2),
    }
    for name in ("candidate", "null"):
        prediction[f"{name}_weights"] = np.full((2, 2), 0.5)
        prediction[f"{name}_locations"] = np.zeros((2, 2))
        prediction[f"{name}_scales"] = np.tile([0.1, 0.2], (2, 1))
        prediction[f"{name}_expected_absolute_delta"] = np.zeros(2)
    if short_candidate:
        prediction["candidate_point"] = np.zeros(1)
    if target_field:
        prediction["target"] = np.zeros(2)
    prediction_path = directory / f"prediction{fold}_{seed}.npz"
    np.savez_compressed(prediction_path, **prediction)
    checkpoints = {}
    for stage in ("point", "residual"):
        checkpoints[stage] = {}
        for name in ("candidate", "null"):
            path = directory / f"{name}_{stage}{fold}_{seed}.pt"
            path.write_bytes(f"{name}-{stage}".encode())
            checkpoints[stage][name] = str(path)
    frozen_parents = {}
    for name in ("v13_point", "v14_encoder"):
        path = directory / f"{name}_parent{fold}_{seed}.pt"
        path.write_bytes(name.encode())
        frozen_parents[name] = str(path)
    row = {
        "schema_version": FOLD_SCHEMA,
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "seed": seed,
        "point_epochs": epochs,
        "calibration_epochs": epochs,
        "candidate_connectivity": FULL_CROSS_CONSTRUCT,
        "null_connectivity": BLOCK_DIAGONAL_NULL,
        "cross_construct_operator": POSITION_ALIGNED_OPERATOR,
        "candidate_parameter_count": 100,
        "null_parameter_count": 100,
        "candidate_trainable_parameter_count": 50,
        "null_trainable_parameter_count": 50,
        "frozen_parent_seed": 0,
        "initial_parent_replay_max_abs_difference": {
            "candidate": 0.0,
            "null": 0.0,
        },
        "frozen_parent_checkpoints": frozen_parents,
        "n_validated_puzzle_coordinate_frames": 20,
        "training_histories": {
            "candidate_point": [0.5] * epochs,
            "null_point": [0.6] * epochs,
            "candidate_residual": [0.4] * epochs,
            "null_residual": [0.45] * epochs,
        },
        "point_checkpoints": checkpoints["point"],
        "residual_checkpoints": checkpoints["residual"],
        "residual_parameter_counts": {"candidate": 63748, "null": 63748},
        "prediction_artifact": str(prediction_path),
        "n_registered_prediction_rows": 2,
        "invariants": {
            "outcome_blind_puzzle_set_inputs": True,
            "exact_parameter_and_initialization_match": True,
            "candidate_full_cross_construct_attention": True,
            "null_block_diagonal_attention": True,
            "puzzle_balanced_training": True,
            "position_aligned_cross_construct_attention": True,
            "puzzle_coordinate_frames_validated": True,
            "frozen_v13_point_parent": True,
            "frozen_v14_context_encoder": True,
            "zero_initialized_parent_replay_at_1e_7": True,
            "point_frozen_during_calibration": True,
            "v10_residual_family_reused": True,
            "puzzle_balanced_residual_calibration": True,
            "median_constraint_all_held_rows": True,
            "prediction_target_free": True,
            "held_score_computed": False,
            "external_outcome_accessed": False,
        },
    }
    path = directory / f"puzzle_set_fold_result_fold{fold}_seed{seed}.json"
    path.write_text(json.dumps(row), encoding="utf-8")


def test_merger_accepts_only_the_exact_complete_prediction_universe(
    tmp_path: Path,
) -> None:
    _write_fold(tmp_path, fold=0)
    _write_fold(tmp_path, fold=1)
    result = merge_complete_universe(
        tmp_path,
        expected_folds=[0, 1],
        expected_seeds=[0],
        expected_point_epochs=1,
        expected_calibration_epochs=1,
        expected_parameter_count=100,
        expected_trainable_parameter_count=50,
    )
    assert result["schema_version"] == MERGED_SCHEMA
    assert result["status"] == "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
    assert [row["outer_fold"] for row in result["folds"]] == [0, 1]
    assert result["merge_integrity"]["partial_scores_inspected"] is False


def test_merger_rejects_missing_fold(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    try:
        merge_complete_universe(
            tmp_path,
            expected_folds=[0, 1],
            expected_seeds=[0],
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "incomplete or unexpected" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted a missing fold")


def test_merger_rejects_target_bearing_prediction(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0, target_field=True)
    try:
        merge_complete_universe(
            tmp_path,
            expected_folds=[0],
            expected_seeds=[0],
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "target_free" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted target-bearing prediction")


def test_merger_rejects_wrong_epoch_or_parameter_count(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0, epochs=2)
    try:
        merge_complete_universe(
            tmp_path,
            expected_folds=[0],
            expected_seeds=[0],
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "epoch freeze" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted an epoch mismatch")


def test_merger_rejects_misaligned_prediction_rows(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0, short_candidate=True)
    try:
        merge_complete_universe(
            tmp_path,
            expected_folds=[0],
            expected_seeds=[0],
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "aligned_rows" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted misaligned prediction rows")


def test_merger_rejects_biological_key_overlap_across_folds(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    _write_fold(tmp_path, fold=1)
    first = tmp_path / "prediction0_0.npz"
    second = tmp_path / "prediction1_0.npz"
    with np.load(first, allow_pickle=True) as handle:
        payload = {name: handle[name] for name in handle.files}
    payload["outer_fold"] = np.full(2, 1, dtype=np.int64)
    np.savez_compressed(second, **payload)
    try:
        merge_complete_universe(
            tmp_path,
            expected_folds=[0, 1],
            expected_seeds=[0],
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "repeats biological keys" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted cross-fold key overlap")


def test_merger_rejects_distribution_that_moves_the_point_median(
    tmp_path: Path,
) -> None:
    _write_fold(tmp_path, fold=0)
    path = tmp_path / "prediction0_0.npz"
    with np.load(path, allow_pickle=True) as handle:
        payload = {name: handle[name] for name in handle.files}
    payload["candidate_locations"] = np.full((2, 2), 0.5)
    np.savez_compressed(path, **payload)
    try:
        merge_complete_universe(
            tmp_path,
            expected_folds=[0],
            expected_seeds=[0],
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "median_preserved" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted a shifted distribution median")
