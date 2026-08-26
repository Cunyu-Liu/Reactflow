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
        "candidate_point": np.zeros(2),
        "null_point": np.zeros(2),
    }
    if short_candidate:
        prediction["candidate_point"] = np.zeros(1)
    if target_field:
        prediction["target"] = np.zeros(2)
    prediction_path = directory / f"prediction{fold}_{seed}.npz"
    np.savez_compressed(prediction_path, **prediction)
    candidate_checkpoint = directory / f"candidate{fold}_{seed}.pt"
    null_checkpoint = directory / f"null{fold}_{seed}.pt"
    candidate_checkpoint.write_bytes(b"candidate")
    null_checkpoint.write_bytes(b"null")
    row = {
        "schema_version": FOLD_SCHEMA,
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "seed": seed,
        "epochs": epochs,
        "candidate_connectivity": FULL_CROSS_CONSTRUCT,
        "null_connectivity": BLOCK_DIAGONAL_NULL,
        "candidate_parameter_count": 100,
        "null_parameter_count": 100,
        "candidate_history": [0.5] * epochs,
        "null_history": [0.6] * epochs,
        "candidate_checkpoint": str(candidate_checkpoint),
        "null_checkpoint": str(null_checkpoint),
        "prediction_artifact": str(prediction_path),
        "n_registered_prediction_rows": 2,
        "invariants": {
            "outcome_blind_puzzle_set_inputs": True,
            "exact_parameter_and_initialization_match": True,
            "candidate_full_cross_construct_attention": True,
            "null_block_diagonal_attention": True,
            "puzzle_balanced_training": True,
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
        expected_epochs=1,
        expected_parameter_count=100,
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
            expected_epochs=1,
            expected_parameter_count=100,
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
            expected_epochs=1,
            expected_parameter_count=100,
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
            expected_epochs=1,
            expected_parameter_count=100,
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
            expected_epochs=1,
            expected_parameter_count=100,
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
            expected_epochs=1,
            expected_parameter_count=100,
        )
    except ValueError as error:
        assert "repeats biological keys" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted cross-fold key overlap")
