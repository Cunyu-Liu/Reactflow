from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reactflow_delta.merge_model_rescue_v2 import merge_screen_folds
from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    MEAN_CANDIDATE,
)


def _write_fold(tmp_path, fold: int, *, seed: int = 0) -> None:
    candidates = {}
    for candidate in (MEAN_CANDIDATE, CALIBRATED_CANDIDATE):
        prediction = tmp_path / f"{candidate}_prediction_{fold}.npz"
        mean = tmp_path / f"{candidate}_mean_{fold}.pt"
        calibration = tmp_path / f"{candidate}_calibration_{fold}.pt"
        for artifact in (prediction, mean, calibration):
            artifact.write_bytes(b"artifact")
        candidates[candidate] = {
            "prediction_artifact": str(prediction),
            "mean_checkpoint": str(mean),
            "calibration_checkpoint": str(calibration),
            "score": {"must_not_be_used_by_merge": fold},
        }
    row = {
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "seed": seed,
        "baseline": {"model_id": "b1_rfd_direct_aligned", "score": {}},
        "candidates": candidates,
    }
    path = tmp_path / f"v2_fold_result_fold{fold}_seed0.json"
    path.write_text(json.dumps(row), encoding="utf-8")


def test_merge_requires_and_orders_complete_frozen_fold_universe(tmp_path):
    for fold in reversed(range(20)):
        _write_fold(tmp_path, fold)

    result = merge_screen_folds(tmp_path)

    assert result["merge_integrity"] == {
        "n_folds": 20,
        "fold_ids": list(range(20)),
        "unique_folds": True,
        "all_referenced_artifacts_present": True,
        "partial_scores_inspected_before_merge": False,
    }
    assert [row["outer_fold"] for row in result["folds"]] == list(range(20))
    assert result["qualification"]["r2m4_authorized_before_qualification"] is False


def test_merge_rejects_incomplete_fold_universe(tmp_path):
    for fold in range(19):
        _write_fold(tmp_path, fold)

    with pytest.raises(ValueError, match="exactly folds 0 through 19"):
        merge_screen_folds(tmp_path)


def test_merge_rejects_non_frozen_seed_or_missing_artifact(tmp_path):
    for fold in range(20):
        _write_fold(tmp_path, fold, seed=1 if fold == 3 else 0)

    with pytest.raises(ValueError, match="frozen seed 0"):
        merge_screen_folds(tmp_path)

    _write_fold(tmp_path, 3)
    row_path = tmp_path / "v2_fold_result_fold4_seed0.json"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    missing = row["candidates"][MEAN_CANDIDATE]["prediction_artifact"]
    row_path.write_text(json.dumps(row), encoding="utf-8")
    Path(missing).unlink()
    with pytest.raises(FileNotFoundError, match="missing prediction_artifact"):
        merge_screen_folds(tmp_path)
