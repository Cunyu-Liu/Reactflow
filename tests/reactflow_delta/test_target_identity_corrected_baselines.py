from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reactflow_delta.merge_target_identity_corrected_baselines import merge_folds
from scripts.reactflow_delta.model_rescue_v5_probe import WeightedRidgeStats
from scripts.reactflow_delta.qualify_target_identity_corrected_baselines import qualify
from scripts.reactflow_delta.run_target_identity_corrected_baselines import (
    FOLD_SCHEMA,
    PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
    _stats_equal,
)
from scripts.reactflow_delta.score_target_identity_corrected_baselines import (
    SCHEMA as SCORE_SCHEMA,
)


def test_feature30_replay_compares_all_sufficient_statistics() -> None:
    left = WeightedRidgeStats.zeros(3, 2)
    right = WeightedRidgeStats.zeros(3, 2)
    x = np.asarray([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    y = np.asarray([[0.1, 0.1], [-0.2, 0.2]])
    weight = np.asarray([0.5, 0.5])
    left.add_rows(x, y, weight)
    right.add_rows(x, y, weight)
    assert _stats_equal(left, right)
    right.xty[0, 0] += 1.0e-5
    assert not _stats_equal(left, right)


def _write_fold(directory: Path, fold: int) -> None:
    prediction = directory / f"tic2a_corrected_predictions_fold{fold}.npz"
    model = directory / f"tic2a_corrected_models_fold{fold}.json"
    keys = np.asarray([f"key-{fold}-0", f"key-{fold}-1"], dtype=object)
    values = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.full(2, fold, dtype=np.int64),
        "registered_status": np.full(2, "covered", dtype=object),
    }
    for index, field in enumerate(PREDICTION_FIELDS):
        values[field] = np.asarray([index / 10.0, index / 10.0 + 0.01])
    np.savez_compressed(prediction, **values)
    model.write_text("{}\n")
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": "TIC2A",
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "prediction_artifact": str(prediction),
        "model_artifact": str(model),
        "n_registered_prediction_rows": 2,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "v5_v6_feature30_stats_replay_pass": True,
        "v5_v6_feature30_prediction_replay_pass": True,
        "held_target_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "legacy_prediction_reused": False,
        "external_outcome_accessed": False,
    }
    (directory / f"tic2a_corrected_fold_result_fold{fold}.json").write_text(
        json.dumps(result) + "\n"
    )


def test_merge_requires_all_prediction_only_folds(tmp_path: Path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    merged = merge_folds(tmp_path)
    assert merged["status"] == "TIC2A_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged["merge_integrity"]["complete_fold_universe"] is True
    assert merged["merge_integrity"]["target_identity_exact"] is True
    assert merged["merge_integrity"]["legacy_prediction_reused"] is False


def test_merge_rejects_missing_fold(tmp_path: Path) -> None:
    for fold in range(19):
        _write_fold(tmp_path, fold)
    with pytest.raises(ValueError, match="incomplete"):
        merge_folds(tmp_path)


def test_qualifier_reports_all_fixed_pairwise_effects_without_selection() -> None:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
                "direct18_signed_delta_mae": 0.30,
                "direct18_absolute_delta_mae": 0.25,
                "v5_feature30_signed_delta_mae": 0.29,
                "v5_feature30_absolute_delta_mae": 0.24,
                "v6_feature41_signed_delta_mae": 0.28,
                "v6_feature41_absolute_delta_mae": 0.23,
            }
        )
    result = qualify(
        {
            "schema_version": SCORE_SCHEMA,
            "status": "TIC2A_COMPLETE_CORRECTED_SCORE_PASS",
            "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
            "scores": rows,
        }
    )
    assert result["status"] == "TIC2A_CORRECTED_BASELINE_REBUILD_PASS"
    assert set(result["comparisons"]) == {
        "direct18_to_v5_feature30",
        "v5_feature30_to_v6_feature41",
        "direct18_to_v6_feature41",
    }
    assert result["model_selection_performed"] is False
    assert result["v7_candidate_evaluated"] is False
