from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reactflow_delta.merge_model_rescue_v5_probe import merge_folds
from scripts.reactflow_delta.qualify_model_rescue_v5_probe import qualify
from scripts.reactflow_delta.run_model_rescue_v5_probe import PREDICTION_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v5_probe import _puzzle_macro


def _key(method: str, design_pos: int, mutation: str, position: int) -> str:
    return (
        f"openknot_m2|P1|{method}|P1_{method}|{design_pos}|{mutation}|{position}"
    )


def test_puzzle_macro_balances_positions_then_mutants_then_methods() -> None:
    losses = {
        _key("A", 3, "A>G", 0): 0.0,
        _key("A", 3, "A>G", 1): 0.0,
        _key("A", 7, "A>G", 0): 2.0,
        _key("B", 2, "G>A", 0): 1.0,
    }
    # A: mean(mutant means [0, 2]) = 1; B: 1; puzzle = 1.
    assert _puzzle_macro(losses) == pytest.approx(1.0)
    # Direct position pooling would be 0.75, so this fixture detects the exact
    # missing-position imbalance that the frozen hierarchy must neutralize.
    assert np.mean(list(losses.values())) == pytest.approx(0.75)


def _write_prediction(path: Path, fold: int, *, include_target: bool = False) -> None:
    key = f"openknot_m2|P{fold:02d}|M|P{fold:02d}_M|0|A>G|0"
    fields = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray([key], dtype=object),
        "biological_scoring_key": np.asarray([key], dtype=object),
        "outer_fold": np.asarray([fold], dtype=np.int64),
        "baseline_signed_delta": np.asarray([0.0]),
        "baseline_absolute_delta": np.asarray([0.0]),
        "candidate_signed_delta": np.asarray([0.1]),
        "candidate_absolute_delta": np.asarray([0.1]),
        "registered_status": np.asarray(["covered"], dtype=object),
    }
    if include_target:
        fields["target"] = np.asarray([0.2])
    np.savez_compressed(path, **fields)


def _write_fold(tmp_path: Path, fold: int, *, include_target: bool = False) -> None:
    prediction = tmp_path / f"v5_probe_predictions_fold{fold}.npz"
    model = tmp_path / f"v5_probe_models_fold{fold}.json"
    _write_prediction(prediction, fold, include_target=include_target)
    model.write_text("{}\n", encoding="utf-8")
    result = {
        "schema_version": "reactflow_delta.model_rescue_v5_probe_fold.v1",
        "phase": "V5M2",
        "outer_fold": fold,
        "held_puzzle": f"P{fold:02d}",
        "prediction_artifact": str(prediction),
        "model_artifact": str(model),
        "n_registered_prediction_rows": 1,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }
    (tmp_path / f"v5_probe_fold_result_fold{fold}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )


def test_merge_requires_complete_prediction_only_universe(tmp_path: Path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    merged = merge_folds(tmp_path)
    assert merged["status"] == "V5M2_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged["merge_integrity"]["prediction_only_fields"] is True


def test_merge_rejects_incomplete_or_target_contaminated_artifacts(
    tmp_path: Path,
) -> None:
    for fold in range(19):
        _write_fold(tmp_path, fold)
    with pytest.raises(ValueError, match="incomplete"):
        merge_folds(tmp_path)
    _write_fold(tmp_path, 19, include_target=True)
    with pytest.raises(ValueError, match="target-side"):
        merge_folds(tmp_path)


def _score_rows(signed_gain: float, absolute_gain: float) -> dict:
    return {
        "status": "V5M2_COMPLETE_SCORE_PASS",
        "scores": [
            {
                "outer_fold": fold,
                "baseline_signed_delta_mae": 0.2,
                "candidate_signed_delta_mae": 0.2 - signed_gain,
                "baseline_absolute_delta_mae": 0.1,
                "candidate_absolute_delta_mae": 0.1 - absolute_gain,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
            for fold in range(20)
        ],
    }


def test_qualifier_mechanically_opens_only_the_frozen_eligibility_gate() -> None:
    passed = qualify(_score_rows(signed_gain=0.003, absolute_gain=0.0))
    assert passed["overall_status"] == "V5M2_STRUCTURE_DELTA_SIGNAL_ELIGIBLE"
    assert passed["v5m3_authorized"] is True
    failed = qualify(_score_rows(signed_gain=0.001, absolute_gain=0.0))
    assert failed["overall_status"] == "MODEL_RESCUE_V5_FAIL"
    assert failed["v5m3_authorized"] is False


def test_qualifier_rejects_duplicate_fold_universe() -> None:
    scores = _score_rows(signed_gain=0.003, absolute_gain=0.0)
    scores["scores"][-1]["outer_fold"] = 18
    with pytest.raises(ValueError, match="outer folds 0 through 19"):
        qualify(scores)
