from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.merge_post_v13_route_diagnostics import merge_folds
from scripts.reactflow_delta.post_v13_route_diagnostics import (
    accumulate_train_stats,
    coherent_signed_magnitude,
    normalized_reliability_weights,
)
from scripts.reactflow_delta.qualify_post_v13_route_diagnostics import qualify
from scripts.reactflow_delta.run_post_v13_route_diagnostics import (
    CORRECTED_REFERENCE_SCHEMA,
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
    assert_corrected_feature41_replay,
    assert_prediction_authority,
    predict_registered_held,
)
from scripts.reactflow_delta.score_post_v13_route_diagnostics import (
    SCHEMA as SCORE_SCHEMA,
    assert_score_authority,
    puzzle_macro,
)


class _ArrayCache:
    def __init__(self, width: int, length: int = 3) -> None:
        self.width = width
        self.length = length

    def get(self, _record: SimpleNamespace) -> np.ndarray:
        return np.zeros((self.length, self.width), dtype=np.float32)


class _TrainingUniverse:
    def __init__(
        self,
        targets: dict[str, np.ndarray | None],
        errors: dict[str, np.ndarray | None],
    ) -> None:
        self.targets = targets
        self.errors = errors
        construct = SimpleNamespace(
            sequence="AUG",
            wt_observed=np.asarray([True, True, True]),
            wt_reactivity=np.asarray([0.1, 0.2, 0.3]),
            wt_error=np.asarray([0.1, 0.2, 0.3]),
            region_map=np.asarray(["design_region", "other", "other"]),
        )
        self.constructs = {"P01_method_a": construct, "P01_method_b": construct}

    def get_construct(self, construct_id: str) -> SimpleNamespace:
        return self.constructs[construct_id]

    def mutant_full_profile(
        self, wt_id: str, _design_pos: int, _ref: str, _alt: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.targets[wt_id], self.errors[wt_id]


def _record(cell: str, wt_id: str, *, puzzle: str = "P01") -> SimpleNamespace:
    return SimpleNamespace(
        construct_id=cell,
        wt_id=wt_id,
        puzzle=puzzle,
        method=cell,
        design_pos=0,
        full_pos=0,
        ref="A",
        alt="U",
    )


def test_reliability_weights_are_positive_mean_one_and_use_neutral_missing_raw() -> None:
    weights = normalized_reliability_weights(
        np.asarray([0.1, np.nan, 0.0]), np.asarray([0.2, 0.2, 0.2])
    )
    raw_valid = 1.0 / (0.1**2 + 0.2**2 + 0.05**2)
    expected = np.asarray([raw_valid, 1.0, 1.0])
    expected /= expected.mean()
    assert np.all(weights > 0)
    assert weights.mean() == pytest.approx(1.0)
    assert np.allclose(weights, expected)


def test_noise_aware_stats_preserve_cell_and_mutant_exposure() -> None:
    targets = {
        "a1": np.asarray([0.2, 0.4, 0.6]),
        "a2": np.asarray([0.3, 0.5, 0.7]),
        "b1": np.asarray([0.15, 0.25, 0.35]),
        "missing": None,
    }
    errors = {
        "a1": np.asarray([0.1, 0.2, 0.3]),
        "a2": np.asarray([0.3, 0.2, 0.1]),
        "b1": np.asarray([0.2, 0.2, 0.2]),
        "missing": None,
    }
    univ = _TrainingUniverse(targets, errors)
    records = [
        _record("P01_method_a", "a1"),
        _record("P01_method_a", "a2"),
        _record("P01_method_b", "b1"),
        _record("P01_method_b", "missing"),
    ]
    ordinary, noise, counts = accumulate_train_stats(
        univ, records, _ArrayCache(12), _ArrayCache(11)
    )
    assert ordinary.sum_weight == pytest.approx(2.0)
    assert noise.sum_weight == pytest.approx(2.0)
    assert counts == {
        "n_train_puzzles": 1,
        "n_train_cells": 2,
        "n_train_valid_mutants": 3,
        "n_train_qualified_positions": 9,
    }


def test_coherent_reconstruction_uses_signed_head_only_for_direction() -> None:
    result = coherent_signed_magnitude(
        np.asarray([-0.2, 0.0, 0.3]), np.asarray([0.4, 0.7, -0.1])
    )
    assert np.array_equal(result, np.asarray([-0.4, 0.0, 0.0]))


def _constant_model(width: int, signed: float, absolute: float) -> dict:
    return {
        "mean_x": np.zeros(width, dtype=np.float64),
        "scale_x": np.ones(width, dtype=np.float64),
        "mean_y": np.asarray([signed, absolute], dtype=np.float64),
        "coefficient": np.zeros((width, 2), dtype=np.float64),
        "alpha": 1.0,
    }


class _PredictionUniverse:
    def __init__(self) -> None:
        self.construct = SimpleNamespace(
            sequence="AUG",
            wt_observed=np.asarray([True, True, True]),
            wt_reactivity=np.asarray([0.1, 0.2, 0.3]),
            wt_error=np.asarray([0.1, 0.1, 0.1]),
            region_map=np.asarray(["design_region", "other", "other"]),
        )

    def get_construct(self, _construct_id: str) -> SimpleNamespace:
        return self.construct

    def mutant_full_profile(self, *_args: object) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("held target/error entered post-V13 prediction")


def test_prediction_path_is_held_target_and_error_invariant() -> None:
    prediction = predict_registered_held(
        _PredictionUniverse(),
        [_record("P01_method_a", "held")],
        _ArrayCache(12),
        _ArrayCache(11),
        _constant_model(41, signed=-0.2, absolute=0.4),
        _constant_model(41, signed=-0.1, absolute=0.3),
        outer_fold=0,
    )
    assert len(prediction["keys"]) == 3
    assert np.array_equal(prediction["baseline_signed_delta"], np.full(3, -0.2))
    assert np.array_equal(prediction["noise_aware_signed_delta"], np.full(3, -0.1))
    assert np.array_equal(prediction["coherent_signed_delta"], np.full(3, -0.4))


def test_corrected_feature41_replay_requires_keys_and_both_heads(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    keys = np.asarray(["a", "b"], dtype=object)
    np.savez_compressed(
        reference,
        schema_version=np.asarray(CORRECTED_REFERENCE_SCHEMA),
        keys=keys,
        v6_feature41_signed_delta=np.asarray([0.1, -0.2]),
        v6_feature41_absolute_delta=np.asarray([0.2, 0.3]),
    )
    prediction = {
        "keys": keys.copy(),
        "baseline_signed_delta": np.asarray([0.1, -0.2]),
        "baseline_absolute_delta": np.asarray([0.2, 0.3]),
    }
    assert_corrected_feature41_replay(prediction, reference)
    prediction["baseline_absolute_delta"][1] += 1e-8
    with pytest.raises(ValueError, match="feature41 replay failed"):
        assert_corrected_feature41_replay(prediction, reference)


def _write_fold(root: Path, fold: int, *, target_field: bool = False) -> None:
    prediction_path = root / f"post_v13_diag_predictions_fold{fold}.npz"
    model_path = root / f"post_v13_diag_models_fold{fold}.json"
    reference_path = root / f"tic2a_corrected_predictions_fold{fold}.npz"
    key = np.asarray([f"openknot_m2|P{fold + 1:02d}|m|c|0|A>U|0"], dtype=object)
    values = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": key,
        "biological_scoring_key": key.copy(),
        "outer_fold": np.asarray([fold], dtype=np.int64),
        "registered_status": np.asarray(["covered"], dtype=object),
        "baseline_signed_delta": np.asarray([0.0]),
        "baseline_absolute_delta": np.asarray([0.0]),
        "noise_aware_signed_delta": np.asarray([0.0]),
        "noise_aware_absolute_delta": np.asarray([0.0]),
        "coherent_signed_delta": np.asarray([0.0]),
    }
    if target_field:
        values["target"] = np.asarray([0.0])
    np.savez_compressed(prediction_path, **values)
    model_path.write_text("{}\n")
    reference_path.write_text("reference\n")
    row = {
        "schema_version": FOLD_SCHEMA,
        "phase": "PV13D2",
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "prediction_artifact": str(prediction_path),
        "model_artifact": str(model_path),
        "corrected_feature41_reference": str(reference_path),
        "n_registered_prediction_rows": 1,
        "corrected_feature41_replay_pass": True,
        "held_target_or_error_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "model_or_threshold_selection_performed": False,
        "external_outcome_accessed": False,
    }
    (root / f"post_v13_diag_fold_result_fold{fold}.json").write_text(
        json.dumps(row) + "\n"
    )


def test_merge_requires_complete_prediction_only_universe(tmp_path: Path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    merged = merge_folds(tmp_path)
    assert merged["status"] == "PV13D2_COMPLETE_UNSCORED_MERGE_PASS"

    (tmp_path / "post_v13_diag_fold_result_fold19.json").unlink()
    with pytest.raises(ValueError, match="universe incomplete"):
        merge_folds(tmp_path)


def test_merge_rejects_target_side_fields(tmp_path: Path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold, target_field=fold == 0)
    with pytest.raises(ValueError, match="target-side fields"):
        merge_folds(tmp_path)


def test_puzzle_macro_balances_methods_after_mutants() -> None:
    losses = {
        "openknot_m2|P01|method_a|cell_a|0|A>U|0": 0.0,
        "openknot_m2|P01|method_a|cell_a|1|A>U|0": 2.0,
        "openknot_m2|P01|method_b|cell_b|0|A>U|0": 10.0,
    }
    assert puzzle_macro(losses) == pytest.approx(5.5)


def _scores(
    *,
    noise_signed_gain: float,
    noise_absolute_gain: float,
    coherent_signed_gain: float,
    coherent_absolute_gain: float,
) -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "baseline_signed_delta_mae": 1.0,
                "noise_aware_signed_delta_mae": 1.0 - noise_signed_gain,
                "coherent_signed_delta_mae": 1.0 - coherent_signed_gain,
                "baseline_point_absolute_delta_mae": 1.0,
                "noise_aware_point_absolute_delta_mae": 1.0
                - noise_absolute_gain,
                "coherent_point_absolute_delta_mae": 1.0
                - coherent_absolute_gain,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "PV13D3_COMPLETE_SCORE_PASS",
        "target_join_after_complete_merge": True,
        "corrected_feature41_replay_all_folds": True,
        "partial_fold_scores_inspected": False,
        "model_or_threshold_selection_performed": False,
        "external_outcome_accessed": False,
        "scores": rows,
    }


def test_qualifier_applies_gates_and_deterministic_both_pass_rule() -> None:
    both = qualify(
        _scores(
            noise_signed_gain=0.006,
            noise_absolute_gain=0.006,
            coherent_signed_gain=0.006,
            coherent_absolute_gain=0.011,
        )
    )
    assert both["route_support"] == {
        "noise_aware": True,
        "coherent_factorization": True,
    }
    assert both["selected_next_route"] == "NOISE_AWARE_POINT_TRAINING"

    coherent = qualify(
        _scores(
            noise_signed_gain=0.004,
            noise_absolute_gain=0.006,
            coherent_signed_gain=0.006,
            coherent_absolute_gain=0.011,
        )
    )
    assert coherent["status"] == "PV13D3_COHERENT_FACTORIZATION_ROUTE_SELECTED"

    neither = qualify(
        _scores(
            noise_signed_gain=0.004,
            noise_absolute_gain=0.004,
            coherent_signed_gain=0.004,
            coherent_absolute_gain=0.009,
        )
    )
    assert neither["selected_next_route"] == (
        "WT_PROFILE_SELF_SUPERVISED_PRETRAINING_ONLY"
    )


def _write_authority(
    root: Path,
    *,
    phase: str,
    training: bool | str,
    held: bool | str,
) -> None:
    path = root / "configs/reactflow_delta/active_contract.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "authority": {"current_phase": phase},
                "training_allowed": training,
                "candidate_model_training_allowed": False,
                "held_score_read_allowed": held,
                "partial_fold_score_read_allowed": False,
                "new_external_outcome_access_allowed": False,
            }
        )
    )


def test_prediction_and_score_authorities_are_mutually_exclusive(tmp_path: Path) -> None:
    _write_authority(
        tmp_path,
        phase="PV13D2",
        training="FIXED_WEIGHTED_RIDGE_DIAGNOSTIC_ONLY",
        held=False,
    )
    assert_prediction_authority(tmp_path)
    with pytest.raises(RuntimeError, match="outside PV13D3"):
        assert_score_authority(tmp_path)

    _write_authority(
        tmp_path,
        phase="PV13D3",
        training=False,
        held="PV13D_COMPLETE_MERGE_SCORE_ONCE_ONLY",
    )
    assert_score_authority(tmp_path)
    with pytest.raises(RuntimeError, match="outside PV13D2"):
        assert_prediction_authority(tmp_path)


@pytest.mark.parametrize(
    "script",
    (
        "scripts/reactflow_delta/merge_post_v13_route_diagnostics.py",
        "scripts/reactflow_delta/qualify_post_v13_route_diagnostics.py",
    ),
)
def test_pipeline_entrypoints_resolve_project_package(script: str) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
