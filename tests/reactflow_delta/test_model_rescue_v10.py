from __future__ import annotations

import numpy as np
from pathlib import Path
import subprocess
import torch

from scripts.reactflow_delta.merge_model_rescue_v10 import merge_folds
from scripts.reactflow_delta.model_rescue_v2 import gaussian_mixture_crps_torch
from scripts.reactflow_delta.model_rescue_v10 import (
    INPUT_WIDTH,
    CapacitySymmetricResidual,
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    initialize_asymmetric_from_symmetric,
    mixture_cdf_at_point,
    parameter_count,
)
from scripts.reactflow_delta.qualify_model_rescue_v10 import qualify
from scripts.reactflow_delta.score_model_rescue_v10 import (
    SCHEMA as SCORE_SCHEMA,
    merged_integrity_pass,
)


def test_v10_parameter_counts_match_frozen_contract() -> None:
    assert parameter_count(CapacitySymmetricResidual()) == 63491
    assert parameter_count(MedianAsymmetricResidual()) == 63748


def test_v10_asymmetric_initialization_is_exact_symmetric_nested_null() -> None:
    torch.manual_seed(17)
    symmetric = CapacitySymmetricResidual().double()
    asymmetric = MedianAsymmetricResidual().double()
    initialize_asymmetric_from_symmetric(symmetric, asymmetric)
    inputs = torch.randn(128, INPUT_WIDTH, dtype=torch.float64)
    point = torch.randn(128, dtype=torch.float64)
    sw, sl, ss = symmetric(point, inputs)
    aw, al, ass = asymmetric(point, inputs)
    assert torch.equal(sw, aw)
    assert torch.equal(ss, ass)
    assert torch.equal(sl, al)
    target = torch.randn(128, dtype=torch.float64)
    assert torch.equal(
        gaussian_mixture_crps_torch(sl, ss, sw, target),
        gaussian_mixture_crps_torch(al, ass, aw, target),
    )


def test_v10_asymmetric_cdf_constraint_and_gradients_are_finite() -> None:
    torch.manual_seed(19)
    model = MedianAsymmetricResidual()
    inputs = torch.randn(1024, INPUT_WIDTH)
    point = torch.randn(1024)
    weights, locations, scales = model(point, inputs)
    cdf = mixture_cdf_at_point(point, weights, locations, scales)
    assert torch.allclose(cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0)
    target = torch.randn(1024)
    loss = gaussian_mixture_crps_torch(
        locations, scales, weights, target
    ).mean()
    loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert float(model.output_layer.weight.grad[3].abs().sum()) > 0.0
    assert float(model.output_layer.bias.grad[3].abs()) > 0.0


def test_v10_asymmetric_cdf_constraint_survives_boundary_allocations() -> None:
    torch.manual_seed(23)
    model = MedianAsymmetricResidual()
    inputs = torch.randn(2048, INPUT_WIDTH)
    point = torch.randn(2048)
    with torch.no_grad():
        model.output_layer.weight[3].normal_(mean=0.0, std=4.0)
        model.output_layer.bias[3] = 8.0
    weights, locations, scales = model(point, inputs)
    cdf = mixture_cdf_at_point(point, weights, locations, scales)
    assert weights.dtype == torch.float64
    assert locations.dtype == torch.float64
    assert scales.dtype == torch.float64
    assert torch.allclose(cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0)


def test_v10_train_only_standardization_and_input_width() -> None:
    feature41 = np.arange(5 * 41, dtype=np.float64).reshape(5, 41)
    point = np.linspace(-1.0, 1.0, 5)
    direct = np.arange(5 * 201, dtype=np.float64).reshape(5, 201)
    values = calibration_input(feature41, point, direct)
    assert values.shape == (5, INPUT_WIDTH)
    standardizer = TrainOnlyStandardizer.fit([values[:3]])
    train = standardizer.transform_numpy(values[:3])
    assert np.allclose(train.mean(axis=0)[np.std(values[:3], axis=0) > 0], 0.0)
    held = standardizer.transform_numpy(values[3:])
    assert held.shape == (2, INPUT_WIDTH)


def _write_fold(directory, fold: int) -> None:
    keys = np.asarray([f"key-{fold}-0", f"key-{fold}-1"], dtype=object)
    point_f = np.asarray([0.0, 0.1])
    point_m = np.asarray([0.02, 0.12])
    prediction = {
        "schema_version": np.asarray("reactflow_delta.model_rescue_v10_prediction.v1"),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.full(2, fold),
        "seed": np.zeros(2, dtype=np.int64),
        "registered_status": np.full(2, "covered", dtype=object),
        "feature41_point": point_f,
        "meanaligned_point": point_m,
        "historical_v9_weights": np.full((2, 2), 0.5),
        "historical_v9_locations": np.repeat(point_m[:, None], 2, axis=1),
        "historical_v9_scales": np.full((2, 2), 0.2),
        "historical_v9_expected_absolute_delta": np.full(2, 0.2),
    }
    for name, point in (
        ("feature41_symmetric", point_f),
        ("feature41_asymmetric", point_f),
        ("meanaligned_symmetric", point_m),
        ("meanaligned_asymmetric", point_m),
    ):
        prediction[f"{name}_weights"] = np.full((2, 2), 0.5)
        prediction[f"{name}_locations"] = np.repeat(point[:, None], 2, axis=1)
        prediction[f"{name}_scales"] = np.full((2, 2), 0.2)
        prediction[f"{name}_expected_absolute_delta"] = np.full(2, 0.2)
    prediction_path = directory / f"prediction-{fold}.npz"
    np.savez_compressed(prediction_path, **prediction)
    checkpoints = {}
    for name in (
        "feature41_symmetric",
        "feature41_asymmetric",
        "meanaligned_symmetric",
        "meanaligned_asymmetric",
    ):
        path = directory / f"{name}-{fold}.pt"
        path.write_bytes(b"checkpoint")
        checkpoints[name] = str(path)
    row = {
        "schema_version": "reactflow_delta.model_rescue_v10_fold.v1",
        "phase": "V10M2",
        "outer_fold": fold,
        "seed": 0,
        "epochs": 40,
        "prediction_artifact": str(prediction_path),
        "n_registered_prediction_rows": 2,
        "checkpoints": checkpoints,
        "parameter_counts": {
            "feature41_symmetric": 63491,
            "feature41_asymmetric": 63748,
            "meanaligned_symmetric": 63491,
            "meanaligned_asymmetric": 63748,
        },
        "invariants": {
            "target_profile_identity_exact": True,
            "v8_point_replay_at_1e_7": True,
            "tic2a_feature41_replay_at_1e_7": True,
            "outer_train_only_standardization": True,
            "trained_v8_direct_features_only": True,
            "fair_feature41_and_meanaligned_head_families": True,
            "median_constraint_all_held_rows": True,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
    }
    (directory / f"v10_fold_result_fold{fold}_seed0.json").write_text(
        __import__("json").dumps(row) + "\n"
    )


def test_v10_merge_requires_complete_prediction_only_universe(tmp_path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    merged = merge_folds(tmp_path)
    assert merged["status"] == "V10M2_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged_integrity_pass(merged["merge_integrity"])
    (tmp_path / "v10_fold_result_fold19_seed0.json").unlink()
    try:
        merge_folds(tmp_path)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("V10 merge accepted an incomplete fold universe")


def _complete_score_fixture(asymmetric_crps: float = 0.13) -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
                "feature41_signed_delta_mae": 0.20,
                "meanaligned_signed_delta_mae": 0.18,
                "feature41_absolute_delta_mae": 0.15,
                "meanaligned_asymmetric_absolute_delta_mae": 0.135,
                "feature41_asymmetric_crps": 0.15,
                "meanaligned_asymmetric_crps": asymmetric_crps,
                "meanaligned_symmetric_crps": 0.135,
                "historical_v9_crps": 0.14,
                "feature41_asymmetric_coverage68": 0.68,
                "meanaligned_asymmetric_coverage68": 0.67,
                "feature41_asymmetric_coverage95": 0.95,
                "meanaligned_asymmetric_coverage95": 0.94,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "V10M3_COMPLETE_SCORE_PASS",
        "scores": rows,
    }


def test_v10_qualifier_requires_overall_and_asymmetry_increment() -> None:
    passed = qualify(_complete_score_fixture())
    assert passed["status"] == "V10M3_TOP_JOURNAL_SCREEN_PASS"
    assert passed["gate_passed"] is True
    failed = qualify(_complete_score_fixture(asymmetric_crps=0.136))
    assert failed["status"] == "V10M3_TOP_JOURNAL_SCREEN_FAIL"
    assert failed["gates"]["asymmetric_vs_symmetric_relative_gain_ge_1pct"] is False
    assert failed["v10m4_authorized"] is False


def test_v10_smoke_controller_is_prediction_only() -> None:
    root = Path(__file__).resolve().parents[2]
    controller = root / "scripts/reactflow_delta/run_model_rescue_v10_smoke_controller.sh"
    subprocess.run(["bash", "-n", str(controller)], check=True)
    text = controller.read_text(encoding="utf-8")
    assert "--phase V10M1" in text
    assert "--epochs 3" in text
    assert "qualify_model_rescue_v10_smoke" in text
    assert "score_model_rescue_v10" not in text


def test_v10_screen_controller_is_complete_universe_before_score() -> None:
    root = Path(__file__).resolve().parents[2]
    controller = root / "scripts/reactflow_delta/run_model_rescue_v10_screen_controller.sh"
    subprocess.run(["bash", "-n", str(controller)], check=True)
    text = controller.read_text(encoding="utf-8")
    assert "--phase V10M2" in text
    assert "--epochs 40" in text
    assert "fold<20" in text
    assert "merge_model_rescue_v10" in text
    assert "score_model_rescue_v10" not in text
