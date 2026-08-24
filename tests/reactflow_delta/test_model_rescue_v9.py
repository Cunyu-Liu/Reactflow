from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from scripts.reactflow_delta.diagnose_model_rescue_v9_residuals import (
    hierarchical_weights,
    residual_statistics,
    weighted_quantile,
)
from scripts.reactflow_delta.model_rescue_v2 import gaussian_mixture_crps_torch
from scripts.reactflow_delta.model_rescue_v9 import (
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
    EquiCalibratedZeroMeanMixture,
    assert_zero_mean_distribution,
    expected_absolute_delta,
    mutant_balanced_crps,
)
from scripts.reactflow_delta.merge_model_rescue_v9 import merge_folds
from scripts.reactflow_delta.qualify_model_rescue_v9 import qualify as qualify_screen
from scripts.reactflow_delta.qualify_model_rescue_v9_smoke import qualify
from scripts.reactflow_delta.score_model_rescue_v9 import (
    SCHEMA as SCORE_SCHEMA,
    merged_integrity_pass,
)
from scripts.reactflow_delta.model_rescue_v6_probe import CANDIDATE_PROBE_FEATURE_NAMES


def test_v9_distribution_preserves_signed_mean_and_has_positive_magnitude() -> None:
    torch.manual_seed(0)
    model = EquiCalibratedZeroMeanMixture()
    mean = torch.tensor([0.0, 0.1, -0.2])
    features = torch.randn(3, 41)
    weights, locations, scales = model(mean, features)
    assert_zero_mean_distribution(mean, weights, locations)
    assert torch.all(scales > 0)
    assert torch.allclose(weights.sum(-1), torch.ones(3))
    magnitude = expected_absolute_delta(weights, locations, scales)
    assert torch.all(magnitude >= mean.abs())


def test_v9_zero_mean_check_is_stable_for_large_float32_means() -> None:
    model = EquiCalibratedZeroMeanMixture()
    mean = torch.tensor([1.0e8, -1.0e8], dtype=torch.float32)
    weights, locations, _scales = model(mean, torch.zeros(2, 41))
    assert_zero_mean_distribution(mean, weights, locations)
    assert torch.equal(locations[:, 0], mean)
    assert torch.equal(locations[:, 1], mean)


def test_v9_mutant_balanced_crps_matches_explicit_mutant_average() -> None:
    target = torch.tensor([0.1, -0.2, 0.4])
    locations = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.2, 0.2]])
    scales = torch.full((3, 2), 0.1)
    weights = torch.full((3, 2), 0.5)
    index = torch.tensor([0, 0, 1])
    actual = mutant_balanced_crps(
        weights, locations, scales, target, index, n_mutants=2
    )
    position = gaussian_mixture_crps_torch(locations, scales, weights, target)
    expected = 0.5 * (position[:2].mean() + position[2])
    assert torch.allclose(actual, expected)


def test_v9_baseline_and_candidate_heads_can_start_identically() -> None:
    torch.manual_seed(17)
    baseline = EquiCalibratedZeroMeanMixture()
    torch.manual_seed(17)
    candidate = EquiCalibratedZeroMeanMixture()
    for left, right in zip(baseline.parameters(), candidate.parameters()):
        assert torch.equal(left, right)


def _write_smoke_fold(directory: Path, fold: int) -> None:
    keys = np.asarray([f"key-{fold}-0", f"key-{fold}-1"], dtype=object)
    prediction = directory / f"prediction-{fold}.npz"
    mean = np.asarray([0.0, 0.1])
    locations = np.repeat(mean[:, None], 2, axis=1)
    weights = np.full((2, 2), 0.5)
    scales = np.full((2, 2), 0.2)
    expected_abs = np.asarray([0.15, 0.2])
    np.savez_compressed(
        prediction,
        schema_version=np.asarray(PREDICTION_SCHEMA),
        keys=keys,
        biological_scoring_key=keys.copy(),
        outer_fold=np.full(2, fold),
        seed=np.zeros(2),
        registered_status=np.full(2, "covered", dtype=object),
        feature41_delta_mean=mean,
        feature41_weights=weights,
        feature41_locations=locations,
        feature41_scales=scales,
        feature41_expected_absolute_delta=expected_abs,
        meanaligned_delta_mean=mean,
        meanaligned_weights=weights,
        meanaligned_locations=locations,
        meanaligned_scales=scales,
        meanaligned_expected_absolute_delta=expected_abs,
    )
    baseline = directory / f"baseline-{fold}.pt"
    candidate = directory / f"candidate-{fold}.pt"
    state = EquiCalibratedZeroMeanMixture().state_dict()
    torch.save(state, baseline)
    torch.save(state, candidate)
    row = {
        "schema_version": FOLD_SCHEMA,
        "phase": "V9M1",
        "evidence_status": "ENGINEERING_SMOKE_ONLY",
        "outer_fold": fold,
        "seed": 0,
        "calibration_epochs": 3,
        "baseline_calibration_checkpoint": str(baseline),
        "candidate_calibration_checkpoint": str(candidate),
        "prediction_artifact": str(prediction),
        "n_registered_prediction_rows": 2,
        "invariants": {
            "target_profile_identity_exact": True,
            "v8_mean_replay_at_1e_7": True,
            "tic2a_feature41_replay_at_1e_7": True,
            "identical_residual_head_class_and_budget": True,
            "both_component_locations_equal_frozen_mean": True,
            "residual_changed_signed_point_mean": False,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
    }
    (directory / f"v9_fold_result_fold{fold}_seed0.json").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


def test_v9_smoke_qualifier_checks_distribution_invariants(tmp_path: Path) -> None:
    _write_smoke_fold(tmp_path, 0)
    _write_smoke_fold(tmp_path, 1)
    result = qualify(tmp_path)
    assert result["status"] == "V9M1_ENGINEERING_SMOKE_PASS"
    assert result["scores_read"] is False
    assert result["v9m2_authorized"] is True


def test_v9_smoke_controller_is_prediction_only() -> None:
    root = Path(__file__).resolve().parents[2]
    controller = root / "scripts/reactflow_delta/run_model_rescue_v9_smoke_controller.sh"
    subprocess.run(["bash", "-n", str(controller)], check=True)
    text = controller.read_text(encoding="utf-8")
    assert "--phase V9M1" in text
    assert "--folds \"${fold}\"" in text
    assert "qualify_model_rescue_v9_smoke" in text
    assert "score_model_rescue" not in text


def test_v9_screen_controller_uses_all_folds_and_never_scores() -> None:
    root = Path(__file__).resolve().parents[2]
    controller = root / "scripts/reactflow_delta/run_model_rescue_v9_screen_controller.sh"
    subprocess.run(["bash", "-n", str(controller)], check=True)
    text = controller.read_text(encoding="utf-8")
    assert "--phase V9M2" in text
    assert "--epochs 40" in text
    assert "merge_model_rescue_v9" in text
    assert "score_model_rescue" not in text
    for fold in range(20):
        assert str(fold) in text


def test_v9m2_merge_requires_all_twenty_prediction_only_folds(tmp_path: Path) -> None:
    for fold in range(20):
        _write_smoke_fold(tmp_path, fold)
        path = tmp_path / f"v9_fold_result_fold{fold}_seed0.json"
        row = json.loads(path.read_text())
        row["phase"] = "V9M2"
        row["evidence_status"] = "DEVELOPMENT_CONSUMED_PREDICTION_ONLY_SCREEN"
        row["calibration_epochs"] = 40
        path.write_text(json.dumps(row) + "\n")
    merged = merge_folds(tmp_path)
    assert merged["status"] == "V9M2_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged["merge_integrity"]["zero_mean_residual_all_folds"] is True
    (tmp_path / "v9_fold_result_fold19_seed0.json").unlink()
    try:
        merge_folds(tmp_path)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("V9 merge accepted an incomplete fold universe")


def test_v9_complete_scorer_accepts_safe_false_merge_invariants(
    tmp_path: Path,
) -> None:
    for fold in range(20):
        _write_smoke_fold(tmp_path, fold)
        path = tmp_path / f"v9_fold_result_fold{fold}_seed0.json"
        row = json.loads(path.read_text())
        row["phase"] = "V9M2"
        row["evidence_status"] = "DEVELOPMENT_CONSUMED_PREDICTION_ONLY_SCREEN"
        row["calibration_epochs"] = 40
        path.write_text(json.dumps(row) + "\n")
    integrity = merge_folds(tmp_path)["merge_integrity"]
    assert integrity["partial_scores_inspected"] is False
    assert integrity["external_outcome_accessed"] is False
    assert merged_integrity_pass(integrity) is True
    integrity["partial_scores_inspected"] = True
    assert merged_integrity_pass(integrity) is False


def test_post_v9_diagnostic_weights_balance_method_mutant_and_position() -> None:
    methods = np.asarray(["a", "a", "a", "b", "b"], dtype=object)
    mutants = np.asarray(["a1", "a1", "a2", "b1", "b1"], dtype=object)
    weights = hierarchical_weights(methods, mutants)
    assert np.isclose(weights[methods == "a"].sum(), 0.5)
    assert np.isclose(weights[methods == "b"].sum(), 0.5)
    assert np.isclose(weights[mutants == "a1"].sum(), 0.25)
    assert np.isclose(weights[mutants == "a2"].sum(), 0.25)
    assert np.isclose(weights[mutants == "b1"].sum(), 0.5)


def test_post_v9_weighted_quantiles_and_asymmetry_are_deterministic() -> None:
    values = np.asarray([-2.0, 0.0, 1.0, 4.0])
    weights = np.asarray([0.1, 0.4, 0.3, 0.2])
    assert weighted_quantile(values, weights, 0.1) == -2.0
    assert weighted_quantile(values, weights, 0.5) == 0.0
    assert weighted_quantile(values, weights, 0.9) == 4.0
    result = residual_statistics(values, weights)
    assert result["q10"] == -2.0
    assert result["q50"] == 0.0
    assert result["q90"] == 4.0
    assert np.isclose(result["normalized_quantile_asymmetry"], 1.0 / 3.0)


def _complete_score_fixture(candidate_absolute: float = 0.14) -> dict:
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
                "meanaligned_absolute_delta_mae": candidate_absolute,
                "feature41_crps": 0.16,
                "meanaligned_crps": 0.14,
                "feature41_coverage68": 0.68,
                "meanaligned_coverage68": 0.67,
                "feature41_coverage95": 0.95,
                "meanaligned_coverage95": 0.94,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "V9M3_COMPLETE_SCORE_PASS",
        "scores": rows,
    }


def test_v9_top_journal_qualifier_requires_signed_absolute_and_crps() -> None:
    passed = qualify_screen(_complete_score_fixture())
    assert passed["status"] == "V9M3_TOP_JOURNAL_SCREEN_PASS"
    assert passed["gate_passed"] is True
    failed = qualify_screen(_complete_score_fixture(candidate_absolute=0.151))
    assert failed["status"] == "V9M3_TOP_JOURNAL_SCREEN_FAIL"
    assert failed["gates"]["absolute_relative_gain_ge_1pct"] is False
    assert failed["v9m4_authorized"] is False


def test_v9_feature41_contract_binds_exact_feature_order() -> None:
    assert len(CANDIDATE_PROBE_FEATURE_NAMES) == 41
    assert CANDIDATE_PROBE_FEATURE_NAMES[:3] == (
        "signed_sequence_distance",
        "absolute_sequence_distance",
        "log_absolute_sequence_distance",
    )
