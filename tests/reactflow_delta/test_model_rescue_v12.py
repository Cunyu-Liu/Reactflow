import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.model_rescue_v12 import (
    GATE_PARAMETERS,
    MonotoneRegimeGate,
    fit_monotone_gate,
    fixed_parent_null,
    gated_point,
    hierarchy_weights,
    trainable_parameter_count,
)
from scripts.reactflow_delta.model_rescue_v10 import (
    MedianAsymmetricResidual,
    mixture_cdf_at_point,
)
from scripts.reactflow_delta.run_model_rescue_v12 import (
    _load_parent_prediction,
    build_inner_crossfit_ledger,
)
from scripts.reactflow_delta.qualify_model_rescue_v12_smoke import qualify as qualify_smoke
from scripts.reactflow_delta.merge_model_rescue_v12 import merge_folds
from scripts.reactflow_delta.qualify_model_rescue_v12 import qualify
from scripts.reactflow_delta.score_model_rescue_v12 import assert_score_authority
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.assemble_model_rescue_v12_formal import assemble_fold
from scripts.reactflow_delta.qualify_model_rescue_v12_formal import (
    qualify as qualify_formal,
)
from scripts.reactflow_delta.score_model_rescue_v12_formal import (
    assert_score_authority as assert_formal_score_authority,
)
from scripts.reactflow_delta.diagnose_model_rescue_v12 import (
    assert_diagnostic_authority,
    gate_geometry,
    oracle_diagnostics,
    route_summary,
    summarize_gate_geometries,
)
from scripts.reactflow_delta.validate_model_rescue_v12_contract import (
    assert_run_authority,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v12_contract_preserves_v11_and_opens_only_score_blind_screen() -> None:
    result = validate_contract(ROOT)
    assert result["status"] == "V12_CONTRACT_VALIDATION_PASS"
    assert result["phase"] == "V12M3"
    assert result["training_allowed"] == "V12_TWENTY_FOLD_SCORE_BLIND_SCREEN_ONLY"
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    assert active["gate_state"]["V12M2"] == "ENGINEERING_SMOKE_PASS"
    assert active["gate_state"]["V12M3"] == "AUTHORIZED_SCORE_BLIND_SCREEN_ONLY"
    assert active["gate_state"]["V12M4"] == "NOT_AUTHORIZED"
    assert_run_authority(ROOT, "V12M3")
    with pytest.raises(RuntimeError, match="sole active authority"):
        assert_run_authority(ROOT, "V12M4")


def test_gate_has_four_parameters_and_is_monotone_in_both_inputs() -> None:
    gate = MonotoneRegimeGate().to(dtype=torch.float64)
    assert trainable_parameter_count(gate) == GATE_PARAMETERS
    distance = torch.tensor([0.0, 1.0, 5.0, 20.0], dtype=torch.float64)
    feature = torch.full_like(distance, 0.1)
    by_distance = gate(distance, feature)
    assert torch.all(by_distance[1:] >= by_distance[:-1])
    magnitude = torch.tensor([0.0, 0.01, 0.05, 0.2], dtype=torch.float64)
    by_magnitude = gate(torch.full_like(magnitude, 10.0), magnitude)
    assert torch.all(by_magnitude[1:] >= by_magnitude[:-1])
    assert torch.all((by_distance > 0.0) & (by_distance < 1.0))
    assert torch.all((by_magnitude > 0.0) & (by_magnitude < 1.0))


def test_candidate_composition_and_fixed_one_null_are_exact() -> None:
    feature = torch.tensor([-0.2, 0.0, 0.3], dtype=torch.float64)
    parent = torch.tensor([-0.1, 0.2, -0.4], dtype=torch.float64)
    gate = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)
    candidate = gated_point(feature, parent, gate)
    assert torch.equal(candidate[[0]], feature[[0]])
    assert torch.allclose(candidate[[2]], parent[[2]], atol=1e-15, rtol=0.0)
    assert torch.equal(fixed_parent_null(feature, parent), parent)


def test_hierarchy_weights_balance_puzzles_methods_mutants_and_positions() -> None:
    puzzles = ["P1"] * 6 + ["P2"] * 2
    methods = ["A", "A", "A", "A", "B", "B", "C", "C"]
    mutants = ["A1", "A2", "A2", "A2", "B1", "B1", "C1", "C1"]
    weights = hierarchy_weights(puzzles, methods, mutants)
    arrays = [np.asarray(value, dtype=object) for value in (puzzles, methods, mutants)]
    assert np.isclose(weights[arrays[0] == "P1"].sum(), 0.5)
    assert np.isclose(weights[arrays[0] == "P2"].sum(), 0.5)
    assert np.isclose(weights[(arrays[0] == "P1") & (arrays[1] == "A")].sum(), 0.25)
    assert np.isclose(weights[arrays[2] == "A1"].sum(), 0.125)
    assert np.isclose(weights[arrays[2] == "A2"].sum(), 0.125)


def test_gate_fit_is_deterministic_and_uses_only_declared_arrays() -> None:
    feature = np.asarray([0.0, 0.02, 0.1, 0.2] * 2, dtype=np.float64)
    parent = feature + np.asarray([0.3, 0.2, 0.1, 0.05] * 2)
    target = feature + np.asarray([0.0, 0.02, 0.08, 0.05] * 2)
    distance = np.asarray([0.0, 1.0, 10.0, 30.0] * 2)
    labels = {
        "puzzles": ["P1"] * 4 + ["P2"] * 4,
        "methods": ["A"] * 4 + ["B"] * 4,
        "mutants": ["M1", "M1", "M2", "M2"] * 2,
    }
    first = fit_monotone_gate(
        feature41_point=feature,
        parent_v11_point=parent,
        target_delta=target,
        absolute_distance=distance,
        steps=20,
        learning_rate=0.01,
        device="cpu",
        **labels,
    )
    second = fit_monotone_gate(
        feature41_point=feature,
        parent_v11_point=parent,
        target_delta=target,
        absolute_distance=distance,
        steps=20,
        learning_rate=0.01,
        device="cpu",
        **labels,
    )
    assert first.gate.to_dict() == second.gate.to_dict()
    assert first.history == second.history
    assert np.isfinite(first.history).all()
    assert first.history[-1] < first.history[0]


def test_inner_crossfit_holds_each_outer_train_puzzle_once() -> None:
    fold = SimpleNamespace(
        train_puzzles=["P1", "P2", "P3", "P4", "P5"],
        inner_groups=[["P1", "P5"], ["P2"], ["P3"], ["P4"]],
    )
    ledger = build_inner_crossfit_ledger(fold)
    held = [puzzle for row in ledger for puzzle in row["held_puzzles"]]
    assert sorted(held) == sorted(fold.train_puzzles)
    assert len(held) == len(set(held))
    for row in ledger:
        assert not set(row["train_puzzles"]) & set(row["held_puzzles"])
        assert set(row["train_puzzles"]) | set(row["held_puzzles"]) == set(
            fold.train_puzzles
        )


def test_reused_v10_residual_head_preserves_float64_candidate_median() -> None:
    torch.manual_seed(0)
    head = MedianAsymmetricResidual()
    point = torch.tensor([-0.2, 0.0, 0.3], dtype=torch.float64)
    standardized = torch.zeros((3, 244), dtype=torch.float32)
    weights, locations, scales = head(point, standardized)
    cdf = mixture_cdf_at_point(point, weights, locations, scales)
    assert torch.allclose(cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0)
    assert weights.dtype == locations.dtype == scales.dtype == torch.float64


def test_smoke_qualifier_rejects_target_bearing_predictions(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        '{"schema_version":"reactflow_delta.model_rescue_v12_inner_crossfit_ledger.v1",'
        '"outer_train_puzzles_covered_once":true,"target_values_stored":false}'
    )
    invariants = {
        "inner_crossfit_complete": True,
        "outer_held_target_used_for_gate_fit": False,
        "method_used_as_gate_input": False,
        "parent_v11_exact_replay": True,
        "gate_range_pass": True,
        "candidate_distribution_median_fixed": True,
        "prediction_only_artifact": True,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "unexpected_keys": 0,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }
    for fold in (0, 1):
        prediction = tmp_path / f"prediction{fold}.npz"
        np.savez_compressed(
            prediction,
            schema_version=np.asarray("reactflow_delta.model_rescue_v12_prediction.v1"),
            keys=np.asarray(["k"], dtype=object),
            biological_scoring_key=np.asarray(["k"], dtype=object),
            candidate_point=np.asarray([0.0]),
            gate_value=np.asarray([0.5]),
            target=np.asarray([0.0]),
        )
        (tmp_path / f"v12_fold_result_fold{fold}_seed0.json").write_text(
            json.dumps(
                {
                    "schema_version": "reactflow_delta.model_rescue_v12_fold.v1",
                    "phase": "V12M2",
                    "inner_crossfit_ledger": str(ledger),
                    "prediction_artifact": str(prediction),
                    "invariants": invariants,
                }
            )
        )
    with pytest.raises(ValueError, match="targets or scores"):
        qualify_smoke(tmp_path)


def test_complete_merger_rejects_an_incomplete_fold_universe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="universe incomplete"):
        merge_folds(tmp_path, "V12M3")


def _passing_v12_score() -> dict[str, object]:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "feature41_signed_delta_mae": 1.0,
                "parent_v11_signed_delta_mae": 0.95,
                "candidate_signed_delta_mae": 0.8,
                "feature41_absolute_delta_mae": 1.0,
                "parent_v11_point_absolute_delta_mae": 0.95,
                "candidate_point_absolute_delta_mae": 0.8,
                "candidate_distribution_absolute_delta_mae": 0.8,
                "historical_v10_distribution_absolute_delta_mae": 0.95,
                "feature41_crps": 1.0,
                "parent_v11_crps": 0.95,
                "candidate_crps": 0.8,
                "historical_v10_crps": 0.96,
                "feature41_coverage68": 0.68,
                "candidate_coverage68": 0.68,
                "feature41_coverage95": 0.95,
                "candidate_coverage95": 0.95,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
        )
    return {
        "schema_version": "reactflow_delta.model_rescue_v12_score.v1",
        "status": "V12M3_COMPLETE_SCORE_PASS",
        "scores": rows,
    }


def test_v12_qualifier_requires_every_frozen_top_journal_gate() -> None:
    scores = _passing_v12_score()
    passed = qualify(scores)
    assert passed["status"] == "V12M3_TOP_JOURNAL_SCREEN_PASS"
    assert all(passed["gates"].values())
    for row in scores["scores"]:
        row["candidate_crps"] = 0.951
    failed = qualify(scores)
    assert failed["status"] == "V12M3_TOP_JOURNAL_SCREEN_FAIL"
    assert failed["gates"]["task_crps_gain_vs_feature41_ge_5pct"] is False
    assert failed["v12m4_authorized"] is False


def test_scientific_scorer_remains_closed_during_score_blind_training() -> None:
    with pytest.raises(RuntimeError, match="training must be closed"):
        assert_score_authority(ROOT)


def test_v12_scoring_estimand_balances_methods_before_puzzle_mean() -> None:
    losses = {
        "openknot_m2|P1|A|C1|1|A>G|0": 0.0,
        "openknot_m2|P1|A|C1|2|C>U|0": 0.0,
        "openknot_m2|P1|B|C1|3|G>A|0": 2.0,
    }
    assert _puzzle_macro(losses) == 1.0


def test_v12_parent_loader_requires_exact_seed0_biological_keys(tmp_path: Path) -> None:
    path = tmp_path / "parent.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray("reactflow_delta.model_rescue_v11_prediction.v1"),
        outer_fold=np.asarray([0]),
        seed=np.asarray([0]),
        keys=np.asarray(["expected"], dtype=object),
        biological_scoring_key=np.asarray(["different"], dtype=object),
        registered_status=np.asarray(["covered"], dtype=object),
    )
    with pytest.raises(ValueError, match="biological scoring keys"):
        _load_parent_prediction(path, 0)


def _write_formal_source(path: Path, seed: int) -> None:
    keys = np.asarray(["k0", "k1"], dtype=object)
    one_component_weights = np.asarray([[1.0], [1.0]])
    two_component_weights = np.asarray([[0.25, 0.75], [0.25, 0.75]])
    deterministic = {
        "feature41_weights": one_component_weights,
        "feature41_locations": np.asarray([[0.0], [0.0]]),
        "feature41_scales": np.asarray([[0.1], [0.1]]),
        "feature41_expected_absolute_delta": np.asarray([0.1, 0.1]),
        "parent_weights": two_component_weights,
        "parent_locations": np.asarray([[0.0, 0.1], [0.0, 0.1]]),
        "parent_scales": np.asarray([[0.1, 0.2], [0.1, 0.2]]),
        "parent_expected_absolute_delta": np.asarray([0.2, 0.2]),
        "historical_v10_weights": two_component_weights,
        "historical_v10_locations": np.asarray([[0.0, 0.1], [0.0, 0.1]]),
        "historical_v10_scales": np.asarray([[0.1, 0.2], [0.1, 0.2]]),
        "historical_v10_expected_absolute_delta": np.asarray([0.2, 0.2]),
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray("reactflow_delta.model_rescue_v12_prediction.v1"),
        keys=keys,
        biological_scoring_key=keys.copy(),
        outer_fold=np.full(2, 0),
        seed=np.full(2, seed),
        feature41_point=np.asarray([0.0, 0.0]),
        v11_parent_point=np.asarray([1.0, 1.0]),
        candidate_point=np.full(2, float(seed)),
        candidate_weights=two_component_weights,
        candidate_locations=np.full((2, 2), float(seed)),
        candidate_scales=np.asarray([[0.1, 0.2], [0.1, 0.2]]),
        candidate_expected_absolute_delta=np.full(2, float(seed)),
        **deterministic,
    )


def test_formal_assembler_uses_all_five_seeds_with_equal_mass(tmp_path: Path) -> None:
    rows = []
    for seed in range(5):
        path = tmp_path / f"seed{seed}.npz"
        _write_formal_source(path, seed)
        rows.append({"seed": seed, "prediction_artifact": str(path)})
    result = assemble_fold(rows, fold=0, out_dir=tmp_path)
    with np.load(result["prediction_artifact"], allow_pickle=True) as prediction:
        assert prediction["candidate_weights"].shape == (2, 10)
        assert np.allclose(prediction["candidate_weights"].sum(axis=1), 1.0)
        for seed in range(5):
            assert np.allclose(
                prediction["candidate_weights"][:, 2 * seed : 2 * seed + 2].sum(axis=1),
                0.2,
            )
        assert np.allclose(prediction["candidate_point"], 2.0)


def test_formal_qualifier_repeats_screen_gates_and_requires_four_positive_seeds() -> None:
    screen = {
        "schema_version": "reactflow_delta.model_rescue_v12_qualification.v1",
        "status": "V12M3_TOP_JOURNAL_SCREEN_PASS",
        "gate_passed": True,
    }
    rows = _passing_v12_score()["scores"]
    scores = {
        "schema_version": "reactflow_delta.model_rescue_v12_formal_score.v1",
        "status": "V12M4_COMPLETE_FORMAL_SCORE_PASS",
        "mixture_scores": rows,
        "individual_seed_scores": {str(seed): rows for seed in range(5)},
        "equal_seed_mixture": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }
    result = qualify_formal(scores, screen)
    assert result["status"] == "V12M4_TOP_JOURNAL_FORMAL_PASS"
    assert all(result["gates"].values())
    assert set(result["positive_seed_counts"].values()) == {5}


def test_formal_scorer_remains_closed_before_exact_screen_pass() -> None:
    with pytest.raises(RuntimeError, match="closed outside V12M4"):
        assert_formal_score_authority(ROOT)


def test_post_v12_diagnostics_remain_closed_during_score_blind_training() -> None:
    with pytest.raises(RuntimeError, match="require training closed"):
        assert_diagnostic_authority(ROOT)


def test_post_v12_gate_geometry_replays_frozen_monotone_surface() -> None:
    keys = np.asarray(
        [
            "openknot_m2|P1|A|C1|1|A>G|0",
            "openknot_m2|P1|B|C2|2|C>U|0",
        ],
        dtype=object,
    )
    result = gate_geometry(
        {
            "keys": keys,
            "gate_value": np.asarray([0.2, 0.8]),
            "gate_distance_factor": np.asarray([0.4, 0.9]),
            "gate_magnitude_factor": np.asarray([0.5, 0.89]),
        },
        {
            "b_distance": 0.0,
            "raw_w_distance": 0.0,
            "b_magnitude": 0.0,
            "raw_w_magnitude": 0.0,
        },
    )
    surface = np.asarray(result["surface"]["gate"])
    assert np.all(surface[1:, :] >= surface[:-1, :])
    assert np.all(surface[:, 1:] >= surface[:, :-1])
    assert result["weighted_fractions"]["gate_lt_0.25"] == 0.5
    summary = summarize_gate_geometries(
        [{"gate_geometry": result} for _ in range(20)]
    )
    assert summary["parameters"]["b_distance"]["standard_deviation"] == 0.0
    assert len(summary["surface_grid"]) == 36


def test_post_v12_oracle_reports_all_prefrozen_d3_contrasts() -> None:
    observations = {
        "method": np.asarray(["A", "A", "A", "A", "B", "B", "B", "B"]),
        "mutant": np.asarray(["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"]),
        "distance": np.asarray([0, 1, 6, 21, 0, 2, 8, 30], dtype=float),
        "target": np.asarray([0.0, 0.1, 0.3, 0.5, 0.0, -0.1, -0.3, -0.5]),
        "feature41_point": np.asarray(
            [0.02, 0.02, 0.04, 0.06, -0.02, -0.02, -0.04, -0.06]
        ),
        "parent_point": np.asarray(
            [0.04, 0.09, 0.25, 0.42, -0.04, -0.09, -0.25, -0.42]
        ),
        "candidate_point": np.asarray(
            [0.03, 0.08, 0.22, 0.38, -0.03, -0.08, -0.22, -0.38]
        ),
        "gate_value": np.asarray([0.1, 0.3, 0.5, 0.8, 0.1, 0.3, 0.5, 0.8]),
    }
    result = oracle_diagnostics(
        observations,
        {"feature41_absolute_delta_mae": 0.235},
    )
    assert "parent_v11" in result["oracles"]["global"]["gains"]
    assert "candidate_v12" in result["oracles"]["global"]["gains"]
    assert "global_minus_2d" in result["comparisons"]
    assert "distance" in result["regime_residual_associations"]
    assert "magnitude" in result["regime_residual_associations"]
    assert "v12_gate_to_2d_oracle_gate_weighted_correlation" in result


def _post_v12_route_folds() -> list[dict[str, object]]:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "oracle": {
                    "observed_point_losses": {
                        "feature41_signed_delta_mae": 1.0,
                        "feature41_absolute_delta_mae": 1.0,
                        "candidate_signed_delta_mae": 0.9,
                        "candidate_point_absolute_delta_mae": 0.9,
                    },
                    "oracles": {
                        "global": {
                            "signed_delta_mae": 0.8,
                            "point_absolute_delta_mae": 0.8,
                        },
                        "distance_by_magnitude": {
                            "signed_delta_mae": 0.6,
                            "point_absolute_delta_mae": 0.6,
                            "signed_relative_gain_vs_feature41": 0.4,
                            "point_absolute_relative_gain_vs_feature41": 0.4,
                        },
                    },
                },
                "distribution": {
                    "global": {
                        "candidate": {
                            "lower_miss90": 0.1,
                            "upper_miss90": 0.0,
                            "lower_miss95": 0.1,
                            "upper_miss95": 0.0,
                        }
                    }
                },
            }
        )
    return rows


def test_post_v12_route_requires_diagnostic_supported_bottleneck() -> None:
    point_gate_names = {
        "signed_gain_vs_feature41_ge_10pct",
        "signed_gain_vs_parent_v11_ge_1pct",
        "signed_ci_lower_each_gt_zero",
        "signed_positive_puzzles_vs_feature41_ge_16",
        "signed_positive_puzzles_vs_parent_v11_ge_14",
        "point_absolute_gain_vs_feature41_ge_5pct",
        "point_absolute_gain_vs_parent_v11_ge_1pct",
        "point_absolute_ci_lower_each_gt_zero",
        "point_absolute_positive_puzzles_vs_feature41_ge_16",
        "point_absolute_positive_puzzles_vs_parent_v11_ge_14",
    }
    coordinate = route_summary(
        _post_v12_route_folds(), {"gates": {name: False for name in point_gate_names}}
    )
    assert coordinate["decisions"]["route"] == (
        "LOW_CAPACITY_NON_PRODUCT_GATE_AMENDMENT_ELIGIBLE_NOT_AUTHORIZED"
    )
    distribution = route_summary(
        _post_v12_route_folds(), {"gates": {name: True for name in point_gate_names}}
    )
    assert distribution["decisions"]["route"] == (
        "RESIDUAL_DISTRIBUTION_ONLY_AMENDMENT_ELIGIBLE_NOT_AUTHORIZED"
    )
    assert distribution["decisions"]["new_model_authorized"] is False
