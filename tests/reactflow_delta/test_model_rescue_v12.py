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
from scripts.reactflow_delta.run_model_rescue_v12 import build_inner_crossfit_ledger
from scripts.reactflow_delta.qualify_model_rescue_v12_smoke import qualify as qualify_smoke
from scripts.reactflow_delta.validate_model_rescue_v12_contract import (
    assert_run_authority,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v12_contract_preserves_v11_and_keeps_training_closed() -> None:
    result = validate_contract(ROOT)
    assert result["status"] == "V12_CONTRACT_VALIDATION_PASS"
    assert result["phase"] == "V12M1"
    assert result["training_allowed"] is False
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    assert active["gate_state"]["V12M2"] == "NOT_AUTHORIZED"
    assert active["gate_state"]["V12M3"] == "NOT_AUTHORIZED"
    assert active["gate_state"]["V12M4"] == "NOT_AUTHORIZED"
    with pytest.raises(RuntimeError, match="sole active authority"):
        assert_run_authority(ROOT, "V12M2")


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
