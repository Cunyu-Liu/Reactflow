from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.model_rescue_v2 import ConditionalScaleMixtureCalibrator
from scripts.reactflow_delta.model_rescue_v3 import (
    DisagreementGate,
    apply_disagreement_gate_numpy,
    build_inner_crossfit_ledger,
    fit_convex_l1_alpha,
    fit_disagreement_gate,
    hierarchy_position_weights,
)
from scripts.reactflow_delta.run_model_rescue_v3 import (
    assert_run_authority,
    validate_outer_expert_reuse,
)


def test_exact_convex_l1_and_frozen_disagreement_gate() -> None:
    b1 = np.zeros(21)
    mean = np.r_[np.ones(20) * 2.0, np.asarray([10.0])]
    target = np.r_[np.ones(20) * 1.5, np.asarray([0.0])]
    weight = np.ones(21) / 21
    assert fit_convex_l1_alpha(target, b1, mean, weight) == 0.75
    gate = fit_disagreement_gate(target, b1, mean, weight)
    assert gate.quantile == 0.95
    assert 0.0 <= gate.alpha_low <= 1.0
    assert 0.0 <= gate.alpha_high <= 1.0
    with pytest.raises(ValueError, match="frozen"):
        fit_disagreement_gate(target, b1, mean, weight, quantile=0.9)


def test_gate_application_has_no_target_argument_and_is_convex() -> None:
    b1 = np.asarray([0.0, 2.0])
    mean = np.asarray([1.0, -2.0])
    gate = DisagreementGate(threshold=2.0, alpha_low=0.8, alpha_high=0.25)
    blend, alpha, disagreement = apply_disagreement_gate_numpy(b1, mean, gate)
    np.testing.assert_allclose(alpha, [0.8, 0.25])
    np.testing.assert_allclose(disagreement, [1.0, 4.0])
    assert np.all(blend >= np.minimum(b1, mean))
    assert np.all(blend <= np.maximum(b1, mean))


def test_inner_crossfit_ledger_is_disjoint_and_complete() -> None:
    outer = [f"P{i:02d}" for i in range(1, 9)]
    groups = [["P01", "P05"], ["P02", "P06"], ["P03", "P07"], ["P04", "P08"]]
    ledger = build_inner_crossfit_ledger(outer, groups)
    assert len(ledger) == 4
    held = [p for row in ledger for p in row["held_puzzles"]]
    assert sorted(held) == outer
    assert all(not (set(row["held_puzzles"]) & set(row["train_puzzles"])) for row in ledger)
    with pytest.raises(ValueError, match="exactly once"):
        build_inner_crossfit_ledger(outer, [groups[0], groups[0], groups[2], groups[3]])


def test_hierarchy_weights_equalize_puzzle_method_and_mutant() -> None:
    puzzle = np.asarray(["P1"] * 6 + ["P2"] * 2, dtype=object)
    method = np.asarray(["A"] * 4 + ["B"] * 2 + ["A"] * 2, dtype=object)
    mutant = np.asarray(["u1"] * 3 + ["u2"] + ["u3"] * 2 + ["u4"] * 2, dtype=object)
    weight = hierarchy_position_weights(puzzle, method, mutant)
    assert np.isclose(weight[puzzle == "P1"].sum(), 0.5)
    assert np.isclose(weight[(puzzle == "P1") & (method == "A")].sum(), 0.25)
    assert np.isclose(weight[mutant == "u1"].sum(), weight[mutant == "u2"].sum())


def test_zero_mean_calibrator_cannot_backpropagate_to_blended_mean() -> None:
    calibrator = ConditionalScaleMixtureCalibrator(feature_dim=5, hidden=4)
    blended = torch.randn(2, 3, requires_grad=True)
    features = torch.randn(2, 3, 5, requires_grad=True)
    weights, locations, scales = calibrator(blended, features)
    loss = (weights + locations + scales).sum()
    loss.backward()
    assert blended.grad is None
    assert features.grad is None
    torch.testing.assert_close(locations[..., 0], blended.detach())
    torch.testing.assert_close(locations[..., 1], blended.detach())


def test_runner_enforces_phase_specific_training_authority(tmp_path) -> None:
    config = tmp_path / "configs" / "reactflow_delta"
    config.mkdir(parents=True)
    active = {
        "authority": {"current_phase": "R3M2"},
        "runnable_phases": ["R3M2"],
        "training_allowed": "ENGINEERING_SMOKE_ONLY",
        "new_external_outcome_access_allowed": False,
    }
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(active), encoding="utf-8"
    )
    assert_run_authority(tmp_path, "R3M2", smoke=True)
    with pytest.raises(RuntimeError, match="closed"):
        assert_run_authority(tmp_path, "R3M3", smoke=False)
    with pytest.raises(RuntimeError, match="smoke only"):
        assert_run_authority(tmp_path, "R3M2", smoke=False)


def test_smoke_rejects_forty_epoch_outer_expert_reuse(tmp_path) -> None:
    with pytest.raises(ValueError, match="at most three epochs"):
        validate_outer_expert_reuse(
            seed=0,
            smoke=True,
            b1_result_dir=tmp_path / "b1",
            mean_result_dir=tmp_path / "mean",
        )
    validate_outer_expert_reuse(
        seed=0,
        smoke=False,
        b1_result_dir=tmp_path / "b1",
        mean_result_dir=tmp_path / "mean",
    )
