from __future__ import annotations

import copy

import torch

from scripts.reactflow_delta.model_rescue_v4 import (
    MutationConditionedDualTower,
    V4ModelConfig,
    ZeroMeanResidualCalibrator,
    capacity_ratio,
    freeze_mean_model,
    residual_crps,
    trainable_parameter_count,
)


def _tiny_config(*, pair_enabled: bool = True, foundation_dim: int = 12) -> V4ModelConfig:
    return V4ModelConfig(
        d_model=32,
        heads=4,
        wt_blocks=2,
        response_blocks=2 if pair_enabled else 3,
        ff_dim=64,
        pair_dim=16,
        pair_heads=4,
        pair_blocks=2 if pair_enabled else 0,
        foundation_dim=foundation_dim,
        dropout=0.0,
        max_relative_distance=8,
        pair_enabled=pair_enabled,
    )


def _inputs(*, foundation_dim: int = 12):
    torch.manual_seed(7)
    length = 7
    batch = 3
    base = torch.tensor([0, 1, 2, 3, 0, 2, 1])
    sequence = torch.nn.functional.one_hot(base, 4).float()
    react = torch.randn(length)
    error = torch.rand(length)
    observed = torch.tensor([True, True, False, True, True, False, True])
    position = torch.linspace(-1.0, 1.0, length)
    region = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1, 1, 0, 0]), 2).float()
    edit = torch.tensor([1, 3, 5])
    wt_foundation = torch.randn(length, foundation_dim)
    mutant_foundation = wt_foundation[None].repeat(batch, 1, 1)
    mutant_foundation[0, 1] += 0.5
    mutant_foundation[1, 3] -= 0.25
    mutant_foundation[2, 5] += 0.75
    return {
        "sequence_one_hot": sequence,
        "wt_reactivity": react,
        "wt_error": error,
        "wt_observed": observed,
        "position": position,
        "region_one_hot": region,
        "edit_idx": edit,
        "refs": ["C", "U", "G"],
        "alts": ["A", "C", "A"],
        "wt_foundation": wt_foundation,
        "mutant_foundation": mutant_foundation,
    }


def test_dual_tower_outputs_full_construct_and_backpropagates_through_pair_source() -> None:
    model = MutationConditionedDualTower(_tiny_config())
    mean, features = model.forward_mean_and_features(**_inputs())
    assert mean.shape == (3, 7)
    assert features.shape == (3, 7, 32)
    assert torch.isfinite(mean).all()
    mean.abs().mean().backward()
    pair_grad = model.sequence_to_pair[0].weight.grad
    assert pair_grad is not None
    assert torch.isfinite(pair_grad).all()
    assert float(pair_grad.abs().sum()) > 0.0


def test_full_output_is_not_masked_by_wt_observation_and_same_allele_is_zero() -> None:
    model = MutationConditionedDualTower(_tiny_config())
    inputs = _inputs()
    inputs["refs"][0] = "A"
    inputs["alts"][0] = "A"
    mean = model.forward_mean(**inputs)
    assert torch.equal(mean[0], torch.zeros_like(mean[0]))
    assert torch.isfinite(mean[:, ~inputs["wt_observed"]]).all()
    assert not torch.equal(mean[1, ~inputs["wt_observed"]], torch.zeros(2))


def test_prediction_path_has_no_target_argument_and_foundation_is_detached() -> None:
    model = MutationConditionedDualTower(_tiny_config())
    inputs = _inputs()
    inputs["wt_foundation"].requires_grad_(True)
    inputs["mutant_foundation"].requires_grad_(True)
    mean = model.forward_mean(**inputs)
    mean.square().mean().backward()
    assert inputs["wt_foundation"].grad is None
    assert inputs["mutant_foundation"].grad is None


def test_zero_mean_calibration_cannot_move_or_backpropagate_to_mean() -> None:
    model = MutationConditionedDualTower(_tiny_config())
    inputs = _inputs()
    mean, features = model.forward_mean_and_features(**inputs)
    state_before = copy.deepcopy(model.state_dict())
    freeze_mean_model(model)
    calibrator = ZeroMeanResidualCalibrator(feature_dim=32, hidden=16)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=1e-3)
    target = torch.randn_like(mean)
    mask = torch.ones_like(mean, dtype=torch.bool)
    loss = residual_crps(calibrator, mean.detach(), features.detach(), target, mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    weights, locations, scales = calibrator(mean, features)
    assert torch.equal(locations[..., 0], mean.detach())
    assert torch.equal(locations[..., 1], mean.detach())
    assert torch.allclose(weights.sum(-1), torch.ones_like(mean))
    assert torch.all(scales > 0)
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, expected in state_before.items():
        assert torch.equal(model.state_dict()[name], expected)


def test_default_primary_and_functional_null_meet_frozen_capacity_contract() -> None:
    primary = MutationConditionedDualTower(V4ModelConfig.primary())
    null = MutationConditionedDualTower(V4ModelConfig.capacity_null())
    primary_count = trainable_parameter_count(primary)
    assert 35_000_000 <= primary_count <= 45_000_000
    assert capacity_ratio(primary, null) <= 0.05
    assert null.config.pair_enabled is False
    assert len(null.response_blocks) == 6
