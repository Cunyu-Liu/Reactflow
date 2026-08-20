from __future__ import annotations

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v2 import (
    ConditionalScaleMixtureCalibrator,
    GlobalResidualCalibrator,
    MeanAlignedModel,
    assert_mean_state_unchanged,
    cell_balanced_l1,
    freeze_mean_model,
    gaussian_mixture_crps_torch,
    mean_state_snapshot,
)


def test_torch_mixture_crps_matches_existing_numpy_closed_form():
    locations = torch.tensor([[0.1, 0.4], [-0.3, 0.2]], dtype=torch.float64)
    scales = torch.tensor([[0.2, 0.6], [0.4, 0.1]], dtype=torch.float64)
    weights = torch.tensor([[0.7, 0.3], [0.25, 0.75]], dtype=torch.float64)
    target = torch.tensor([0.25, -0.1], dtype=torch.float64)
    actual = gaussian_mixture_crps_torch(locations, scales, weights, target)
    expected = weighted_gaussian_mixture_crps(
        locations.numpy(), scales.numpy(), weights.numpy(), target.numpy()
    )
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-10, rtol=1e-10)


def test_single_component_mixture_reduces_to_gaussian_crps():
    locations = torch.tensor([[0.2], [-0.4]], dtype=torch.float64)
    scales = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    weights = torch.ones_like(locations)
    target = torch.tensor([0.5, 0.1], dtype=torch.float64)
    actual = gaussian_mixture_crps_torch(locations, scales, weights, target)
    expected = weighted_gaussian_mixture_crps(
        locations.numpy(), scales.numpy(), weights.numpy(), target.numpy()
    )
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-10, rtol=1e-10)


def test_zero_mean_residual_components_share_exact_point_mean():
    delta_mean = torch.tensor([[0.2, -0.1], [0.0, 0.3]])
    features = torch.randn(2, 2, 9)
    calibrator = ConditionalScaleMixtureCalibrator(feature_dim=9, hidden=64)
    weights, locations, scales = calibrator(delta_mean, features)
    assert torch.equal(locations[..., 0], delta_mean)
    assert torch.equal(locations[..., 1], delta_mean)
    assert torch.allclose((weights * locations).sum(-1), delta_mean, atol=1e-7, rtol=0)
    assert torch.all(scales > 0)
    assert torch.all(scales[..., 1] >= scales[..., 0])


def test_calibration_backward_cannot_change_or_reach_mean_model():
    torch.manual_seed(7)
    mean_model = MeanAlignedModel(d=16, heads=4, hidden=8)
    freeze_mean_model(mean_model)
    before = mean_state_snapshot(mean_model)
    calibrator = ConditionalScaleMixtureCalibrator(feature_dim=41, hidden=64)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=1e-3)
    delta_mean = torch.randn(2, 5)
    frozen_features = torch.randn(2, 5, 41)
    weights, locations, scales = calibrator(delta_mean, frozen_features)
    loss = gaussian_mixture_crps_torch(
        locations, scales, weights, torch.randn(2, 5)
    ).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in mean_model.parameters())
    assert_mean_state_unchanged(before, mean_model)


def test_global_calibrator_has_one_same_center_component():
    delta_mean = torch.tensor([[0.1, -0.2]])
    weights, locations, scales = GlobalResidualCalibrator()(delta_mean)
    assert weights.shape == locations.shape == scales.shape == (1, 2, 1)
    assert torch.equal(locations[..., 0], delta_mean)
    assert torch.all(scales > 0)


def test_cell_l1_averages_positions_then_only_observed_mutants():
    prediction = torch.tensor([[0.0, 2.0], [100.0, 100.0], [1.0, 4.0]])
    target = torch.tensor([[0.0, 4.0], [5.0, 5.0], [2.0, 8.0]])
    qualified = torch.tensor([[True, True], [False, False], [True, False]])
    wt = torch.zeros(2)
    # Mutant 1 loss=(0+2)/2=1; fully missing mutant is excluded; mutant 3 loss=1.
    assert torch.equal(cell_balanced_l1(prediction, target, qualified, wt), torch.tensor(1.0))
