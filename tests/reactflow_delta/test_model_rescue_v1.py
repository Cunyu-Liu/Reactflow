from __future__ import annotations

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v1 import (
    AlignedDeltaModel,
    aligned_mixture_loss,
    weighted_gaussian_mixture_crps,
)
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian


def test_weighted_mixture_crps_reduces_to_single_gaussian():
    loc = np.array([[0.2], [-0.4]])
    scale = np.array([[0.3], [0.7]])
    weight = np.ones_like(loc)
    y = np.array([0.1, 0.5])
    got = weighted_gaussian_mixture_crps(loc, scale, weight, y)
    expected = np.array([crps_gaussian(loc[i, 0], scale[i, 0], y[i]) for i in range(2)])
    assert np.allclose(got, expected)


def test_sparse_distribution_has_fixed_zero_mean_and_normalized_weights():
    torch.manual_seed(0)
    model = AlignedDeltaModel(k_rank=0, sparse=True, d=16, heads=4, hidden=8)
    H = torch.randn(5, 16)
    edit = torch.tensor([1, 3])
    dists = torch.arange(5)[None, :].repeat(2, 1).float() - edit[:, None]
    mask = torch.ones(2, 5, dtype=torch.bool)
    weights, locations, scales = model.forward_distribution(H, edit, dists, ["A", "C"], ["G", "U"], mask)
    assert weights.shape == locations.shape == scales.shape == (2, 5, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 5))
    assert torch.allclose(locations[..., 0], torch.zeros(2, 5))
    assert torch.all(scales > 0)


def test_loss_is_finite_with_missing_targets_removed_before_arithmetic():
    torch.manual_seed(0)
    model = AlignedDeltaModel(k_rank=0, sparse=True, d=16, heads=4, hidden=8)
    H = torch.randn(4, 16)
    edit = torch.tensor([1])
    dists = torch.arange(4)[None, :].float() - 1
    target = torch.tensor([[0.1, float("nan"), 0.4, 0.0]])
    qualified = torch.tensor([[True, False, True, True]])
    prediction_mask = torch.ones_like(qualified)
    wt = torch.tensor([0.0, 0.2, 0.1, 0.0])
    loss = aligned_mixture_loss(
        model, H, edit, dists, ["A"], ["G"], target, prediction_mask, qualified, wt, huber_lambda=0.1
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.all(torch.isfinite(p.grad)) for p in model.parameters())
