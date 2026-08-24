#!/usr/bin/env python3
"""Equi-calibrated zero-mean residual distributions for Model Rescue v9."""

from __future__ import annotations

import math

import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v2 import (
    expected_abs_normal,
    gaussian_mixture_crps_torch,
)
from scripts.reactflow_delta.run_p3_lrso_v3 import SCALE_FLOOR


FEATURE41_WIDTH = 41
CALIBRATION_FEATURE_WIDTH = FEATURE41_WIDTH + 2
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v9_prediction.v1"
FOLD_SCHEMA = "reactflow_delta.model_rescue_v9_fold.v1"


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1e-8)))


class EquiCalibratedZeroMeanMixture(nn.Module):
    """Two Gaussian scales centered exactly on an immutable point mean."""

    def __init__(self, feature_width: int = FEATURE41_WIDTH, hidden: int = 64) -> None:
        super().__init__()
        if feature_width != FEATURE41_WIDTH or hidden != 64:
            raise ValueError("V9 freezes feature width 41 and residual hidden size 64")
        self.head = nn.Sequential(
            nn.Linear(CALIBRATION_FEATURE_WIDTH, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        with torch.no_grad():
            self.head[-1].bias[0] = 0.0
            self.head[-1].bias[1] = _inverse_softplus(0.08)
            self.head[-1].bias[2] = _inverse_softplus(0.20)

    def forward(
        self, signed_mean: torch.Tensor, feature41: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feature41.shape[-1] != FEATURE41_WIDTH:
            raise ValueError("V9 residual input is not the frozen feature41 basis")
        mean = signed_mean.detach()
        inputs = torch.cat(
            [feature41.detach(), mean.unsqueeze(-1), mean.abs().unsqueeze(-1)], dim=-1
        )
        raw = self.head(inputs)
        narrow_weight = torch.sigmoid(raw[..., 0])
        narrow_scale = SCALE_FLOOR + torch.nn.functional.softplus(raw[..., 1])
        wide_scale = narrow_scale + torch.nn.functional.softplus(raw[..., 2])
        weights = torch.stack([narrow_weight, 1.0 - narrow_weight], dim=-1)
        locations = torch.stack([mean, mean], dim=-1)
        scales = torch.stack([narrow_scale, wide_scale], dim=-1)
        return weights, locations, scales


def expected_absolute_delta(
    weights: torch.Tensor, locations: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    value = torch.zeros_like(locations[..., 0])
    for component in range(locations.shape[-1]):
        value = value + weights[..., component] * expected_abs_normal(
            locations[..., component], scales[..., component]
        )
    return value


def mutant_balanced_crps(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    target_delta: torch.Tensor,
    mutant_index: torch.Tensor,
    n_mutants: int,
) -> torch.Tensor:
    """Position -> mutant -> one puzzle-method cell CRPS."""
    if n_mutants <= 0 or target_delta.numel() == 0:
        raise ValueError("V9 calibration cell must contain qualified mutants")
    values = gaussian_mixture_crps_torch(
        locations, scales, weights, target_delta
    )
    sums = torch.zeros(n_mutants, device=values.device, dtype=values.dtype)
    counts = torch.zeros(n_mutants, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, mutant_index, values)
    counts.scatter_add_(0, mutant_index, torch.ones_like(values))
    if not bool((counts > 0).all()):
        raise ValueError("V9 mutant index contains an empty mutant")
    return (sums / counts).mean()


def assert_zero_mean_distribution(
    signed_mean: torch.Tensor,
    weights: torch.Tensor,
    locations: torch.Tensor,
    *,
    atol: float = 1e-7,
) -> None:
    if not torch.allclose(
        locations[..., 0], signed_mean, atol=atol, rtol=0.0
    ) or not torch.allclose(
        locations[..., 1], signed_mean, atol=atol, rtol=0.0
    ):
        raise RuntimeError("V9 residual distribution changed a component location")
    residual_mean = (
        weights * (locations - signed_mean.detach().unsqueeze(-1))
    ).sum(-1)
    if not torch.allclose(
        residual_mean, torch.zeros_like(signed_mean), atol=atol, rtol=0.0
    ):
        raise RuntimeError("V9 residual distribution changed the signed point mean")
