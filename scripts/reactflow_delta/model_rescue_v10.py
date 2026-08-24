#!/usr/bin/env python3
"""Median-preserving residual distributions for Model Rescue v10."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v9 import expected_absolute_delta
from scripts.reactflow_delta.run_p3_lrso_v3 import SCALE_FLOOR


FEATURE41_WIDTH = 41
POINT_FEATURE_WIDTH = 2
DIRECT_FEATURE_WIDTH = 201
INPUT_WIDTH = FEATURE41_WIDTH + POINT_FEATURE_WIDTH + DIRECT_FEATURE_WIDTH
HIDDEN_WIDTH = 256
CDF_EPSILON = 1.0e-4
WEIGHT_FLOOR = 0.1
STANDARDIZATION_SCALE_FLOOR = 1.0e-6
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v10_prediction.v1"
FOLD_SCHEMA = "reactflow_delta.model_rescue_v10_fold.v1"


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1e-8)))


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, rows: list[np.ndarray]) -> "TrainOnlyStandardizer":
        if not rows:
            raise ValueError("V10 standardizer requires outer-train rows")
        values = np.concatenate(rows, axis=0).astype(np.float64, copy=False)
        if values.ndim != 2 or values.shape[1] != INPUT_WIDTH:
            raise ValueError("V10 standardizer input width changed")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale >= STANDARDIZATION_SCALE_FLOOR, scale, 1.0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise RuntimeError("V10 standardizer contains nonfinite statistics")
        return cls(mean=mean, scale=scale)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != INPUT_WIDTH:
            raise ValueError("V10 transform input width changed")
        return ((values - self.mean) / self.scale).astype(np.float32)

    def transform_torch(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != INPUT_WIDTH:
            raise ValueError("V10 transform input width changed")
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        scale = torch.as_tensor(self.scale, device=values.device, dtype=values.dtype)
        return (values - mean) / scale


def calibration_input(
    feature41: np.ndarray, point: np.ndarray, direct_features: np.ndarray
) -> np.ndarray:
    feature41 = np.asarray(feature41, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    direct_features = np.asarray(direct_features, dtype=np.float64)
    if feature41.ndim != 2 or feature41.shape[1] != FEATURE41_WIDTH:
        raise ValueError("V10 feature41 width changed")
    if point.shape != (len(feature41),):
        raise ValueError("V10 point vector is not aligned")
    if direct_features.shape != (len(feature41), DIRECT_FEATURE_WIDTH):
        raise ValueError("V10 direct features are not aligned")
    return np.concatenate(
        [feature41, point[:, None], np.abs(point)[:, None], direct_features], axis=1
    )


class CapacitySymmetricResidual(nn.Module):
    """Parameter-matched same-location scale-mixture null."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(INPUT_WIDTH, HIDDEN_WIDTH)
        self.output_layer = nn.Linear(HIDDEN_WIDTH, 3)
        self._initialize_output()

    def _initialize_output(self) -> None:
        with torch.no_grad():
            self.output_layer.bias[0] = 0.0
            self.output_layer.bias[1] = _inverse_softplus(0.08)
            self.output_layer.bias[2] = _inverse_softplus(0.20)

    def raw(self, standardized_input: torch.Tensor) -> torch.Tensor:
        return self.output_layer(torch.relu(self.input_layer(standardized_input)))

    def forward(
        self, point: torch.Tensor, standardized_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep the learned network in float32, but construct the reported
        # distribution in float64.  The asymmetric head uses inverse-normal
        # CDFs to impose an exact median; float32 inverse/forward CDF
        # round-trips can miss the frozen 3e-6 invariant near the allocation
        # boundary even when the algebraic constraint is satisfied.
        raw = self.raw(standardized_input).to(torch.float64)
        point = point.detach().to(torch.float64)
        narrow_weight = WEIGHT_FLOOR + (1.0 - 2.0 * WEIGHT_FLOOR) * torch.sigmoid(
            raw[..., 0]
        )
        narrow_scale = SCALE_FLOOR + torch.nn.functional.softplus(raw[..., 1])
        wide_scale = narrow_scale + torch.nn.functional.softplus(raw[..., 2])
        weights = torch.stack([narrow_weight, 1.0 - narrow_weight], dim=-1)
        locations = torch.stack([point, point], dim=-1)
        scales = torch.stack([narrow_scale, wide_scale], dim=-1)
        return weights, locations, scales


class MedianAsymmetricResidual(nn.Module):
    """Two-Gaussian mixture whose CDF at the frozen point is exactly 0.5."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(INPUT_WIDTH, HIDDEN_WIDTH)
        self.output_layer = nn.Linear(HIDDEN_WIDTH, 4)
        self._initialize_output()

    def _initialize_output(self) -> None:
        with torch.no_grad():
            self.output_layer.bias[0] = 0.0
            self.output_layer.bias[1] = _inverse_softplus(0.08)
            self.output_layer.bias[2] = _inverse_softplus(0.20)
            self.output_layer.weight[3].zero_()
            self.output_layer.bias[3] = 0.0

    def raw(self, standardized_input: torch.Tensor) -> torch.Tensor:
        return self.output_layer(torch.relu(self.input_layer(standardized_input)))

    @staticmethod
    def allocations(
        weight: torch.Tensor, allocation_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = torch.as_tensor(
            CDF_EPSILON, device=weight.device, dtype=weight.dtype
        )
        one = torch.ones((), device=weight.device, dtype=weight.dtype)
        lower_from_b = (0.5 - (1.0 - weight) * (1.0 - epsilon)) / weight
        upper_from_b = (0.5 - (1.0 - weight) * epsilon) / weight
        lower = torch.maximum(epsilon, lower_from_b)
        upper = torch.minimum(one - epsilon, upper_from_b)
        unit = torch.sigmoid(allocation_raw)
        a = torch.where(
            unit >= 0.5,
            0.5 + (upper - 0.5) * (2.0 * unit - 1.0),
            0.5 - (0.5 - lower) * (1.0 - 2.0 * unit),
        )
        b = (0.5 - weight * a) / (1.0 - weight)
        # Preserve the exact nested-null forward value without cutting the
        # allocation gradient at its zero initialization.
        b = torch.where(
            allocation_raw == 0.0,
            b + (torch.full_like(b, 0.5) - b).detach(),
            b,
        )
        return a, b

    def forward(
        self, point: torch.Tensor, standardized_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.raw(standardized_input).to(torch.float64)
        point = point.detach().to(torch.float64)
        narrow_weight = WEIGHT_FLOOR + (1.0 - 2.0 * WEIGHT_FLOOR) * torch.sigmoid(
            raw[..., 0]
        )
        narrow_scale = SCALE_FLOOR + torch.nn.functional.softplus(raw[..., 1])
        wide_scale = narrow_scale + torch.nn.functional.softplus(raw[..., 2])
        a, b = self.allocations(narrow_weight, raw[..., 3])
        narrow_residual_location = -narrow_scale * torch.special.ndtri(a)
        wide_residual_location = -wide_scale * torch.special.ndtri(b)
        weights = torch.stack([narrow_weight, 1.0 - narrow_weight], dim=-1)
        locations = torch.stack(
            [
                point + narrow_residual_location,
                point + wide_residual_location,
            ],
            dim=-1,
        )
        scales = torch.stack([narrow_scale, wide_scale], dim=-1)
        return weights, locations, scales


def initialize_asymmetric_from_symmetric(
    symmetric: CapacitySymmetricResidual, asymmetric: MedianAsymmetricResidual
) -> None:
    """Copy every common parameter and initialize only allocation at the null."""
    with torch.no_grad():
        asymmetric.input_layer.weight.copy_(symmetric.input_layer.weight)
        asymmetric.input_layer.bias.copy_(symmetric.input_layer.bias)
        asymmetric.output_layer.weight[:3].copy_(symmetric.output_layer.weight)
        asymmetric.output_layer.bias[:3].copy_(symmetric.output_layer.bias)
        asymmetric.output_layer.weight[3].zero_()
        asymmetric.output_layer.bias[3].zero_()


def mixture_cdf_at_point(
    point: torch.Tensor,
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    z = (point.detach().unsqueeze(-1) - locations) / scales
    return torch.sum(weights * torch.special.ndtr(z), dim=-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def distribution_expected_absolute_delta(
    weights: torch.Tensor, locations: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    return expected_absolute_delta(weights, locations, scales)
