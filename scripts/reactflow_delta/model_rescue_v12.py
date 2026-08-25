#!/usr/bin/env python3
"""Continuous monotone residual shrinkage for Model Rescue v12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CANDIDATE = "v12_v11_monotone_regime_shrinkage"
TASK_MATCHED_NULL = "v12_gate_fixed_one_parent_null"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v12_prediction.v1"
MAGNITUDE_NORMALIZATION = 0.05
GATE_PARAMETERS = 4


class MonotoneRegimeGate(nn.Module):
    """Product gate that increases with edit distance and feature41 magnitude."""

    def __init__(self) -> None:
        super().__init__()
        self.b_distance = nn.Parameter(torch.zeros(()))
        self.raw_w_distance = nn.Parameter(torch.zeros(()))
        self.b_magnitude = nn.Parameter(torch.zeros(()))
        self.raw_w_magnitude = nn.Parameter(torch.zeros(()))

    def factors(
        self,
        absolute_distance: torch.Tensor,
        feature41_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if absolute_distance.shape != feature41_point.shape:
            raise ValueError("distance and feature41 must have identical shapes")
        if not torch.isfinite(absolute_distance).all() or not torch.isfinite(
            feature41_point
        ).all():
            raise ValueError("gate inputs must be finite")
        if (absolute_distance < 0).any():
            raise ValueError("absolute distance cannot be negative")
        distance_feature = torch.log1p(absolute_distance)
        magnitude_feature = torch.log1p(
            torch.abs(feature41_point) / MAGNITUDE_NORMALIZATION
        )
        distance = torch.sigmoid(
            self.b_distance + F.softplus(self.raw_w_distance) * distance_feature
        )
        magnitude = torch.sigmoid(
            self.b_magnitude + F.softplus(self.raw_w_magnitude) * magnitude_feature
        )
        return distance, magnitude

    def forward(
        self,
        absolute_distance: torch.Tensor,
        feature41_point: torch.Tensor,
    ) -> torch.Tensor:
        distance, magnitude = self.factors(absolute_distance, feature41_point)
        return distance * magnitude

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(parameter.detach().cpu())
            for name, parameter in self.named_parameters()
        }


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def gated_point(
    feature41_point: torch.Tensor,
    parent_v11_point: torch.Tensor,
    gate_value: torch.Tensor,
) -> torch.Tensor:
    if not (
        feature41_point.shape == parent_v11_point.shape == gate_value.shape
    ):
        raise ValueError("feature41, parent point, and gate must be aligned")
    if (gate_value < 0).any() or (gate_value > 1).any():
        raise ValueError("gate values must lie in [0, 1]")
    return feature41_point + gate_value * (parent_v11_point - feature41_point)


def fixed_parent_null(
    feature41_point: torch.Tensor, parent_v11_point: torch.Tensor
) -> torch.Tensor:
    if feature41_point.shape != parent_v11_point.shape:
        raise ValueError("feature41 and parent point must be aligned")
    return parent_v11_point


def hierarchy_weights(
    puzzles: Sequence[str],
    methods: Sequence[str],
    mutants: Sequence[str],
) -> np.ndarray:
    """Puzzle→method→mutant→position weights for inner-OOF gate fitting."""

    puzzles_array = np.asarray(puzzles, dtype=object)
    methods_array = np.asarray(methods, dtype=object)
    mutants_array = np.asarray(mutants, dtype=object)
    if not (
        puzzles_array.ndim == methods_array.ndim == mutants_array.ndim == 1
        and puzzles_array.shape == methods_array.shape == mutants_array.shape
        and len(puzzles_array) > 0
    ):
        raise ValueError("puzzle, method, and mutant labels must be aligned vectors")
    weights = np.zeros(len(puzzles_array), dtype=np.float64)
    puzzle_values = sorted(set(map(str, puzzles_array)))
    for puzzle in puzzle_values:
        puzzle_mask = puzzles_array == puzzle
        method_values = sorted(set(map(str, methods_array[puzzle_mask])))
        for method in method_values:
            method_mask = puzzle_mask & (methods_array == method)
            mutant_values = sorted(set(map(str, mutants_array[method_mask])))
            for mutant in mutant_values:
                position_mask = method_mask & (mutants_array == mutant)
                weights[position_mask] = (
                    1.0
                    / len(puzzle_values)
                    / len(method_values)
                    / len(mutant_values)
                    / int(position_mask.sum())
                )
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("V12 hierarchy weights do not sum to one")
    return weights


@dataclass(frozen=True)
class GateFitResult:
    gate: MonotoneRegimeGate
    history: list[float]


def fit_monotone_gate(
    *,
    feature41_point: np.ndarray,
    parent_v11_point: np.ndarray,
    target_delta: np.ndarray,
    absolute_distance: np.ndarray,
    puzzles: Sequence[str],
    methods: Sequence[str],
    mutants: Sequence[str],
    steps: int,
    learning_rate: float,
    device: str | torch.device,
) -> GateFitResult:
    """Fit the frozen four-parameter gate on legal inner-OOF predictions."""

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            feature41_point,
            parent_v11_point,
            target_delta,
            absolute_distance,
        )
    ]
    if any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays):
        raise ValueError("V12 gate arrays must be aligned one-dimensional vectors")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("V12 gate arrays must be finite")
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("gate steps and learning rate must be positive")
    weights = hierarchy_weights(puzzles, methods, mutants)
    gate = MonotoneRegimeGate().to(device=device, dtype=torch.float64)
    if trainable_parameter_count(gate) != GATE_PARAMETERS:
        raise RuntimeError("V12 gate parameter count differs from contract")
    feature = torch.as_tensor(arrays[0], device=device)
    parent = torch.as_tensor(arrays[1], device=device)
    target = torch.as_tensor(arrays[2], device=device)
    distance = torch.as_tensor(arrays[3], device=device)
    torch_weights = torch.as_tensor(weights, device=device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[float] = []
    for _step in range(steps):
        value = gate(distance, feature)
        prediction = gated_point(feature, parent, value)
        loss = torch.sum(torch_weights * torch.abs(target - prediction))
        optimizer.zero_grad()
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in gate.parameters()
        ):
            raise RuntimeError("V12 gate produced a missing or non-finite gradient")
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return GateFitResult(gate=gate, history=history)
