#!/usr/bin/env python3
"""Inactive mathematical core for post-V14 P2 quantile residual fitting.

This module deliberately contains no authority, artifact, held-score, or device
policy code.  A later authorized runtime owns those boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr
import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v2 import gaussian_mixture_crps_torch
from scripts.reactflow_delta.model_rescue_v10 import (
    DIRECT_FEATURE_WIDTH as V10_DIRECT_FEATURE_WIDTH,
    FEATURE41_WIDTH as V10_FEATURE41_WIDTH,
    INPUT_WIDTH as V10_INPUT_WIDTH,
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    distribution_expected_absolute_delta,
    parameter_count as v10_parameter_count,
)


FEATURE41_WIDTH = V10_FEATURE41_WIDTH
POINT_FEATURE_WIDTH = 2
DIRECT_FEATURE_WIDTH = V10_DIRECT_FEATURE_WIDTH
INPUT_WIDTH = V10_INPUT_WIDTH
CANDIDATE_HIDDEN_WIDTH = 248
N_QUANTILES = 13
N_GAPS = 12
MEDIAN_INDEX = 6
GAP_FLOOR = 1.0e-4
EXPECTED_PARAMETER_COUNT = 63_748

TAUS = (
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
)
QUADRATURE_WEIGHTS = (
    0.0375,
    0.0375,
    0.075,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.075,
    0.0375,
    0.0375,
)

INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0 = (
    "INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0"
)
INITIAL_GRID_REPLAY_ATOL = 1.0e-6
INITIAL_GRID_REPLAY_RTOL = 0.0
POINT_REPLAY_ATOL = 1.0e-7
POINT_REPLAY_RTOL = 0.0

ADAM_LEARNING_RATE = 1.0e-3
ADAM_WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 5.0
PUZZLE_ORDER_SEED_MULTIPLIER = 100_003
V10_INITIAL_NARROW_SCALE_RAW_TARGET = 0.08
V10_INITIAL_WIDE_GAP_RAW_TARGET = 0.20
INVERSE_CDF_BOUND_SIGMAS = 16.0
INVERSE_CDF_BISECTION_STEPS = 128


class MonotoneQuantileResidual(nn.Module):
    """244 -> 248 -> 12 positive gaps around an exact detached median."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(INPUT_WIDTH, CANDIDATE_HIDDEN_WIDTH)
        self.output_layer = nn.Linear(CANDIDATE_HIDDEN_WIDTH, N_GAPS)

    def raw(self, standardized_input: torch.Tensor) -> torch.Tensor:
        if standardized_input.ndim != 2 or standardized_input.shape[1] != INPUT_WIDTH:
            raise ValueError("P2 standardized input width changed")
        if standardized_input.dtype != torch.float32:
            raise ValueError("P2 learned layers require float32 input")
        if not bool(torch.isfinite(standardized_input).all()):
            raise ValueError("P2 standardized input contains nonfinite values")
        return self.output_layer(torch.relu(self.input_layer(standardized_input)))

    def forward(
        self, point: torch.Tensor, standardized_input: torch.Tensor
    ) -> torch.Tensor:
        if point.ndim != 1 or point.shape[0] != standardized_input.shape[0]:
            raise ValueError("P2 frozen point is not aligned")
        if not bool(torch.isfinite(point).all()):
            raise ValueError("P2 frozen point contains nonfinite values")
        raw_gaps = self.raw(standardized_input).to(torch.float64)
        gaps = GAP_FLOOR + torch.nn.functional.softplus(raw_gaps)
        median = point.detach().to(torch.float64).unsqueeze(-1)
        lower = median - torch.flip(
            torch.cumsum(torch.flip(gaps[..., :MEDIAN_INDEX], dims=(-1,)), dim=-1),
            dims=(-1,),
        )
        upper = median + torch.cumsum(gaps[..., MEDIAN_INDEX:], dim=-1)
        return torch.cat([lower, median, upper], dim=-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def inverse_softplus(value: float | np.ndarray) -> float | np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if not np.isfinite(values).all() or bool((values <= 0.0).any()):
        raise ValueError("inverse softplus requires finite positive values")
    result = values + np.log(-np.expm1(-values))
    if values.ndim == 0:
        return float(result)
    return result


def initialize_p2_v10_comparator(model: MedianAsymmetricResidual) -> None:
    """Freeze the input-independent P2-specific V10 replay initialization."""

    bias = np.asarray(
        [
            0.0,
            inverse_softplus(V10_INITIAL_NARROW_SCALE_RAW_TARGET),
            inverse_softplus(V10_INITIAL_WIDE_GAP_RAW_TARGET),
            0.0,
        ],
        dtype=np.float64,
    )
    with torch.no_grad():
        model.output_layer.weight.zero_()
        model.output_layer.bias.copy_(
            torch.as_tensor(
                bias,
                dtype=model.output_layer.bias.dtype,
                device=model.output_layer.bias.device,
            )
        )


def gaussian_mixture_quantiles_float64(
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    taus: Sequence[float] = TAUS,
) -> np.ndarray:
    """Invert aligned Gaussian mixtures by fixed bounded float64 bisection."""

    weights = np.asarray(weights, dtype=np.float64)
    locations = np.asarray(locations, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    taus_array = np.asarray(taus, dtype=np.float64)
    if not (
        weights.ndim == locations.ndim == scales.ndim == 2
        and weights.shape == locations.shape == scales.shape
        and weights.shape[0] > 0
        and weights.shape[1] > 0
    ):
        raise ValueError("P2 Gaussian mixture arrays are not aligned matrices")
    if taus_array.ndim != 1 or len(taus_array) == 0:
        raise ValueError("P2 inverse-CDF taus must be a nonempty vector")
    if not (
        np.isfinite(weights).all()
        and np.isfinite(locations).all()
        and np.isfinite(scales).all()
        and np.isfinite(taus_array).all()
    ):
        raise ValueError("P2 inverse-CDF inputs contain nonfinite values")
    if bool((weights < 0.0).any()) or bool((scales <= 0.0).any()):
        raise ValueError("P2 Gaussian mixture weights/scales are invalid")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("P2 Gaussian mixture weights do not sum to one")
    if bool((taus_array <= 0.0).any()) or bool((taus_array >= 1.0).any()):
        raise ValueError("P2 inverse-CDF taus must lie strictly inside (0, 1)")
    if bool((np.diff(taus_array) <= 0.0).any()):
        raise ValueError("P2 inverse-CDF taus must be strictly increasing")

    lower_bound = np.min(
        locations - INVERSE_CDF_BOUND_SIGMAS * scales, axis=1
    )
    upper_bound = np.max(
        locations + INVERSE_CDF_BOUND_SIGMAS * scales, axis=1
    )

    def mixture_cdf(value: np.ndarray) -> np.ndarray:
        z = (value[:, None] - locations) / scales
        return np.sum(weights * ndtr(z), axis=1, dtype=np.float64)

    if bool((mixture_cdf(lower_bound) >= taus_array[0]).any()) or bool(
        (mixture_cdf(upper_bound) <= taus_array[-1]).any()
    ):
        raise RuntimeError("P2 fixed inverse-CDF bounds do not bracket the grid")

    result = np.empty((weights.shape[0], len(taus_array)), dtype=np.float64)
    for column, tau in enumerate(taus_array):
        lower = lower_bound.copy()
        upper = upper_bound.copy()
        for _ in range(INVERSE_CDF_BISECTION_STEPS):
            midpoint = (lower + upper) * 0.5
            below = mixture_cdf(midpoint) < tau
            lower = np.where(below, midpoint, lower)
            upper = np.where(below, upper, midpoint)
        result[:, column] = (lower + upper) * 0.5
    return result


def registered_v10_initial_grid(
    comparator: MedianAsymmetricResidual,
    point: torch.Tensor,
    standardized_input: torch.Tensor,
) -> np.ndarray:
    comparator.eval()
    with torch.no_grad():
        weights, locations, scales = comparator(point, standardized_input)
    return gaussian_mixture_quantiles_float64(
        weights.detach().cpu().numpy(),
        locations.detach().cpu().numpy(),
        scales.detach().cpu().numpy(),
    )


def initialize_candidate_from_registered_grid(
    candidate: MonotoneQuantileResidual, registered_grid: np.ndarray
) -> None:
    registered_grid = np.asarray(registered_grid, dtype=np.float64)
    if registered_grid.shape != (N_QUANTILES,) or not np.isfinite(
        registered_grid
    ).all():
        raise ValueError("P2 registered initial grid is invalid")
    target_gaps = np.diff(registered_grid)
    if bool((target_gaps <= GAP_FLOOR).any()):
        raise ValueError("P2 target adjacent gap must be strictly greater than 1e-4")
    raw_bias = inverse_softplus(target_gaps - GAP_FLOOR)
    with torch.no_grad():
        candidate.output_layer.weight.zero_()
        candidate.output_layer.bias.copy_(
            torch.as_tensor(
                raw_bias,
                dtype=candidate.output_layer.bias.dtype,
                device=candidate.output_layer.bias.device,
            )
        )


def assert_initial_grid_replay(
    candidate_grid: np.ndarray,
    registered_comparator_grid: np.ndarray,
    *,
    atol: float = INITIAL_GRID_REPLAY_ATOL,
    rtol: float = INITIAL_GRID_REPLAY_RTOL,
) -> None:
    if atol != INITIAL_GRID_REPLAY_ATOL or rtol != INITIAL_GRID_REPLAY_RTOL:
        raise ValueError(
            "P2 initial replay requires INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0"
        )
    candidate_grid = np.asarray(candidate_grid, dtype=np.float64)
    registered_comparator_grid = np.asarray(
        registered_comparator_grid, dtype=np.float64
    )
    if (
        candidate_grid.shape != registered_comparator_grid.shape
        or candidate_grid.shape[-1:] != (N_QUANTILES,)
        or not np.isfinite(candidate_grid).all()
        or not np.isfinite(registered_comparator_grid).all()
    ):
        raise ValueError("P2 initial replay grids are invalid or unaligned")
    if not np.allclose(
        candidate_grid,
        registered_comparator_grid,
        atol=INITIAL_GRID_REPLAY_ATOL,
        rtol=INITIAL_GRID_REPLAY_RTOL,
    ):
        raise RuntimeError("P2 candidate initial grid does not replay registered V10")


def build_grid_matched_models(
    seed: int, device: str | torch.device = "cpu"
) -> tuple[MonotoneQuantileResidual, MedianAsymmetricResidual]:
    """Create both 63,748-parameter heads from the same seed and P2 init."""

    torch.manual_seed(seed)
    candidate = MonotoneQuantileResidual().to(device)
    torch.manual_seed(seed)
    comparator = MedianAsymmetricResidual().to(device)
    initialize_p2_v10_comparator(comparator)

    zero_input = torch.zeros((1, INPUT_WIDTH), dtype=torch.float32, device=device)
    zero_point = torch.zeros(1, dtype=torch.float64, device=device)
    registered = registered_v10_initial_grid(comparator, zero_point, zero_input)[0]
    initialize_candidate_from_registered_grid(candidate, registered)
    candidate.eval()
    with torch.no_grad():
        realized = candidate(zero_point, zero_input).detach().cpu().numpy()[0]
    assert_initial_grid_replay(realized, registered)
    if parameter_count(candidate) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("P2 candidate parameter count changed")
    if v10_parameter_count(comparator) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("P2 matched V10 parameter count changed")
    return candidate, comparator


def _fixed_vector(
    values: torch.Tensor, *, name: str, last_width: int | None = None
) -> torch.Tensor:
    if last_width is None:
        if values.ndim != 1:
            raise ValueError(f"{name} must be a vector")
    elif values.ndim < 1 or values.shape[-1] != last_width:
        raise ValueError(f"{name} has the wrong final width")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return values


def weighted_pinball_training_surrogate(
    target: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Per-position candidate training surrogate; this is not CRPS."""

    _fixed_vector(quantiles, name="P2 quantiles", last_width=N_QUANTILES)
    _fixed_vector(target, name="P2 target")
    if quantiles.shape[:-1] != target.shape:
        raise ValueError("P2 target and quantiles are not aligned")
    target64 = target.to(torch.float64)
    quantiles64 = quantiles.to(torch.float64)
    taus = torch.as_tensor(TAUS, dtype=torch.float64, device=quantiles.device)
    weights = torch.as_tensor(
        QUADRATURE_WEIGHTS, dtype=torch.float64, device=quantiles.device
    )
    residual = target64.unsqueeze(-1) - quantiles64
    pinball = residual * (taus - (residual < 0.0).to(torch.float64))
    return 2.0 * torch.sum(weights * pinball, dim=-1)


def finite_atom_scientific_crps(
    target: torch.Tensor, atom_values: torch.Tensor
) -> torch.Tensor:
    """Exact CRPS of the declared fixed-mass finite atom distribution."""

    _fixed_vector(atom_values, name="P2 atom values", last_width=N_QUANTILES)
    _fixed_vector(target, name="P2 target")
    if atom_values.shape[:-1] != target.shape:
        raise ValueError("P2 target and atom values are not aligned")
    target64 = target.to(torch.float64)
    atom64 = atom_values.to(torch.float64)
    weights = torch.as_tensor(
        QUADRATURE_WEIGHTS, dtype=torch.float64, device=atom_values.device
    )
    first = torch.sum(weights * torch.abs(target64.unsqueeze(-1) - atom64), dim=-1)
    pair_distance = torch.abs(atom64.unsqueeze(-1) - atom64.unsqueeze(-2))
    pair_weights = weights.unsqueeze(-1) * weights.unsqueeze(-2)
    return first - 0.5 * torch.sum(pair_weights * pair_distance, dim=(-2, -1))


def weighted_expected_absolute_delta(atom_values: torch.Tensor) -> torch.Tensor:
    """Exact E|delta| for the declared fixed-mass finite atom distribution."""

    _fixed_vector(atom_values, name="P2 atom values", last_width=N_QUANTILES)
    weights = torch.as_tensor(
        QUADRATURE_WEIGHTS, dtype=torch.float64, device=atom_values.device
    )
    return torch.sum(weights * torch.abs(atom_values.to(torch.float64)), dim=-1)


def equal_puzzle_hierarchy_mean(
    position_values: torch.Tensor,
    mutant_index: torch.Tensor,
    method_cell_index: torch.Tensor,
    puzzle_index: torch.Tensor,
) -> torch.Tensor:
    """Position -> equal mutant -> equal method cell -> equal puzzle mean."""

    _fixed_vector(position_values, name="P2 position values")
    vectors = (mutant_index, method_cell_index, puzzle_index)
    if any(value.ndim != 1 or value.shape != position_values.shape for value in vectors):
        raise ValueError("P2 hierarchy indices are not aligned vectors")
    puzzle_means = []
    for puzzle in torch.unique(puzzle_index, sorted=True):
        puzzle_mask = puzzle_index == puzzle
        cell_means = []
        for cell in torch.unique(method_cell_index[puzzle_mask], sorted=True):
            cell_mask = puzzle_mask & (method_cell_index == cell)
            mutant_means = []
            for mutant in torch.unique(mutant_index[cell_mask], sorted=True):
                mutant_mask = cell_mask & (mutant_index == mutant)
                mutant_means.append(position_values[mutant_mask].mean())
            cell_means.append(torch.stack(mutant_means).mean())
        puzzle_means.append(torch.stack(cell_means).mean())
    if not puzzle_means:
        raise ValueError("P2 hierarchy requires outer-train rows")
    return torch.stack(puzzle_means).mean()


@dataclass(frozen=True)
class P2OuterTrainRows:
    feature41: np.ndarray
    direct_features: np.ndarray
    target_delta: np.ndarray
    puzzle: Sequence[Any]
    method_cell: Sequence[Any]
    mutant: Sequence[Any]


@dataclass(frozen=True)
class PairedFitResult:
    candidate: MonotoneQuantileResidual
    comparator: MedianAsymmetricResidual
    standardizer: TrainOnlyStandardizer
    seed: int
    epochs: int
    puzzle_orders: tuple[tuple[Any, ...], ...]
    candidate_history: tuple[float, ...]
    comparator_history: tuple[float, ...]
    candidate_quantiles: np.ndarray
    candidate_expected_absolute_delta: np.ndarray
    v10_replay_weights: np.ndarray
    v10_replay_locations: np.ndarray
    v10_replay_scales: np.ndarray
    v10_replay_expected_absolute_delta: np.ndarray


def _aligned_labels(values: Sequence[Any], row_count: int, name: str) -> np.ndarray:
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1 or len(labels) != row_count:
        raise ValueError(f"P2 {name} labels are not aligned")
    return labels


def _factorize_in_first_seen_order(labels: np.ndarray) -> tuple[np.ndarray, tuple[Any, ...]]:
    mapping: dict[Any, int] = {}
    ordered = []
    codes = np.empty(len(labels), dtype=np.int64)
    for index, raw_label in enumerate(labels.tolist()):
        try:
            code = mapping.get(raw_label)
        except TypeError as exc:
            raise ValueError("P2 hierarchy labels must be hashable") from exc
        if code is None:
            code = len(ordered)
            mapping[raw_label] = code
            ordered.append(raw_label)
        codes[index] = code
    return codes, tuple(ordered)


def _finite_gradients(model: nn.Module, label: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite P2 {label} gradient in {name}")


def fit_paired_quantile_and_v10(
    rows: P2OuterTrainRows,
    frozen_point: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str | torch.device = "cpu",
) -> PairedFitResult:
    """Fit candidate and matched replay on one identical authorized row universe."""

    feature41 = np.asarray(rows.feature41, dtype=np.float64)
    direct_features = np.asarray(rows.direct_features, dtype=np.float64)
    target_delta = np.asarray(rows.target_delta, dtype=np.float64)
    point_snapshot = np.asarray(frozen_point, dtype=np.float64).copy()
    if feature41.ndim != 2 or feature41.shape[1] != FEATURE41_WIDTH:
        raise ValueError("P2 feature41 width changed")
    row_count = len(feature41)
    if row_count == 0:
        raise ValueError("P2 fitting requires outer-train rows")
    if direct_features.shape != (row_count, DIRECT_FEATURE_WIDTH):
        raise ValueError("P2 direct features are not aligned")
    if target_delta.shape != (row_count,) or point_snapshot.shape != (row_count,):
        raise ValueError("P2 target or frozen point is not aligned")
    if not (
        np.isfinite(feature41).all()
        and np.isfinite(direct_features).all()
        and np.isfinite(target_delta).all()
        and np.isfinite(point_snapshot).all()
    ):
        raise ValueError("P2 fitting inputs contain nonfinite values")
    if epochs <= 0:
        raise ValueError("P2 fitting epochs must be positive")

    puzzle_labels = _aligned_labels(rows.puzzle, row_count, "puzzle")
    method_labels = _aligned_labels(rows.method_cell, row_count, "method-cell")
    mutant_labels = _aligned_labels(rows.mutant, row_count, "mutant")
    puzzle_codes, ordered_puzzles = _factorize_in_first_seen_order(puzzle_labels)
    method_codes, _ = _factorize_in_first_seen_order(method_labels)
    mutant_codes, _ = _factorize_in_first_seen_order(mutant_labels)

    raw_input = calibration_input(feature41, point_snapshot, direct_features)
    standardizer = TrainOnlyStandardizer.fit([raw_input])
    standardized = standardizer.transform_numpy(raw_input)
    candidate, comparator = build_grid_matched_models(seed, device)
    candidate.train()
    comparator.train()
    candidate_optimizer = torch.optim.Adam(
        candidate.parameters(), lr=ADAM_LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY
    )
    comparator_optimizer = torch.optim.Adam(
        comparator.parameters(), lr=ADAM_LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY
    )

    candidate_history = []
    comparator_history = []
    puzzle_orders = []
    for epoch in range(epochs):
        order = list(range(len(ordered_puzzles)))
        random.Random(seed * PUZZLE_ORDER_SEED_MULTIPLIER + epoch).shuffle(order)
        puzzle_orders.append(tuple(ordered_puzzles[index] for index in order))
        candidate_losses = []
        comparator_losses = []
        for puzzle_code in order:
            row_indices = np.flatnonzero(puzzle_codes == puzzle_code)
            x = torch.as_tensor(
                standardized[row_indices], dtype=torch.float32, device=device
            )
            target = torch.as_tensor(
                target_delta[row_indices], dtype=torch.float64, device=device
            )
            mutant = torch.as_tensor(
                mutant_codes[row_indices], dtype=torch.int64, device=device
            )
            method = torch.as_tensor(
                method_codes[row_indices], dtype=torch.int64, device=device
            )
            puzzle = torch.as_tensor(
                puzzle_codes[row_indices], dtype=torch.int64, device=device
            )

            candidate_point = torch.as_tensor(
                point_snapshot[row_indices], dtype=torch.float64, device=device
            ).clone().requires_grad_(True)
            candidate_quantiles = candidate(candidate_point, x)
            candidate_loss = equal_puzzle_hierarchy_mean(
                weighted_pinball_training_surrogate(target, candidate_quantiles),
                mutant,
                method,
                puzzle,
            )
            candidate_optimizer.zero_grad(set_to_none=True)
            candidate_loss.backward()
            if candidate_point.grad is not None:
                raise RuntimeError("P2 candidate propagated a frozen-point gradient")
            _finite_gradients(candidate, "candidate")
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), GRADIENT_CLIP_NORM)
            candidate_optimizer.step()
            candidate_losses.append(float(candidate_loss.detach().cpu()))

            comparator_point = torch.as_tensor(
                point_snapshot[row_indices], dtype=torch.float64, device=device
            ).clone().requires_grad_(True)
            weights, locations, scales = comparator(comparator_point, x)
            comparator_loss = equal_puzzle_hierarchy_mean(
                gaussian_mixture_crps_torch(locations, scales, weights, target),
                mutant,
                method,
                puzzle,
            )
            comparator_optimizer.zero_grad(set_to_none=True)
            comparator_loss.backward()
            if comparator_point.grad is not None:
                raise RuntimeError("P2 comparator propagated a frozen-point gradient")
            _finite_gradients(comparator, "matched V10")
            torch.nn.utils.clip_grad_norm_(comparator.parameters(), GRADIENT_CLIP_NORM)
            comparator_optimizer.step()
            comparator_losses.append(float(comparator_loss.detach().cpu()))

        candidate_history.append(float(np.mean(candidate_losses)))
        comparator_history.append(float(np.mean(comparator_losses)))

    if not (
        len(candidate_history) == len(comparator_history) == epochs
        and np.isfinite(candidate_history).all()
        and np.isfinite(comparator_history).all()
    ):
        raise RuntimeError("P2 paired fitting history is incomplete or nonfinite")
    if not np.array_equal(np.asarray(frozen_point, dtype=np.float64), point_snapshot):
        raise RuntimeError("P2 paired fitting changed the frozen point")

    candidate.eval()
    comparator.eval()
    with torch.no_grad():
        all_input = torch.as_tensor(standardized, dtype=torch.float32, device=device)
        all_point = torch.as_tensor(point_snapshot, dtype=torch.float64, device=device)
        candidate_quantiles = candidate(all_point, all_input)
        weights, locations, scales = comparator(all_point, all_input)
        candidate_absolute = weighted_expected_absolute_delta(candidate_quantiles)
        comparator_absolute = distribution_expected_absolute_delta(
            weights, locations, scales
        )
    candidate_quantiles_numpy = candidate_quantiles.detach().cpu().numpy()
    if not np.array_equal(candidate_quantiles_numpy[:, MEDIAN_INDEX], point_snapshot):
        raise RuntimeError("P2 candidate median drifted from the frozen point")
    if not bool((np.diff(candidate_quantiles_numpy, axis=1) > 0.0).all()):
        raise RuntimeError("P2 candidate quantiles are not strictly increasing")

    return PairedFitResult(
        candidate=candidate,
        comparator=comparator,
        standardizer=standardizer,
        seed=seed,
        epochs=epochs,
        puzzle_orders=tuple(puzzle_orders),
        candidate_history=tuple(candidate_history),
        comparator_history=tuple(comparator_history),
        candidate_quantiles=candidate_quantiles_numpy,
        candidate_expected_absolute_delta=candidate_absolute.detach().cpu().numpy(),
        v10_replay_weights=weights.detach().cpu().numpy(),
        v10_replay_locations=locations.detach().cpu().numpy(),
        v10_replay_scales=scales.detach().cpu().numpy(),
        v10_replay_expected_absolute_delta=comparator_absolute.detach().cpu().numpy(),
    )
