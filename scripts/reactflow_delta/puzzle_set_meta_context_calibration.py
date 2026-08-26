#!/usr/bin/env python3
"""Point-frozen V10 residual calibration for the proposed puzzle-set model."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v2 import gaussian_mixture_crps_torch
from scripts.reactflow_delta.model_rescue_v10 import (
    CapacitySymmetricResidual,
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    distribution_expected_absolute_delta,
    initialize_asymmetric_from_symmetric,
    mixture_cdf_at_point,
    parameter_count,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    PuzzleSetMetaContextPointModel,
)


POINT_NAMES = ("candidate", "null")
EXPECTED_RESIDUAL_PARAMETERS = 63_748


def freeze_point_model(model: PuzzleSetMetaContextPointModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _assert_snapshot(
    expected: dict[str, torch.Tensor], module: torch.nn.Module, label: str
) -> None:
    actual = module.state_dict()
    for name, value in expected.items():
        if not torch.equal(value, actual[name].detach().cpu()):
            raise RuntimeError(f"puzzle-set calibration changed frozen {label}: {name}")


def make_exact_residual_pair(
    *, seed: int, device: str | torch.device
) -> tuple[MedianAsymmetricResidual, MedianAsymmetricResidual]:
    """Create identical V10 median-asymmetric residual heads for both arms."""

    torch.manual_seed(int(seed))
    symmetric = CapacitySymmetricResidual().to(device)
    candidate = MedianAsymmetricResidual().to(device)
    initialize_asymmetric_from_symmetric(symmetric, candidate)
    null = copy.deepcopy(candidate)
    if parameter_count(candidate) != EXPECTED_RESIDUAL_PARAMETERS or parameter_count(
        null
    ) != EXPECTED_RESIDUAL_PARAMETERS:
        raise RuntimeError("puzzle-set residual parameter count changed")
    for left, right in zip(candidate.parameters(), null.parameters()):
        if not torch.equal(left.detach(), right.detach()):
            raise RuntimeError("puzzle-set residual initialization differs")
    return candidate, null


def build_calibration_cells(
    puzzle_batches: Sequence[dict[str, Any]],
    *,
    candidate: PuzzleSetMetaContextPointModel,
    null: PuzzleSetMetaContextPointModel,
) -> list[dict[str, Any]]:
    """Materialize outer-train-only rows after both point models are frozen."""

    if any(parameter.requires_grad for parameter in candidate.parameters()) or any(
        parameter.requires_grad for parameter in null.parameters()
    ):
        raise RuntimeError("puzzle-set point models must be frozen before calibration")
    output: list[dict[str, Any]] = []
    candidate.eval()
    null.eval()
    with torch.no_grad():
        for puzzle_batch in puzzle_batches:
            contexts = puzzle_batch["contexts"]
            cells = puzzle_batch["cells"]
            if len(contexts) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
                raise ValueError("puzzle-set calibration requires eight WT contexts")
            observed = [context[3].bool() for context in contexts]
            candidate_hidden = candidate.encode_puzzle_set(contexts)
            null_hidden = null.encode_puzzle_set(contexts)
            candidate_mixed = candidate.meta_context.mix_construct_tokens(
                candidate_hidden, observed
            )
            null_mixed = null.meta_context.mix_construct_tokens(null_hidden, observed)
            for source in cells:
                focal = int(source["focal_construct_index"])
                candidate_point, _ = candidate.forward_from_encoded(
                    candidate_hidden,
                    candidate_mixed,
                    focal,
                    source["edit_index"],
                    source["signed_distance"],
                    source["refs"],
                    source["alts"],
                    source["feature41_point"],
                    source["prediction_mask"],
                )
                null_point, _ = null.forward_from_encoded(
                    null_hidden,
                    null_mixed,
                    focal,
                    source["edit_index"],
                    source["signed_distance"],
                    source["refs"],
                    source["alts"],
                    source["feature41_point"],
                    source["prediction_mask"],
                )
                qualified = source["qualified_mask"].detach().cpu().numpy()
                target = source["target"].detach().cpu().numpy()
                wt = source["wt"].detach().cpu().numpy()
                feature41 = np.asarray(source["feature41_basis"], dtype=np.float32)
                direct = np.asarray(source["direct_features"], dtype=np.float32)
                expected = tuple(candidate_point.shape)
                if (
                    qualified.shape != expected
                    or target.shape != expected
                    or feature41.shape != (*expected, 41)
                    or direct.shape != (*expected, 201)
                ):
                    raise ValueError("puzzle-set calibration cell is misaligned")

                candidate_values = candidate_point.detach().cpu().numpy()
                null_values = null_point.detach().cpu().numpy()
                feature_rows = []
                direct_rows = []
                candidate_rows = []
                null_rows = []
                target_rows = []
                mutant_index = []
                valid_mutants = 0
                for mutant in range(len(qualified)):
                    receiver = np.flatnonzero(qualified[mutant])
                    if not len(receiver):
                        continue
                    feature_rows.append(feature41[mutant, receiver])
                    direct_rows.append(direct[mutant, receiver])
                    candidate_rows.append(candidate_values[mutant, receiver])
                    null_rows.append(null_values[mutant, receiver])
                    target_rows.append(target[mutant, receiver] - wt[receiver])
                    mutant_index.append(
                        np.full(len(receiver), valid_mutants, dtype=np.int64)
                    )
                    valid_mutants += 1
                if valid_mutants:
                    output.append(
                        {
                            "puzzle": str(puzzle_batch["puzzle"]),
                            "construct_id": str(source["construct_id"]),
                            "feature41": np.concatenate(feature_rows).astype(np.float32),
                            "direct_features": np.concatenate(direct_rows).astype(
                                np.float32
                            ),
                            "candidate_point": np.concatenate(candidate_rows).astype(
                                np.float32
                            ),
                            "null_point": np.concatenate(null_rows).astype(np.float32),
                            "target_delta": np.concatenate(target_rows).astype(
                                np.float32
                            ),
                            "mutant_index": np.concatenate(mutant_index),
                            "n_mutants": valid_mutants,
                        }
                    )
    if not output:
        raise RuntimeError("puzzle-set residual calibration produced no cells")
    return output


def _prepare_inputs(
    cells: list[dict[str, Any]], point_name: str
) -> tuple[TrainOnlyStandardizer, list[np.ndarray]]:
    point_field = f"{point_name}_point"
    raw = [
        calibration_input(
            cell["feature41"], cell[point_field], cell["direct_features"]
        )
        for cell in cells
    ]
    standardizer = TrainOnlyStandardizer.fit(raw)
    return standardizer, [standardizer.transform_numpy(value) for value in raw]


def _mutant_balanced_crps(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    mutant_index: torch.Tensor,
    n_mutants: int,
) -> torch.Tensor:
    values = gaussian_mixture_crps_torch(locations, scales, weights, target)
    sums = torch.zeros(n_mutants, device=values.device, dtype=values.dtype)
    counts = torch.zeros(n_mutants, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, mutant_index, values)
    counts.scatter_add_(0, mutant_index, torch.ones_like(values))
    if not bool((counts > 0.0).all()):
        raise RuntimeError("puzzle-set calibration contains an empty mutant")
    return (sums / counts).mean()


def fit_puzzle_balanced_residual_head(
    head: MedianAsymmetricResidual,
    cells: list[dict[str, Any]],
    inputs: list[np.ndarray],
    point_field: str,
    *,
    device: str,
    epochs: int,
    seed: int,
) -> list[float]:
    """Position -> mutant -> method cell -> puzzle CRPS optimization."""

    if len(cells) != len(inputs) or epochs < 1:
        raise ValueError("puzzle-set residual training inputs are incomplete")
    by_puzzle: dict[str, list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        by_puzzle[str(cell["puzzle"])].append(index)
    puzzles = sorted(by_puzzle)
    if not puzzles:
        raise ValueError("puzzle-set residual training requires puzzles")
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0.0)
    history = []
    for epoch in range(int(epochs)):
        order = list(range(len(puzzles)))
        random.Random(int(seed) * 100_003 + int(epoch)).shuffle(order)
        puzzle_losses = []
        for puzzle_index in order:
            cell_losses = []
            for index in by_puzzle[puzzles[puzzle_index]]:
                cell = cells[index]
                x = torch.tensor(inputs[index], device=device)
                point = torch.tensor(cell[point_field], device=device)
                target = torch.tensor(cell["target_delta"], device=device)
                mutant_index = torch.tensor(cell["mutant_index"], device=device)
                weights, locations, scales = head(point, x)
                cell_losses.append(
                    _mutant_balanced_crps(
                        weights,
                        locations,
                        scales,
                        target,
                        mutant_index,
                        int(cell["n_mutants"]),
                    )
                )
            loss = torch.stack(cell_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError(
                        f"nonfinite puzzle-set residual gradient in {name}"
                    )
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            puzzle_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(puzzle_losses)))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("puzzle-set residual history is incomplete or nonfinite")
    return history


def fit_residual_pair(
    puzzle_batches: Sequence[dict[str, Any]],
    *,
    candidate: PuzzleSetMetaContextPointModel,
    null: PuzzleSetMetaContextPointModel,
    epochs: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Freeze both points and fit one exact V10-family head per arm."""

    freeze_point_model(candidate)
    freeze_point_model(null)
    candidate_snapshot = _snapshot(candidate)
    null_snapshot = _snapshot(null)
    cells = build_calibration_cells(
        puzzle_batches, candidate=candidate, null=null
    )
    candidate_head, null_head = make_exact_residual_pair(seed=seed, device=device)
    heads = {"candidate": candidate_head, "null": null_head}
    standardizers = {}
    histories = {}
    for name in POINT_NAMES:
        standardizer, inputs = _prepare_inputs(cells, name)
        standardizers[name] = standardizer
        histories[name] = fit_puzzle_balanced_residual_head(
            heads[name],
            cells,
            inputs,
            f"{name}_point",
            device=device,
            epochs=int(epochs),
            seed=int(seed),
        )
    _assert_snapshot(candidate_snapshot, candidate, "candidate point")
    _assert_snapshot(null_snapshot, null, "null point")
    if any(parameter.grad is not None for parameter in candidate.parameters()) or any(
        parameter.grad is not None for parameter in null.parameters()
    ):
        raise RuntimeError("puzzle-set calibration produced point gradients")
    return {
        "heads": heads,
        "standardizers": standardizers,
        "histories": histories,
        "n_calibration_cells": len(cells),
    }


def calibrated_distribution(
    *,
    point: np.ndarray,
    feature41: np.ndarray,
    direct_features: np.ndarray,
    head: MedianAsymmetricResidual,
    standardizer: TrainOnlyStandardizer,
) -> dict[str, np.ndarray]:
    """Apply one train-only calibrator while preserving the supplied median."""

    raw = calibration_input(feature41, point, direct_features)
    device = next(head.parameters()).device
    standardized = torch.tensor(
        standardizer.transform_numpy(raw), device=device
    )
    point_tensor = torch.tensor(np.asarray(point, dtype=np.float64), device=device)
    head.eval()
    with torch.no_grad():
        weights, locations, scales = head(point_tensor, standardized)
        cdf = mixture_cdf_at_point(point_tensor, weights, locations, scales)
        if not torch.allclose(
            cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0
        ):
            raise RuntimeError("puzzle-set residual calibration moved point median")
        expected_absolute = distribution_expected_absolute_delta(
            weights, locations, scales
        )
    return {
        "weights": weights.cpu().numpy().astype(np.float64),
        "locations": locations.cpu().numpy().astype(np.float64),
        "scales": scales.cpu().numpy().astype(np.float64),
        "expected_absolute_delta": expected_absolute.cpu()
        .numpy()
        .astype(np.float64),
    }
