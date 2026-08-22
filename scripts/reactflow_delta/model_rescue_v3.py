#!/usr/bin/env python3
"""Core disagreement-gated expert blend for Model Rescue v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch


CANDIDATE = "b1_meanaligned_disagreement_gate_calibrated_residual"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v3_prediction.v1"
INNER_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v3_inner_prediction.v1"
GATE_QUANTILE = 0.95


@dataclass(frozen=True)
class DisagreementGate:
    threshold: float
    alpha_low: float
    alpha_high: float
    quantile: float = GATE_QUANTILE

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(valid.any()) or not 0.0 <= q <= 1.0:
        raise ValueError("weighted_quantile requires valid rows and q in [0,1]")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def fit_convex_l1_alpha(
    target: np.ndarray,
    b1: np.ndarray,
    meanaligned: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Exact weighted-L1 convex alpha from prediction breakpoints."""
    target = np.asarray(target, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    meanaligned = np.asarray(meanaligned, dtype=float)
    weights = np.asarray(weights, dtype=float)
    difference = meanaligned - b1
    valid = (
        np.isfinite(target)
        & np.isfinite(b1)
        & np.isfinite(meanaligned)
        & np.isfinite(weights)
        & (weights > 0)
        & (np.abs(difference) > 1e-12)
    )
    if not bool(valid.any()):
        return 1.0
    breakpoint = (target[valid] - b1[valid]) / difference[valid]
    effective_weight = weights[valid] * np.abs(difference[valid])
    alpha = weighted_quantile(breakpoint, effective_weight, 0.5)
    return float(np.clip(alpha, 0.0, 1.0))


def fit_disagreement_gate(
    target: np.ndarray,
    b1: np.ndarray,
    meanaligned: np.ndarray,
    hierarchy_weights: np.ndarray,
    *,
    quantile: float = GATE_QUANTILE,
) -> DisagreementGate:
    if quantile != GATE_QUANTILE:
        raise ValueError(f"v3 gate quantile is frozen at {GATE_QUANTILE}")
    disagreement = np.abs(np.asarray(b1, dtype=float) - np.asarray(meanaligned, dtype=float))
    threshold = weighted_quantile(disagreement, hierarchy_weights, quantile)
    high = disagreement > threshold
    low = ~high
    if not bool(low.any()) or not bool(high.any()):
        raise ValueError("frozen q95 gate must contain both low and high rows")
    alpha_low = fit_convex_l1_alpha(
        np.asarray(target)[low],
        np.asarray(b1)[low],
        np.asarray(meanaligned)[low],
        np.asarray(hierarchy_weights)[low],
    )
    alpha_high = fit_convex_l1_alpha(
        np.asarray(target)[high],
        np.asarray(b1)[high],
        np.asarray(meanaligned)[high],
        np.asarray(hierarchy_weights)[high],
    )
    return DisagreementGate(
        threshold=threshold,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        quantile=quantile,
    )


def apply_disagreement_gate_numpy(
    b1: np.ndarray, meanaligned: np.ndarray, gate: DisagreementGate
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b1 = np.asarray(b1, dtype=float)
    meanaligned = np.asarray(meanaligned, dtype=float)
    disagreement = np.abs(b1 - meanaligned)
    alpha = np.where(
        disagreement > gate.threshold, gate.alpha_high, gate.alpha_low
    )
    blend = b1 + alpha * (meanaligned - b1)
    return blend, alpha, disagreement


def apply_disagreement_gate_torch(
    b1: torch.Tensor,
    meanaligned: torch.Tensor,
    gate: DisagreementGate,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    disagreement = torch.abs(b1 - meanaligned)
    low = torch.as_tensor(gate.alpha_low, dtype=b1.dtype, device=b1.device)
    high = torch.as_tensor(gate.alpha_high, dtype=b1.dtype, device=b1.device)
    alpha = torch.where(disagreement > gate.threshold, high, low)
    blend = b1 + alpha * (meanaligned - b1)
    return blend, alpha, disagreement


def build_inner_crossfit_ledger(
    outer_train_puzzles: Iterable[str], inner_groups: Iterable[Iterable[str]]
) -> list[dict[str, Any]]:
    """Validate and materialize disjoint inner-held puzzle groups."""
    outer = sorted(set(outer_train_puzzles))
    groups = [sorted(set(group)) for group in inner_groups]
    if len(groups) != 4:
        raise ValueError("v3 requires exactly four inner puzzle groups")
    flattened = [puzzle for group in groups for puzzle in group]
    if sorted(flattened) != outer or len(flattened) != len(set(flattened)):
        raise ValueError("inner groups must cover each outer-train puzzle exactly once")
    rows = []
    outer_set = set(outer)
    for inner_fold, held in enumerate(groups):
        held_set = set(held)
        train = sorted(outer_set - held_set)
        if not held or held_set & set(train):
            raise ValueError("inner held groups must be nonempty and disjoint from train")
        rows.append(
            {
                "inner_fold": inner_fold,
                "train_puzzles": train,
                "held_puzzles": held,
                "held_zero_exposure": True,
            }
        )
    return rows


def hierarchy_position_weights(
    puzzle: np.ndarray,
    method: np.ndarray,
    mutant: np.ndarray,
) -> np.ndarray:
    """Weights for puzzle -> method -> mutant -> position exact macro L1."""
    puzzle = np.asarray(puzzle, dtype=object)
    method = np.asarray(method, dtype=object)
    mutant = np.asarray(mutant, dtype=object)
    if not (len(puzzle) == len(method) == len(mutant)) or len(puzzle) == 0:
        raise ValueError("hierarchy weight inputs must be nonempty and aligned")
    puzzles = sorted(set(puzzle.tolist()))
    result = np.zeros(len(puzzle), dtype=float)
    for p in puzzles:
        p_selected = puzzle == p
        methods = sorted(set(method[p_selected].tolist()))
        for m in methods:
            cell = p_selected & (method == m)
            mutants = sorted(set(mutant[cell].tolist()))
            for u in mutants:
                positions = cell & (mutant == u)
                result[positions] = 1.0 / (
                    len(puzzles)
                    * len(methods)
                    * len(mutants)
                    * int(positions.sum())
                )
    if not np.isclose(result.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"hierarchy weights sum to {result.sum()}, expected one")
    return result
