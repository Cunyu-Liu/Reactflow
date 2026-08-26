#!/usr/bin/env python3
"""Core utilities for the frozen post-V13 route diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache as UnconstrainedFeatureCache,
    WeightedRidgeStats,
    baseline_features as direct_features,
    cell_position_weights,
)
from scripts.reactflow_delta.model_rescue_v5_schema import (
    FEATURE_NAMES as UNCONSTRAINED_FEATURE_NAMES,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    CANDIDATE_PROBE_FEATURE_NAMES as FEATURE41_NAMES,
    ConstrainedFeatureCache,
)


ERROR_FLOOR = 0.05


def normalized_reliability_weights(
    mutant_error: np.ndarray,
    wt_error: np.ndarray,
    *,
    error_floor: float = ERROR_FLOOR,
) -> np.ndarray:
    """Return inverse-variance weights with mean one inside one mutant.

    Missing or nonpositive error pairs receive neutral raw weight one.  The
    final normalization preserves the total exposure of the mutant.
    """

    mutant = np.asarray(mutant_error, dtype=np.float64)
    wt = np.asarray(wt_error, dtype=np.float64)
    if mutant.shape != wt.shape or mutant.ndim != 1:
        raise ValueError("reliability errors must be same-shape vectors")
    if error_floor <= 0:
        raise ValueError("reliability error floor must be positive")
    if len(mutant) == 0:
        return np.zeros(0, dtype=np.float64)
    valid = np.isfinite(mutant) & (mutant > 0) & np.isfinite(wt) & (wt > 0)
    raw = np.ones(len(mutant), dtype=np.float64)
    raw[valid] = 1.0 / (mutant[valid] ** 2 + wt[valid] ** 2 + error_floor**2)
    mean = float(raw.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise RuntimeError("reliability normalization is non-finite")
    result = raw / mean
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise RuntimeError("reliability weights must be finite and positive")
    return result


def coherent_signed_magnitude(
    signed_prediction: np.ndarray, absolute_prediction: np.ndarray
) -> np.ndarray:
    signed = np.asarray(signed_prediction, dtype=np.float64)
    magnitude = np.asarray(absolute_prediction, dtype=np.float64)
    if signed.shape != magnitude.shape:
        raise ValueError("signed and magnitude predictions must have the same shape")
    if not np.isfinite(signed).all() or not np.isfinite(magnitude).all():
        raise ValueError("coherent reconstruction requires finite predictions")
    return np.sign(signed) * np.maximum(magnitude, 0.0)


def feature41_matrix(
    construct: Any,
    record: Any,
    receiver: np.ndarray,
    unconstrained_cache: UnconstrainedFeatureCache,
    constrained_cache: ConstrainedFeatureCache,
) -> np.ndarray:
    direct = direct_features(construct, record, receiver)
    unconstrained = unconstrained_cache.get(record)[receiver]
    constrained = constrained_cache.get(record)[receiver]
    matrix = np.concatenate([direct, unconstrained, constrained], axis=1)
    if matrix.shape != (len(receiver), len(FEATURE41_NAMES)):
        raise RuntimeError("post-V13 feature41 width changed")
    if not np.isfinite(matrix).all():
        raise RuntimeError("post-V13 feature41 contains non-finite values")
    return matrix


def accumulate_train_stats(
    univ: Any,
    records: list[Any],
    unconstrained_cache: UnconstrainedFeatureCache,
    constrained_cache: ConstrainedFeatureCache,
) -> tuple[WeightedRidgeStats, WeightedRidgeStats, dict[str, int]]:
    """Build matched ordinary and noise-aware outer-train statistics."""

    if not records:
        raise ValueError("post-V13 diagnostic training records cannot be empty")
    puzzles = {str(record.puzzle) for record in records}
    ordinary = WeightedRidgeStats.zeros(len(FEATURE41_NAMES), 2)
    noise_aware = WeightedRidgeStats.zeros(len(FEATURE41_NAMES), 2)
    by_cell: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_cell[str(record.construct_id)].append(record)

    valid_mutants = 0
    qualified_positions = 0
    for construct_id, cell_records in sorted(by_cell.items()):
        construct = univ.get_construct(construct_id)
        prepared: list[
            tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        counts: list[int] = []
        for record in cell_records:
            target, target_error = univ.mutant_full_profile(
                record.wt_id, record.design_pos, record.ref, record.alt
            )
            qualified = (
                np.zeros(len(construct.sequence), dtype=bool)
                if target is None
                else construct.wt_observed.astype(bool) & np.isfinite(target)
            )
            receiver = np.flatnonzero(qualified)
            if len(receiver):
                signed = (
                    target[receiver].astype(np.float64)
                    - construct.wt_reactivity[receiver].astype(np.float64)
                )
                mutant_error = (
                    np.full(len(receiver), np.nan, dtype=np.float64)
                    if target_error is None
                    else np.asarray(target_error[receiver], dtype=np.float64)
                )
                wt_error = np.asarray(construct.wt_error[receiver], dtype=np.float64)
            else:
                signed = np.zeros(0, dtype=np.float64)
                mutant_error = np.zeros(0, dtype=np.float64)
                wt_error = np.zeros(0, dtype=np.float64)
            prepared.append((record, receiver, signed, mutant_error, wt_error))
            counts.append(len(receiver))

        base_weights = cell_position_weights(np.asarray(counts, dtype=np.int64))
        for prepared_row, base_weight in zip(prepared, base_weights):
            record, receiver, signed, mutant_error, wt_error = prepared_row
            if not len(receiver):
                continue
            matrix = feature41_matrix(
                construct,
                record,
                receiver,
                unconstrained_cache,
                constrained_cache,
            )
            target_matrix = np.column_stack([signed, np.abs(signed)])
            reliability = normalized_reliability_weights(mutant_error, wt_error)
            ordinary.add_rows(matrix, target_matrix, base_weight)
            noise_aware.add_rows(matrix, target_matrix, base_weight * reliability)
            valid_mutants += 1
            qualified_positions += len(receiver)

    if not np.isclose(ordinary.sum_weight, noise_aware.sum_weight, atol=1e-12):
        raise RuntimeError("noise-aware weighting changed total puzzle exposure")
    return ordinary, noise_aware, {
        "n_train_puzzles": len(puzzles),
        "n_train_cells": len(by_cell),
        "n_train_valid_mutants": valid_mutants,
        "n_train_qualified_positions": qualified_positions,
    }
