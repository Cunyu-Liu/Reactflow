#!/usr/bin/env python3
"""Fixed incremental-ridge utilities for the Model Rescue v6 eligibility probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from scripts.reactflow_delta.model_rescue_v5_probe import (
    BASELINE_FEATURE_NAMES as DIRECT_FEATURE_NAMES,
    EnsembleFeatureCache as UnconstrainedFeatureCache,
    WeightedRidgeStats,
    baseline_features as direct_features,
    cell_position_weights,
)
from scripts.reactflow_delta.model_rescue_v5_schema import (
    FEATURE_NAMES as UNCONSTRAINED_FEATURE_NAMES,
)
from scripts.reactflow_delta.model_rescue_v6_schema import (
    CACHE_SCHEMA,
    FEATURE_NAMES as CONSTRAINED_CACHE_FEATURE_NAMES,
    PROBE_FEATURE_INDICES,
    PROBE_FEATURE_NAMES as CONSTRAINED_PROBE_FEATURE_NAMES,
)


BASELINE_PROBE_FEATURE_NAMES = DIRECT_FEATURE_NAMES + UNCONSTRAINED_FEATURE_NAMES
CANDIDATE_PROBE_FEATURE_NAMES = (
    BASELINE_PROBE_FEATURE_NAMES + CONSTRAINED_PROBE_FEATURE_NAMES
)


def _cache_key(
    puzzle: Any,
    method: Any,
    design_pos: Any,
    ref: Any,
    alt: Any,
) -> tuple[str, str, int, str, str]:
    return (
        str(puzzle),
        str(method),
        int(design_pos),
        str(ref).replace("T", "U"),
        str(alt).replace("T", "U"),
    )


class ConstrainedFeatureCache:
    """Qualified 12-channel cache exposed through the frozen 11-channel basis."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("v6 probe requires the qualified v6 cache schema")
        names = tuple(json.loads(self.handle.attrs["feature_names"]))
        if names != CONSTRAINED_CACHE_FEATURE_NAMES:
            raise ValueError("v6 constrained cache feature universe changed")

        def strings(name: str) -> list[str]:
            return [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in self.handle[name][:]
            ]

        puzzles = strings("puzzle")
        methods = strings("method")
        refs = strings("ref")
        alts = strings("alt")
        positions = self.handle["design_pos"][:]
        lengths = {len(puzzles), len(methods), len(refs), len(alts), len(positions)}
        if len(lengths) != 1:
            raise ValueError("v6 cache metadata columns have inconsistent lengths")
        self.index: dict[tuple[str, str, int, str, str], int] = {}
        for index, values in enumerate(zip(puzzles, methods, positions, refs, alts)):
            key = _cache_key(*values)
            if key in self.index:
                raise ValueError(f"duplicate v6 biological cache row {key}")
            self.index[key] = index
        self.features = np.asarray(self.handle["features"][:], dtype=np.float32)
        if self.features.ndim != 3 or self.features.shape[0] != len(self.index):
            raise ValueError("v6 cache feature tensor and metadata row counts differ")
        if self.features.shape[2] != len(CONSTRAINED_CACHE_FEATURE_NAMES):
            raise ValueError("v6 cache feature width differs from the frozen schema")

    def close(self) -> None:
        self.handle.close()

    def get(self, record: Any) -> np.ndarray:
        key = _cache_key(
            record.puzzle,
            record.method,
            record.design_pos,
            record.ref,
            record.alt,
        )
        if key not in self.index:
            raise KeyError(f"v6 cache is missing biological mutant key {key}")
        full = np.asarray(self.features[self.index[key]], dtype=np.float32)
        value = full[:, PROBE_FEATURE_INDICES]
        if value.ndim != 2 or value.shape[1] != len(CONSTRAINED_PROBE_FEATURE_NAMES):
            raise RuntimeError(f"invalid constrained probe feature shape for {key}")
        return value


def validate_cache_alignment(
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
) -> dict[str, int | bool]:
    same_keys = set(unconstrained.index) == set(constrained.index)
    if not same_keys:
        missing = set(unconstrained.index) - set(constrained.index)
        unexpected = set(constrained.index) - set(unconstrained.index)
        raise ValueError(
            "v5/v6 cache biological universes differ: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if unconstrained.features.ndim != 3 or constrained.features.ndim != 3:
        raise ValueError("v5/v6 cache feature tensors must be three-dimensional")
    if unconstrained.features.shape[:2] != constrained.features.shape[:2]:
        raise ValueError("v5/v6 cache mutant or receiver dimensions differ")
    return {
        "biological_key_universe_equal": True,
        "registered_mutants": len(unconstrained.index),
        "receiver_length": int(unconstrained.features.shape[1]),
        "unconstrained_width": int(unconstrained.features.shape[2]),
        "constrained_cache_width": int(constrained.features.shape[2]),
        "constrained_probe_width": len(CONSTRAINED_PROBE_FEATURE_NAMES),
    }


def _target_profile(univ: Any, record: Any) -> np.ndarray | None:
    target, _error = univ.mutant_full_profile(
        record.wt_id, record.design_pos, record.ref, record.alt
    )
    return target


def accumulate_train_stats(
    univ: Any,
    records: list[Any],
    unconstrained_cache: UnconstrainedFeatureCache,
    constrained_cache: ConstrainedFeatureCache,
) -> tuple[WeightedRidgeStats, WeightedRidgeStats, dict[str, int]]:
    baseline_stats = WeightedRidgeStats.zeros(len(BASELINE_PROBE_FEATURE_NAMES), 2)
    candidate_stats = WeightedRidgeStats.zeros(len(CANDIDATE_PROBE_FEATURE_NAMES), 2)
    by_cell: dict[str, list[Any]] = {}
    for record in records:
        by_cell.setdefault(record.construct_id, []).append(record)
    n_positions = 0
    n_valid_mutants = 0
    for construct_id, cell_records in sorted(by_cell.items()):
        construct = univ.get_construct(construct_id)
        prepared: list[tuple[Any, np.ndarray, np.ndarray]] = []
        counts: list[int] = []
        for record in cell_records:
            target = _target_profile(univ, record)
            if target is None:
                qualified = np.zeros(len(construct.sequence), dtype=bool)
            else:
                qualified = construct.wt_observed.astype(bool) & np.isfinite(target)
            receiver = np.flatnonzero(qualified)
            signed = (
                target[receiver].astype(np.float64)
                - construct.wt_reactivity[receiver].astype(np.float64)
                if len(receiver)
                else np.zeros(0, dtype=np.float64)
            )
            prepared.append((record, receiver, signed))
            counts.append(len(receiver))
        weights = cell_position_weights(np.asarray(counts))
        for (record, receiver, signed), weight in zip(prepared, weights):
            if not len(receiver):
                continue
            direct = direct_features(construct, record, receiver)
            unconstrained = unconstrained_cache.get(record)[receiver]
            constrained = constrained_cache.get(record)[receiver]
            baseline = np.concatenate([direct, unconstrained], axis=1)
            candidate = np.concatenate([baseline, constrained], axis=1)
            target = np.column_stack([signed, np.abs(signed)])
            baseline_stats.add_rows(baseline, target, weight)
            candidate_stats.add_rows(candidate, target, weight)
            n_positions += len(receiver)
            n_valid_mutants += 1
    return baseline_stats, candidate_stats, {
        "n_train_cells": len(by_cell),
        "n_train_valid_mutants": n_valid_mutants,
        "n_train_qualified_positions": n_positions,
    }


def prediction_features(
    construct: Any,
    record: Any,
    receiver: np.ndarray,
    unconstrained_cache: UnconstrainedFeatureCache,
    constrained_cache: ConstrainedFeatureCache,
) -> tuple[np.ndarray, np.ndarray]:
    direct = direct_features(construct, record, receiver)
    unconstrained = unconstrained_cache.get(record)[receiver]
    constrained = constrained_cache.get(record)[receiver]
    baseline = np.concatenate([direct, unconstrained], axis=1)
    candidate = np.concatenate([baseline, constrained], axis=1)
    if baseline.shape[1] != len(BASELINE_PROBE_FEATURE_NAMES):
        raise RuntimeError("v6 baseline probe feature width changed")
    if candidate.shape[1] != len(CANDIDATE_PROBE_FEATURE_NAMES):
        raise RuntimeError("v6 candidate probe feature width changed")
    return baseline, candidate
