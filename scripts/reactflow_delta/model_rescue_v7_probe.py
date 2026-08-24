#!/usr/bin/env python3
"""Fixed feature utilities for the corrected V7M2 dependency eligibility probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache as UnconstrainedFeatureCache,
    WeightedRidgeStats,
    cell_position_weights,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    CANDIDATE_PROBE_FEATURE_NAMES as FEATURE41_NAMES,
    ConstrainedFeatureCache,
    prediction_features as v6_prediction_features,
)
from scripts.reactflow_delta.model_rescue_v7_schema import (
    CACHE_SCHEMA,
    FEATURE_NAMES as DEPENDENCY_FEATURE_NAMES,
)


CANDIDATE_PROBE_FEATURE_NAMES = FEATURE41_NAMES + DEPENDENCY_FEATURE_NAMES


def biological_cache_key(
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


def _strings(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ]


class DependencyFeatureCache:
    """Qualified dependency6 cache indexed by corrected biological identity."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("V7M2 requires the qualified V7 dependency cache schema")
        names = tuple(json.loads(self.handle.attrs.get("feature_names", "[]")))
        if names != DEPENDENCY_FEATURE_NAMES:
            raise ValueError("V7M2 dependency feature basis changed")

        puzzles = _strings(self.handle["puzzle"])
        methods = _strings(self.handle["method"])
        refs = _strings(self.handle["ref"])
        alts = _strings(self.handle["alt"])
        design_positions = np.asarray(self.handle["design_pos"][:], dtype=np.int64)
        full_positions = np.asarray(self.handle["full_pos"][:], dtype=np.int64)
        lengths = {
            len(puzzles),
            len(methods),
            len(refs),
            len(alts),
            len(design_positions),
            len(full_positions),
        }
        if len(lengths) != 1:
            raise ValueError("V7 dependency cache metadata lengths differ")
        self.index: dict[tuple[str, str, int, str, str], int] = {}
        for index, values in enumerate(
            zip(puzzles, methods, design_positions, refs, alts)
        ):
            key = biological_cache_key(*values)
            if key in self.index:
                raise ValueError(f"duplicate V7 dependency biological key {key}")
            self.index[key] = index
        self.full_positions = full_positions
        self.features = self.handle["features"]
        if self.features.ndim != 3 or self.features.shape[0] != len(self.index):
            raise ValueError("V7 dependency feature tensor shape is invalid")
        if self.features.shape[2] != len(DEPENDENCY_FEATURE_NAMES):
            raise ValueError("V7 dependency feature width changed")

    def close(self) -> None:
        self.handle.close()

    def get(self, record: Any) -> np.ndarray:
        key = biological_cache_key(
            record.puzzle,
            record.method,
            record.design_pos,
            record.ref,
            record.alt,
        )
        if key not in self.index:
            raise KeyError(f"V7 dependency cache is missing biological key {key}")
        index = self.index[key]
        source = int(self.full_positions[index])
        if source != int(record.full_pos):
            raise ValueError(
                f"V7 dependency source coordinate differs for {key}: "
                f"cache={source} record={record.full_pos}"
            )
        value = np.asarray(self.features[index], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(DEPENDENCY_FEATURE_NAMES):
            raise RuntimeError(f"V7 dependency feature shape is invalid for {key}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"V7 dependency features are non-finite for {key}")
        if not np.array_equal(
            value[source], np.zeros(len(DEPENDENCY_FEATURE_NAMES), dtype=np.float32)
        ):
            raise RuntimeError(f"V7 dependency self edge is not zero for {key}")
        return value


def validate_cache_alignment(
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
    dependency: DependencyFeatureCache,
) -> dict[str, int | bool]:
    baseline_keys = set(unconstrained.index)
    if baseline_keys != set(constrained.index) or baseline_keys != set(dependency.index):
        raise ValueError(
            "V5, V6, and V7 biological cache universes are not identical: "
            f"v5={len(baseline_keys)} v6={len(constrained.index)} "
            f"v7={len(dependency.index)}"
        )
    receiver_shapes = {
        int(unconstrained.features.shape[1]),
        int(constrained.features.shape[1]),
        int(dependency.features.shape[1]),
    }
    if len(receiver_shapes) != 1:
        raise ValueError("V5, V6, and V7 receiver lengths differ")
    return {
        "biological_key_universe_equal": True,
        "registered_mutants": len(baseline_keys),
        "receiver_length": receiver_shapes.pop(),
        "baseline_width": len(FEATURE41_NAMES),
        "candidate_width": len(CANDIDATE_PROBE_FEATURE_NAMES),
    }


def _target_profile(univ: Any, record: Any) -> np.ndarray | None:
    target, _error = univ.mutant_full_profile(
        record.wt_id, record.design_pos, record.ref, record.alt
    )
    return target


def accumulate_candidate_train_stats(
    univ: Any,
    records: list[Any],
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
    dependency: DependencyFeatureCache,
) -> tuple[WeightedRidgeStats, dict[str, int]]:
    """Accumulate only the frozen 47-feature candidate sufficient statistics."""

    stats = WeightedRidgeStats.zeros(len(CANDIDATE_PROBE_FEATURE_NAMES), 2)
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
            qualified = (
                construct.wt_observed.astype(bool) & np.isfinite(target)
                if target is not None
                else np.zeros(len(construct.sequence), dtype=bool)
            )
            receiver = np.flatnonzero(qualified)
            signed = (
                target[receiver].astype(np.float64)
                - construct.wt_reactivity[receiver].astype(np.float64)
                if len(receiver)
                else np.zeros(0, dtype=np.float64)
            )
            prepared.append((record, receiver, signed))
            counts.append(len(receiver))
        weights = cell_position_weights(np.asarray(counts, dtype=np.int64))
        for (record, receiver, signed), weight in zip(prepared, weights):
            if not len(receiver):
                continue
            _feature30, feature41 = v6_prediction_features(
                construct,
                record,
                receiver,
                unconstrained,
                constrained,
            )
            dependency6 = dependency.get(record)[receiver]
            feature47 = np.concatenate([feature41, dependency6], axis=1)
            if feature47.shape != (len(receiver), len(CANDIDATE_PROBE_FEATURE_NAMES)):
                raise RuntimeError("V7 candidate feature width changed")
            target = np.column_stack([signed, np.abs(signed)])
            stats.add_rows(feature47, target, weight)
            n_positions += len(receiver)
            n_valid_mutants += 1
    return stats, {
        "n_train_cells": len(by_cell),
        "n_train_valid_mutants": n_valid_mutants,
        "n_train_qualified_positions": n_positions,
    }


def prediction_features(
    construct: Any,
    record: Any,
    receiver: np.ndarray,
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
    dependency: DependencyFeatureCache,
) -> tuple[np.ndarray, np.ndarray]:
    _feature30, feature41 = v6_prediction_features(
        construct,
        record,
        receiver,
        unconstrained,
        constrained,
    )
    dependency6 = dependency.get(record)[receiver]
    feature47 = np.concatenate([feature41, dependency6], axis=1)
    if feature41.shape[1] != len(FEATURE41_NAMES):
        raise RuntimeError("V7 baseline feature width changed")
    if feature47.shape[1] != len(CANDIDATE_PROBE_FEATURE_NAMES):
        raise RuntimeError("V7 candidate feature width changed")
    return feature41, feature47
