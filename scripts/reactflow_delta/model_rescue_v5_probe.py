#!/usr/bin/env python3
"""Fixed weighted-ridge utilities for the v5 ensemble-delta eligibility probe."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import h5py
import numpy as np

from scripts.reactflow_delta.model_rescue_v5_schema import (
    CACHE_SCHEMA,
    FEATURE_NAMES,
)


ALPHABET = "ACGU"
BASELINE_FEATURE_NAMES = (
    "signed_sequence_distance",
    "absolute_sequence_distance",
    "log_absolute_sequence_distance",
    "edit_position",
    "receiver_position",
    "same_site",
    "receiver_design_region",
    "mutation_ref_a",
    "mutation_ref_c",
    "mutation_ref_g",
    "mutation_ref_u",
    "mutation_alt_a",
    "mutation_alt_c",
    "mutation_alt_g",
    "mutation_alt_u",
    "wt_reactivity",
    "wt_error_precision",
    "wt_observed",
)
MUTANT_SUFFIX = re.compile(r"_mm_(\d+)_([ACGTU])_([ACGTU])$")


def canonical_mutant_id(row_id: str) -> str:
    match = MUTANT_SUFFIX.search(str(row_id))
    if match is None:
        raise ValueError(f"invalid mutant id {row_id}")
    position, ref, alt = match.groups()
    prefix = row_id[: match.start()]
    return f"{prefix}_mm_{position}_{ref.replace('T', 'U')}_{alt.replace('T', 'U')}"


def record_mutant_id(record: Any) -> str:
    prefix = record.wt_id[:-3] if str(record.wt_id).endswith("_wt") else str(record.wt_id)
    return canonical_mutant_id(
        f"{prefix}_mm_{int(record.design_pos)}_{record.ref}_{record.alt}"
    )


class EnsembleFeatureCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        if self.handle.attrs.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("v5 probe requires the qualified v5 cache schema")
        names = tuple(json.loads(self.handle.attrs["feature_names"]))
        if names != FEATURE_NAMES:
            raise ValueError("v5 cache feature universe differs from the frozen contract")
        raw_ids = self.handle["row_id"][:]
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in raw_ids]
        self.index: dict[str, int] = {}
        for index, row_id in enumerate(ids):
            canonical = canonical_mutant_id(row_id)
            if canonical in self.index:
                raise ValueError(f"duplicate canonical cache row {canonical}")
            self.index[canonical] = index
        self.features = self.handle["features"]

    def close(self) -> None:
        self.handle.close()

    def get(self, record: Any) -> np.ndarray:
        row_id = record_mutant_id(record)
        if row_id not in self.index:
            raise KeyError(f"v5 cache is missing {row_id}")
        value = np.asarray(self.features[self.index[row_id]], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
            raise RuntimeError(f"invalid cached feature shape for {row_id}")
        return value


def baseline_features(
    construct: Any,
    record: Any,
    receiver_positions: np.ndarray,
) -> np.ndarray:
    receiver = np.asarray(receiver_positions, dtype=np.int64)
    length = len(construct.sequence)
    if receiver.ndim != 1 or np.any(receiver < 0) or np.any(receiver >= length):
        raise ValueError("receiver positions are outside the construct")
    edit = int(record.full_pos)
    if not 0 <= edit < length:
        raise ValueError("corrected edit position is outside the construct")
    distance = receiver - edit
    abs_distance = np.abs(distance)
    ref = str(record.ref).replace("T", "U")
    alt = str(record.alt).replace("T", "U")
    if ref not in ALPHABET or alt not in ALPHABET:
        raise ValueError("mutation identity is outside the RNA alphabet")
    ref_one_hot = np.zeros((len(receiver), 4), dtype=np.float32)
    alt_one_hot = np.zeros((len(receiver), 4), dtype=np.float32)
    ref_one_hot[:, ALPHABET.index(ref)] = 1.0
    alt_one_hot[:, ALPHABET.index(alt)] = 1.0
    observed = construct.wt_observed.astype(bool)
    raw_reactivity = construct.wt_reactivity.astype(np.float32)
    fill = float(np.nanmean(raw_reactivity[observed])) if observed.any() else 0.0
    wt_reactivity = np.where(observed, raw_reactivity, fill)[receiver]
    raw_error = construct.wt_error.astype(np.float32)
    precision = np.where(
        observed & np.isfinite(raw_error) & (raw_error > 0),
        -np.log(np.maximum(raw_error, 1e-6)),
        0.0,
    )[receiver]
    result = np.column_stack(
        [
            distance / max(length - 1, 1),
            abs_distance / max(length - 1, 1),
            np.log1p(abs_distance) / math.log(max(length, 2)),
            np.full(len(receiver), edit / max(length - 1, 1)),
            receiver / max(length - 1, 1),
            (receiver == edit).astype(np.float32),
            (construct.region_map[receiver] == "design_region").astype(np.float32),
            ref_one_hot,
            alt_one_hot,
            wt_reactivity,
            precision,
            observed[receiver].astype(np.float32),
        ]
    ).astype(np.float32)
    if result.shape != (len(receiver), len(BASELINE_FEATURE_NAMES)):
        raise RuntimeError("v5 baseline feature width changed")
    if not np.isfinite(result).all():
        raise RuntimeError("v5 baseline features are non-finite")
    return result


@dataclass
class WeightedRidgeStats:
    sum_weight: float
    sum_x: np.ndarray
    sum_x2: np.ndarray
    xtx: np.ndarray
    sum_y: np.ndarray
    xty: np.ndarray

    @classmethod
    def zeros(cls, n_features: int, n_targets: int = 2) -> "WeightedRidgeStats":
        return cls(
            sum_weight=0.0,
            sum_x=np.zeros(n_features, dtype=np.float64),
            sum_x2=np.zeros(n_features, dtype=np.float64),
            xtx=np.zeros((n_features, n_features), dtype=np.float64),
            sum_y=np.zeros(n_targets, dtype=np.float64),
            xty=np.zeros((n_features, n_targets), dtype=np.float64),
        )

    def add_rows(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        w = np.asarray(weight, dtype=np.float64)
        if x64.ndim != 2 or y64.ndim != 2 or w.shape != (len(x64),):
            raise ValueError("weighted ridge rows have incompatible shapes")
        if len(x64) != len(y64) or x64.shape[1] != len(self.sum_x):
            raise ValueError("weighted ridge feature or target width changed")
        if np.any(w <= 0) or not np.isfinite(x64).all() or not np.isfinite(y64).all():
            raise ValueError("weighted ridge rows must be finite with positive weights")
        wx = w[:, None] * x64
        self.sum_weight += float(w.sum())
        self.sum_x += wx.sum(axis=0)
        self.sum_x2 += (wx * x64).sum(axis=0)
        self.xtx += x64.T @ wx
        self.sum_y += (w[:, None] * y64).sum(axis=0)
        self.xty += x64.T @ (w[:, None] * y64)

    def __iadd__(self, other: "WeightedRidgeStats") -> "WeightedRidgeStats":
        self.sum_weight += other.sum_weight
        self.sum_x += other.sum_x
        self.sum_x2 += other.sum_x2
        self.xtx += other.xtx
        self.sum_y += other.sum_y
        self.xty += other.xty
        return self


def sum_stats(rows: Iterable[WeightedRidgeStats]) -> WeightedRidgeStats:
    values = list(rows)
    if not values:
        raise ValueError("cannot sum an empty weighted-statistics collection")
    result = WeightedRidgeStats.zeros(len(values[0].sum_x), len(values[0].sum_y))
    for row in values:
        result += row
    return result


def fit_weighted_standardized_ridge(
    stats: WeightedRidgeStats, alpha: float = 1.0
) -> dict[str, np.ndarray | float]:
    if stats.sum_weight <= 0 or alpha < 0:
        raise ValueError("weighted ridge requires positive weight and nonnegative alpha")
    mean_x = stats.sum_x / stats.sum_weight
    variance = np.maximum(stats.sum_x2 / stats.sum_weight - mean_x**2, 0.0)
    scale_x = np.sqrt(variance)
    scale_x = np.where(scale_x < 1e-8, 1.0, scale_x)
    mean_y = stats.sum_y / stats.sum_weight
    centered_xtx = stats.xtx - stats.sum_weight * np.outer(mean_x, mean_x)
    centered_xty = stats.xty - np.outer(stats.sum_x, mean_y)
    ztz = centered_xtx / np.outer(scale_x, scale_x)
    zty = centered_xty / scale_x[:, None]
    coefficient = np.linalg.solve(ztz + alpha * np.eye(len(mean_x)), zty)
    return {
        "mean_x": mean_x,
        "scale_x": scale_x,
        "mean_y": mean_y,
        "coefficient": coefficient,
        "alpha": float(alpha),
    }


def predict_weighted_ridge(
    model: dict[str, np.ndarray | float], x: np.ndarray
) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    standardized = (value - model["mean_x"]) / model["scale_x"]
    return np.asarray(model["mean_y"] + standardized @ model["coefficient"])


def cell_position_weights(qualified_counts: np.ndarray) -> list[np.ndarray]:
    counts = np.asarray(qualified_counts, dtype=np.int64)
    valid = counts > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return [np.zeros(0, dtype=np.float64) for _ in counts]
    return [
        np.full(int(count), 1.0 / (n_valid * int(count)), dtype=np.float64)
        if count > 0
        else np.zeros(0, dtype=np.float64)
        for count in counts
    ]
