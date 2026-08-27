#!/usr/bin/env python3
"""Frozen linear cross-construct route probe for post-V14 branch 5.

This module contains no runtime authority and reads no targets by itself.  The
candidate and its matched null differ only in whether the seven non-focal WT
constructs are read at the correct coordinate or at the fixed shift-17
coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v14 import V14PointModel


CONSTRUCTS_PER_PUZZLE = 8
HIDDEN_WIDTH = 256
RAW_SUMMARY_WIDTH = HIDDEN_WIDTH + 4
PROBE_FEATURE_WIDTH = 2 * RAW_SUMMARY_WIDTH
ALIGNED_SHIFT = 0
MATCHED_NULL_SHIFT = 17
RIDGE_ALPHA = 1.0


def zero_preserving_v14_content_hidden(
    model: V14PointModel,
    context: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Remove the V14 coordinate/region-only path from non-focal hidden state."""

    if model.training:
        raise ValueError("branch5 content contrast requires frozen V14 eval mode")
    if len(context) != 6:
        raise ValueError("branch5 V14 context must contain six aligned tensors")
    sequence, reactivity, precision, observed, position, region = context
    reference = (
        torch.zeros_like(sequence),
        torch.zeros_like(reactivity),
        torch.zeros_like(precision),
        torch.zeros_like(observed),
        position,
        region,
    )
    with torch.no_grad():
        actual = model.encode(context, None)
        coordinate_only = model.encode(reference, None)
    content = actual - coordinate_only
    if content.shape != actual.shape or not torch.isfinite(content).all():
        raise RuntimeError("branch5 V14 content contrast is invalid")
    return content


def nonfocal_linear_summary(
    hidden: torch.Tensor,
    reactivity: torch.Tensor,
    observed: torch.Tensor,
    *,
    focal_index: int,
    shift: int,
) -> torch.Tensor:
    """Return the 260D summary from zero-preserving V14 content hidden state."""

    if (
        hidden.ndim != 3
        or hidden.shape[0] != CONSTRUCTS_PER_PUZZLE
        or hidden.shape[2] != HIDDEN_WIDTH
    ):
        raise ValueError("branch5 hidden tensor must have shape (8,L,256)")
    length = int(hidden.shape[1])
    if reactivity.shape != (CONSTRUCTS_PER_PUZZLE, length) or observed.shape != (
        CONSTRUCTS_PER_PUZZLE,
        length,
    ):
        raise ValueError("branch5 WT value or observation tensor is misaligned")
    if not 0 <= int(focal_index) < CONSTRUCTS_PER_PUZZLE:
        raise ValueError("branch5 focal index is outside the eight-construct set")
    if int(shift) not in (ALIGNED_SHIFT, MATCHED_NULL_SHIFT):
        raise ValueError("branch5 shift must be exactly 0 or 17")

    keep = [index for index in range(CONSTRUCTS_PER_PUZZLE) if index != focal_index]
    nonfocal_hidden = torch.roll(hidden[keep], shifts=int(shift), dims=1)
    nonfocal_reactivity = torch.roll(reactivity[keep], shifts=int(shift), dims=1)
    nonfocal_observed = torch.roll(
        observed[keep].to(dtype=torch.bool), shifts=int(shift), dims=1
    )
    hidden_mean = nonfocal_hidden.mean(dim=0)
    observed_float = nonfocal_observed.to(dtype=hidden.dtype)
    safe_values = torch.where(
        nonfocal_observed,
        nonfocal_reactivity.to(dtype=hidden.dtype),
        torch.zeros((), dtype=hidden.dtype, device=hidden.device),
    )
    support_count = observed_float.sum(dim=0)
    support_fraction = support_count / float(CONSTRUCTS_PER_PUZZLE - 1)
    pooled_safe_value = safe_values.mean(dim=0)
    observed_mean = safe_values.sum(dim=0) / support_count.clamp_min(1.0)
    centered = torch.where(
        nonfocal_observed,
        safe_values - observed_mean.unsqueeze(0),
        torch.zeros((), dtype=hidden.dtype, device=hidden.device),
    )
    observed_std = torch.sqrt(
        centered.square().sum(dim=0) / support_count.clamp_min(1.0)
    )
    no_support = support_count == 0
    observed_mean = observed_mean.masked_fill(no_support, 0.0)
    observed_std = observed_std.masked_fill(no_support, 0.0)
    result = torch.cat(
        [
            hidden_mean,
            pooled_safe_value[:, None],
            observed_mean[:, None],
            observed_std[:, None],
            support_fraction[:, None],
        ],
        dim=-1,
    )
    if result.shape != (length, RAW_SUMMARY_WIDTH) or not torch.isfinite(result).all():
        raise RuntimeError("branch5 non-focal summary is invalid")
    return result


def source_receiver_features(
    summary: torch.Tensor, edit_index: torch.Tensor
) -> torch.Tensor:
    """Concatenate source and receiver summaries for every mutant-position row."""

    if summary.ndim != 2 or summary.shape[1] != RAW_SUMMARY_WIDTH:
        raise ValueError("branch5 summary width changed")
    if edit_index.ndim != 1 or edit_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("branch5 edit index must be a one-dimensional integer tensor")
    length = int(summary.shape[0])
    if bool(((edit_index < 0) | (edit_index >= length)).any()):
        raise ValueError("branch5 edit index lies outside the construct")
    source = summary[edit_index][:, None, :].expand(-1, length, -1)
    receiver = summary[None, :, :].expand(len(edit_index), -1, -1)
    result = torch.cat([source, receiver], dim=-1)
    if result.shape != (len(edit_index), length, PROBE_FEATURE_WIDTH):
        raise RuntimeError("branch5 source-receiver feature shape changed")
    return result


def puzzle_method_balanced_weights(
    puzzle_cell_masks: Iterable[Iterable[np.ndarray]],
) -> list[list[np.ndarray]]:
    """Exact puzzle→cell→mutant→position weights, scaled to mean one."""

    puzzles = [
        [np.asarray(mask, dtype=bool) for mask in cells] for cells in puzzle_cell_masks
    ]
    if not puzzles:
        raise ValueError("branch5 weighting requires outer-train puzzles")
    output: list[list[np.ndarray]] = []
    total_qualified_rows = 0
    for cells in puzzles:
        valid_cells = [mask for mask in cells if mask.ndim == 2 and bool(mask.any())]
        if not valid_cells or len(valid_cells) != len(cells):
            raise ValueError("branch5 weighting received an empty or malformed cell")
        puzzle_output = []
        for mask in valid_cells:
            valid_mutants = mask.any(axis=1)
            n_mutants = int(valid_mutants.sum())
            if n_mutants == 0:
                raise ValueError("branch5 cell has no qualified mutant")
            weights = np.zeros(mask.shape, dtype=np.float64)
            for mutant in np.flatnonzero(valid_mutants):
                n_positions = int(mask[mutant].sum())
                weights[mutant, mask[mutant]] = 1.0 / (
                    len(puzzles) * len(valid_cells) * n_mutants * n_positions
                )
                total_qualified_rows += n_positions
            puzzle_output.append(weights)
        output.append(puzzle_output)
    if total_qualified_rows <= 0:
        raise ValueError("branch5 weighting found no qualified rows")
    # The hierarchy above sums to one.  Scaling to N gives nonzero rows mean 1,
    # so alpha=1 has a fold-invariant meaning.
    return [[weights * total_qualified_rows for weights in cells] for cells in output]


@dataclass
class ProbeRidgeStats:
    sum_weight: float
    sum_x: np.ndarray
    sum_x2: np.ndarray
    xtx: np.ndarray
    sum_y: float
    xty: np.ndarray

    @classmethod
    def zeros(cls, width: int = PROBE_FEATURE_WIDTH) -> "ProbeRidgeStats":
        return cls(
            sum_weight=0.0,
            sum_x=np.zeros(width, dtype=np.float64),
            sum_x2=np.zeros(width, dtype=np.float64),
            xtx=np.zeros((width, width), dtype=np.float64),
            sum_y=0.0,
            xty=np.zeros(width, dtype=np.float64),
        )

    def add_rows(
        self, features: np.ndarray, residual: np.ndarray, weight: np.ndarray
    ) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(residual, dtype=np.float64)
        w = np.asarray(weight, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.sum_x):
            raise ValueError("branch5 ridge feature matrix has invalid shape")
        if y.shape != (len(x),) or w.shape != (len(x),):
            raise ValueError("branch5 ridge residual or weight is misaligned")
        if (
            not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isfinite(w).all()
            or np.any(w <= 0)
        ):
            raise ValueError("branch5 ridge rows must be finite with positive weight")
        wx = w[:, None] * x
        self.sum_weight += float(w.sum())
        self.sum_x += wx.sum(axis=0)
        self.sum_x2 += (wx * x).sum(axis=0)
        self.xtx += x.T @ wx
        self.sum_y += float(np.dot(w, y))
        self.xty += x.T @ (w * y)


def fit_probe_ridge(
    stats: ProbeRidgeStats, *, alpha: float = RIDGE_ALPHA
) -> dict[str, np.ndarray | float]:
    if float(alpha) != RIDGE_ALPHA or stats.sum_weight <= 0:
        raise ValueError("branch5 ridge is frozen to alpha=1 with positive weight")
    mean_x = stats.sum_x / stats.sum_weight
    variance = np.maximum(stats.sum_x2 / stats.sum_weight - mean_x**2, 0.0)
    scale_x = np.sqrt(variance)
    scale_x = np.where(scale_x < 1e-8, 1.0, scale_x)
    mean_y = stats.sum_y / stats.sum_weight
    centered_xtx = stats.xtx - stats.sum_weight * np.outer(mean_x, mean_x)
    centered_xty = stats.xty - stats.sum_x * mean_y
    ztz = centered_xtx / np.outer(scale_x, scale_x)
    zty = centered_xty / scale_x
    coefficient = np.linalg.solve(ztz + RIDGE_ALPHA * np.eye(len(mean_x)), zty)
    result = {
        "mean_x": mean_x,
        "scale_x": scale_x,
        "mean_y": float(mean_y),
        "coefficient": coefficient,
        "alpha": RIDGE_ALPHA,
    }
    if not all(
        (
            np.isfinite(value).all()
            if isinstance(value, np.ndarray)
            else np.isfinite(value)
        )
        for value in result.values()
    ):
        raise RuntimeError("branch5 ridge fit is nonfinite")
    return result


def predict_probe_ridge(
    model: dict[str, np.ndarray | float], features: np.ndarray
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != PROBE_FEATURE_WIDTH:
        raise ValueError("branch5 ridge prediction feature width changed")
    standardized = (x - np.asarray(model["mean_x"])) / np.asarray(model["scale_x"])
    prediction = float(model["mean_y"]) + standardized @ np.asarray(
        model["coefficient"]
    )
    if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
        raise RuntimeError("branch5 ridge prediction is invalid")
    return prediction
