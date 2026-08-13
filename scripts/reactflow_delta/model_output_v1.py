#!/usr/bin/env python3
"""model_output_v1 (contract 11.7, 7.4): coordinate-aligned full-construct output.

Every registered mutant has a coordinate-aligned output of length = canonical full
construct. Fields separated: mutant latent mean, delta mean, model_scale/interval,
coverage, region map, and full predictive distribution components. model_scale is
distinct from measurement error. Five-seed mixture requires family/location/scale/
df/weight per component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DistributionComponent:
    family: str  # gaussian | student_t
    location: np.ndarray  # length L
    scale: np.ndarray     # length L, positive
    df: float | None = None  # mandatory for student_t
    weight: float = 1.0


@dataclass
class ModelOutput:
    construct_id: str
    mutation_key: str  # pos_ref>alt
    outer_fold: int
    latent_mean: np.ndarray    # length L
    delta_mean: np.ndarray     # length L (mutant - WT anchor)
    model_scale: np.ndarray    # length L, positive (predictive, NOT measurement error)
    measurement_error: np.ndarray | None = None  # length L, train-only noise floor, separate field
    region_map: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    coverage: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # model coverage
    components: list[DistributionComponent] = field(default_factory=list)


def validate_model_output(out: ModelOutput, canonical_len: int,
                          region_map: np.ndarray) -> dict[str, Any]:
    problems: list[str] = []
    for name, arr in [("latent_mean", out.latent_mean), ("delta_mean", out.delta_mean),
                      ("model_scale", out.model_scale)]:
        if arr.shape[0] != canonical_len:
            problems.append(f"{name} length {arr.shape[0]} != canonical {canonical_len}")
    if out.region_map.shape[0] != canonical_len:
        problems.append(f"region_map length {out.region_map.shape[0]} != canonical {canonical_len}")
    elif len(out.region_map) > 0 and not np.array_equal(out.region_map, region_map):
        problems.append("region_map not equal to canonical construct region map")
    if np.any(out.model_scale <= 0):
        problems.append("model_scale must be positive (no zero/negative predictive scale)")
    if out.coverage.shape[0] != canonical_len:
        problems.append(f"coverage length {out.coverage.shape[0]} != canonical {canonical_len}")
    if out.components:
        weights_sum = sum(c.weight for c in out.components)
        if abs(weights_sum - 1.0) > 1e-6:
            problems.append(f"mixture weights sum {weights_sum} != 1")
        for c in out.components:
            if c.family == "student_t" and (c.df is None or c.df <= 2):
                problems.append("student_t requires df>2 for finite variance")
            if c.scale.shape[0] != canonical_len:
                problems.append(f"component scale length {c.scale.shape[0]} != canonical {canonical_len}")
    return {
        "schema_version": "reactflow_delta.model_output_v1.validation.v1",
        "all_pass": len(problems) == 0,
        "problems": problems,
    }
