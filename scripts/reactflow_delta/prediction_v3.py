#!/usr/bin/env python3
"""prediction_v3: schema + validator for per-component-position prediction rows.

Contract 11.3:
  - every registered mutant/position has a row or explicit failure status
  - component_key = biological_scoring_key + model_id + seed_or_component_id (unique in-model)
  - biological_scoring_key = dataset+puzzle+method+construct+mutation+position+outer_fold (no model/seed)
  - candidate/baseline biological scoring key sets must be exactly equal
  - distribution family/location/positive scale; Student-t requires df; mixture_weight
Outcome-blind: validator is structural only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionRow:
    biological_scoring_key: str
    model_id: str
    seed_or_component_id: Any
    outer_fold: int
    distribution_family: str  # gaussian | student_t
    location: float
    scale: float  # positive
    df: float | None = None
    mixture_weight: float = 1.0
    mutant_latent_mean: float | None = None
    delta_mean: float | None = None
    target_available: bool = True
    model_coverage: bool = True
    failure_reason: str | None = None

    @property
    def component_key(self) -> str:
        return f"{self.biological_scoring_key}|{self.model_id}|{self.seed_or_component_id}"


def validate_rows(rows: list[PredictionRow], *, expected_keys: set[str] | None = None,
                  expected_seeds: list | None = None) -> dict[str, Any]:
    problems: list[str] = []
    component_keys: set[str] = set()
    bio_keys: set[str] = set()
    for r in rows:
        if r.component_key in component_keys:
            problems.append(f"duplicate component_key: {r.component_key}")
        component_keys.add(r.component_key)
        bio_keys.add(r.biological_scoring_key)
        if r.scale <= 0:
            problems.append(f"non-positive scale for {r.component_key}")
        if r.distribution_family == "student_t" and (r.df is None or r.df <= 1):
            problems.append(f"student_t requires df>1 for {r.component_key}")
        if r.distribution_family not in ("gaussian", "student_t"):
            problems.append(f"unknown family {r.distribution_family} for {r.component_key}")
        if not (0 < r.mixture_weight <= 1):
            problems.append(f"bad mixture_weight {r.mixture_weight} for {r.component_key}")
        if not r.model_coverage and r.failure_reason is None:
            problems.append(f"no-coverage row missing failure_reason: {r.component_key}")
    if expected_keys is not None and bio_keys != expected_keys:
        problems.append(f"biological scoring key set mismatch: {len(bio_keys)} vs expected {len(expected_keys)}")
    if expected_seeds is not None:
        present = {r.seed_or_component_id for r in rows}
        if set(expected_seeds) != present:
            problems.append(f"seed set mismatch: {sorted(present)} vs {expected_seeds}")
    return {
        "schema_version": "reactflow_delta.prediction_v3.validation.v1",
        "all_pass": len(problems) == 0,
        "n_rows": len(rows),
        "n_component_keys": len(component_keys),
        "n_biological_scoring_keys": len(bio_keys),
        "problems": problems,
    }
