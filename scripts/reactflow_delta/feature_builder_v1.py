#!/usr/bin/env python3
"""feature_builder_v1: outcome-blind feature builder (contract 4.3, 18.1).

Builds predictor features from ONLY legal prospective inputs:
  WT sequence, WT observed reactivity, wt_reported_error_input, WT observed mask,
  construct coordinate, region, assay condition, exact directional mutation.

Held mutant response/error/mask are NEVER used to build features, cohort, scale,
normalizer, hyperparameters, or whether a prediction row is produced. The
held-response invariance fixture asserts that permuting held mutant outcomes does
not change the features/predictions for any held mutant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LEGAL_TOKENS = {"wt_reactivity", "wt_error", "wt_mask", "region", "position", "ref", "alt"}


@dataclass
class MutationInput:
    puzzle: str
    method: str
    construct_id: str
    pos: int
    ref: str
    alt: str


@dataclass
class FeatureVector:
    wt_reactivity: np.ndarray
    wt_error: np.ndarray
    wt_mask: np.ndarray
    region_onehot: np.ndarray
    pos_scalar: float
    ref_alt_onehot: np.ndarray


def onehot_base(b: str) -> np.ndarray:
    idx = {"A": 0, "C": 1, "G": 2, "U": 3}.get(b.upper())
    if idx is None:
        raise ValueError(f"invalid base {b}")
    v = np.zeros(4)
    v[idx] = 1.0
    return v


def build_features(mut: MutationInput, wt_reactivity: np.ndarray, wt_error: np.ndarray,
                   wt_mask: np.ndarray, region_map: np.ndarray) -> FeatureVector:
    """Build features strictly from WT profile + mutation identity. Outcome-blind."""
    pos = mut.pos
    L = len(wt_reactivity)
    region_onehot = np.zeros(L)
    region_onehot[region_map == "design_region"] = 1.0
    # ref=alt => mutation-induced mean must be zero downstream; we still carry identity
    return FeatureVector(
        wt_reactivity=wt_reactivity.copy(),
        wt_error=wt_error.copy(),
        wt_mask=wt_mask.copy(),
        region_onehot=region_onehot,
        pos_scalar=float(pos),
        ref_alt_onehot=np.concatenate([onehot_base(mut.ref), onehot_base(mut.alt)]),
    )


def _eq(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype == bool or b.dtype == bool or a.dtype == object or b.dtype == object:
        return np.array_equal(a, b)
    return bool(np.allclose(a, b, equal_nan=True))


def held_response_invariance(features: FeatureVector, features_permuted: FeatureVector) -> bool:
    """Features must be unchanged when held mutant outcomes are permuted (input-blind)."""
    arrays = [
        ("wt_reactivity", features.wt_reactivity, features_permuted.wt_reactivity),
        ("wt_error", features.wt_error, features_permuted.wt_error),
        ("wt_mask", features.wt_mask, features_permuted.wt_mask),
        ("region_onehot", features.region_onehot, features_permuted.region_onehot),
        ("ref_alt_onehot", features.ref_alt_onehot, features_permuted.ref_alt_onehot),
    ]
    for name, a, b in arrays:
        if not _eq(a, b):
            return False
    if features.pos_scalar != features_permuted.pos_scalar:
        return False
    return True
