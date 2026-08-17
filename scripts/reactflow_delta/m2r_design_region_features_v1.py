#!/usr/bin/env python3
"""m2r_design_region_features_v1.py — LEGAL design-region aggregate features.

MOTIVATION (method-level, not data-level):
The rescue_factor target is defined as a DESIGN-REGION RMSD ratio:
  rescue = 1 - RMSD(double,wt) / sqrt(RMSD(sA,wt)^2 + RMSD(sB,wt)^2)
over the design region [sub_start, sub_end).

The current 236-dim feature set uses only LOCAL windows around the pair sites
(i,j), but NEVER exposes the design-region aggregates.  The DENOMINATOR
sqrt(RMSD(sA,wt)^2 + RMSD(sB,wt)^2) is ENTIRELY LEGAL (WT + single-mutant
profiles) — the model never sees it.  Adding it is a clean, honest method-level
lever that directly targets the target's structure.

Features added (8 dims, all legal / non-circular):
  * rmsd_sA_wt  : RMSD(singleA, WT) over design region
  * rmsd_sB_wt  : RMSD(singleB, WT) over design region
  * denom_legal : sqrt(rmsd_sA_wt^2 + rmsd_sB_wt^2)  [the rescue denominator]
  * delta_sA_wt_mean : mean(singleA - WT) over design region (signed disruption)
  * delta_sB_wt_mean : mean(singleB - WT) over design region
  * wt_region_mean   : mean(WT reactivity) over design region
  * sA_region_mean   : mean(singleA reactivity) over design region
  * sB_region_mean   : mean(singleB reactivity) over design region

For the CIRCULAR ORACLE (used only in ceiling audit, not in the legal model):
  * rmsd_double_wt : RMSD(double, WT) over design region
  * double_region_mean : mean(double reactivity) over design region

These features are computed from the full-profile arrays of each M2RPair sample,
using the design-region mask (sub_start, sub_end).  They are ADDED to the
existing 236-dim window features, not replacing them.
"""
from __future__ import annotations

import numpy as np

# re-export the noise-floor helpers for consistency
def _design_mask(n: int, sub_start: int, sub_end: int) -> np.ndarray:
    """Boolean mask over design region [sub_start, sub_end) (0-indexed)."""
    m = np.zeros(n, dtype=bool)
    lo = max(sub_start - 1, 0) if sub_start is not None else 0
    hi = sub_end if sub_end is not None else n
    m[lo:hi] = True
    return m


def _rmsd_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """RMSD over masked region, NaN if <3 valid positions."""
    m = np.isfinite(a) & np.isfinite(b) & mask
    if m.sum() < 3:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _region_mean(profile: np.ndarray, mask: np.ndarray) -> float:
    """Mean of profile over masked region, NaN if no valid positions."""
    m = np.isfinite(profile) & mask
    if m.sum() == 0:
        return np.nan
    return float(np.mean(profile[m]))


def _prof(p, n: int) -> np.ndarray:
    """Convert list/array of length n to float64 array, NaN for None/invalid."""
    a = np.full(n, np.nan, dtype=np.float64)
    for k in range(min(len(p), n)):
        try:
            v = float(p[k])
            a[k] = v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            pass
    return a


def build_design_region_features(sample, n: int) -> np.ndarray:
    """Build the 8-dim legal design-region feature vector for one M2RPair.

    Returns (8,) ndarray.  NaN features are filled with 0.0 (the model will
    learn to ignore them via the tree structure, or the Ridge will shrink them).
    """
    mask = _design_mask(n, sample.sub_start, sample.sub_end)
    wt = _prof(sample.wt_reactivity, n)
    ra = _prof(sample.singleA_reactivity, n)
    rb = _prof(sample.singleB_reactivity, n)

    rmsd_sA = _rmsd_region(wt, ra, mask)
    rmsd_sB = _rmsd_region(wt, rb, mask)
    denom = np.sqrt(rmsd_sA**2 + rmsd_sB**2) if (np.isfinite(rmsd_sA) and np.isfinite(rmsd_sB) and rmsd_sA + rmsd_sB > 0) else np.nan

    delta_sA = _region_mean(ra - wt, mask)   # signed disruption
    delta_sB = _region_mean(rb - wt, mask)
    wt_mean = _region_mean(wt, mask)
    sA_mean = _region_mean(ra, mask)
    sB_mean = _region_mean(rb, mask)

    feats = np.array([rmsd_sA, rmsd_sB, denom,
                      delta_sA, delta_sB,
                      wt_mean, sA_mean, sB_mean], dtype=np.float64)
    feats[~np.isfinite(feats)] = 0.0
    return feats


def build_design_region_oracle_features(sample, n: int) -> np.ndarray:
    """Build the CIRCULAR oracle features (double-mutant profile aggregates).

    Returns (2,) ndarray: [rmsd_double_wt, double_region_mean].
    These are ONLY for the ceiling audit — never for the legal model.
    """
    mask = _design_mask(n, sample.sub_start, sample.sub_end)
    wt = _prof(sample.wt_reactivity, n)
    rd = _prof(sample.double_reactivity, n)

    rmsd_d = _rmsd_region(wt, rd, mask)
    d_mean = _region_mean(rd, mask)

    feats = np.array([rmsd_d, d_mean], dtype=np.float64)
    feats[~np.isfinite(feats)] = 0.0
    return feats


def design_region_feature_names() -> list[str]:
    return [
        "dr_rmsd_sA_wt", "dr_rmsd_sB_wt", "dr_denom_legal",
        "dr_delta_sA_mean", "dr_delta_sB_mean",
        "dr_wt_region_mean", "dr_sA_region_mean", "dr_sB_region_mean",
    ]


def oracle_feature_names() -> list[str]:
    return ["dr_rmsd_double_wt", "dr_double_region_mean"]


def build_all(samples, n: int) -> np.ndarray:
    """Build (n_samples, 8) legal design-region feature matrix."""
    return np.stack([build_design_region_features(s, n) for s in samples])


def build_all_oracle(samples, n: int) -> np.ndarray:
    """Build (n_samples, 2) circular oracle feature matrix."""
    return np.stack([build_design_region_oracle_features(s, n) for s in samples])