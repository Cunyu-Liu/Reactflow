#!/usr/bin/env python3
"""m2_caller_v1 — per-position-error changer caller for OpenKnot M2 (no replicates).

Why a dedicated caller for M2
-----------------------------
The reactflow_delta CallerV4 (endpoint_v6, STRICT_INDUCTIVE_WT_ALLOWED) derives its
per-position noise sigma and its spatial-block null from WT-WT REPLICATES within the
train fold (ICC reliability gate -> NO_CALL when replicates are absent).  M2 has NO
WT replicates for any design, so CallerV4's reliability gate would return NO_CALL for
every M2 unit.  To label M2 changers we therefore use the dataset's own per-position
``reactivity_error`` as the noise scale, with an explicit documented statistical
assumption (below).

Changer definition (per single-nt mutant, judged on the mutation response):
  * per-position z-score over the aligned profile
        z[i] = (mut[i] - wt[i]) / sqrt(wt_err[i]^2 + mut_err[i]^2)
    only where both reactivities are finite and the pooled error > 0.
  * cluster statistic T = max over contiguous eligible windows (length <=
    CLUSTER_WINDOW) of the size-normalised RMS of the squared z's:
        T(window) = sqrt( mean_{i in window} z[i]^2 ).
    A size-normalised RMS keeps a short hotspot and a long weak cluster comparable,
    matching the size-normalisation fix of the frozen v2/v4 callers.
  * null: because no WT replicates exist, the null is the max-T distribution of iid
    standard-normal z profiles over the SAME eligibility mask (B = N_NULL samples).
    This encodes the explicit assumption that M2's per-position reactivity_error is
    calibrated so that under no-change each z[i] ~ N(0,1).
  * p-value uses (b+1)/(B+1); label "1" (changer) iff p <= ALPHA, else "0".

Determinism: fixed documented RNG seed; byte-reproducible for identical input.

This module ONLY generates labels + reliability-free caller for M2.  It does not
build the sequence->spectrum predictor (that is the runner + residual model).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

M2_CALLER_SCHEMA = "reactflow_delta.m2_caller.v1"
RNG_SEED = 20260812            # documented deterministic seed
CLUSTER_WINDOW = 15            # max eligible positions per cluster window
N_NULL = 2000                  # Gaussian-null samples
ALPHA = 0.05                   # one-sided significance for the cluster statistic
PLUS_ONE_NULL = True           # p = (b+1)/(B+1)

_SQRT2 = math.sqrt(2.0)


def _finite(v: Optional[float]) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def per_position_z(wt_react, mut_react, wt_err, mut_err, mask) -> list[Optional[float]]:
    """Per-position z = (mut-wt)/sqrt(wt_err^2 + mut_err^2); None where invalid."""
    n = len(mask)
    z: list[Optional[float]] = []
    for i in range(n):
        if not mask[i]:
            z.append(None)
            continue
        wt, mu = wt_react[i], mut_react[i]
        if not (_finite(wt) and _finite(mu)):
            z.append(None)
            continue
        we, me = wt_err[i], mut_err[i]
        s = _SQRT2
        if _finite(we) and _finite(me):
            s = math.sqrt(we * we + me * me)
        if not (s > 0):
            z.append(None)
            continue
        z.append((float(mu) - float(wt)) / s)
    return z


def max_cluster_rms(z, mask, cluster_window=CLUSTER_WINDOW) -> float:
    """Size-normalised RMS cluster statistic over contiguous eligible windows."""
    z = list(z)
    mask = list(mask)
    n = len(mask)
    best = 0.0
    i = 0
    while i < n:
        if not mask[i] or z[i] is None:
            i += 1
            continue
        # extend contiguous eligible run
        j = i
        while j < n and mask[j] and z[j] is not None:
            j += 1
        # sliding windows of length 1..cluster_window within [i, j)
        zsq = [z[k] * z[k] for k in range(i, j)]
        L = j - i
        for length in range(1, min(L, cluster_window) + 1):
            s = sum(zsq[:length])
            best = max(best, math.sqrt(max(s, 0.0) / length))
            for end in range(length, L):
                s += zsq[end] - zsq[end - length]
                # s is a sum of squares; guard tiny negative float-roundoff
                best = max(best, math.sqrt(max(s, 0.0) / length))
        i = j
    return float(best)


def gaussian_null(mask, cluster_window=CLUSTER_WINDOW, n_null=N_NULL, seed=RNG_SEED,
                  rng=None) -> np.ndarray:
    """Max-T null from iid standard-normal z over the same mask.

    Only eligible positions get noise (the rest are ignored by the statistic),
    so the null carries the same spatial support / sparsity as the observation.
    """
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if rng is None:
        rng = np.random.default_rng(seed)
    stats = np.empty(n_null, dtype=np.float64)
    for b in range(n_null):
        z = rng.standard_normal(n)
        z[~mask] = 0.0
        # max_cluster_rms treats mask[i]=0 as excluded, so pass z and mask
        stats[b] = max_cluster_rms(z, mask.astype(int), cluster_window)
    return stats


@dataclass
class M2CallResult:
    pair_id: str
    label: str                 # "1" | "0"
    statistic: float
    p_value: float

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id, "label": self.label,
            "statistic": self.statistic, "p_value": self.p_value,
        }


def call_mutant(
    pair_id, wt_react, mut_react, wt_err, mut_err, mask,
    cluster_window=CLUSTER_WINDOW, n_null=N_NULL, alpha=ALPHA, seed=RNG_SEED,
    null: Optional[np.ndarray] = None, rng=None,
) -> M2CallResult:
    """Label one mutant as changer ("1") or not ("0").

    ``null`` (precomputed gaussian_null) may be passed to avoid recomputation when
    many mutants share the same eligibility mask; otherwise it is computed here.
    """
    z = per_position_z(wt_react, mut_react, wt_err, mut_err, mask)
    stat = max_cluster_rms(z, mask, cluster_window)
    if null is None:
        null = gaussian_null(mask, cluster_window, n_null, seed, rng=rng)
    if PLUS_ONE_NULL:
        b = int((null >= stat).sum())
        p = float((b + 1) / (len(null) + 1))
    else:
        p = float((null >= stat).mean())
    label = "1" if p <= alpha else "0"
    return M2CallResult(pair_id=pair_id, label=label, statistic=stat, p_value=p)
