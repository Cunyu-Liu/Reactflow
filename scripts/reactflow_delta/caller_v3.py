#!/usr/bin/env python3
"""R3v3 — fold-local changer caller v3 (endpoint_v3 label-generation).

Implements the endpoint_v3 caller contract (configs/reactflow_delta/
endpoint_v3.yaml, caller_contract; proposal
docs/contracts/amendments/ReactFlowDelta_endpoint_v3_proposal_20260807.md).

Why v3 (calibration fix)
------------------------
The R5 P2 learnability gate (frozen caller_v2) produced an effectively
constant binary changer label (3 changers / 6359 pool pairs) because the
frozen caller_v2 noise model was miscalibrated:
  * cross-study reactivity scale heterogeneity (median spread ~293000x);
  * reported per-position errors are miscalibrated (median ~2.5x, mean ~8.4x
    smaller than empirical replicate scatter).

caller_v3 changes ONLY the noise model (no label/unit/score/metric change):
  * z_i = (mut_i - wt_i) / (sqrt(2) * sigma_i), where sigma_i is the
    per-position EMPIRICAL scatter (sample SD, ddof=1) across the replicate
    group's WT profiles.
  * The spatial-block null is built from WT-WT replicate disagreement scaled
    by the SAME empirical scatter.
  * Per-study (median/MAD) normalization is a common linear scale and cancels
    in the ratio, so it is documented but not applied inside z.

Fold-locality (audit §13.3.2 / §13.3.5)
----------------------------------------
The statistical null and unit reliability are computed ONLY from TRAIN-fold
replicate groups (outer outcome invisible).  The per-group empirical scatter
(sigma) may optionally be sourced from a wider set of WT replicate groups
(noise_replicate_groups); WT reactivity is an ALLOWED input per endpoint_v3
and does not carry the mutant outcome, and correct per-study scaling requires
each held-out study's own WT scale.  The null always remains train-only.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from caller_v2 import (
    CallerV2,
    CallResult,
    CallerV2Error,
    ReplicateGroup,
    spatial_block_null,
    _p_value,
    input_sha256,
    _finite,
    CLUSTER_WINDOW,
    N_NULL,
    NULL_BLOCK_LEN,
    RNG_SEED,
    ALPHA,
    ICC_THRESHOLD,
    MIN_REPLICATES,
    MIN_REPLICATE_GROUPS,
    PLUS_ONE_NULL,
)

CALLER_V3_SCHEMA = "reactflow_delta.caller_v3.v1"


def _empirical_scatter(g: ReplicateGroup) -> np.ndarray:
    """Per-position empirical scatter = sample SD (ddof=1) across WT profiles.

    Returns a length-``min(len(profile))`` numpy array; positions with < 2
    finite replicate values are NaN (handled by caller fallback).
    """
    L = min(len(p) for p in g.wt_profiles)
    arr = np.full((g.n_replicates, L), np.nan, dtype=float)
    for r in range(g.n_replicates):
        for i in range(L):
            v = g.wt_profiles[r][i]
            if _finite(v):
                arr[r, i] = float(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.nanstd(arr, axis=0, ddof=1)
    return sd


def _med_positive(sigma: Optional[np.ndarray]) -> Optional[float]:
    """Median of finite, positive sigma entries (used as per-group fallback)."""
    if sigma is None:
        return None
    vals = [float(s) for s in sigma if _finite(s) and s > 0]
    if not vals:
        return None
    return float(np.median(vals))


class CallerV3(CallerV2):
    """Fold-local changer caller with empirical-scatter noise recalibration.

    Usage mirrors CallerV2::

        caller = CallerV3(seed=RNG_SEED)
        caller.fit(train_groups, [], noise_replicate_groups=all_pool_groups)
        results = [caller.call(pair) for pair in pairs]
        manifest = caller.manifest(results)
    """

    def __init__(
        self,
        seed: int = RNG_SEED,
        cluster_window: int = CLUSTER_WINDOW,
        n_null: int = N_NULL,
        block_len: int = NULL_BLOCK_LEN,
        alpha: float = ALPHA,
        icc_threshold: float = ICC_THRESHOLD,
        min_replicates: int = MIN_REPLICATES,
        min_replicate_groups: int = MIN_REPLICATE_GROUPS,
    ) -> None:
        super().__init__(
            seed=seed,
            cluster_window=cluster_window,
            n_null=n_null,
            block_len=block_len,
            alpha=alpha,
            icc_threshold=icc_threshold,
            min_replicates=min_replicates,
            min_replicate_groups=min_replicate_groups,
        )
        self._sigma_by_group: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        replicate_groups: Sequence[ReplicateGroup],
        pairs: Sequence,
        noise_replicate_groups: Optional[Sequence[ReplicateGroup]] = None,
    ) -> "CallerV3":
        """Fit on train-fold replicate/control data only.

        Parameters
        ----------
        replicate_groups:
            TRAIN-fold replicate groups.  Used for reliability + spatial-block
            null (outer outcome invisible).
        pairs:
            Train-side pairs (only for the input SHA-256 fingerprint).
        noise_replicate_groups:
            Optional source of per-group empirical scatter.  Defaults to
            ``replicate_groups``.  Pass the full-pool WT replicate groups so
            each held-out study is scaled by its own WT scale (an ALLOWED
            input; the null stays train-only).
        """
        usable = [g for g in replicate_groups if g.n_replicates >= self.min_replicates]
        self._reliability_by_group, self._global_reliability = self._reliability_for(
            replicate_groups)
        self._structure_ok = len(usable) >= self.min_replicate_groups
        if not self._structure_ok:
            self._global_reliability = None

        noise_groups = (
            noise_replicate_groups
            if noise_replicate_groups is not None else replicate_groups
        )
        self._sigma_by_group = {}
        for g in noise_groups:
            if g.n_replicates < 2:
                continue
            self._sigma_by_group[g.group_key] = _empirical_scatter(g)

        null_profiles, mask = self._null_z_profiles_empirical(usable)
        self._null = spatial_block_null(
            null_profiles,
            mask if mask is not None else [],
            n_null=self.n_null,
            block_len=self.block_len,
            seed=self.seed,
        )

        self._input_hash = input_sha256(usable, pairs, self.seed)
        self._params = {
            "seed": self.seed,
            "cluster_window": self.cluster_window,
            "n_null": self.n_null,
            "block_len": self.block_len,
            "alpha": self.alpha,
            "icc_threshold": self.icc_threshold,
            "min_replicates": self.min_replicates,
            "min_replicate_groups": self.min_replicate_groups,
            "p_value_rule": "(b+1)/(B+1)",
            "statistic": "size-normalised RMS sliding-window cluster",
            "null": "spatial-block bootstrap (contiguous blocks of WT-WT disagreement)",
            "noise_model": "empirical per-position scatter (sample SD, ddof=1) "
                           "across WT replicates; reported errors NOT used "
                           "(miscalibrated in caller_v2)",
            "z_definition": "z_i = (mut_i - wt_i) / (sqrt(2) * sigma_i)",
            "per_study_normalization": "cancels in the z ratio; documented not applied",
            "reliability": "ICC(1,1) per replicate group",
            "global_reliability": self._global_reliability,
            "n_null_sampled": len(self._null),
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # empirical-scatter null profiles (train-fold only)
    # ------------------------------------------------------------------
    def _null_z_profiles_empirical(
        self, groups: Sequence[ReplicateGroup],
    ) -> tuple[list[list[float]], Optional[list[int]]]:
        profiles: list[list[float]] = []
        mask: Optional[list[int]] = None
        for g in groups:
            k = g.n_replicates
            if k < 2:
                continue
            sigma = self._sigma_by_group.get(g.group_key)
            mask = g.eligibility_mask
            length = min(len(p) for p in g.wt_profiles)
            eligible_idx = [i for i in range(min(length, len(mask))) if mask[i]]
            if len(eligible_idx) < 2:
                continue
            for a in range(k):
                for b in range(a + 1, k):
                    prof: list[float] = []
                    for i in eligible_idx:
                        wa, wb = g.wt_profiles[a][i], g.wt_profiles[b][i]
                        s = None
                        if (
                            sigma is not None
                            and i < len(sigma)
                            and _finite(sigma[i])
                            and sigma[i] > 0
                        ):
                            s = float(sigma[i])
                        if s is None:
                            ea, eb = g.wt_errors[a][i], g.wt_errors[b][i]
                            if _finite(ea) and _finite(eb) and (ea > 0 or eb > 0):
                                s = math.sqrt(float(ea) ** 2 + float(eb) ** 2)
                        if s is None or not (_finite(wa) and _finite(wb)):
                            continue
                        prof.append((float(wa) - float(wb)) / (s * math.sqrt(2.0)))
                    if len(prof) >= 2:
                        profiles.append(prof)
        return profiles, mask

    # ------------------------------------------------------------------
    # empirical-scatter per-position z for a candidate pair
    # ------------------------------------------------------------------
    def _z_for_pair(self, pair) -> tuple[list[Optional[float]], list[int]]:
        sigma = self._sigma_by_group.get(pair.group_key)
        med_sigma = _med_positive(sigma)
        n = len(pair)
        z: list[Optional[float]] = []
        for i in range(n):
            if not pair.eligibility_mask[i]:
                z.append(None)
                continue
            wt, mut = pair.wt_reactivity[i], pair.mutant_reactivity[i]
            if not (_finite(wt) and _finite(mut)):
                z.append(None)
                continue
            s: Optional[float] = None
            if (
                sigma is not None
                and i < len(sigma)
                and _finite(sigma[i])
                and sigma[i] > 0
            ):
                s = float(sigma[i])
            if s is None and med_sigma is not None:
                s = med_sigma
            if s is None:
                we, me = pair.wt_error[i], pair.mutant_error[i]
                if _finite(we) and _finite(me) and (we > 0 or me > 0):
                    s = math.sqrt(float(we) ** 2 + float(me) ** 2)
            if s is None or not (s > 0):
                z.append(None)
                continue
            z.append((float(mut) - float(wt)) / (s * math.sqrt(2.0)))
        return z, list(pair.eligibility_mask)

    # ------------------------------------------------------------------
    # call
    # ------------------------------------------------------------------
    def call(self, pair) -> CallResult:
        if not self._fitted:
            raise CallerV2Error("caller not fitted: call fit() on train-fold replicate groups first")

        rel = self._unit_reliability(pair.group_key)
        if not self._structure_ok or rel is None or rel < self.icc_threshold:
            return CallResult(
                pair_id=pair.pair_id, label="NO_CALL",
                statistic=None, p_value=None, reliability=rel,
                group_key=pair.group_key)

        z, eligible = self._z_for_pair(pair)
        if not any(eligible):
            return CallResult(pair_id=pair.pair_id, label="NO_CALL",
                              statistic=None, p_value=None, reliability=rel,
                              group_key=pair.group_key)

        stat = self._cluster_with(z, eligible, self.cluster_window)
        p = _p_value(self._null, stat)
        label = "1" if p <= self.alpha else "0"
        return CallResult(pair_id=pair.pair_id, label=label, statistic=stat,
                          p_value=p, reliability=rel, group_key=pair.group_key)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def sigma_for(self, group_key: tuple) -> Optional[np.ndarray]:
        """Copy of the per-group empirical scatter (None if group absent)."""
        s = self._sigma_by_group.get(group_key)
        return None if s is None else np.array(s)

    @property
    def schema(self) -> str:
        return CALLER_V3_SCHEMA
