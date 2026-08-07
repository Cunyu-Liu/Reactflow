#!/usr/bin/env python3
"""R3 — fold-local changer caller v2 (label-generation).

Implements the frozen endpoint caller contract (configs/reactflow_delta/
endpoint_v2.yaml, caller_contract; audit contract §13.2 R3, §13.3 items 2 & 5).

Semantics
---------
C_i = 1  ("changer")  iff mutation i induces a *significant experimental
        reactivity response*, judged by a spatial-cluster statistic computed on
        control-standardised per-position reactivity deltas, tested against a
        deterministic spatial-block null built ONLY from train-fold WT-WT
        replicate disagreement.  The caller is fit ONLY on the training fold;
        any attempt to access an outer (validation/test) row raises a hard
        error (OuterFoldAccessError) and writes a consumption event.
C_i = 0  ("nonchanger") when the statistic is not significant but the unit is
        reliable enough to make that binary assertion.
NO_CALL  when the train-fold unit reliability (ICC-style) is below the
        documented threshold: the caller refuses to force a binary label.

Design notes vs. the legacy ph0x caller (NOT reused as final; kept read-only)
---------------------------------------------------------------------------
* Sliding-window bias fixed: the legacy ``_max_cluster`` accumulated an
  ever-growing sum-of-squares, so the "max" was always the full run (up to the
  window cap) — a long run always won and there was no well-defined cluster
  boundary (edge/overlap/length bias).  v2 uses a SIZE-NORMALISED scan
  statistic (RMS of the squared z-scores over each contiguous eligible window
  of <= CLUSTER_WINDOW positions), so a short hotspot and a long weak cluster
  are compared on equal footing and the max is a real extremum, not a
  monotone artifact.
* ICC heterogeneity: reliability is computed per replicate group (ICC 1,1) and
  gated per unit, instead of a single pooled number.
* Spatial-block null: the legacy null independently resampled per-position
  null z's, destroying spatial correlation and making the null artificially
  tight.  v2 builds the null by BLOCK-PERMUTING contiguous segments from real
  train-fold WT-WT difference profiles (spatial block bootstrap), preserving
  local spatial correlation.
* Determinism: a fixed documented RNG seed (RNG_SEED) and a deterministic
  PRNG (random.Random) make null/labels byte-reproducible for identical input.
  An input SHA-256 is recorded so "same input -> same output" is checkable.

Fold-locality enforcement (audit §13.3.2 / §13.3.5)
---------------------------------------------------
A FoldLocalLoader admits ONLY rows whose study role is in the train roles.
* Reading an outer (validation/test) row raises OuterFoldAccessError (hard
  fail) *and* records a consumption event (event_type="OUTER_ROW_ACCESS").
* A Seal is broken permanently on the first outer access; restore() raises
  SealViolationError, so aggregate reporting can never "restore sealed".

Reliability / NO_CALL
---------------------
For each train-fold replicate group (same sequence + probe + temperature) an
ICC(1,1) is estimated from the WT profiles.  If a group has fewer than
MIN_REPLICATES replicates, or its ICC < ICC_THRESHOLD (documented default
0.50), every pair whose WT belongs to that group returns NO_CALL rather than a
forced binary label.  A global train reliability summary is also emitted.
If the train fold as a whole has < MIN_REPLICATE_GROUPS usable replicate
groups, the caller is considered unreliable and returns NO_CALL for all units
(honest negative result rather than a fabricated label).

This module ONLY generates labels + reliability + determinism + fold-locality
guard.  It does NOT build the binary-vs-probability predictor (that is R4/R5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

CALLER_SCHEMA = "reactflow_delta.caller_v2.v1"

# ---------------------------------------------------------------------------
# Documented, frozen constants
# ---------------------------------------------------------------------------
RNG_SEED = 20260807          # deterministic seed (documented in manifest)
CLUSTER_WINDOW = 15          # max eligible positions per cluster window
MIN_ELIGIBLE_IN_WINDOW = 1   # min eligible positions for a window to count
N_NULL = 2000                # spatial-block null permutations
NULL_BLOCK_LEN = 5           # spatial block length for block permutation (contiguous positions)
ALPHA = 0.05                 # one-sided significance for the cluster statistic
ICC_THRESHOLD = 0.50         # documented reliability threshold (below -> NO_CALL)
MIN_REPLICATES = 2           # min distinct WT profiles for a usable replicate group
MIN_REPLICATE_GROUPS = 5     # min usable replicate groups for the caller to be reliable at all
PLUS_ONE_NULL = True         # p-value uses (b+1)/(B+1) per audit §13.3.5


class CallerV2Error(Exception):
    """Base error for the v2 caller."""


class OuterFoldAccessError(CallerV2Error):
    """Raised when the caller/loader attempts to read a non-train (outer) row."""


class SealViolationError(CallerV2Error):
    """Raised when something tries to restore a broken seal."""


# ---------------------------------------------------------------------------
# Consumption / exposure tracking (audit §13.3.2)
# ---------------------------------------------------------------------------
class ConsumptionLedger:
    """Append-only log of data-access events.  Never editable, never restorable."""

    def __init__(self) -> None:
        self._events: list[dict] = []

    def record(self, row_id: str, role: str, event_type: str) -> None:
        self._events.append({
            "row_id": str(row_id),
            "role": role,
            "event_type": event_type,
            "ordinal": len(self._events),
        })

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    @property
    def n_events(self) -> int:
        return len(self._events)

    def has_outer_access(self) -> bool:
        return any(e["event_type"] == "OUTER_ROW_ACCESS" for e in self._events)

    def to_manifest(self) -> dict:
        return {
            "n_events": len(self._events),
            "outer_access_occurred": self.has_outer_access(),
            "events": list(self._events),
        }


class Seal:
    """A seal that, once broken by outer access, can never be restored.

    ``restore()`` deliberately does not exist; any attempt to "un-break" the
    seal via break() with a restore flag raises SealViolationError so that
    aggregate reporting can never silently restore a sealed state.
    """

    def __init__(self, ledger: Optional[ConsumptionLedger] = None) -> None:
        self._ledger = ledger or ConsumptionLedger()
        self._sealed = True

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def break_seal(self, row_id: str, role: str) -> None:
        """Permanently break the seal because an outer row was touched."""
        self._ledger.record(row_id, role, "OUTER_ROW_ACCESS")
        self._sealed = False

    def record_train(self, row_id: str, role: str) -> None:
        self._ledger.record(row_id, role, "TRAIN_ROW_ACCESS")

    def restore(self) -> None:
        # Aggregate reporting must NOT be able to restore sealed (audit §13.3.2).
        raise SealViolationError(
            "Seal cannot be restored: once broken by outer-row access, the "
            "sealed state is permanent and aggregate reporting must not restore it."
        )


# ---------------------------------------------------------------------------
# Data model (pure-Python, stdlib only; no numpy dependency)
# ---------------------------------------------------------------------------
@dataclass
class PairFeatures:
    """Per-pair caller input: aligned per-position arrays over the WT profile.

    All arrays share the same length (== len(eligibility_mask)).
    """
    pair_id: str
    wt_reactivity: list[float]
    mutant_reactivity: list[float]
    wt_error: list[float]
    mutant_error: list[float]
    eligibility_mask: list[int]          # 1 = eligible, 0 = excluded
    group_key: tuple = ("", "")          # (study, probe) replicate-group key
    role: str = "train"

    def __len__(self) -> int:
        return len(self.eligibility_mask)

    def canonical_bytes(self) -> bytes:
        """Stable serialisation used for input hashing / determinism checks."""
        payload = {
            "pair_id": self.pair_id,
            "wt": self.wt_reactivity,
            "mut": self.mutant_reactivity,
            "werr": self.wt_error,
            "merr": self.mutant_error,
            "mask": self.eligibility_mask,
            "group": list(self.group_key),
            "role": self.role,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class ReplicateGroup:
    """A train-fold replicate block: >=MIN_REPLICATES distinct WT profiles.

    Used for ICC reliability estimation and for building the spatial-block null.
    ``group_key`` must be the full replicate-group identity
    ``(study, canonical_sequence, probe, temperature)`` so distinct replicate
    groups never collide in the reliability map.
    """
    group_key: tuple
    wt_profiles: list[list[float]]   # each a full-length WT reactivity profile
    wt_errors: list[list[float]]
    eligibility_mask: list[int]
    study: str = ""

    @property
    def n_replicates(self) -> int:
        return len(self.wt_profiles)


@dataclass
class CallResult:
    pair_id: str
    label: str                      # "1" | "0" | "NO_CALL"
    statistic: Optional[float]      # T_i (RMS cluster statistic)
    p_value: Optional[float]
    reliability: Optional[float]    # ICC of the unit's replicate group
    group_key: tuple

    def is_changer(self) -> bool:
        return self.label == "1"

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "label": self.label,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "reliability": self.reliability,
            "group_key": list(self.group_key),
        }


@dataclass
class CallerManifest:
    params: dict
    null_quantile: float
    null_median: float
    global_reliability: Optional[float]
    reliability_by_group: dict
    input_sha256: str
    n_units: int
    n_changer: int
    n_nonchanger: int
    n_no_call: int
    seal_outer_access: bool

    def to_dict(self) -> dict:
        return {
            "schema_version": CALLER_SCHEMA,
            "params": self.params,
            "null_quantile": self.null_quantile,
            "null_median": self.null_median,
            "global_reliability": self.global_reliability,
            "reliability_by_group": self.reliability_by_group,
            "input_sha256": self.input_sha256,
            "label_counts": {
                "n_units": self.n_units,
                "changer": self.n_changer,
                "nonchanger": self.n_nonchanger,
                "no_call": self.n_no_call,
            },
            "seal_outer_access": self.seal_outer_access,
        }


# ---------------------------------------------------------------------------
# Fold-locality guard (loader)
# ---------------------------------------------------------------------------
class FoldLocalLoader:
    """Admits rows ONLY from train roles; outer access -> hard fail + event.

    ``role_of(study)`` must map a study prefix to a split role.  Any row whose
    role is not in ``train_roles`` raises OuterFoldAccessError and breaks the
    seal (recording a consumption event).  This makes reading an outer row both
    a hard error and a recorded exposure — never silently possible.
    """

    def __init__(self, split_roles: dict[str, str],
                 train_roles: Sequence[str] = ("train",),
                 ledger: Optional[ConsumptionLedger] = None) -> None:
        self.split_roles = dict(split_roles)
        self.train_roles = set(train_roles)
        self.ledger = ledger or ConsumptionLedger()
        self.seal = Seal(self.ledger)

    @staticmethod
    def study_of(source_accession: str) -> str:
        return (source_accession or "").split("_")[0]

    def _role_for(self, study: str) -> str:
        return self.split_roles.get(study, "UNASSIGNED")

    def assert_train(self, pair_id: str, study: str) -> str:
        """Verify a row is train; raise (and record exposure) otherwise."""
        role = self._role_for(study)
        if role not in self.train_roles:
            # Hard fail + permanent exposure record.  Do NOT continue.
            self.seal.break_seal(pair_id, role)
            raise OuterFoldAccessError(
                f"Fold-locality violation: pair {pair_id!r} (study {study!r}, "
                f"role {role!r}) is an OUTER row; the caller may only read "
                f"train roles {sorted(self.train_roles)}.  Access refused."
            )
        self.seal.record_train(pair_id, role)
        return role


# ---------------------------------------------------------------------------
# Core numerics (pure Python)
# ---------------------------------------------------------------------------
def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def compute_eligible_mask(eligibility_reason_codes: Sequence[Any]) -> list[int]:
    """Mask: 1 where the pair code is ELIGIBLE, else 0 (matches endpoint mask)."""
    return [1 if (c == "ELIGIBLE") else 0 for c in eligibility_reason_codes]


def per_position_z(pair: PairFeatures) -> tuple[list[Optional[float]], list[int]]:
    """Control-standardised delta z_i = (mut - wt) / sqrt(wt_err^2 + mut_err^2).

    Returns (z, eligible) where z is None where not measurable (ineligible,
    missing, or zero/unknown noise) and eligible is the position mask.
    """
    n = len(pair)
    z: list[Optional[float]] = []
    for i in range(n):
        if not pair.eligibility_mask[i]:
            z.append(None)
            continue
        wt, mut = pair.wt_reactivity[i], pair.mutant_reactivity[i]
        we, me = pair.wt_error[i], pair.mutant_error[i]
        if not (_finite(wt) and _finite(mut) and _finite(we) and _finite(me)):
            z.append(None)
            continue
        noise = math.sqrt(float(we) ** 2 + float(me) ** 2)
        if not math.isfinite(noise) or noise <= 0:
            z.append(None)
            continue
        z.append((float(mut) - float(wt)) / noise)
    return z, list(pair.eligibility_mask)


def max_cluster_stat(z: Sequence[Optional[float]],
                     eligible: Sequence[int]) -> float:
    """SIZE-NORMALISED sliding-window cluster statistic (fixes legacy bias).

    Slides a window over each contiguous run of eligible positions (windows of
    up to CLUSTER_WINDOW eligible positions).  For a window of ``k`` eligible
    positions with squared-z sum ``S``, the statistic is ``sqrt(S/k)`` — the
    RMS z over the window.  Normalising by sqrt(k) removes the legacy
    monotone-growth / edge / overlap / length bias, so a short strong hotspot
    and a long weak cluster are comparable and the max is a genuine extremum.
    """
    best = 0.0
    n = len(z)
    i = 0
    while i < n:
        if not eligible[i] or z[i] is None:
            i += 1
            continue
        # contiguous eligible run starting at i
        run_z: list[float] = []
        j = i
        while j < n and eligible[j]:
            if z[j] is not None:
                run_z.append(float(z[j]))
            j += 1
        # scan windows of up to CLUSTER_WINDOW eligible positions within run
        run_len = len(run_z)
        for start in range(run_len):
            s = 0.0
            for end in range(start, min(start + CLUSTER_WINDOW, run_len)):
                s += run_z[end] ** 2
                k = end - start + 1
                if k < MIN_ELIGIBLE_IN_WINDOW:
                    continue
                val = math.sqrt(s / k)
                if val > best:
                    best = val
        i = j
    return best


def icc_one_way(profiles: Sequence[Sequence[float]],
                mask: Sequence[int]) -> Optional[float]:
    """ICC(1,1) over positions (targets) x replicates (raters), one-way random.

    k = number of replicates, L = number of eligible positions.
      MSB = between-position mean square
      MSW = within-position mean square
      ICC(1,1) = (MSB - MSW) / (MSB + (k-1)*MSW)
    Returns None if not computable (e.g. < 2 replicates or < 2 eligible pos).
    """
    k = len(profiles)
    if k < 2:
        return None
    length = min((len(p) for p in profiles), default=0)
    eligible_idx = [i for i in range(min(length, len(mask))) if mask[i]]
    if len(eligible_idx) < 2:
        return None
    L = len(eligible_idx)
    data = [[profiles[r][i] for r in range(k)] for i in eligible_idx]
    # grand mean
    total = sum(sum(row) for row in data)
    grand = total / (L * k)
    # per-position means
    pos_means = [sum(row) / k for row in data]
    # MSB between positions
    ssb = k * sum((pm - grand) ** 2 for pm in pos_means)
    msb = ssb / (L - 1)
    # MSW within positions
    ssw = sum(sum((v - pm) ** 2 for v in row) for row, pm in zip(data, pos_means))
    msw = ssw / (L * (k - 1))
    if msw == 0 and msb == 0:
        return None
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return float(icc)


# ---------------------------------------------------------------------------
# Spatial-block null (deterministic)
# ---------------------------------------------------------------------------
def _null_z_profiles(replicate_groups: Sequence[ReplicateGroup]
                     ) -> tuple[list[list[float]], list[int]]:
    """Build null z-profiles (WT-WT disagreement) from train replicate groups.

    For each replicate group and each distinct pair of WT profiles (a,b),
    compute per-eligible-position (prof_a - prof_b)/sqrt(err_a^2 + err_b^2).
    Each such vector is a full null profile that inherently carries the local
    spatial correlation of real WT-WT disagreement.
    """
    profiles: list[list[float]] = []
    mask = None
    for g in replicate_groups:
        k = g.n_replicates
        if k < 2:
            continue
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
                    ea, eb = g.wt_errors[a][i], g.wt_errors[b][i]
                    if not (_finite(wa) and _finite(wb) and _finite(ea) and _finite(eb)):
                        continue
                    noise = math.sqrt(float(ea) ** 2 + float(eb) ** 2)
                    if not math.isfinite(noise) or noise <= 0:
                        continue
                    prof.append((float(wa) - float(wb)) / noise)
                if len(prof) >= 2:
                    profiles.append(prof)
    return profiles, mask


def spatial_block_null(null_profiles: Sequence[Sequence[float]],
                       mask: Sequence[int],
                       n_null: int = N_NULL,
                       block_len: int = NULL_BLOCK_LEN,
                       seed: int = RNG_SEED) -> list[float]:
    """Deterministic spatial-block null distribution of the cluster statistic.

    Resamples pseudo-profiles by concatenating random contiguous BLOCKS of
    length ``block_len`` drawn from the real null profiles (a spatial block
    bootstrap), then computes the same size-normalised cluster statistic.
    Block-permuting contiguous positions preserves local spatial correlation,
    so the null is not artificially tight (unlike the legacy independent
    per-position resampling).  Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    if not null_profiles:
        return []
    all_ones = [1] * (max(len(p) for p in null_profiles))
    lengths = [len(p) for p in null_profiles]
    target_len = sorted(lengths)[len(lengths) // 2]  # median length
    target_len = max(target_len, block_len)
    nulls: list[float] = []
    for _ in range(n_null):
        pseudo: list[float] = []
        while len(pseudo) < target_len:
            prof = rng.choice(null_profiles)
            start = rng.randrange(0, len(prof))
            seg = [prof[(start + t) % len(prof)] for t in range(block_len)]
            pseudo.extend(seg)
        pseudo = pseudo[:target_len]
        nulls.append(max_cluster_stat(pseudo, all_ones[:target_len]))
    nulls.sort()
    return nulls


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """q-th quantile of a sorted list (linear interpolation)."""
    if not sorted_values:
        raise CallerV2Error("empty null distribution")
    n = len(sorted_values)
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo]) * (1 - frac) + float(sorted_values[hi]) * frac


def _p_value(sorted_null: Sequence[float], obs: float, plus_one: bool = PLUS_ONE_NULL) -> float:
    """(b+1)/(B+1) one-sided p-value per audit §13.3.5."""
    B = len(sorted_null)
    if B == 0:
        return 1.0
    b = sum(1 for v in sorted_null if v >= obs)
    if plus_one:
        return (b + 1) / (B + 1)
    return b / B


# ---------------------------------------------------------------------------
# The caller
# ---------------------------------------------------------------------------
def input_sha256(replicate_groups: Sequence[ReplicateGroup],
                 pairs: Sequence[PairFeatures],
                 seed: int) -> str:
    """Stable SHA-256 of the caller input (replicate groups + pairs + seed)."""
    h = hashlib.sha256()
    h.update(seed.to_bytes(8, "big"))
    for g in sorted(replicate_groups, key=lambda g: repr(g.group_key)):
        h.update(repr(g.group_key).encode())
        h.update(b"|" + b";".join(repr(p).encode() for p in g.wt_profiles))
    for p in sorted(pairs, key=lambda p: p.pair_id):
        h.update(p.canonical_bytes())
    return h.hexdigest()


class CallerV2:
    """Deterministic, fold-local changer caller.

    Usage::
        caller = CallerV2(seed=RNG_SEED)
        caller.fit(train_replicate_groups, train_pairs_for_hash)   # train-only
        results = [caller.call(pair) for pair in train_pairs]
        manifest = caller.manifest(results)
    """

    def __init__(self, seed: int = RNG_SEED,
                 cluster_window: int = CLUSTER_WINDOW,
                 n_null: int = N_NULL,
                 block_len: int = NULL_BLOCK_LEN,
                 alpha: float = ALPHA,
                 icc_threshold: float = ICC_THRESHOLD,
                 min_replicates: int = MIN_REPLICATES,
                 min_replicate_groups: int = MIN_REPLICATE_GROUPS) -> None:
        self.seed = int(seed)
        self.cluster_window = int(cluster_window)
        self.n_null = int(n_null)
        self.block_len = int(block_len)
        self.alpha = float(alpha)
        self.icc_threshold = float(icc_threshold)
        self.min_replicates = int(min_replicates)
        self.min_replicate_groups = int(min_replicate_groups)
        self._fitted = False
        self._structure_ok = False
        self._reliability_by_group: dict[tuple, Optional[float]] = {}
        self._null: list[float] = []
        self._global_reliability: Optional[float] = None
        self._input_hash: Optional[str] = None
        self._params: dict = {}

    def _reliability_for(self, groups: Sequence[ReplicateGroup]) -> tuple[dict, Optional[float]]:
        rel: dict[tuple, Optional[float]] = {}
        vals: list[float] = []
        for g in groups:
            if g.n_replicates < self.min_replicates:
                # insufficient replicate structure -> unit reliability is None -> NO_CALL
                rel[g.group_key] = None
                continue
            icc = icc_one_way(g.wt_profiles, g.eligibility_mask)
            rel[g.group_key] = icc
            if icc is not None:
                vals.append(icc)
        global_rel = (sum(vals) / len(vals)) if vals else None
        return rel, global_rel

    def fit(self, replicate_groups: Sequence[ReplicateGroup],
            pairs: Sequence[PairFeatures]) -> "CallerV2":
        """Fit on train-fold replicate/control data only.

        Computes per-group ICC reliability, the global reliability, and the
        deterministic spatial-block null.  Does not touch any pair outcome; the
        null is built solely from WT-WT replicate disagreement.
        """
        usable = [g for g in replicate_groups if g.n_replicates >= self.min_replicates]
        # reliability map covers ALL groups (insufficient ones are mapped to None)
        self._reliability_by_group, self._global_reliability = self._reliability_for(replicate_groups)
        self._structure_ok = len(usable) >= self.min_replicate_groups
        if not self._structure_ok:
            # too little replicate structure in the train fold -> globally unreliable
            self._global_reliability = None

        null_profiles, mask = _null_z_profiles(usable)
        self._null = spatial_block_null(
            null_profiles, mask if mask is not None else [],
            n_null=self.n_null, block_len=self.block_len, seed=self.seed)

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
            "reliability": "ICC(1,1) per replicate group",
            "global_reliability": self._global_reliability,
            "n_null_sampled": len(self._null),
        }
        self._fitted = True
        return self

    def _unit_reliability(self, group_key: tuple) -> Optional[float]:
        if group_key in self._reliability_by_group:
            return self._reliability_by_group[group_key]
        return self._global_reliability

    def call(self, pair: PairFeatures) -> CallResult:
        """Call a single pair; returns a binary label or NO_CALL.

        fold-locality: this method does not itself consult any outer row — it
        operates purely on the train-side PairFeatures already admitted by the
        loader.  The loader (FoldLocalLoader.assert_train) is the enforcement
        point for reading outer rows.
        """
        if not self._fitted:
            raise CallerV2Error("caller not fitted: call fit() on train-fold replicate groups first")

        # reliability gate -> NO_CALL (audit §13.3 R3: "reliability过低返回NO_CALL")
        rel = self._unit_reliability(pair.group_key)
        if not self._structure_ok or rel is None or rel < self.icc_threshold:
            return CallResult(
                pair_id=pair.pair_id, label="NO_CALL",
                statistic=None, p_value=None, reliability=rel,
                group_key=pair.group_key)

        z, eligible = per_position_z(pair)
        if not any(eligible):
            return CallResult(pair_id=pair.pair_id, label="NO_CALL",
                              statistic=None, p_value=None, reliability=rel,
                              group_key=pair.group_key)

        stat = self._cluster_with(z, eligible, self.cluster_window)
        p = _p_value(self._null, stat)
        label = "1" if p <= self.alpha else "0"
        return CallResult(pair_id=pair.pair_id, label=label, statistic=stat,
                          p_value=p, reliability=rel, group_key=pair.group_key)

    @staticmethod
    def _cluster_with(z, eligible, window) -> float:
        # size-normalised cluster with an explicit window cap
        best = 0.0
        n = len(z)
        i = 0
        while i < n:
            if not eligible[i] or z[i] is None:
                i += 1
                continue
            run_z: list[float] = []
            j = i
            while j < n and eligible[j]:
                if z[j] is not None:
                    run_z.append(float(z[j]))
                j += 1
            run_len = len(run_z)
            for start in range(run_len):
                s = 0.0
                for end in range(start, min(start + window, run_len)):
                    s += run_z[end] ** 2
                    k = end - start + 1
                    if k < MIN_ELIGIBLE_IN_WINDOW:
                        continue
                    val = math.sqrt(s / k)
                    if val > best:
                        best = val
            i = j
        return best

    @property
    def null_quantile(self) -> Optional[float]:
        if not self._null:
            return None
        return _quantile(self._null, 1 - self.alpha)

    @property
    def null_median(self) -> Optional[float]:
        if not self._null:
            return None
        return _quantile(self._null, 0.5)

    @property
    def null_distribution(self) -> list[float]:
        """Copy of the fitted spatial-block null distribution (deterministic)."""
        return list(self._null)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def reliability_for(self, group_key: tuple) -> Optional[float]:
        return self._unit_reliability(group_key)

    def manifest(self, results: Sequence[CallResult]) -> CallerManifest:
        n_changer = sum(1 for r in results if r.is_changer())
        n_non = sum(1 for r in results if r.label == "0")
        n_nc = sum(1 for r in results if r.label == "NO_CALL")
        return CallerManifest(
            params=self._params,
            null_quantile=self.null_quantile,
            null_median=self.null_median,
            global_reliability=self._global_reliability,
            reliability_by_group={repr(k): v for k, v in self._reliability_by_group.items()},
            input_sha256=self._input_hash or "",
            n_units=len(results),
            n_changer=n_changer,
            n_nonchanger=n_non,
            n_no_call=n_nc,
            seal_outer_access=False,
        )


# ---------------------------------------------------------------------------
# CLI: run the caller over a v2 canonical dataset + split (train-fold only)
# ---------------------------------------------------------------------------
def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def build_pair_features(pair: dict, wt_rec: dict, mut_rec: dict) -> PairFeatures:
    """Join a primary pair with its WT and mutant canonical records."""
    wl = wt_rec.get("reactivity_layers", {})
    ml = mut_rec.get("reactivity_layers", {})
    wt_react = list(wl.get("train_frozen", {}).get("reactivity") or wl.get("raw", {}).get("reactivity") or [])
    wt_err = list(wl.get("train_frozen", {}).get("error") or wl.get("raw", {}).get("error") or [])
    mut_react = list(ml.get("train_frozen", {}).get("reactivity") or ml.get("raw", {}).get("reactivity") or [])
    mut_err = list(ml.get("train_frozen", {}).get("error") or ml.get("raw", {}).get("error") or [])
    codes = pair.get("eligibility_reason_codes") or []
    mask = compute_eligible_mask(codes)
    # full replicate-group identity of the pair's WT, so reliability maps
    # to the correct (study, sequence, probe, temperature) replicate block
    grp = (_study_of(pair.get("source_accession") or ""),
           wt_rec.get("canonical_sequence") or "",
           tuple(wt_rec.get("probe") or []),
           tuple(wt_rec.get("temperature") or []))
    return PairFeatures(
        pair_id=f"{pair.get('source_accession')}:{pair.get('mutant_profile_index')}",
        wt_reactivity=wt_react,
        mutant_reactivity=mut_react,
        wt_error=wt_err,
        mutant_error=mut_err,
        eligibility_mask=mask,
        group_key=grp,
        role=pair.get("_role", "train"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="R3 fold-local caller v2 (label generation)")
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--pairs-jsonl", type=Path, required=True)
    ap.add_argument("--split-yaml", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    import yaml
    split = yaml.safe_load(args.split_yaml.read_text(encoding="utf-8"))
    assignment = split.get("assignment", {})
    train_studies = {s for s, r in assignment.items() if r == "train"}

    loader = FoldLocalLoader(assignment, train_roles=("train",))

    # index canonical records by (accession, profile_index, asset)
    canon_index: dict[tuple, dict] = {}
    with open(args.canonical_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            canon_index[(r.get("source_accession"), r.get("source_profile_index"),
                         r.get("source_asset_name"))] = r

    replicate_groups: dict[tuple, dict] = {}
    pairs: list[PairFeatures] = []
    with open(args.pairs_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            pair = json.loads(line)
            study = _study_of(pair.get("source_accession") or "")
            loader.assert_train(pair.get("source_accession") or "", study)  # fold-local
            wt_rec = canon_index.get((pair.get("source_accession"),
                                      pair.get("wt_profile_index"),
                                      pair.get("asset_name")))
            mut_rec = canon_index.get((pair.get("source_accession"),
                                       pair.get("mutant_profile_index"),
                                       pair.get("asset_name")))
            if wt_rec is None or mut_rec is None:
                continue
            pf = build_pair_features(pair, wt_rec, mut_rec)
            pairs.append(pf)

    # replicate groups from train-fold WT records (same seq+probe+temp, >=2 profiles)
    # NOTE: building replicate groups from the full canonical stream is handled by
    # the caller of this module; the CLI builds them from train-fold WT records.
    wt_by_key: dict[tuple, dict] = {}
    with open(args.canonical_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("is_wt"):
                continue
            study = _study_of(r.get("source_accession") or "")
            if study not in train_studies:
                continue
            key = (study, r.get("canonical_sequence"), tuple(r.get("probe") or []),
                   tuple(r.get("temperature") or []))
            wt_by_key.setdefault(key, []).append(r)

    rep_groups: list[ReplicateGroup] = []
    for (study, seq, probe, temp), recs in wt_by_key.items():
        if len(recs) < 2:
            continue
        rl0 = recs[0].get("reactivity_layers", {})
        mask = compute_eligible_mask(rl0.get("eligibility_reason_codes") or [])
        profs = [list(r.get("reactivity_layers", {}).get("train_frozen", {}).get("reactivity") or []) for r in recs]
        errs = [list(r.get("reactivity_layers", {}).get("train_frozen", {}).get("error") or []) for r in recs]
        rep_groups.append(ReplicateGroup(
            group_key=(study, seq, tuple(probe), tuple(temp)),
            wt_profiles=profs, wt_errors=errs,
            eligibility_mask=mask, study=study))

    caller = CallerV2(seed=args.seed).fit(rep_groups, pairs)
    results = [caller.call(p) for p in pairs]
    manifest = caller.manifest(results)
    manifest.seal_outer_access = loader.seal.has_outer_access()
    args.out_manifest.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "n_units": manifest.n_units,
        "changer": manifest.n_changer,
        "nonchanger": manifest.n_nonchanger,
        "no_call": manifest.n_no_call,
        "global_reliability": manifest.global_reliability,
        "null_quantile": manifest.null_quantile,
        "seal_outer_access": loader.seal.has_outer_access(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
