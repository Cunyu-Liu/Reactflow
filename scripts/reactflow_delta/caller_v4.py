#!/usr/bin/env python3
"""caller_v4 — information-permission frozen fold-local changer caller (endpoint_v6).

Fixes the Scheme-3 information-flow violation (audit locked fact 12 / operating
principle 7): the target-eligibility mask and any statistic computed from the
held-out publication's pooled WT replicates must never enter a prospective
caller's parameters.

Two explicit caller modes (endpoint_v6.caller_v4_modes):

Primary  STRICT_INDUCTIVE_WT_ALLOWED:
    * null, reliability, and noise/scatter models are fit ONLY from outer-train
      publications' replicate groups (train-only).
    * a held pair may use its OWN deployment-legal WT sequence/profile/error
      (already carried in PairFeatures).
    * it may NOT use the held publication's pooled replicate statistics to
      change caller parameters (sigma). For a held pair whose group is not
      present in the train sigma map, we fall back to a train-global median
      sigma derived from outer-train groups only — never the held group's own
      pooled scatter.

Sensitivity  WT_REPLICATE_CONDITIONED_TRANSDUCTIVE:
    * run only when the deployment scenario guarantees same-domain WT
      replicates are available at inference time.
    * the held group's own WT replicates may be used to estimate sigma
      (transductive conditioning). Must be named/reported separately and never
      mixed with the primary result.

Determinism / fold-locality: identical to CallerV3 (spatial-block null from
train WT-WT disagreement; per-position empirical scatter; ICC(1,1) reliability;
cluster statistic; (b+1)/(B+1) p-value). The ONLY changes vs CallerV3 are:
    * an explicit mode flag that enforces the sigma source boundary;
    * a train-global median-sigma fallback in STRICT mode;
    * a sensitivity transition/coverage manifest that reports label agreement
      between modes and per-publication flip rates.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from caller_v2 import (
    RNG_SEED,
    CLUSTER_WINDOW,
    N_NULL,
    NULL_BLOCK_LEN,
    ALPHA,
    ICC_THRESHOLD,
    MIN_REPLICATES,
    MIN_REPLICATE_GROUPS,
    CallResult,
    CallerV2Error,
    ReplicateGroup,
    input_sha256,
    spatial_block_null,
    _finite,
)
from caller_v3 import CallerV3, _empirical_scatter, _med_positive

CALLER_V4_SCHEMA = "reactflow_delta.caller_v4.v1"

MODE_STRICT = "STRICT_INDUCTIVE_WT_ALLOWED"
MODE_TRANSDUCTIVE = "WT_REPLICATE_CONDITIONED_TRANSDUCTIVE"

# Default stability gate (endpoint_v6.caller_stability_gate). Pre-registered;
# cannot be tuned after observing results.
GATE_OVERALL_LABEL_FLIP = 0.10     # overall label flip <= 10%
GATE_PERPUB_LABEL_FLIP = 0.25      # any non-tiny publication flip <= 25%
GATE_OVERALL_CALLABLE = 0.70       # overall callable coverage >= 70%
GATE_PERPUB_CALLABLE = 0.50        # each inference publication callable >= 50%
GATE_MIN_PUB_FOR_PERPUB_FLIP = 20  # publications smaller than this are "tiny"


class CallerV4(CallerV3):
    """Information-permission frozen changer caller (endpoint_v6).

    Usage mirrors CallerV3::

        caller = CallerV4(mode=MODE_STRICT, seed=RNG_SEED)
        caller.fit(train_groups, train_pairs)          # train-only
        results = [caller.call(pair) for pair in pairs]
        manifest = caller.sensitivity_manifest(train_results, held_results)
    """

    def __init__(
        self,
        mode: str = MODE_STRICT,
        seed: int = RNG_SEED,
        cluster_window: int = CLUSTER_WINDOW,
        n_null: int = N_NULL,
        block_len: int = NULL_BLOCK_LEN,
        alpha: float = ALPHA,
        icc_threshold: float = ICC_THRESHOLD,
        min_replicates: int = MIN_REPLICATES,
        min_replicate_groups: int = MIN_REPLICATE_GROUPS,
    ) -> None:
        if mode not in (MODE_STRICT, MODE_TRANSDUCTIVE):
            raise CallerV2Error(f"unknown caller_v4 mode: {mode!r}")
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
        self.mode = mode
        self._train_median_sigma: Optional[float] = None
        self._train_sigma_by_group: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # fit: STRICT enforces sigma source = outer-train only
    # ------------------------------------------------------------------
    def fit(
        self,
        replicate_groups: Sequence[ReplicateGroup],
        pairs: Sequence,
        noise_replicate_groups: Optional[Sequence[ReplicateGroup]] = None,
    ) -> "CallerV4":
        """Fit on train-fold replicate groups only.

        In STRICT mode, ``noise_replicate_groups`` MUST be None (or exactly the
        train groups). Passing extra (e.g. held-out) groups is rejected to
        prevent held-publication pooled scatter from entering caller params.
        """
        if self.mode == MODE_STRICT and noise_replicate_groups is not None:
            raise CallerV2Error(
                "STRICT_INDUCTIVE_WT_ALLOWED forbids noise_replicate_groups: "
                "sigma must be derived from outer-train replicate groups only."
            )

        usable = [g for g in replicate_groups if g.n_replicates >= self.min_replicates]
        self._reliability_by_group, self._global_reliability = self._reliability_for(
            replicate_groups)
        self._structure_ok = len(usable) >= self.min_replicate_groups
        if not self._structure_ok:
            self._global_reliability = None

        # sigma strictly from TRAIN replicate groups
        self._sigma_by_group = {}
        for g in replicate_groups:
            if g.n_replicates < 2:
                continue
            self._sigma_by_group[g.group_key] = _empirical_scatter(g)
        self._train_sigma_by_group = dict(self._sigma_by_group)

        # train-global median sigma fallback (STRICT): median of positive
        # train scatter entries, used for held pairs whose group is absent.
        meds = [
            _med_positive(s)
            for s in self._train_sigma_by_group.values()
            if _med_positive(s) is not None
        ]
        self._train_median_sigma = float(np.median(meds)) if meds else None

        null_profiles, mask = _null_z_profiles_v4(usable, self._sigma_by_group)
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
                           "across TRAIN WT replicates",
            "z_definition": "z_i = (mut_i - wt_i) / (sqrt(2) * sigma_i)",
            "mode": self.mode,
            "sigma_source": ("outer-train only; held-pair fallback = train-global median"
                             if self.mode == MODE_STRICT else
                             "held-pair own WT replicates allowed (transductive)"),
            "train_median_sigma": self._train_median_sigma,
            "reliability": "ICC(1,1) per replicate group",
            "global_reliability": self._global_reliability,
            "n_null_sampled": len(self._null),
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # add_held_wt_replicates: TRANSDUCTIVE only
    # ------------------------------------------------------------------
    def add_held_wt_replicates(self, groups: Sequence[ReplicateGroup]) -> "CallerV4":
        """Transductive mode only: condition sigma on held groups' WT replicates.

        Not allowed in STRICT mode (would leak held pooled scatter into caller).
        """
        if self.mode != MODE_TRANSDUCTIVE:
            raise CallerV2Error(
                "add_held_wt_replicates is only valid in "
                "WT_REPLICATE_CONDITIONED_TRANSDUCTIVE mode."
            )
        for g in groups:
            if g.n_replicates < 2:
                continue
            self._sigma_by_group[g.group_key] = _empirical_scatter(g)
        return self

    # ------------------------------------------------------------------
    # per-position z: STRICT fallback uses train-global median, never held
    # ------------------------------------------------------------------
    def _z_for_pair(self, pair) -> tuple[list[Optional[float]], list[int]]:
        sigma = self._sigma_by_group.get(pair.group_key)
        if self.mode == MODE_STRICT:
            # sigma map is train-only; if the pair's group is absent (a held
            # publication), fall back to the train-global median. We never
            # compute the held group's own pooled scatter here.
            med_sigma = self._train_median_sigma
        else:
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
                    s = math_sqrt_sq(we, me)
            if s is None or not (s > 0):
                z.append(None)
                continue
            z.append((float(mut) - float(wt)) / (s * math_sqrt_2()))
        return z, list(pair.eligibility_mask)

    # ------------------------------------------------------------------
    # sensitivity manifest: label agreement + transition + coverage
    # ------------------------------------------------------------------
    def sensitivity_manifest(
        self,
        primary_results: Sequence[CallResult],
        sensitivity_results: Sequence[CallResult],
        publication_of: dict[str, str],
    ) -> dict:
        by_id = {r.pair_id: r for r in primary_results}
        n = 0
        agree = 0
        flip_0_1 = flip_1_0 = flip_no_call = 0
        per_pub_flip: dict[str, list] = {}
        per_pub_call: dict[str, list] = {}
        for r in sensitivity_results:
            p = by_id.get(r.pair_id)
            if p is None:
                continue
            n += 1
            pub = publication_of.get(r.pair_id, "UNKNOWN")
            per_pub_call.setdefault(pub, []).append(1 if r.label != "NO_CALL" else 0)
            if r.label == p.label:
                agree += 1
                continue
            if p.label == "NO_CALL" and r.label != "NO_CALL":
                flip_no_call += 1
            elif p.label == "0" and r.label == "1":
                flip_0_1 += 1
            elif p.label == "1" and r.label == "0":
                flip_1_0 += 1
            per_pub_flip.setdefault(pub, []).append(1)
        # per-publication label flip counts (only pairs with a label in both)
        per_pub_flip_rate: dict[str, float] = {}
        per_pub_call_cov: dict[str, float] = {}
        for pub, flips in per_pub_flip.items():
            total = len(per_pub_call.get(pub, []))
            per_pub_flip_rate[pub] = len(flips) / total if total else None
        for pub, calls in per_pub_call.items():
            per_pub_call_cov[pub] = sum(calls) / len(calls) if calls else None

        overall_flip = (n - agree) / n if n else None
        overall_callable = sum(per_pub_call[p] if False else 0 for p in per_pub_call) if False else None
        # pooled callable coverage over all sensitivity rows
        all_calls = [c for calls in per_pub_call.values() for c in calls]
        overall_callable = sum(all_calls) / len(all_calls) if all_calls else None

        # stability gate evaluation
        gate = {
            "overall_label_flip": overall_flip,
            "overall_label_flip_gate": GATE_OVERALL_LABEL_FLIP,
            "worst_nontiny_pub_flip": _worst_flip(per_pub_flip_rate, per_pub_call),
            "per_pub_flip_gate": GATE_PERPUB_LABEL_FLIP,
            "overall_callable_coverage": overall_callable,
            "overall_callable_gate": GATE_OVERALL_CALLABLE,
            "worst_inference_pub_callable": _worst_call(per_pub_call_cov),
            "per_pub_callable_gate": GATE_PERPUB_CALLABLE,
        }
        gate["pass"] = _evaluate_gate(gate)

        return {
            "schema_version": CALLER_V4_SCHEMA,
            "mode_primary": self.mode,
            "mode_sensitivity": MODE_STRICT if self.mode == MODE_TRANSDUCTIVE else MODE_TRANSDUCTIVE,
            "n_paired_rows": n,
            "label_agreement": (agree / n if n else None),
            "transition_matrix": {
                "0_to_1": flip_0_1,
                "1_to_0": flip_1_0,
                "no_call_to_call": flip_no_call,
                "total_disagree": n - agree,
            },
            "overall_call_primary_coverage": overall_callable,
            "publication_macro_call_coverage": (
                sum(per_pub_call_cov.values()) / len(per_pub_call_cov)
                if per_pub_call_cov else None),
            "per_publication_flip_rate": per_pub_flip_rate,
            "per_publication_call_coverage": per_pub_call_cov,
            "stability_gate": gate,
            "params": self._params,
        }


def math_sqrt_2() -> float:
    import math
    return math.sqrt(2.0)


def math_sqrt_sq(a: float, b: float) -> float:
    import math
    return math.sqrt(float(a) ** 2 + float(b) ** 2)


def _null_z_profiles_v4(groups, sigma_by_group):
    """Replicates caller_v3's empirical null-profile builder over train groups."""
    import math
    profiles: list[list[float]] = []
    mask: Optional[list[int]] = None
    for g in groups:
        k = g.n_replicates
        if k < 2:
            continue
        sigma = sigma_by_group.get(g.group_key)
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
                    if sigma is not None and i < len(sigma) and _finite(sigma[i]) and sigma[i] > 0:
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


def _worst_flip(per_pub_flip_rate, per_pub_call) -> Optional[float]:
    worst = None
    for pub, rate in per_pub_flip_rate.items():
        if rate is None:
            continue
        n = len(per_pub_call.get(pub, []))
        if n < GATE_MIN_PUB_FOR_PERPUB_FLIP:
            continue  # "tiny" publication exempt from per-pub flip gate
        if worst is None or rate > worst:
            worst = rate
    return worst


def _worst_call(per_pub_call_cov) -> Optional[float]:
    worst = None
    for pub, cov in per_pub_call_cov.items():
        if cov is None:
            continue
        if worst is None or cov < worst:
            worst = cov
    return worst


def _evaluate_gate(gate) -> bool:
    if gate["overall_label_flip"] is None or gate["overall_label_flip"] > gate["overall_label_flip_gate"]:
        return False
    if gate["worst_nontiny_pub_flip"] is not None and gate["worst_nontiny_pub_flip"] > gate["per_pub_flip_gate"]:
        return False
    if gate["overall_callable_coverage"] is None or gate["overall_callable_coverage"] < gate["overall_callable_gate"]:
        return False
    if gate["worst_inference_pub_callable"] is not None and gate["worst_inference_pub_callable"] < gate["per_pub_callable_gate"]:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit("caller_v4 is a library; use run_caller_v4_sensitivity.py")