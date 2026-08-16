#!/usr/bin/env python3
"""evaluate_v4 — endpoint_v4 metric: publication-macro AUPRC over NON-DEGENERATE publications.

Route A (authority epoch 16, user grant).  This is the reference implementation of
the endpoint_v4 primary metric semantics:

  endpoint_v2/v3 (frozen evaluate_v2.publication_macro_auprc):
    ANY constant-label publication  => whole macro returns UNIDENTIFIABLE
    (degenerate_policies.constant_label, fail-closed, no silent exclusion).

  endpoint_v4 (this module, publication_macro_auprc_non_degenerate):
    constant-label (degenerate) publications are explicitly listed and EXCLUDED;
    macro AUPRC is computed over the remaining NON-DEGENERATE (mixed-label)
    publications only.  The excluded set is ALWAYS returned and must be written to
    results (degenerate_publications field) — no silent dropping.

unit/label/score/mask/information-permission/caller are UNCHANGED from endpoint_v3.
Only the metric degeneracy policy is relaxed (per contract change-control, this is a
new endpoint version, not an in-place edit of evaluate_v2.py).

This module is READ-ONLY w.r.t. evaluate_v2.py: it imports its numeric AP and the
block-helper internals, never modifies them.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

from evaluate_v2 import (
    UNIDENTIFIABLE,
    _average_precision_numeric,
    _flatten,
    _group_blocks,
    is_unidentifiable,
    permutation_p_value,
)


def _group(publications, labels, scores):
    groups: dict[Any, list] = defaultdict(list)
    for pub, lab, sc in zip(publications, labels, scores):
        groups[pub].append((int(bool(lab)), float(sc)))
    return groups


def non_degenerate_publications(publications: Sequence, labels: Sequence) -> tuple[list, list]:
    """Partition publications into (non_degenerate_sorted, degenerate_sorted).

    A publication is degenerate iff its label set is constant (all-positive or
    all-negative).  Ordering is sorted-by-str for determinism.
    """
    groups: dict[Any, list[int]] = defaultdict(list)
    for pub, lab in zip(publications, labels):
        groups[pub].append(int(bool(lab)))
    nondeg: list = []
    deg: list = []
    for pub in sorted(groups, key=str):
        if len(set(groups[pub])) <= 1:
            deg.append(pub)
        else:
            nondeg.append(pub)
    return nondeg, deg


def per_publication_ap(publications: Sequence, labels: Sequence,
                       scores: Sequence) -> dict[Any, Any]:
    """Per-publication numeric AP, or None for a constant-label publication."""
    groups = _group(publications, labels, scores)
    out: dict[Any, Any] = {}
    for pub in sorted(groups, key=str):
        labs = [t for t, _ in groups[pub]]
        scos = [s for _, s in groups[pub]]
        if len(set(labs)) <= 1:
            out[pub] = None
        else:
            out[pub] = float(_average_precision_numeric(labs, scos))
    return out


def publication_macro_auprc_non_degenerate(
        publications: Sequence, labels: Sequence, scores: Sequence
) -> tuple[Any, list, list]:
    """Macro AUPRC over NON-DEGENERATE publications.

    Returns (statistic, non_degenerate_pubs, degenerate_pubs).  statistic is a
    float if >=1 non-degenerate publication exists, else UNIDENTIFIABLE.
    """
    groups = _group(publications, labels, scores)
    if not groups:
        return UNIDENTIFIABLE, [], []
    nondeg: list = []
    deg: list = []
    aps: list[float] = []
    for pub in sorted(groups, key=str):
        labs = [t for t, _ in groups[pub]]
        scos = [s for _, s in groups[pub]]
        if len(set(labs)) <= 1:
            deg.append(pub)
        else:
            nondeg.append(pub)
            aps.append(float(_average_precision_numeric(labs, scos)))
    if not aps:
        return UNIDENTIFIABLE, nondeg, deg
    return sum(aps) / float(len(aps)), nondeg, deg


def bootstrap_ci_non_degenerate(
        publications: Sequence, labels: Sequence, scores: Sequence,
        seed: int = 0, n_boot: int = 1000, alpha: float = 0.05,
) -> dict[str, Any]:
    """Publication-block bootstrap percentile CI of the non-degenerate macro.

    Resample the NON-degenerate publications (blocks) with replacement, recompute
    the macro over the resampled non-degenerate pubs each time, take the
    (alpha/2, 1-alpha/2) percentiles.  Degenerate publications are excluded from
    the estimate (they contribute no AP) but reported.
    """
    rng = random.Random(seed)
    _, nondeg, deg = publication_macro_auprc_non_degenerate(
        publications, labels, scores)
    if is_unidentifiable(_real_if_float(publication_macro_auprc_non_degenerate(
            publications, labels, scores)[0])):
        return {"ci_low": None, "ci_high": None, "point": UNIDENTIFIABLE,
                "n_boot": n_boot, "non_degenerate": nondeg, "degenerate": deg}
    if len(nondeg) < 3:
        # contract: <3 non-degenerate pubs => no confirmatory CI
        return {"ci_low": None, "ci_high": None, "point": None,
                "n_boot": n_boot, "non_degenerate": nondeg, "degenerate": deg,
                "note": "PUBLICATION_LT_3_NO_CONFIRMATORY_CI"}

    groups = _group(publications, labels, scores)
    # only non-degenerate pubs participate in the estimate
    nd_groups = {p: groups[p] for p in nondeg}
    nd_pubs = list(nondeg)
    boots: list[float] = []
    for _ in range(n_boot):
        resampled_pubs = [rng.choice(nd_pubs) for _ in nd_pubs]
        pubs2: list[Any] = []
        labs2: list[int] = []
        scos2: list[float] = []
        for p in resampled_pubs:
            for lab, sc in nd_groups[p]:
                pubs2.append(p)
                labs2.append(lab)
                scos2.append(sc)
        stat, _, _ = publication_macro_auprc_non_degenerate(pubs2, labs2, scos2)
        if not is_unidentifiable(stat):
            boots.append(float(stat))
    if not boots:
        return {"ci_low": None, "ci_high": None, "point": None,
                "n_boot": n_boot, "non_degenerate": nondeg, "degenerate": deg,
                "note": "NO_VALID_BOOTSTRAP"}
    boots.sort()
    lo = boots[int(math.floor((alpha / 2.0) * (len(boots) - 1)))]
    hi = boots[int(math.ceil((1.0 - alpha / 2.0) * (len(boots) - 1)))]
    return {"ci_low": float(lo), "ci_high": float(hi),
            "point": float(sum(boots) / len(boots)),
            "n_boot": n_boot, "non_degenerate": nondeg, "degenerate": deg}


def paired_bootstrap_delta_ci(
        publications: Sequence, labels: Sequence,
        model_scores: Sequence, baseline_scores: Sequence,
        seed: int = 0, n_boot: int = 1000, alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired publication-block bootstrap CI of (model_macro - baseline_macro).

    The SAME resample of non-degenerate publications is applied to model and
    baseline scores, so the CI reflects the paired difference.  Returns
    ci_low/ci_high (None if non-degenerate pubs < 3 or not identifiable).
    """
    _, nondeg, _ = publication_macro_auprc_non_degenerate(
        publications, labels, model_scores)
    if len(nondeg) < 3:
        return {"ci_low": None, "ci_high": None, "n_boot": n_boot,
                "note": "PUBLICATION_LT_3_NO_CONFIRMATORY_CI"}

    mg = _group(publications, labels, model_scores)
    bg = _group(publications, labels, baseline_scores)
    nd_pubs = list(nondeg)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        resampled = [rng.choice(nd_pubs) for _ in nd_pubs]
        m_l, m_s, b_l, b_s = [], [], [], []
        for p in resampled:
            for lab, sc in mg[p]:
                m_l.append(lab)
                m_s.append(sc)
            for lab, sc in bg[p]:
                b_l.append(lab)
                b_s.append(sc)
        m_stat, _, _ = publication_macro_auprc_non_degenerate(resampled, m_l, m_s)
        b_stat, _, _ = publication_macro_auprc_non_degenerate(resampled, b_l, b_s)
        if is_unidentifiable(m_stat) or is_unidentifiable(b_stat):
            continue
        deltas.append(float(m_stat) - float(b_stat))
    if not deltas:
        return {"ci_low": None, "ci_high": None, "n_boot": n_boot,
                "note": "NO_VALID_PAIRED_BOOTSTRAP"}
    deltas.sort()
    lo = deltas[int(math.floor((alpha / 2.0) * (len(deltas) - 1)))]
    hi = deltas[int(math.ceil((1.0 - alpha / 2.0) * (len(deltas) - 1)))]
    return {"ci_low": float(lo), "ci_high": float(hi),
            "point": float(sum(deltas) / len(deltas)), "n_boot": n_boot}


def permutation_test_non_degenerate(
        publications: Sequence, labels: Sequence, scores: Sequence,
        seed: int = 0, n_perm: int = 1000,
) -> dict[str, Any]:
    """Publication-level block permutation test for the non-degenerate macro.

    Mirrors evaluate_v2.permutation_test: blocks are publications (outer unit),
    score-blocks are permuted within equal-size classes (no mixed blocks),
    publication-macro AUPRC (non-degenerate variant) is computed under the null,
    p = (b+1)/(B+1).  Degenerate (constant-label) publications are excluded from
    each permutation's macro by the relaxed policy.
    """
    real, nondeg, deg = publication_macro_auprc_non_degenerate(
        publications, labels, scores)
    if is_unidentifiable(real):
        return {"statistic": UNIDENTIFIABLE, "p_value": None, "b": 0,
                "n_perm": n_perm, "null": [], "non_degenerate": nondeg,
                "degenerate": deg}
    pub_ids, label_blocks, score_blocks = _group_blocks(publications, labels, scores)
    size_classes: dict[int, list[int]] = defaultdict(list)
    for idx, lb in enumerate(label_blocks):
        size_classes[len(lb)].append(idx)

    rng = random.Random(seed)
    null_stats: list[float] = []
    b = 0
    for _ in range(n_perm):
        perm_score_blocks = [None] * len(score_blocks)
        for size, idxs in size_classes.items():
            perm_idxs = idxs[:]
            rng.shuffle(perm_idxs)
            for orig, dest in zip(idxs, perm_idxs):
                perm_score_blocks[dest] = score_blocks[orig]
        perm_labels = _flatten(label_blocks)
        perm_scores = _flatten(perm_score_blocks)
        perm_pubs = [p for p, lb in zip(pub_ids, label_blocks) for _ in lb]
        stat, _, _ = publication_macro_auprc_non_degenerate(
            perm_pubs, perm_labels, perm_scores)
        if is_unidentifiable(stat):
            continue
        null_stats.append(float(stat))
        if float(stat) >= float(real):
            b += 1

    p_value = permutation_p_value(float(real), null_stats)
    return {"statistic": float(real), "p_value": p_value, "b": b,
            "n_perm": n_perm, "null": sorted(null_stats),
            "n_null_numeric": len(null_stats),
            "non_degenerate": nondeg, "degenerate": deg}


def _real_if_float(stat: Any) -> Any:
    return stat
