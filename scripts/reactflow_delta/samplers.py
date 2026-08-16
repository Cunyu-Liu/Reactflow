#!/usr/bin/env python3
"""Phase 3 samplers: nested leave-one-publication-out fold structure + magnitude targets.

Outer unit = publication (same PMID = one publication). For each fold, one publication
is held out; all studies of the remaining publications form the train-studies for the
fold-local caller (caller_v3), and models train on train-publication changers and are
evaluated on held-out-publication changers.
"""
from __future__ import annotations

import math


def publication_folds(pubs, pair_recs, pub_study):
    """Yield per-fold metadata. (Caller/label computation is done by the run.)"""
    for held_pub in pubs:
        train_studies = set()
        for p_ in pubs:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_pids = [pid for pid in pair_recs if pair_recs[pid]["pub"] != held_pub]
        held_pids = [pid for pid in pair_recs if pair_recs[pid]["pub"] == held_pub]
        yield {
            "held_pub": held_pub,
            "train_studies": train_studies,
            "train_pids": train_pids,
            "held_pids": held_pids,
        }


def pair_magnitude(pf):
    """(magnitude, weight) from PairFeatures over ELIGIBLE positions."""
    wt = pf.wt_reactivity
    mu = pf.mutant_reactivity
    mask = pf.eligibility_mask
    L = min(len(wt), len(mu), len(mask))
    vals = []
    for i in range(L):
        if not mask[i]:
            continue
        a, b = float(wt[i]), float(mu[i])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        vals.append(abs(b - a))
    if not vals:
        return (None, 0)
    return (float(sum(vals)) / len(vals), len(vals))
