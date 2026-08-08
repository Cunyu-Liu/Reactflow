#!/usr/bin/env python3
"""evaluate_v5 — endpoint_v5 conditional-magnitude metric: conditional WMAE skill.

Route B (authority epoch 17, user grant).  This is the reference implementation of
the endpoint_v5 CONDITIONAL magnitude metric.  It does NOT modify the frozen
primary endpoint (endpoint_v4 / evaluate_v4 / evaluate_v2).  It imports only the
shared UNIDENTIFIABLE sentinel and block helpers from evaluate_v2.

Task (endpoint_v5.conditional):
  * Unit    = matched WT-exact_single-mutant pair, restricted to TRUE CHANGERS
              (C_i=1 as adjudicated by the fold-local caller_v3).
  * Target  = y_i = profile-level magnitude =
              mean over ELIGIBLE positions of |mutant_react[i] - wt_react[i]|
              (raw absolute reactivity change).  weight w_i = n_eligible_positions.
  * Score   = hat{y}_i (regression head direct output).
  * Metric  = conditional WMAE skill = 1 - WMAE_model / WMAE_baseline, where
              WMAE = sum_i w_i |y_i - pred_i| / sum_i w_i and the baseline is the
              strongest same-information trivial (train-changer weighted mean).

Resampling (publication is the outer unit):
  * paired publication-block bootstrap CI of the skill (lower bound > 0 => GO).
  * publication-block permutation p = (b+1)/(B+1)  (reported, may be degenerate).

Degeneracy (fail-closed, mirroring evaluate_v4 conventions):
  * no true changers in held-out  => UNIDENTIFIABLE
  * <3 publications with changers  => PUBLICATION_LT_3_NO_CONFIRMATORY_CI
  * held-out magnitude constant    => UNIDENTIFIABLE
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

from evaluate_v2 import (
    UNIDENTIFIABLE,
    _flatten,
    _group_blocks,
    is_unidentifiable,
    permutation_p_value,
)


# ---------------------------------------------------------------------------
# Core weighted MAE / skill
# ---------------------------------------------------------------------------
def _wmae(y: Sequence[float], pred: Sequence[float],
          w: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for yi, pi, wi in zip(y, pred, w):
        num += float(wi) * abs(float(yi) - float(pi))
        den += float(wi)
    if den <= 0.0:
        raise ValueError("sum of weights must be > 0")
    return num / den


def _weighted_mean(y: Sequence[float], w: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for yi, wi in zip(y, w):
        num += float(wi) * float(yi)
        den += float(wi)
    if den <= 0.0:
        raise ValueError("sum of weights must be > 0")
    return num / den


def conditional_wmae_skill(
        publications: Sequence,
        y: Sequence[float],
        weights: Sequence[float],
        model_pred: Sequence[float],
        baseline_pred: Sequence[float],
) -> dict[str, Any]:
    """Conditional WMAE skill over the pooled held-out true-changers.

    Returns dict with skill, wmae_model, wmae_baseline, n_changers,
    n_publications, or UNIDENTIFIABLE/notes per degeneracy policy.
    """
    y = [float(v) for v in y]
    w = [float(v) for v in weights]
    mp = [float(v) for v in model_pred]
    bp = [float(v) for v in baseline_pred]

    if not y:
        return {"skill": UNIDENTIFIABLE, "n_changers": 0, "n_publications": 0,
                "note": "NO_TRUE_CHANGER_HELDOUT"}
    # constant-magnitude (no learnable variance) -> unidentifiable
    if _weighted_mean(y, w) <= 0.0 or max(y) - min(y) <= 1e-12:
        return {"skill": UNIDENTIFIABLE, "n_changers": len(y),
                "n_publications": len(set(publications)),
                "note": "CONSTANT_MAGNITUDE_HELDOUT"}

    wmae_model = _wmae(y, mp, w)
    wmae_base = _wmae(y, bp, w)
    if wmae_base <= 0.0:
        return {"skill": UNIDENTIFIABLE, "wmae_model": wmae_model,
                "wmae_baseline": wmae_base, "n_changers": len(y),
                "n_publications": len(set(publications)),
                "note": "BASELINE_ZERO_WMAE"}
    skill = 1.0 - wmae_model / wmae_base
    return {"skill": float(skill), "wmae_model": float(wmae_model),
            "wmae_baseline": float(wmae_base), "n_changers": len(y),
            "n_publications": len(set(publications))}


def paired_bootstrap_skill_ci(
        publications: Sequence, y: Sequence[float], weights: Sequence[float],
        model_pred: Sequence[float], baseline_pred: Sequence[float],
        seed: int = 0, n_boot: int = 1000, alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired publication-block bootstrap percentile CI of the conditional skill.

    Resample publications (blocks) with replacement; recompute the pooled
    conditional skill over the resampled changers each draw; take the
    (alpha/2, 1-alpha/2) percentiles of the skill distribution.
    """
    pubs = list(publications)
    n_pubs = len(set(pubs))
    if n_pubs < 3:
        return {"ci_low": None, "ci_high": None, "skill": None,
                "n_publications": n_pubs, "n_boot": n_boot,
                "note": "PUBLICATION_LT_3_NO_CONFIRMATORY_CI"}

    # group changers by publication
    groups: dict[Any, dict] = defaultdict(lambda: {"y": [], "w": [], "mp": [], "bp": []})
    for p, yi, wi, mi, bi in zip(pubs, y, weights, model_pred, baseline_pred):
        groups[p]["y"].append(float(yi))
        groups[p]["w"].append(float(wi))
        groups[p]["mp"].append(float(mi))
        groups[p]["bp"].append(float(bi))
    pub_ids = sorted(groups, key=str)

    rng = random.Random(seed)
    skills: list[float] = []
    for _ in range(n_boot):
        resampled_pubs = [rng.choice(pub_ids) for _ in pub_ids]
        ry, rw, rmp, rbp = [], [], [], []
        for p in resampled_pubs:
            ry.extend(groups[p]["y"])
            rw.extend(groups[p]["w"])
            rmp.extend(groups[p]["mp"])
            rbp.extend(groups[p]["bp"])
        base = _wmae(ry, rbp, rw)
        if base <= 0.0:
            continue
        skills.append(1.0 - _wmae(ry, rmp, rw) / base)
    if not skills:
        return {"ci_low": None, "ci_high": None, "skill": None,
                "n_publications": n_pubs, "n_boot": n_boot,
                "note": "NO_VALID_BOOTSTRAP"}
    skills.sort()
    lo = skills[int(math.floor((alpha / 2.0) * (len(skills) - 1)))]
    hi = skills[int(math.ceil((1.0 - alpha / 2.0) * (len(skills) - 1)))]
    return {"ci_low": float(lo), "ci_high": float(hi),
            "skill": float(sum(skills) / len(skills)),
            "n_publications": n_pubs, "n_boot": n_boot}


def permutation_test_skill(
        publications: Sequence, y: Sequence[float], weights: Sequence[float],
        model_pred: Sequence[float], baseline_pred: Sequence[float],
        seed: int = 0, n_perm: int = 1000,
) -> dict[str, Any]:
    """Publication-level block permutation test for the conditional skill.

    Score-blocks (model predictions) are permuted within equal-size classes
    (publications are the outer unit; no mixed blocks), the conditional skill is
    recomputed under the null, p = (b+1)/(B+1).
    """
    res = conditional_wmae_skill(publications, y, weights, model_pred, baseline_pred)
    real = res.get("skill")
    if res.get("n_changers", 0) == 0 or is_unidentifiable(real):
        return {"statistic": UNIDENTIFIABLE, "p_value": None, "b": 0,
                "n_perm": n_perm, "null": [], "note": "NO_TRUE_CHANGER_HELDOUT"}
    yf = [float(v) for v in y]
    base = _wmae(yf, [float(v) for v in baseline_pred],
                 [float(v) for v in weights])
    if base <= 0.0:
        return {"statistic": UNIDENTIFIABLE, "p_value": None, "b": 0,
                "n_perm": n_perm, "null": [], "note": "SKILL_UNIDENTIFIABLE_OR_BASE_ZERO"}

    # group the four aligned arrays by publication (block = publication)
    groups: dict[Any, dict] = defaultdict(lambda: {"y": [], "w": [], "mp": [], "bp": []})
    for p, yi, wi, mi, bi in zip(publications, y, weights, model_pred, baseline_pred):
        groups[p]["y"].append(float(yi))
        groups[p]["w"].append(float(wi))
        groups[p]["mp"].append(float(mi))
        groups[p]["bp"].append(float(bi))
    pub_ids = sorted(groups, key=str)
    y_blocks = [groups[p]["y"] for p in pub_ids]
    mp_blocks = [groups[p]["mp"] for p in pub_ids]
    w_blocks = [groups[p]["w"] for p in pub_ids]
    bp_blocks = [groups[p]["bp"] for p in pub_ids]

    size_classes: dict[int, list[int]] = defaultdict(list)
    for idx, lb in enumerate(y_blocks):
        size_classes[len(lb)].append(idx)

    rng = random.Random(seed)
    null_skills: list[float] = []
    b = 0
    for _ in range(n_perm):
        perm_mp_blocks = [None] * len(mp_blocks)
        for size, idxs in size_classes.items():
            perm_idxs = idxs[:]
            rng.shuffle(perm_idxs)
            for orig, dest in zip(idxs, perm_idxs):
                perm_mp_blocks[dest] = mp_blocks[orig]
        perm_y = _flatten(y_blocks)
        perm_mp = _flatten(perm_mp_blocks)
        pw = _flatten(w_blocks)
        wsum = sum(pw)
        if wsum <= 0.0:
            continue
        # weights & baseline stay with their original publications (unchanged)
        null_wmae_m = _wmae(perm_y, perm_mp, pw)
        null_skill = 1.0 - null_wmae_m / base
        null_skills.append(float(null_skill))
        if float(null_skill) >= float(real):
            b += 1

    p_value = permutation_p_value(float(real), null_skills)
    return {"statistic": float(real), "p_value": p_value, "b": b,
            "n_perm": n_perm, "null": sorted(null_skills),
            "n_null_numeric": len(null_skills)}
