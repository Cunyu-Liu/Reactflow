#!/usr/bin/env python3
"""R4: reference-consistent evaluator v2 for the frozen endpoint_v2 semantics.

Implements the standard metric / resampling package behind the frozen
`configs/reactflow_delta/endpoint_v2.yaml` spec and contract §13.2/R4, §13.3.

Primary estimand (RFD_ENDPOINT_V2):
    unit      = matched WT-exact_single-mutant pair
    label     = C_i (binary changer, frozen caller)
    score     = pair-level P(C_i=1)  (direct model output)
    metric    = publication-macro AUPRC        (publication is the OUTER resampling unit)
    resampling= publication-level block permutation with p = (b+1)/(B+1)
Conditional = WMAE skill on |delta_r| magnitude.
Secondary   = WMAE skill / Spearman on signed delta at eligible positions.

Degenerate / UNIDENTIFIABLE boundaries (endpoint_v2 `degenerate_policies`, §13.3):
  * constant label (all one class)                 -> UNIDENTIFIABLE  (no number)
  * pair-any all-positive                          -> DEGENERATE      -> UNIDENTIFIABLE
  * publication < 3 (no confirmatory CI)           -> UNIDENTIFIABLE
  * missing info is NOT zero-filled (coverage)     -> missing stays missing
  * tied AP is row-order invariant                 -> guaranteed by distinct-threshold grouping
  * same PMID = one publication = one outer unit

Implementation notes
--------------------
This module is intentionally dependency-free (pure Python stdlib).  The
required runtime (`/mnt/cunyuliu/reactflow_delta_runtime_py311/bin/python`)
does NOT ship numpy/scipy/sklearn, so every metric is implemented from scratch
and cross-checked against a hand-verified reference that reproduces
scikit-learn's published algorithm and documented outputs (e.g.
``average_precision_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.83``).  A
scikit-learn cross-check is still wired into the test suite and runs whenever
sklearn is importable (it is skipped when absent).  All randomness is seeded
and reproducible: same input -> same output.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Sentinel codes
# ---------------------------------------------------------------------------
UNIDENTIFIABLE = "UNIDENTIFIABLE"
DEGENERATE = "DEGENERATE"

# scikit-learn documented reference: average_precision_score([0,0,1,1],
# [0.1,0.4,0.35,0.8]) == 0.83 (exact 5/6).
_SKLEARN_AP_DOC_EXAMPLE = 5.0 / 6.0


def is_unidentifiable(x: Any) -> bool:
    """True if ``x`` is a degenerate/unidentifiable sentinel rather than a number."""
    return x is UNIDENTIFIABLE or x is DEGENERATE or x == UNIDENTIFIABLE


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# ---------------------------------------------------------------------------
# Average precision (tie-aware; exact clone of sklearn's binary algorithm)
# ---------------------------------------------------------------------------
def _average_precision_numeric(y_true: Sequence, y_score: Sequence) -> float:
    """Numeric average precision, exactly reproducing scikit-learn's binary
    ``average_precision_score`` (``_binary_uninterpolated_average_precision``
    via ``precision_recall_curve``).

    The distinct-threshold grouping makes the result INVARIANT to the row order
    of tied scores (contract §13.3 item 4 / ``tied_ap_row_order_invariant``).

    Edge handling (documented):
      * all-positive labels -> 1.0   (every precision step == 1)
      * all-negative labels -> 0.0   (no positive to retrieve; sklearn raises,
                                      we return the limiting value 0.0)
    Callers that need the endpoint degenerate policy must guard on constant
    labels *before* calling this (see ``average_precision_tie_aware``).
    """
    y_true = [int(bool(t)) for t in y_true]
    y_score = [float(s) for s in y_score]
    n = len(y_true)
    if n == 0:
        return 0.0
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    if n_pos == n:
        return 1.0

    # stable descending sort by score (Python sorted is stable)
    order = sorted(range(n), key=lambda i: y_score[i], reverse=True)
    yt = [y_true[i] for i in order]
    ys = [y_score[i] for i in order]

    # cumulative positive counts
    tps_at = [0] * n
    c = 0
    for i in range(n):
        c += yt[i]
        tps_at[i] = c

    # distinct score-change indices + final index (mirrors np.where(np.diff) r_[...])
    distinct = [i for i in range(n - 1) if ys[i] != ys[i + 1]]
    threshold_idxs = distinct + [n - 1]

    tps = [tps_at[i] for i in threshold_idxs]
    fps = [1 + idx - tp for idx, tp in zip(threshold_idxs, tps)]
    precision = [(tp / (tp + fp)) if (tp + fp) else 0.0 for tp, fp in zip(tps, fps)]
    recall = [tp / n_pos for tp in tps]

    # keep only the segment up to full recall, then append terminal points
    # (recall = 1 / precision = 1 at the top; recall = 0 / precision = 1 end cap).
    last_ind = tps.index(tps[-1])  # == np.searchsorted(tps, tps[-1], side="left")
    sl = slice(last_ind, None, -1)
    prec_curve = precision[sl] + [1.0]
    rec_curve = recall[sl] + [0.0]

    # AP = - sum( diff(recall) * precision[:-1] )   (step / uninterpolated integral)
    ap = 0.0
    for i in range(1, len(rec_curve)):
        ap += -(rec_curve[i] - rec_curve[i - 1]) * prec_curve[i - 1]
    return max(0.0, ap)


def average_precision_tie_aware(y_true: Sequence, y_score: Sequence) -> Any:
    """Endpoint-level average precision.

    Applies ``degenerate_policies.constant_label``: if the label vector is
    constant (all 0 or all 1) the metric is UNIDENTIFIABLE (no number), because
    a single-class label set makes AP degenerate / non-informative.  Otherwise
    returns the tie-aware, row-order-invariant AP (see ``_average_precision_numeric``).
    """
    yt = [int(bool(t)) for t in y_true]
    if len(yt) == 0:
        return UNIDENTIFIABLE
    if len(set(yt)) <= 1:
        return UNIDENTIFIABLE
    return _average_precision_numeric(yt, y_score)


# ---------------------------------------------------------------------------
# Brier score (proper scoring rule for probabilities)
# ---------------------------------------------------------------------------
def brier_score(y_true: Sequence, y_score: Sequence) -> float:
    """Brier score = mean((y_true - y_score)^2).  Proper scoring rule for
    probabilities; always defined (it evaluates calibration, not ranking), so it
    is NOT guarded by the constant-label policy.
    """
    yt = [float(t) for t in y_true]
    ys = [float(s) for s in y_score]
    n = len(yt)
    if n == 0:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(yt, ys)) / float(n)


# ---------------------------------------------------------------------------
# Coverage (missing stays missing; never zero-filled)
# ---------------------------------------------------------------------------
def coverage(eligible: Sequence, predicted: Sequence) -> float:
    """Fraction of *eligible* positions/pairs that carry a non-missing prediction.

    ``eligible[i]`` truthy => position i is eligible (denominator).
    A prediction counts as covered only if it is a finite number; NaN / None /
    missing stays missing and is NOT zero-filled (``missing_info_not_zero``).
    Returns 0.0 when there are no eligible positions.
    """
    n = len(eligible)
    if n == 0:
        return 0.0
    covered = 0
    for i in range(n):
        if not eligible[i]:
            continue
        if i < len(predicted) and predicted[i] is not None and _is_finite(predicted[i]):
            covered += 1
    return covered / float(sum(1 for e in eligible if e))


# ---------------------------------------------------------------------------
# Weighted mean absolute error skill
# ---------------------------------------------------------------------------
def _wmae(y_true: Sequence[float], y_pred: Sequence[float],
          weights: Sequence[float]) -> float:
    num = sum(w * abs(t - p) for t, p, w in zip(y_true, y_pred, weights))
    den = sum(weights)
    if den <= 0:
        return float("nan")
    return num / den


def wmae_skill(y_true: Sequence, y_pred: Sequence, weights: Optional[Sequence] = None,
               reference: Optional[Sequence] = None) -> Any:
    """Weighted-MAE skill vs a reference prediction.

        Skill = 1 - WMAE(pred) / WMAE(reference)

    WMAE = weighted mean absolute error.  ``reference`` defaults to the constant
    (trivial) baseline that predicts the weighted mean of ``y_true``, which is
    the standard 'predict-the-mean' reference; a candidate that simply matches
    that constant therefore has skill 0, and a perfect candidate (pred == true)
    has skill 1.

    signed vs absolute semantics: the *error* is always the absolute deviation
    (MAE) — this is correct for both the conditional (|delta| magnitude) and the
    secondary (signed delta) tasks; what differs is the target passed in by the
    caller (use ``absolute_target`` to map signed deltas to magnitudes for the
    conditional task).  When the reference itself is perfect (WMAE == 0) the
    skill is undefined -> UNIDENTIFIABLE.
    """
    n = len(y_true)
    if n == 0:
        return UNIDENTIFIABLE
    w = [1.0] * n if weights is None else [float(x) for x in weights]
    yt = [float(t) for t in y_true]
    yp = [float(p) for p in y_pred]
    if reference is None:
        den = sum(w)
        mean = sum(w[i] * yt[i] for i in range(n)) / den if den > 0 else 0.0
        ref = [mean] * n
    else:
        ref = [float(r) for r in reference]

    wmae_pred = _wmae(yt, yp, w)
    wmae_ref = _wmae(yt, ref, w)
    if not math.isfinite(wmae_pred) or not math.isfinite(wmae_ref):
        return UNIDENTIFIABLE
    if wmae_ref == 0.0:
        # reference is perfect; if the model is also perfect the ratio is 0/0
        # and we define skill 0 (equal to the perfect reference), else undefined.
        return 0.0 if wmae_pred == 0.0 else UNIDENTIFIABLE
    return 1.0 - wmae_pred / wmae_ref


def absolute_target(signed: Sequence) -> list[float]:
    """Map signed deltas to |delta| magnitudes (conditional-task target)."""
    return [abs(float(x)) for x in signed]


# ---------------------------------------------------------------------------
# Rank correlations (secondary continuous ranking)
# ---------------------------------------------------------------------------
def _rank(data: Sequence[float]) -> list[float]:
    """Average-rank ties (same as scipy.stats.rankdata(average))."""
    n = len(data)
    order = sorted(range(n), key=lambda i: data[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and data[order[j + 1]] == data[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1..n
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Any:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    varx = sum((a - mx) ** 2 for a in x)
    vary = sum((b - my) ** 2 for b in y)
    if varx == 0 or vary == 0:
        return UNIDENTIFIABLE
    return cov / math.sqrt(varx * vary)


def spearman(x: Sequence, y: Sequence) -> Any:
    """Spearman rank correlation = Pearson on average-ranks.  Returns
    UNIDENTIFIABLE when either input is constant (zero rank variance)."""
    if len(x) == 0:
        return UNIDENTIFIABLE
    return _pearson(_rank(x), _rank(y))


def kendall(x: Sequence, y: Sequence) -> Any:
    """Kendall's tau-b (ties handled).  Returns UNIDENTIFIABLE on degenerate input."""
    n = len(x)
    concordant = discordant = tx = ty = 0
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            a = xi - x[j]
            b = yi - y[j]
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tx) * (concordant + discordant + ty))
    if denom == 0:
        return UNIDENTIFIABLE
    return (concordant - discordant) / denom


# ---------------------------------------------------------------------------
# Publication-macro AUPRC (primary metric; publication = outer unit)
# ---------------------------------------------------------------------------
def publication_macro_auprc(publications: Sequence, labels: Sequence,
                            scores: Sequence) -> Any:
    """Publication-macro AUPRC.

    Computes AP within each publication (score = P(C_i=1)), then macro-averages
    over publications (each publication contributes equally).

    Degenerate policy (fail-closed, no silent exclusion): if ANY publication has
    a constant label set (all-positive or all-negative — including the
    ``pair_any_all_positive`` case), the whole metric is UNIDENTIFIABLE rather
    than a number.  This enforces ``degenerate_policies.constant_label`` and
    ``degenerate_policies.pair_any_all_positive`` at the outer-unit level: a
    degenerate publication must not drag or inflate the macro by an undefined AP.
    """
    groups: dict[Any, list] = defaultdict(list)
    for pub, lab, sc in zip(publications, labels, scores):
        groups[pub].append((int(bool(lab)), float(sc)))
    if not groups:
        return UNIDENTIFIABLE
    pub_aps: list[float] = []
    for pub in sorted(groups, key=str):
        labs = [t for t, _ in groups[pub]]
        scos = [s for _, s in groups[pub]]
        if len(set(labs)) <= 1:
            return UNIDENTIFIABLE
        pub_aps.append(_average_precision_numeric(labs, scos))
    return sum(pub_aps) / float(len(pub_aps))


# ---------------------------------------------------------------------------
# Plus-one permutation p-value
# ---------------------------------------------------------------------------
def permutation_p_value(real: float, null_stats: Sequence[float]) -> float:
    """p = (b + 1) / (B + 1), where b = #null stats >= real and B = #null stats
    (contract §13.3 item 5).  Guaranteed in [1/(B+1), 1]."""
    b = sum(1 for v in null_stats if v >= real)
    return (b + 1) / (len(null_stats) + 1)


def _group_blocks(publications: Sequence, labels: Sequence, scores: Sequence):
    groups: dict[Any, list] = defaultdict(list)
    for pub, lab, sc in zip(publications, labels, scores):
        groups[pub].append((int(bool(lab)), float(sc)))
    pub_ids = sorted(groups, key=str)
    label_blocks = [[t for t, _ in groups[p]] for p in pub_ids]
    score_blocks = [[s for _, s in groups[p]] for p in pub_ids]
    return pub_ids, label_blocks, score_blocks


def _flatten(blocks: Sequence[Sequence]) -> list:
    return [v for b in blocks for v in b]


def permutation_test(publications: Sequence, labels: Sequence, scores: Sequence,
                     seed: int = 0, n_perm: int = 1000) -> dict[str, Any]:
    """Publication-level block permutation test for the primary metric.

    Blocks are the publications (outer unit).  A permutation pairs each
    label-block with a score-block of the *same* size chosen uniformly at random
    within that size class (random matching / block permutation).  This keeps
    every publication's rows intact as a block and never mixes rows across
    blocks (``no mixed blocks``); publications whose size is unique cannot be
    permuted and remain as observed, which is the correct exchangeable-null
    treatment.  Within each permutation the publication-macro AUPRC is computed
    and the null distribution is built; the p-value is (b+1)/(B+1).

    Deterministic given ``seed`` (Python's ``random.Random``).
    """
    pub_ids, label_blocks, score_blocks = _group_blocks(publications, labels, scores)
    if not pub_ids:
        return {"statistic": UNIDENTIFIABLE, "p_value": None, "b": 0,
                "n_perm": n_perm, "null": []}
    real = publication_macro_auprc(publications, labels, scores)

    # group block indices by size so we only permute within equal-size classes
    size_classes: dict[int, list[int]] = defaultdict(list)
    for idx, lb in enumerate(label_blocks):
        size_classes[len(lb)].append(idx)

    rng = random.Random(seed)
    null_stats: list[float] = []
    b = 0
    for _ in range(n_perm):
        perm_score_blocks = [None] * len(score_blocks)
        for size, idxs in size_classes.items():
            # random matching: permute the score-blocks among this size class
            perm_idxs = idxs[:]
            rng.shuffle(perm_idxs)
            for orig, dest in zip(idxs, perm_idxs):
                perm_score_blocks[dest] = score_blocks[orig]
        perm_labels = _flatten(label_blocks)
        perm_scores = _flatten(perm_score_blocks)
        perm_pubs = [p for p, lb in zip(pub_ids, label_blocks) for _ in lb]
        stat = publication_macro_auprc(perm_pubs, perm_labels, perm_scores)
        null_stats.append(stat if not is_unidentifiable(stat) else float("nan"))
        if not is_unidentifiable(stat) and stat >= real:
            b += 1

    return {
        "statistic": real,
        "p_value": permutation_p_value(real, [s for s in null_stats if not math.isnan(s)]),
        "b": b,
        "n_perm": n_perm,
        "null": sorted(null_stats),
    }


# ---------------------------------------------------------------------------
# Cluster (publication-level) bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_publication_ci(per_pub_values: Sequence[float], seed: int = 0,
                             n_boot: int = 1000, alpha: float = 0.05,
                             statistic: Optional[Callable] = None) -> Any:
    """Cluster bootstrap confidence interval for a scalar summarised over
    publications (the outer resampling unit).

    Resamples the publication-level values with replacement (n_pub draws) and
    computes the percentile CI at ``alpha``.  Resampling whole publications keeps
    blocks intact (no mixed blocks).

    Confirmatory rule (``degenerate_policies.publication_lt_3``): with fewer than
    3 distinct publications no confirmatory CI can be produced -> UNIDENTIFIABLE.
    """
    values = [float(v) for v in per_pub_values]
    n_pub = len(values)
    if n_pub < 3:
        return UNIDENTIFIABLE
    statistic = statistic or statistics.mean
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n_pub)] for _ in range(n_pub)]
        boots.append(float(statistic(sample)))
    boots.sort()
    lo = boots[int(round((alpha / 2.0) * (n_boot - 1)))]
    hi = boots[int(round((1.0 - alpha / 2.0) * (n_boot - 1)))]
    return {
        "lower": lo,
        "upper": hi,
        "statistic": float(statistic(values)),
        "n_pub": n_pub,
        "n_boot": n_boot,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Assembled primary endpoint evaluation
# ---------------------------------------------------------------------------
def evaluate_primary(publications: Sequence, labels: Sequence, scores: Sequence,
                     seed: int = 0, n_perm: int = 1000, n_boot: int = 1000) -> dict[str, Any]:
    """Run the full frozen primary-endpoint evaluation (publication-macro AUPRC),
    returning metric + confirmatory CI + permutation p.

    Degenerate handling:
      * global constant label                        -> metric UNIDENTIFIABLE
      * any publication constant                     -> metric UNIDENTIFIABLE
      * < 3 distinct publications                    -> CI UNIDENTIFIABLE (no confirmatory CI)
      * permutation p computed with (b+1)/(B+1)
    """
    pubs = [str(p) for p in publications]
    labs = [int(bool(t)) for t in labels]
    scores = [float(s) for s in scores]

    result: dict[str, Any] = {}
    if len(set(labs)) <= 1:
        result["metric"] = UNIDENTIFIABLE
    else:
        result["metric"] = publication_macro_auprc(pubs, labs, scores)

    n_pub = len(set(pubs))
    if n_pub < 3:
        result["ci"] = UNIDENTIFIABLE
    else:
        # per-publication AP values for the cluster bootstrap
        groups: dict[str, list] = defaultdict(list)
        for p, l, s in zip(pubs, labs, scores):
            groups[p].append((l, s))
        per_pub_ap: list[float] = []
        for p in groups:
            gl = [t for t, _ in groups[p]]
            gs = [s for _, s in groups[p]]
            if len(set(gl)) <= 1:
                per_pub_ap = [UNIDENTIFIABLE]
                break
            per_pub_ap.append(_average_precision_numeric(gl, gs))
        if any(is_unidentifiable(v) for v in per_pub_ap):
            result["ci"] = UNIDENTIFIABLE
        else:
            result["ci"] = bootstrap_publication_ci(per_pub_ap, seed=seed, n_boot=n_boot)

    perm = permutation_test(pubs, labs, scores, seed=seed, n_perm=n_perm)
    result["permutation"] = perm
    return result
