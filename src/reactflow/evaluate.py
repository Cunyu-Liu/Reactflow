"""Evaluation protocol for the C5 eFold head-to-head (cycle C5.4).

This module turns raw model outputs into the honest, comparable, differentiated
numbers the C5 plan promises.  It has four responsibilities:

1. **Structure accuracy per generalization tier.**  Pair-level F1 and MCC are
   aggregated separately for the ``in_clan`` / ``cross_clan`` / ``novel_clan``
   tiers so the *generalization gap* can be read off directly.  Both the
   *macro* average (mean of per-structure scores, the convention eFold and the
   RNA secondary-structure literature use) and the *micro* average (pooled
   confusion counts) are reported.

2. **Reactivity quality.**  Two complementary views:

   * **shape** via Pearson and Spearman rank correlation -- both are invariant
     to a positive affine rescaling of the prediction, so they measure *profile
     shape* without needing calibration (see
     :func:`reactflow.symbolic.verify_pearson_affine_invariance`);
   * **magnitude** via MAE *after* a weighted affine calibration
     ``r = alpha*rhat + gamma`` is fit and applied -- magnitude only becomes
     meaningful once the scale/offset are matched.

3. **Uncertainty calibration.**  The Expected and Maximum Calibration Error
   (ECE / MCE) over reliability bins.

4. **Honest head-to-head bookkeeping.**  A two-column comparison table that keeps
   **cited** numbers (from an external DOI) and **locally recomputed** numbers in
   *separate columns that are never merged or averaged*.  Cells with no local
   number yet are marked ``pending`` rather than back-filled -- this is the
   project's standing honesty red line.

Everything here is pure standard library; ``import reactflow.evaluate`` pulls in
no numpy/torch.

Mathematical summary
--------------------
* F1  = ``2 TP / (2 TP + FP + FN)``.
* MCC = ``(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))``.
* Spearman rho = Pearson correlation of the *average-tie ranks* of the two
  series.
* ECE = ``sum_b (|B_b|/N) |acc(B_b) - conf(B_b)|`` where ``conf`` is the mean
  predicted probability and ``acc`` the empirical positive rate in bin ``b``.
* generalization gap = ``F1(in_clan) - F1(novel_clan)``; smaller is better.

Complexity
----------
Structure metrics are ``O(sum_k L_k^2)`` over ``k`` structures of length
``L_k``; reactivity metrics ``O(L log L)`` (rank sort); ECE ``O(N + M)`` for
``N`` observations and ``M`` bins.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from reactflow.metrics import mean_absolute_error, pair_confusion
from reactflow.reactivity import fit_weighted_affine_calibration, weighted_pearson

TIERS: Tuple[str, ...] = ("in_clan", "cross_clan", "novel_clan")
DEFAULT_DISTANCE_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("short", 1, 11),
    ("medium", 12, 23),
    ("long", 24, None),
)


# --------------------------------------------------------------------------- #
# structure accuracy per tier
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StructurePrediction:
    """One predicted/target pair matrix tagged with a generalization tier.

    Attributes:
        predicted: predicted pair matrix (upper-triangle cells ``> 0.5`` count as
            a predicted pair).
        target: ground-truth pair matrix.
        tier: tier label.  The standard OOD protocol uses :data:`TIERS`
            (``in_clan`` / ``cross_clan`` / ``novel_clan``), while real benchmark
            runs may also use names such as ``archiveII`` or ``lncRNA``.

    Complexity: O(L^2) matrix storage per prediction.
    """

    predicted: Sequence[Sequence[float]]
    target: Sequence[Sequence[float]]
    tier: str


@dataclass(frozen=True)
class TierMetrics:
    """Aggregated structure metrics for a single tier.

    ``mean_f1`` / ``mean_mcc`` are macro averages (mean of per-structure scores).
    ``micro_f1`` / ``micro_mcc`` are computed from the pooled confusion counts.

    Complexity: O(1) summary storage.
    """

    tier: str
    count: int
    mean_f1: float
    mean_mcc: float
    micro_f1: float
    micro_mcc: float


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Return ``2TP/(2TP+FP+FN)`` (0 when undefined).

    Complexity: O(1).
    """

    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    """Return the Matthews correlation coefficient from counts (0 when undefined).

    Complexity: O(1).
    """

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator


def structure_metrics_by_tier(
    predictions: Sequence[StructurePrediction],
) -> Dict[str, TierMetrics]:
    """Aggregate F1/MCC per generalization tier (macro and micro).

    For each tier the macro scores average the per-structure F1/MCC, while the
    micro scores are computed once from the pooled ``TP/FP/FN/TN`` counts.  A
    tier with no predictions is simply absent from the returned mapping.

    Complexity: ``O(sum_k L_k^2)``.
    """

    macro_f1: Dict[str, List[float]] = {}
    macro_mcc: Dict[str, List[float]] = {}
    pooled: Dict[str, Dict[str, int]] = {}
    for item in predictions:
        if not item.tier:
            raise ValueError("tier label must be non-empty")
        confusion = pair_confusion(item.predicted, item.target)
        tp, fp, fn, tn = confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]
        macro_f1.setdefault(item.tier, []).append(_f1_from_counts(tp, fp, fn))
        macro_mcc.setdefault(item.tier, []).append(_mcc_from_counts(tp, fp, fn, tn))
        acc = pooled.setdefault(item.tier, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        acc["tp"] += tp
        acc["fp"] += fp
        acc["fn"] += fn
        acc["tn"] += tn

    result: Dict[str, TierMetrics] = {}
    for tier, f1s in macro_f1.items():
        counts = pooled[tier]
        result[tier] = TierMetrics(
            tier=tier,
            count=len(f1s),
            mean_f1=sum(f1s) / len(f1s),
            mean_mcc=sum(macro_mcc[tier]) / len(macro_mcc[tier]),
            micro_f1=_f1_from_counts(counts["tp"], counts["fp"], counts["fn"]),
            micro_mcc=_mcc_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"]),
        )
    return result


def _pair_confusion_by_distance(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    min_distance: int,
    max_distance: Optional[int],
) -> Dict[str, int]:
    """Compute pair confusion over a base-pair distance slice.

    Formula: restrict the usual upper-triangle candidate set to
    ``D = {(i,j): i < j, min_distance <= j-i <= max_distance}``, with
    ``max_distance=None`` denoting an unbounded upper tail.  Counts are then
    ``TP = |P_hat cap P cap D|``, ``FP = |(P_hat \\ P) cap D|``,
    ``FN = |(P \\ P_hat) cap D|`` and ``TN = |D \\ (P_hat union P)|``.
    Complexity: O(L^2).
    """

    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    if max_distance is not None and max_distance < min_distance:
        raise ValueError("max_distance must be >= min_distance")
    if len(predicted) != len(target) or any(len(a) != len(b) for a, b in zip(predicted, target)):
        raise ValueError("predicted and target matrices must have the same shape")
    size = len(predicted)
    tp = fp = fn = tn = 0
    for i in range(size):
        for j in range(i + 1, size):
            distance = j - i
            if distance < min_distance or (max_distance is not None and distance > max_distance):
                continue
            pred = float(predicted[i][j]) > 0.5
            truth = float(target[i][j]) > 0.5
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def structure_distance_bin_metrics_by_tier(
    predictions: Sequence[StructurePrediction],
    *,
    bins: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_DISTANCE_BINS,
) -> Dict[str, Dict[str, TierMetrics]]:
    """Aggregate F1/MCC by tier and base-pair distance bin.

    The default bins are ``short: 1..11``, ``medium: 12..23`` and
    ``long: >=24``.  They expose whether novel-family errors are driven by local
    stems or long-range contacts, which is critical for cross-family RNA
    secondary-structure claims.

    Formula: for every tier ``t`` and bin ``b=[d_min,d_max]``, compute per-sample
    F1/MCC from distance-filtered confusion counts and average them for macro
    scores; micro scores use pooled counts over the same distance slice.
    Complexity: O(B * sum_k L_k^2), where B is the number of bins.
    """

    macro_f1: Dict[str, Dict[str, List[float]]] = {}
    macro_mcc: Dict[str, Dict[str, List[float]]] = {}
    pooled: Dict[str, Dict[str, Dict[str, int]]] = {}
    for label, min_distance, max_distance in bins:
        if not label:
            raise ValueError("distance-bin label must be non-empty")
        for item in predictions:
            if not item.tier:
                raise ValueError("tier label must be non-empty")
            confusion = _pair_confusion_by_distance(
                item.predicted,
                item.target,
                min_distance=min_distance,
                max_distance=max_distance,
            )
            tp, fp, fn, tn = confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]
            macro_f1.setdefault(item.tier, {}).setdefault(label, []).append(_f1_from_counts(tp, fp, fn))
            macro_mcc.setdefault(item.tier, {}).setdefault(label, []).append(_mcc_from_counts(tp, fp, fn, tn))
            acc = pooled.setdefault(item.tier, {}).setdefault(label, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            acc["tp"] += tp
            acc["fp"] += fp
            acc["fn"] += fn
            acc["tn"] += tn

    result: Dict[str, Dict[str, TierMetrics]] = {}
    for tier, bins_by_label in macro_f1.items():
        result[tier] = {}
        for label, f1s in bins_by_label.items():
            counts = pooled[tier][label]
            result[tier][label] = TierMetrics(
                tier=f"{tier}:{label}",
                count=len(f1s),
                mean_f1=sum(f1s) / len(f1s),
                mean_mcc=sum(macro_mcc[tier][label]) / len(macro_mcc[tier][label]),
                micro_f1=_f1_from_counts(counts["tp"], counts["fp"], counts["fn"]),
                micro_mcc=_mcc_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"]),
            )
    return result


@dataclass(frozen=True)
class GeneralizationGap:
    """The headline OOD gap ``F1(in_clan) - F1(novel_clan)``.

    A *smaller* gap means the model degrades less on entirely unseen clans, which
    is exactly ReactFlow's differentiation claim against eFold.

    Complexity: O(1) summary storage.
    """

    in_clan_f1: float
    novel_clan_f1: float
    gap: float


def generalization_gap(
    tier_metrics: Mapping[str, TierMetrics],
    *,
    use_micro: bool = False,
) -> GeneralizationGap:
    """Compute the in-clan vs novel-clan F1 gap from :func:`structure_metrics_by_tier`.

    ``use_micro`` selects the pooled (micro) F1 instead of the macro average.
    Both required tiers must be present.

    Complexity: O(1).
    """

    for tier in ("in_clan", "novel_clan"):
        if tier not in tier_metrics:
            raise ValueError(f"tier {tier!r} missing; cannot compute generalization gap")
    attr = "micro_f1" if use_micro else "mean_f1"
    in_f1 = getattr(tier_metrics["in_clan"], attr)
    novel_f1 = getattr(tier_metrics["novel_clan"], attr)
    return GeneralizationGap(in_clan_f1=in_f1, novel_clan_f1=novel_f1, gap=in_f1 - novel_f1)


# --------------------------------------------------------------------------- #
# reactivity quality
# --------------------------------------------------------------------------- #
def _average_ranks(values: Sequence[float]) -> List[float]:
    """Return average-tie ranks (0-based fractional positions).

    Tied values receive the mean of the contiguous positions they occupy, which
    is the standard fractional ranking used by Spearman's rho.

    Complexity: O(L log L).
    """

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson correlation of the average-tie ranks.

    Non-finite pairs are dropped before ranking.  When either ranked series has
    zero variance (e.g. all-constant input) the correlation is undefined and 0.0
    is returned, mirroring :func:`reactflow.reactivity.weighted_pearson`.

    Complexity: O(L log L).
    """

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    valid = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if not valid:
        raise ValueError("no finite observations")
    xs = [a for a, _ in valid]
    ys = [b for _, b in valid]
    rank_x = _average_ranks(xs)
    rank_y = _average_ranks(ys)
    return weighted_pearson(rank_x, rank_y, [1.0] * len(valid))


def calibrated_mae(
    predicted: Sequence[float],
    target: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """MAE after a weighted affine calibration is fit and applied.

    The calibration ``r = alpha*rhat + gamma`` is fit by weighted least squares
    (:func:`reactflow.reactivity.fit_weighted_affine_calibration`); the reported
    error is the (unweighted) mean absolute error of the calibrated prediction
    against the target over finite positions.

    Complexity: O(L).
    """

    if weights is None:
        weights = [1.0] * len(predicted)
    alpha, gamma = fit_weighted_affine_calibration(predicted, target, weights)
    calibrated = [alpha * float(p) + gamma for p in predicted]
    return mean_absolute_error(calibrated, target)


@dataclass(frozen=True)
class ReactivityMetrics:
    """Reactivity-profile quality: shape (correlations) + magnitude (calibrated MAE).

    Complexity: O(1) summary storage.
    """

    count: int
    pearson: float
    spearman: float
    calibrated_mae: float


def reactivity_metrics(
    predicted: Sequence[float],
    target: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> ReactivityMetrics:
    """Bundle Pearson, Spearman and calibrated MAE for a reactivity profile.

    Complexity: O(L log L).
    """

    if len(predicted) != len(target):
        raise ValueError("predicted and target must have the same length")
    if weights is None:
        weights = [1.0] * len(predicted)
    count = sum(
        1
        for p, t in zip(predicted, target)
        if math.isfinite(float(p)) and math.isfinite(float(t))
    )
    return ReactivityMetrics(
        count=count,
        pearson=weighted_pearson(predicted, target, weights),
        spearman=spearman_correlation(predicted, target),
        calibrated_mae=calibrated_mae(predicted, target, weights),
    )


# --------------------------------------------------------------------------- #
# calibration (ECE / MCE)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationBin:
    """One reliability-diagram bin.

    Complexity: O(1) summary storage.
    """

    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability summary over ``n_bins`` equal-width probability bins.

    Complexity: O(B) storage for B bins.
    """

    n_bins: int
    total: int
    ece: float
    mce: float
    bins: Tuple[CalibrationBin, ...]


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute ECE and MCE over equal-width reliability bins.

    Each observation is a predicted probability ``p_i in [0, 1]`` that a pair is
    present, paired with a binary outcome ``y_i in {0, 1}``.  Bin ``b`` collects
    the probabilities in ``[b/M, (b+1)/M)`` (the last bin is closed on the right
    so ``p = 1`` lands in it), and

        conf(b) = mean p_i over B_b,   acc(b) = mean y_i over B_b,
        ECE = sum_b (|B_b|/N) |acc(b) - conf(b)|,
        MCE = max_b |acc(b) - conf(b)|  (over non-empty bins).

    Complexity: O(N + M).
    """

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if not probabilities:
        raise ValueError("at least one observation is required")

    sum_conf = [0.0] * n_bins
    sum_acc = [0.0] * n_bins
    counts = [0] * n_bins
    total = 0
    for prob, label in zip(probabilities, labels):
        p = float(prob)
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"probability {prob!r} outside [0, 1]")
        y = int(label)
        if y not in (0, 1):
            raise ValueError(f"label {label!r} is not binary")
        index = min(int(p * n_bins), n_bins - 1)
        sum_conf[index] += p
        sum_acc[index] += y
        counts[index] += 1
        total += 1

    bins: List[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        lower = b / n_bins
        upper = (b + 1) / n_bins
        if counts[b] == 0:
            bins.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        confidence = sum_conf[b] / counts[b]
        accuracy = sum_acc[b] / counts[b]
        gap = abs(accuracy - confidence)
        ece += (counts[b] / total) * gap
        mce = max(mce, gap)
        bins.append(CalibrationBin(lower, upper, counts[b], confidence, accuracy))
    return CalibrationReport(n_bins=n_bins, total=total, ece=ece, mce=mce, bins=tuple(bins))


# --------------------------------------------------------------------------- #
# honest head-to-head comparison (cited vs local, never merged)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ComparisonRow:
    """One test set's cited-vs-local row.

    ``cited_f1`` comes from an external publication (with ``citation``); it is
    *never* combined with ``local_f1``.  ``local_f1 is None`` means ReactFlow has
    not yet been evaluated on this set -- reported as ``pending``, not guessed.

    Complexity: O(1) summary storage.
    """

    test_set: str
    cited_f1: Optional[float]
    citation: Optional[str]
    local_f1: Optional[float]

    @property
    def status(self) -> str:
        """Return the honesty status of this row.

        Complexity: O(1).
        """

        has_cited = self.cited_f1 is not None
        has_local = self.local_f1 is not None
        if has_cited and has_local:
            return "both"
        if has_cited and not has_local:
            return "local-pending"
        if has_local and not has_cited:
            return "local-only"
        return "empty"


@dataclass(frozen=True)
class ComparisonTable:
    """A collection of :class:`ComparisonRow` sorted by test-set name.

    Complexity: O(K) storage for K comparison rows.
    """

    rows: Tuple[ComparisonRow, ...]


def build_comparison_table(
    cited: Mapping[str, Tuple[float, str]],
    local: Mapping[str, float],
) -> ComparisonTable:
    """Assemble a head-to-head table keeping cited and local numbers separate.

    ``cited`` maps a test-set name to ``(f1, citation)`` reported by an external
    source; ``local`` maps a test-set name to a locally recomputed F1.  The two
    inputs are deliberately *separate arguments*: this function never adds,
    averages, or otherwise fuses a cited number with a local one -- it only lines
    them up in adjacent columns.  Test sets present in only one input still get a
    row (with the missing side left ``None``).

    Complexity: O(K log K) for ``K`` distinct test sets.
    """

    names = sorted(set(cited) | set(local))
    rows: List[ComparisonRow] = []
    for name in names:
        cited_entry = cited.get(name)
        if cited_entry is not None:
            cited_f1: Optional[float] = float(cited_entry[0])
            citation: Optional[str] = str(cited_entry[1])
        else:
            cited_f1 = None
            citation = None
        local_value = local.get(name)
        local_f1 = None if local_value is None else float(local_value)
        rows.append(
            ComparisonRow(test_set=name, cited_f1=cited_f1, citation=citation, local_f1=local_f1)
        )
    return ComparisonTable(rows=tuple(rows))


def render_comparison_markdown(table: ComparisonTable) -> str:
    """Render the comparison as a Markdown table with clearly separated columns.

    The cited column always shows its source; the local column shows the number
    or the literal ``pending`` so a reader can never mistake an unfilled cell for
    a real result.

    Complexity: O(K).
    """

    lines = [
        "| Test set | Cited F1 (source) | Local recompute F1 | Status |",
        "|---|---|---|---|",
    ]
    for row in table.rows:
        if row.cited_f1 is None:
            cited_cell = "—"
        else:
            source = row.citation or "cited"
            cited_cell = f"{row.cited_f1:.3f} ({source})"
        local_cell = "pending" if row.local_f1 is None else f"{row.local_f1:.3f}"
        lines.append(f"| {row.test_set} | {cited_cell} | {local_cell} | {row.status} |")
    return "\n".join(lines) + "\n"
