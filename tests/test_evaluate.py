"""Tests for the C5.4 evaluation protocol (reactflow.evaluate).

These lock down the closed-form behaviour promised by the C5 plan:

* per-tier structure F1/MCC with *macro* (per-structure mean) and *micro*
  (pooled confusion) aggregation deliberately disagreeing;
* the generalization gap ``F1(in_clan) - F1(novel_clan)`` in both aggregations;
* Spearman as Pearson-of-ranks (monotone, non-linear-monotone, tied, constant);
* calibrated MAE absorbing a positive affine rescale of the prediction;
* ECE/MCE for perfect and known-miscalibrated inputs plus input validation;
* the honest cited-vs-local comparison table never fusing the two columns.
"""

import math

import pytest

from reactflow.constraints import dotbracket_to_matrix
from reactflow.evaluate import (
    ComparisonRow,
    StructurePrediction,
    build_comparison_table,
    calibrated_mae,
    expected_calibration_error,
    generalization_gap,
    reactivity_metrics,
    render_comparison_markdown,
    spearman_correlation,
    structure_distance_bin_metrics_by_tier,
    structure_metrics_by_tier,
)


# ``((..))`` pairs positions (0,5) and (1,4); ``(....)`` keeps only (0,5).
TARGET = dotbracket_to_matrix("((..))")
PRED_PERFECT = dotbracket_to_matrix("((..))")
PRED_PARTIAL = dotbracket_to_matrix("(....)")  # tp=1, fp=0, fn=1 vs TARGET


def _mixed_predictions():
    """Predictions whose macro and micro tier scores differ on purpose."""

    return [
        StructurePrediction(PRED_PERFECT, TARGET, "in_clan"),
        StructurePrediction(PRED_PERFECT, TARGET, "in_clan"),
        StructurePrediction(PRED_PARTIAL, TARGET, "cross_clan"),
        StructurePrediction(PRED_PERFECT, TARGET, "novel_clan"),
        StructurePrediction(PRED_PARTIAL, TARGET, "novel_clan"),
    ]


# --------------------------------------------------------------------------- #
# structure metrics per tier
# --------------------------------------------------------------------------- #
def test_structure_metrics_by_tier_macro_and_micro():
    metrics = structure_metrics_by_tier(_mixed_predictions())
    assert set(metrics) == {"in_clan", "cross_clan", "novel_clan"}

    in_clan = metrics["in_clan"]
    assert in_clan.count == 2
    assert in_clan.mean_f1 == pytest.approx(1.0)
    assert in_clan.mean_mcc == pytest.approx(1.0)
    assert in_clan.micro_f1 == pytest.approx(1.0)
    assert in_clan.micro_mcc == pytest.approx(1.0)

    cross = metrics["cross_clan"]
    assert cross.count == 1
    assert cross.mean_f1 == pytest.approx(2 / 3)  # 2*1/(2*1+0+1)
    assert cross.micro_f1 == pytest.approx(2 / 3)  # single structure: macro == micro
    assert cross.mean_mcc == pytest.approx(13 / math.sqrt(364))

    novel = metrics["novel_clan"]
    assert novel.count == 2
    assert novel.mean_f1 == pytest.approx((1.0 + 2 / 3) / 2)  # macro = 5/6
    assert novel.micro_f1 == pytest.approx(6 / 7)  # pooled tp=3, fn=1 -> 6/7
    assert abs(novel.mean_f1 - novel.micro_f1) > 1e-6  # macro and micro really differ


def test_structure_metrics_all_negative_structure_is_zero():
    empty = dotbracket_to_matrix("....")  # no pairs anywhere
    metrics = structure_metrics_by_tier([StructurePrediction(empty, empty, "in_clan")])
    # tp=fp=fn=0 -> F1 denominator 0 -> 0.0; MCC denominator 0 -> 0.0
    assert metrics["in_clan"].mean_f1 == 0.0
    assert metrics["in_clan"].mean_mcc == 0.0
    assert metrics["in_clan"].micro_f1 == 0.0
    assert metrics["in_clan"].micro_mcc == 0.0


def test_structure_metrics_accepts_benchmark_tier_and_rejects_empty_label():
    metrics = structure_metrics_by_tier([StructurePrediction(PRED_PERFECT, TARGET, "archiveII")])
    assert metrics["archiveII"].mean_f1 == pytest.approx(1.0)

    with pytest.raises(ValueError, match="tier label"):
        structure_metrics_by_tier([StructurePrediction(PRED_PERFECT, TARGET, "")])


def _matrix(size, pairs):
    """Return a symmetric pair matrix for distance-bin tests."""

    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i, j in pairs:
        matrix[i][j] = 1
        matrix[j][i] = 1
    return matrix


def test_structure_distance_bin_metrics_by_tier_separates_long_range_errors():
    target = _matrix(30, [(1, 10), (0, 29)])
    predicted = _matrix(30, [(1, 10)])

    metrics = structure_distance_bin_metrics_by_tier(
        [StructurePrediction(predicted, target, "novel_clan")],
        bins=(("short", 1, 11), ("long", 24, None)),
    )

    short = metrics["novel_clan"]["short"]
    long = metrics["novel_clan"]["long"]
    assert short.count == 1
    assert short.mean_f1 == pytest.approx(1.0)
    assert short.micro_f1 == pytest.approx(1.0)
    assert long.mean_f1 == pytest.approx(0.0)
    assert long.micro_f1 == pytest.approx(0.0)
    assert long.mean_mcc == pytest.approx(0.0)


def test_structure_distance_bin_metrics_validates_bins_and_labels():
    with pytest.raises(ValueError, match="min_distance"):
        structure_distance_bin_metrics_by_tier(
            [StructurePrediction(PRED_PERFECT, TARGET, "in_clan")],
            bins=(("bad", 0, 1),),
        )
    with pytest.raises(ValueError, match="distance-bin label"):
        structure_distance_bin_metrics_by_tier(
            [StructurePrediction(PRED_PERFECT, TARGET, "in_clan")],
            bins=(("", 1, 2),),
        )


# --------------------------------------------------------------------------- #
# generalization gap
# --------------------------------------------------------------------------- #
def test_generalization_gap_macro_and_micro():
    metrics = structure_metrics_by_tier(_mixed_predictions())

    macro = generalization_gap(metrics)
    assert macro.in_clan_f1 == pytest.approx(1.0)
    assert macro.novel_clan_f1 == pytest.approx(5 / 6)
    assert macro.gap == pytest.approx(1 / 6)

    micro = generalization_gap(metrics, use_micro=True)
    assert micro.in_clan_f1 == pytest.approx(1.0)
    assert micro.novel_clan_f1 == pytest.approx(6 / 7)
    assert micro.gap == pytest.approx(1 - 6 / 7)


def test_generalization_gap_requires_both_tiers():
    metrics = structure_metrics_by_tier(
        [StructurePrediction(PRED_PERFECT, TARGET, "in_clan")]
    )
    with pytest.raises(ValueError, match="novel_clan"):
        generalization_gap(metrics)


# --------------------------------------------------------------------------- #
# spearman rank correlation
# --------------------------------------------------------------------------- #
def test_spearman_monotone_and_ties():
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    # rank-monotone but non-linear -> still 1.0 (rank, not value, correlation)
    assert spearman_correlation([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # tied predictor: average-tie ranks give a closed-form 4.5/sqrt(22.5)
    assert spearman_correlation([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(
        4.5 / math.sqrt(22.5)
    )


def test_spearman_constant_and_validation():
    assert spearman_correlation([1, 1, 1], [1, 2, 3]) == 0.0  # zero-variance ranks
    with pytest.raises(ValueError, match="same length"):
        spearman_correlation([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="no finite"):
        spearman_correlation([float("nan")], [1.0])


# --------------------------------------------------------------------------- #
# calibrated MAE
# --------------------------------------------------------------------------- #
def test_calibrated_mae_absorbs_positive_affine_rescale():
    target = [1.0, 2.0, 3.0, 4.0]
    predicted = [2.0 * t + 1.0 for t in target]  # perfectly affine in target
    assert calibrated_mae(predicted, target) == pytest.approx(0.0, abs=1e-9)

    rescaled = [5.0 * p - 2.0 for p in predicted]  # another positive affine map
    assert calibrated_mae(rescaled, target) == pytest.approx(
        calibrated_mae(predicted, target), abs=1e-9
    )


def test_calibrated_mae_accepts_explicit_weights():
    target = [1.0, 2.0, 3.0, 4.0]
    predicted = [2.0 * t + 1.0 for t in target]
    assert calibrated_mae(predicted, target, weights=[1.0, 1.0, 1.0, 1.0]) == pytest.approx(
        0.0, abs=1e-9
    )


# --------------------------------------------------------------------------- #
# calibration error (ECE / MCE)
# --------------------------------------------------------------------------- #
def test_ece_perfect_calibration_is_zero():
    report = expected_calibration_error([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])
    assert report.ece == pytest.approx(0.0)
    assert report.mce == pytest.approx(0.0)
    assert report.total == 4
    assert report.n_bins == 10


def test_ece_known_miscalibration():
    # 0.9 -> bin 9 (acc 1.0, gap 0.1); 0.1 -> bin 1 (acc 0.0, gap 0.1)
    report = expected_calibration_error([0.9, 0.1], [1, 0])
    assert report.ece == pytest.approx(0.1)
    assert report.mce == pytest.approx(0.1)


def test_ece_same_bin_averages_accuracy():
    # both 0.8 land in bin 8; confidence 0.8, accuracy 0.5 -> gap 0.3
    report = expected_calibration_error([0.8, 0.8], [1, 0])
    assert report.ece == pytest.approx(0.3)
    assert report.mce == pytest.approx(0.3)


def test_ece_probability_one_lands_in_last_bin():
    report = expected_calibration_error([1.0], [1], n_bins=5)
    assert report.n_bins == 5
    assert len(report.bins) == 5
    assert report.bins[-1].count == 1  # p == 1 closed into the final bin
    assert report.ece == pytest.approx(0.0)


def test_ece_validation():
    with pytest.raises(ValueError, match="same length"):
        expected_calibration_error([0.5], [1, 0])
    with pytest.raises(ValueError, match="positive"):
        expected_calibration_error([0.5], [1], n_bins=0)
    with pytest.raises(ValueError, match="at least one"):
        expected_calibration_error([], [])
    with pytest.raises(ValueError, match="outside"):
        expected_calibration_error([1.5], [1])
    with pytest.raises(ValueError, match="outside"):
        expected_calibration_error([float("nan")], [1])
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error([0.5], [2])


# --------------------------------------------------------------------------- #
# reactivity metrics bundle
# --------------------------------------------------------------------------- #
def test_reactivity_metrics_bundle_and_validation():
    target = [0.1, 0.2, 0.5, 0.9]
    predicted = [2.0 * t + 0.1 for t in target]
    metrics = reactivity_metrics(predicted, target)
    assert metrics.count == 4
    assert metrics.pearson == pytest.approx(1.0)
    assert metrics.spearman == pytest.approx(1.0)
    assert metrics.calibrated_mae == pytest.approx(0.0, abs=1e-9)

    weighted = reactivity_metrics(predicted, target, weights=[1.0, 1.0, 1.0, 1.0])
    assert weighted.pearson == pytest.approx(1.0)

    with pytest.raises(ValueError, match="same length"):
        reactivity_metrics([1.0, 2.0], [1.0])


def test_reactivity_metrics_counts_finite_only():
    metrics = reactivity_metrics([1.0, 2.0, float("nan")], [1.0, 2.0, 3.0])
    assert metrics.count == 2


# --------------------------------------------------------------------------- #
# honest cited-vs-local comparison
# --------------------------------------------------------------------------- #
def test_comparison_table_keeps_cited_and_local_separate():
    cited = {
        "viral_mRNA": (0.73, "eFold 10.1126/sciadv.adz4967"),
        "lncRNA": (0.44, "eFold 10.1126/sciadv.adz4967"),
    }
    local = {"viral_mRNA": 0.55, "synthetic_pilot": 0.40}
    table = build_comparison_table(cited, local)

    rows = {row.test_set: row for row in table.rows}
    assert [row.test_set for row in table.rows] == sorted(rows)  # sorted by name

    both = rows["viral_mRNA"]
    assert both.cited_f1 == 0.73  # cited kept verbatim, never averaged with local
    assert both.local_f1 == 0.55
    assert both.status == "both"

    pending = rows["lncRNA"]
    assert pending.cited_f1 == 0.44
    assert pending.local_f1 is None
    assert pending.status == "local-pending"

    local_only = rows["synthetic_pilot"]
    assert local_only.cited_f1 is None
    assert local_only.local_f1 == 0.40
    assert local_only.status == "local-only"


def test_comparison_row_empty_status():
    row = ComparisonRow(test_set="unmeasured", cited_f1=None, citation=None, local_f1=None)
    assert row.status == "empty"


def test_render_comparison_markdown_marks_pending_and_missing():
    cited = {"lncRNA": (0.44, "eFold")}
    local = {"synthetic_pilot": 0.40}
    md = render_comparison_markdown(build_comparison_table(cited, local))

    assert "| Test set |" in md  # header row present
    assert "0.440" in md  # cited number formatted
    assert "0.400" in md  # local number formatted
    assert "pending" in md  # lncRNA has no local recompute yet
    assert "—" in md  # synthetic_pilot has no cited number
    assert "local-pending" in md
    assert "local-only" in md
