"""R4: unit tests for ``scripts/reactflow_delta/evaluate_v2.py``.

Covers the frozen endpoint_v2 evaluator (contract §13.2/R4, §13.3):

  * tied AP row-order invariance (tie-aware, sklearn-consistent)
  * constant label / pair-any all-positive -> UNIDENTIFIABLE (no number)
  * publication < 3 -> UNIDENTIFIABLE confirmatory CI (no mixed blocks)
  * plus-one permutation p = (b + 1) / (B + 1), p in [1/(B+1), 1]
  * signed vs absolute task handling
  * coverage with missing (never zero-filled)
  * cross-check vs scikit-learn reference (when importable) and a hand-verified
    reference reproducing sklearn's documented outputs (sklearn is NOT shipped
    in the required runtime, so the hand-verified reference is the fallback)
  * WMAE skill sanity (perfect model -> 1, constant-equal baseline -> 0)
  * determinism (same seed -> same result)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:  # sklearn is NOT shipped in the required runtime; detect availability
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

from scripts.reactflow_delta.evaluate_v2 import (  # noqa: E402
    UNIDENTIFIABLE,
    absolute_target,
    average_precision_tie_aware,
    bootstrap_publication_ci,
    brier_score,
    coverage,
    evaluate_primary,
    is_unidentifiable,
    kendall,
    permutation_p_value,
    permutation_test,
    publication_macro_auprc,
    spearman,
    wmae_skill,
)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Tied AP row-order invariance + sklearn-consistency
# ---------------------------------------------------------------------------
def test_ap_matches_sklearn_documented_example():
    # sklearn docs: average_precision_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.83
    ap = average_precision_tie_aware([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert isinstance(ap, float)
    assert _close(ap, 5.0 / 6.0, 1e-12)


def test_ap_perfect_ranking_is_one():
    assert _close(average_precision_tie_aware([1, 1, 0], [0.9, 0.7, 0.3]), 1.0, 1e-12)


def test_ap_tied_row_order_invariant():
    labels = [1, 0, 1, 0, 1, 1, 0, 0, 1]
    scores = [0.5, 0.5, 0.3, 0.9, 0.7, 0.7, 0.2, 0.1, 0.5]
    # several deterministic orderings of the same (label, tied-score) multiset
    import random
    rng = random.Random(42)
    aps = []
    for _ in range(200):
        idx = list(range(len(labels)))
        rng.shuffle(idx)
        labs = [labels[i] for i in idx]
        scos = [scores[i] for i in idx]
        aps.append(average_precision_tie_aware(labs, scos))
    assert all(_close(a, aps[0], 1e-12) for a in aps), f"tied AP order-dependent: {aps}"


def test_ap_constant_label_unidentifiable():
    assert average_precision_tie_aware([1, 1, 1], [0.9, 0.5, 0.1]) == UNIDENTIFIABLE
    assert average_precision_tie_aware([0, 0, 0], [0.9, 0.5, 0.1]) == UNIDENTIFIABLE


# ---------------------------------------------------------------------------
# Constant label / pair-any all-positive -> UNIDENTIFIABLE
# ---------------------------------------------------------------------------
def test_publication_macro_all_positive_publication_unidentifiable():
    # pair-any all-positive in one publication -> DEGENERATE -> UNIDENTIFIABLE
    pubs = ["p1", "p1", "p1", "p2", "p2"]
    labels = [1, 1, 1, 0, 1]  # p1 constant-positive
    scores = [0.9, 0.7, 0.5, 0.3, 0.8]
    assert publication_macro_auprc(pubs, labels, scores) == UNIDENTIFIABLE


def test_publication_macro_constant_negative_publication_unidentifiable():
    pubs = ["p1", "p1", "p2", "p2", "p2"]
    labels = [0, 0, 0, 1, 1]
    scores = [0.6, 0.4, 0.3, 0.8, 0.9]
    assert publication_macro_auprc(pubs, labels, scores) == UNIDENTIFIABLE


def test_publication_macro_auprc_normal():
    pubs = ["p1", "p1", "p1", "p2", "p2", "p2"]
    labels = [1, 0, 1, 1, 1, 0]
    scores = [0.9, 0.2, 0.7, 0.8, 0.6, 0.3]
    v = publication_macro_auprc(pubs, labels, scores)
    assert isinstance(v, float) and 0.0 <= v <= 1.0
    # equals the mean of the two per-publication APs
    ap1 = average_precision_tie_aware([1, 0, 1], [0.9, 0.2, 0.7])
    ap2 = average_precision_tie_aware([1, 1, 0], [0.8, 0.6, 0.3])
    assert _close(v, (ap1 + ap2) / 2.0, 1e-12)


# ---------------------------------------------------------------------------
# publication < 3 -> UNIDENTIFIABLE confirmatory CI (no mixed blocks)
# ---------------------------------------------------------------------------
def test_bootstrap_ci_lt3_publications_unidentifiable():
    assert bootstrap_publication_ci([0.5, 0.6]) == UNIDENTIFIABLE
    assert bootstrap_publication_ci([0.5]) == UNIDENTIFIABLE


def test_evaluate_primary_lt3_publications_ci_unidentifiable():
    pubs = ["a", "a", "b", "b", "b"]
    labels = [1, 0, 1, 1, 0]
    scores = [0.9, 0.2, 0.8, 0.6, 0.3]
    res = evaluate_primary(pubs, labels, scores, n_boot=200, n_perm=100)
    assert res["ci"] == UNIDENTIFIABLE
    assert isinstance(res["metric"], float)


def test_evaluate_primary_ge3_publications_ci_is_dict():
    pubs = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    labels = [1, 0, 1, 1, 1, 0, 0, 1, 1]
    scores = [0.9, 0.2, 0.7, 0.8, 0.6, 0.3, 0.4, 0.85, 0.75]
    res = evaluate_primary(pubs, labels, scores, n_boot=500, n_perm=200)
    assert isinstance(res["ci"], dict)
    assert res["ci"]["lower"] <= res["ci"]["upper"]
    assert isinstance(res["permutation"]["p_value"], float)


# ---------------------------------------------------------------------------
# Plus-one permutation: p = (b+1)/(B+1), p in [1/(B+1), 1]
# ---------------------------------------------------------------------------
def test_permutation_p_value_plus_one():
    null = [0.0, 0.5, 1.0]
    # real=1.0 -> b = count(null >= 1.0) = 1 -> (1+1)/(3+1) = 0.5
    assert _close(permutation_p_value(1.0, null), 0.5, 1e-12)
    # real=2.0 -> b=0 -> 1/(3+1)=0.25
    assert _close(permutation_p_value(2.0, null), 0.25, 1e-12)
    # real=-1.0 -> b=3 -> 4/4=1.0
    assert _close(permutation_p_value(-1.0, null), 1.0, 1e-12)


def test_permutation_p_value_bounds():
    for B in (3, 10, 99):
        null = [float(i) for i in range(B)]
        for real in (-1.0, B / 2.0, B + 1.0):
            p = permutation_p_value(real, null)
            assert 1.0 / (B + 1.0) <= p <= 1.0


def test_permutation_test_deterministic_and_bounded():
    pubs = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    labels = [1, 0, 1, 0] * 3
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.95, 0.05, 0.85, 0.15]
    r1 = permutation_test(pubs, labels, scores, seed=7, n_perm=200)
    r2 = permutation_test(pubs, labels, scores, seed=7, n_perm=200)
    assert r1["p_value"] == r2["p_value"]
    assert 1.0 / (200 + 1.0) <= r1["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Signed vs absolute handling
# ---------------------------------------------------------------------------
def test_absolute_target():
    assert absolute_target([-3, -1, 0, 2, 5]) == [3.0, 1.0, 0.0, 2.0, 5.0]


def test_absolute_task_treats_sign_by_magnitude():
    # signed_abs_same_rank: in the absolute task, -4 and +4 are the same rank
    signed = [-4.0, 4.0, 2.0, 1.0]
    pred = [3.0, 3.0, 1.0, 0.5]  # ranks of pred align with |signed|
    # |signed| = [4,4,2,1] -> ties -4/+4 at rank 2.5; pred ranks 2.5,2.5,1.5,1? not 1.0
    # use a pred exactly reflecting the absolute ranking: high mag -> high pred
    pred2 = [0.9, 0.9, 0.5, 0.2]
    r = spearman(absolute_target(signed), pred2)
    assert isinstance(r, float)
    assert _close(r, 1.0, 1e-9)
    # signed self-consistency
    assert _close(spearman(signed, signed), 1.0, 1e-12)


def test_signed_vs_absolute_wmae_target():
    y_signed = [-2.0, 3.0, 1.0, -4.0]
    y_abs = absolute_target(y_signed)  # [2,3,1,4]
    # a magnitude prediction on the absolute target
    pred = [2.0, 3.0, 1.0, 4.0]
    assert _close(wmae_skill(y_abs, pred), 1.0, 1e-12)  # perfect on absolute target


# ---------------------------------------------------------------------------
# Coverage: missing is not zero-filled
# ---------------------------------------------------------------------------
def test_coverage_missing_not_zero_filled():
    eligible = [True, True, True, True, False]
    predicted = [0.5, None, float("nan"), 0.0, 0.9]
    # eligible positions: indices 0..3 (index 4 not eligible)
    # covered: idx0=0.5 (finite), idx1=None (missing, not counted),
    #          idx2=nan (missing, not counted), idx3=0.0 (finite, counts as covered)
    c = coverage(eligible, predicted)
    assert _close(c, 2.0 / 4.0, 1e-12)
    # zero is a real (non-missing) prediction -> counts; missing does not
    assert coverage([True, True], [1.0, 2.0]) == 1.0
    assert coverage([True, True], [0.0, None]) == 0.5


def test_coverage_empty():
    assert coverage([], []) == 0.0
    assert coverage([True, False], [0.5]) == 1.0  # only one eligible


# ---------------------------------------------------------------------------
# WMAE skill sanity
# ---------------------------------------------------------------------------
def test_wmae_skill_perfect_model():
    y_true = [1.0, 2.0, 3.0]
    assert _close(wmae_skill(y_true, y_true), 1.0, 1e-12)


def test_wmae_skill_constant_equal_baseline_zero():
    y_true = [1.0, 2.0, 3.0]
    pred = [2.0, 2.0, 2.0]  # equals the constant (mean) baseline -> skill 0
    assert _close(wmae_skill(y_true, pred), 0.0, 1e-12)


def test_wmae_skill_better_than_baseline_positive():
    y_true = [0.0, 1.0, 2.0, 3.0]
    pred = [0.2, 0.9, 2.1, 2.9]  # close to truth -> positive skill
    s = wmae_skill(y_true, pred)
    assert isinstance(s, float) and 0.0 < s < 1.0


def test_wmae_skill_weights_respected():
    # weights concentrate the metric: zero-weight on a wrong prediction removes
    # its penalty, raising skill (reference is an explicit zero baseline here).
    y_true = [0.0, 10.0, 20.0]
    pred = [0.0, 5.0, 20.0]      # wrong only on the middle position
    ref = [0.0, 0.0, 0.0]        # explicit trivial baseline (WMAE_ref = 10)
    s_equal = wmae_skill(y_true, pred, weights=[1.0, 1.0, 1.0], reference=ref)
    s_drop = wmae_skill(y_true, pred, weights=[1.0, 0.0, 1.0], reference=ref)
    assert _close(s_drop, 1.0, 1e-12)          # middle error weighted out -> perfect
    assert 0.0 < s_equal < 1.0 and s_equal < s_drop  # equal weights keep the penalty


# ---------------------------------------------------------------------------
# Cross-check vs scikit-learn (optional) + hand-verified reference
# ---------------------------------------------------------------------------
def test_brier_hand_reference():
    # mean((y_true - y_score)^2)
    v = brier_score([0.0, 1.0, 1.0], [0.2, 0.7, 0.9])
    expected = (0.04 + 0.09 + 0.01) / 3.0
    assert _close(v, expected, 1e-12)


def test_spearman_hand_reference():
    # monotonic -> 1
    assert _close(spearman([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, 1e-12)
    # anti-monotonic -> -1
    assert _close(spearman([1, 2, 3], [3, 2, 1]), -1.0, 1e-12)


def test_kendall_sign():
    assert _close(kendall([1, 2, 3, 4], [1, 2, 3, 4]), 1.0, 1e-12)
    assert _close(kendall([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, 1e-12)


@pytest.mark.skipif(not HAVE_SKLEARN, reason="sklearn not installed in runtime")
def test_sklearn_ap_crosscheck():
    from sklearn.metrics import average_precision_score, brier_score_loss

    labels = [1, 0, 1, 0, 1, 1, 0, 0, 1]
    scores = [0.5, 0.5, 0.3, 0.9, 0.7, 0.7, 0.2, 0.1, 0.5]
    got = average_precision_tie_aware(labels, scores)
    ref = average_precision_score(labels, scores)
    assert _close(got, ref, 1e-9)

    yt = [0, 1, 1, 0, 1]
    ys = [0.1, 0.9, 0.8, 0.3, 0.4]
    assert _close(brier_score(yt, ys), brier_score_loss(yt, ys), 1e-12)


def test_sklearn_not_installed_is_documented():
    # In the required runtime sklearn is absent -> the hand-verified reference is
    # the authoritative cross-check; this test documents that the evaluator still
    # matches sklearn's documented output (0.83) without sklearn being present.
    assert _close(average_precision_tie_aware([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
                  5.0 / 6.0, 1e-12)
    assert HAVE_SKLEARN in (True, False)  # informational
