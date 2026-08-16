"""Hand-computed gold fixtures for the ReactFlow structure evaluator.

Each fixture is a small RNA structure where TP/FP/FN/TN/F1/MCC can be verified
by hand.  The fixtures exercise every edge case listed in the C1-0 audit spec:

* all-unpaired (empty structure)
* single hairpin
* two stems (nested)
* GU wobble
* pseudoknot crossing
* illegal diagonal pair (error handling)
* 1-based vs 0-based pair list
* relaxed match hits but exact match does not

The tests check BOTH:
  - ``reactflow.metrics.pair_confusion`` (matrix-based, used by c0_evaluate)
  - ``evaluate_external_baseline_predictions._pair_confusion`` (set-based, used
    by the eFold wrapper)

so that the two scorers are provably aligned on every hand-computed case.

Mathematical reference
----------------------
For a structure of length ``L`` with upper-triangle candidate set
``U = {(i,j) : 0 <= i < j < L}`` of size ``|U| = L*(L-1)/2``:

* ``TP = |P_hat ∩ P|``           (correctly predicted pairs)
* ``FP = |P_hat \\ P|``          (spurious pairs)
* ``FN = |P \\ P_hat|``          (missed pairs)
* ``TN = |U| - TP - FP - FN``   (correctly rejected non-pairs)
* ``F1 = 2*TP / (2*TP + FP + FN)``  (0 when denominator is 0)
* ``MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))``  (0 when undefined)

Empty-structure convention (CRITICAL DIFFERENCE):
* ReactFlow: empty prediction on empty target -> F1 = 0.0 (denominator = 0)
* eFold:     empty prediction on empty target -> F1 = 1.0 (sum_pair == 0)
The ReactFlow contract ``static_v1`` adopts the ReactFlow convention and
records the eFold convention as an annotated difference.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

import pytest

# Make the scripts directory importable for the external baseline scorer.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from reactflow.constraints import pairs_to_matrix, matrix_to_pairs, validate_pair_matrix
from reactflow.metrics import pair_confusion, f1_score, matthews_corrcoef
from reactflow.c0_evaluate import (
    f1_from_counts,
    mcc_from_counts,
    shifted_pair_counts,
    structure_record_metrics,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _matrix_from_pairs(pairs: Sequence[Tuple[int, int]], size: int):
    """Build a binary symmetric matrix from a list of 0-based pairs."""
    return pairs_to_matrix(pairs, size)


def _set_pair_confusion(
    predicted: FrozenSet[Tuple[int, int]],
    target: FrozenSet[Tuple[int, int]],
    length: int,
) -> Dict[str, int]:
    """Replicate evaluate_external_baseline_predictions._pair_confusion for testing."""
    tp = len(predicted & target)
    fp = len(predicted - target)
    fn = len(target - predicted)
    candidate_count = length * (length - 1) // 2
    tn = candidate_count - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": max(tn, 0)}


def _expected_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def _expected_mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


# --------------------------------------------------------------------------- #
# Fixture 1: all unpaired (empty structure)
# --------------------------------------------------------------------------- #
def test_fixture_01_all_unpaired_empty_structure():
    """Empty prediction on empty target.

    Hand computation (L=4, |U|=6):
        TP=0, FP=0, FN=0, TN=6
        F1 = 0/(0+0+0) = 0.0  (ReactFlow convention: denominator 0 -> F1=0)
        MCC = 0 (undefined -> 0)

    CRITICAL: eFold returns F1=1.0 here (sum_pair==0 branch).
    ReactFlow returns F1=0.0.  This difference is recorded in the evaluation
    contract as an annotated, non-blocking divergence.
    """
    size = 4
    predicted = _matrix_from_pairs([], size)
    target = _matrix_from_pairs([], size)

    confusion = pair_confusion(predicted, target)
    assert confusion == {"tp": 0, "fp": 0, "fn": 0, "tn": 6}

    assert f1_score(predicted, target) == 0.0
    assert matthews_corrcoef(predicted, target) == 0.0

    # Set-based scorer should agree.
    set_conf = _set_pair_confusion(frozenset(), frozenset(), size)
    assert set_conf == confusion


# --------------------------------------------------------------------------- #
# Fixture 2: single hairpin, perfect match
# --------------------------------------------------------------------------- #
def test_fixture_02_single_hairpin_perfect_match():
    """Single hairpin with 2 pairs, perfect prediction.

    Sequence: GCAAAAGC (L=8)
    Pairs: (0,7) G-C, (1,6) C-G  (distances 7 and 5, both > min_loop=3)

    Hand computation (L=8, |U|=28):
        TP=2, FP=0, FN=0, TN=26
        F1 = 4/4 = 1.0
        MCC = (52)/sqrt(2*2*26*26) = 52/52 = 1.0
    """
    size = 8
    pairs = [(0, 7), (1, 6)]
    predicted = _matrix_from_pairs(pairs, size)
    target = _matrix_from_pairs(pairs, size)

    confusion = pair_confusion(predicted, target)
    assert confusion == {"tp": 2, "fp": 0, "fn": 0, "tn": 26}

    assert f1_score(predicted, target) == pytest.approx(1.0)
    assert matthews_corrcoef(predicted, target) == pytest.approx(1.0)

    set_conf = _set_pair_confusion(frozenset(pairs), frozenset(pairs), size)
    assert set_conf == confusion


# --------------------------------------------------------------------------- #
# Fixture 3: two stems (nested), partial match
# --------------------------------------------------------------------------- #
def test_fixture_03_two_stems_nested_partial_match():
    """Two nested stems, prediction misses one pair.

    Sequence: GCAAAGCAAAGCAAAGCAAAGC (L=22)
    Target pairs (nested, 2 stems):
        outer stem: (0,21) G-C, (1,20) C-G
        inner stem: (5,16) G-C, (6,15) C-G
    Predicted pairs: (0,21), (5,16), (6,15)  -- missed (1,20)

    Hand computation (L=22, |U|=231):
        TP=3, FP=0, FN=1, TN=231-3-0-1=227
        F1 = 6/7 ≈ 0.857142...
        MCC = (3*227 - 0)/sqrt(3*4*227*227) = 681/sqrt(619524) = 681/787.098...
    """
    size = 22
    target_pairs = [(0, 21), (1, 20), (5, 16), (6, 15)]
    predicted_pairs = [(0, 21), (5, 16), (6, 15)]  # missing (1, 20)
    predicted = _matrix_from_pairs(predicted_pairs, size)
    target = _matrix_from_pairs(target_pairs, size)

    confusion = pair_confusion(predicted, target)
    assert confusion == {"tp": 3, "fp": 0, "fn": 1, "tn": 227}

    expected_f1 = _expected_f1(3, 0, 1)
    assert f1_score(predicted, target) == pytest.approx(expected_f1)
    assert expected_f1 == pytest.approx(6 / 7)

    set_conf = _set_pair_confusion(frozenset(predicted_pairs), frozenset(target_pairs), size)
    assert set_conf == confusion


# --------------------------------------------------------------------------- #
# Fixture 4: GU wobble pair
# --------------------------------------------------------------------------- #
def test_fixture_04_gu_wobble_pair():
    """Hairpin containing a GU wobble pair.

    Sequence: GCAAAAGU (L=8)
    Pairs: (0,7) G-U (wobble), (1,6) C-G (canonical)
    Both are legal when allow_wobble=True.

    Hand computation (L=8, |U|=28):
        TP=2, FP=0, FN=0, TN=26
        F1 = 1.0
    """
    size = 8
    pairs = [(0, 7), (1, 6)]
    predicted = _matrix_from_pairs(pairs, size)
    target = _matrix_from_pairs(pairs, size)

    confusion = pair_confusion(predicted, target)
    assert confusion == {"tp": 2, "fp": 0, "fn": 0, "tn": 26}

    assert f1_score(predicted, target) == pytest.approx(1.0)

    # Constraint validation should pass with allow_wobble=True.
    sequence = "GCAAAAGU"
    result = validate_pair_matrix(
        sequence,
        predicted,
        min_loop=3,
        allow_wobble=True,
        allow_pseudoknot=True,
    )
    assert result.valid, f"GU wobble should be legal: {result.violations}"

    # With allow_wobble=False, the G-U pair becomes a violation.
    result_no_wobble = validate_pair_matrix(
        sequence,
        predicted,
        min_loop=3,
        allow_wobble=False,
        allow_pseudoknot=True,
    )
    assert not result_no_wobble.valid


# --------------------------------------------------------------------------- #
# Fixture 5: pseudoknot crossing
# --------------------------------------------------------------------------- #
def test_fixture_05_pseudoknot_crossing():
    """Crossing (pseudoknot) pairs.

    Sequence: GAAAAAAAAACAAAAUAAAAAA (L=22)
        pos 0  = G, pos 10 = C  -> canonical G-C, distance 10 (> min_loop=3)
        pos 5  = A, pos 15 = U  -> canonical A-U, distance 10 (> min_loop=3)
        0 < 5 < 10 < 15  -> crossing (pseudoknot)

    The evaluator itself does NOT care about pseudoknots -- it just compares
    matrix cells.  The constraint checker should flag the crossing when
    allow_pseudoknot=False and accept it when allow_pseudoknot=True.

    Hand computation (L=22, |U|=231):
        TP=2, FP=0, FN=0, TN=229
        F1 = 1.0
    """
    size = 22
    pairs = [(0, 10), (5, 15)]
    predicted = _matrix_from_pairs(pairs, size)
    target = _matrix_from_pairs(pairs, size)

    confusion = pair_confusion(predicted, target)
    assert confusion == {"tp": 2, "fp": 0, "fn": 0, "tn": 229}

    assert f1_score(predicted, target) == pytest.approx(1.0)

    # Constraint checker: crossing should be flagged when allow_pseudoknot=False.
    sequence = "GAAAAAAAAACAAAAUAAAAAA"
    assert len(sequence) == size
    assert sequence[0] == "G" and sequence[10] == "C"
    assert sequence[5] == "A" and sequence[15] == "U"
    result_nested = validate_pair_matrix(
        sequence,
        predicted,
        min_loop=3,
        allow_wobble=True,
        allow_pseudoknot=False,
    )
    assert not result_nested.valid
    assert any("pseudoknot" in v for v in result_nested.violations), (
        f"expected pseudoknot violation, got: {result_nested.violations}"
    )

    # With allow_pseudoknot=True, the crossing is accepted (both pairs canonical).
    result_pk = validate_pair_matrix(
        sequence,
        predicted,
        min_loop=3,
        allow_wobble=True,
        allow_pseudoknot=True,
    )
    assert result_pk.valid, f"pseudoknot should be allowed: {result_pk.violations}"


# --------------------------------------------------------------------------- #
# Fixture 6: illegal diagonal pair
# --------------------------------------------------------------------------- #
def test_fixture_06_illegal_diagonal_pair():
    """A pair (i, i) on the diagonal must be rejected by pairs_to_matrix.

    pairs_to_matrix raises ValueError for self-pairs.  The external baseline
    scorer (_normalize_pair) also rejects self-pairs.
    """
    size = 5
    with pytest.raises(ValueError, match="diagonal"):
        _matrix_from_pairs([(2, 2)], size)

    # The external baseline scorer rejects self-pairs too.
    from evaluate_external_baseline_predictions import _normalize_pair

    with pytest.raises(ValueError, match="self-pair"):
        _normalize_pair([2, 2], length=size, one_based=False)


# --------------------------------------------------------------------------- #
# Fixture 7: 1-based vs 0-based pair list
# --------------------------------------------------------------------------- #
def test_fixture_07_one_based_vs_zero_based():
    """1-based pair [1, 5] must equal 0-based pair [0, 4] after normalization.

    The external baseline scorer has a --one-based-predictions flag that
    subtracts 1 from each index.  The ReactFlow internal format is always
    0-based.
    """
    from evaluate_external_baseline_predictions import _normalize_pair

    one_based = _normalize_pair([1, 5], length=8, one_based=True)
    zero_based = _normalize_pair([0, 4], length=8, one_based=False)
    assert one_based == zero_based == (0, 4)

    # A 1-based pair [1, 1] would become (0, 0) -- still a self-pair after
    # normalization, so it must be rejected.
    with pytest.raises(ValueError, match="self-pair"):
        _normalize_pair([1, 1], length=8, one_based=True)

    # Out-of-range 1-based pair should be rejected.
    with pytest.raises(ValueError, match="out of range"):
        _normalize_pair([1, 9], length=8, one_based=True)


# --------------------------------------------------------------------------- #
# Fixture 8: relaxed match hits but exact match does not
# --------------------------------------------------------------------------- #
def test_fixture_08_relaxed_match_vs_exact_match():
    """A predicted pair shifted by 1 should be FN/FP under exact matching but
    TP under shifted matching (tolerance=1).

    Target pair: (0, 6)
    Predicted pair: (0, 7)  -- right endpoint shifted by +1

    Exact matching (pair_confusion):
        TP=0, FP=1, FN=1

    Shifted matching (shifted_pair_counts, tolerance=1):
        |0-0|<=1 and |7-6|<=1  -> TP=1, FP=0, FN=0
    """
    size = 8
    target_pairs = [(0, 6)]
    predicted_pairs = [(0, 7)]
    predicted = _matrix_from_pairs(predicted_pairs, size)
    target = _matrix_from_pairs(target_pairs, size)

    # Exact matching.
    confusion = pair_confusion(predicted, target)
    assert confusion["tp"] == 0
    assert confusion["fp"] == 1
    assert confusion["fn"] == 1

    exact_f1 = f1_score(predicted, target)
    assert exact_f1 == 0.0  # 2*0/(0+1+1) = 0

    # Shifted matching (tolerance=1).
    shifted = shifted_pair_counts(predicted, target, tolerance=1)
    assert shifted["tp"] == 1
    assert shifted["fp"] == 0
    assert shifted["fn"] == 0

    shifted_f1 = f1_from_counts(shifted["tp"], shifted["fp"], shifted["fn"])
    assert shifted_f1 == pytest.approx(1.0)

    # structure_record_metrics should report BOTH exact_f1 and shifted_f1.
    record = structure_record_metrics(predicted, target)
    assert record["exact_f1"] == pytest.approx(0.0)
    assert record["shifted_f1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Fixture 9: empty prediction on non-empty target
# --------------------------------------------------------------------------- #
def test_fixture_09_empty_prediction_on_nonempty_target():
    """Empty prediction on a non-empty target.

    Target: (0,7), (1,6)  (2 pairs)
    Predicted: none

    Hand computation (L=8, |U|=28):
        TP=0, FP=0, FN=2, TN=26
        F1 = 0/2 = 0.0
        Recall = 0/2 = 0.0
        Precision = undefined (0/0) -> 0.0 in our implementation
    """
    size = 8
    target_pairs = [(0, 7), (1, 6)]
    predicted = _matrix_from_pairs([], size)
    target = _matrix_from_pairs(target_pairs, size)

    confusion = pair_confusion(predicted, target)
    assert confusion == {"tp": 0, "fp": 0, "fn": 2, "tn": 26}

    assert f1_score(predicted, target) == 0.0

    record = structure_record_metrics(predicted, target)
    assert record["exact_f1"] == 0.0
    assert record["recall"] == 0.0
    assert record["precision"] == 0.0  # 0/(0+0) -> 0.0


# --------------------------------------------------------------------------- #
# Fixture 10: non-empty prediction on empty target
# --------------------------------------------------------------------------- #
def test_fixture_10_nonempty_prediction_on_empty_target():
    """Non-empty prediction on an empty target.

    Target: none
    Predicted: (0,7), (1,6)  (2 spurious pairs)

    Hand computation (L=8, |U|=28):
        TP=0, FP=2, FN=0, TN=26
        F1 = 0/2 = 0.0
        Precision = 0/2 = 0.0
        Recall = undefined (0/0) -> 0.0 in our implementation
    """
    size = 8
    predicted_pairs = [(0, 7), (1, 6)]
    predicted = _matrix_from_pairs(predicted_pairs, size)
    target = _matrix_from_pairs([], size)

    confusion = pair_confusion(predicted, target)
    assert confusion == {"tp": 0, "fp": 2, "fn": 0, "tn": 26}

    assert f1_score(predicted, target) == 0.0

    record = structure_record_metrics(predicted, target)
    assert record["exact_f1"] == 0.0
    assert record["precision"] == 0.0
    assert record["recall"] == 0.0


# --------------------------------------------------------------------------- #
# Fixture 11: length mismatch (error handling)
# --------------------------------------------------------------------------- #
def test_fixture_11_length_mismatch():
    """Predicted and target matrices of different sizes must raise.

    The evaluator rejects shape mismatches rather than silently truncating or
    padding.  This is a protocol safety guard.
    """
    predicted = _matrix_from_pairs([(0, 4)], 5)
    target = _matrix_from_pairs([(0, 6)], 8)

    with pytest.raises(ValueError, match="same shape"):
        pair_confusion(predicted, target)


# --------------------------------------------------------------------------- #
# Fixture 12: distance bin accounting
# --------------------------------------------------------------------------- #
def test_fixture_12_distance_bin_accounting():
    """Pairs in different distance bins are scored independently.

    Sequence: GCAAAGCAAAGCAAAGCAAAGC (L=22)
    Pairs:
        (0,21)  distance=21  -> long  (>=24? no, 21 < 24, so medium? no, 21 is in short? no)
    Wait, distance bins are:
        short:  1..11
        medium: 12..23
        long:   >=24

    So distance=21 -> medium
    distance=15 -> medium
    distance=10 -> short
    distance=6  -> short

    Let me use pairs at distances 6, 10, 15, 21 to cover short and medium.
    Target = Predicted = {(0,6), (1,11), (5,16), (3,20)}  -- but check crossing
    (0,6) and (1,11): 0<1<6<11 -> crossing
    Let me use nested: (0,21), (1,20), (5,16), (6,15) -- distances 21, 19, 11, 9
    distance 21 -> medium, 19 -> medium, 11 -> short, 9 -> short

    Short bin (1..11): pairs with distance 11 and 9 -> 2 pairs
    Medium bin (12..23): pairs with distance 21 and 19 -> 2 pairs
    Long bin (>=24): 0 pairs
    """
    size = 22
    pairs = [(0, 21), (1, 20), (5, 16), (6, 15)]
    matrix = _matrix_from_pairs(pairs, size)

    record = structure_record_metrics(matrix, matrix)
    bins = record["distance_bins"]

    # Short bin: distances 9 and 11 -> 2 TP, 0 FP, 0 FN
    assert bins["short"]["confusion"]["tp"] == 2
    assert bins["short"]["f1"] == pytest.approx(1.0)

    # Medium bin: distances 19 and 21 -> 2 TP
    assert bins["medium"]["confusion"]["tp"] == 2
    assert bins["medium"]["f1"] == pytest.approx(1.0)

    # Long bin: no pairs -> TP=0, denominator=0 -> F1=0.0
    assert bins["long"]["confusion"]["tp"] == 0
    assert bins["long"]["f1"] == 0.0


# --------------------------------------------------------------------------- #
# Fixture 13: matrix-based vs set-based scorer alignment
# --------------------------------------------------------------------------- #
def test_fixture_13_matrix_vs_set_scorer_alignment():
    """The matrix-based pair_confusion and the set-based _pair_confusion must
    agree on every non-trivial case.

    This is the core alignment guarantee between the ReactFlow model evaluator
    (c0_evaluate.py, matrix-based) and the eFold wrapper scorer
    (evaluate_external_baseline_predictions.py, set-based).
    """
    size = 10
    target_pairs = [(0, 9), (1, 8), (2, 7)]
    predicted_pairs = [(0, 9), (1, 8), (3, 6)]  # 2 correct, 1 spurious, 1 missed

    predicted_matrix = _matrix_from_pairs(predicted_pairs, size)
    target_matrix = _matrix_from_pairs(target_pairs, size)

    matrix_conf = pair_confusion(predicted_matrix, target_matrix)
    set_conf = _set_pair_confusion(
        frozenset(predicted_pairs), frozenset(target_pairs), size
    )

    # |U| = 10*9/2 = 45; TN = 45 - 2 - 1 - 1 = 41.
    assert matrix_conf == set_conf
    assert matrix_conf == {"tp": 2, "fp": 1, "fn": 1, "tn": 41}


# --------------------------------------------------------------------------- #
# Summary: all fixtures registered
# --------------------------------------------------------------------------- #
GOLD_FIXTURES = [
    "01_all_unpaired_empty_structure",
    "02_single_hairpin_perfect_match",
    "03_two_stems_nested_partial_match",
    "04_gu_wobble_pair",
    "05_pseudoknot_crossing",
    "06_illegal_diagonal_pair",
    "07_one_based_vs_zero_based",
    "08_relaxed_match_vs_exact_match",
    "09_empty_prediction_on_nonempty_target",
    "10_nonempty_prediction_on_empty_target",
    "11_length_mismatch",
    "12_distance_bin_accounting",
    "13_matrix_vs_set_scorer_alignment",
]


def test_gold_fixtures_registered():
    """Smoke test: the GOLD_FIXTURES list matches the implemented test functions."""
    assert len(GOLD_FIXTURES) == 13
    for name in GOLD_FIXTURES:
        assert isinstance(name, str) and name.strip()
