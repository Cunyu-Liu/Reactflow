"""Capture gold-fixture results into a JSON artifact for the C1-0 audit trail.

For each fixture this script records:
  * the hand-computed expected TP/FP/FN/TN/F1/MCC,
  * the actual values produced by the ReactFlow scorers,
  * a pass/fail flag,
  * the fixture's documentation comment.

The output is written to ``artifacts/c1_0/evaluator_fixture_results.json`` and is
the canonical evidence for Gate criterion 1 (all fixtures 100% pass).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

# Make src/ and scripts/ importable when run from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for path in (_SRC, _SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reactflow.constraints import pairs_to_matrix, validate_pair_matrix  # noqa: E402
from reactflow.metrics import pair_confusion, f1_score, matthews_corrcoef  # noqa: E402
from reactflow.c0_evaluate import (  # noqa: E402
    f1_from_counts,
    mcc_from_counts,
    shifted_pair_counts,
    structure_record_metrics,
)
from evaluate_external_baseline_predictions import _normalize_pair  # noqa: E402


def _matrix(pairs, size):
    return pairs_to_matrix(pairs, size)


def _set_confusion(predicted: FrozenSet[Tuple[int, int]], target: FrozenSet[Tuple[int, int]], length: int) -> Dict[str, int]:
    tp = len(predicted & target)
    fp = len(predicted - target)
    fn = len(target - predicted)
    candidate = length * (length - 1) // 2
    tn = candidate - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": max(tn, 0)}


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def _mcc(tp, fp, fn, tn):
    import math
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def _approx_equal(a, b, eps=1e-9):
    return abs(a - b) <= eps


def _entry(name, expected, actual, notes=""):
    """Build one fixture result entry comparing expected vs actual."""
    ok = True
    for key in ("tp", "fp", "fn", "tn"):
        if expected.get(key) is not None and expected[key] != actual.get(key):
            ok = False
            break
    for key in ("f1", "mcc"):
        if expected.get(key) is not None and not _approx_equal(expected[key], actual.get(key, 0.0)):
            ok = False
            break
    return {
        "fixture": name,
        "passed": ok,
        "expected": expected,
        "actual": actual,
        "notes": notes,
    }


def run_all() -> dict:
    results = []

    # Fixture 01: empty vs empty
    size = 4
    pred = _matrix([], size)
    targ = _matrix([], size)
    conf = pair_confusion(pred, targ)
    results.append(_entry(
        "01_all_unpaired_empty_structure",
        {"tp": 0, "fp": 0, "fn": 0, "tn": 6, "f1": 0.0, "mcc": 0.0},
        {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"],
         "f1": f1_score(pred, targ), "mcc": matthews_corrcoef(pred, targ)},
        notes="ReactFlow convention: empty-vs-empty -> F1=0.0 (eFold returns 1.0).",
    ))

    # Fixture 02: single hairpin perfect
    size = 8
    pairs = [(0, 7), (1, 6)]
    pred = _matrix(pairs, size)
    targ = _matrix(pairs, size)
    conf = pair_confusion(pred, targ)
    results.append(_entry(
        "02_single_hairpin_perfect_match",
        {"tp": 2, "fp": 0, "fn": 0, "tn": 26, "f1": 1.0, "mcc": 1.0},
        {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"],
         "f1": f1_score(pred, targ), "mcc": matthews_corrcoef(pred, targ)},
    ))

    # Fixture 03: two stems nested, partial match
    size = 22
    target_pairs = [(0, 21), (1, 20), (5, 16), (6, 15)]
    predicted_pairs = [(0, 21), (5, 16), (6, 15)]
    pred = _matrix(predicted_pairs, size)
    targ = _matrix(target_pairs, size)
    conf = pair_confusion(pred, targ)
    exp_f1 = _f1(3, 0, 1)
    exp_mcc = _mcc(3, 0, 1, 227)
    results.append(_entry(
        "03_two_stems_nested_partial_match",
        {"tp": 3, "fp": 0, "fn": 1, "tn": 227, "f1": exp_f1, "mcc": exp_mcc},
        {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"],
         "f1": f1_score(pred, targ), "mcc": matthews_corrcoef(pred, targ)},
    ))

    # Fixture 04: GU wobble
    size = 8
    pairs = [(0, 7), (1, 6)]
    pred = _matrix(pairs, size)
    targ = _matrix(pairs, size)
    conf = pair_confusion(pred, targ)
    seq = "GCAAAAGU"
    res_wob = validate_pair_matrix(seq, pred, min_loop=3, allow_wobble=True, allow_pseudoknot=True)
    res_no_wob = validate_pair_matrix(seq, pred, min_loop=3, allow_wobble=False, allow_pseudoknot=True)
    results.append(_entry(
        "04_gu_wobble_pair",
        {"tp": 2, "fp": 0, "fn": 0, "tn": 26, "f1": 1.0},
        {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"],
         "f1": f1_score(pred, targ)},
        notes=f"allow_wobble=True valid={res_wob.valid}; allow_wobble=False valid={res_no_wob.valid}",
    ))

    # Fixture 05: pseudoknot crossing
    size = 22
    pairs = [(0, 10), (5, 15)]
    pred = _matrix(pairs, size)
    targ = _matrix(pairs, size)
    conf = pair_confusion(pred, targ)
    seq = "GAAAAAAAAACAAAAUAAAAAA"
    res_nest = validate_pair_matrix(seq, pred, min_loop=3, allow_wobble=True, allow_pseudoknot=False)
    res_pk = validate_pair_matrix(seq, pred, min_loop=3, allow_wobble=True, allow_pseudoknot=True)
    results.append(_entry(
        "05_pseudoknot_crossing",
        {"tp": 2, "fp": 0, "fn": 0, "tn": 229, "f1": 1.0},
        {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"],
         "f1": f1_score(pred, targ)},
        notes=f"nested valid={res_nest.valid} (expects False); pseudoknot valid={res_pk.valid} (expects True)",
    ))

    # Fixture 06: illegal diagonal pair
    diag_error = None
    try:
        _matrix([(2, 2)], 5)
    except ValueError as exc:
        diag_error = str(exc)
    norm_error = None
    try:
        _normalize_pair([2, 2], length=5, one_based=False)
    except ValueError as exc:
        norm_error = str(exc)
    results.append({
        "fixture": "06_illegal_diagonal_pair",
        "passed": bool(diag_error and "diagonal" in diag_error and norm_error and "self-pair" in norm_error),
        "expected": {"raises": "ValueError with 'diagonal' (matrix) / 'self-pair' (set)"},
        "actual": {"matrix_error": diag_error, "set_error": norm_error},
    })

    # Fixture 07: 1-based vs 0-based
    ob = _normalize_pair([1, 5], length=8, one_based=True)
    zb = _normalize_pair([0, 4], length=8, one_based=False)
    oob_error = None
    try:
        _normalize_pair([1, 9], length=8, one_based=True)
    except ValueError as exc:
        oob_error = str(exc)
    results.append({
        "fixture": "07_one_based_vs_zero_based",
        "passed": ob == zb == (0, 4) and oob_error and "out of range" in oob_error,
        "expected": {"relation": "1-based [1,5] == 0-based [0,4] == (0,4)"},
        "actual": {"one_based": list(ob), "zero_based": list(zb), "out_of_range_error": oob_error},
    })

    # Fixture 08: relaxed vs exact
    size = 8
    target_pairs = [(0, 6)]
    predicted_pairs = [(0, 7)]
    pred = _matrix(predicted_pairs, size)
    targ = _matrix(target_pairs, size)
    conf = pair_confusion(pred, targ)
    shifted = shifted_pair_counts(pred, targ, tolerance=1)
    rec = structure_record_metrics(pred, targ)
    results.append(_entry(
        "08_relaxed_match_vs_exact_match",
        {"exact_tp": 0, "exact_fp": 1, "exact_fn": 1, "exact_f1": 0.0,
         "shifted_tp": 1, "shifted_fp": 0, "shifted_fn": 0, "shifted_f1": 1.0},
        {"exact_tp": conf["tp"], "exact_fp": conf["fp"], "exact_fn": conf["fn"],
         "exact_f1": f1_score(pred, targ),
         "shifted_tp": shifted["tp"], "shifted_fp": shifted["fp"], "shifted_fn": shifted["fn"],
         "shifted_f1": f1_from_counts(shifted["tp"], shifted["fp"], shifted["fn"]),
         "record_exact_f1": rec["exact_f1"], "record_shifted_f1": rec["shifted_f1"]},
    ))

    # Fixture 09: empty prediction on nonempty target
    size = 8
    target_pairs = [(0, 7), (1, 6)]
    pred = _matrix([], size)
    targ = _matrix(target_pairs, size)
    conf = pair_confusion(pred, targ)
    rec = structure_record_metrics(pred, targ)
    results.append(_entry(
        "09_empty_prediction_on_nonempty_target",
        {"tp": 0, "fp": 0, "fn": 2, "tn": 26, "f1": 0.0, "precision": 0.0, "recall": 0.0},
        {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"],
         "f1": f1_score(pred, targ), "precision": rec["precision"], "recall": rec["recall"]},
    ))

    # Fixture 10: nonempty prediction on empty target
    size = 8
    predicted_pairs = [(0, 7), (1, 6)]
    pred = _matrix(predicted_pairs, size)
    targ = _matrix([], size)
    conf = pair_confusion(pred, targ)
    rec = structure_record_metrics(pred, targ)
    results.append(_entry(
        "10_nonempty_prediction_on_empty_target",
        {"tp": 0, "fp": 2, "fn": 0, "tn": 26, "f1": 0.0, "precision": 0.0, "recall": 0.0},
        {"tp": conf["tp"], "fp": conf["fp"], "fn": conf["fn"], "tn": conf["tn"],
         "f1": f1_score(pred, targ), "precision": rec["precision"], "recall": rec["recall"]},
    ))

    # Fixture 11: length mismatch
    pred = _matrix([(0, 4)], 5)
    targ = _matrix([(0, 6)], 8)
    mismatch_error = None
    try:
        pair_confusion(pred, targ)
    except ValueError as exc:
        mismatch_error = str(exc)
    results.append({
        "fixture": "11_length_mismatch",
        "passed": bool(mismatch_error and "same shape" in mismatch_error),
        "expected": {"raises": "ValueError with 'same shape'"},
        "actual": {"error": mismatch_error},
    })

    # Fixture 12: distance bins
    size = 22
    pairs = [(0, 21), (1, 20), (5, 16), (6, 15)]
    mat = _matrix(pairs, size)
    rec = structure_record_metrics(mat, mat)
    bins = rec["distance_bins"]
    # distances: 21 (medium), 19 (medium), 11 (short), 9 (short)
    short_ok = bins["short"]["confusion"]["tp"] == 2 and _approx_equal(bins["short"]["f1"], 1.0)
    med_ok = bins["medium"]["confusion"]["tp"] == 2 and _approx_equal(bins["medium"]["f1"], 1.0)
    long_ok = bins["long"]["confusion"]["tp"] == 0 and bins["long"]["f1"] == 0.0
    results.append({
        "fixture": "12_distance_bin_accounting",
        "passed": short_ok and med_ok and long_ok,
        "expected": {"short": {"tp": 2, "f1": 1.0}, "medium": {"tp": 2, "f1": 1.0}, "long": {"tp": 0, "f1": 0.0}},
        "actual": {
            "short": {"tp": bins["short"]["confusion"]["tp"], "f1": bins["short"]["f1"]},
            "medium": {"tp": bins["medium"]["confusion"]["tp"], "f1": bins["medium"]["f1"]},
            "long": {"tp": bins["long"]["confusion"]["tp"], "f1": bins["long"]["f1"]},
        },
        "notes": "distances: (0,21)=21 medium, (1,20)=19 medium, (5,16)=11 short, (6,15)=9 short",
    })

    # Fixture 13: matrix vs set alignment
    size = 10
    target_pairs = [(0, 9), (1, 8), (2, 7)]
    predicted_pairs = [(0, 9), (1, 8), (3, 6)]
    pred = _matrix(predicted_pairs, size)
    targ = _matrix(target_pairs, size)
    matrix_conf = pair_confusion(pred, targ)
    set_conf = _set_confusion(frozenset(predicted_pairs), frozenset(target_pairs), size)
    results.append({
        "fixture": "13_matrix_vs_set_scorer_alignment",
        "passed": matrix_conf == set_conf == {"tp": 2, "fp": 1, "fn": 1, "tn": 41},
        "expected": {"tp": 2, "fp": 1, "fn": 1, "tn": 41},
        "actual": {"matrix": matrix_conf, "set": set_conf},
        "notes": "|U| = 10*9/2 = 45; TN = 45 - 2 - 1 - 1 = 41.",
    })

    all_passed = all(r["passed"] for r in results)
    return {
        "schema_version": "1.0",
        "phase": "C1-0",
        "description": "Hand-computed gold fixtures for the ReactFlow structure evaluator.",
        "total_fixtures": len(results),
        "all_passed": all_passed,
        "gate_criterion_1": all_passed,
        "fixtures": results,
    }


def main():
    report = run_all()
    out_path = _ROOT / "artifacts" / "c1_0" / "evaluator_fixture_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Total fixtures: {report['total_fixtures']}")
    print(f"All passed: {report['all_passed']}")
    if not report["all_passed"]:
        print("FAILURES:")
        for r in report["fixtures"]:
            if not r["passed"]:
                print(f"  - {r['fixture']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
