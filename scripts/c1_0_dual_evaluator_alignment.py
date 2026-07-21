#!/usr/bin/env python3
"""C1-0 Task 3: Dual evaluator alignment (eFold vs ReactFlow).

This script compares three F1 evaluators on a fixed battery of small RNA
secondary-structure test cases:

1. eFold's official ``efold.core.metrics.f1`` (torch-tensor, full-matrix sum
   over all cells, including the symmetric lower triangle).
2. ReactFlow's matrix-based ``reactflow.metrics.f1_score`` (upper-triangle
   cell confusion, ``2TP/(2TP+FP+FN)``).
3. ReactFlow's set-based scorer in
   ``scripts.evaluate_external_baseline_predictions._pair_confusion`` +
   ``_f1_from_counts`` (operates on frozensets of ``(i, j)`` pairs).

Equivalence rationale
---------------------
For a symmetric binary pair matrix with zero diagonal, the eFold full-matrix
sum doubles every quantity consistently:

    sum(pred)      = 2 * (TP + FP)
    sum(true)      = 2 * (TP + FN)
    sum(pred*true) = 2 * TP

so

    efold_f1 = 2 * sum(pred*true) / (sum(pred) + sum(true))
             = 2 * (2*TP) / (2*(TP+FP) + 2*(TP+FN))
             = 2*TP / (2*TP + FP + FN)

which is identical to ReactFlow's upper-triangle F1.  The only documented
disagreement is the empty-vs-empty convention: eFold returns 1.0
(``sum_pair == 0`` branch) while ReactFlow returns 0.0 (``denominator == 0``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from efold.core.metrics import f1 as efold_f1

from reactflow.constraints import matrix_to_pairs, pairs_to_matrix
from reactflow.metrics import f1_score as reactflow_matrix_f1

from evaluate_external_baseline_predictions import _pair_confusion, _f1_from_counts


# Tolerance for ReactFlow matrix-vs-set agreement (both float64, exact).
TOL = 1e-9
# Tolerance for eFold-vs-ReactFlow agreement.  eFold's f1 returns a Python
# float via ``.item()`` on a torch.float32 scalar, so its ULP error for values
# near 1.0 is ~1.2e-7 (float32 epsilon).  1e-6 leaves comfortable headroom
# while still catching genuine formula disagreements.
EFOLD_TOL = 1e-6


def _matrix_to_torch(matrix: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert a list-of-lists matrix to a torch.float32 tensor."""

    return torch.tensor(matrix, dtype=torch.float32)


def _frozenset_from_matrix(matrix: Sequence[Sequence[float]]) -> frozenset:
    """Convert an upper-triangle binary matrix to a frozenset of (i, j) pairs."""

    return frozenset(matrix_to_pairs(matrix))


def _reactflow_set_f1(
    pred_matrix: Sequence[Sequence[float]],
    target_matrix: Sequence[Sequence[float]],
) -> float:
    """Run the ReactFlow set-based scorer from the external baseline script."""

    length = len(pred_matrix)
    pred_pairs = _frozenset_from_matrix(pred_matrix)
    target_pairs = _frozenset_from_matrix(target_matrix)
    confusion = _pair_confusion(pred_pairs, target_pairs, length=length)
    return _f1_from_counts(confusion["tp"], confusion["fp"], confusion["fn"])


def _build_matrix(pairs: Sequence[Tuple[int, int]], size: int) -> List[List[float]]:
    """Build a symmetric binary matrix from a list of (i, j) pairs."""

    matrix = pairs_to_matrix(list(pairs), size)
    return [list(row) for row in matrix]


# Each test case: (name, size, predicted_pairs, target_pairs)
TEST_CASES: List[Tuple[str, int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]] = [
    # 1. Empty vs empty - the critical convention difference (eFold=1.0, ReactFlow=0.0)
    ("empty_vs_empty_L5", 5, [], []),
    # 2. Perfect match - single hairpin
    ("perfect_match_single_hairpin_L10", 10, [(3, 8)], [(3, 8)]),
    # 3. Perfect match - two nested pairs (helix)
    ("perfect_match_two_pairs_L12", 12, [(1, 10), (3, 8)], [(1, 10), (3, 8)]),
    # 4. Partial match - missed pair (recall < 1)
    ("partial_missed_pair_L12", 12, [(1, 10)], [(1, 10), (3, 8)]),
    # 5. Partial match - spurious pair (precision < 1)
    ("partial_spurious_pair_L12", 12, [(1, 10), (3, 8)], [(1, 10)]),
    # 6. Partial match - both missed and spurious
    ("partial_missed_and_spurious_L15", 15, [(2, 12), (4, 9)], [(2, 12), (5, 10)]),
    # 7. GU wobble pair - evaluator should not care about chemistry
    ("gu_wobble_pair_L8", 8, [(1, 6)], [(1, 6)]),
    # 8. Pseudoknot crossing - perfect match (evaluator ignores crossings)
    ("pseudoknot_crossing_perfect_L10", 10, [(1, 5), (3, 7)], [(1, 5), (3, 7)]),
    # 9. Pseudoknot crossing - partial match
    ("pseudoknot_crossing_partial_L10", 10, [(1, 5), (3, 7)], [(1, 5), (6, 9)]),
    # 10. All-unpaired target with non-empty prediction
    ("all_unpaired_target_nonempty_pred_L8", 8, [(1, 6)], []),
    # 11. Empty prediction with non-empty target
    ("empty_pred_nonempty_target_L8", 8, [], [(1, 6)]),
    # 12. Larger perfect match
    ("larger_perfect_L20", 20, [(2, 18), (4, 16), (6, 14)], [(2, 18), (4, 16), (6, 14)]),
    # 13. Larger mixed (perfect + missed + spurious)
    ("larger_mixed_L20", 20, [(2, 18), (4, 16), (6, 14)], [(2, 18), (5, 15), (7, 12)]),
    # 14. Length 1 (degenerate, no pairs possible) -> empty-vs-empty convention
    ("degenerate_L1", 1, [], []),
    # 15. Length 0 (fully degenerate) -> empty-vs-empty convention
    ("degenerate_L0", 0, [], []),
]


def _classify_difference(
    pred_pairs: Sequence[Tuple[int, int]],
    target_pairs: Sequence[Tuple[int, int]],
    efold_val: float,
    matrix_val: float,
    set_val: float,
) -> str:
    """Return a human-readable reason for any evaluator disagreement."""

    # ReactFlow matrix and set scorers both operate in float64, so they must
    # agree bit-for-bit (modulo a tiny tolerance for safety).
    matrix_set_agree = abs(matrix_val - set_val) < TOL
    # eFold returns a torch.float32 scalar via ``.item()``, so we compare with
    # the float32-aware tolerance.
    efold_matrix_agree = abs(efold_val - matrix_val) < EFOLD_TOL

    if matrix_set_agree and efold_matrix_agree:
        return "none"

    # Empty-vs-empty convention: eFold returns 1.0, ReactFlow returns 0.0.
    if (
        len(pred_pairs) == 0
        and len(target_pairs) == 0
        and abs(efold_val - 1.0) < TOL
        and abs(matrix_val - 0.0) < TOL
        and abs(set_val - 0.0) < TOL
    ):
        return "empty_vs_empty_convention"

    if not matrix_set_agree:
        return (
            f"reactflow_matrix_vs_set_mismatch(matrix={matrix_val}, set={set_val})"
        )

    return (
        f"unexpected_difference(efold={efold_val}, matrix={matrix_val}, set={set_val})"
    )


def _run_one(
    name: str,
    size: int,
    pred_pairs_seq: Sequence[Tuple[int, int]],
    target_pairs_seq: Sequence[Tuple[int, int]],
) -> dict:
    """Run all three evaluators on one test case and return the result dict."""

    pred_pairs = [tuple(p) for p in pred_pairs_seq]
    target_pairs = [tuple(p) for p in target_pairs_seq]

    pred_matrix = _build_matrix(pred_pairs, size)
    target_matrix = _build_matrix(target_pairs, size)

    # eFold expects torch tensors of shape (L, L). For size 0 we build an
    # empty (0, 0) tensor directly; torch.sum on it returns 0, so the
    # sum_pair == 0 branch fires and eFold returns 1.0.
    pred_tensor = _matrix_to_torch(pred_matrix) if size > 0 else torch.zeros(
        (0, 0), dtype=torch.float32
    )
    target_tensor = _matrix_to_torch(target_matrix) if size > 0 else torch.zeros(
        (0, 0), dtype=torch.float32
    )
    efold_val = float(efold_f1(pred_tensor, target_tensor))

    matrix_val = float(reactflow_matrix_f1(pred_matrix, target_matrix))
    set_val = float(_reactflow_set_f1(pred_matrix, target_matrix))

    reason = _classify_difference(
        pred_pairs, target_pairs, efold_val, matrix_val, set_val
    )
    agree = reason == "none"

    return {
        "test_case": name,
        "size": size,
        "predicted_pairs": [list(p) for p in pred_pairs],
        "target_pairs": [list(p) for p in target_pairs],
        "efold_f1": efold_val,
        "reactflow_matrix_f1": matrix_val,
        "reactflow_set_f1": set_val,
        "agree": agree,
        "difference_reason": reason,
    }


def main() -> int:
    results = [_run_one(name, size, pred, target) for name, size, pred, target in TEST_CASES]

    non_empty_results = [
        r for r in results if r["difference_reason"] != "empty_vs_empty_convention"
    ]
    non_empty_agree_count = sum(1 for r in non_empty_results if r["agree"])
    unexpected_diffs = [
        r for r in results
        if r["difference_reason"] not in ("none", "empty_vs_empty_convention")
    ]

    summary = {
        "schema_version": 1,
        "description": (
            "Dual evaluator alignment: eFold official f1 vs ReactFlow matrix "
            "f1_score vs ReactFlow set-based _pair_confusion.  The only "
            "documented disagreement is the empty-vs-empty convention "
            "(eFold=1.0, ReactFlow=0.0).  eFold returns a torch.float32 "
            "scalar, so eFold-vs-ReactFlow comparisons use a float32-aware "
            "tolerance."
        ),
        "tolerance_matrix_vs_set": TOL,
        "tolerance_efold_vs_reactflow": EFOLD_TOL,
        "test_case_count": len(results),
        "agree_count": sum(1 for r in results if r["agree"]),
        "empty_vs_empty_count": sum(
            1 for r in results if r["difference_reason"] == "empty_vs_empty_convention"
        ),
        "unexpected_difference_count": len(unexpected_diffs),
        "non_empty_test_case_count": len(non_empty_results),
        "non_empty_agree_count": non_empty_agree_count,
        "non_empty_all_agree": non_empty_agree_count == len(non_empty_results),
        "unexpected_differences": unexpected_diffs,
        "results": results,
    }

    output_path = Path("artifacts/c1_0/efold_dual_alignment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "test_cases": summary["test_case_count"],
                "agree": summary["agree_count"],
                "empty_vs_empty": summary["empty_vs_empty_count"],
                "non_empty_all_agree": summary["non_empty_all_agree"],
                "unexpected_differences": summary["unexpected_difference_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
