"""Evaluation metrics for RNA pair prediction and reactivity profiles."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from reactflow.constraints import matrix_to_pairs


def pair_confusion(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> Dict[str, int]:
    """Compute pair-level TP/FP/FN/TN over upper-triangle cells.

    Formula:
    ``TP = |P_hat cap P|``, ``FP = |P_hat \\ P|``, ``FN = |P \\ P_hat|``.
    ``TN`` counts non-pair upper-triangle candidates.

    Complexity: O(L^2).
    """

    if len(predicted) != len(target) or any(len(a) != len(b) for a, b in zip(predicted, target)):
        raise ValueError("predicted and target matrices must have the same shape")
    size = len(predicted)
    tp = fp = fn = tn = 0
    for i in range(size):
        for j in range(i + 1, size):
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


def f1_score(predicted: Sequence[Sequence[float]], target: Sequence[Sequence[float]]) -> float:
    """Return pair-level F1 score ``2TP/(2TP+FP+FN)``.

    Complexity: O(L^2).
    """

    c = pair_confusion(predicted, target)
    denominator = 2 * c["tp"] + c["fp"] + c["fn"]
    return 0.0 if denominator == 0 else 2 * c["tp"] / denominator


def matthews_corrcoef(predicted: Sequence[Sequence[float]], target: Sequence[Sequence[float]]) -> float:
    """Return pair-level Matthews correlation coefficient.

    Formula:
    ``MCC=(TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))``.

    Complexity: O(L^2).
    """

    c = pair_confusion(predicted, target)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator


def mean_absolute_error(predicted: Sequence[float], target: Sequence[float]) -> float:
    """Return MAE over finite positions.

    Complexity: O(L).
    """

    if len(predicted) != len(target):
        raise ValueError("predicted and target lengths must match")
    values = [
        abs(float(a) - float(b))
        for a, b in zip(predicted, target)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if not values:
        raise ValueError("no finite observations")
    return sum(values) / len(values)
