"""Contact-map denoising auxiliary for the ReactFlow partner-class posterior.

Scientific role
---------------
ReactFlow's primary DFM objective denoises each position's partner class.  That is
the right state space for sampling legal structures, but it is still a
per-position objective.  RNA secondary structure, however, is a pairwise object:
if position ``i`` chooses ``j``, position ``j`` must choose ``i``.  The
contact-map auxiliary below adds a *pair-consistency* training signal without
replacing the structure-distribution model.

For an unordered candidate pair ``i < j`` the model-induced soft contact
probability is

    P_ij = 0.5 * (pi_i[j + 1] + pi_j[i + 1]),

where ``pi_i`` is the partner-class posterior and class ``0`` is unpaired.  The
target ``Y_ij`` comes from the clean legal structure.  We optimize a balanced
binary cross entropy over legal candidate contacts:

    L_contact = mean_pos[-log P_ij] + negative_weight * mean_neg[-log(1-P_ij)].

For RF-CF2 long-range recovery, those means can become weighted means with
``w_ij = long_range_weight`` when ``j - i >= long_range_min_distance`` and
``w_ij = 1`` otherwise.  Setting ``long_range_weight=1`` recovers the original
objective exactly.

The split positive/negative averaging is intentional.  A naive BCE over all
``O(L^2)`` legal non-pairs would swamp the sparse true contacts and encourage the
model to predict "no pair" everywhere.  This term therefore behaves as a
denoising auxiliary for pair consistency, not as a generic contact-map classifier.

The gradient is exact and pure standard library.  It is verified against central
finite differences in tests and symbolically in :mod:`reactflow.symbolic`.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


_EPS = 1e-6


def _validate_shapes(
    marginals: Sequence[Sequence[float]],
    target_pair_matrix: Sequence[Sequence[int]],
    legal_pair: Sequence[Sequence[bool]],
) -> int:
    """Validate the ``L x (L+1)``, ``L x L`` and ``L x L`` inputs and return ``L``."""

    size = len(marginals)
    if size == 0:
        raise ValueError("at least one position is required")
    if len(target_pair_matrix) != size or len(legal_pair) != size:
        raise ValueError("target_pair_matrix and legal_pair must match marginals length")
    for i, row in enumerate(marginals):
        if len(row) != size + 1:
            raise ValueError(f"marginal row {i} must have length L+1")
    for i, row in enumerate(target_pair_matrix):
        if len(row) != size:
            raise ValueError(f"target pair row {i} must have length L")
    for i, row in enumerate(legal_pair):
        if len(row) != size:
            raise ValueError(f"legal pair row {i} must have length L")
    return size


def soft_contact_matrix(marginals: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
    """Return symmetric soft contacts ``P_ij = 0.5(pi_i[j+1] + pi_j[i+1])``.

    Complexity: O(L^2).
    """

    size = len(marginals)
    if size == 0:
        raise ValueError("at least one position is required")
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i, row in enumerate(marginals):
        if len(row) != size + 1:
            raise ValueError(f"marginal row {i} must have length L+1")
    for i in range(size):
        for j in range(i + 1, size):
            value = 0.5 * (float(marginals[i][j + 1]) + float(marginals[j][i + 1]))
            matrix[i][j] = value
            matrix[j][i] = value
    return tuple(tuple(row) for row in matrix)


def _candidate_counts(
    target_pair_matrix: Sequence[Sequence[int]],
    legal_pair: Sequence[Sequence[bool]],
) -> Tuple[int, int]:
    """Count positive and negative unordered legal candidate contacts.

    Complexity: O(L^2).
    """

    size = len(target_pair_matrix)
    positives = 0
    negatives = 0
    for i in range(size):
        for j in range(i + 1, size):
            target = int(target_pair_matrix[i][j]) or int(target_pair_matrix[j][i])
            legal = bool(legal_pair[i][j]) and bool(legal_pair[j][i])
            if target and not legal:
                raise ValueError(f"target pair ({i},{j}) is not legal under the current mask")
            if not legal:
                continue
            if target:
                positives += 1
            else:
                negatives += 1
    return positives, negatives


def _validate_long_range_params(long_range_min_distance: int, long_range_weight: float) -> None:
    """Validate RF-CF2 long-range contact weighting parameters.

    Formula: candidate ``(i, j)`` is long-range iff
    ``j - i >= long_range_min_distance``.  The distance threshold is positive
    and the weight is non-negative so the positive/negative BCE components stay
    weighted means over legal candidates.  Complexity: O(1).
    """

    if long_range_min_distance < 1:
        raise ValueError("long_range_min_distance must be positive")
    if long_range_weight < 0.0:
        raise ValueError("long_range_weight must be non-negative")


def _candidate_weight(
    i: int,
    j: int,
    *,
    long_range_min_distance: int,
    long_range_weight: float,
) -> float:
    """Return ``w_ij`` for one unordered candidate pair.

    Formula: ``w_ij = long_range_weight`` if the span ``j-i`` is at least the
    threshold and ``1`` otherwise.  Complexity: O(1).
    """

    return float(long_range_weight) if (j - i) >= long_range_min_distance else 1.0


def _candidate_weight_sums(
    target_pair_matrix: Sequence[Sequence[int]],
    legal_pair: Sequence[Sequence[bool]],
    *,
    long_range_min_distance: int,
    long_range_weight: float,
) -> Tuple[float, float]:
    """Return legal positive/negative candidate weight sums.

    Formula: ``W_pos = sum_{i<j} w_ij 1[Y_ij=1]`` and
    ``W_neg = sum_{i<j} w_ij 1[Y_ij=0]``, restricted to legal candidates.  A
    target contact outside the legal mask is rejected because it is impossible
    under the current RNA pairing constraints.  Complexity: O(L^2).
    """

    size = len(target_pair_matrix)
    pos_weight = 0.0
    neg_weight = 0.0
    for i in range(size):
        for j in range(i + 1, size):
            target = int(target_pair_matrix[i][j]) or int(target_pair_matrix[j][i])
            legal = bool(legal_pair[i][j]) and bool(legal_pair[j][i])
            if target and not legal:
                raise ValueError(f"target pair ({i},{j}) is not legal under the current mask")
            if not legal:
                continue
            weight = _candidate_weight(
                i,
                j,
                long_range_min_distance=long_range_min_distance,
                long_range_weight=long_range_weight,
            )
            if target:
                pos_weight += weight
            else:
                neg_weight += weight
    return pos_weight, neg_weight


def contact_denoising_bce(
    marginals: Sequence[Sequence[float]],
    target_pair_matrix: Sequence[Sequence[int]],
    legal_pair: Sequence[Sequence[bool]],
    *,
    negative_weight: float = 0.25,
    long_range_min_distance: int = 24,
    long_range_weight: float = 1.0,
    eps: float = _EPS,
) -> float:
    """Return balanced BCE between induced soft contacts and the clean contact map.

    Positives and negatives are averaged separately, then negatives are scaled by
    ``negative_weight``.  For RF-CF2, each average may be a weighted mean where
    pairs with span at least ``long_range_min_distance`` receive
    ``long_range_weight``.  If a sequence has no positive legal contact, the
    positive part is a safe no-op; likewise for negatives.

    Complexity: O(L^2).
    """

    if negative_weight < 0.0:
        raise ValueError("negative_weight must be non-negative")
    _validate_long_range_params(long_range_min_distance, long_range_weight)
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    size = _validate_shapes(marginals, target_pair_matrix, legal_pair)
    pos_weight, neg_weight_sum = _candidate_weight_sums(
        target_pair_matrix,
        legal_pair,
        long_range_min_distance=long_range_min_distance,
        long_range_weight=long_range_weight,
    )
    pos_total = 0.0
    neg_total = 0.0
    for i in range(size):
        for j in range(i + 1, size):
            if not (bool(legal_pair[i][j]) and bool(legal_pair[j][i])):
                continue
            target = int(target_pair_matrix[i][j]) or int(target_pair_matrix[j][i])
            prob = 0.5 * (float(marginals[i][j + 1]) + float(marginals[j][i + 1]))
            prob = min(1.0 - eps, max(eps, prob))
            weight = _candidate_weight(
                i,
                j,
                long_range_min_distance=long_range_min_distance,
                long_range_weight=long_range_weight,
            )
            if target:
                pos_total += weight * -math.log(prob)
            else:
                neg_total += weight * -math.log(1.0 - prob)
    loss = 0.0
    if pos_weight > 0.0:
        loss += pos_total / pos_weight
    if neg_weight_sum > 0.0:
        loss += negative_weight * neg_total / neg_weight_sum
    return loss


def contact_denoising_logit_gradient(
    marginals: Sequence[Sequence[float]],
    target_pair_matrix: Sequence[Sequence[int]],
    legal_pair: Sequence[Sequence[bool]],
    *,
    lambda_contact: float,
    negative_weight: float = 0.25,
    long_range_min_distance: int = 24,
    long_range_weight: float = 1.0,
    eps: float = _EPS,
) -> List[List[float]]:
    """Exact gradient of ``lambda_contact * L_contact`` into partner-class logits.

    For one unordered pair ``i < j``, ``P_ij = 0.5(pi_i[j+1] + pi_j[i+1])`` and
    BCE sensitivity is ``dL/dP = (P - Y) / (P(1-P))`` times the balanced
    positive/negative scale.  With RF-CF2 long-range weighting, that scale is
    ``w_ij / W_pos`` or ``negative_weight * w_ij / W_neg``.  The gradient to row
    ``i`` follows from the softmax Jacobian of class ``j+1``:

        d pi_i[j+1] / d logit_i[k] = pi_i[j+1] (1[k=j+1] - pi_i[k]).

    Row ``j`` receives the symmetric contribution for class ``i+1``.  Only legal
    unordered candidates are considered.

    Complexity: O(L^3) in the small stdlib loop because each pair contributes to
    all classes in two rows.  The torch backend uses a vectorized autograd form.
    """

    if negative_weight < 0.0:
        raise ValueError("negative_weight must be non-negative")
    _validate_long_range_params(long_range_min_distance, long_range_weight)
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    size = _validate_shapes(marginals, target_pair_matrix, legal_pair)
    pos_weight, neg_weight_sum = _candidate_weight_sums(
        target_pair_matrix,
        legal_pair,
        long_range_min_distance=long_range_min_distance,
        long_range_weight=long_range_weight,
    )
    grads = [[0.0 for _ in range(size + 1)] for _ in range(size)]
    if lambda_contact == 0.0:
        return grads

    for i in range(size):
        for j in range(i + 1, size):
            if not (bool(legal_pair[i][j]) and bool(legal_pair[j][i])):
                continue
            target = 1.0 if (int(target_pair_matrix[i][j]) or int(target_pair_matrix[j][i])) else 0.0
            weight = _candidate_weight(
                i,
                j,
                long_range_min_distance=long_range_min_distance,
                long_range_weight=long_range_weight,
            )
            if target:
                if pos_weight <= 0.0:
                    continue
                pair_scale = weight / pos_weight
            else:
                if neg_weight_sum <= 0.0 or negative_weight == 0.0:
                    continue
                pair_scale = negative_weight * weight / neg_weight_sum
            prob = 0.5 * (float(marginals[i][j + 1]) + float(marginals[j][i + 1]))
            prob = min(1.0 - eps, max(eps, prob))
            dloss_dprob = pair_scale * (prob - target) / (prob * (1.0 - prob))
            coeff = float(lambda_contact) * 0.5 * dloss_dprob

            class_ij = j + 1
            prob_ij = float(marginals[i][class_ij])
            for k in range(size + 1):
                indicator = 1.0 if k == class_ij else 0.0
                grads[i][k] += coeff * prob_ij * (indicator - float(marginals[i][k]))

            class_ji = i + 1
            prob_ji = float(marginals[j][class_ji])
            for k in range(size + 1):
                indicator = 1.0 if k == class_ji else 0.0
                grads[j][k] += coeff * prob_ji * (indicator - float(marginals[j][k]))
    return grads
