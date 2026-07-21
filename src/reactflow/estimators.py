"""Expectation estimators for reactivity-consistency training.

The proposal uses two practical estimators:

1. Mean-field denoising marginals:
       E[f_i(S)] = a q_i + b ebar_i + c.
2. Monte Carlo samples:
       E[f_i(S)] ~= (1/M) sum_m f_i(S_m).

This module implements both estimators plus a Gumbel-Softmax partner relaxation
that can later be replaced by an autodiff tensor backend without changing the
mathematical contract.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from reactflow.reactivity import ReactivityForwardOperator


def mean_field_expected_reactivity(
    sequence: str,
    unpaired_prob: Sequence[float],
    *,
    probe: str,
    edge_prob: Optional[Sequence[float]] = None,
    operator: Optional[ReactivityForwardOperator] = None,
) -> Tuple[float, ...]:
    """Estimate expected reactivity from denoised marginals.

    Formula:
        ``rhat_i = a_{k,t_i} q_i + b_{k,t_i} ebar_i + c_{k,t_i}``.

    Complexity: O(L).
    """

    op = operator or ReactivityForwardOperator()
    return op.from_expectations(sequence, unpaired_prob, edge_prob, probe)


def monte_carlo_expected_reactivity(
    sequence: str,
    structures: Sequence[Sequence[Sequence[float]]],
    *,
    probe: str,
    operator: Optional[ReactivityForwardOperator] = None,
) -> Tuple[float, ...]:
    """Estimate expected reactivity from sampled structures.

    The estimator

        ``rhat_i = (1/M) sum_m f_i(S_m)``

    is unbiased for independent samples from ``p_theta(S|x)``.  The function is
    deterministic for a fixed input list of sampled matrices.

    Complexity: O(M L^2) because each structure requires matrix-derived
    unpaired/edge features.
    """

    if not structures:
        raise ValueError("at least one sampled structure is required")
    op = operator or ReactivityForwardOperator()
    total = [0.0 for _ in sequence]
    for matrix in structures:
        values = op.from_structure(sequence, matrix, probe)
        if len(values) != len(total):
            raise ValueError("sampled structure length does not match sequence")
        for index, value in enumerate(values):
            total[index] += value
    count = float(len(structures))
    return tuple(value / count for value in total)


def gumbel_softmax_partner_probabilities(
    logits: Sequence[float],
    *,
    temperature: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Tuple[float, ...]:
    """Sample a relaxed categorical partner vector via Gumbel-Softmax.

    For logits ``l_c`` and iid ``g_c=-log(-log(U_c))``:

        y_c = exp((l_c + g_c)/tau) / sum_d exp((l_d + g_d)/tau).

    Lower ``tau`` approaches a one-hot categorical sample; higher ``tau`` yields
    smoother probabilities.  This is the scalar-list equivalent of the standard
    differentiable estimator used in tensor frameworks.

    Complexity: O(C), C = number of partner categories.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not logits:
        raise ValueError("logits must be non-empty")
    local_rng = rng or random.Random()
    perturbed: List[float] = []
    for logit in logits:
        u = min(1.0 - 1e-12, max(1e-12, local_rng.random()))
        gumbel = -math.log(-math.log(u))
        perturbed.append((float(logit) + gumbel) / temperature)
    max_value = max(perturbed)
    exp_values = [math.exp(value - max_value) for value in perturbed]
    total = sum(exp_values)
    return tuple(value / total for value in exp_values)


def hard_partner_from_relaxed(probabilities: Sequence[float]) -> Tuple[int, ...]:
    """Convert relaxed partner probabilities to a one-hot straight-through value.

    This function implements the forward hardening step of straight-through
    Gumbel-Softmax.  In autodiff frameworks, the backward pass would reuse the
    relaxed probabilities; here we expose the deterministic hard value for tests
    and non-autodiff pilots.

    Complexity: O(C).
    """

    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    total = sum(float(p) for p in probabilities)
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    index = max(range(len(probabilities)), key=lambda idx: float(probabilities[idx]))
    return tuple(1 if idx == index else 0 for idx in range(len(probabilities)))
