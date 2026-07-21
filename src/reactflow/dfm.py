"""Discrete flow matching (DFM) primitives for RNA 2D partner-class generation.

State space
-----------
For a sequence of length ``L`` the 2D structure is modeled per position ``i`` as
a categorical *partner choice* over ``K = L + 1`` classes:

    class 0        -> unpaired (the empty partner, "varnothing")
    class 1..L     -> paired with sequence index ``class - 1``.

A per-position factorized model outputs the denoised posterior marginal
``pi_i^theta(. | S_t, x)``.  This is the mean-field marginal used by estimator
(A) of the research plan; global legality/symmetry are restored at sampling time
by the greedy projection in :mod:`reactflow.constraints`.

Mixture probability path
-------------------------
We use the *linear / mixture* conditional path between a source distribution
``p0`` and the clean data class ``x1``:

    p_{t|1}(z | x1) = (1 - t) * p0(z) + t * 1[z = x1],   t in [0, 1].

At ``t = 0`` this is the source ``p0``; at ``t = 1`` it is the Kronecker delta at
``x1``.  It is normalized for every ``t`` because ``p0`` is a distribution:

    sum_z p_{t|1}(z|x1) = (1 - t) * 1 + t * 1 = 1.

Its time derivative is constant in ``t``:

    d/dt p_{t|1}(z|x1) = 1[z = x1] - p0(z).

Denoising objective
-------------------
The network predicts the clean-data posterior ``p^theta_{1|t}(x1 | x_t)`` via
per-position logits.  The training loss is the denoising cross-entropy

    L_DFM = E_{t, x1, x_t~p_{t|1}(.|x1)} [ -log p^theta_{1|t}(x1 | x_t) ],

whose gradient w.r.t. the logits is the classic ``softmax(logits) - onehot(x1)``.

Generative rate matrix
----------------------
Sampling integrates a continuous-time Markov chain whose conditional rate matrix
(Campbell et al., 2024) reproduces the mixture path:

    R*_t(z -> j | x1) = ReLU( d/dt p_{t|1}(j|x1) - d/dt p_{t|1}(z|x1) )
                        / ( Z_t * p_{t|1}(z|x1) ),   j != z,

with ``Z_t`` the number of states with positive path mass.  The unconditional
sampling rate marginalizes over the denoiser posterior:

    R^theta_t(z -> j) = sum_{x1} p^theta_{1|t}(x1 | x_t=z) * R*_t(z -> j | x1).

This module is implemented purely with the Python standard library so that the
mathematical core is deterministic, unit-testable, and verifiable with SymPy
before any tensor autodiff backend is introduced.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def softmax(logits: Sequence[float]) -> Tuple[float, ...]:
    """Return the numerically stable softmax of ``logits``.

    Formula: ``p_c = exp(l_c - m) / sum_d exp(l_d - m)`` with ``m = max_c l_c``.
    Subtracting the max prevents overflow without changing the result.

    Complexity: O(C) for ``C`` classes.
    """

    if not logits:
        raise ValueError("logits must be non-empty")
    max_logit = max(float(value) for value in logits)
    exp_values = [math.exp(float(value) - max_logit) for value in logits]
    total = sum(exp_values)
    return tuple(value / total for value in exp_values)


def log_softmax(logits: Sequence[float]) -> Tuple[float, ...]:
    """Return the numerically stable log-softmax of ``logits``.

    Formula: ``log p_c = (l_c - m) - log sum_d exp(l_d - m)``.  Computing the log
    directly avoids taking ``log`` of a possibly tiny softmax probability.

    Complexity: O(C).
    """

    if not logits:
        raise ValueError("logits must be non-empty")
    max_logit = max(float(value) for value in logits)
    shifted = [float(value) - max_logit for value in logits]
    log_sum = math.log(sum(math.exp(value) for value in shifted))
    return tuple(value - log_sum for value in shifted)


def cross_entropy_from_logits(logits: Sequence[float], target_index: int) -> float:
    """Return ``-log softmax(logits)[target_index]``.

    This is the per-position denoising cross-entropy contribution to ``L_DFM``.

    Complexity: O(C).
    """

    if not 0 <= target_index < len(logits):
        raise ValueError("target_index out of range")
    return -log_softmax(logits)[target_index]


def softmax_cross_entropy_gradient(logits: Sequence[float], target_index: int) -> Tuple[float, ...]:
    """Return ``d(cross_entropy)/d(logits) = softmax(logits) - onehot(target)``.

    Derivation: with ``p = softmax(l)`` and loss ``-log p_y``,
    ``dL/dl_c = p_c - 1[c = y]``.  The one-hot subtraction is the only place the
    label enters the gradient.

    Complexity: O(C).
    """

    if not 0 <= target_index < len(logits):
        raise ValueError("target_index out of range")
    probs = softmax(logits)
    return tuple(prob - (1.0 if index == target_index else 0.0) for index, prob in enumerate(probs))


def uniform_source(num_classes: int) -> Tuple[float, ...]:
    """Return the uniform source distribution ``p0(z) = 1/K``.

    Complexity: O(K).
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    value = 1.0 / num_classes
    return tuple(value for _ in range(num_classes))


def mixture_path_marginal(t: float, data_index: int, source: Sequence[float]) -> Tuple[float, ...]:
    """Return ``p_{t|1}(. | x1) = (1 - t) p0 + t onehot(x1)``.

    Complexity: O(K).
    """

    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    num_classes = len(source)
    if not 0 <= data_index < num_classes:
        raise ValueError("data_index out of range")
    return tuple(
        (1.0 - t) * float(prob) + (t if index == data_index else 0.0)
        for index, prob in enumerate(source)
    )


def mixture_path_time_derivative(data_index: int, source: Sequence[float]) -> Tuple[float, ...]:
    """Return ``d/dt p_{t|1}(. | x1) = onehot(x1) - p0`` (constant in ``t``).

    Complexity: O(K).
    """

    num_classes = len(source)
    if not 0 <= data_index < num_classes:
        raise ValueError("data_index out of range")
    return tuple((1.0 if index == data_index else 0.0) - float(prob) for index, prob in enumerate(source))


def sample_path_index(
    t: float,
    data_index: int,
    source: Sequence[float],
    *,
    rng: random.Random,
) -> int:
    """Sample ``x_t ~ p_{t|1}(. | x1)`` for the mixture path.

    Complexity: O(K).
    """

    marginal = mixture_path_marginal(t, data_index, source)
    draw = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(marginal):
        cumulative += prob
        if draw <= cumulative:
            return index
    return len(marginal) - 1


def conditional_rate_matrix(
    t: float,
    data_index: int,
    source: Sequence[float],
) -> Tuple[Tuple[float, ...], ...]:
    """Return the Campbell conditional rate matrix ``R*_t(z -> j | x1)``.

    For off-diagonal entries ``j != z``:

        R*_t(z -> j) = ReLU( d_t p(j) - d_t p(z) ) / ( Z_t * p(z) ),

    where ``d_t p`` is :func:`mixture_path_time_derivative`, ``p`` is
    :func:`mixture_path_marginal`, and ``Z_t`` is the count of states with
    ``p(z) > 0``.  Diagonal entries hold the negative row sum so each row of the
    generator sums to zero (a valid CTMC generator).

    The matrix satisfies the Kolmogorov forward (master) equation
    ``d_t p(z) = sum_{z'} R*_t(z' -> z) p(z')`` for the mixture path; this is
    checked symbolically in :mod:`reactflow.symbolic`.

    Complexity: O(K^2) time and memory.
    """

    if not 0.0 <= t < 1.0:
        raise ValueError("conditional rate matrix requires t in [0, 1)")
    num_classes = len(source)
    marginal = mixture_path_marginal(t, data_index, source)
    derivative = mixture_path_time_derivative(data_index, source)
    support = sum(1 for prob in marginal if prob > 0.0)
    if support == 0:
        raise ValueError("path marginal has empty support")
    rates = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    for z in range(num_classes):
        p_z = marginal[z]
        if p_z <= 0.0:
            continue
        row_sum = 0.0
        for j in range(num_classes):
            if j == z:
                continue
            flux = derivative[j] - derivative[z]
            rate = max(0.0, flux) / (support * p_z)
            rates[z][j] = rate
            row_sum += rate
        rates[z][z] = -row_sum
    return tuple(tuple(row) for row in rates)


def posterior_transition_rates(
    t: float,
    current_index: int,
    posterior: Sequence[float],
    source: Sequence[float],
) -> Tuple[float, ...]:
    """Return the marginal sampling rates ``R^theta_t(current -> .)``.

    The rate row is the posterior expectation of the conditional rates:

        R^theta_t(z -> j) = sum_{x1} posterior[x1] * R*_t(z -> j | x1).

    Only ``current_index`` (``z``) is evaluated, so this returns a single rate
    row (length ``K``) rather than the full matrix.

    For the uniform source used by ReactFlow sampling, the mixture path has the
    closed form ``R(z->j)=posterior[j]/(1-t)`` for ``j != z``.  That fast path is
    O(K); a generic O(K^2) fallback is retained for non-uniform research paths.
    """

    num_classes = len(source)
    if len(posterior) != num_classes:
        raise ValueError("posterior and source must have the same length")
    if not 0 <= current_index < num_classes:
        raise ValueError("current_index out of range")
    if not 0.0 <= t < 1.0:
        raise ValueError("sampling rates require t in [0, 1)")

    if num_classes > 0:
        uniform_value = 1.0 / num_classes
        if all(abs(float(value) - uniform_value) <= 1e-12 for value in source):
            scale = 1.0 / (1.0 - t)
            aggregate = [
                0.0 if index == current_index else max(0.0, float(posterior[index])) * scale
                for index in range(num_classes)
            ]
            aggregate[current_index] = -sum(aggregate)
            return tuple(aggregate)

    marginal_cache = {}
    derivative_cache = {}
    aggregate = [0.0 for _ in range(num_classes)]
    for data_index, weight in enumerate(posterior):
        if weight <= 0.0:
            continue
        if data_index not in marginal_cache:
            marginal_cache[data_index] = mixture_path_marginal(t, data_index, source)
            derivative_cache[data_index] = mixture_path_time_derivative(data_index, source)
        marginal = marginal_cache[data_index]
        derivative = derivative_cache[data_index]
        p_z = marginal[current_index]
        if p_z <= 0.0:
            continue
        support = sum(1 for prob in marginal if prob > 0.0)
        for j in range(num_classes):
            if j == current_index:
                continue
            flux = derivative[j] - derivative[current_index]
            aggregate[j] += float(weight) * max(0.0, flux) / (support * p_z)
    aggregate[current_index] = -sum(aggregate[j] for j in range(num_classes) if j != current_index)
    return tuple(aggregate)


def euler_step_distribution(
    current: Sequence[float],
    rate_rows: Sequence[Sequence[float]],
    dt: float,
) -> Tuple[float, ...]:
    """Advance a categorical distribution by one Euler CTMC step.

    Using generator ``R`` (rows sum to zero) the forward equation is
    ``d/dt p = p R``; the explicit Euler update is ``p_{t+dt} = p_t (I + dt R)``.
    The result is renormalized to correct first-order truncation drift.

    Complexity: O(K^2).
    """

    num_classes = len(current)
    if any(len(row) != num_classes for row in rate_rows) or len(rate_rows) != num_classes:
        raise ValueError("rate matrix must be square and match the distribution")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    updated: List[float] = []
    for j in range(num_classes):
        value = float(current[j]) + dt * sum(float(current[z]) * float(rate_rows[z][j]) for z in range(num_classes))
        updated.append(max(0.0, value))
    total = sum(updated)
    if total <= 0.0:
        raise ValueError("euler step produced a non-positive mass")
    return tuple(value / total for value in updated)


def denoising_cross_entropy(
    logit_rows: Sequence[Sequence[float]],
    target_indices: Sequence[int],
) -> float:
    """Return mean per-position denoising cross-entropy ``L_DFM`` for one sample.

    ``logit_rows[i]`` are the class logits for position ``i`` and
    ``target_indices[i]`` is the clean partner class ``x1_i``.

    Complexity: O(L * K).
    """

    if len(logit_rows) != len(target_indices):
        raise ValueError("logit_rows and target_indices must have the same length")
    if not logit_rows:
        raise ValueError("at least one position is required")
    total = 0.0
    for logits, target in zip(logit_rows, target_indices):
        total += cross_entropy_from_logits(logits, int(target))
    return total / len(logit_rows)
