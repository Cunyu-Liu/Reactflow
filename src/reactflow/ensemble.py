"""Variance-aware (heteroscedastic) reactivity observation model.

Scientific motivation
----------------------
Chemical probing does not label a single structure; it reports a signal averaged
over the structural *ensemble* ``p_theta(S | x)``.  ReactFlow's first-moment term
only matches the mean ``E[f_i(S)]`` to the measured reactivity ``r_i``, which is
fundamentally *unidentifiable*: many ensembles share the same mean profile.  A
minimal, physically meaningful extra constraint is the ensemble's own
*dispersion*.

Under the factorized mean-field structure posterior, position ``i`` is unpaired
with probability ``q_i = pi_i[0]`` (a Bernoulli variable ``u_i``).  With the
minimal affine operator ``f_i = a_i u_i + c_i`` the induced structural variance of
the reactivity at position ``i`` is

    Var[f_i] = a_i^2 Var[u_i] = a_i^2 q_i (1 - q_i).

We therefore model the observed reactivity as heteroscedastic Gaussian

    r_i ~ N( mu_i, v_i ),
    mu_i = alpha (a_i q_i + c_i) + gamma,
    v_i  = beta * a_i^2 q_i (1 - q_i) + tau^2,

where ``alpha, gamma`` are the affine calibration already used by the first-moment
term, ``beta >= 0`` scales the structural variance into reactivity units, and
``tau^2 > 0`` is a measurement-noise floor that keeps the likelihood finite even
when the model is confident (``q_i in {0, 1}``).

The training term is the per-position weighted negative log-likelihood (dropping
the constant ``0.5 log 2pi``):

    ell_calib = (1 / W) sum_i w_i [ (mu_i - r_i)^2 / (2 v_i) + 0.5 log v_i ],
    W = sum_i w_i   (over finite, positively-weighted positions).

Why this is *custom*, not generic
----------------------------------
This is not a generic entropy or dropout regularizer: the variance ``v_i`` is
derived from the *same* Bernoulli structure posterior that produces the mean, and
from the *same* probe coefficient ``a_i``.  It couples confidence and residual in
the direction the chemistry dictates -- a position the model claims is confidently
paired/unpaired (low ``q_i(1-q_i)``) is given a small tolerance, so a large
reactivity residual there is penalized hard; a structurally ambiguous position is
allowed a larger residual.  This directly targets hypothesis H4 (ensemble
identifiability) in the project README.

Reduction / safety
------------------
With ``beta = 0`` the variance is the constant ``tau^2`` and the term reduces to a
scaled weighted MSE plus a constant, so it never contradicts the first-moment
objective.  Callers keep ``lambda_calib = 0`` by default, making the whole term a
no-op so existing training trajectories are bit-for-bit unchanged.

Every gradient below is derived by hand, checked against SymPy
(:mod:`reactflow.symbolic`) and against central finite differences in the test
suite, matching the rest of the project's rigor contract.  The module is pure
standard library.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


_VAR_FLOOR = 1e-6


def structural_variance(
    q_values: Sequence[float],
    a_values: Sequence[float],
    *,
    beta: float,
    tau_squared: float,
) -> Tuple[float, ...]:
    """Return per-position reactivity variance ``v_i = beta a_i^2 q_i(1-q_i)+tau^2``.

    ``tau_squared`` is floored at :data:`_VAR_FLOOR` so the downstream Gaussian
    log-likelihood and its gradient stay finite for confident positions.

    Complexity: O(L).
    """

    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    floor = max(float(tau_squared), _VAR_FLOOR)
    variances: List[float] = []
    for q, a in zip(q_values, a_values):
        qc = min(1.0, max(0.0, float(q)))
        var = float(beta) * float(a) * float(a) * qc * (1.0 - qc) + floor
        variances.append(var)
    return tuple(variances)


def heteroscedastic_reactivity_nll(
    q_values: Sequence[float],
    target: Sequence[float],
    weights: Sequence[float],
    a_values: Sequence[float],
    c_values: Sequence[float],
    *,
    alpha: float,
    gamma: float,
    beta: float,
    tau_squared: float,
) -> float:
    """Return the weighted heteroscedastic Gaussian NLL ``ell_calib``.

    ``mu_i = alpha (a_i q_i + c_i) + gamma`` and ``v_i`` from
    :func:`structural_variance`.  Only finite, positively-weighted positions
    contribute; the sum is normalized by the effective weight ``W``.  Returns
    ``0.0`` when no position is valid so the term is a safe no-op on empty masks.

    Complexity: O(L).
    """

    size = len(q_values)
    if not (len(target) == len(weights) == len(a_values) == len(c_values) == size):
        raise ValueError("all inputs must share the sequence length")
    variances = structural_variance(q_values, a_values, beta=beta, tau_squared=tau_squared)
    weight_sum = 0.0
    total = 0.0
    for i in range(size):
        w = float(weights[i])
        r = float(target[i])
        if not (math.isfinite(w) and w > 0.0 and math.isfinite(r)):
            continue
        q = min(1.0, max(0.0, float(q_values[i])))
        mu = float(alpha) * (float(a_values[i]) * q + float(c_values[i])) + float(gamma)
        v = variances[i]
        residual = mu - r
        total += w * (residual * residual / (2.0 * v) + 0.5 * math.log(v))
        weight_sum += w
    if weight_sum <= 0.0:
        return 0.0
    return total / weight_sum


def heteroscedastic_reactivity_logit_gradient(
    marginals: Sequence[Sequence[float]],
    target: Sequence[float],
    weights: Sequence[float],
    a_values: Sequence[float],
    c_values: Sequence[float],
    *,
    alpha: float,
    gamma: float,
    beta: float,
    tau_squared: float,
    lambda_calib: float,
) -> List[List[float]]:
    """Exact gradient of ``lambda_calib * ell_calib`` into every logit row.

    Writing ``q = pi_i[0]``, ``mu = alpha (a q + c) + gamma`` and
    ``v = beta a^2 q(1-q) + tau^2`` the per-position sensitivity is

        s_i = d/dq [ (mu-r)^2/(2v) + 0.5 log v ]
            = (mu - r) * (alpha a) / v
              + ( 0.5/v - (mu - r)^2 / (2 v^2) ) * beta a^2 (1 - 2q).

    The first summand is the mean channel (identical in spirit to the
    first-moment gradient); the second is the *variance channel* that is unique to
    this observation model.  The logit gradient then uses the class-0 softmax
    Jacobian ``d q / d logit_i[k] = q (1[k=0] - pi_i[k])`` exactly as the
    reactivity-magnitude and thermodynamic terms do:

        d ell / d logit_i[k] = lambda_calib * (w_i / W) * s_i * q * (1[k=0] - pi_i[k]).

    Complexity: O(L * K).
    """

    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    floor = max(float(tau_squared), _VAR_FLOOR)
    size = len(marginals)
    if not (len(target) == len(weights) == len(a_values) == len(c_values) == size):
        raise ValueError("all inputs must share the sequence length")

    weight_sum = 0.0
    for i in range(size):
        w = float(weights[i])
        if math.isfinite(w) and w > 0.0 and math.isfinite(float(target[i])):
            weight_sum += w

    grads: List[List[float]] = []
    for i in range(size):
        row = marginals[i]
        num_classes = len(row)
        grad_row = [0.0 for _ in range(num_classes)]
        w = float(weights[i])
        r = float(target[i])
        if weight_sum > 0.0 and math.isfinite(w) and w > 0.0 and math.isfinite(r):
            q = float(row[0])
            a = float(a_values[i])
            c = float(c_values[i])
            mu = float(alpha) * (a * q + c) + float(gamma)
            v = float(beta) * a * a * q * (1.0 - q) + floor
            residual = mu - r
            dmu_dq = float(alpha) * a
            dv_dq = float(beta) * a * a * (1.0 - 2.0 * q)
            s_i = residual * dmu_dq / v + (0.5 / v - residual * residual / (2.0 * v * v)) * dv_dq
            coeff = float(lambda_calib) * (w / weight_sum) * s_i
            for k in range(num_classes):
                indicator = 1.0 if k == 0 else 0.0
                grad_row[k] = coeff * q * (indicator - float(row[k]))
        grads.append(grad_row)
    return grads
