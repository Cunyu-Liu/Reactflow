"""Differentiable reactivity forward operator and consistency losses.

Core equation
-------------
For probe ``k`` and base type ``t_i`` at position ``i``:

    f_i^(k)(S) = a_{k,t_i} u_i(S) + b_{k,t_i} e_i(S) + c_{k,t_i}.

``u_i`` is the unpaired indicator and ``e_i`` is a helix-edge/fraying context.
Because ``f`` is affine in structure features, expectation commutes:

    E[f_i(S)] = a E[u_i] + b E[e_i] + c.

This module implements the exact affine forward operator and deterministic loss
terms.  It is intentionally independent of PyTorch/JAX so the mathematical core
can be unit-tested before neural parameterization is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from reactflow.constraints import edge_context_indicators, unpaired_indicators
from reactflow.data import normalize_probe_name, probe_base_mask


BASES = ("A", "C", "G", "U")


@dataclass(frozen=True)
class ReactivityParameters:
    """Probe/base-specific affine parameters.

    The default values encode only qualitative chemistry: unpaired A/C receives
    strong DMS signal; 2A3 reports all bases and has a larger edge-context term.
    These are initialization values, not claimed fitted constants.

    Complexity: O(|probes| * |bases|) storage.
    """

    unpaired: Mapping[Tuple[str, str], float]
    edge: Mapping[Tuple[str, str], float]
    bias: Mapping[Tuple[str, str], float]

    @staticmethod
    def defaults() -> "ReactivityParameters":
        """Return physically monotone default parameters.

        Complexity: O(1).
        """

        unpaired: Dict[Tuple[str, str], float] = {}
        edge: Dict[Tuple[str, str], float] = {}
        bias: Dict[Tuple[str, str], float] = {}
        for base in BASES:
            unpaired[("2A3", base)] = 1.0
            edge[("2A3", base)] = 0.25
            bias[("2A3", base)] = 0.02
            if base in {"A", "C"}:
                unpaired[("DMS", base)] = 1.0
                edge[("DMS", base)] = 0.10
                bias[("DMS", base)] = 0.01
            else:
                unpaired[("DMS", base)] = 0.0
                edge[("DMS", base)] = 0.0
                bias[("DMS", base)] = 0.0
        return ReactivityParameters(unpaired=unpaired, edge=edge, bias=bias)

    def coefficient(self, probe: str, base: str) -> Tuple[float, float, float]:
        """Return ``(a,b,c)`` for probe/base with validation.

        Complexity: O(1).
        """

        key = (normalize_probe_name(probe), base.upper())
        if key[1] not in BASES:
            raise ValueError(f"invalid RNA base: {base}")
        return float(self.unpaired[key]), float(self.edge[key]), float(self.bias[key])


@dataclass(frozen=True)
class LossBreakdown:
    """Decomposed reactivity consistency loss.

    Complexity: O(1) summary storage.
    """

    total: float
    magnitude: float
    shape: float
    alpha: float
    gamma: float
    effective_weight_sum: float


class ReactivityForwardOperator:
    """Affine structure-to-reactivity operator.

    The class is stateless except for calibrated parameters, which makes it easy
    to use both in tests and inside future neural training loops.

    Formula: ``f_i(S) = a_i u_i(S) + b_i e_i(S) + c_i`` and therefore
    ``E[f_i(S)] = a_i q_i + b_i E[e_i] + c_i``.  Complexity is O(L^2) from a hard
    pair matrix or O(L) when expectations are already supplied.
    """

    def __init__(self, parameters: Optional[ReactivityParameters] = None) -> None:
        """Create an operator with explicit or default chemistry parameters."""

        self.parameters = parameters or ReactivityParameters.defaults()

    def from_structure(
        self,
        sequence: str,
        pair_matrix: Sequence[Sequence[float]],
        probe: str,
    ) -> Tuple[float, ...]:
        """Compute ``f(S)`` for a concrete hard/soft structure matrix.

        It first derives ``u_i`` and ``e_i`` from the matrix, then delegates to
        ``from_expectations``.  For hard structures the result is exactly the
        single-structure forward model; for soft matrices it is the affine
        expectation under the supplied marginals.

        Complexity: O(L^2) time and O(L) memory.
        """

        return self.from_expectations(
            sequence=sequence,
            unpaired_prob=unpaired_indicators(pair_matrix),
            edge_prob=edge_context_indicators(pair_matrix),
            probe=probe,
        )

    def from_expectations(
        self,
        sequence: str,
        unpaired_prob: Sequence[float],
        edge_prob: Optional[Sequence[float]],
        probe: str,
    ) -> Tuple[float, ...]:
        """Compute ``E[f(S)] = a q_i + b ebar_i + c`` from expectations.

        ``unpaired_prob`` is ``q_i=Pr(u_i=1)`` and ``edge_prob`` is
        ``E[e_i]``.  If ``edge_prob`` is omitted it is treated as zero, giving
        the minimal first-order operator used in early pilots.

        Complexity: O(L).
        """

        sequence = sequence.upper()
        if len(unpaired_prob) != len(sequence):
            raise ValueError("unpaired_prob length must match sequence length")
        if edge_prob is None:
            edge_prob = tuple(0.0 for _ in sequence)
        if len(edge_prob) != len(sequence):
            raise ValueError("edge_prob length must match sequence length")

        probe = normalize_probe_name(probe)
        values: List[float] = []
        for base, q, ebar in zip(sequence, unpaired_prob, edge_prob):
            a, b, c = self.parameters.coefficient(probe, base)
            q_clamped = min(1.0, max(0.0, float(q)))
            e_clamped = min(1.0, max(0.0, float(ebar)))
            values.append(a * q_clamped + b * e_clamped + c)
        return tuple(values)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return ``sum_i w_i x_i / sum_i w_i`` with finite-value filtering.

    Complexity: O(L).
    """

    numerator = 0.0
    denominator = 0.0
    for value, weight in zip(values, weights):
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            numerator += weight * value
            denominator += weight
    if denominator <= 0:
        raise ValueError("sum of valid weights must be positive")
    return numerator / denominator


def weighted_pearson(x: Sequence[float], y: Sequence[float], weights: Sequence[float]) -> float:
    """Compute weighted Pearson correlation.

    Formula:

        corr_w(x,y) = sum w_i (x_i-mu_x)(y_i-mu_y)
                      / sqrt(sum w_i (x_i-mu_x)^2 sum w_i (y_i-mu_y)^2).

    If either variance is zero, the shape signal is undefined; the function
    returns 0 so the shape loss ``1-corr`` becomes neutral-to-bad rather than
    producing NaN.

    Complexity: O(L).
    """

    if not (len(x) == len(y) == len(weights)):
        raise ValueError("x, y and weights must have the same length")
    valid = [
        (float(a), float(b), float(w))
        for a, b, w in zip(x, y, weights)
        if math.isfinite(a) and math.isfinite(b) and math.isfinite(w) and w > 0
    ]
    if not valid:
        raise ValueError("no finite weighted observations")
    vx, vy, vw = zip(*valid)
    mux = weighted_mean(vx, vw)
    muy = weighted_mean(vy, vw)
    cov = sum(w * (a - mux) * (b - muy) for a, b, w in valid)
    varx = sum(w * (a - mux) ** 2 for a, _, w in valid)
    vary = sum(w * (b - muy) ** 2 for _, b, w in valid)
    if varx <= 0 or vary <= 0:
        return 0.0
    return cov / math.sqrt(varx * vary)


def fit_weighted_affine_calibration(
    predicted: Sequence[float],
    target: Sequence[float],
    weights: Sequence[float],
) -> Tuple[float, float]:
    """Fit ``alpha, gamma`` minimizing ``sum w_i (alpha*x_i+gamma-y_i)^2``.

    Normal equations:

        [Sxx Sx] [alpha] = [Sxy]
        [Sx  Sw] [gamma]   [Sy ]

    where ``Sxx=sum w x^2``, ``Sx=sum w x``, ``Sw=sum w``,
    ``Sxy=sum w x y`` and ``Sy=sum w y``.  If the determinant is singular,
    the best stable fallback is ``alpha=1`` and a weighted mean shift.

    Complexity: O(L).
    """

    if not (len(predicted) == len(target) == len(weights)):
        raise ValueError("predicted, target and weights must have the same length")
    valid = [
        (float(x), float(y), float(w))
        for x, y, w in zip(predicted, target, weights)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and w > 0
    ]
    if not valid:
        raise ValueError("no finite weighted observations")
    sxx = sum(w * x * x for x, _, w in valid)
    sx = sum(w * x for x, _, w in valid)
    sw = sum(w for _, _, w in valid)
    sxy = sum(w * x * y for x, y, w in valid)
    sy = sum(w * y for _, y, w in valid)
    determinant = sxx * sw - sx * sx
    if abs(determinant) < 1e-12:
        gamma = sy / sw - sx / sw
        return 1.0, gamma
    alpha = (sxy * sw - sy * sx) / determinant
    gamma = (sxx * sy - sx * sxy) / determinant
    return alpha, gamma


def weighted_mse(predicted: Sequence[float], target: Sequence[float], weights: Sequence[float]) -> float:
    """Return ``sum w_i (pred_i-target_i)^2 / sum w_i``.

    Complexity: O(L).
    """

    numerator = 0.0
    denominator = 0.0
    for pred, truth, weight in zip(predicted, target, weights):
        if math.isfinite(pred) and math.isfinite(truth) and math.isfinite(weight) and weight > 0:
            numerator += weight * (pred - truth) ** 2
            denominator += weight
    if denominator <= 0:
        raise ValueError("sum of valid weights must be positive")
    return numerator / denominator


def reactivity_consistency_loss(
    predicted: Sequence[float],
    target: Sequence[float],
    weights: Sequence[float],
    *,
    lambda_magnitude: float = 1.0,
    lambda_shape: float = 1.0,
    calibrate: bool = True,
) -> LossBreakdown:
    """Compute calibrated magnitude + shape reactivity loss.

    The implemented objective is

        L = lambda_mag * MSE_w(alpha * pred + gamma, target)
            + lambda_shape * (1 - corr_w(pred, target)).

    The shape term intentionally uses uncalibrated predictions because Pearson
    is already shift/scale invariant.  Calibration only affects the magnitude
    term.  This mirrors the proposal's separation between absolute scale and
    profile shape.

    Complexity: O(L).
    """

    if not (len(predicted) == len(target) == len(weights)):
        raise ValueError("predicted, target and weights must have the same length")
    if calibrate:
        alpha, gamma = fit_weighted_affine_calibration(predicted, target, weights)
    else:
        alpha, gamma = 1.0, 0.0
    calibrated = tuple(alpha * float(value) + gamma for value in predicted)
    magnitude = weighted_mse(calibrated, target, weights)
    shape = 1.0 - weighted_pearson(predicted, target, weights)
    total = lambda_magnitude * magnitude + lambda_shape * shape
    effective_weight_sum = sum(
        float(w)
        for value, truth, w in zip(predicted, target, weights)
        if math.isfinite(value) and math.isfinite(truth) and math.isfinite(w) and w > 0
    )
    return LossBreakdown(
        total=total,
        magnitude=magnitude,
        shape=shape,
        alpha=alpha,
        gamma=gamma,
        effective_weight_sum=effective_weight_sum,
    )


def masked_unit_weights(sequence: str, probe: str, target: Sequence[float]) -> Tuple[float, ...]:
    """Build unit weights from probe/base validity and finite targets.

    This convenience function is useful for quick pilots before experimental
    error fields are loaded.  Complexity: O(L).
    """

    base_mask = probe_base_mask(sequence.upper(), probe)
    return tuple(1.0 if keep and math.isfinite(float(value)) else 0.0 for keep, value in zip(base_mask, target))
