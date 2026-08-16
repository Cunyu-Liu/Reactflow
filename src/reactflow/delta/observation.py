"""Monotone probe observation operator (v3 EPRO §4.6, Phase O0 T-O0.10).

Maps the latent accessibility response ``h`` to a predicted probe reactivity
difference ``Delta r`` via a *monotone* observation function with probe-specific
heads.

Construction (§4.6):

    a_bar = A(z_bar)                    # midpoint accessibility
    a_m = a_bar + h / 2                 # mutant endpoint
    a_w = a_bar - h / 2                 # WT endpoint
    Delta r_hat = f_p(a_m) - f_p(a_w)   # probe-specific monotone observation

Monotonicity is enforced *by parameterization*: each observation head ``f_p``
is a non-negative combination of fixed non-decreasing basis functions, so its
derivative is non-negative everywhere on the frozen domain. DMS and SHAPE/2A3
use separate heads. The observation head does *not* read the study ID.

Because ``f_p`` is applied identically to ``a_m`` and ``a_w``, the observation
inherits the endpoint-swap antisymmetry: swapping endpoints flips the sign of
``h`` (hence ``a_m <-> a_w``) and therefore flips the sign of ``Delta r_hat``.

Numpy-only (runs in ``editflow311`` without torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

OBSERVATION_SCHEMA_VERSION = "reactflow-delta-o0-observation-v1"

# Frozen probe domain (accessibility values are bounded for monotonicity audit).
PROBE_DOMAIN_MIN = -6.0
PROBE_DOMAIN_MAX = 6.0

# Supported probe chemistries (separate heads, §4.6).
SUPPORTED_PROBES = frozenset({"DMS", "SHAPE", "2A3"})


# ---------------------------------------------------------------------------
# Monotone basis functions
# ---------------------------------------------------------------------------


def _monotone_basis(a: np.ndarray) -> np.ndarray:
    """Fixed non-decreasing basis functions on accessibility ``a``.

    Returns a ``(n_basis, n)`` array. Each row is a non-decreasing function of
    ``a``. Used as a fixed basis; the observation head forms a non-negative
    linear combination, which is non-decreasing by construction.

    Basis:
      1. identity (linear, slope 1)
      2. softplus (non-decreasing, smooth ReLU)
      3. tanh (monotone, bounded)
      4. cubic-soft (a^3 clipped, monotone on the frozen domain)
    """

    a = np.asarray(a, dtype=float)
    b1 = a
    b2 = np.log1p(np.exp(np.clip(a, -30.0, 30.0)))  # softplus
    b3 = np.tanh(a)
    b4 = np.clip(a, -3.0, 3.0) ** 3  # monotone on frozen domain
    return np.stack([b1, b2, b3, b4], axis=0)


_N_BASIS = 4


# ---------------------------------------------------------------------------
# Observation head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationHead:
    """Probe-specific monotone observation head ``f_p`` (§4.6).

    ``weights`` is a non-negative vector of length ``n_basis``; the head is
    ``f_p(a) = sum_k weights_k * basis_k(a)``, which is non-decreasing because
    every basis function is non-decreasing and weights are non-negative.
    """

    probe: str
    weights: np.ndarray  # (n_basis,) non-negative
    domain_min: float = PROBE_DOMAIN_MIN
    domain_max: float = PROBE_DOMAIN_MAX

    def __post_post_init__(self) -> None:
        pass

    def evaluate(self, a: np.ndarray) -> np.ndarray:
        """Evaluate the monotone head ``f_p(a)``."""

        a = np.asarray(a, dtype=float)
        basis = _monotone_basis(a)
        return self.weights @ basis  # (n,)

    def derivative(self, a: np.ndarray) -> np.ndarray:
        """Analytic derivative ``f_p'(a)`` (must be >= 0 on the domain)."""

        a = np.asarray(a, dtype=float)
        # d/da of each basis:
        d1 = np.ones_like(a)
        d2 = 1.0 / (1.0 + np.exp(-np.clip(a, -30.0, 30.0)))  # sigmoid
        d3 = 1.0 - np.tanh(a) ** 2  # sech^2
        d4 = 3.0 * np.clip(np.clip(a, -3.0, 3.0), -3.0, 3.0) ** 2
        # On the clipped region d4 derivative of clip is 0 outside, but inside
        # the frozen domain the cubic is monotone. Recompute carefully:
        d4 = np.where(np.abs(a) <= 3.0, 3.0 * a ** 2, 0.0)
        deriv = np.stack([d1, d2, d3, d4], axis=0)
        return self.weights @ deriv  # (n,)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "weights": [float(w) for w in self.weights],
            "weights_non_negative": bool(np.all(self.weights >= -1e-15)),
            "domain_min": float(self.domain_min),
            "domain_max": float(self.domain_max),
            "monotone": bool(np.all(self.weights >= -1e-15)),
        }


def default_head(probe: str) -> ObservationHead:
    """Construct a default monotone head for a probe chemistry."""

    if probe not in SUPPORTED_PROBES:
        raise ValueError(f"unsupported probe {probe!r}; supported: {sorted(SUPPORTED_PROBES)}")
    # Distinct non-negative weights per probe (DMS more linear, SHAPE/2A3 softer).
    table = {
        "DMS": np.array([0.6, 0.2, 0.1, 0.05], dtype=float),
        "SHAPE": np.array([0.4, 0.3, 0.2, 0.05], dtype=float),
        "2A3": np.array([0.35, 0.35, 0.2, 0.05], dtype=float),
    }
    return ObservationHead(probe=probe, weights=table[probe])


# ---------------------------------------------------------------------------
# Observation operator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationResult:
    """Output of :func:`observe_delta_r`."""

    delta_r_hat: np.ndarray  # (n,) predicted reactivity difference
    a_bar: np.ndarray  # (n,) midpoint accessibility
    a_m: np.ndarray  # (n,) mutant endpoint accessibility
    a_w: np.ndarray  # (n,) WT endpoint accessibility
    probe: str
    head: ObservationHead
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "probe": self.probe,
            "head": self.head.to_audit_dict(),
            "delta_r_hat_max_abs": float(np.max(np.abs(self.delta_r_hat)))
            if self.delta_r_hat.size
            else 0.0,
        }


def _default_midpoint_accessibility(z_bar_features: np.ndarray) -> np.ndarray:
    """``a_bar = A(z_bar)`` midpoint accessibility from symmetric background."""

    z_bar_features = np.asarray(z_bar_features, dtype=float)
    # Linear map from symmetric background to midpoint accessibility.
    return 0.5 * z_bar_features[0] - 0.1 * z_bar_features[2]  # sum/2 - 0.1*|diff|


def observe_delta_r(
    h: np.ndarray,
    z_bar_features: np.ndarray,
    *,
    probe: str = "DMS",
    head: ObservationHead | None = None,
    accessibility_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ObservationResult:
    """Predict ``Delta r_hat`` from the latent response ``h`` (§4.6).

    Parameters
    ----------
    h : (n,) array
        Total latent accessibility response (linear + nonlinear).
    z_bar_features : (n_features, n) array
        Swap-invariant symmetric background features.
    probe : str
        Probe chemistry (selects the observation head).
    head : ObservationHead, optional
        Override the default monotone head.
    accessibility_fn : callable, optional
        Override the default midpoint accessibility function.
    """

    h = np.asarray(h, dtype=float)
    z_bar_features = np.asarray(z_bar_features, dtype=float)

    if head is None:
        head = default_head(probe)
    afn = accessibility_fn or _default_midpoint_accessibility

    a_bar = afn(z_bar_features)
    a_m = a_bar + h / 2.0
    a_w = a_bar - h / 2.0
    delta_r = head.evaluate(a_m) - head.evaluate(a_w)

    return ObservationResult(
        delta_r_hat=delta_r,
        a_bar=a_bar,
        a_m=a_m,
        a_w=a_w,
        probe=probe,
        head=head,
    )


# ---------------------------------------------------------------------------
# Invariant audit
# ---------------------------------------------------------------------------


def check_observation_invariants(
    h: np.ndarray,
    z_bar_features: np.ndarray,
    *,
    probes: tuple[str, ...] = ("DMS", "SHAPE", "2A3"),
    monotonicity_tol: float = 1e-9,
) -> dict[str, Any]:
    """Audit probe monotonicity and swap-antisymmetry of observation (§5.3).

    Monotonicity: for each probe head, ``f_p'(a) >= 0`` everywhere on the
    frozen domain ``[PROBE_DOMAIN_MIN, PROBE_DOMAIN_MAX]``.
    """

    # Sweep the frozen domain densely.
    a_grid = np.linspace(PROBE_DOMAIN_MIN, PROBE_DOMAIN_MAX, 401)

    per_probe: dict[str, Any] = {}
    for p in probes:
        head = default_head(p)
        deriv = head.derivative(a_grid)
        min_deriv = float(np.min(deriv))
        per_probe[p] = {
            "min_derivative": min_deriv,
            "weights_non_negative": bool(np.all(head.weights >= -1e-15)),
            "pass": min_deriv >= -monotonicity_tol,
        }

    # Swap antisymmetry: observe with -h flips sign of delta_r.
    res_pos = observe_delta_r(h, z_bar_features, probe="DMS")
    res_neg = observe_delta_r(-h, z_bar_features, probe="DMS")
    swap_err = float(np.max(np.abs(res_pos.delta_r_hat + res_neg.delta_r_hat))) if res_pos.delta_r_hat.size else 0.0

    return {
        "monotonicity": {
            "per_probe": per_probe,
            "all_pass": all(v["pass"] for v in per_probe.values()),
        },
        "swap_antisymmetry": {
            "max_abs_err": swap_err,
            "pass": swap_err < 1e-6,
        },
        "domain": {"min": PROBE_DOMAIN_MIN, "max": PROBE_DOMAIN_MAX},
    }
