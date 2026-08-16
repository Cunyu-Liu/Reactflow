"""Odd nonlinear switch operator (v3 EPRO §4.5, Phase O0 T-O0.9).

Models finite-amplitude conformational switching (riboSNitch / strand
displacement). The switch gate is endpoint-swap-invariant and the nonlinear
response is an *odd* function of the linear response, so the total response
satisfies ``h(-b) = -h(b)``.

Construction (§4.5):

    f = F(z_bar, |b|, c)          # fragility, swap-invariant
    pi = sigmoid(f)               # switch gate in [0, 1], swap-invariant
    h_nl = pi * tanh(S(z_bar) @ h_lin)   # nonlinear response
    h = h_lin + h_nl              # total latent accessibility response

Oddness proof (no bias, tanh odd, S independent of b):

  * ``h_lin(-b) = (I-K)^{-1}(-b) = -h_lin(b)``  (K is b-independent);
  * ``tanh(-x) = -tanh(x)``  (tanh is odd);
  * ``S(z_bar)`` depends only on ``z_bar`` (b-independent);
  * ``pi`` depends on ``z_bar`` and ``|b|`` (both even under ``b -> -b``);
  * therefore ``h_nl(-b) = pi * tanh(S @ (-h_lin)) = pi * (-tanh(S @ h_lin)) =
    -h_nl(b)``;
  * and ``h(-b) = h_lin(-b) + h_nl(-b) = -h_lin(b) - h_nl(b) = -h(b)``.

The switch has *no bias* term: ``tanh(0) = 0`` and ``S`` has no additive bias.
``no_switch`` ablation (``pi = 0``) is supported via :func:`no_switch_response`.

Numpy-only (runs in ``editflow311`` without torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

SWITCH_SCHEMA_VERSION = "reactflow-delta-o0-switch-v1"


# ---------------------------------------------------------------------------
# Fragility and switch gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchResult:
    """Output of :func:`compute_switch`."""

    fragility: np.ndarray  # (n,) f
    gate: np.ndarray  # (n,) pi = sigmoid(f), swap-invariant
    h_nl: np.ndarray  # (n,) nonlinear response
    h: np.ndarray  # (n,) total response h_lin + h_nl
    gate_mean: float
    gate_max: float
    no_bias: bool
    schema_version: str = SWITCH_SCHEMA_VERSION

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_mean": float(self.gate_mean),
            "gate_max": float(self.gate_max),
            "no_bias": bool(self.no_bias),
            "h_norm": float(np.linalg.norm(self.h)),
            "h_nl_norm": float(np.linalg.norm(self.h_nl)),
        }


def _default_fragility(
    z_bar_features: np.ndarray, abs_b: np.ndarray
) -> np.ndarray:
    """Swap-invariant fragility ``f = F(z_bar, |b|)``.

    ``z_bar_features`` is a ``(n_features, n)`` stack of swap-invariant
    background features; ``abs_b`` is ``|b|`` (even under ``b -> -b``). Both
    inputs are swap-invariant, so ``f`` is swap-invariant.
    """

    # Magnitude of symmetric background + |b| contribution.
    mag = np.sum(np.abs(z_bar_features), axis=0) + np.abs(abs_b)
    # Centered, non-negative fragility.
    f = np.log1p(np.exp(np.clip(0.3 * mag - 1.0, -30.0, 30.0)))
    return f


def _default_S(z_bar_features: np.ndarray) -> np.ndarray:
    """Swap-invariant, bias-free mixing matrix ``S(z_bar)`` (no directional bias).

    ``S`` is symmetric and has zero diagonal (no self-bias). It depends only on
    the swap-invariant ``z_bar`` and carries no additive bias.
    """

    n = z_bar_features.shape[1]
    node_mag = np.sum(np.abs(z_bar_features), axis=0)
    # Symmetric off-diagonal mixing (no directional bias => S symmetric).
    S = np.sqrt(np.maximum(node_mag[:, None] * node_mag[None, :], 0.0))
    S = 0.1 * S  # small stable scale
    np.fill_diagonal(S, 0.0)  # no self-bias
    # Symmetrize exactly.
    S = 0.5 * (S + S.T)
    return S


def compute_switch(
    h_lin: np.ndarray,
    z_bar_features: np.ndarray,
    b: np.ndarray,
    *,
    fragility_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    S_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    gate_offset: float = 0.0,
) -> SwitchResult:
    """Compute the odd nonlinear switch response (§4.5).

    Parameters
    ----------
    h_lin : (n,) array
        Linear response from the stable solver.
    z_bar_features : (n_features, n) array
        Swap-invariant symmetric background features (e.g. from
        :meth:`SymmetricBackground.stack`).
    b : (n,) array
        Node forcing vector (used for ``|b|`` in fragility).
    fragility_fn, S_fn : callable, optional
        Override the default swap-invariant, bias-free constructions.
    gate_offset : float
        Scalar added to fragility before sigmoid. Default 0 keeps the gate
        centered. Must not introduce b-dependence.
    """

    h_lin = np.asarray(h_lin, dtype=float)
    z_bar_features = np.asarray(z_bar_features, dtype=float)
    b = np.asarray(b, dtype=float)

    ffn = fragility_fn or _default_fragility
    sfn = S_fn or _default_S

    f = ffn(z_bar_features, np.abs(b)) + float(gate_offset)
    pi = 1.0 / (1.0 + np.exp(-f))  # sigmoid, swap-invariant
    S = sfn(z_bar_features)  # symmetric, no bias
    h_nl = pi * np.tanh(S @ h_lin)
    h = h_lin + h_nl

    return SwitchResult(
        fragility=f,
        gate=pi,
        h_nl=h_nl,
        h=h,
        gate_mean=float(np.mean(pi)),
        gate_max=float(np.max(pi)),
        no_bias=True,
    )


def no_switch_response(h_lin: np.ndarray) -> np.ndarray:
    """``no_switch`` ablation: return ``h_lin`` unchanged (§4.5, §5.2).

    With the switch gate disabled (``pi = 0``), ``h_nl = 0`` and ``h = h_lin``.
    """

    return np.asarray(h_lin, dtype=float).copy()


def check_switch_oddness(
    h_lin: np.ndarray,
    z_bar_features: np.ndarray,
    b: np.ndarray,
    *,
    odd_tol: float = 1e-6,
    **kwargs: Any,
) -> dict[str, Any]:
    """Audit the odd switch invariant ``h(-b) = -h(b)`` (§5.3).

    ``h_lin`` must be the linear response to ``b``; the audit recomputes the
    switch with ``-b`` (and ``-h_lin``) and checks ``h(-b) == -h(b)``.
    """

    res_pos = compute_switch(h_lin, z_bar_features, b, **kwargs)
    res_neg = compute_switch(-h_lin, z_bar_features, -b, **kwargs)

    odd_err = float(np.max(np.abs(res_neg.h + res_pos.h))) if res_pos.h.size else 0.0
    gate_invar_err = float(np.max(np.abs(res_neg.gate - res_pos.gate))) if res_pos.gate.size else 0.0
    h_nl_odd_err = float(np.max(np.abs(res_neg.h_nl + res_pos.h_nl))) if res_pos.h_nl.size else 0.0

    # No-bias check: switch of zero h_lin must be zero h_nl.
    res_zero = compute_switch(np.zeros_like(h_lin), z_bar_features, b, **kwargs)
    zero_bias_err = float(np.max(np.abs(res_zero.h_nl))) if res_zero.h_nl.size else 0.0

    return {
        "oddness": {
            "max_abs_err": odd_err,
            "tol": odd_tol,
            "pass": odd_err < odd_tol,
        },
        "gate_swap_invariance": {
            "max_abs_err": gate_invar_err,
            "pass": gate_invar_err < 1e-12,
        },
        "h_nl_oddness": {
            "max_abs_err": h_nl_odd_err,
            "pass": h_nl_odd_err < odd_tol,
        },
        "no_bias": {
            "max_abs_err": zero_bias_err,
            "pass": zero_bias_err < odd_tol,
        },
    }
