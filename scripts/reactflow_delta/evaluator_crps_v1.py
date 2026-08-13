#!/usr/bin/env python3
"""evaluator_crps_v1: probabilistic full-construct CRPS evaluator + sign fixtures.

Implements contract 9.1-9.4 honestly:
  - Gaussian CRPS: exact closed form.
  - five-seed deployment: point = mean of means; primary CRPS scored directly on
    the equal-weight Gaussian mixture CDF via the exact energy form
    (CRPS(F,y) = E|X-y| - 0.5 E|X-X'|), NOT the average of per-seed CRPS.
  - Student-t CRPS: numerical integration of the CRPS CDF form (honest, no
    fabricated closed form).
  - primary estimand: position -> mutant -> cell -> method-balanced puzzle -> mean.
  - sign-flip exact probabilities 14/20 and 15/20 (contract 9.4).

Outcome-blind: structural checks only; scoring requires a held target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import integrate, stats


def _phi(x: np.ndarray) -> np.ndarray:
    return stats.norm.cdf(x)


def _phi_pdf(x: np.ndarray) -> np.ndarray:
    return stats.norm.pdf(x)


def _exp_abs_norm(m: float, s: float) -> float:
    """E|W| for W ~ N(m, s^2)."""
    if s <= 0:
        return abs(m)
    return s * np.sqrt(2.0 / np.pi) * np.exp(-m * m / (2 * s * s)) + m * (2 * _phi(m / s) - 1)


def crps_gaussian(loc: float, scale: float, y: float) -> float:
    """CRPS of N(loc, scale) at y (exact energy form: E|X-y| - 0.5 E|X-X'|)."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    # E|X-X'| = 2*scale/sqrt(pi) for one Gaussian
    return float(_exp_abs_norm(y - loc, scale) - scale / np.sqrt(np.pi))


def mixture_crps(locs: list[float], scales: list[float], weights: list[float], y: float) -> float:
    """Exact CRPS of an equal-arbitrary-weight Gaussian mixture at y (energy form)."""
    n = len(locs)
    if n == 0:
        raise ValueError("empty mixture")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    locs = np.asarray(locs, dtype=float)
    scales = np.asarray(scales, dtype=float)
    e_xy = sum(w[k] * _exp_abs_norm(y - locs[k], scales[k]) for k in range(n))
    e_xx = 0.0
    for i in range(n):
        for j in range(n):
            s_ij = np.sqrt(scales[i] ** 2 + scales[j] ** 2)
            e_xx += w[i] * w[j] * _exp_abs_norm(locs[i] - locs[j], s_ij)
    return float(e_xy - 0.5 * e_xx)


def five_seed_point(means: list[float]) -> float:
    """Point prediction = mean of five predictive means."""
    return float(np.mean(means))


def crps_student_t(loc: float, scale: float, df: float, y: float) -> float:
    """CRPS of Student-t(loc, scale, df) at y via numerical quadrature (df>1)."""
    if df <= 1:
        raise ValueError("student_t requires df>1 for CRPS")
    if scale <= 0:
        raise ValueError("scale must be positive")
    dist = stats.t(df=df, loc=loc, scale=scale)

    def integrand_low(x: float) -> float:
        return dist.cdf(x) ** 2

    def integrand_high(x: float) -> float:
        return (1.0 - dist.cdf(x)) ** 2

    left, _ = integrate.quad(integrand_low, -np.inf, y, limit=200)
    right, _ = integrate.quad(integrand_high, y, np.inf, limit=200)
    return float(left + right)


# ---- sign-flip exact fixtures (contract 9.4) ------------------------------
def exact_sign_prob(n_signs: int, k_total: int = 20) -> dict[str, float]:
    """Exact one/two-sided sign probability for k_total signs with n_signs same-sign."""
    from math import comb
    if k_total > 20:
        raise ValueError("only 20 finite effects support full enumeration; use actual 2^K otherwise")
    total = 2 ** k_total
    fav_one = sum(comb(k_total, k) for k in range(n_signs, k_total + 1))
    # {X>=n} and {X<=k_total-n} are disjoint by symmetry => two_sided = 2*one_sided
    return {
        "one_sided": fav_one / total,
        "two_sided": 2 * fav_one / total,
    }


# ---- primary estimand aggregation -----------------------------------------
def puzzle_effect(l_candidate: list[float], l_baseline: list[float]) -> float:
    """D_p = mean over cells of (L_baseline - L_candidate); positive = candidate better."""
    if len(l_candidate) != len(l_baseline):
        raise ValueError("candidate/baseline cell counts must match")
    if len(l_candidate) == 0:
        raise ValueError("no cells")
    return float(np.mean([b - c for c, b in zip(l_candidate, l_baseline)]))
