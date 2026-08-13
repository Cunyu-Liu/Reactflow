#!/usr/bin/env python3
"""p2_learnability: frozen Direct learnability procedure (contract 9.2, 13).

Implements the single adaptive P2 procedure (NOT "pick whichever model is
significant"):
  - T* : selected among {ZeroResponse, train-only median} by outer-train grouped
        inner OOF primary CRPS + deterministic tie-break.
  - Direct*: selected among {regularized direct, frozen nonlinear, flat MLP,
        RNet static-delta, RFD-Direct} by the same inner OOF CRPS + tie-break.
  - D_p^P2 = L_p^T* - L_p^Direct*  (positive => candidate better).
  - theta = mean over 20 puzzles; two-sided 95% puzzle-level t CI.
  - leave-one-puzzle influence and exhaustive sign-flip T (studentized).

This is the statistical core (GPU-free). The per-fold training loop that produces
the inner OOF CRPS for each baseline is launched separately on GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TRIVIAL_METHODS = {"zero", "train_median"}
DIRECT_METHODS = {"reg_direct", "nonlinear", "flat_mlp", "rnet_static_delta", "rfd_direct"}


def select_inner_t_star(inner_crps: dict[str, float], tie_break_order: list[str]) -> str:
    """T* = argmin over trivial methods of inner OOF primary CRPS; deterministic tie-break."""
    return _select(inner_crps, TRIVIAL_METHODS, tie_break_order)


def select_inner_direct_star(inner_crps: dict[str, float], tie_break_order: list[str]) -> str:
    """Direct* = argmin over direct methods of inner OOF primary CRPS."""
    return _select(inner_crps, DIRECT_METHODS, tie_break_order)


def _select(inner_crps: dict[str, float], allowed: set[str], tie_break_order: list[str]) -> str:
    present = {m: inner_crps[m] for m in allowed if m in inner_crps}
    if not present:
        raise ValueError(f"no valid methods present among {sorted(allowed)}")
    best_val = min(present.values())
    best = [m for m, v in present.items() if v == best_val]
    # deterministic tie-break: first in the frozen precedence order
    for m in tie_break_order:
        if m in best:
            return m
    return sorted(best)[0]


def d_p_p2(l_t_star: float, l_direct_star: float) -> float:
    """D_p^P2 = L_p^T* - L_p^Direct*; positive means direct candidate better."""
    return l_t_star - l_direct_star


def puzzle_level_ci20(effects: list[float], alpha: float = 0.05) -> dict[str, Any]:
    """theta mean/SD/two-sided 95% t CI over the 20 puzzle effects."""
    if len(effects) != 20:
        return {"planned_n_not_met": True, "n_effects": len(effects),
                "message": "PLANNED_INFERENCE_N_NOT_MET: need 20 finite D_p; report actual-K sensitivity only"}
    arr = np.asarray(effects, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    se = sd / np.sqrt(len(arr))
    from scipy import stats
    tcrit = stats.t.ppf(1 - alpha / 2, df=len(arr) - 1)
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return {
        "planned_n_not_met": False,
        "n_effects": len(arr),
        "mean": mean,
        "sd": sd,
        "ci_low": lo,
        "ci_high": hi,
        "ci_low_gt_0": lo > 0.0,
    }


def studentized_sign_flip(effects: list[float], n_draws: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Exhaustive/sampled two-sided sign-flip on T = mean(D)/ (sd(D)/sqrt(K))."""
    arr = np.asarray(effects, dtype=float)
    K = len(arr)
    sd = arr.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return {"status": "UNIDENTIFIABLE_SIGN_FLIP",
                "reason": "zero/degenerate standard error; no epsilon added"}
    t_obs = arr.mean() / (sd / np.sqrt(K))
    if K <= 20:
        # exhaustive 2^K sign enumeration
        import itertools
        count = 0
        total = 0
        t_obs_abs = abs(t_obs)
        for bits in itertools.product([1, -1], repeat=K):
            flipped = arr * np.asarray(bits)
            s = flipped.std(ddof=1)
            if s == 0:
                continue
            t = flipped.mean() / (s / np.sqrt(K))
            if abs(t) >= t_obs_abs:
                count += 1
            total += 1
        p = count / total if total else float("nan")
    else:
        rng = np.random.RandomState(seed)
        cnt = 0
        for _ in range(n_draws):
            bits = rng.choice([1, -1], size=K)
            flipped = arr * bits
            s = flipped.std(ddof=1)
            if s == 0:
                continue
            t = flipped.mean() / (s / np.sqrt(K))
            if abs(t) >= t_obs_abs:
                cnt += 1
        p = (cnt + 1) / (n_draws + 1)  # plus-one correction
    return {"status": "OK", "T_obs": t_obs, "K": K, "p_value": p, "mode": "exhaustive" if K <= 20 else "monte_carlo_plus_one"}


def leave_one_puzzle_influence(effects: list[float], puzzles: list[str]) -> dict[str, Any]:
    """Recompute theta excluding each puzzle; report sensitivity."""
    rows = []
    for i, p in enumerate(puzzles):
        rest = [e for j, e in enumerate(effects) if j != i]
        rows.append({"excluded": p, "theta": float(np.mean(rest))})
    return {"schema_version": "reactflow_delta.p2.lop_influence.v1", "rows": rows,
            "max_abs_shift": max(abs(r["theta"] - float(np.mean(effects))) for r in rows)}
