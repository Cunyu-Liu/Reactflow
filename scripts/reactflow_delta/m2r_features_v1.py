#!/usr/bin/env python3
"""m2r_features_v1.py — feature engineering for the M2R rescue_factor task.

Task (LEGAL formulation, non-circular)
--------------------------------------
Given: WT sequence + WT reactivity/error, the two single-mutant profiles
(singleA at i, singleB at j), and the target structure — predict the DOUBLE
mutant's rescue_factor (base-pair support).

Rescue factor is computed from the DOUBLE mutant's profile (which we do NOT
include in features — that would be circular), so this is a genuine
"predict whether a candidate pair is real" task.

Features per pair (i,j):
  * WT reactivity + error at i, j, and in windows around both sites
  * singleA / singleB reactivity (+error) at i, j, and windows
  * derived single-mutant "disruption" signals (how much each single changes
    reactivity at its own site and at the partner site)
  * target-structure features: paired flag, bracket depth, crossing/pseudoknot
    membership at i and j, edit distance, design-region length
  * sequence features: one-hot base at i and j, ref/alt allele, canonical
    pairing (Watson-Crick / wobble) of the candidate pair, mutation types
  * position features: relative positions, edit distance

All features are WT/legal: nothing derived from the double-mutant profile.
"""
from __future__ import annotations

import numpy as np

WINDOW = 7          # half-window around each site: 2*WINDOW+1 positions
BASES = "ACGU"
WC_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE = {("G", "U"), ("U", "G")}


def _nan_to(v, default=0.0):
    try:
        x = float(v)
        return default if not np.isfinite(x) else x
    except (TypeError, ValueError):
        return default


def dot_to_depth(structure):
    """Return (paired, depth) arrays over the full structure string."""
    n = len(structure)
    paired = np.zeros(n, dtype=np.float64)
    depth = np.zeros(n, dtype=np.float64)
    stack = []
    openers = "([{"
    closers = ")]}"
    for i, ch in enumerate(structure):
        if ch in openers:
            stack.append(ch)
            paired[i] = 1.0
            depth[i] = len(stack)
        elif ch in closers:
            paired[i] = 1.0
            depth[i] = len(stack)
            if stack:
                stack.pop()
        else:
            depth[i] = len(stack)
    return paired, depth


def _window(arr, center, W=WINDOW, fill=0.0):
    """Extract a length 2W+1 window around ``center`` from ``arr`` (padding)."""
    out = []
    for k in range(-W, W + 1):
        idx = center + k
        out.append(_nan_to(arr[idx], fill) if 0 <= idx < len(arr) else fill)
    return np.array(out, dtype=np.float64)


def _base_oh(base):
    return np.array([1.0 if base == b else 0.0 for b in BASES], dtype=np.float64)


def _norm(x):
    """Robust normalize a single reactivity value."""
    return float(np.tanh(_nan_to(x)))


def build_pair_features(s: "M2RPair") -> np.ndarray:
    """Feature vector for one M2RPair sample.  All features are WT/single-mutant
    legal (no double-mutant profile)."""
    seq = s.sequence
    i, j = s.editA_seq_pos, s.editB_seq_pos
    n = len(seq)

    parts = []

    # ---- 1. WT reactivity/error at i, j + windows ----
    parts.append(np.array([_norm(s.wt_reactivity[i]), _norm(s.wt_reactivity[j]),
                           _norm(s.wt_error[i]), _norm(s.wt_error[j])]))
    parts.append(_window(s.wt_reactivity, i))
    parts.append(_window(s.wt_reactivity, j))
    parts.append(_window(s.wt_error, i))
    parts.append(_window(s.wt_error, j))

    # ---- 2. single-mutant reactivity/error at i, j + windows ----
    parts.append(np.array([_norm(s.singleA_reactivity[i]), _norm(s.singleA_reactivity[j]),
                           _norm(s.singleA_error[i]), _norm(s.singleA_error[j])]))
    parts.append(_window(s.singleA_reactivity, i))
    parts.append(_window(s.singleA_reactivity, j))
    parts.append(_window(s.singleB_reactivity, i))
    parts.append(_window(s.singleB_reactivity, j))
    parts.append(_window(s.singleA_error, i))
    parts.append(_window(s.singleA_error, j))
    parts.append(_window(s.singleB_error, i))
    parts.append(_window(s.singleB_error, j))

    # ---- 3. derived disruption signals (single-mutant change vs WT) ----
    def chg(mut_react, center):
        # mean absolute change in a window around center, singleA vs WT
        w0 = _window(s.wt_reactivity, center)
        wm = _window(mut_react, center)
        return float(np.abs(wm - w0).mean())
    parts.append(np.array([
        chg(s.singleA_reactivity, i), chg(s.singleA_reactivity, j),
        chg(s.singleB_reactivity, i), chg(s.singleB_reactivity, j),
    ]))
    # signed delta at the sites themselves
    parts.append(np.array([
        _norm(s.singleA_reactivity[i]) - _norm(s.wt_reactivity[i]),
        _norm(s.singleA_reactivity[j]) - _norm(s.wt_reactivity[j]),
        _norm(s.singleB_reactivity[i]) - _norm(s.wt_reactivity[i]),
        _norm(s.singleB_reactivity[j]) - _norm(s.wt_reactivity[j]),
    ]))

    # ---- 4. target-structure features ----
    tgt = s.target_structure
    if len(tgt) < n:
        tgt = tgt + "." * (n - len(tgt))
    pa, dp = dot_to_depth(tgt[:n])
    parts.append(np.array([pa[i], pa[j], dp[i], dp[j],
                           float(abs(i - j) / max(n, 1)),
                           float(i / max(n, 1)), float(j / max(n, 1))]))

    # ---- 5. sequence / pairing features ----
    base_i = seq[i] if i < len(seq) else "N"
    base_j = seq[j] if j < len(seq) else "N"
    parts.append(_base_oh(base_i))
    parts.append(_base_oh(base_j))
    parts.append(np.array([
        1.0 if (base_i, base_j) in WC_PAIRS else 0.0,
        1.0 if (base_i, base_j) in WOBBLE else 0.0,
    ]))

    return np.concatenate(parts).astype(np.float64)


def feature_names() -> list[str]:
    names = []
    names += ["react_i", "react_j", "err_i", "err_j"]
    names += [f"wt_i_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"wt_j_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"wte_i_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"wte_j_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += ["A_i", "A_j", "Ae_i", "Ae_j"]
    names += [f"Ai_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"Aj_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"Bi_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"Bj_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"Aei_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"Aej_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"Bei_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += [f"Bej_{k}" for k in range(-WINDOW, WINDOW + 1)]
    names += ["chgAi", "chgAj", "chgBi", "chgBj"]
    names += ["dAi", "dAj", "dBi", "dBj"]
    names += ["str_pa_i", "str_pa_j", "str_dp_i", "str_dp_j",
              "edit_dist_norm", "rel_i", "rel_j"]
    names += [f"oh_i_{b}" for b in BASES]
    names += [f"oh_j_{b}" for b in BASES]
    names += ["wc_pair", "wobble"]
    return names


def build_all(samples) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build feature matrix + target + design keys for all samples."""
    X = np.stack([build_pair_features(s) for s in samples])
    y = np.array([s.rescue_factor for s in samples], dtype=np.float64)
    keys = [s.design_id for s in samples]
    return X, y, keys, feature_names()
