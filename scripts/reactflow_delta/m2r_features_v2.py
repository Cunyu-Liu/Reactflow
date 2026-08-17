#!/usr/bin/env python3
"""m2r_features_v2.py — v2 legal feature group for the M2R rescue_factor task.

MOTIVATION (method-level, capturing the double-mutant interaction):
The v1 feature set (230 dims) treats the two single mutants mostly
INDEPENDENTLY: windows around i from singleA, windows around j from singleB,
plus local disruption magnitudes.  The rescue factor, however, is fundamentally
an INTERACTION quantity:
    rescue = 1 - rD / sqrt(rA^2 + rB^2)
with rX = RMSD(mutant X, WT) over the design region.  High rescue means the
double mutant RESTORES the WT structure (small rD).  The two legal cues that
predict rD without seeing the double-mutant profile are:

  (a) CROSS-MUTANT DISRUPTION OVERLAP: if singleA and singleB disrupt the SAME
      structural region (dA(k) = |A_k - wt_k| and dB(k) = |B_k - wt_k| both
      large at the same positions k), the two edits are structurally coupled
      and their combination is far more likely to restore the WT structure
      than if they disrupt disjoint regions.  This overlap is 100% legal
      (single mutants + WT only).

  (b) DISRUPTION MAGNITUDES rA, rB (the legal denominator of the rescue
      formula): directly the sqrt-sum of the two single-mutant RMSDs.

  (c) STRUCTURAL STEM CONTEXT in the target structure: the rescue of a pair
      (i,j) depends on whether i and j sit in a long, well-nested stem vs a
      shallow/peripheral element, and on the pseudoknot nesting depth.

Everything here uses ONLY: WT + singleA + singleB reactivity/error, the target
structure (design intent), sequence, and the M2_structure (independent data
source).  Nothing from the double-mutant reactivity profile.
"""
from __future__ import annotations

import numpy as np

WINDOW = 7  # must match m2r_features_v1


def _nan_to(v, default=0.0):
    try:
        x = float(v)
        return default if not np.isfinite(x) else x
    except (TypeError, ValueError):
        return default


def _prof(p):
    return np.array([x if x is not None and np.isfinite(x) else np.nan
                     for x in p], dtype=np.float64)


def dot_to_depth(structure):
    """Return (paired, depth, partner) arrays over the structure string.

    partner[k] = index of the base k is paired with, else -1.  Handles
    nested and pseudoknotted pairs by first-match with a stack per bracket.
    """
    n = len(structure)
    paired = np.zeros(n, dtype=np.float64)
    depth = np.zeros(n, dtype=np.float64)
    partner = np.full(n, -1, dtype=np.int64)
    # group by bracket type
    from collections import defaultdict
    stacks = defaultdict(list)  # bracket char -> stack of indices
    depth_stack = []
    for idx, ch in enumerate(structure):
        if ch in "([{":
            stacks[ch].append(idx)
            depth_stack.append(len(depth_stack) + 1)
            paired[idx] = 1.0
            depth[idx] = len(depth_stack)
        elif ch in ")]}":
            open_ch = {"}": "{", "]": "[", ")": "("}[ch]
            st = stacks[open_ch]
            depth[idx] = len(depth_stack)
            if st:
                mate = st.pop()
                partner[idx] = mate
                partner[mate] = idx
                paired[idx] = 1.0
                if depth_stack:
                    depth_stack.pop()
        else:
            depth[idx] = len(depth_stack)
    return paired, depth, partner


def _stem_lengths(paired, partner):
    """For each position, the length of the contiguous paired run (stem) it
    belongs to; 0 for unpaired positions."""
    n = len(paired)
    stem = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        if paired[i] == 1.0:
            j = i
            while j < n and paired[j] == 1.0 and partner[j] != -1:
                j += 1
            # contiguous run [i, j) is a stem segment; assign its length
            stem[i:j] = j - i
            i = j
        else:
            i += 1
    return stem


def _region_mask(n, sub_start, sub_end):
    m = np.zeros(n, dtype=bool)
    lo = max(sub_start - 1, 0) if sub_start is not None else 0
    hi = sub_end if sub_end is not None else n
    m[lo:hi] = True
    return m


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return 0.0
    aa = a[m]; bb = b[m]
    va = aa - aa.mean(); vb = bb - bb.mean()
    den = np.sqrt(np.sum(va ** 2) * np.sum(vb ** 2))
    if den <= 0:
        return 0.0
    return float(np.sum(va * vb) / den)


def _rmsd(a, b, mask):
    m = np.isfinite(a) & np.isfinite(b) & mask
    if m.sum() < 3:
        return 0.0
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def build_v2_features(s: "M2RPair") -> np.ndarray:
    """v2 legal feature vector for one M2RPair sample.  Concatenated with v1."""
    n = len(s.sequence)
    i, j = s.editA_seq_pos, s.editB_seq_pos
    wt = _prof(s.wt_reactivity)
    ra = _prof(s.singleA_reactivity)
    rb = _prof(s.singleB_reactivity)
    mask = _region_mask(n, s.sub_start, s.sub_end)
    parts = []

    # ---- A. cross-mutant disruption overlap (design region) ----
    dA = np.abs(ra - wt)
    dB = np.abs(rb - wt)
    mr = np.isfinite(dA) & np.isfinite(dB) & mask
    parts.append(np.array([
        _pearson(dA, dB),
        float(np.nanmean(np.minimum(dA, dB)[mr])) if mr.sum() >= 3 else 0.0,
        float(np.nanmean(np.abs(dA - dB)[mr])) if mr.sum() >= 3 else 0.0,
    ]))
    # overlap in a local window around i and around j
    wi = slice(max(i - WINDOW, 0), min(i + WINDOW + 1, n))
    wj = slice(max(j - WINDOW, 0), min(j + WINDOW + 1, n))
    dA_wi = np.minimum(dA[wi], dB[wi])
    dA_wj = np.minimum(dA[wj], dB[wj])
    parts.append(np.array([
        _pearson(dA[wi], dB[wi]),
        _pearson(dA[wj], dB[wj]),
        float(np.nanmean(dA_wi)) if np.isfinite(dA_wi).sum() >= 3 else 0.0,
        float(np.nanmean(dA_wj)) if np.isfinite(dA_wj).sum() >= 3 else 0.0,
    ]))

    # ---- B. disruption magnitudes (legal denominator of the rescue formula) ----
    rA = _rmsd(wt, ra, mask)
    rB = _rmsd(wt, rb, mask)
    r_sum = np.sqrt(rA ** 2 + rB ** 2)
    parts.append(np.array([
        rA, rB, r_sum,
        float(rA * rB),
        _nan_to(1.0 - r_sum / (rA + rB + 1e-9)),  # overlap fraction proxy
    ]))

    # ---- C. structural stem context (target structure, design intent) ----
    tgt = s.target_structure
    if len(tgt) < n:
        tgt = tgt + "." * (n - len(tgt))
    pa, dp, partner = dot_to_depth(tgt[:n])
    stem = _stem_lengths(pa, partner)
    # position of i and j in their stem (index from the 5' stem end, normalized)
    def _pos_in_stem(idx):
        if pa[idx] != 1.0 or partner[idx] == -1:
            return 0.0, 0.0
        # walk 5' to the stem start
        k = idx
        while k > 0 and pa[k - 1] == 1.0 and partner[k - 1] != -1:
            k -= 1
        start = k
        k = idx
        while k < n - 1 and pa[k + 1] == 1.0 and partner[k + 1] != -1:
            k += 1
        L = max(k - start + 1, 1)
        return float((idx - start) / L), float(L)
    pos_i, len_i = _pos_in_stem(i)
    pos_j, len_j = _pos_in_stem(j)
    parts.append(np.array([
        stem[i], stem[j],
        pos_i, pos_j, len_i, len_j,
        dp[i], dp[j],
    ]))

    # ---- D. M2_structure cross context (independent data source) ----
    # whether BOTH i and j are paired in the experimentally observed M2
    # structure, and the product of their M2 depths (coupling in M2)
    m2str = getattr(s, "m2_structure", "") or ""
    m2_sub = getattr(s, "sub_start", None)
    if m2str and m2_sub is not None:
        mpa, mdp, _ = dot_to_depth(m2str)
        idx_i = i - (m2_sub - 1)
        idx_j = j - (m2_sub - 1)
        def _val(arr, idx):
            return float(arr[idx]) if 0 <= idx < len(arr) else 0.0
        pa_i = _val(mpa, idx_i); pa_j = _val(mpa, idx_j)
        parts.append(np.array([
            pa_i * pa_j, _val(mdp, idx_i) * _val(mdp, idx_j),
        ]))
    else:
        parts.append(np.zeros(2))

    return np.concatenate(parts).astype(np.float64)


def v2_feature_names() -> list[str]:
    names = []
    names += ["xcorr_design", "xmin_design", "xabsdiff_design"]
    names += ["xcorr_wi", "xcorr_wj", "xmin_wi", "xmin_wj"]
    names += ["rA", "rB", "rAB_sqrt_sum", "rA_rB", "overlap_frac"]
    names += ["stem_i", "stem_j", "pos_i", "pos_j", "len_i", "len_j",
              "depth_i", "depth_j"]
    names += ["m2_both_paired", "m2_depth_prod"]
    return names


def build_all_v2(samples) -> tuple[np.ndarray, list[str]]:
    X = np.stack([build_v2_features(s) for s in samples])
    return X, v2_feature_names()
