#!/usr/bin/env python3
"""response_spectrum_scinv_v1 — STRICT-legal scale-invariant per-position response spectrum.

Root-cause context
------------------
The endpoint_v6 conditional-magnitude target collapses each true-changer pair to a
single scalar
    y = mean over ELIGIBLE |mut[i] - wt[i]|            (absolute change)
(and its scale-invariant variant y_rel = that / WT level).  Both destroy the
per-position structure of the mutation response: a single edit can raise reactivity
at some positions and lower it at others (opening/closing base pairs), and those
signed/spatial effects are averaged away.  The Phase 2 / scinv experiments showed
the scalar magnitude carries no transferable signal (permutation p = 1.0), largely
because the mean collapses the informative per-position signal into a number that is
dominated by the pair's own WT level / assay scale.

This module replaces the scalar with a FULL response spectrum aligned to the
model's local window around the EDIT SITE:

    for each window position k in [0, WINDOW)  (sequence index = ei - HALF + k):
        y[k] = (mut[idx] - wt[idx]) / scale        # SIGNED, scale-invariant response
        w[k] = 1.0  if idx eligible AND finite WT+mut, else 0.0
    scale   = mean WT reactivity over ELIGIBLE positions   (STRICT-legal WT anchor)

Only the WT reactivity anchor (an ALLOWED input) forms the scale.  The mutant
reactivity / target mask never enter the scale.  The output is a vector aligned to
the same local window the model already reads, so a position-aware model can predict
the full mutation-response profile directly (multiple supervised targets per pair)
instead of a single collapsed number.  This preserves SIGN and SPATIAL structure,
which is the information the scalar magnitude discarded.

The functions take plain lists/ints so the module is self-contained and unit-testable
without the run_p2_v3 dependency chain.
"""
from __future__ import annotations

import math

SCALE_EPS = 1e-6


def _scale(wt_reactivity, eligibility_mask, mode="mean_level"):
    """WT-level scale over ELIGIBLE finite positions.

    mean_level : arithmetic mean (matches magnitude_scale_invariant_v1).
    Returns a float >= SCALE_EPS, or None if there are no eligible finite WT values.
    """
    vals = []
    L = min(len(wt_reactivity), len(eligibility_mask))
    for i in range(L):
        if not eligibility_mask[i]:
            continue
        a = float(wt_reactivity[i])
        if math.isfinite(a):
            vals.append(a)
    if not vals:
        return None
    if mode == "mad":
        m = sum(vals) / len(vals)
        s = sum(abs(v - m) for v in vals) / len(vals)
    else:
        s = sum(vals) / len(vals)
    return s if s > SCALE_EPS else SCALE_EPS


def pair_response_spectrum(
    wt_reactivity, mutant_reactivity, eligibility_mask,
    edit_index, window=21, scale_mode="mean_level",
):
    """Return (y_vec, w_vec, scale) aligned to the local window around the edit site.

    Args:
        wt_reactivity      : WT reactivity array (aligned to sequence positions).
        mutant_reactivity  : mutant reactivity array (same alignment).
        eligibility_mask   : aligned eligibility mask (bool/list).
        edit_index         : sequence index of the edited site (int).
        window             : window width (odd).  Default 21.
        scale_mode         : 'mean_level' | 'mad'.

    Returns:
        y_vec : list of length `window`, signed scale-invariant response per window
                position (0.0 where not predictable/eligible).
        w_vec : list of length `window`, 1.0 where eligible AND finite WT+mut, else 0.0.
        scale : float used (>=SCALE_EPS), or None if no eligible finite WT.

    Window position k maps to sequence index idx = edit_index - (window//2) + k.
    Deterministic and fold-invariant.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    half = window // 2
    scale = _scale(wt_reactivity, eligibility_mask, mode=scale_mode)
    if scale is None:
        return [0.0] * window, [0.0] * window, None

    L = min(len(wt_reactivity), len(mutant_reactivity), len(eligibility_mask))
    y_vec = []
    w_vec = []
    for k in range(window):
        idx = edit_index - half + k
        elig = bool(eligibility_mask[idx]) if 0 <= idx < len(eligibility_mask) else False
        if not elig:
            y_vec.append(0.0)
            w_vec.append(0.0)
            continue
        if not (0 <= idx < L):
            y_vec.append(0.0)
            w_vec.append(0.0)
            continue
        a = float(wt_reactivity[idx])
        b = float(mutant_reactivity[idx])
        if not (math.isfinite(a) and math.isfinite(b)):
            y_vec.append(0.0)
            w_vec.append(0.0)
            continue
        y_vec.append((b - a) / scale)
        w_vec.append(1.0)
    return y_vec, w_vec, scale


def response_spectrum_magnitude(wt_reactivity, mutant_reactivity, eligibility_mask):
    """Scalar summary of the full spectrum used only for diagnostics (not the target).

    Returns (mean |y_i| over eligible, n_eligible) matching the old scalar magnitude
    so a comparison script can report both the old and new targets side by side.
    """
    L = min(len(wt_reactivity), len(mutant_reactivity), len(eligibility_mask))
    abs_deltas = []
    for i in range(L):
        if not eligibility_mask[i]:
            continue
        a = float(wt_reactivity[i])
        b = float(mutant_reactivity[i])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        abs_deltas.append(abs(b - a))
    if not abs_deltas:
        return (None, 0)
    return (float(sum(abs_deltas) / len(abs_deltas)), len(abs_deltas))