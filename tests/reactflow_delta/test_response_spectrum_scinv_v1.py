#!/usr/bin/env python3
"""Tests for response_spectrum_scinv_v1 — scale-invariant full-spectrum response target."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import response_spectrum_scinv_v1 as rss


def _mask(n, on):
    return [bool(i in on) for i in range(n)]


def test_even_window_raises():
    with pytest.raises(ValueError):
        rss.pair_response_spectrum([1.0] * 10, [1.0] * 10, _mask(10, range(10)),
                                   5, window=20)


def test_no_eligible_returns_none_scale():
    y, w, scale = rss.pair_response_spectrum([1.0] * 5, [1.0] * 5, _mask(5, []),
                                             2, window=5)
    assert scale is None
    assert y == [0.0] * 5
    assert w == [0.0] * 5


def test_scale_invariance_doubling_wt_and_mut():
    """Scaling WT and mut by a constant leaves every response element invariant."""
    n = 9
    edit = 4
    wt = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    mut = [1.5, 1.8, 3.5, 4.2, 5.5, 3.2, 2.8, 2.5, 1.2]
    mask = _mask(n, range(n))
    y1, w1, s1 = rss.pair_response_spectrum(wt, mut, mask, edit, window=9)
    y2, w2, s2 = rss.pair_response_spectrum([2 * v for v in wt], [2 * v for v in mut],
                                            mask, edit, window=9)
    assert w1 == w2 == [1.0] * 9
    # scale = mean WT level so it doubles with the WT; the RESPONSE is invariant
    assert s2 == pytest.approx(2.0 * s1, rel=1e-9)
    for a, b in zip(y1, y2):
        assert a == pytest.approx(b, rel=1e-9)


def test_signed_preserved():
    """A negative response (reactivity decreases) must be negative after scaling."""
    n = 7
    edit = 3
    wt = [1.0] * n
    mut = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]  # drop at position 2
    mask = _mask(n, range(n))
    y, w, scale = rss.pair_response_spectrum(wt, mut, mask, edit, window=7)
    assert y[2] < 0.0
    assert w[2] == 1.0
    assert scale == pytest.approx(1.0, rel=1e-9)


def test_inelegible_and_out_of_bounds_weight_zero():
    """Ineligible and out-of-window positions must have weight 0, y 0."""
    n = 5
    edit = 2
    wt = [1.0, 2.0, 3.0, 4.0, 5.0]
    mut = [1.5, 2.5, 3.5, 4.5, 5.5]
    mask = _mask(n, [1, 3])  # only positions 1 and 3 eligible
    y, w, scale = rss.pair_response_spectrum(wt, mut, mask, edit, window=5)
    # window covers idx 0..4; eligible at 1 and 3
    assert w[0] == 0.0 and w[1] == 1.0 and w[2] == 0.0 and w[3] == 1.0 and w[4] == 0.0
    assert y[0] == 0.0 and y[4] == 0.0
    # scale = mean WT over eligible (1 and 3) = (2+4)/2 = 3
    assert scale == pytest.approx(3.0, rel=1e-9)
    # position 1: (2.5-2)/3 = 0.1666...
    assert y[1] == pytest.approx(0.5 / 3.0, rel=1e-9)


def test_non_finite_skipped():
    n = 5
    edit = 2
    wt = [1.0, 2.0, float("nan"), 4.0, 5.0]
    mut = [1.5, 2.5, 3.5, 4.5, 5.5]
    mask = _mask(n, range(n))
    y, w, scale = rss.pair_response_spectrum(wt, mut, mask, edit, window=5)
    # position 2 (window k=2) has nan WT -> weight 0
    assert w[2] == 0.0
    assert scale is not None


def test_mad_scale_mode():
    n = 5
    edit = 2
    wt = [1.0, 2.0, 3.0, 4.0, 5.0]
    mut = [1.5, 2.5, 3.5, 4.5, 5.5]
    mask = _mask(n, range(n))
    _, _, s_mean = rss.pair_response_spectrum(wt, mut, mask, edit, window=5,
                                               scale_mode="mean_level")
    _, _, s_mad = rss.pair_response_spectrum(wt, mut, mask, edit, window=5,
                                             scale_mode="mad")
    assert s_mean == pytest.approx(3.0, rel=1e-9)
    assert s_mad == pytest.approx(sum(abs(v - 3.0) for v in wt) / len(wt), rel=1e-9)
    assert s_mad < s_mean


def test_zero_scale_floor():
    wt = [0.0, 0.0, 0.0]
    mut = [0.0, 0.0, 0.0]
    mask = _mask(3, range(3))
    y, w, scale = rss.pair_response_spectrum(wt, mut, mask, 1, window=3)
    assert scale == rss.SCALE_EPS
    assert w == [1.0, 1.0, 1.0]
    assert all(v == 0.0 for v in y)


def test_response_spectrum_magnitude_matches_scalar():
    wt = [1.0, 2.0, 3.0, 4.0]
    mut = [2.0, 2.5, 4.0, 5.0]
    mask = _mask(4, range(4))
    mag, n = rss.response_spectrum_magnitude(wt, mut, mask)
    assert n == 4
    assert mag == pytest.approx((1.0 + 0.5 + 1.0 + 1.0) / 4.0, rel=1e-9)