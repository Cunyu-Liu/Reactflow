"""Tests for the odd nonlinear switch operator (T-O0.9, §4.5, §5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from reactflow.delta.switch import (
    check_switch_oddness,
    compute_switch,
    no_switch_response,
)
from reactflow.delta.susceptibility import build_symmetric_background


def _rng():
    return np.random.default_rng(2)


def _features(n=16):
    rng = _rng()
    zb = build_symmetric_background(rng.standard_normal(n), rng.standard_normal(n))
    return zb.stack()


# ---------------------------------------------------------------------------
# T-O0.9: odd switch h(-b) = -h(b)
# ---------------------------------------------------------------------------


class TestOddSwitch:
    def test_h_odd(self):
        rng = _rng()
        n = 16
        zbf = _features(n)
        h_lin = rng.standard_normal(n)
        b = rng.standard_normal(n)
        res_pos = compute_switch(h_lin, zbf, b)
        res_neg = compute_switch(-h_lin, zbf, -b)
        err = np.max(np.abs(res_pos.h + res_neg.h))
        assert err < 1e-6

    def test_h_nl_odd(self):
        rng = _rng()
        n = 12
        zbf = _features(n)
        h_lin = rng.standard_normal(n)
        b = rng.standard_normal(n)
        res_pos = compute_switch(h_lin, zbf, b)
        res_neg = compute_switch(-h_lin, zbf, -b)
        err = np.max(np.abs(res_pos.h_nl + res_neg.h_nl))
        assert err < 1e-6

    def test_gate_swap_invariant(self):
        rng = _rng()
        n = 14
        zbf = _features(n)
        h_lin = rng.standard_normal(n)
        b = rng.standard_normal(n)
        res_pos = compute_switch(h_lin, zbf, b)
        res_neg = compute_switch(-h_lin, zbf, -b)
        # Gate depends on z_bar and |b|, both invariant under b -> -b.
        err = np.max(np.abs(res_pos.gate - res_neg.gate))
        assert err < 1e-12

    def test_tanh_odd_no_bias(self):
        """Switch of zero h_lin must produce zero h_nl (no bias, tanh(0)=0)."""
        rng = _rng()
        n = 16
        zbf = _features(n)
        b = rng.standard_normal(n)
        res_zero = compute_switch(np.zeros(n), zbf, b)
        assert np.max(np.abs(res_zero.h_nl)) < 1e-7
        # Total response equals h_lin (zero) plus zero h_nl.
        assert np.max(np.abs(res_zero.h)) < 1e-7

    def test_audit_oddness(self):
        rng = _rng()
        n = 16
        zbf = _features(n)
        h_lin = rng.standard_normal(n)
        b = rng.standard_normal(n)
        audit = check_switch_oddness(h_lin, zbf, b)
        assert audit["oddness"]["pass"]
        assert audit["gate_swap_invariance"]["pass"]
        assert audit["no_bias"]["pass"]

    def test_no_switch_ablation(self):
        """no_switch ablation returns h_lin unchanged (§4.5)."""
        rng = _rng()
        h_lin = rng.standard_normal(10)
        h = no_switch_response(h_lin)
        assert np.allclose(h, h_lin)
        # Must be independent of z_bar / b.
        assert h.shape == h_lin.shape


class TestSwitchProperties:
    def test_gate_in_unit_interval(self):
        rng = _rng()
        n = 12
        zbf = _features(n)
        h_lin = rng.standard_normal(n)
        b = rng.standard_normal(n)
        res = compute_switch(h_lin, zbf, b)
        assert np.all(res.gate >= 0.0)
        assert np.all(res.gate <= 1.0)

    def test_S_symmetric_no_bias(self):
        from reactflow.delta.switch import _default_S
        zbf = _features(14)
        S = _default_S(zbf)
        assert np.allclose(S, S.T)
        assert np.all(np.diag(S) == 0.0)

    def test_audit_dict(self):
        rng = _rng()
        n = 12
        zbf = _features(n)
        h_lin = rng.standard_normal(n)
        b = rng.standard_normal(n)
        res = compute_switch(h_lin, zbf, b)
        audit = res.to_audit_dict()
        assert audit["no_bias"] is True
        assert audit["h_norm"] >= 0.0
