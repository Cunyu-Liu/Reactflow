"""Tests for the monotone probe observation operator (T-O0.10, §4.6, §5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from reactflow.delta.observation import (
    OBSERVATION_SCHEMA_VERSION,
    PROBE_DOMAIN_MAX,
    PROBE_DOMAIN_MIN,
    SUPPORTED_PROBES,
    ObservationHead,
    check_observation_invariants,
    default_head,
    observe_delta_r,
)
from reactflow.delta.susceptibility import build_symmetric_background


def _rng():
    return np.random.default_rng(3)


def _features(n=16):
    rng = _rng()
    zb = build_symmetric_background(rng.standard_normal(n), rng.standard_normal(n))
    return zb.stack()


# ---------------------------------------------------------------------------
# T-O0.10: monotone probe observation
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_default_heads_monotone_on_domain(self):
        a = np.linspace(PROBE_DOMAIN_MIN, PROBE_DOMAIN_MAX, 401)
        for probe in SUPPORTED_PROBES:
            head = default_head(probe)
            deriv = head.derivative(a)
            assert np.min(deriv) >= -1e-9, f"probe {probe} not monotone"

    def test_weights_non_negative(self):
        for probe in SUPPORTED_PROBES:
            head = default_head(probe)
            assert np.all(head.weights >= 0.0)

    def test_monotone_head_increasing(self):
        a = np.linspace(PROBE_DOMAIN_MIN, PROBE_DOMAIN_MAX, 201)
        for probe in SUPPORTED_PROBES:
            head = default_head(probe)
            vals = head.evaluate(a)
            diffs = np.diff(vals)
            assert np.min(diffs) >= -1e-9, f"probe {probe} head not non-decreasing"

    def test_custom_head_must_be_monotone(self):
        # Non-negative weights => monotone.
        w = np.array([0.5, 0.3, 0.1, 0.2])
        head = ObservationHead(probe="DMS", weights=w)
        a = np.linspace(PROBE_DOMAIN_MIN, PROBE_DOMAIN_MAX, 301)
        assert np.min(head.derivative(a)) >= -1e-9

    def test_audit_all_probes_pass(self):
        rng = _rng()
        n = 16
        zbf = _features(n)
        h = rng.standard_normal(n)
        audit = check_observation_invariants(h, zbf, probes=("DMS", "SHAPE", "2A3"))
        assert audit["monotonicity"]["all_pass"]
        for p in ("DMS", "SHAPE", "2A3"):
            assert audit["monotonicity"]["per_probe"][p]["pass"]


class TestProbeSpecificHeads:
    def test_distinct_heads(self):
        dms = default_head("DMS")
        shape = default_head("SHAPE")
        twoa3 = default_head("2A3")
        assert not np.allclose(dms.weights, shape.weights)
        assert not np.allclose(shape.weights, twoa3.weights)

    def test_rejects_unknown_probe(self):
        with pytest.raises(ValueError):
            default_head("CMCT")

    def test_observation_uses_correct_head(self):
        rng = _rng()
        n = 12
        zbf = _features(n)
        h = rng.standard_normal(n)
        res_dms = observe_delta_r(h, zbf, probe="DMS")
        res_shape = observe_delta_r(h, zbf, probe="SHAPE")
        assert res_dms.head.probe == "DMS"
        assert res_shape.head.probe == "SHAPE"
        # Different heads => different predictions (in general).
        assert not np.allclose(res_dms.delta_r_hat, res_shape.delta_r_hat)


class TestSwapAntisymmetry:
    def test_swap_flips_sign(self):
        rng = _rng()
        n = 16
        zbf = _features(n)
        h = rng.standard_normal(n)
        res_pos = observe_delta_r(h, zbf, probe="DMS")
        res_neg = observe_delta_r(-h, zbf, probe="DMS")
        err = np.max(np.abs(res_pos.delta_r_hat + res_neg.delta_r_hat))
        assert err < 1e-6

    def test_zero_h_zero_delta_r(self):
        rng = _rng()
        n = 12
        zbf = _features(n)
        res = observe_delta_r(np.zeros(n), zbf, probe="SHAPE")
        # f_p(a_bar) - f_p(a_bar) = 0
        assert np.max(np.abs(res.delta_r_hat)) < 1e-7

    def test_head_does_not_read_study_id(self):
        """Observation head must not depend on study ID (§4.6)."""
        rng = _rng()
        n = 10
        zbf = _features(n)
        h = rng.standard_normal(n)
        # No study parameter in the API at all.
        res = observe_delta_r(h, zbf, probe="DMS")
        assert "study" not in res.to_audit_dict()
        assert "study_id" not in res.to_audit_dict()

    def test_endpoints_split_correctly(self):
        rng = _rng()
        n = 8
        zbf = _features(n)
        h = rng.standard_normal(n)
        res = observe_delta_r(h, zbf, probe="DMS")
        # a_m = a_bar + h/2, a_w = a_bar - h/2
        assert np.allclose(res.a_m - res.a_w, h)
        assert np.allclose(res.a_m + res.a_w, 2.0 * res.a_bar)
