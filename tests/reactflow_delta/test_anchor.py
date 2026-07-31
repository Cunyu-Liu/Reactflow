"""Tests for the P2 WT-anchored posterior update + access guard (T-O0.11, §4.8, §5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from reactflow.delta.anchor import (
    P2_ALLOWED_INPUT_FIELDS,
    P2_FORBIDDEN_INPUT_FIELDS,
    P2AnchorGuard,
    check_p2_anchor_invariants,
    static_audit,
)


# ---------------------------------------------------------------------------
# T-O0.11: P2 anchor access guard (mutant profile access == 0)
# ---------------------------------------------------------------------------


class TestStaticAudit:
    def test_static_audit_passes(self):
        audit = static_audit()
        assert audit["pass"]
        assert audit["vocab_overlap"] == []
        assert audit["forbidden_in_signature"] == []

    def test_forbidden_disjoint_from_allowed(self):
        assert P2_ALLOWED_INPUT_FIELDS.isdisjoint(P2_FORBIDDEN_INPUT_FIELDS)

    def test_mutant_fields_in_forbidden(self):
        for f in ("mut_reactivity", "mutant_reactivity", "mutant_profile", "delta_reactivity"):
            assert f in P2_FORBIDDEN_INPUT_FIELDS


class TestRuntimeGuard:
    def test_wt_update_no_mutant_access(self):
        guard = P2AnchorGuard()
        n = 16
        rng = np.random.default_rng(0)
        q = rng.standard_normal(n)
        delta_a = guard.wt_anchor_update(q, sigma=0.3, probe="DMS")
        assert delta_a.shape == (n,)
        audit = guard.audit()
        assert audit["mutant_access_count"] == 0
        assert audit["pass"]

    def test_access_log_records_wt_fields(self):
        guard = P2AnchorGuard()
        q = np.zeros(8)
        guard.wt_anchor_update(q, sigma=0.5, probe="SHAPE", prior=np.zeros(8))
        audit = guard.audit()
        assert "wt_observation_residual" in audit["wt_access_fields"]
        assert "measurement_variance" in audit["wt_access_fields"]
        assert "wt_accessibility_prior" in audit["wt_access_fields"]
        assert audit["mutant_access_count"] == 0

    def test_forbidden_field_raises(self):
        guard = P2AnchorGuard()
        with pytest.raises(RuntimeError):
            guard._record("mut_reactivity")
        with pytest.raises(RuntimeError):
            guard._record("mutant_profile")

    def test_gain_validation(self):
        guard = P2AnchorGuard()
        with pytest.raises(ValueError):
            guard.wt_anchor_update(np.zeros(4), sigma=0.1, gain=1.5)
        with pytest.raises(ValueError):
            guard.wt_anchor_update(np.zeros(4), sigma=0.1, gain=-0.1)

    def test_update_finite(self):
        guard = P2AnchorGuard()
        rng = np.random.default_rng(1)
        q = rng.standard_normal(32)
        delta_a = guard.wt_anchor_update(q, sigma=0.2, probe="DMS")
        assert np.all(np.isfinite(delta_a))

    def test_no_mutant_argument_in_signature(self):
        """wt_anchor_update must not accept any mutant-side argument."""
        import inspect
        sig = inspect.signature(P2AnchorGuard.wt_anchor_update)
        params = set(sig.parameters)
        forbidden = params & P2_FORBIDDEN_INPUT_FIELDS
        assert forbidden == set()


class TestFullAudit:
    def test_check_p2_anchor_invariants_passes(self):
        audit = check_p2_anchor_invariants()
        assert audit["pass"]
        assert audit["static"]["pass"]
        assert audit["runtime"]["pass"]
        assert audit["runtime"]["mutant_access_count"] == 0

    def test_p2_does_not_build_second_model(self):
        """P2 is a WT-side update; it produces delta_a_w only (§4.8)."""
        guard = P2AnchorGuard()
        out = guard.wt_anchor_update(np.ones(10), sigma=0.1, probe="DMS")
        # Output is a single WT accessibility update vector.
        assert out.ndim == 1
        assert out.shape[0] == 10
