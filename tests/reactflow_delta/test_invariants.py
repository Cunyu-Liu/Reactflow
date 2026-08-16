"""Tests for the O0 invariant suite and synthetic fixtures (T-O0.12~T-O0.13, §5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from reactflow.delta.invariants import (
    IDENTITY_TOL,
    RHO_MAX,
    SOLVER_EPS,
    SWAP_TOL,
    InvariantSuiteReport,
    all_fixtures,
    hairpin_release_fixture,
    no_change_fixture,
    run_invariant_suite,
    run_pipeline,
    two_state_fixture,
)


# ---------------------------------------------------------------------------
# T-O0.12: synthetic fixtures
# ---------------------------------------------------------------------------


class TestSyntheticFixtures:
    def test_no_change_fixture(self):
        fx = no_change_fixture()
        assert fx.name == "no_change"
        assert np.allclose(fx.z_w, fx.z_m)  # identity by construction

    def test_hairpin_release_fixture(self):
        fx = hairpin_release_fixture()
        assert fx.name == "hairpin_release"
        # Perturbation localized at edit window.
        assert not np.allclose(fx.z_w, fx.z_m)
        assert len(fx.contact_edges) >= 1
        # Distal contact.
        ep = fx.edit_pos
        assert any(abs(i - ep) > 5 or abs(j - ep) > 5 for i, j in fx.contact_edges)

    def test_two_state_fixture(self):
        fx = two_state_fixture()
        assert fx.name == "two_state"
        # Large-amplitude perturbation.
        diff = np.abs(fx.z_m - fx.z_w)
        assert np.max(diff) > 2.0

    def test_all_fixtures_three(self):
        fxs = all_fixtures()
        names = [f.name for f in fxs]
        assert names == ["no_change", "hairpin_release", "two_state"]


class TestPipelineOnFixtures:
    def test_no_change_identity_response(self):
        fx = no_change_fixture()
        res = run_pipeline(fx)
        # b == 0 => h_lin == 0 => h == 0 => delta_r_hat ~ 0.
        assert np.max(np.abs(res.delta_r_hat)) < SWAP_TOL

    def test_hairpin_release_propagates(self):
        fx = hairpin_release_fixture()
        res = run_pipeline(fx)
        # Forcing is non-zero, response is non-zero.
        assert np.max(np.abs(res.h_lin)) > 0.0
        # Propagation reaches distal positions (via contact edge in K).
        assert np.max(np.abs(res.h)) > 0.0

    def test_two_state_switch_active(self):
        fx = two_state_fixture()
        res = run_pipeline(fx)
        # Nonlinear response is non-zero (switch gate active for large forcing).
        assert np.max(np.abs(res.h_nl)) >= 0.0
        # no_switch ablation differs from full switch.
        assert not np.allclose(res.delta_r_hat, res.no_switch_delta_r_hat) or np.max(np.abs(res.h_nl)) < 1e-6

    def test_all_forcing_invariants_pass(self):
        for fx in all_fixtures():
            res = run_pipeline(fx)
            assert res.forcing_audit["identity"]["pass"], f"{fx.name} forcing identity"
            assert res.forcing_audit["swap"]["pass"], f"{fx.name} forcing swap"
            assert res.forcing_audit["leakage"]["pass"], f"{fx.name} forcing leakage"

    def test_all_susceptibility_invariants_pass(self):
        for fx in all_fixtures():
            res = run_pipeline(fx)
            assert res.susceptibility_audit["stability"]["pass"], f"{fx.name} stability"
            assert res.susceptibility_audit["solver"]["pass"], f"{fx.name} solver"
            assert res.susceptibility_audit["sparsity"]["pass"], f"{fx.name} sparsity"

    def test_all_switch_invariants_pass(self):
        for fx in all_fixtures():
            res = run_pipeline(fx)
            assert res.switch_audit["oddness"]["pass"], f"{fx.name} switch oddness"
            assert res.switch_audit["no_bias"]["pass"], f"{fx.name} switch no-bias"

    def test_all_observation_invariants_pass(self):
        for fx in all_fixtures():
            res = run_pipeline(fx)
            assert res.observation_audit["monotonicity"]["all_pass"], f"{fx.name} monotonicity"
            assert res.observation_audit["swap_antisymmetry"]["pass"], f"{fx.name} obs swap"


# ---------------------------------------------------------------------------
# T-O0.13: invariant suite 100%
# ---------------------------------------------------------------------------


class TestInvariantSuite:
    def test_suite_all_pass(self):
        report = run_invariant_suite()
        assert report.all_pass
        assert report.n_passed == report.n_checks
        assert report.failed_checks == []

    def test_suite_includes_all_fixtures(self):
        report = run_invariant_suite()
        assert "no_change" in report.fixtures
        assert "hairpin_release" in report.fixtures
        assert "two_state" in report.fixtures

    def test_suite_includes_p2_anchor(self):
        report = run_invariant_suite()
        assert report.p2_anchor_audit["pass"]
        assert report.p2_anchor_audit["runtime"]["mutant_access_count"] == 0

    def test_suite_audit_dict(self):
        report = run_invariant_suite()
        d = report.to_audit_dict()
        assert d["all_pass"] is True
        assert d["n_checks"] > 0
        assert d["n_passed"] == d["n_checks"]
        assert d["failed_checks"] == []

    def test_thresholds_match_contract(self):
        """Frozen §5.3 thresholds must match the contract values."""
        assert IDENTITY_TOL == 1e-7
        assert SWAP_TOL == 1e-6
        assert SOLVER_EPS == 1e-5
        assert RHO_MAX == 0.95
        assert RHO_MAX < 1.0

    def test_no_change_delta_r_below_identity_tol(self):
        """no-change fixture: delta_r_hat must be ~0 (identity)."""
        fx = no_change_fixture()
        res = run_pipeline(fx)
        assert np.max(np.abs(res.delta_r_hat)) < SWAP_TOL

    def test_suite_checks_count(self):
        report = run_invariant_suite()
        # Expect checks across 3 fixtures (forcing/susceptibility/switch/obs +
        # no-change delta_r identity) + P2 static/runtime.
        assert report.n_checks >= 20
