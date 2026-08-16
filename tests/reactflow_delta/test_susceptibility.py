"""Tests for the stable susceptibility operator (T-O0.7~T-O0.8, §4.4, §5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from reactflow.delta.susceptibility import (
    build_kernel,
    build_symmetric_background,
    check_susceptibility_invariants,
    solve_response,
)


def _rng():
    return np.random.default_rng(1)


# ---------------------------------------------------------------------------
# Symmetric background swap-invariance
# ---------------------------------------------------------------------------


class TestSymmetricBackground:
    def test_swap_invariant(self):
        rng = _rng()
        z_w = rng.standard_normal(16)
        z_m = rng.standard_normal(16)
        zb = build_symmetric_background(z_w, z_m)
        zb_swap = build_symmetric_background(z_m, z_w)
        assert np.allclose(zb.z_sum, zb_swap.z_sum)
        assert np.allclose(zb.z_prod, zb_swap.z_prod)
        assert np.allclose(zb.z_abs_diff, zb_swap.z_abs_diff)

    def test_stack_shape(self):
        rng = _rng()
        z_w = rng.standard_normal(10)
        z_m = rng.standard_normal(10)
        zb = build_symmetric_background(z_w, z_m)
        assert zb.stack().shape == (3, 10)


# ---------------------------------------------------------------------------
# T-O0.7: spectral / stability
# ---------------------------------------------------------------------------


class TestStability:
    def test_rho_below_max(self):
        rng = _rng()
        z_w = rng.standard_normal(16)
        z_m = rng.standard_normal(16)
        zb = build_symmetric_background(z_w, z_m)
        edges = [(i, i + 1) for i in range(15)] + [(5, 12)]
        K = build_kernel(zb, edges, rho_max=0.9)
        assert K.rho <= 0.9 + 1e-12
        assert K.rho_max < 1.0

    def test_rho_max_in_open_interval(self):
        rng = _rng()
        zb = build_symmetric_background(rng.standard_normal(8), rng.standard_normal(8))
        edges = [(i, i + 1) for i in range(7)]
        K = build_kernel(zb, edges, rho_max=0.95)
        assert 0.0 < K.rho_max < 1.0

    def test_rejects_invalid_rho_max(self):
        zb = build_symmetric_background(np.zeros(4), np.zeros(4))
        for bad in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                build_kernel(zb, [], rho_max=bad)

    def test_kernel_symmetric(self):
        rng = _rng()
        zb = build_symmetric_background(rng.standard_normal(12), rng.standard_normal(12))
        edges = [(i, i + 1) for i in range(11)]
        K = build_kernel(zb, edges, rho_max=0.9)
        assert K.symmetric
        assert np.allclose(K.K, K.K.T)

    def test_kernel_swap_invariant(self):
        rng = _rng()
        z_w = rng.standard_normal(12)
        z_m = rng.standard_normal(12)
        edges = [(i, i + 1) for i in range(11)]
        K = build_kernel(build_symmetric_background(z_w, z_m), edges, rho_max=0.9)
        K_swap = build_kernel(build_symmetric_background(z_m, z_w), edges, rho_max=0.9)
        assert np.allclose(K.K, K_swap.K)

    def test_sparsity(self):
        rng = _rng()
        zb = build_symmetric_background(rng.standard_normal(12), rng.standard_normal(12))
        edges = [(0, 1), (5, 6), (10, 11)]
        K = build_kernel(zb, edges, rho_max=0.9)
        # Only 6 symmetric off-diagonal entries should be non-zero.
        assert np.count_nonzero(K.K) == 6


# ---------------------------------------------------------------------------
# T-O0.8: stable solver + residual audit
# ---------------------------------------------------------------------------


class TestSolver:
    def test_direct_residual(self):
        rng = _rng()
        zb = build_symmetric_background(rng.standard_normal(16), rng.standard_normal(16))
        edges = [(i, i + 1) for i in range(15)]
        K = build_kernel(zb, edges, rho_max=0.9)
        b = rng.standard_normal(16)
        sol = solve_response(b, K.K, method="direct", eps_solver=1e-5)
        assert sol.relative_residual < 1e-5

    def test_neumann_residual(self):
        rng = _rng()
        zb = build_symmetric_background(rng.standard_normal(16), rng.standard_normal(16))
        edges = [(i, i + 1) for i in range(15)]
        K = build_kernel(zb, edges, rho_max=0.9)
        b = rng.standard_normal(16)
        sol = solve_response(b, K.K, method="neumann", eps_solver=1e-5, max_iter=5000)
        assert sol.relative_residual < 1e-5

    def test_direct_matches_neumann(self):
        rng = _rng()
        zb = build_symmetric_background(rng.standard_normal(12), rng.standard_normal(12))
        edges = [(i, i + 1) for i in range(11)]
        K = build_kernel(zb, edges, rho_max=0.8)
        b = rng.standard_normal(12)
        sol_d = solve_response(b, K.K, method="direct")
        sol_n = solve_response(b, K.K, method="neumann", eps_solver=1e-8, max_iter=10000)
        assert np.allclose(sol_d.h, sol_n.h, atol=1e-4)

    def test_residual_audit_dict(self):
        rng = _rng()
        zb = build_symmetric_background(rng.standard_normal(8), rng.standard_normal(8))
        edges = [(i, i + 1) for i in range(7)]
        K = build_kernel(zb, edges, rho_max=0.9)
        b = rng.standard_normal(8)
        sol = solve_response(b, K.K)
        audit = sol.to_audit_dict()
        assert audit["pass"]
        assert audit["relative_residual"] < 1e-5

    def test_aggregated_audit(self):
        rng = _rng()
        z_w = rng.standard_normal(16)
        z_m = rng.standard_normal(16)
        edges = [(i, i + 1) for i in range(15)] + [(3, 10)]
        audit = check_susceptibility_invariants(z_w, z_m, edges, rho_max=0.9)
        assert audit["swap_invariance"]["pass"]
        assert audit["stability"]["pass"]
        assert audit["sparsity"]["pass"]
        assert audit["solver"]["pass"]
