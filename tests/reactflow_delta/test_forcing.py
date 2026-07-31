"""Tests for the local forcing operator (T-O0.1~T-O0.6, §4.3, §5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from reactflow.delta.forcing import (
    ForcingSupport,
    build_forcing_support,
    check_forcing_invariants,
    compute_forcing,
)


def _rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# T-O0.1 / T-O0.2: no-edit identity + hard-zero forcing
# ---------------------------------------------------------------------------


class TestNoEditIdentity:
    def test_support_construction(self):
        sup = build_forcing_support(20, edit_pos=10, local_window=3)
        assert sup.n == 20
        # Node window covers [7, 14).
        assert sup.node_mask.sum() == 7
        # Edit position is in the support.
        assert sup.node_mask[10]

    def test_identity_zero_forcing(self):
        rng = _rng()
        z = rng.standard_normal(20)
        sup = build_forcing_support(20, edit_pos=10, local_window=3)
        res = compute_forcing(z, z, sup)
        assert np.max(np.abs(res.node_forcing)) < 1e-7
        assert np.max(np.abs(res.edge_forcing)) < 1e-7

    def test_identity_audit_passes(self):
        rng = _rng()
        z = rng.standard_normal(16)
        sup = build_forcing_support(16, edit_pos=8, local_window=2)
        audit = check_forcing_invariants(z, z, sup)
        assert audit["identity"]["pass"]
        assert audit["identity"]["max_abs_node"] < 1e-7


# ---------------------------------------------------------------------------
# T-O0.3 / T-O0.4: endpoint swap + signed local node/edge forcing
# ---------------------------------------------------------------------------


class TestEndpointSwap:
    def test_swap_node_antisymmetry(self):
        rng = _rng()
        z_w = rng.standard_normal(20)
        z_m = rng.standard_normal(20)
        sup = build_forcing_support(20, edit_pos=10, local_window=3)
        b_fwd = compute_forcing(z_w, z_m, sup)
        b_bwd = compute_forcing(z_m, z_w, sup)
        err = np.max(np.abs(b_fwd.node_forcing + b_bwd.node_forcing))
        assert err < 1e-6

    def test_swap_edge_antisymmetry(self):
        rng = _rng()
        z_w = rng.standard_normal(20)
        z_m = rng.standard_normal(20)
        sup = build_forcing_support(20, edit_pos=10, local_window=3,
                                     contact_edges=[(8, 18)])
        b_fwd = compute_forcing(z_w, z_m, sup)
        b_bwd = compute_forcing(z_m, z_w, sup)
        err = np.max(np.abs(b_fwd.edge_forcing + b_bwd.edge_forcing))
        assert err < 1e-6

    def test_swap_audit_passes(self):
        rng = _rng()
        z_w = rng.standard_normal(16)
        z_m = rng.standard_normal(16)
        sup = build_forcing_support(16, edit_pos=8, local_window=2,
                                     contact_edges=[(6, 12)])
        audit = check_forcing_invariants(z_w, z_m, sup)
        assert audit["swap"]["pass"]
        assert audit["swap"]["max_abs_node_err"] < 1e-6


# ---------------------------------------------------------------------------
# T-O0.5 / T-O0.6: forcing support leakage + sparse symmetric-background K
# ---------------------------------------------------------------------------


class TestForcingLeakage:
    def test_off_support_node_zero(self):
        rng = _rng()
        z_w = rng.standard_normal(20)
        z_m = rng.standard_normal(20)
        sup = build_forcing_support(20, edit_pos=10, local_window=2)
        res = compute_forcing(z_w, z_m, sup)
        off = res.node_forcing[~sup.node_mask]
        assert np.max(np.abs(off)) == 0.0

    def test_off_support_edge_zero(self):
        rng = _rng()
        z_w = rng.standard_normal(20)
        z_m = rng.standard_normal(20)
        sup = build_forcing_support(20, edit_pos=10, local_window=2,
                                     contact_edges=[(10, 18)])
        res = compute_forcing(z_w, z_m, sup)
        off = res.edge_forcing[~sup.edge_mask]
        assert np.max(np.abs(off)) == 0.0

    def test_leakage_audit_passes(self):
        rng = _rng()
        z_w = rng.standard_normal(16)
        z_m = rng.standard_normal(16)
        sup = build_forcing_support(16, edit_pos=8, local_window=2)
        audit = check_forcing_invariants(z_w, z_m, sup)
        assert audit["leakage"]["pass"]
        assert audit["leakage"]["off_support_node"] == 0.0

    def test_remote_position_no_direct_mutation_token(self):
        """Remote positions (outside support) read no forcing (§4.3 rule 5)."""
        rng = _rng()
        z_w = rng.standard_normal(30)
        z_m = z_w.copy()
        z_m[15] += 5.0  # large perturbation at edit
        sup = build_forcing_support(30, edit_pos=15, local_window=2)
        res = compute_forcing(z_w, z_m, sup)
        # Position 0 is far from edit; forcing must be 0 there.
        assert res.node_forcing[0] == 0.0
        assert res.node_forcing[29] == 0.0

    def test_audit_dict_has_support(self):
        rng = _rng()
        z_w = rng.standard_normal(12)
        z_m = rng.standard_normal(12)
        sup = build_forcing_support(12, edit_pos=6, local_window=2)
        res = compute_forcing(z_w, z_m, sup)
        audit = res.to_audit_dict()
        assert audit["support"]["edit_pos"] == 6
        assert audit["off_support_node_leakage"] == 0.0
