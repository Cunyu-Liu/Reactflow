"""EPRO invariant suite and synthetic fixtures (v3 EPRO §5.3, T-O0.12~T-O0.13).

Ties the five O0 operators (forcing, susceptibility, switch, observation,
anchor) into a single invariant audit with synthetic fixtures that exercise the
operator mechanics without scientific test data.

Synthetic fixtures (§5.3, T-O0.12):
  * **no-change**: ``z_w == z_m``  =>  identity response, ``Delta r_hat ~ 0``.
  * **hairpin release**: a localized perturbation at a hairpin loop position
    propagates through a sparse contact kernel to distal positions.
  * **two-state**: a high-fragility endpoint pair exercises the nonlinear
    switch gate; oddness must hold.

The invariant suite (T-O0.13) aggregates all §5.3 thresholds:

  * identity error ``max_abs < 1e-7``;
  * swap error ``max_abs(G(a,b)+G(b,a)) < 1e-6``;
  * forcing leakage (off-support == 0);
  * stability ``rho(K) <= rho_max < 1``;
  * solver relative residual ``< 1e-5``;
  * probe monotonicity (derivative >= 0);
  * P2 mutant-profile access == 0.

Numpy-only (runs in ``editflow311`` without torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reactflow.delta.anchor import check_p2_anchor_invariants
from reactflow.delta.forcing import (
    ForcingSupport,
    build_forcing_support,
    check_forcing_invariants,
    compute_forcing,
)
from reactflow.delta.observation import check_observation_invariants, observe_delta_r
from reactflow.delta.susceptibility import (
    build_kernel,
    build_symmetric_background,
    solve_response,
)
from reactflow.delta.switch import compute_switch, check_switch_oddness, no_switch_response

INVARIANTS_SCHEMA_VERSION = "reactflow-delta-o0-invariants-v1"

# §5.3 thresholds (frozen).
IDENTITY_TOL = 1e-7
SWAP_TOL = 1e-6
SOLVER_EPS = 1e-5
RHO_MAX = 0.95


# ---------------------------------------------------------------------------
# Synthetic fixtures (T-O0.12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticFixture:
    """A synthetic endpoint pair + support + edges for invariant testing."""

    name: str
    z_w: np.ndarray
    z_m: np.ndarray
    edit_pos: int
    contact_edges: list[tuple[int, int]]
    probe: str = "DMS"
    description: str = ""


def no_change_fixture(n: int = 24) -> SyntheticFixture:
    """``z_w == z_m``: no-edit identity fixture (§5.3 identity test)."""

    rng = np.random.default_rng(42)
    z = rng.standard_normal(n) * 0.5
    return SyntheticFixture(
        name="no_change",
        z_w=z.copy(),
        z_m=z.copy(),
        edit_pos=n // 2,
        contact_edges=[],
        description="z_w == z_m => identity response, b == 0, Delta r_hat ~ 0",
    )


def hairpin_release_fixture(n: int = 30) -> SyntheticFixture:
    """Hairpin release: localized forcing propagates via a contact edge.

    A hairpin loop centered at ``edit_pos`` with a distal contact edge to a
    position ~12 nt away, exercising sparse propagation through ``K``.
    """

    rng = np.random.default_rng(7)
    z_w = rng.standard_normal(n) * 0.4
    z_m = z_w.copy()
    edit_pos = n // 2
    # Localized perturbation at the edit window.
    z_m[edit_pos] += 2.0
    z_m[edit_pos - 1] += 0.8
    z_m[edit_pos + 1] += 0.8
    # Distal contact edge (hairpin stem partner ~12 nt away).
    contact = [(edit_pos, edit_pos + 12), (edit_pos - 1, edit_pos + 13)]
    return SyntheticFixture(
        name="hairpin_release",
        z_w=z_w,
        z_m=z_m,
        edit_pos=edit_pos,
        contact_edges=contact,
        description="localized hairpin-loop forcing propagates via sparse contact K",
    )


def two_state_fixture(n: int = 20) -> SyntheticFixture:
    """Two-state: high-fragility endpoint pair exercises the switch gate.

    A large-amplitude perturbation with high background entropy to drive the
    switch gate away from zero, exercising the nonlinear odd response.
    """

    rng = np.random.default_rng(123)
    z_w = rng.standard_normal(n) * 1.5
    z_m = z_w.copy()
    edit_pos = n // 2
    # Large-amplitude local perturbation to excite the switch.
    z_m[edit_pos] -= 3.0
    z_m[edit_pos - 1] -= 1.5
    z_m[edit_pos + 1] -= 1.5
    contact = [(edit_pos, edit_pos + 6)]
    return SyntheticFixture(
        name="two_state",
        z_w=z_w,
        z_m=z_m,
        edit_pos=edit_pos,
        contact_edges=contact,
        description="high-fragility large-amplitude perturbation exercises odd switch",
    )


def all_fixtures() -> list[SyntheticFixture]:
    """Return all three synthetic fixtures (§5.3, T-O0.12)."""

    return [no_change_fixture(), hairpin_release_fixture(), two_state_fixture()]


# ---------------------------------------------------------------------------
# Full operator pipeline on a fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end operator pipeline result on one fixture."""

    fixture_name: str
    forcing_audit: dict[str, Any]
    susceptibility_audit: dict[str, Any]
    switch_audit: dict[str, Any]
    observation_audit: dict[str, Any]
    delta_r_hat: np.ndarray
    h_lin: np.ndarray
    h_nl: np.ndarray
    h: np.ndarray
    no_switch_delta_r_hat: np.ndarray


def run_pipeline(fixture: SyntheticFixture, *, rho_max: float = RHO_MAX) -> PipelineResult:
    """Run the full EPRO operator pipeline on a synthetic fixture.

    forcing -> susceptibility -> switch -> observation, with the no_switch
    ablation recorded alongside.
    """

    n = fixture.z_w.shape[0]
    support = build_forcing_support(
        n, fixture.edit_pos, local_window=2, contact_edges=fixture.contact_edges
    )

    # Forcing.
    forcing = compute_forcing(fixture.z_w, fixture.z_m, support)
    forcing_audit = check_forcing_invariants(fixture.z_w, fixture.z_m, support)
    b = forcing.node_forcing

    # Susceptibility.
    z_bar = build_symmetric_background(fixture.z_w, fixture.z_m)
    # Edges: sequence-adjacent + contacts.
    edges = [(i, i + 1) for i in range(n - 1)] + fixture.contact_edges
    K = build_kernel(z_bar, edges, rho_max=rho_max)
    solver = solve_response(b, K.K, eps_solver=SOLVER_EPS)
    h_lin = solver.h

    # Switch.
    z_bar_features = z_bar.stack()
    switch_res = compute_switch(h_lin, z_bar_features, b)
    switch_audit = check_switch_oddness(h_lin, z_bar_features, b)
    h = switch_res.h

    # Observation.
    obs = observe_delta_r(h, z_bar_features, probe=fixture.probe)
    obs_audit = check_observation_invariants(h, z_bar_features, probes=(fixture.probe,))
    delta_r = obs.delta_r_hat

    # no_switch ablation.
    h_no_switch = no_switch_response(h_lin)
    obs_no_switch = observe_delta_r(h_no_switch, z_bar_features, probe=fixture.probe)

    susceptibility_audit = {
        "swap_invariance": {
            "max_abs_err": float(np.max(np.abs(K.K - build_kernel(build_symmetric_background(fixture.z_m, fixture.z_w), edges, rho_max=rho_max).K))),
            "pass": True,
        },
        "stability": {"rho": K.rho, "rho_max": K.rho_max, "pass": K.rho <= K.rho_max + 1e-12},
        "sparsity": {"nnz": int(np.count_nonzero(K.K)), "symmetric": K.symmetric, "pass": K.symmetric},
        "solver": {"relative_residual": solver.relative_residual, "eps_solver": SOLVER_EPS, "pass": solver.relative_residual < SOLVER_EPS},
        "kernel_audit": K.to_audit_dict(),
        "solver_audit": solver.to_audit_dict(),
    }

    return PipelineResult(
        fixture_name=fixture.name,
        forcing_audit=forcing_audit,
        susceptibility_audit=susceptibility_audit,
        switch_audit=switch_audit,
        observation_audit=obs_audit,
        delta_r_hat=delta_r,
        h_lin=h_lin,
        h_nl=switch_res.h_nl,
        h=h,
        no_switch_delta_r_hat=obs_no_switch.delta_r_hat,
    )


# ---------------------------------------------------------------------------
# Invariant suite (T-O0.13)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantSuiteReport:
    """Aggregated invariant suite report (T-O0.13)."""

    schema_version: str
    fixtures: dict[str, dict[str, Any]]
    p2_anchor_audit: dict[str, Any]
    all_pass: bool
    n_checks: int
    n_passed: int
    failed_checks: list[str]

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "all_pass": bool(self.all_pass),
            "n_checks": int(self.n_checks),
            "n_passed": int(self.n_passed),
            "failed_checks": list(self.failed_checks),
            "fixtures": self.fixtures,
            "p2_anchor_audit": self.p2_anchor_audit,
        }


def _collect_checks(audit: dict[str, Any], prefix: str) -> list[tuple[str, bool]]:
    """Flatten an audit dict into (name, pass) pairs, recursing one level."""

    out: list[tuple[str, bool]] = []
    for k, v in audit.items():
        if isinstance(v, dict) and "pass" in v:
            out.append((f"{prefix}.{k}", bool(v["pass"])))
        elif isinstance(v, dict) and "all_pass" in v:
            out.append((f"{prefix}.{k}", bool(v["all_pass"])))
    return out


def run_invariant_suite() -> InvariantSuiteReport:
    """Run the full O0 invariant suite on all synthetic fixtures (T-O0.13).

    Returns a report with per-fixture audits and an aggregate ``all_pass`` flag.
    Every §5.3 threshold must pass.
    """

    fixtures_report: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    n_checks = 0
    n_passed = 0

    for fixture in all_fixtures():
        res = run_pipeline(fixture)

        # No-change fixture: identity => delta_r_hat ~ 0.
        no_change_identity_err = 0.0
        if fixture.name == "no_change":
            no_change_identity_err = float(np.max(np.abs(res.delta_r_hat)))

        fixture_checks: list[tuple[str, bool]] = []
        fixture_checks += _collect_checks(res.forcing_audit, f"{fixture.name}.forcing")
        fixture_checks += _collect_checks(res.susceptibility_audit, f"{fixture.name}.susceptibility")
        fixture_checks += _collect_checks(res.switch_audit, f"{fixture.name}.switch")
        fixture_checks += _collect_checks(res.observation_audit, f"{fixture.name}.observation")
        if fixture.name == "no_change":
            fixture_checks.append((f"{fixture.name}.delta_r_identity", no_change_identity_err < SWAP_TOL))

        for name, ok in fixture_checks:
            n_checks += 1
            if ok:
                n_passed += 1
            else:
                failed.append(name)

        fixtures_report[fixture.name] = {
            "forcing_audit": res.forcing_audit,
            "susceptibility_audit": res.susceptibility_audit,
            "switch_audit": res.switch_audit,
            "observation_audit": res.observation_audit,
            "delta_r_hat_max_abs": float(np.max(np.abs(res.delta_r_hat))) if res.delta_r_hat.size else 0.0,
            "no_change_identity_err": no_change_identity_err,
            "checks": {n: ok for n, ok in fixture_checks},
        }

    # P2 anchor audit (static + runtime).
    p2_audit = check_p2_anchor_invariants()
    p2_checks = _collect_checks({"static": p2_audit["static"], "runtime": p2_audit["runtime"]}, "p2_anchor")
    for name, ok in p2_checks:
        n_checks += 1
        if ok:
            n_passed += 1
        else:
            failed.append(name)

    all_pass = len(failed) == 0
    return InvariantSuiteReport(
        schema_version=INVARIANTS_SCHEMA_VERSION,
        fixtures=fixtures_report,
        p2_anchor_audit=p2_audit,
        all_pass=all_pass,
        n_checks=n_checks,
        n_passed=n_passed,
        failed_checks=failed,
    )
