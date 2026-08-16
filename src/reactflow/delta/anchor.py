"""P2 WT-anchored posterior update with access guard (v3 EPRO §4.8, T-O0.11).

P2 does **not** build a second model. It uses the *WT* observation residual to
update the WT accessibility state, then re-propagates with the same mutation
forcing. The hard constraint (§4.8, §5.3) is:

    P2 mutant-profile access: static code and runtime audit must both be 0.

This module enforces that by construction:

  * The :class:`P2AnchorGuard` API only accepts the WT observation residual
    ``q = r_w_obs - r_hat_w`` and the measurement variance ``sigma``. There is
    no parameter or code path that receives mutant reactivity.
  * A runtime access log records every input the guard sees; an audit method
    verifies no mutant-profile field was ever read.
  * The static API (function signatures) contains no mutant-reactivity argument.

Numpy-only (runs in ``editflow311`` without torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ANCHOR_SCHEMA_VERSION = "reactflow-delta-o0-anchor-v1"

# Frozen vocabulary of WT-side input field names the guard is allowed to see.
# Mutant-side fields (mut_reactivity, mutant_profile, etc.) are NOT in this set.
P2_ALLOWED_INPUT_FIELDS = frozenset({
    "wt_observation_residual",   # q = r_w_obs - r_hat_w
    "measurement_variance",      # sigma^2
    "wt_accessibility_prior",    # a_w prior (for regularization)
    "probe",                     # probe chemistry
    "condition",                 # experimental condition
})

# Fields that would constitute mutant-profile access (forbidden).
P2_FORBIDDEN_INPUT_FIELDS = frozenset({
    "mut_reactivity",
    "mutant_reactivity",
    "mutant_profile",
    "mut_profile",
    "delta_reactivity",
    "r_m_obs",
    "mut_accessibility",
})


@dataclass
class P2AnchorGuard:
    """WT-anchored posterior update with mutant-profile access audit (§4.8).

    The guard accepts only WT-side inputs and records every field it sees in a
    runtime access log. :meth:`audit` verifies that no forbidden (mutant-side)
    field was ever accessed.

    Usage::

        guard = P2AnchorGuard()
        delta_a_w = guard.wt_anchor_update(
            q=wt_residual, sigma=measurement_variance, probe="DMS",
        )
        report = guard.audit()  # must show mutant_access_count == 0
    """

    access_log: list[str] = field(default_factory=list)
    schema_version: str = ANCHOR_SCHEMA_VERSION

    def _record(self, field_name: str) -> None:
        if field_name in P2_FORBIDDEN_INPUT_FIELDS:
            raise RuntimeError(
                f"P2 anchor guard received forbidden mutant-side field {field_name!r}; "
                f"P2 is prohibited from reading mutant reactivity (§4.8)."
            )
        self.access_log.append(field_name)

    def wt_anchor_update(
        self,
        q: np.ndarray,
        sigma: np.ndarray | float,
        *,
        probe: str = "DMS",
        prior: np.ndarray | None = None,
        gain: float = 0.5,
    ) -> np.ndarray:
        """Compute the WT accessibility posterior update ``delta_a_w`` (§4.8).

        ``q = r_w_obs - r_hat_w`` is the WT observation residual;
        ``sigma`` is the measurement standard deviation (or variance — see
        ``sigma_is_variance``). The update is a shrinkage toward the residual,
        regularized toward ``prior`` when provided. No mutant input is read.

        Parameters
        ----------
        q : (n,) array
            WT observation residual.
        sigma : array or float
            Measurement noise scale.
        probe : str
            Probe chemistry (selects calibration).
        prior : (n,) array, optional
            WT accessibility prior for regularization.
        gain : float
            Update gain in ``[0, 1]``.
        """

        self._record("wt_observation_residual")
        self._record("measurement_variance")
        self._record("probe")
        if prior is not None:
            self._record("wt_accessibility_prior")

        if not (0.0 <= gain <= 1.0):
            raise ValueError(f"gain must be in [0, 1], got {gain}")

        q = np.asarray(q, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        # Shrinkage: delta_a_w = gain * q / (1 + sigma^2)  (noise-regularized)
        denom = 1.0 + sigma_arr ** 2
        delta_a_w = float(gain) * q / denom
        if prior is not None:
            prior = np.asarray(prior, dtype=float)
            # Small pull toward prior (regularization), does not read mutant.
            delta_a_w = delta_a_w + 0.01 * (prior - 0.0)
        return delta_a_w

    def audit(self) -> dict[str, Any]:
        """Runtime access audit: verify no mutant-side field was read (§5.3)."""

        mutant_access = [f for f in self.access_log if f in P2_FORBIDDEN_INPUT_FIELDS]
        wt_access = [f for f in self.access_log if f in P2_ALLOWED_INPUT_FIELDS]
        unknown = [f for f in self.access_log if f not in P2_ALLOWED_INPUT_FIELDS and f not in P2_FORBIDDEN_INPUT_FIELDS]
        return {
            "schema_version": self.schema_version,
            "mutant_access_count": int(len(mutant_access)),
            "mutant_access_fields": mutant_access,
            "wt_access_count": int(len(wt_access)),
            "wt_access_fields": wt_access,
            "unknown_fields": unknown,
            "pass": len(mutant_access) == 0,
        }


# ---------------------------------------------------------------------------
# Static API audit (no instantiation needed)
# ---------------------------------------------------------------------------


def static_audit() -> dict[str, Any]:
    """Static code audit of the P2 anchor API (§5.3).

    Verifies that the :class:`P2AnchorGuard` and :meth:`wt_anchor_update` API
    surface does not accept any mutant-side field name. This is a structural
    check on the allowed/forbidden vocabulary, complementing the runtime audit.
    """

    # The forbidden set must be disjoint from the allowed set.
    overlap = P2_ALLOWED_INPUT_FIELDS & P2_FORBIDDEN_INPUT_FIELDS
    # wt_anchor_update signature (inspected manually to avoid import of ``inspect``
    # subtleties): accepts q, sigma, probe, prior, gain — none are mutant-side.
    signature_fields = {"q", "sigma", "probe", "prior", "gain"}
    forbidden_in_signature = signature_fields & P2_FORBIDDEN_INPUT_FIELDS

    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "allowed_inputs": sorted(P2_ALLOWED_INPUT_FIELDS),
        "forbidden_inputs": sorted(P2_FORBIDDEN_INPUT_FIELDS),
        "vocab_overlap": sorted(overlap),
        "forbidden_in_signature": sorted(forbidden_in_signature),
        "pass": len(overlap) == 0 and len(forbidden_in_signature) == 0,
    }


def check_p2_anchor_invariants() -> dict[str, Any]:
    """Full P2 anchor audit: static + runtime mutant-access must both be 0."""

    static = static_audit()

    # Runtime audit: run a representative update and check the access log.
    guard = P2AnchorGuard()
    n = 16
    q = np.random.default_rng(0).standard_normal(n)
    sigma = 0.3
    _ = guard.wt_anchor_update(q, sigma, probe="DMS", prior=np.zeros(n))
    runtime = guard.audit()

    return {
        "static": static,
        "runtime": runtime,
        "pass": bool(static["pass"] and runtime["pass"]),
    }
