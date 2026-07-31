"""Local mutation forcing operator (v3 EPRO §4.3, Phase O0 T-O0.1~T-O0.6).

The forcing operator ``b = B(z_w, z_m, e, c)`` maps a pair of endpoint latent
states (WT ``z_w``, mutant ``z_m``) to a directed perturbation vector supported
on an edit-centered local window. Three hard invariants are enforced *by
construction* (§4.3, §5.3):

  1. **No-edit identity**: ``x_w == x_m  =>  b = 0``.
  2. **Endpoint-swap antisymmetry**: ``B(z_m, z_w) = -B(z_w, z_m)``.
  3. **Support leakage**: forcing is strictly zero outside ``forcing_support_mask``.

Construction. Write the node forcing as

    b_i = w_i(z_w, z_m) * (z_w_i - z_m_i)

where ``w_i`` is a *symmetric* function of the endpoints (``w_i(z_w,z_m) =
w_i(z_m,z_w)``), computed only from symmetric features (sums, products, absolute
differences) and restricted to the support mask. Then:

  * identity: ``z_w == z_m  =>  (z_w - z_m) == 0  =>  b == 0``;
  * swap: ``B(z_m,z_w) = w_i(z_m,z_w)*(z_m-z_w) = w_i(z_w,z_m)*(-(z_w-z_m)) = -b_i``;
  * leakage: ``w_i`` is multiplied by the support mask, so off-support ``b_i = 0``.

Edge forcing ``b_{ij}`` follows the same antisymmetric decomposition on edges
incident to the edit window. Edge forcing is recorded as an auditable artifact;
the linear response ``h_lin = (I-K)^{-1} b`` uses the node forcing vector.

This module is numpy-only (runs in ``editflow311`` without torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

FORCING_SCHEMA_VERSION = "reactflow-delta-o0-forcing-v1"


# ---------------------------------------------------------------------------
# Support mask construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForcingSupport:
    """Edit-centered support mask for node and edge forcing (§4.3).

    ``node_mask`` is a bool array of length ``n``; ``edge_mask`` is a bool
    ``(n, n)`` array. Both are ``True`` only where forcing is permitted. The
    edit position (0-indexed) is recorded for auditability.
    """

    node_mask: np.ndarray  # bool (n,)
    edge_mask: np.ndarray  # bool (n, n)
    edit_pos: int  # 0-indexed edit position
    local_window: int  # node forcing window radius
    n: int

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "edit_pos": int(self.edit_pos),
            "local_window": int(self.local_window),
            "n": int(self.n),
            "node_support_count": int(self.node_mask.sum()),
            "edge_support_count": int(self.edge_mask.sum()),
        }


def build_forcing_support(
    n: int,
    edit_pos: int,
    *,
    local_window: int = 3,
    contact_edges: list[tuple[int, int]] | None = None,
) -> ForcingSupport:
    """Build the edit-centered forcing support mask (§4.3 rules 3-4).

    Node forcing is allowed within ``local_window`` of the edit position
    (sequence-distance window). Edge forcing is allowed on sequence-adjacent
    edges and on explicitly provided contact edges, restricted to edges that
    touch the edit-centered node window.

    Parameters
    ----------
    n : int
        Sequence length.
    edit_pos : int
        0-indexed edit position.
    local_window : int
        Radius of the edit-centered node window.
    contact_edges : list of (i, j) tuples, optional
        Sparse contact edges (e.g. union of WT/mutant BPP contacts). Edges
        touching the node window are added to the edge support.
    """

    if not (0 <= edit_pos < n):
        raise ValueError(f"edit_pos {edit_pos} out of range [0, {n})")
    if local_window < 0:
        raise ValueError("local_window must be non-negative")

    node_mask = np.zeros(n, dtype=bool)
    lo = max(0, edit_pos - local_window)
    hi = min(n, edit_pos + local_window + 1)
    node_mask[lo:hi] = True

    edge_mask = np.zeros((n, n), dtype=bool)
    # Sequence-adjacent edges within the node window.
    for i in range(lo, hi - 1):
        edge_mask[i, i + 1] = True
        edge_mask[i + 1, i] = True
    # Contact edges that touch the node window (at least one endpoint in window).
    if contact_edges is not None:
        for i, j in contact_edges:
            if not (0 <= i < n and 0 <= j < n) or i == j:
                continue
            if node_mask[i] or node_mask[j]:
                edge_mask[i, j] = True
                edge_mask[j, i] = True

    return ForcingSupport(
        node_mask=node_mask,
        edge_mask=edge_mask,
        edit_pos=int(edit_pos),
        local_window=int(local_window),
        n=int(n),
    )


# ---------------------------------------------------------------------------
# Forcing result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForcingResult:
    """Output of :func:`compute_forcing`."""

    node_forcing: np.ndarray  # (n,) b_i
    edge_forcing: np.ndarray  # (n, n) b_{ij} (antisymmetric matrix)
    support: ForcingSupport
    schema_version: str = FORCING_SCHEMA_VERSION

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "support": self.support.to_audit_dict(),
            "node_forcing_max_abs": float(np.max(np.abs(self.node_forcing)))
            if self.node_forcing.size
            else 0.0,
            "edge_forcing_max_abs": float(np.max(np.abs(self.edge_forcing)))
            if self.edge_forcing.size
            else 0.0,
            "off_support_node_leakage": float(
                np.max(np.abs(self.node_forcing[~self.support.node_mask]))
            )
            if self.node_forcing.size
            else 0.0,
            "off_support_edge_leakage": float(
                np.max(np.abs(self.edge_forcing[~self.support.edge_mask]))
            )
            if self.edge_forcing.size
            else 0.0,
        }


# ---------------------------------------------------------------------------
# Default symmetric weight functions
# ---------------------------------------------------------------------------


def _default_node_weight(
    z_w: np.ndarray, z_m: np.ndarray
) -> np.ndarray:
    """Symmetric node weight ``w_i(z_w, z_m) = w_i(z_m, z_w)``.

    Uses symmetric features: ``z_w + z_m`` (sum) and ``|z_m - z_w|`` (absolute
    difference). Both are invariant under endpoint swap, so the weight is
    symmetric. Output is non-negative (a magnitude scale).
    """

    s = z_w + z_m  # symmetric
    d = np.abs(z_m - z_w)  # symmetric
    # A fixed, non-negative scalar field. softplus-like to keep non-negative.
    w = np.log1p(np.exp(np.clip(0.5 * s + 0.25 * d, -30.0, 30.0)))
    return w


def _default_edge_weight(
    z_w: np.ndarray, z_m: np.ndarray
) -> np.ndarray:
    """Symmetric edge weight matrix (non-negative, swap-invariant)."""

    s = (z_w[:, None] + z_m[None, :]) + (z_m[:, None] + z_w[None, :])  # symmetric
    w = np.log1p(np.exp(np.clip(0.25 * s, -30.0, 30.0)))
    # zero diagonal (no self-edge forcing)
    np.fill_diagonal(w, 0.0)
    return w


# ---------------------------------------------------------------------------
# Core forcing operator
# ---------------------------------------------------------------------------


def compute_forcing(
    z_w: np.ndarray,
    z_m: np.ndarray,
    support: ForcingSupport,
    *,
    node_weight_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    edge_weight_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    node_scale: float = 1.0,
    edge_scale: float = 1.0,
) -> ForcingResult:
    """Compute the directed local forcing ``b = B(z_w, z_m)`` (§4.3).

    Guarantees by construction:
      * ``z_w == z_m  =>  b == 0`` (identity);
      * ``B(z_m, z_w) == -B(z_w, z_m)`` (swap antisymmetry);
      * off-support forcing is exactly zero (leakage).

    Parameters
    ----------
    z_w, z_m : (n,) float arrays
        WT and mutant endpoint latent states.
    support : ForcingSupport
        Edit-centered support mask.
    node_weight_fn, edge_weight_fn : callable, optional
        Symmetric weight functions. Defaults are non-negative symmetric fields.
    node_scale, edge_scale : float
        Global scalar multipliers.
    """

    z_w = np.asarray(z_w, dtype=float)
    z_m = np.asarray(z_m, dtype=float)
    if z_w.shape != z_m.shape:
        raise ValueError(f"z_w shape {z_w.shape} != z_m shape {z_m.shape}")
    if z_w.ndim != 1:
        raise ValueError("z_w, z_m must be 1-D arrays")
    if z_w.shape[0] != support.n:
        raise ValueError(
            f"z_w length {z_w.shape[0]} != support.n {support.n}"
        )

    nwt = node_weight_fn or _default_node_weight
    ewt = edge_weight_fn or _default_edge_weight

    # Node forcing: b_i = w_i(z_w,z_m) * (z_w_i - z_m_i) * scale, masked.
    w_node = nwt(z_w, z_m)
    delta_node = z_w - z_m  # antisymmetric under swap
    b_node = node_scale * w_node * delta_node
    b_node = b_node * support.node_mask  # zero outside support

    # Edge forcing: b_{ij} = w_{ij}(z_w,z_m) * (z_w_i - z_m_j ... ) -- use
    # antisymmetric endpoint difference on each edge. A clean antisymmetric
    # form: b_{ij} = w_{ij} * ((z_w_i - z_m_i) + (z_w_j - z_m_j)) is symmetric
    # in i,j which is wrong. Instead use b_{ij} = w_{ij} * (z_w_i - z_m_j) but
    # that is NOT antisymmetric under swap. The correct construction: edge
    # forcing must flip sign under swap. Use b_{ij} = w_{ij} * (delta_i - delta_j)
    # where delta = z_w - z_m (antisymmetric vector). Then:
    #   swap: delta -> -delta  =>  b_{ij} -> -b_{ij}.  Good.
    #   identity: delta = 0  =>  b_{ij} = 0.  Good.
    w_edge = ewt(z_w, z_m)
    delta = z_w - z_m  # (n,) antisymmetric
    # b_{ij} = w_{ij} * (delta_i - delta_j)
    b_edge = edge_scale * w_edge * (delta[:, None] - delta[None, :])
    b_edge = b_edge * support.edge_mask
    np.fill_diagonal(b_edge, 0.0)

    return ForcingResult(
        node_forcing=b_node,
        edge_forcing=b_edge,
        support=support,
    )


def check_forcing_invariants(
    z_w: np.ndarray,
    z_m: np.ndarray,
    support: ForcingSupport,
    *,
    identity_tol: float = 1e-7,
    swap_tol: float = 1e-6,
    **kwargs: Any,
) -> dict[str, Any]:
    """Audit the three forcing invariants (§5.3) for a given endpoint pair."""

    # Identity: z_w == z_m => b == 0
    b_identity = compute_forcing(z_w, z_w, support, **kwargs)
    identity_max_abs = float(np.max(np.abs(b_identity.node_forcing))) if b_identity.node_forcing.size else 0.0
    identity_edge_max = float(np.max(np.abs(b_identity.edge_forcing))) if b_identity.edge_forcing.size else 0.0

    # Swap: B(z_m, z_w) == -B(z_w, z_m)
    b_forward = compute_forcing(z_w, z_m, support, **kwargs)
    b_backward = compute_forcing(z_m, z_w, support, **kwargs)
    swap_node_err = float(np.max(np.abs(b_forward.node_forcing + b_backward.node_forcing))) if b_forward.node_forcing.size else 0.0
    swap_edge_err = float(np.max(np.abs(b_forward.edge_forcing + b_backward.edge_forcing))) if b_forward.edge_forcing.size else 0.0

    # Leakage: off-support forcing == 0
    leak_node = float(np.max(np.abs(b_forward.node_forcing[~support.node_mask]))) if b_forward.node_forcing.size else 0.0
    leak_edge = float(np.max(np.abs(b_forward.edge_forcing[~support.edge_mask]))) if b_forward.edge_forcing.size else 0.0

    return {
        "identity": {
            "max_abs_node": identity_max_abs,
            "max_abs_edge": identity_edge_max,
            "pass": max(identity_max_abs, identity_edge_max) < identity_tol,
            "tol": identity_tol,
        },
        "swap": {
            "max_abs_node_err": swap_node_err,
            "max_abs_edge_err": swap_edge_err,
            "pass": max(swap_node_err, swap_edge_err) < swap_tol,
            "tol": swap_tol,
        },
        "leakage": {
            "off_support_node": leak_node,
            "off_support_edge": leak_edge,
            "pass": max(leak_node, leak_edge) == 0.0,
        },
    }
