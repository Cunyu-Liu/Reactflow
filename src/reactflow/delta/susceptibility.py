"""Stable susceptibility operator (v3 EPRO §4.4, Phase O0 T-O0.7~T-O0.8).

Builds a sparse, endpoint-swap-invariant propagation kernel ``K = K(z_bar, c)``
with controlled spectral radius ``rho(K) <= rho_max < 1``, and solves the linear
response

    h_lin = (I - K)^{-1} b

with an auditable relative residual

    ||h - b - K h|| / (||b|| + eps) < eps_solver.

Invariants enforced *by construction* (§4.4, §5.3):

  * **Swap invariance**: ``z_bar = Sym(z_w, z_m)`` is symmetric, and edges are
    the union of WT/mutant contacts (swap-invariant), so ``K`` is swap-invariant.
  * **Sparsity**: ``K`` is only non-zero on the provided edge set.
  * **Stability**: ``K`` is rescaled so ``rho(K) <= rho_max < 1``.
  * **Solver residual**: the solved response satisfies the residual bound.

Numpy-only (runs in ``editflow311`` without torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SUSCEPTIBILITY_SCHEMA_VERSION = "reactflow-delta-o0-susceptibility-v1"


# ---------------------------------------------------------------------------
# Symmetric background (endpoint-swap invariant)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmetricBackground:
    """``z_bar = Sym(z_w, z_m)`` (§4.4), swap-invariant by construction.

    Features: ``z_sum = z_w + z_m``, ``z_prod = z_w * z_m``, ``z_abs_diff =
    |z_m - z_w|``. All three are invariant under ``z_w <-> z_m``.
    """

    z_sum: np.ndarray  # (n,)
    z_prod: np.ndarray  # (n,)
    z_abs_diff: np.ndarray  # (n,)
    n: int

    def stack(self) -> np.ndarray:
        """Return (3, n) stacked feature matrix."""

        return np.stack([self.z_sum, self.z_prod, self.z_abs_diff], axis=0)


def build_symmetric_background(z_w: np.ndarray, z_m: np.ndarray) -> SymmetricBackground:
    """Build the swap-invariant symmetric background (§4.4)."""

    z_w = np.asarray(z_w, dtype=float)
    z_m = np.asarray(z_m, dtype=float)
    if z_w.shape != z_m.shape:
        raise ValueError("z_w and z_m must have the same shape")
    return SymmetricBackground(
        z_sum=z_w + z_m,
        z_prod=z_w * z_m,
        z_abs_diff=np.abs(z_m - z_w),
        n=z_w.shape[0],
    )


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelResult:
    """Output of :func:`build_kernel`."""

    K: np.ndarray  # (n, n) sparse propagation kernel
    edges: np.ndarray  # (2, n_edges) edge index pairs
    rho_raw: float  # spectral radius before rescaling
    rho: float  # spectral radius after rescaling
    rho_max: float  # target upper bound
    rescale_factor: float  # factor applied to enforce rho <= rho_max
    symmetric: bool  # whether K is symmetric
    schema_version: str = SUSCEPTIBILITY_SCHEMA_VERSION

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_edges": int(self.edges.shape[1]) if self.edges.size else 0,
            "rho_raw": float(self.rho_raw),
            "rho": float(self.rho),
            "rho_max": float(self.rho_max),
            "rescale_factor": float(self.rescale_factor),
            "symmetric": bool(self.symmetric),
            "rho_pass": float(self.rho) <= float(self.rho_max) + 1e-12,
            "nnz": int(np.count_nonzero(self.K)),
        }


def _spectral_radius(K: np.ndarray) -> float:
    """Spectral radius ``rho(K) = max |lambda_i|``.

    For non-symmetric matrices uses ``np.linalg.eigvals``; for symmetric
    matrices uses ``np.linalg.eigvalsh`` (faster, real output).
    """

    if K.shape[0] == 0:
        return 0.0
    # Detect symmetry exactly (within FP epsilon).
    if np.allclose(K, K.T, atol=1e-12):
        ev = np.linalg.eigvalsh(K)
    else:
        ev = np.linalg.eigvals(K)
    return float(np.max(np.abs(ev)))


def build_kernel(
    z_bar: SymmetricBackground,
    edges: list[tuple[int, int]] | np.ndarray,
    *,
    rho_max: float = 0.95,
    edge_weight_fn: Any | None = None,
) -> KernelResult:
    """Build the sparse, swap-invariant, stable propagation kernel ``K``.

    The kernel is non-zero only on the provided ``edges`` (sequence-adjacent
    edges + WT/mutant sparse contact union, §4.4). Edge weights are derived
    from symmetric background features (swap-invariant). The kernel is then
    rescaled so ``rho(K) <= rho_max < 1``.

    Parameters
    ----------
    z_bar : SymmetricBackground
        Swap-invariant endpoint features.
    edges : list of (i, j) or (2, n_edges) array
        Sparse edge set. Self-loops are ignored.
    rho_max : float
        Target upper bound on the spectral radius, ``0 <= rho_max < 1``.
    """

    if not (0.0 < rho_max < 1.0):
        raise ValueError(f"rho_max must be in (0, 1), got {rho_max}")

    n = z_bar.n
    # Normalize edges to a (2, n_edges) int array.
    if isinstance(edges, list):
        if not edges:
            edge_arr = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_arr = np.asarray(edges, dtype=np.int64).T
    else:
        edge_arr = np.asarray(edges, dtype=np.int64)
        if edge_arr.ndim == 2 and edge_arr.shape[0] != 2 and edge_arr.shape[1] == 2:
            edge_arr = edge_arr.T

    # Build raw symmetric kernel from symmetric features.
    K = np.zeros((n, n), dtype=float)
    # Symmetric per-node magnitude from the background (swap-invariant).
    node_mag = np.log1p(np.exp(np.clip(0.5 * z_bar.z_sum + 0.25 * z_bar.z_abs_diff, -30.0, 30.0)))

    if edge_arr.size > 0:
        for k in range(edge_arr.shape[1]):
            i, j = int(edge_arr[0, k]), int(edge_arr[1, k])
            if i == j or not (0 <= i < n and 0 <= j < n):
                continue
            # Symmetric weight: geometric mean of endpoint magnitudes.
            w = float(np.sqrt(node_mag[i] * node_mag[j]))
            K[i, j] = w
            K[j, i] = w  # enforce symmetry

    # Rescale to enforce rho(K) <= rho_max.
    rho_raw = _spectral_radius(K)
    if rho_raw > 1e-15:
        rescale = min(1.0, rho_max / rho_raw)
    else:
        rescale = 1.0
    K = K * rescale
    rho = _spectral_radius(K)

    symmetric = bool(np.allclose(K, K.T, atol=1e-12))

    return KernelResult(
        K=K,
        edges=edge_arr,
        rho_raw=rho_raw,
        rho=rho,
        rho_max=float(rho_max),
        rescale_factor=float(rescale),
        symmetric=symmetric,
    )


# ---------------------------------------------------------------------------
# Stable solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverResult:
    """Output of :func:`solve_response`."""

    h: np.ndarray  # (n,) linear response
    residual: np.ndarray  # (n,) b + K h - h
    relative_residual: float  # ||residual|| / (||b|| + eps)
    method: str
    n_iter: int
    eps_solver: float

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "relative_residual": float(self.relative_residual),
            "eps_solver": float(self.eps_solver),
            "pass": float(self.relative_residual) < self.eps_solver,
            "method": self.method,
            "n_iter": int(self.n_iter),
            "h_norm": float(np.linalg.norm(self.h)),
        }


def solve_response(
    b: np.ndarray,
    K: np.ndarray,
    *,
    eps_solver: float = 1e-5,
    max_iter: int = 1000,
    method: str = "direct",
) -> SolverResult:
    """Solve ``(I - K) h = b`` for the linear response (§4.4).

    The ``direct`` method uses ``np.linalg.solve``. A ``neumann`` iterative
    method is also provided for differentiable settings. In both cases the
    relative residual is computed and must satisfy

        ||h - b - K h|| / (||b|| + eps) < eps_solver.

    Parameters
    ----------
    b : (n,) array
        Node forcing vector.
    K : (n, n) array
        Stable propagation kernel with ``rho(K) < 1``.
    eps_solver : float
        Relative residual tolerance.
    max_iter : int
        Iteration cap for the Neumann method.
    method : {"direct", "neumann"}
        Solver to use.
    """

    b = np.asarray(b, dtype=float)
    K = np.asarray(K, dtype=float)
    n = b.shape[0]
    eps = 1e-30

    if method == "direct":
        A = np.eye(n) - K
        h = np.linalg.solve(A, b)
        n_iter = 1
    elif method == "neumann":
        # h = sum_{k=0}^{inf} K^k b  (converges since rho(K) < 1)
        h = np.zeros_like(b)
        term = b.copy()
        n_iter = 0
        for k in range(max_iter):
            h = h + term
            term = K @ term
            n_iter += 1
            if np.linalg.norm(term) < eps_solver * (np.linalg.norm(b) + eps):
                break
    else:
        raise ValueError(f"unknown method {method!r}")

    residual = b + K @ h - h  # = (I - K) h - b should be ~0; here we want h - b - K h
    # The contract writes: ||h - b - K h|| / (||b|| + eps). Note h = b + K h for the
    # fixed point, so h - b - K h = 0 at convergence.
    residual = h - b - K @ h
    rel_res = float(np.linalg.norm(residual) / (np.linalg.norm(b) + eps))

    return SolverResult(
        h=h,
        residual=residual,
        relative_residual=rel_res,
        method=method,
        n_iter=int(n_iter),
        eps_solver=float(eps_solver),
    )


# ---------------------------------------------------------------------------
# Invariant audit
# ---------------------------------------------------------------------------


def check_susceptibility_invariants(
    z_w: np.ndarray,
    z_m: np.ndarray,
    edges: list[tuple[int, int]] | np.ndarray,
    *,
    rho_max: float = 0.95,
    eps_solver: float = 1e-5,
) -> dict[str, Any]:
    """Audit swap-invariance, stability, and solver residual (§5.3)."""

    z_bar = build_symmetric_background(z_w, z_m)
    z_bar_swap = build_symmetric_background(z_m, z_w)

    K = build_kernel(z_bar, edges, rho_max=rho_max)
    K_swap = build_kernel(z_bar_swap, edges, rho_max=rho_max)

    swap_invariance_err = float(np.max(np.abs(K.K - K_swap.K))) if K.K.size else 0.0

    # Solver on a small forcing vector.
    b = np.zeros(z_w.shape[0], dtype=float)
    b[0] = 1.0
    sol = solve_response(b, K.K, eps_solver=eps_solver)

    return {
        "swap_invariance": {
            "max_abs_err": swap_invariance_err,
            "pass": swap_invariance_err < 1e-12,
        },
        "stability": {
            "rho": K.rho,
            "rho_max": K.rho_max,
            "pass": K.rho <= K.rho_max + 1e-12 and K.rho_max < 1.0,
        },
        "sparsity": {
            "nnz": int(np.count_nonzero(K.K)),
            "symmetric": K.symmetric,
            "pass": K.symmetric,
        },
        "solver": {
            "relative_residual": sol.relative_residual,
            "eps_solver": eps_solver,
            "pass": sol.relative_residual < eps_solver,
        },
        "kernel_audit": K.to_audit_dict(),
        "solver_audit": sol.to_audit_dict(),
    }
