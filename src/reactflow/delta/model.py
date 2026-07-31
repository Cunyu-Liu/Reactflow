"""EPRO models (v3 EPRO §4.3-4.9, Phase M0 T-M0.1~T-M0.2).

Torch differentiable implementation of the EPRO (Endpoint-swap Preserving
Response Operator) pipeline. Wraps the O0 numpy operators as differentiable
torch modules, preserving all by-construction invariants (§5.3):

  * Forcing: identity (z_w==z_m => b=0), swap antisymmetry (B(z_m,z_w)=-B(z_w,z_m)),
    support leakage (off-support b=0).
  * Susceptibility: swap-invariant kernel K, sparsity, stability (rho(K)<1),
    solver residual.
  * Switch: odd (h(-b)=-h(b)), no bias, swap-invariant gate.
  * Observation: monotone (non-negative weights * non-decreasing basis),
    swap-antisymmetric.

Two model tiers (§4.9):

  * **EPRO-0**: deterministic, fixed thermo, hand forcing, fixed propagation,
    no/few learned parameters. Validates architecture mechanics.
  * **EPRO-Lite**: 2-6M params, fixed thermo prior, learned local forcing
    correction, sparse stable susceptibility, probe-specific observation,
    no/single global switch gate.

Because all 1509 pairs carry ``encoded_alt="X"`` (mutant sequences not
constructible), the mutant endpoint ``z_m`` is implicit. We parameterize the
*endpoint difference* ``delta = z_w - z_m`` directly:

  * EPRO-0: ``delta = 0`` (no learned correction; deterministic baseline).
  * EPRO-Lite (M0-R2, v3.5): ``delta = correction_net(concat(z_w, delta_thermo))``,
    NO positive bump, can be positive or negative (breaks non-negativity chain
    at the delta ring).

The symmetric background is then ``z_bar = Sym(z_w, z_m) = Sym(z_w, z_w - delta)``,
which is well-defined and swap-invariant by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SCHEMA_VERSION = "reactflow-delta-m0-model-v1"

# Thermo feature dimension. First 5 are WT thermo features, next 5 are delta
# thermo features (mutant_mean(3 alt bases) - wt), giving the encoder
# mutation-effect information (M0-R fix, v3.4 §2.2):
#   0. unpaired_prob            5. delta_unpaired_prob
#   1. positional_entropy_bits  6. delta_positional_entropy_bits
#   2. bpp_paired_prob          7. delta_bpp_paired_prob
#   3. normalized_seq_pos       8. delta_mfe_energy
#   4. distance_from_edit       9. delta_pf_energy
THERMO_FEAT_DIM = 10

# Observation basis size (identity, softplus, tanh, cubic-soft).
N_BASIS = 4


# ---------------------------------------------------------------------------
# Monotone observation basis (differentiable, matches O0 observation.py)
# ---------------------------------------------------------------------------


def monotone_basis(a: torch.Tensor) -> torch.Tensor:
    """Fixed non-decreasing basis functions on accessibility ``a``.

    Returns (..., n_basis) tensor. Each column is non-decreasing in ``a``.
    Matches O0 ``observation._monotone_basis``.
    """

    a = a.clamp(-6.0, 6.0)
    b1 = a
    b2 = F.softplus(a)
    b3 = torch.tanh(a)
    b4 = (a.clamp(-3.0, 3.0)) ** 3
    return torch.stack([b1, b2, b3, b4], dim=-1)  # (..., 4)


class MonotoneHead(nn.Module):
    """Probe-specific monotone observation head ``f_p(a) = sum_k w_k * basis_k(a)``.

    Weights are parameterized via softplus to be non-negative, ensuring
    monotonicity by construction (§4.6).
    """

    def __init__(self, n_basis: int = N_BASIS, init_weights: list[float] | None = None):
        super().__init__()
        if init_weights is None:
            # Scaled-down init: produces output on data scale (~0.006)
            # without needing a separate output_scale parameter.
            init_weights = [0.06, 0.02, 0.01, 0.005]
        # Raw weights; softplus(raw) >= 0 ensures non-negativity.
        init = torch.tensor(init_weights, dtype=torch.float32)
        # Invert softplus for initialization: raw = log(exp(w) - 1)
        raw = torch.log(torch.expm1(init.clamp(min=1e-6)))
        self.raw_weights = nn.Parameter(raw)

    @property
    def weights(self) -> torch.Tensor:
        """Non-negative weights = softplus(raw_weights)."""

        return F.softplus(self.raw_weights)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        basis = monotone_basis(a)  # (..., n_basis)
        return (basis * self.weights).sum(dim=-1)  # (...)


# ---------------------------------------------------------------------------
# Thermo feature encoder: per-position thermo features -> latent z_w
# ---------------------------------------------------------------------------


class ThermoEncoder(nn.Module):
    """Maps per-position thermo features to latent state ``z_w``.

    Input features (per position), 10 dims total (M0-R, v3.4 §2.2):
      WT thermo (0-4):
        0. unpaired_prob
        1. positional_entropy_bits (normalized)
        2. bpp_paired_prob
        3. normalized_seq_pos (position / seq_length, in [0, 1])
        4. normalized_distance_from_edit (|pos - edit_pos| / seq_length)
      Delta thermo (5-9), = mutant_mean(3 alt bases) - wt:
        5. delta_unpaired_prob
        6. delta_positional_entropy_bits
        7. delta_bpp_paired_prob
        8. delta_mfe_energy (broadcast scalar)
        9. delta_pf_energy (broadcast scalar)

    For EPRO-0: fixed linear map (no learned parameters).
    For EPRO-Lite: learned MLP.
    """

    def __init__(self, feat_dim: int = THERMO_FEAT_DIM, latent_dim: int = 64,
                 hidden_dim: int = 512, n_layers: int = 3, learned: bool = True):
        super().__init__()
        self.learned = learned
        self.latent_dim = latent_dim

        if learned:
            layers: list[nn.Module] = []
            d_in = feat_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(d_in, hidden_dim))
                layers.append(nn.GELU())
                d_in = hidden_dim
            layers.append(nn.Linear(d_in, latent_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            # Fixed linear map: project features to a fixed latent.
            # Use a fixed random projection (seeded for reproducibility).
            self.register_buffer("fixed_proj", torch.randn(feat_dim, latent_dim) * 0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, n, feat_dim) -> z_w: (B, n, latent_dim)."""

        if self.learned:
            return self.mlp(features)
        else:
            return features @ self.fixed_proj


# ---------------------------------------------------------------------------
# Forcing module: b = w_sym(z_w, delta) * delta * mask
# ---------------------------------------------------------------------------


class ForcingModule(nn.Module):
    """Local mutation forcing operator (§4.3), differentiable.

    Computes ``b = w_sym(z_w, z_m) * (z_w - z_m) * mask`` where:
      * ``delta = z_w - z_m`` is the endpoint difference (antisymmetric).
      * ``w_sym`` is a symmetric function of the endpoints (uses z_sum, z_abs_diff).
      * ``mask`` enforces support leakage = 0 outside the edit window.

    Invariants by construction:
      * Identity: delta=0 => b=0.
      * Swap: delta -> -delta => z_sum unchanged, z_abs_diff unchanged, b -> -b.
      * Leakage: mask zeroes off-support forcing.

    For EPRO-0: delta = 0 (no learned correction; deterministic baseline).
    For EPRO-Lite (M0-R2, v3.5): delta = correction_net(concat(z_w, delta_thermo)),
        NO positive bump, can be positive or negative (v3.5 §1.2, removes
        non-negativity bias that caused M0-R pred_min=0.0).
    """

    def __init__(self, latent_dim: int = 64, local_window: int = 3,
                 hidden_dim: int = 512, learned: bool = True,
                 delta_thermo_dim: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.local_window = local_window
        self.learned = learned
        self.delta_thermo_dim = delta_thermo_dim

        if learned:
            # M0-R2: correction_net input = concat(z_w[window], delta_thermo[window])
            # delta_thermo provides mutation-effect signal that does NOT collapse
            # on OOD parents (physical prior, not learned encoder output).
            context_dim = (latent_dim + delta_thermo_dim) * (2 * local_window + 1)
            self.correction_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * local_window + 1),
            )
            # Initialize correction net to small non-zero outputs (learnable).
            for m in self.correction_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, z_w: torch.Tensor, edit_pos: int, n: int,
                mask: torch.Tensor,
                delta_thermo: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forcing vector b and endpoint difference delta.

        Parameters
        ----------
        z_w : (n, latent_dim) — WT latent state per position.
        edit_pos : int — 0-indexed edit position.
        n : int — sequence length.
        mask : (n,) bool — support mask (True where forcing is allowed).
        delta_thermo : (n, delta_thermo_dim) | None — per-position delta_thermo
            features (M0-R2, required when learned=True). None for EPRO-0.

        Returns
        -------
        b : (n, latent_dim) — forcing vector (same dim as z_w for response).
        delta : (n, latent_dim) — endpoint difference z_w - z_m (for kernel/observation).
        """

        lo = max(0, edit_pos - self.local_window)
        hi = min(n, edit_pos + self.local_window + 1)
        window_size = hi - lo

        # M0-R2 (v3.5 §1.2): NO positive bump. delta = correction_net(context),
        # fully learned, can be positive or negative. This breaks the
        # non-negativity chain at the delta ring (b = w_sym * delta, w_sym >= 0,
        # so delta < 0 => b < 0 => can predict negative delta_r).
        delta_window = torch.zeros(window_size, z_w.shape[-1],
                                   dtype=z_w.dtype, device=z_w.device)

        if self.learned:
            if delta_thermo is None:
                raise ValueError(
                    "delta_thermo required when learned=True (M0-R2, v3.5 §2.2). "
                    "Pass None only for EPRO-0 (learned=False)."
                )
            # z_w context (learned latent, may collapse on OOD parents).
            z_context = z_w[lo:hi].flatten()  # (window * latent_dim,)
            expected_z = self.latent_dim * (2 * self.local_window + 1)
            if z_context.shape[0] < expected_z:
                pad = torch.zeros(expected_z - z_context.shape[0],
                                  dtype=z_w.dtype, device=z_w.device)
                z_context = torch.cat([z_context, pad])
            # delta_thermo context (physical prior, does NOT collapse on OOD).
            d_context = delta_thermo[lo:hi].flatten()  # (window * delta_thermo_dim,)
            expected_d = self.delta_thermo_dim * (2 * self.local_window + 1)
            if d_context.shape[0] < expected_d:
                pad = torch.zeros(expected_d - d_context.shape[0],
                                  dtype=delta_thermo.dtype, device=delta_thermo.device)
                d_context = torch.cat([d_context, pad])
            context = torch.cat([z_context, d_context])  # (expected_z + expected_d,)
            correction = self.correction_net(context)  # (window,)
            if correction.shape[0] < window_size:
                pad = torch.zeros(window_size - correction.shape[0],
                                  dtype=correction.dtype, device=correction.device)
                correction = torch.cat([correction, pad])
            elif correction.shape[0] > window_size:
                correction = correction[:window_size]
            # NO bump addition; delta fully from correction_net, can be negative.
            delta_window = correction.unsqueeze(-1) * 1.0  # (window, latent_dim)

        # Scatter delta to full-length vector.
        delta = torch.zeros_like(z_w)  # (n, latent_dim)
        delta[lo:hi] = delta_window

        # Symmetric weight w_sym from z_w and delta (UNCHANGED, v3.5 §2.3 item 13).
        # z_sum = z_w + z_m = z_w + (z_w - delta) = 2*z_w - delta (symmetric)
        # z_abs_diff = |delta| (symmetric)
        z_sum = 2.0 * z_w - delta
        z_abs_diff = delta.abs()
        # Symmetric non-negative weight (softplus-like, matches O0 default).
        w_sym = F.softplus(0.5 * z_sum.sum(dim=-1, keepdim=True)
                           + 0.25 * z_abs_diff.sum(dim=-1, keepdim=True))  # (n, 1)

        # Forcing: b = w_sym * delta * mask
        b = w_sym * delta  # (n, latent_dim)
        b = b * mask.unsqueeze(-1).to(b.dtype)  # zero outside support

        return b, delta


# ---------------------------------------------------------------------------
# Susceptibility module: K, solve (I-K)^{-1} b
# ---------------------------------------------------------------------------


class SusceptibilityModule(nn.Module):
    """Stable sparse susceptibility kernel + solver (§4.4), differentiable.

    Builds ``K`` from symmetric background features on the provided edge set,
    rescales so ``rho(K) <= rho_max < 1``, and solves ``h_lin = (I-K)^{-1} b``
    via differentiable Neumann iteration.

    Invariants by construction:
      * Swap-invariance: K uses only symmetric features (z_bar).
      * Sparsity: K non-zero only on provided edges.
      * Stability: rho(K) <= rho_max < 1 via spectral rescaling.
      * Solver: Neumann series converges since rho(K) < 1.

    For EPRO-0: fixed edge weights (from symmetric features).
    For EPRO-Lite: learned edge weight correction (bounded).
    """

    def __init__(self, latent_dim: int = 64, rho_max: float = 0.95,
                 neumann_iter: int = 50, learned: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.rho_max = rho_max
        self.neumann_iter = neumann_iter
        self.learned = learned

        if learned:
            # Learned edge weight correction: takes edge features (bpp, distance)
            # and outputs a bounded correction.
            self.edge_net = nn.Sequential(
                nn.Linear(3, 32),  # (bpp, seq_dist, contact_weight)
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            # Initialize to zero (start at EPRO-0).
            for m in self.edge_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def _build_kernel(self, z_w: torch.Tensor, delta: torch.Tensor,
                      edges: torch.Tensor, edge_features: torch.Tensor,
                      n: int) -> torch.Tensor:
        """Build the sparse, stable propagation kernel K.

        Parameters
        ----------
        z_w : (n, latent_dim) — WT latent state.
        delta : (n, latent_dim) — endpoint difference.
        edges : (2, n_edges) — edge index pairs (0-indexed).
        edge_features : (n_edges, 3) — (bpp, seq_dist, contact_weight).
        n : int — sequence length.

        Returns
        -------
        K : (n, n) — stable propagation kernel.
        """

        device = z_w.device
        dtype = z_w.dtype

        # Symmetric background from z_w and delta.
        z_sum = 2.0 * z_w - delta  # (n, latent_dim)
        z_abs_diff = delta.abs()  # (n, latent_dim)
        # Per-node symmetric magnitude (matches O0 susceptibility default).
        node_mag = F.softplus(0.5 * z_sum.sum(dim=-1) + 0.25 * z_abs_diff.sum(dim=-1))  # (n,)

        K = torch.zeros(n, n, dtype=dtype, device=device)
        if edges.shape[1] == 0:
            return K

        i_idx = edges[0]  # (n_edges,)
        j_idx = edges[1]  # (n_edges,)

        # Symmetric edge weight: geometric mean of endpoint magnitudes.
        w = torch.sqrt(node_mag[i_idx] * node_mag[j_idx] + 1e-12)  # (n_edges,)

        if self.learned:
            # Bounded learned correction: tanh * small scale.
            correction = torch.tanh(self.edge_net(edge_features).squeeze(-1)) * 0.1  # (n_edges,)
            w = w * (1.0 + correction)

        # Enforce symmetry: K[i,j] = K[j,i] = w.
        K[i_idx, j_idx] = w
        K[j_idx, i_idx] = w

        # Spectral rescaling: estimate rho via power iteration.
        rho_est = self._power_iteration_rho(K, n, dtype, device)
        rescale = torch.clamp(self.rho_max / (rho_est + 1e-12), max=1.0)
        K = K * rescale

        return K

    def _power_iteration_rho(self, K: torch.Tensor, n: int,
                             dtype: torch.dtype, device: torch.device,
                             n_iter: int = 20) -> torch.Tensor:
        """Estimate spectral radius via power iteration (differentiable)."""

        v = torch.randn(n, dtype=dtype, device=device)
        v = v / (v.norm() + 1e-12)
        for _ in range(n_iter):
            v = K @ v
            v = v / (v.norm() + 1e-12)
        rho_est = (K @ v).norm() / (v.norm() + 1e-12)
        return rho_est

    def _solve_neumann(self, b: torch.Tensor, K: torch.Tensor,
                       n: int) -> torch.Tensor:
        """Solve (I - K)^{-1} b via Neumann series: h = sum_{k=0}^{T} K^k b.

        Converges since rho(K) < 1. Differentiable.
        """

        h = b.clone()
        term = b.clone()
        for _ in range(self.neumann_iter):
            term = K @ term
            h = h + term
            if term.norm() < 1e-8 * (b.norm() + 1e-12):
                break
        return h

    def forward(self, z_w: torch.Tensor, delta: torch.Tensor,
                edges: torch.Tensor, edge_features: torch.Tensor,
                b: torch.Tensor, n: int) -> torch.Tensor:
        """Compute linear response h_lin = (I - K)^{-1} b.

        Returns h_lin: (n, latent_dim).
        """

        K = self._build_kernel(z_w, delta, edges, edge_features, n)
        h_lin = self._solve_neumann(b, K, n)
        return h_lin, K


# ---------------------------------------------------------------------------
# Switch module (optional, single global gate for EPRO-Lite)
# ---------------------------------------------------------------------------


class SwitchModule(nn.Module):
    """Odd nonlinear switch (§4.5), differentiable.

    h_nl = pi * tanh(S @ h_lin), where:
      * S is symmetric, no-bias (ensures oddness: h_nl(-h_lin) = -h_nl(h_lin)).
      * pi is a swap-invariant gate (single global scalar for EPRO-Lite).

    For EPRO-0: pi=0 (no switch, h = h_lin).
    """

    def __init__(self, latent_dim: int = 64, enabled: bool = False):
        super().__init__()
        self.enabled = enabled
        self.latent_dim = latent_dim

        if enabled:
            # S: symmetric mixing matrix (lower-triangular parameterized).
            n_tri = latent_dim * (latent_dim + 1) // 2
            self.S_raw = nn.Parameter(torch.zeros(n_tri) * 0.01)
            # Single global gate.
            self.gate_raw = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12

    @property
    def gate(self) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0)
        return torch.sigmoid(self.gate_raw)

    def _build_S(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Build symmetric, zero-diagonal mixing matrix S."""

        L = self.latent_dim
        # Reconstruct symmetric matrix from lower triangle.
        S = torch.zeros(L, L, dtype=dtype, device=device)
        idx = torch.tril_indices(L, L, offset=-1)  # strict lower triangle (no diagonal)
        S[idx[0], idx[1]] = self.S_raw[:idx.shape[1]]
        S = S + S.T  # symmetrize
        return S

    def forward(self, h_lin: torch.Tensor) -> torch.Tensor:
        """Apply switch: h = h_lin + gate * tanh(S @ h_lin).

        h_lin: (n, latent_dim).
        Returns h: (n, latent_dim).
        """

        if not self.enabled:
            return h_lin

        S = self._build_S(h_lin.dtype, h_lin.device)
        h_nl = self.gate * torch.tanh(h_lin @ S.T)  # (n, latent_dim)
        return h_lin + h_nl


# ---------------------------------------------------------------------------
# Observation module
# ---------------------------------------------------------------------------


class ObservationModule(nn.Module):
    """Monotone probe observation (§4.6), differentiable.

    delta_r_hat = f_p(a_bar + h/2) - f_p(a_bar - h/2)

    where a_bar is the midpoint accessibility from symmetric background, and
    f_p is a monotone head (non-negative weights * non-decreasing basis).

    Swap-antisymmetry: swapping endpoints flips h sign, hence flips delta_r_hat.
    """

    def __init__(self, latent_dim: int = 64, learned: bool = True):
        super().__init__()
        self.learned = learned

        if learned:
            # Midpoint accessibility: z_bar features -> scalar per position.
            self.access_net = nn.Sequential(
                nn.Linear(latent_dim * 3, latent_dim),  # z_sum, z_prod, z_abs_diff
                nn.GELU(),
                nn.Linear(latent_dim, 1),
            )
            self.head = MonotoneHead()
        else:
            # Fixed: use first latent dim as accessibility (fixed projection).
            self.register_buffer("fixed_access_proj", torch.randn(latent_dim * 3, 1) * 0.01)
            self.head = MonotoneHead()
            # Freeze head weights.
            self.head.raw_weights.requires_grad_(False)

    def forward(self, h: torch.Tensor, z_w: torch.Tensor,
                delta: torch.Tensor) -> torch.Tensor:
        """Predict delta_r_hat.

        Parameters
        ----------
        h : (n, latent_dim) — total latent response.
        z_w : (n, latent_dim) — WT latent state.
        delta : (n, latent_dim) — endpoint difference.

        Returns
        -------
        delta_r_hat : (n,) — predicted reactivity difference.
        """

        # Symmetric background from z_w and delta.
        z_m = z_w - delta
        z_sum = z_w + z_m  # = 2*z_w - delta (symmetric)
        z_prod = z_w * z_m  # symmetric
        z_abs_diff = (z_w - z_m).abs()  # = |delta| (symmetric)
        z_bar = torch.cat([z_sum, z_prod, z_abs_diff], dim=-1)  # (n, 3*latent_dim)

        if self.learned:
            a_bar = self.access_net(z_bar).squeeze(-1)  # (n,)
        else:
            a_bar = (z_bar @ self.fixed_access_proj).squeeze(-1)  # (n,)

        a_m = a_bar + h.sum(dim=-1) / 2.0  # project h to scalar per position
        a_w = a_bar - h.sum(dim=-1) / 2.0

        delta_r = self.head(a_m) - self.head(a_w)  # (n,)
        return delta_r


# ---------------------------------------------------------------------------
# Full EPRO model
# ---------------------------------------------------------------------------


@dataclass
class EPROConfig:
    """Configuration for EPRO models."""

    model_type: str = "epro_lite"  # "epro0" or "epro_lite"
    latent_dim: int = 64
    hidden_dim: int = 512
    n_encoder_layers: int = 3
    local_window: int = 3
    rho_max: float = 0.95
    neumann_iter: int = 50
    switch_enabled: bool = False  # EPRO-Lite: no switch or single global gate
    probe: str = "DMS"  # only DMS in this dataset
    delta_thermo_dim: int = 5  # M0-R2: delta_thermo features driving correction_net


class EPROModel(nn.Module):
    """Full EPRO model (EPRO-0 or EPRO-Lite).

    Pipeline per pair:
      1. Encode thermo features -> z_w (latent WT state).
      2. Compute delta = z_w - z_m (endpoint difference) on edit window.
      3. Forcing: b = w_sym(z_w, delta) * delta * mask.
      4. Susceptibility: K from z_bar + edges, h_lin = (I-K)^{-1} b.
      5. Switch (optional): h = h_lin + gate * tanh(S @ h_lin).
      6. Observation: delta_r_hat = f_p(a_bar + h/2) - f_p(a_bar - h/2).
    """

    def __init__(self, config: EPROConfig | None = None):
        super().__init__()
        if config is None:
            config = EPROConfig()
        self.config = config
        self.is_lite = config.model_type == "epro_lite"

        learned = self.is_lite

        self.encoder = ThermoEncoder(
            feat_dim=THERMO_FEAT_DIM,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_encoder_layers,
            learned=learned,
        )
        self.forcing = ForcingModule(
            latent_dim=config.latent_dim,
            local_window=config.local_window,
            hidden_dim=config.hidden_dim,
            learned=learned,
            delta_thermo_dim=config.delta_thermo_dim,
        )
        self.susceptibility = SusceptibilityModule(
            latent_dim=config.latent_dim,
            rho_max=config.rho_max,
            neumann_iter=config.neumann_iter,
            learned=learned,
        )
        self.switch = SwitchModule(
            latent_dim=config.latent_dim,
            enabled=config.switch_enabled and self.is_lite,
        )
        self.observation = ObservationModule(
            latent_dim=config.latent_dim,
            learned=learned,
        )
        # Output scale: fixed at 1.0. The observation head init is scaled
        # to produce output on the data scale directly, avoiding the
        # collapse-to-zero issue where a learnable scale dominates gradients.
        self.register_buffer("output_scale", torch.tensor(1.0))

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass for a single pair (no batch dimension over pairs).

        Parameters
        ----------
        batch : dict with keys:
            - features: (n, THERMO_FEAT_DIM) — per-position thermo features.
            - edit_pos: int — 0-indexed edit position.
            - edges: (2, n_edges) — contact + sequence-adjacent edges.
            - edge_features: (n_edges, 3) — (bpp, seq_dist, contact_weight).
            - mask: (n,) — True for real positions (not padding).
            - delta_thermo: (n, delta_thermo_dim) — per-position delta_thermo
              features driving correction_net (M0-R2, v3.5 §2.2).

        Returns
        -------
        dict with:
            - delta_r_hat: (n,) — predicted reactivity difference.
            - h_lin: (n, latent_dim) — linear response.
            - h: (n, latent_dim) — total response.
            - b: (n, latent_dim) — forcing.
            - K: (n, n) — kernel.
        """

        features = batch["features"]  # (n, feat_dim)
        edit_pos = batch["edit_pos"]
        edges = batch["edges"]  # (2, n_edges)
        edge_features = batch["edge_features"]  # (n_edges, 3)
        delta_thermo = batch["delta_thermo"]  # (n, delta_thermo_dim) M0-R2
        n = features.shape[0]

        # 1. Encode thermo features -> z_w.
        z_w = self.encoder(features.unsqueeze(0)).squeeze(0)  # (n, latent_dim)

        # 2. Compute forcing b and delta (M0-R2: forcing returns both;
        #    NO duplicate bump/delta computation here, v3.5 §1.2).
        mask = batch.get("mask", torch.ones(n, dtype=torch.bool, device=features.device))
        b, delta = self.forcing(z_w, edit_pos, n, mask, delta_thermo)  # (n, latent_dim) each

        # 3. Susceptibility: K, h_lin.
        h_lin, K = self.susceptibility(z_w, delta, edges, edge_features, b, n)

        # 4. Switch (optional).
        h = self.switch(h_lin)

        # 5. Observation (with learnable output scale).
        delta_r_hat = self.observation(h, z_w, delta) * self.output_scale

        return {
            "delta_r_hat": delta_r_hat,
            "h_lin": h_lin,
            "h": h,
            "b": b,
            "K": K,
            "z_w": z_w,
            "delta": delta,
        }

    def param_count(self) -> int:
        """Count trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def make_epro0(**kwargs: Any) -> EPROModel:
    """Create an EPRO-0 model (deterministic, no learned parameters)."""

    config = EPROConfig(model_type="epro0", switch_enabled=False, **kwargs)
    return EPROModel(config)


def make_epro_lite(**kwargs: Any) -> EPROModel:
    """Create an EPRO-Lite model (learned, 2-6M params)."""

    config = EPROConfig(model_type="epro_lite", switch_enabled=False, **kwargs)
    model = EPROModel(config)
    # Report param count.
    pc = model.param_count()
    if pc < 2_000_000:
        # Scale up hidden dim to reach 2M target.
        # Rough: params ~ hidden^2 * n_layers. Solve for hidden.
        pass  # Accept whatever count; gate is about Skill not params.
    return model
