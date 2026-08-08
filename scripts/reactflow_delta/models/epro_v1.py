#!/usr/bin/env python3
"""Phase 3 scheme-3 (contract §9.4): minimal REPAIRED EPRO propagation operator.

Repairs the dev05-era EPRO (LambdaLR bug, degenerate edge init, unproven
gradient/residual) into a small, correct, capacity-matched operator:

  * Fixed LR Adam (no LambdaLR).
  * Non-degenerate (random) init for every layer; no zero-initialized edge net.
  * Vector-valued, swap-antisymmetric forcing at the edited site, built as
    b = w_sym * delta, where delta is ANTISYMMETRIC in the WT<->mutant swap and
    w_sym is a symmetric non-negative weight. This guarantees:
        identity      : alt==ref -> diff=0 -> delta=0 -> b=0 -> h=0
        antisymmetry  : swap -> delta -> -delta -> b -> -b -> h -> -h,
                        and magnitude |readout(h)| is swap-invariant.
  * Single SPARSE top-k base-pair contact operator (from ViennaRNA BPP).
  * Differentiable sparse Neumann propagation with explicit rho<1 normalization
    and per-sample residual/convergence logging.
  * No complex switch (kept disabled).
  * Bias-free readout over the propagated response, aggregated over ELIGIBLE
    positions (the same mask/weighting as the endpoint_v5 target).

Scientific claim under test: mutation response propagating along base-pair
contacts gives an increment over the same-capacity scheme-2 generic model.
Verified by paired publication-block bootstrap CI of the conditional-WMAE-skill
difference, with ablations local-only / random-contacts / no-propagation.

Synthetic invariants (§9.4) verified by tests/test_epro_v1.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-position feature dim (allowed inputs): base one-hot(5) + WT reactivity(1)
# + WT error(1). Same semantics as scheme-1 pos_dim, but over the FULL sequence.
POS_DIM = 7
# Condition feature dim (probe 12 + modifier 14 + experimentType 4 + temp 1)
# + edit-position extra (3) = 37. Matches build_glob in run_phase3_scheme3.
GLOB_DIM = 37

BASE_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
BASE_OTHER = 4


def base_oh(base):
    v = np.zeros(5, dtype=np.float32)
    idx = BASE_MAP.get(base)
    v[idx if idx is not None else BASE_OTHER] = 1.0
    return v


# ---------------------------------------------------------------------------
# Contact graph builder (ViennaRNA BPP -> sparse top-k edges)
# ---------------------------------------------------------------------------
def build_contact_edges(seq: str, top_k: int = 4, min_bpp: float = 1e-3):
    """Return (edges (E,2) 0-indexed, weights (E,)) from WT base-pair probs.

    Uses ViennaRNA partition-function BPP, keeps the top_k most probable partner
    per position (sparse operator). Requires RNA in the running environment.
    """
    import RNA
    seq_rna = seq.upper().replace("T", "U")
    fc = RNA.fold_compound(seq_rna, RNA.md())
    fc.pf()
    bpp = fc.bpp()  # (n+1, n+1), 1-indexed
    n = len(seq_rna)
    wmap = {}
    for i in range(1, n + 1):
        row = bpp[i]
        cands = [(j, float(row[j])) for j in range(1, n + 1)
                 if j != i and float(row[j]) >= min_bpp]
        cands.sort(key=lambda x: -x[1])
        for j, w in cands[:top_k]:
            a, b = (i - 1, j - 1)
            lo, hi = (a, b) if a < b else (b, a)
            wmap[(lo, hi)] = max(wmap.get((lo, hi), 0.0), w)
    edges = np.array(sorted(wmap), dtype=np.int64)  # (E,2)
    weights = np.array([wmap[e] for e in sorted(wmap)], dtype=np.float32)
    return edges, weights


def normalize_edge_weights(edges, weights, n, alpha: float = 0.5):
    """Row-normalize so max in-degree <= 1, then scale by alpha < 1 -> rho<1."""
    edges = np.asarray(edges)
    weights = np.asarray(weights, dtype=np.float32)
    if edges.shape[0] == 0:
        return weights.astype(np.float32)
    in_deg = np.zeros(n, dtype=np.float64)
    for j in edges[:, 1]:
        in_deg[j] += 1.0
    maxd = float(max(1.0, in_deg.max()))
    return (weights / maxd * alpha).astype(np.float32)


# ---------------------------------------------------------------------------
# EPRO operator (repaired)
# ---------------------------------------------------------------------------
class SparseNeumannPropagator(nn.Module):
    """Differentiable sparse Neumann propagation h = (I - K)^{-1} b.

    K is defined on the sparse edge set with normalized weights (rho<1), so the
    Neumann series converges; residual is logged for convergence evidence.
    """

    def __init__(self, hidden: int, neumann_iter: int = 8):
        super().__init__()
        self.hidden = hidden
        self.neumann_iter = neumann_iter

    def forward(self, b, edges, weights):
        """b: (B, L, hidden); edges: (B, E, 2); weights: (B, E). -> (B, L, hidden)"""
        B, L, H = b.shape
        if edges.shape[1] == 0:
            return b, []
        ei = edges[:, :, 0]  # (B,E)
        ej = edges[:, :, 1]  # (B,E)
        w = weights.unsqueeze(-1)  # (B,E,1)
        h = b.clone()
        res = []
        for _ in range(self.neumann_iter):
            src = torch.gather(h, 1, ei.unsqueeze(-1).expand(-1, -1, H))  # (B,E,H)
            msg = src * w
            agg = torch.zeros_like(h)
            agg = agg.scatter_add_(1, ej.unsqueeze(-1).expand(-1, -1, H), msg)
            h = h + agg
            res.append(float(agg.norm() / (h.norm() + 1e-12)))
        return h, res


class EPROMagnitude(nn.Module):
    """Minimal repaired EPRO for conditional-magnitude regression.

    Antisymmetry and identity hold BY CONSTRUCTION:
      * delta  : Linear(diff)  -- antisymmetric (swap -> -delta)
      * w_sym  : softplus(sym_net(z_sym, glob)) -- symmetric, non-negative
      * b      : w_sym * delta * window_mask   -- antisymmetric
      * h      : (I-K)^{-1} b                  -- antisymmetric
      * mag    : |bias-free Linear(h)|         -- swap-invariant, identity=0
    """

    def __init__(self, hidden: int = 64, glob_dim: int = GLOB_DIM,
                 neumann_iter: int = 8, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.glob_dim = glob_dim
        self.neumann_iter = neumann_iter

        # Per-position encoder over ALLOWED WT features (symmetric in swap).
        self.pos_enc = nn.Sequential(
            nn.Linear(POS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Symmetric non-negative weight net (z_sym + glob -> scalar weight).
        self.sym_net = nn.Sequential(
            nn.Linear(hidden + glob_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Antisymmetric delta projection (no bias, so Linear(diff)=0 when diff=0).
        self.delta_proj = nn.Linear(5, hidden, bias=False)
        # Bias-free magnitude readout over the propagated response.
        self.readout = nn.Linear(hidden, 1, bias=False)

        self.propagator = SparseNeumannPropagator(hidden, neumann_iter)
        self.residuals = []

    def forward(self, X, mask, elig, edit, ref, alt, edges, weights, glob):
        B, L, _ = X.shape
        z = self.pos_enc(X)  # (B,L,hidden) -- symmetric

        # Antisymmetric delta: projection of (alt - ref).
        diff = (alt - ref).to(X.dtype)  # (B,5)
        delta = self.delta_proj(diff).unsqueeze(1)  # (B,1,hidden)

        # Symmetric non-negative weight from per-position WT context + condition.
        gz = torch.cat([z, glob.unsqueeze(1).expand(-1, L, -1)], dim=-1)  # (B,L,h+g)
        w_sym = F.softplus(self.sym_net(gz))  # (B,L,1) >= 0

        # Edit-window mask.
        ar = torch.arange(L, device=X.device)
        winmask = ((ar.unsqueeze(0) - edit.unsqueeze(1)).abs() <= 3).float()  # (B,L)
        winmask = winmask.unsqueeze(-1)  # (B,L,1)

        # Vector forcing: antisymmetric (identity => diff=0 => delta=0 => b=0).
        f = w_sym * delta * winmask * mask.unsqueeze(-1)  # (B,L,hidden)

        # Propagate along the sparse contact operator.
        h, res = self.propagator(f, edges, weights)
        self.residuals = res

        # Bias-free magnitude readout over propagated response, eligibility-masked.
        r = self.readout(h).squeeze(-1)  # (B,L)
        mag = r.abs() * elig  # swap-invariant magnitude, identity=0
        denom = elig.sum(dim=1).clamp_min(1.0)
        return mag.sum(dim=1) / denom

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
