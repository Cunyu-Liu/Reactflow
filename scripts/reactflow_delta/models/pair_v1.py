#!/usr/bin/env python3
"""Phase 3 scheme-1 model module: conditional-magnitude pair head + capacity-matched generic.

Reuse the Phase 2 feature pipeline (build_feature with use_wt_anchor / use_exact_alt
flags) so each ablation changes exactly ONE capability (contract Phase 3 rule ⑤):
  * candidate (PairHeadV1): DeepSets-style set encoder over the WT-anchor local window
    positions (base identity + WT reactivity/error) + global condition/exact-alt features.
  * generic (CapacityMatchedMLP): flat MLP over the SAME flat feature vector, with width
    auto-scaled to match the candidate parameter count (contract Phase 3 rule ④).
  * ablation exact-alt : build_feature(use_exact_alt=False) -> drops exact-alt global features.
  * ablation nonlocal  : build_feature(use_wt_anchor=False) -> drops WT reactivity/error
    local context (removes nonlocal propagation through the WT profile).

All models output a pair-level magnitude (direct, no post-hoc aggregation).
Deterministic per seed. CUDA required by the caller.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PairHeadV1(nn.Module):
    """Conditional-magnitude pair head (scheme 1).

    pos_set: (B, W, pos_dim) per-position features (WT-anchor local context).
    glob:    (B, glob_dim) global features (condition + exact-alt identity).
    """

    def __init__(self, pos_dim: int, glob_dim: int, hidden: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.pos_dim = pos_dim
        self.glob_dim = glob_dim
        self.hidden = hidden
        self.phi = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden + glob_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pos_set, glob):
        e = self.phi(pos_set).sum(dim=1)  # permutation-equivariant pooling
        return self.rho(torch.cat([e, glob], dim=1)).squeeze(-1)


class CapacityMatchedMLP(nn.Module):
    """Flat MLP over the full feature vector, width-scaled to match a target param count."""

    def __init__(self, in_dim: int, target_params: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.in_dim = in_dim
        w1, w2 = _search_mlp_widths(in_dim, target_params)
        self.net = nn.Sequential(
            nn.Linear(in_dim, w1), nn.ReLU(),
            nn.Linear(w1, w2), nn.ReLU(),
            nn.Linear(w2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _search_mlp_widths(in_dim: int, target: int, r: int = 2, tol: float = 0.08):
    """Find (w1, w2) with w1 = r*w2 minimizing |params - target|/target < tol."""
    best = None
    best_err = float("inf")
    for w2 in range(8, 512, 8):
        w1 = r * w2
        p = (in_dim * w1 + w1) + (w1 * w2 + w2) + (w2 + 1)
        err = abs(p - target) / max(target, 1)
        if err < best_err:
            best_err = err
            best = (w1, w2)
        if err < tol:
            return best
    return best


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def split_pos_glob(Xflat: torch.Tensor, W: int, pos_dim: int):
    """Split a flat feature batch (B, W*pos_dim + glob_dim) into (pos_set, glob)."""
    pos = Xflat[:, :W * pos_dim].reshape(Xflat.shape[0], W, pos_dim)
    glob = Xflat[:, W * pos_dim:]
    return pos, glob
