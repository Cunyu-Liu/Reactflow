"""Gated multi-encoder fusion with learnable per-backbone gates.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (pair-aware fusion
strategies).

This strategy learns a softmax gate over the ``N`` backbone encoders so the
model can discover how much to trust each encoder's single and pair
representations.  Each backbone's features are first projected to the common
``d_single`` / ``d_pair`` dimensions, then combined via a learned convex
combination.

Formula
-------
Single fusion::

    gate = softmax([g_1, g_2, ..., g_N])       # g_k are learnable logits
    h_k  = LayerNorm(W_k^{single} s_k)          # project backbone k to d_single
    fused_single = sum_k gate_k * h_k

Pair fusion (when all backbones provide pair features)::

    fused_pair = sum_k gate_k * LayerNorm(W_k^{pair} p_k)

When some backbones lack pair features, the missing entries are dropped and
the gates are re-normalised over the backbones that *do* provide pairs.  If
*no* backbone provides a pair stack, ``fused_pair`` falls back to the
outer-product-mean of the fused singles (so the downstream PairFormer always
receives a pair stack).

Gate initialisation: ``g_k = 0`` for all ``k``, so ``softmax(g) = 1/N``
(uniform) at step 0.  This lets training start from an equal-weight ensemble
and learn to specialise.

Complexity
----------
- Time: ``O(B * L * sum_k d_k * d_single)`` for the per-backbone single
  projections (the gate weighting is ``O(B * L * d_single * N)`` which is
  dominated by the projections).  Pair fusion adds
  ``O(B * L^2 * sum_k p_k * d_pair)`` when pairs are available.
- Memory: ``O(N * B * L * d_single)`` for the projected single stacks (held
  simultaneously for the weighted sum), plus the pair equivalent.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..outer import OuterProductMean
from .base import FusionConfig, FusionStrategy


class GatedMultiEncoderFusion(FusionStrategy):
    """Learned gated convex combination of backbone features.

    Args:
        config: :class:`FusionConfig` with ``fusion_type="gated_multi"``.
        input_single_dims: sequence of length ``N`` giving the single-feature
            dimension ``d_k`` of each backbone.
        input_pair_dims: sequence of length ``N`` giving the pair-feature
            dimension ``p_k`` of each backbone (``0`` if never produced).

    Complexity: construction ``O(sum_k d_k * d_single + sum_k p_k * d_pair)``;
    forward as documented in the module docstring.
    """

    def __init__(
        self,
        config: FusionConfig,
        input_single_dims: Sequence[int],
        input_pair_dims: Sequence[int],
    ) -> None:
        if len(input_single_dims) != config.num_backbones:
            raise ValueError(
                f"input_single_dims length ({len(input_single_dims)}) does not "
                f"match num_backbones ({config.num_backbones})"
            )
        if len(input_pair_dims) != config.num_backbones:
            raise ValueError(
                f"input_pair_dims length ({len(input_pair_dims)}) does not "
                f"match num_backbones ({config.num_backbones})"
            )
        super().__init__(config)
        self.input_single_dims: List[int] = list(input_single_dims)
        self.input_pair_dims: List[int] = list(input_pair_dims)

        # Per-backbone single projection: d_k -> d_single
        self.single_projs = nn.ModuleList(
            [
                nn.Linear(d, config.single_dim)
                for d in self.input_single_dims
            ]
        )
        for proj in self.single_projs:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(proj.bias)
        self.single_norms = nn.ModuleList(
            [nn.LayerNorm(config.single_dim) for _ in self.input_single_dims]
        )

        # Per-backbone pair projection: p_k -> d_pair (only for backbones that
        # can produce pairs, i.e. p_k > 0).
        self.pair_projs = nn.ModuleList(
            [
                nn.Linear(p, config.pair_dim) if p > 0 else None
                for p in self.input_pair_dims
            ]
        )
        for proj in self.pair_projs:
            if proj is not None:
                nn.init.normal_(proj.weight, mean=0.0, std=0.02)
                nn.init.zeros_(proj.bias)
        self.pair_norms = nn.ModuleList(
            [
                nn.LayerNorm(config.pair_dim) if p > 0 else None
                for p in self.input_pair_dims
            ]
        )

        # Learnable gate logits (one per backbone).  Zero-init so softmax
        # gives uniform 1/N weights at step 0.
        self.gate_logits = nn.Parameter(
            torch.zeros(config.num_backbones)
        )

        # Outer-product-mean fallback when no pair features are available.
        self.pair_opm = OuterProductMean(
            config.single_dim, config.pair_dim,
        )

        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(
        self,
        single_features: List[torch.Tensor],
        pair_features: List[Optional[torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Gated convex combination of backbone features.

        Args:
            single_features: list of length ``N``, each ``(B, L, d_k)``.
            pair_features: list of length ``N``, each ``(B, L, L, p_k)`` or
                ``None``.
            mask: BoolTensor ``(B, L)`` real-position mask.

        Returns:
            Tuple ``(fused_single, fused_pair)`` with shapes
            ``(B, L, d_single)`` and ``(B, L, L, d_pair)``.

        Complexity: ``O(B * L * sum_k d_k * d_single)`` for singles; pair
            fusion adds ``O(B * L^2 * sum_k p_k * d_pair)``.
        """
        if len(single_features) != self.num_backbones:
            raise ValueError(
                f"expected {self.num_backbones} single feature stacks, "
                f"got {len(single_features)}"
            )
        if len(pair_features) != self.num_backbones:
            raise ValueError(
                f"expected {self.num_backbones} pair feature entries, "
                f"got {len(pair_features)}"
            )

        # -- gated single fusion -------------------------------------------
        gate = F.softmax(self.gate_logits, dim=0)  # (N,)

        proj_singles: List[torch.Tensor] = []
        for k, (feat, proj, norm) in enumerate(
            zip(single_features, self.single_projs, self.single_norms)
        ):
            h = norm(proj(feat))  # (B, L, d_single)
            proj_singles.append(h)

        # Weighted sum: gate_k * h_k
        fused_single = torch.stack(proj_singles, dim=0)  # (N, B, L, d_single)
        fused_single = torch.einsum(
            "n,nbld->bld", gate, fused_single
        )  # (B, L, d_single)
        fused_single = self.dropout(fused_single)

        m = mask.unsqueeze(-1).to(fused_single.dtype)
        fused_single = fused_single * m

        # -- gated pair fusion ---------------------------------------------
        available_indices = [
            k
            for k, (p, dim) in enumerate(zip(pair_features, self.input_pair_dims))
            if p is not None and dim > 0
        ]

        if available_indices:
            # Re-normalise gates over the backbones that provide pairs.
            sub_logits = self.gate_logits[available_indices]
            sub_gate = F.softmax(sub_logits, dim=0)  # (M,)

            proj_pairs: List[torch.Tensor] = []
            for k in available_indices:
                p = pair_features[k]  # type: ignore[assignment]
                proj = self.pair_projs[k]  # type: ignore[assignment]
                norm = self.pair_norms[k]  # type: ignore[assignment]
                proj_pairs.append(norm(proj(p)))  # (B, L, L, d_pair)

            stacked = torch.stack(proj_pairs, dim=0)  # (M, B, L, L, d_pair)
            fused_pair = torch.einsum(
                "m,mbijp->bijp", sub_gate, stacked
            )  # (B, L, L, d_pair)
            fused_pair = self.dropout(fused_pair)
        else:
            # No pair features from any backbone: fall back to OPM.
            B, L, _ = fused_single.shape
            zero_pair = fused_single.new_zeros(B, L, L, self.pair_dim)
            fused_pair = self.pair_opm(fused_single, zero_pair, mask=mask)

        # Mask out pairs involving padding positions.
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).to(
            fused_pair.dtype
        )
        fused_pair = fused_pair * pair_mask

        return fused_single, fused_pair
