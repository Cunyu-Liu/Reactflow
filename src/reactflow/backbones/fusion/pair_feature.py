"""Pair-aware fusion via concatenation + projection.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (pair-aware fusion
strategies).

This strategy fuses both single and pair representations.  Single features
are concatenated and projected to ``d_single`` exactly as in
:class:`SingleOnlyAdapter`.  Pair features are fused in one of two ways:

1. **All pair features available**: concatenate the ``N`` pair stacks and
   project to ``d_pair``::

       fused_pair = LayerNorm(W_pair(concat([p_1, p_2, ..., p_N])) + b)

2. **Some pair features missing** (any ``p_k is None``): fall back to
   computing the pair stack from the fused singles via the
   :class:`~reactflow.backbones.outer.OuterProductMean`::

       fused_pair = OuterProductMean(fused_single, zeros)

   This keeps the output pair stack available even when backbones lack pair
   outputs, at the cost of a less-informative pair initialization.

Formula
-------
::

    fused_single = LayerNorm(W_single(concat([s_1, ..., s_N])) + b)

    if all(p_k is not None):
        fused_pair = LayerNorm(W_pair(concat([p_1, ..., p_N])) + b)
    else:
        fused_pair = OuterProductMean(fused_single, 0)

Complexity
----------
- Time: ``O(B * L * concat_single * d_single)`` for the single projection,
  plus ``O(B * L^2 * concat_pair * d_pair)`` for the pair projection (when
  pairs are available) or ``O(B * L^2 * D^2)`` for the outer-product-mean
  (when pairs are missing, ``D`` = OPM projection dim).
- Memory: ``O(B * L * d_single + B * L^2 * d_pair)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from ..outer import OuterProductMean
from .base import FusionConfig, FusionStrategy


class PairFeatureAdapter(FusionStrategy):
    """Fuse single and pair features via concatenation + projection.

    Args:
        config: :class:`FusionConfig` with ``fusion_type="pair_feature"``.
        input_single_dims: sequence of length ``N`` giving the single-feature
            dimension ``d_k`` of each backbone.
        input_pair_dims: sequence of length ``N`` giving the pair-feature
            dimension ``p_k`` of each backbone.  Entries may be ``0`` for
            backbones that never produce a pair stack (these contribute to
            the concatenation only when a pair tensor is actually passed at
            forward time).

    Complexity: construction ``O(sum d_k * d_single + sum p_k * d_pair)``;
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
        self.concat_single_dim: int = sum(self.input_single_dims)
        self.concat_pair_dim: int = sum(self.input_pair_dims)

        # Single projection: concat -> d_single
        self.single_proj = nn.Linear(self.concat_single_dim, config.single_dim)
        nn.init.normal_(self.single_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.single_proj.bias)
        self.single_norm = nn.LayerNorm(config.single_dim)

        # Pair projection: concat -> d_pair (only used when all pairs present)
        self.pair_proj = nn.Linear(self.concat_pair_dim, config.pair_dim)
        nn.init.normal_(self.pair_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pair_proj.bias)
        self.pair_norm = nn.LayerNorm(config.pair_dim)

        # Outer-product-mean fallback: initialise pair stack from fused singles.
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
        """Fuse single and pair features.

        Args:
            single_features: list of length ``N``, each ``(B, L, d_k)``.
            pair_features: list of length ``N``, each ``(B, L, L, p_k)`` or
                ``None``.
            mask: BoolTensor ``(B, L)`` real-position mask.

        Returns:
            Tuple ``(fused_single, fused_pair)`` with shapes
            ``(B, L, d_single)`` and ``(B, L, L, d_pair)``.

        Complexity: ``O(B * L * concat_single * d_single + B * L^2 *
            concat_pair * d_pair)`` (pairs available) or
            ``O(B * L^2 * D^2)`` (OPM fallback).
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

        # -- fused single ----------------------------------------------------
        cat_single = torch.cat(single_features, dim=-1)  # (B, L, concat_single)
        fused_single = self.single_proj(cat_single)       # (B, L, d_single)
        fused_single = self.single_norm(fused_single)
        fused_single = self.dropout(fused_single)

        m = mask.unsqueeze(-1).to(fused_single.dtype)
        fused_single = fused_single * m

        # -- fused pair ------------------------------------------------------
        all_pairs_present = all(p is not None for p in pair_features)

        if all_pairs_present and self.concat_pair_dim > 0:
            # Concatenate the (now guaranteed non-None) pair stacks.
            pair_list: List[torch.Tensor] = [
                p  # type: ignore[misc]
                for p in pair_features
                if p is not None
            ]
            cat_pair = torch.cat(pair_list, dim=-1)  # (B, L, L, concat_pair)
            fused_pair = self.pair_proj(cat_pair)    # (B, L, L, d_pair)
            fused_pair = self.pair_norm(fused_pair)
            fused_pair = self.dropout(fused_pair)
        else:
            # Fallback: initialise pair stack from the fused singles via the
            # outer-product-mean.  Pass a zero pair so the OPM acts purely as
            # an initializer: z = 0 + update = update.
            B, L, _ = fused_single.shape
            zero_pair = fused_single.new_zeros(B, L, L, self.pair_dim)
            fused_pair = self.pair_opm(fused_single, zero_pair, mask=mask)

        # Mask out pairs involving padding positions.
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).to(
            fused_pair.dtype
        )
        fused_pair = fused_pair * pair_mask

        return fused_single, fused_pair
