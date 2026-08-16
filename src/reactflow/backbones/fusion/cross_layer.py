"""Cross-layer weighted fusion for a single backbone's intermediate layers.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (pair-aware fusion
strategies) and :attr:`BackboneMode.INTERMEDIATE_LAYER_WEIGHTED_SUM`.

When a single foundation backbone exposes multiple intermediate hidden layers
(e.g. a transformer encoder's per-layer outputs ``h_1, h_2, ..., h_K``), this
strategy learns a convex combination of those layers.  This is the
ESM-2-style "last-layer probing" generalised to a learned weighted sum, and
mirrors the :attr:`BackboneMode.INTERMEDIATE_LAYER_WEIGHTED_SUM` operating
mode described in :mod:`reactflow.backbones.foundation.base`.

Formula
-------
::

    alpha = softmax([a_1, a_2, ..., a_K])      # a_l are learnable logits
    h_l   = LayerNorm(W_l^{single} s_l)        # project layer l to d_single
    fused_single = sum_l alpha_l * h_l

    # Pair features (optional per layer):
    fused_pair = sum_{l : p_l is not None} alpha_l' * LayerNorm(W_l^{pair} p_l)

where ``alpha'`` is the softmax re-normalised over the layers that *do*
provide pair features.  If no layer provides a pair stack, ``fused_pair``
falls back to the outer-product-mean of the fused singles.

Weight initialisation: ``a_l = 0`` for all ``l``, so ``softmax(a) = 1/K``
(uniform) at step 0.

Complexity
----------
- Time: ``O(B * L * sum_l d_l * d_single)`` for the per-layer single
  projections; pair projections add ``O(B * L^2 * sum_l p_l * d_pair)``.
- Memory: ``O(K * B * L * d_single)`` for the projected single stacks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..outer import OuterProductMean
from .base import FusionConfig, FusionStrategy


class CrossLayerWeightedFusion(FusionStrategy):
    """Learned weighted sum of intermediate layers from a single backbone.

    Args:
        config: :class:`FusionConfig` with ``fusion_type="cross_layer"``.
            ``num_backbones`` is reinterpreted as the number of layers ``K``.
        input_single_dims: sequence of length ``K`` giving the single-feature
            dimension ``d_l`` of each layer.  Layers may have different
            dimensions (the strategy projects each to ``d_single``).
        input_pair_dims: sequence of length ``K`` giving the pair-feature
            dimension ``p_l`` of each layer (``0`` if the layer never
            produces a pair stack).

    Complexity: construction
    ``O(sum_l d_l * d_single + sum_l p_l * d_pair)``; forward as above.
    """

    def __init__(
        self,
        config: FusionConfig,
        input_single_dims: Sequence[int],
        input_pair_dims: Sequence[int],
    ) -> None:
        K = config.num_backbones
        if len(input_single_dims) != K:
            raise ValueError(
                f"input_single_dims length ({len(input_single_dims)}) does not "
                f"match num_backbones/num_layers ({K})"
            )
        if len(input_pair_dims) != K:
            raise ValueError(
                f"input_pair_dims length ({len(input_pair_dims)}) does not "
                f"match num_backbones/num_layers ({K})"
            )
        super().__init__(config)
        self.num_layers: int = K
        self.input_single_dims: List[int] = list(input_single_dims)
        self.input_pair_dims: List[int] = list(input_pair_dims)

        # Per-layer single projection: d_l -> d_single
        self.single_projs = nn.ModuleList(
            [nn.Linear(d, config.single_dim) for d in self.input_single_dims]
        )
        for proj in self.single_projs:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(proj.bias)
        self.single_norms = nn.ModuleList(
            [nn.LayerNorm(config.single_dim) for _ in self.input_single_dims]
        )

        # Per-layer pair projection: p_l -> d_pair (only for layers with p_l > 0)
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

        # Learnable per-layer mixing logits.  Zero-init so softmax gives
        # uniform 1/K weights at step 0.
        self.layer_logits = nn.Parameter(torch.zeros(K))

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
        """Weighted sum of intermediate layer features.

        Args:
            single_features: list of length ``K``, each ``(B, L, d_l)``.
            pair_features: list of length ``K``, each ``(B, L, L, p_l)`` or
                ``None``.
            mask: BoolTensor ``(B, L)`` real-position mask.

        Returns:
            Tuple ``(fused_single, fused_pair)`` with shapes
            ``(B, L, d_single)`` and ``(B, L, L, d_pair)``.

        Complexity: ``O(B * L * sum_l d_l * d_single)`` for singles; pair
            projections add ``O(B * L^2 * sum_l p_l * d_pair)``.
        """
        K = self.num_layers
        if len(single_features) != K:
            raise ValueError(
                f"expected {K} single feature stacks (one per layer), "
                f"got {len(single_features)}"
            )
        if len(pair_features) != K:
            raise ValueError(
                f"expected {K} pair feature entries (one per layer), "
                f"got {len(pair_features)}"
            )

        # -- weighted single fusion -----------------------------------------
        weights = F.softmax(self.layer_logits, dim=0)  # (K,)

        proj_singles: List[torch.Tensor] = []
        for feat, proj, norm in zip(
            single_features, self.single_projs, self.single_norms
        ):
            h = norm(proj(feat))  # (B, L, d_single)
            proj_singles.append(h)

        stacked = torch.stack(proj_singles, dim=0)  # (K, B, L, d_single)
        fused_single = torch.einsum(
            "k,kbld->bld", weights, stacked
        )  # (B, L, d_single)
        fused_single = self.dropout(fused_single)

        m = mask.unsqueeze(-1).to(fused_single.dtype)
        fused_single = fused_single * m

        # -- weighted pair fusion -------------------------------------------
        available_indices = [
            l
            for l, (p, dim) in enumerate(zip(pair_features, self.input_pair_dims))
            if p is not None and dim > 0
        ]

        if available_indices:
            # Re-normalise weights over layers that provide pairs.
            sub_logits = self.layer_logits[available_indices]
            sub_weights = F.softmax(sub_logits, dim=0)  # (M,)

            proj_pairs: List[torch.Tensor] = []
            for l in available_indices:
                p = pair_features[l]  # type: ignore[assignment]
                proj = self.pair_projs[l]  # type: ignore[assignment]
                norm = self.pair_norms[l]  # type: ignore[assignment]
                proj_pairs.append(norm(proj(p)))  # (B, L, L, d_pair)

            stacked_pairs = torch.stack(proj_pairs, dim=0)  # (M, B, L, L, d_pair)
            fused_pair = torch.einsum(
                "m,mbnij->bnij", sub_weights, stacked_pairs
            )  # (B, L, L, d_pair)
            fused_pair = self.dropout(fused_pair)
        else:
            # No pair features from any layer: fall back to OPM.
            B, L, _ = fused_single.shape
            zero_pair = fused_single.new_zeros(B, L, L, self.pair_dim)
            fused_pair = self.pair_opm(fused_single, zero_pair, mask=mask)

        # Mask out pairs involving padding positions.
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).to(
            fused_pair.dtype
        )
        fused_pair = fused_pair * pair_mask

        return fused_single, fused_pair
