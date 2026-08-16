"""Single-only fusion adapter (ablation baseline).

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (pair-aware fusion
strategies).

This is the **minimal baseline** fusion strategy: it concatenates the single
representations from all backbones and projects to the target single
dimension, producing no pair representation.  The spec
("禁止只做单 token 线性 adapter") forbids this as the *final* model, but it
serves as an ablation baseline to quantify the value of pair-aware fusion.

Formula
-------
::

    fused_single = LayerNorm(W_proj(concat([s_1, s_2, ..., s_N])) + b)

where ``s_k in R^{d_k}`` is the single representation from backbone ``k`` and
``W_proj in R^{d_single x sum_k d_k}``.  ``pair_features`` is always ``None``;
the caller must initialise the pair stack from the fused singles (e.g. via
:class:`~reactflow.backbones.pair_init.SymmetricPairInit`).

Complexity
----------
- Time: ``O(B * L * sum_k d_k * d_single)`` for the projection.
- Memory: ``O(B * L * d_single)`` for the output (plus the concatenation
  buffer ``O(B * L * sum_k d_k)`` which is freed after the projection).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from .base import FusionConfig, FusionStrategy


class SingleOnlyAdapter(FusionStrategy):
    """Concatenate single features from all backbones and project.

    This strategy ignores pair features entirely (returns ``None`` for the
    fused pair representation).  It is the simplest fusion and serves as an
    ablation baseline against which the pair-aware strategies are measured.

    Args:
        config: :class:`FusionConfig` with ``fusion_type="single_only"``.
            ``single_dim`` is the output dimension; ``pair_dim`` is unused
            (no pair output) but kept for interface consistency.
        input_single_dims: sequence of length ``N`` giving the single-feature
            dimension ``d_k`` of each backbone.  The concatenation dimension
            is ``sum_k d_k``.

    Complexity: construction ``O(sum_k d_k * d_single)``; forward as above.
    """

    def __init__(
        self,
        config: FusionConfig,
        input_single_dims: Sequence[int],
    ) -> None:
        if len(input_single_dims) != config.num_backbones:
            raise ValueError(
                f"input_single_dims length ({len(input_single_dims)}) does not "
                f"match num_backbones ({config.num_backbones})"
            )
        if any(d <= 0 for d in input_single_dims):
            raise ValueError("all input_single_dims must be positive")
        super().__init__(config)
        self.input_single_dims: List[int] = list(input_single_dims)
        self.concat_dim: int = sum(self.input_single_dims)

        self.proj = nn.Linear(self.concat_dim, config.single_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(config.single_dim)
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(
        self,
        single_features: List[torch.Tensor],
        pair_features: List[Optional[torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Concatenate and project single features.

        Args:
            single_features: list of length ``N``, each ``(B, L, d_k)``.
            pair_features: ignored (this strategy produces no pair output).
            mask: BoolTensor ``(B, L)``; used to zero padding positions.

        Returns:
            Tuple ``(fused_single, None)`` where ``fused_single`` has shape
            ``(B, L, d_single)``.

        Complexity: ``O(B * L * concat_dim * d_single)``.
        """
        if len(single_features) != self.num_backbones:
            raise ValueError(
                f"expected {self.num_backbones} single feature stacks, "
                f"got {len(single_features)}"
            )
        # Validate per-backbone dims (cheap; catches mismatched configs).
        for k, (feat, d) in enumerate(
            zip(single_features, self.input_single_dims)
        ):
            if feat.size(-1) != d:
                raise ValueError(
                    f"single_features[{k}] has dim {feat.size(-1)}, "
                    f"expected {d}"
                )

        cat = torch.cat(single_features, dim=-1)  # (B, L, concat_dim)
        fused = self.proj(cat)                      # (B, L, d_single)
        fused = self.norm(fused)
        fused = self.dropout(fused)

        # Zero out padding positions.
        m = mask.unsqueeze(-1).to(fused.dtype)
        fused = fused * m

        return fused, None
