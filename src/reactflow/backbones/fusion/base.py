"""Abstract fusion strategy for combining multiple foundation backbones.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (pair-aware fusion
strategies).

A fusion strategy takes the outputs of ``N`` foundation backbones -- each
producing a single representation ``s_i^{(k)} in R^{d_k}`` and (optionally) a
pair representation ``z_ij^{(k)} in R^{p_k}`` -- and combines them into a
unified single representation ``s_i in R^{d_single}`` and a unified pair
representation ``z_ij in R^{d_pair}`` that the downstream PairFormer consumes.

The unified contract lets the trainer swap fusion strategies (single-only
ablation, pair-aware concatenation, gated multi-encoder, cross-layer
weighted sum) without changing the training loop.

Complexity
----------
- Forward complexity is strategy-specific and documented on each subclass.
- All strategies produce outputs of fixed dimensions ``d_single`` and
  ``d_pair`` regardless of the number of input backbones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Configuration for a :class:`FusionStrategy`.

    Formula: ``single_dim`` and ``pair_dim`` pin the output tensor shapes
    ``single in R^{B x L x d_single}`` and
    ``pair in R^{B x L x L x d_pair}``.  ``num_backbones`` is the number of
    input backbone feature stacks the fusion consumes; it is used to size
    concatenation / projection layers.

    Attributes:
        fusion_type: identifier string (e.g. ``"single_only"``,
            ``"pair_feature"``, ``"gated_multi"``, ``"cross_layer"``).
            Used by the factory/registry to select the strategy class.
        single_dim: output single-feature dimension ``d_single``.
        pair_dim: output pair-feature dimension ``d_pair``.
        num_backbones: number ``N`` of backbone feature stacks to fuse.
        dropout: dropout rate applied after fusion (0 disables).
    """

    fusion_type: str = "single_only"
    single_dim: int = 256
    pair_dim: int = 128
    num_backbones: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.single_dim <= 0:
            raise ValueError("single_dim must be positive")
        if self.pair_dim <= 0:
            raise ValueError("pair_dim must be positive")
        if self.num_backbones < 1:
            raise ValueError("num_backbones must be >= 1")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class FusionOutput:
    """Output container for :meth:`FusionStrategy.forward`.

    Formula: ``single_features[b, i, :]`` is the fused per-nucleotide
    representation at position ``i`` of sample ``b``; ``pair_features[b, i, j, :]``
    is the fused per-pair representation; ``mask[b, i]`` is True for real
    (non-padding) positions.  ``pair_features`` is ``None`` when the fusion
    strategy does not produce a pair stack (the caller then initialises pairs
    from singles via :class:`~reactflow.backbones.pair_init.SymmetricPairInit`).

    Attributes:
        single_features: FloatTensor of shape ``(B, L, d_single)``.
        pair_features: optional FloatTensor of shape ``(B, L, L, d_pair)``;
            ``None`` when the fusion does not produce a pair stack.
        mask: BoolTensor of shape ``(B, L)`` where True marks a real position.
    """

    single_features: torch.Tensor
    mask: torch.Tensor
    pair_features: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class FusionStrategy(nn.Module, ABC):
    """Abstract base class for all pair-aware fusion strategies.

    A fusion strategy is an :class:`~torch.nn.Module` that maps the outputs of
    ``N`` foundation backbones to a unified single (and optionally pair)
    representation.  All concrete strategies implement one method:

    - :meth:`forward` -- produce fused ``(single, pair)`` features from the
      per-backbone lists of single and pair feature stacks.

    Args:
        config: :class:`FusionConfig` describing this fusion strategy.

    Complexity: construction is ``O(P)`` (parameter registration); forward
    complexity is strategy-specific and documented on each subclass.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self._config: FusionConfig = config

    # -- public properties --------------------------------------------------

    @property
    def config(self) -> FusionConfig:
        """Return this fusion strategy's configuration (read-only view)."""
        return self._config

    @property
    def single_dim(self) -> int:
        """Return the output single-feature dimension ``d_single``."""
        return self._config.single_dim

    @property
    def pair_dim(self) -> int:
        """Return the output pair-feature dimension ``d_pair``."""
        return self._config.pair_dim

    @property
    def num_backbones(self) -> int:
        """Return the number ``N`` of backbone feature stacks to fuse."""
        return self._config.num_backbones

    # -- abstract API -------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        single_features: List[torch.Tensor],
        pair_features: List[Optional[torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse per-backbone single and pair feature stacks.

        Args:
            single_features: list of length ``N`` where each element is a
                FloatTensor of shape ``(B, L, d_k)`` (the single representation
                from backbone ``k``).  Different backbones may have different
                ``d_k``; the strategy projects to the common ``d_single``.
            pair_features: list of length ``N`` where each element is either a
                FloatTensor of shape ``(B, L, L, p_k)`` (the pair representation
                from backbone ``k``) or ``None`` if that backbone does not
                produce a pair stack.
            mask: BoolTensor of shape ``(B, L)`` where True marks a real
                (non-padding) position.

        Returns:
            A tuple ``(fused_single, fused_pair)`` where ``fused_single`` has
            shape ``(B, L, d_single)`` and ``fused_pair`` is either ``None``
            or a FloatTensor of shape ``(B, L, L, d_pair)``.

        Complexity: strategy-specific; see each subclass.
        """
        raise NotImplementedError
