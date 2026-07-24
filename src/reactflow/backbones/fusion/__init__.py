"""Pair-aware fusion strategies for ReactFlow Phase C1-3.

This package provides strategies for combining the outputs of multiple
foundation backbones (or multiple layers of a single backbone) into a unified
single (per-position) and pair (per-pair) representation that the downstream
PairFormer consumes.

Strategies
----------
- :class:`SingleOnlyAdapter` -- concatenation + projection of single features
  only (ablation baseline; no pair output).
- :class:`PairFeatureAdapter` -- concatenation + projection of both single and
  pair features, with an outer-product-mean fallback when pairs are missing.
- :class:`GatedMultiEncoderFusion` -- learned softmax gates over backbones.
- :class:`CrossLayerWeightedFusion` -- learned weighted sum of intermediate
  layers from a single backbone.
- :class:`TeacherBPPDistillation` -- knowledge-distillation loss from a frozen
  teacher model (not a real fusion strategy; identity pass-through forward).

All strategies share the :class:`FusionStrategy` ABC and :class:`FusionConfig`
configuration dataclass.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (pair-aware fusion
strategies).
"""

from __future__ import annotations

from .base import (
    FusionConfig,
    FusionOutput,
    FusionStrategy,
)
from .cross_layer import CrossLayerWeightedFusion
from .gated_multi import GatedMultiEncoderFusion
from .pair_feature import PairFeatureAdapter
from .single_only import SingleOnlyAdapter
from .teacher_bpp import TeacherBPPDistillation

__all__ = [
    # base
    "FusionStrategy",
    "FusionConfig",
    "FusionOutput",
    # strategies
    "SingleOnlyAdapter",
    "PairFeatureAdapter",
    "GatedMultiEncoderFusion",
    "CrossLayerWeightedFusion",
    "TeacherBPPDistillation",
]
