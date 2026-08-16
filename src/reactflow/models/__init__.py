"""Static structure prediction models for ReactFlow 2.0 (Phase C1-2).

This package contains the new symmetric PairFormer backbone and the
matched-capacity baselines (bilinear pair head, CNN/UNet pair head) used
for the C1-2 pilot experiment.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 332-452.
"""

from .static_pairformer import StaticPairFormer, PairFormerConfig, PairFormerOutput
from .bilinear_pair_head import BilinearPairHead
from .cnn_pair_head import CNNPairHead, UNetPairHead

__all__ = [
    "StaticPairFormer",
    "PairFormerConfig",
    "PairFormerOutput",
    "BilinearPairHead",
    "CNNPairHead",
    "UNetPairHead",
]
