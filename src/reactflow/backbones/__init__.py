"""Backbone building blocks for ReactFlow 2.0 static structure prediction.

This package contains the input embeddings, pair initialization, and
PairFormer block implementations used by the new symmetric pair predictor.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 350-376 (C1-2).
"""

from .embeddings import (
    InputEmbedding,
    NucleotideEmbedding,
    PositionalEmbedding,
    RelativeDistanceEmbedding,
    pair_compatibility_matrix,
    encode_sequence,
    encode_batch,
    bin_relative_distance,
    sinusoidal_positions,
)
from .pair_init import SymmetricPairInit
from .triangle import (
    TriangleMultiplicativeUpdate,
    TriangleAttention,
    PairTransition,
)
from .outer import (
    OuterProductMean,
    PairToSingleAttention,
    SingleRowAttention,
    SingleTransition,
)

__all__ = [
    # embeddings
    "InputEmbedding",
    "NucleotideEmbedding",
    "PositionalEmbedding",
    "RelativeDistanceEmbedding",
    "pair_compatibility_matrix",
    "encode_sequence",
    "encode_batch",
    "bin_relative_distance",
    "sinusoidal_positions",
    # pair init
    "SymmetricPairInit",
    # triangle
    "TriangleMultiplicativeUpdate",
    "TriangleAttention",
    "PairTransition",
    # outer / single
    "OuterProductMean",
    "PairToSingleAttention",
    "SingleRowAttention",
    "SingleTransition",
]
