"""Pair representation modules for the static PairFormer.

Spec reference: ``ReactFlow分阶段执行提示词.md`` line 346 (suggested module
``src/reactflow/pair/``).

This module groups all pair-related building blocks:

- :class:`SymmetricPairInit` -- symmetric pair feature initialization.
- :class:`TriangleMultiplicativeUpdate` -- incoming/outgoing triangle updates.
- :class:`TriangleAttention` -- starting/ending node triangle attention.
- :class:`PairTransition` -- pair-wise feed-forward transition.
- :class:`OuterProductMean` -- single-to-pair outer product mean update.
- :class:`PairToSingleAttention` -- pair-to-single attention communication.

All classes are re-exported from :mod:`reactflow.backbones` for backward
compatibility.  New code should import from this module.
"""

from __future__ import annotations

from ..backbones.pair_init import SymmetricPairInit
from ..backbones.triangle import (
    TriangleMultiplicativeUpdate,
    TriangleAttention,
    PairTransition,
)
from ..backbones.outer import (
    OuterProductMean,
    PairToSingleAttention,
)

__all__ = [
    "SymmetricPairInit",
    "TriangleMultiplicativeUpdate",
    "TriangleAttention",
    "PairTransition",
    "OuterProductMean",
    "PairToSingleAttention",
]
