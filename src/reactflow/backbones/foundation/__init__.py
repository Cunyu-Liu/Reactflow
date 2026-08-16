"""Unified foundation backbone interface for ReactFlow Phase C1-3.

This package provides the abstract :class:`FoundationBackbone` contract and the
concrete backbone implementations (RibonanzaNet2, RiNALMo, ERNIE-RNA, RNA-FM,
and from-scratch), plus a registry/factory for construction by name.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (unified backbone
interface).

Public API
----------
- :class:`FoundationBackbone`, :class:`BackboneConfig`, :class:`BackboneOutput`,
  :class:`BackboneMode`, :class:`BackboneNotAvailableError` -- the contract.
- :class:`RibonanzaNet2Backbone`, :class:`RiNALMoBackbone`,
  :class:`ERNIERNABackbone`, :class:`RNAFMBackbone`,
  :class:`FromScratchBackbone` -- the concrete backbones.
- :data:`BACKBONE_REGISTRY`, :func:`build_backbone`, :func:`list_backbones`,
  :func:`get_default_config` -- the registry/factory.
"""

from __future__ import annotations

from .base import (
    BackboneConfig,
    BackboneMode,
    BackboneNotAvailableError,
    BackboneOutput,
    FoundationBackbone,
)
from .ernie_rna import (
    ERNIERNABackbone,
    default_config as ernie_rna_default_config,
)
from .from_scratch import (
    FromScratchBackbone,
    default_config as from_scratch_default_config,
)
from .registry import (
    BACKBONE_REGISTRY,
    build_backbone,
    get_default_config,
    list_backbones,
)
from .ribonanza import (
    DEFAULT_SHARD_ROOT,
    RIBONANZANET2_FROZEN_DIM,
    RibonanzaNet2Backbone,
    default_config as ribonanza_default_config,
)
from .rinalmo import (
    RiNALMoBackbone,
    default_config as rinalmo_default_config,
)
from .rna_fm import (
    RNAFMBackbone,
    default_config as rna_fm_default_config,
)

__all__ = [
    # contract
    "FoundationBackbone",
    "BackboneConfig",
    "BackboneOutput",
    "BackboneMode",
    "BackboneNotAvailableError",
    # concrete backbones
    "RibonanzaNet2Backbone",
    "RiNALMoBackbone",
    "ERNIERNABackbone",
    "RNAFMBackbone",
    "FromScratchBackbone",
    # registry / factory
    "BACKBONE_REGISTRY",
    "build_backbone",
    "list_backbones",
    "get_default_config",
    # default config factories
    "ribonanza_default_config",
    "rinalmo_default_config",
    "ernie_rna_default_config",
    "rna_fm_default_config",
    "from_scratch_default_config",
    # constants
    "DEFAULT_SHARD_ROOT",
    "RIBONANZANET2_FROZEN_DIM",
]
