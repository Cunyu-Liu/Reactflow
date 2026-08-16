"""Backbone registry and factory for ReactFlow Phase C1-3.

This module provides the central :data:`BACKBONE_REGISTRY` mapping canonical
backbone names to their constructor entry points, plus the
:func:`build_backbone` factory that the trainer calls to instantiate a backbone
from a name and a :class:`~reactflow.backbones.foundation.base.BackboneConfig`.

Every backbone name in the registry has a matching ``default_config()`` helper
so that a caller can obtain a valid config without knowing the per-backbone
constructor arguments.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (unified backbone
interface).

Complexity
----------
- :func:`build_backbone`: ``O(P)`` where ``P`` is the constructed backbone's
  parameter count (parameter registration dominates).
"""

from __future__ import annotations

from typing import Dict, Sequence

from .base import (
    BackboneConfig,
    BackboneMode,
    BackboneNotAvailableError,
    BackboneOutput,
    FoundationBackbone,
)
from .ernie_rna import ERNIERNABackbone
from .ernie_rna import default_config as ernie_rna_default_config
from .from_scratch import FromScratchBackbone
from .from_scratch import default_config as from_scratch_default_config
from .ribonanza import DEFAULT_SHARD_ROOT, RibonanzaNet2Backbone
from .ribonanza import default_config as ribonanza_default_config
from .rinalmo import RiNALMoBackbone
from .rinalmo import default_config as rinalmo_default_config
from .rna_fm import RNAFMBackbone
from .rna_fm import default_config as rna_fm_default_config


# ---------------------------------------------------------------------------
# Constructor entry points
# ---------------------------------------------------------------------------


def _build_ribonanzanet2(config: BackboneConfig, mode: BackboneMode) -> FoundationBackbone:
    """Construct :class:`RibonanzaNet2Backbone` from a config + mode."""
    return RibonanzaNet2Backbone(config, shard_root=DEFAULT_SHARD_ROOT, mode=mode)


def _build_rinalmo(config: BackboneConfig, mode: BackboneMode) -> FoundationBackbone:
    """Construct :class:`RiNALMoBackbone` from a config + mode."""
    return RiNALMoBackbone(config, mode=mode)


def _build_ernie_rna(config: BackboneConfig, mode: BackboneMode) -> FoundationBackbone:
    """Construct :class:`ERNIERNABackbone` from a config + mode."""
    return ERNIERNABackbone(config, mode=mode)


def _build_rna_fm(config: BackboneConfig, mode: BackboneMode) -> FoundationBackbone:
    """Construct :class:`RNAFMBackbone` from a config + mode."""
    return RNAFMBackbone(config, mode=mode)


def _build_from_scratch(config: BackboneConfig, mode: BackboneMode) -> FoundationBackbone:
    """Construct :class:`FromScratchBackbone` from a config + mode."""
    return FromScratchBackbone(
        config,
        single_dim=config.frozen_feature_dim or 64,
        pair_dim=config.pair_feature_dim or 64,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BACKBONE_REGISTRY: Dict[str, Dict[str, object]] = {
    "ribonanzanet2": {
        "class": RibonanzaNet2Backbone,
        "build": _build_ribonanzanet2,
        "default_config": ribonanza_default_config,
        "available": True,
    },
    "rinalmo": {
        "class": RiNALMoBackbone,
        "build": _build_rinalmo,
        "default_config": rinalmo_default_config,
        "available": False,
    },
    "ernie_rna": {
        "class": ERNIERNABackbone,
        "build": _build_ernie_rna,
        "default_config": ernie_rna_default_config,
        "available": False,
    },
    "rna_fm": {
        "class": RNAFMBackbone,
        "build": _build_rna_fm,
        "default_config": rna_fm_default_config,
        "available": False,
    },
    "from_scratch": {
        "class": FromScratchBackbone,
        "build": _build_from_scratch,
        "default_config": from_scratch_default_config,
        "available": True,
    },
}
"""Registry mapping canonical backbone names to their entry points.

Each value is a dict with keys:

- ``class``: the :class:`FoundationBackbone` subclass.
- ``build``: ``build(config, mode) -> FoundationBackbone`` constructor.
- ``default_config``: ``default_config() -> BackboneConfig`` factory.
- ``available``: whether the backbone's weights/features are usable now
  (``True`` for cached/from-scratch, ``False`` for manifest-only).
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backbone(
    name: str,
    config: BackboneConfig,
    *,
    mode: BackboneMode = BackboneMode.FROZEN,
) -> FoundationBackbone:
    """Instantiate a foundation backbone by registry name.

    Formula: look up ``name`` in :data:`BACKBONE_REGISTRY`, then call the
    registered ``build(config, mode)`` entry point.  Complexity: ``O(P)``
    where ``P`` is the constructed backbone's parameter count.

    Args:
        name: canonical backbone name (case-insensitive), one of
            ``"ribonanzanet2"``, ``"rinalmo"``, ``"ernie_rna"``,
            ``"rna_fm"``, ``"from_scratch"``.
        config: :class:`BackboneConfig` for the backbone.
        mode: operating :class:`BackboneMode`.

    Returns:
        A :class:`FoundationBackbone` instance.

    Raises:
        KeyError: if ``name`` is not in the registry.
        BackboneNotAvailableError: if the backbone is manifest-only and the
            caller immediately invokes forward on it.
    """

    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in BACKBONE_REGISTRY:
        valid = ", ".join(sorted(BACKBONE_REGISTRY))
        raise KeyError(
            f"Unknown backbone name {name!r}. Valid names: {valid}."
        )
    entry = BACKBONE_REGISTRY[key]
    build_fn = entry["build"]  # type: ignore[operator]
    return build_fn(config, mode)


def list_backbones() -> Sequence[str]:
    """Return the sorted list of registered backbone names.

    Complexity: ``O(R log R)`` where ``R`` is the registry size.
    """

    return tuple(sorted(BACKBONE_REGISTRY))


def get_default_config(name: str) -> BackboneConfig:
    """Return the default :class:`BackboneConfig` for a registered backbone.

    Formula: look up ``name`` and call its ``default_config()`` factory.
    Complexity: ``O(1)``.

    Args:
        name: canonical backbone name (case-insensitive).

    Raises:
        KeyError: if ``name`` is not in the registry.
    """

    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in BACKBONE_REGISTRY:
        valid = ", ".join(sorted(BACKBONE_REGISTRY))
        raise KeyError(
            f"Unknown backbone name {name!r}. Valid names: {valid}."
        )
    factory = BACKBONE_REGISTRY[key]["default_config"]  # type: ignore[index]
    return factory()  # type: ignore[operator]
