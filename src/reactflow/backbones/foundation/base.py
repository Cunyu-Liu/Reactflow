"""Unified foundation backbone interface for ReactFlow Phase C1-3.

This module defines the abstract contract that every foundation-model backbone
(RibonanzaNet2, RiNALMo, ERNIE-RNA, RNA-FM, and from-scratch) implements.  A
backbone turns a batch of nucleotide-index tensors into a per-position single
representation ``s_i in R^{d_single}`` and (optionally) a per-pair
representation ``z_ij in R^{d_pair}``.  The unified interface lets the PairFormer
trainer swap backbones without changing the training loop.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (unified backbone
interface).

Operating modes
---------------
Every backbone supports a :class:`BackboneMode` that controls which parameters
receive gradients:

- :attr:`BackboneMode.FROZEN` -- backbone parameters are frozen; only
  downstream heads train.  For cached-feature backbones this returns the
  precomputed features directly (no autograd through the encoder).
- :attr:`BackboneMode.LORA` -- low-rank adapters (LoRA, Hu et al. 2021) are
  inserted on top of a frozen base; only the adapter weights update.  The
  update is ``h' = h + B A h`` with ``A in R^{r x d}``, ``B in R^{d x r}``,
  ``r << d``, and ``B`` zero-initialised so the adapter starts as identity.
- :attr:`BackboneMode.FULL_FINE_TUNE` -- all backbone parameters update with
  gradients (highest capacity, highest memory).
- :attr:`BackboneMode.INTERMEDIATE_LAYER_WEIGHTED_SUM` -- a learned scalar
  mixing weight ``alpha_l`` over intermediate hidden layers ``h_l``:
  ``h = sum_l softmax(alpha)_l * h_l``.  Only the mixing weights and downstream
  heads train; the encoder is frozen.  This is the ESM-2-style "last-layer
  probing" generalised to a learned convex combination.

Complexity
----------
- Forward: ``O(B * L * d_single)`` for single features, ``O(B * L^2 * d_pair)``
  for pair features (when computed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Operating mode
# ---------------------------------------------------------------------------


class BackboneMode(Enum):
    """Gradient-control mode for a :class:`FoundationBackbone`.

    Each member's docstring describes the exact behavior enforced by
    :meth:`FoundationBackbone._apply_mode`.
    """

    FROZEN = "frozen"
    """Backbone weights are frozen (``requires_grad=False``); only downstream
    heads train.  For cached-feature backbones the forward returns precomputed
    features with no autograd through the encoder, so memory is ``O(B*L*d)``."""

    LORA = "lora"
    """Low-rank adapters are inserted on top of a frozen base
    (``h' = h + B A h``, ``A in R^{r x d}``, ``B in R^{d x r}``, ``r << d``).
    Only ``A`` and ``B`` (and downstream heads) update.  ``B`` is zero-init so
    the adapter is identity at step 0.  Memory is ``O(B*L*d)`` plus ``O(r*d)``
    adapter params."""

    FULL_FINE_TUNE = "full_fine_tune"
    """All backbone parameters receive gradients.  Highest capacity and highest
    activation memory (``O(B*L*d)`` plus full parameter-gradient memory
    ``O(P)`` where ``P`` is the parameter count)."""

    INTERMEDIATE_LAYER_WEIGHTED_SUM = "intermediate_layer_weighted_sum"
    """Encoder is frozen; a learned convex combination of intermediate hidden
    layers is taken: ``h = sum_l softmax(alpha)_l * h_l`` where ``alpha`` is a
    trainable vector of per-layer logits.  Only ``alpha`` (and downstream heads)
    updates.  Requires a backbone whose intermediate layers are accessible
    (i.e., the live model is loaded, not a single cached feature tensor)."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BackboneNotAvailableError(RuntimeError):
    """Raised when a backbone's weights are not yet downloaded.

    Manifest-only backbones (``downloaded=False``) raise this from
    :meth:`FoundationBackbone.forward` so that downstream code fails loudly
    rather than silently producing random features.  The error message carries
    download instructions.

    Complexity: ``O(1)`` to raise.
    """

    def __init__(self, model_name: str, download_url: str, *, hint: str = "") -> None:
        msg = (
            f"Foundation backbone '{model_name}' is not available: weights have "
            f"not been downloaded.  Download from: {download_url}"
        )
        if hint:
            msg += f"\n{hint}"
        super().__init__(msg)
        self.model_name = model_name
        self.download_url = download_url


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    """Provenance and dimensionality metadata for a foundation backbone.

    This dataclass is the single source of truth for a backbone's identity:
    where it came from, what license it carries, whether its weights are
    available locally, and what feature dimensions it produces.  It is returned
    by :meth:`FoundationBackbone.get_provenance` so any consumer can audit the
    exact model revision without inspecting the backbone internals.

    Formula: ``frozen_feature_dim`` and ``pair_feature_dim`` pin the output
    tensor shapes ``single in R^{B x L x d_single}`` and
    ``pair in R^{B x L x L x d_pair}``.  Complexity: ``O(1)`` metadata.

    Attributes:
        model_name: human-readable identifier, e.g. ``"RibonanzaNet2"``.
        model_source: origin, e.g. ``"github:sh-ogawa/RibonanzaNet2"`` or
            ``"huggingface:lcm-lab/RiNALMo"``.
        model_revision: exact model revision/commit/tag, e.g. a git SHA or
            HuggingFace revision string.
        license: SPDX-style license identifier, e.g. ``"MIT"`` or
            ``"Apache-2.0"``.
        weights_sha256: SHA-256 of the source weight file(s); ``""`` when no
            weights are present (manifest-only entries).
        code_revision: revision of the model-definition code used to export
            frozen features (distinct from ``model_revision`` which is the
            weight checkpoint tag).
        tokenizer: tokenizer name, e.g. ``"ribonanza-bpe"`` or ``"nucleotide"``.
        max_length: maximum sequence length the backbone accepts.
        contamination_status: label from the C1-1 contamination audit, e.g.
            ``"unknown_contamination"`` or ``"not_applicable"``.
        frozen_feature_dim: dimension ``d_single`` of the per-position single
            representation (``0`` if not applicable yet).
        pair_feature_dim: dimension ``d_pair`` of the per-pair representation
            (``0`` if pair features are never produced).
        downloaded: whether the weights / frozen features are available
            locally.  ``False`` for manifest-only entries.
    """

    model_name: str
    model_source: str
    model_revision: str = ""
    license: str = ""
    weights_sha256: str = ""
    code_revision: str = ""
    tokenizer: str = ""
    max_length: int = 0
    contamination_status: str = "unknown"
    frozen_feature_dim: int = 0
    pair_feature_dim: int = 0
    downloaded: bool = False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class BackboneOutput:
    """Output container for :meth:`FoundationBackbone.forward`.

    Formula: ``single_features[b, i, :]`` is the per-nucleotide representation
    at position ``i`` of sample ``b``; ``pair_features[b, i, j, :]`` is the
    per-pair representation; ``mask[b, i]`` is True for real (non-padding)
    positions.  ``pair_features`` is ``None`` when the backbone does not
    produce a pair stack (the caller then initialises pairs from singles via
    :class:`~reactflow.backbones.pair_init.SymmetricPairInit`).

    Attributes:
        single_features: FloatTensor of shape ``(B, L, d_single)``.
        pair_features: optional FloatTensor of shape ``(B, L, L, d_pair)``;
            ``None`` when the backbone emits single features only.
        mask: BoolTensor of shape ``(B, L)`` where True marks a real position.
    """

    single_features: torch.Tensor
    mask: torch.Tensor
    pair_features: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class FoundationBackbone(nn.Module, ABC):
    """Abstract base class for all ReactFlow foundation backbones.

    A backbone is an :class:`~torch.nn.Module` that maps nucleotide indices to
    single (and optionally pair) representations.  All concrete backbones
    implement two methods:

    - :meth:`forward` -- produce a :class:`BackboneOutput` from indices/mask.
    - :meth:`get_provenance` -- return the :class:`BackboneConfig` describing
      the exact model revision, license, and availability.

    The base class stores the operating :class:`BackboneMode` and provides a
    helper :meth:`_apply_mode` that subclasses call after constructing their
    parameters to enforce the gradient-control contract (freeze / unfreeze).

    Args:
        config: :class:`BackboneConfig` describing this backbone.
        mode: :class:`BackboneMode` controlling gradient flow.

    Complexity: construction is ``O(P)`` (parameter registration); forward
    complexity is backbone-specific and documented on each subclass.
    """

    def __init__(self, config: BackboneConfig, mode: BackboneMode = BackboneMode.FROZEN) -> None:
        super().__init__()
        self._config: BackboneConfig = config
        self._mode: BackboneMode = mode

    # -- public properties --------------------------------------------------

    @property
    def config(self) -> BackboneConfig:
        """Return this backbone's configuration (read-only view)."""
        return self._config

    @property
    def mode(self) -> BackboneMode:
        """Return the current operating mode."""
        return self._mode

    @property
    def single_dim(self) -> int:
        """Return the single-feature dimension ``d_single``."""
        return self._config.frozen_feature_dim

    @property
    def pair_dim(self) -> int:
        """Return the pair-feature dimension ``d_pair`` (0 if none)."""
        return self._config.pair_feature_dim

    # -- abstract API -------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        record_ids: Optional[Sequence[str]] = None,
    ) -> BackboneOutput:
        """Compute single (and optionally pair) features for a batch.

        Args:
            indices: LongTensor of shape ``(B, L)`` with nucleotide vocab
                indices (see :mod:`reactflow.backbones.embeddings`).
            mask: optional BoolTensor of shape ``(B, L)`` where True marks a
                real (non-padding) position.  If ``None``, all positions are
                treated as real.
            record_ids: optional sequence of length ``B`` giving the source
                record id for each sample.  Frozen (cached-feature) backbones
                use this to look up precomputed features; live backbones ignore
                it.

        Returns:
            :class:`BackboneOutput` with ``single_features`` of shape
            ``(B, L, d_single)`` and optional ``pair_features`` of shape
            ``(B, L, L, d_pair)``.

        Complexity: backbone-specific; see each subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_provenance(self) -> BackboneConfig:
        """Return the :class:`BackboneConfig` describing this backbone.

        The returned config carries the exact model revision, weights SHA-256,
        license, and contamination status so any consumer can audit the model
        without inspecting backbone internals.

        Complexity: ``O(1)``.
        """
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------

    def _apply_mode(self) -> None:
        """Enforce the gradient-control contract of :attr:`mode`.

        - :attr:`BackboneMode.FROZEN`: set ``requires_grad=False`` on every
          parameter of this module (subclasses that hold LoRA adapters or
          mixing weights should set those to trainable *after* calling this).
        - :attr:`BackboneMode.LORA`: freeze the base; adapter params (added by
          subclasses) remain trainable by default.
        - :attr:`BackboneMode.FULL_FINE_TUNE`: ensure every parameter is
          trainable (``requires_grad=True``).
        - :attr:`BackboneMode.INTERMEDIATE_LAYER_WEIGHTED_SUM`: freeze the base;
          the mixing-weight parameter (added by subclasses) remains trainable.

        Subclasses are expected to call this at the end of ``__init__`` and then
        re-enable gradients on any mode-specific trainable parameters.

        Complexity: ``O(P)`` where ``P`` is the parameter count.
        """
        if self._mode in (BackboneMode.FROZEN, BackboneMode.LORA,
                          BackboneMode.INTERMEDIATE_LAYER_WEIGHTED_SUM):
            for p in self.parameters():
                p.requires_grad_(False)
        elif self._mode == BackboneMode.FULL_FINE_TUNE:
            for p in self.parameters():
                p.requires_grad_(True)
