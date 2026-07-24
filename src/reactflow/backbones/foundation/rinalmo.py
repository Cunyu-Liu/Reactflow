"""RiNALMo foundation backbone for ReactFlow Phase C1-3 (manifest-only).

RiNALMo (Tan et al., 2024) is a ribonucleic acid language model that covers a
wide range of RNA tasks.  This is a **manifest-only** entry: the weights are not
downloaded yet (``downloaded=False``), so :meth:`RiNALMoBackbone.forward` raises
:class:`BackboneNotAvailableError` with download instructions.  When the weights
become available locally, the backbone will support the
:attr:`BackboneMode.FROZEN`, :attr:`BackboneMode.LORA`, and
:attr:`BackboneMode.FULL_FINE_TUNE` modes.

Provenance (per C1-1 audit)
---------------------------
- Source: HuggingFace ``lcm-lab/RiNALMo``
- License: ``MIT``
- Max length: 1024 nucleotides
- Contamination status: ``unknown_contamination`` (not yet audited against the
  ReactFlow training/eval splits)

Complexity
----------
- Construction: ``O(1)``.
- Forward (when available): ``O(B * L * d_single)`` for singles,
  ``O(B * L^2 * d_pair)`` for pairs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .base import (
    BackboneConfig,
    BackboneMode,
    BackboneNotAvailableError,
    BackboneOutput,
    FoundationBackbone,
)

MODEL_NAME = "RiNALMo"
MODEL_SOURCE = "huggingface:lcm-lab/RiNALMo"
DOWNLOAD_URL = "https://huggingface.co/lcm-lab/RiNALMo"
LICENSE = "MIT"
MAX_LENGTH = 1024
CONTAMINATION_STATUS = "unknown_contamination"
EXPECTED_SINGLE_DIM = 1280
"""Documented per-nucleotide embedding dimension of RiNALMo (ESM-2 style)."""

_DOWNLOAD_HINT = (
    "Download the RiNALMo checkpoint and tokenizer, then construct "
    "RiNALMoBackbone with downloaded=True (or set BackboneConfig.downloaded=True) "
    "and implement the weight-loading path. Until then the backbone is a "
    "manifest placeholder."
)


class RiNALMoBackbone(FoundationBackbone):
    """Manifest-only RiNALMo backbone (weights not yet downloaded).

    This class documents RiNALMo's provenance and raises
    :class:`BackboneNotAvailableError` from :meth:`forward` until the weights
    are present.  When ``downloaded=True``, the backbone will load the live
    RiNALMo encoder and support the FROZEN / LORA / FULL_FINE_TUNE modes:

    - :attr:`BackboneMode.FROZEN`: run the encoder under ``torch.no_grad`` and
      return its per-nucleotide representations.
    - :attr:`BackboneMode.LORA`: freeze the encoder and insert LoRA adapters
      into the attention/FFN projections.
    - :attr:`BackboneMode.FULL_FINE_TUNE`: unfreeze the whole encoder.

    Args:
        config: :class:`BackboneConfig` for RiNALMo (see :func:`default_config`).
        mode: operating :class:`BackboneMode`.

    Complexity: ``O(1)`` construction.
    """

    def __init__(
        self,
        config: BackboneConfig,
        *,
        mode: BackboneMode = BackboneMode.FROZEN,
    ) -> None:
        super().__init__(config, mode)

    def forward(
        self,
        indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        record_ids: Optional[Sequence[str]] = None,
    ) -> BackboneOutput:
        """Return single/pair features, or raise if weights are unavailable.

        Raises:
            BackboneNotAvailableError: when ``config.downloaded`` is False.

        Complexity: ``O(1)`` to raise.
        """

        if not self._config.downloaded:
            raise BackboneNotAvailableError(
                MODEL_NAME, DOWNLOAD_URL, hint=_DOWNLOAD_HINT
            )
        # When weights become available, load the live RiNALMo encoder here and
        # dispatch on self._mode (FROZEN / LORA / FULL_FINE_TUNE).
        raise NotImplementedError(
            "RiNALMo live-encoder path is not implemented yet. Set "
            "downloaded=True and implement the weight-loading path."
        )

    def get_provenance(self) -> BackboneConfig:
        """Return the :class:`BackboneConfig` documenting RiNALMo's provenance.

        Complexity: ``O(1)``.
        """

        return self._config


def default_config() -> BackboneConfig:
    """Return the canonical manifest-only :class:`BackboneConfig` for RiNALMo.

    Formula: pins ``downloaded=False``, ``frozen_feature_dim=1280`` (documented
    RiNALMo embedding size), ``max_length=1024``, ``license="MIT"``, and
    ``contamination_status="unknown_contamination"`` per the C1-1 audit.
    Complexity: ``O(1)``.

    Returns:
        A :class:`BackboneConfig` with RiNALMo manifest defaults.
    """

    return BackboneConfig(
        model_name=MODEL_NAME,
        model_source=MODEL_SOURCE,
        model_revision="main",
        license=LICENSE,
        weights_sha256="",
        code_revision="",
        tokenizer="rinalmo-bpe",
        max_length=MAX_LENGTH,
        contamination_status=CONTAMINATION_STATUS,
        frozen_feature_dim=EXPECTED_SINGLE_DIM,
        pair_feature_dim=0,
        downloaded=False,
    )
