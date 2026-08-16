"""RNA-FM foundation backbone for ReactFlow Phase C1-3 (manifest-only).

RNA-FM (Chen et al., 2022; "Interpretable RNA Foundation Model") is a
transformer-based RNA language model pretrained on 23.7 million ncRNA
sequences.  This is a **manifest-only** entry: the weights are not downloaded
yet (``downloaded=False``), so :meth:`RNAFMBackbone.forward` raises
:class:`BackboneNotAvailableError` with download instructions.  When the weights
become available locally, the backbone will support the FROZEN / LORA /
FULL_FINE_TUNE modes.

Provenance (per C1-1 audit)
---------------------------
- Source: HuggingFace ``mailong-rl/RNA-FM``
- License: ``unknown`` (check the model card before redistribution)
- Max length: 1024 nucleotides
- Contamination status: ``unknown`` (not yet audited)

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

MODEL_NAME = "RNA-FM"
MODEL_SOURCE = "huggingface:mailong-rl/RNA-FM"
DOWNLOAD_URL = "https://huggingface.co/mailong-rl/RNA-FM"
LICENSE = "unknown"
MAX_LENGTH = 1024
CONTAMINATION_STATUS = "unknown"
EXPECTED_SINGLE_DIM = 320
"""Documented per-nucleotide embedding dimension of RNA-FM."""

_DOWNLOAD_HINT = (
    "Download the RNA-FM checkpoint and tokenizer, then construct "
    "RNAFMBackbone with downloaded=True (or set BackboneConfig.downloaded=True) "
    "and implement the weight-loading path. Until then the backbone is a "
    "manifest placeholder."
)


class RNAFMBackbone(FoundationBackbone):
    """Manifest-only RNA-FM backbone (weights not yet downloaded).

    This class documents RNA-FM's provenance and raises
    :class:`BackboneNotAvailableError` from :meth:`forward` until the weights
    are present.  When ``downloaded=True``, the backbone will load the live
    RNA-FM encoder and support the FROZEN / LORA / FULL_FINE_TUNE modes:

    - :attr:`BackboneMode.FROZEN`: run the encoder under ``torch.no_grad`` and
      return its per-nucleotide representations.
    - :attr:`BackboneMode.LORA`: freeze the encoder and insert LoRA adapters
      into the attention/FFN projections.
    - :attr:`BackboneMode.FULL_FINE_TUNE`: unfreeze the whole encoder.

    Args:
        config: :class:`BackboneConfig` for RNA-FM (see :func:`default_config`).
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
        raise NotImplementedError(
            "RNA-FM live-encoder path is not implemented yet. Set "
            "downloaded=True and implement the weight-loading path."
        )

    def get_provenance(self) -> BackboneConfig:
        """Return the :class:`BackboneConfig` documenting RNA-FM's provenance.

        Complexity: ``O(1)``.
        """

        return self._config


def default_config() -> BackboneConfig:
    """Return the canonical manifest-only :class:`BackboneConfig` for RNA-FM.

    Formula: pins ``downloaded=False``, ``frozen_feature_dim=320`` (documented
    RNA-FM embedding size), ``max_length=1024``, ``license="unknown"``, and
    ``contamination_status="unknown"`` per the C1-1 audit.  Complexity: ``O(1)``.

    Returns:
        A :class:`BackboneConfig` with RNA-FM manifest defaults.
    """

    return BackboneConfig(
        model_name=MODEL_NAME,
        model_source=MODEL_SOURCE,
        model_revision="main",
        license=LICENSE,
        weights_sha256="",
        code_revision="",
        tokenizer="rna-fm-bpe",
        max_length=MAX_LENGTH,
        contamination_status=CONTAMINATION_STATUS,
        frozen_feature_dim=EXPECTED_SINGLE_DIM,
        pair_feature_dim=0,
        downloaded=False,
    )
