"""RNA-FM foundation backbone metadata for ReactFlow Phase C1-3.

RNA-FM (Chen et al., 2022; "Interpretable RNA Foundation Model") is a
transformer-based RNA language model pretrained on 23.7 million ncRNA
sequences.  ReactFlow V4 used the official ``rna_fm_t12`` checkpoint through a
separate frozen-cache path.  This generic adapter still has no live checkpoint
loader (``downloaded=False``), so :meth:`RNAFMBackbone.forward` raises
:class:`BackboneNotAvailableError`.  The availability flag therefore describes
this adapter, while the provenance below identifies the actual V4 asset.

Provenance (per C1-1 audit)
---------------------------
- Code: ``ml4bio/RNA-FM`` at commit
  ``348951516e0963d22bbb33b3c9fc18c89081d38e`` (``MIT``)
- Checkpoint: ``cuhkaih/rnafm`` at revision
  ``91d4a46d28d8054a7b429955e8fc0c253ba0afd6`` (``Apache-2.0``)
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
MODEL_VARIANT = "rna_fm_t12"
CODE_SOURCE = "github:ml4bio/RNA-FM"
CODE_REVISION = "348951516e0963d22bbb33b3c9fc18c89081d38e"
CODE_LICENSE = "MIT"
CHECKPOINT_SOURCE = "huggingface:cuhkaih/rnafm"
CHECKPOINT_REVISION = "91d4a46d28d8054a7b429955e8fc0c253ba0afd6"
CHECKPOINT_LICENSE = "Apache-2.0"
MODEL_SOURCE = CHECKPOINT_SOURCE
MODEL_REVISION = CHECKPOINT_REVISION
DOWNLOAD_URL = (
    "https://huggingface.co/cuhkaih/rnafm/resolve/"
    f"{CHECKPOINT_REVISION}/RNA-FM_pretrained.pth"
)
LICENSE = CHECKPOINT_LICENSE
MAX_LENGTH = 1024
CONTAMINATION_STATUS = "unknown"
EXPECTED_SINGLE_DIM = 640
"""Documented per-nucleotide embedding dimension of RNA-FM."""

_DOWNLOAD_HINT = (
    "Use the pinned ml4bio/RNA-FM code and cuhkaih/rnafm checkpoint, then "
    "construct "
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
    """Return the canonical :class:`BackboneConfig` metadata for RNA-FM T12.

    Formula: pins the official code and checkpoint commits,
    ``frozen_feature_dim=640``, ``max_length=1024``, the checkpoint's
    ``Apache-2.0`` license, and ``contamination_status="unknown"``.  The
    ``downloaded=False`` value preserves this adapter's fail-closed behavior;
    V4 acquired the checkpoint through its separate frozen-cache loader.
    Complexity: ``O(1)``.

    Returns:
        A :class:`BackboneConfig` with RNA-FM manifest defaults.
    """

    return BackboneConfig(
        model_name=MODEL_NAME,
        model_source=MODEL_SOURCE,
        model_revision=MODEL_REVISION,
        license=LICENSE,
        weights_sha256="",
        code_revision=CODE_REVISION,
        tokenizer="rna-fm-bpe",
        max_length=MAX_LENGTH,
        contamination_status=CONTAMINATION_STATUS,
        frozen_feature_dim=EXPECTED_SINGLE_DIM,
        pair_feature_dim=0,
        downloaded=False,
    )
