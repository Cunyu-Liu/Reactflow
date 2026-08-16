"""From-scratch foundation backbone for ReactFlow Phase C1-3.

The :class:`FromScratchBackbone` trains an RNA encoder from scratch (no
pretrained weights) using the C1-2 input-embedding building blocks:

- :class:`~reactflow.backbones.embeddings.InputEmbedding` (which composes
  :class:`~reactflow.backbones.embeddings.NucleotideEmbedding` and
  :class:`~reactflow.backbones.embeddings.PositionalEmbedding`) for the
  per-nucleotide single representation.
- :class:`~reactflow.backbones.embeddings.RelativeDistanceEmbedding` for the
  initial per-pair representation.

This backbone has no frozen features and no pretrained weights, so the
operating :class:`BackboneMode` is effectively a no-op for FROZEN and LORA
(there is nothing to freeze and no base to adapt), and
:attr:`BackboneMode.FULL_FINE_TUNE` is the only meaningful mode.  All parameters
are trainable regardless of the requested mode.

Complexity
----------
- Forward (single): ``O(B * L * d_single)``.
- Forward (pair): ``O(B * L^2 * d_pair)``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ..embeddings import (
    InputEmbedding,
    NucleotideEmbedding,
    PAD_INDEX,
    PositionalEmbedding,
    RelativeDistanceEmbedding,
)
from .base import (
    BackboneConfig,
    BackboneMode,
    BackboneOutput,
    FoundationBackbone,
)

MODEL_NAME = "FromScratch"
MODEL_SOURCE = "reactflow:c1-2-embeddings"
LICENSE = "Apache-2.0"
MAX_LENGTH = 1024
CONTAMINATION_STATUS = "not_applicable"


class FromScratchBackbone(FoundationBackbone):
    """Backbone that trains embeddings from scratch (no pretrained weights).

    Single features come from :class:`InputEmbedding`
    (``s_i = LayerNorm(NucleotideEmbedding(idx_i) + PositionalEmbedding(i))``),
    and pair features come from :class:`RelativeDistanceEmbedding`
    (``z_ij = E_bin[bin(|i - j|)]``).  Both stacks are trainable.

    Because there is no pretrained backbone, the operating mode is advisory
    only: :attr:`BackboneMode.FROZEN` and :attr:`BackboneMode.LORA` are no-ops
    (they would either freeze the randomly-initialised embeddings -- pointless --
    or attach adapters to a non-existent frozen base), so this backbone always
    trains all parameters (equivalent to :attr:`BackboneMode.FULL_FINE_TUNE`).

    Args:
        config: :class:`BackboneConfig` for the from-scratch backbone.
        single_dim: per-nucleotide feature dimension ``d_single``.
        pair_dim: per-pair feature dimension ``d_pair``.
        mode: operating :class:`BackboneMode` (accepted but FROZEN/LORA are
            no-ops; FULL_FINE_TUNE is the effective behavior).
        max_len: maximum sequence length for positional embeddings.
        use_onehot: whether the nucleotide embedding concatenates a one-hot.
        learnable_pos: whether the positional embedding is learnable.

    Complexity: construction ``O(d_single^2)``; forward as documented above.
    """

    def __init__(
        self,
        config: BackboneConfig,
        *,
        single_dim: int = 64,
        pair_dim: int = 64,
        mode: BackboneMode = BackboneMode.FULL_FINE_TUNE,
        max_len: int = MAX_LENGTH,
        use_onehot: bool = False,
        learnable_pos: bool = False,
    ) -> None:
        super().__init__(config, mode)
        if config.frozen_feature_dim == 0:
            config.frozen_feature_dim = single_dim
        if config.pair_feature_dim == 0:
            config.pair_feature_dim = pair_dim

        # C1-2 input embedding: nucleotide + positional, composed via LayerNorm.
        # ``single_dim`` / ``pair_dim`` are read via the base-class properties
        # (which proxy to ``config.frozen_feature_dim`` / ``config.pair_feature_dim``).
        self.input_embedding = InputEmbedding(
            single_dim=self.single_dim,
            max_len=max_len,
            use_onehot=use_onehot,
            learnable_pos=learnable_pos,
            frozen_feature_dim=0,
        )
        # Relative-distance pair initialisation.
        self.pair_embedding = RelativeDistanceEmbedding(self.pair_dim)

        # NOTE: _apply_mode is intentionally NOT called for FROZEN/LORA because
        # those modes are no-ops for a from-scratch model (freezing random
        # embeddings or adapting a non-existent frozen base is pointless).
        # All parameters stay trainable (full-FT behavior).
        for p in self.parameters():
            p.requires_grad_(True)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        record_ids: Optional[Sequence[str]] = None,
    ) -> BackboneOutput:
        """Compute single and pair features from nucleotide indices.

        Args:
            indices: LongTensor of shape ``(B, L)`` with nucleotide vocab
                indices.
            mask: optional BoolTensor of shape ``(B, L)``.  If ``None``, all
                non-padding positions are treated as real.
            record_ids: ignored (no cached features).

        Returns:
            :class:`BackboneOutput` with ``single_features`` ``(B, L, d_single)``
            and ``pair_features`` ``(B, L, L, d_pair)``.

        Complexity: ``O(B*L*d_single)`` for singles, ``O(B*L^2*d_pair)`` for
        pairs.
        """

        B, L = indices.shape
        device = indices.device

        if mask is None:
            mask = indices != PAD_INDEX
        mask = mask.to(device=device)

        # Single representation: nucleotide + positional embedding, LayerNorm'd.
        single = self.input_embedding(indices)  # (B, L, d_single)
        single = single * mask.unsqueeze(-1).to(single.dtype)

        # Pair representation: relative-distance embedding for an L x L grid.
        pair = self.pair_embedding(L, device=device)  # (L, L, d_pair)
        pair = pair.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        # Mask out pairs involving padding positions and the diagonal.
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).to(pair.dtype)
        pair = pair * pair_mask.unsqueeze(-1)
        diag = torch.eye(L, dtype=torch.bool, device=device)
        pair = pair * (~diag).unsqueeze(0).unsqueeze(-1).to(pair.dtype)

        return BackboneOutput(
            single_features=single,
            pair_features=pair,
            mask=mask,
        )

    def get_provenance(self) -> BackboneConfig:
        """Return the :class:`BackboneConfig` for the from-scratch backbone.

        Complexity: ``O(1)``.
        """

        return self._config


def default_config(
    *,
    single_dim: int = 64,
    pair_dim: int = 64,
) -> BackboneConfig:
    """Return the canonical :class:`BackboneConfig` for the from-scratch backbone.

    Formula: pins ``downloaded=True`` (always available, no external weights),
    ``contamination_status="not_applicable"`` (no external training data), and
    the ReactFlow C1-2 embeddings as ``model_source``.  Complexity: ``O(1)``.

    Args:
        single_dim: per-nucleotide feature dimension.
        pair_dim: per-pair feature dimension.

    Returns:
        A :class:`BackboneConfig` with from-scratch defaults.
    """

    return BackboneConfig(
        model_name=MODEL_NAME,
        model_source=MODEL_SOURCE,
        model_revision="c1-2",
        license=LICENSE,
        weights_sha256="",
        code_revision="c1-2",
        tokenizer="nucleotide",
        max_length=MAX_LENGTH,
        contamination_status=CONTAMINATION_STATUS,
        frozen_feature_dim=single_dim,
        pair_feature_dim=pair_dim,
        downloaded=True,
    )
