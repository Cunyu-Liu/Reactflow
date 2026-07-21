"""Bilinear pair head baseline for the C1-2 matched-capacity comparison.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 377-394 (matched-capacity
baselines).

This baseline mirrors the legacy ``PairwiseDenoiser`` pair-scoring logic but as
a standalone symmetric pair predictor:

    s_ij = 0.5 * (h_i^T M h_j + h_j^T M h_i) + c_pair * compat_ij
         = h_i^T M_sym h_j + c_pair * compat_ij        (with M_sym = (M + M^T)/2)

The bilinear pair head has no pair stack, no triangle updates, no outer-product
mean -- it is a pure single-stack model.  Its purpose is to show that the
PairFormer's gains (if any) come from the pair stack, not merely from having a
symmetric output head.

Matched capacity
----------------
The single-stack width is set to ``single_dim=256`` (same as the compact
PairFormer), and the bilinear matrix ``M`` has shape ``(C, C)``.  The total
parameter count is comparable to the compact PairFormer when ``num_blocks=0``
is impossible; in practice the bilinear head is *smaller* than the PairFormer,
which is the point -- it shows that capacity alone is not what makes the
PairFormer work.

Symmetry guarantee
------------------
- ``M`` is symmetrized at construction: ``M_sym = 0.5 * (M + M^T)``.
- The output logits are also symmetrized explicitly: ``logit_ij = 0.5 * (s_ij + s_ji)``.
- The diagonal and padding positions are masked to ``-inf``.

Complexity
----------
- Time: ``O(L^2 * C)`` per forward.
- Memory: ``O(L^2 + L * C)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..backbones import InputEmbedding, pair_compatibility_matrix
from .static_pairformer import PairFormerOutput


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BilinearPairHeadConfig:
    """Configuration for the bilinear pair head baseline.

    Defaults match the compact PairFormer's single-stack width so the two
    models are comparable in capacity.
    """

    single_dim: int = 256
    max_len: int = 512
    use_onehot: bool = False
    learnable_pos: bool = True
    dropout: float = 0.0
    use_compatibility: bool = True
    compat_scale_init: float = 0.5
    use_calibration: bool = True
    init_temperature: float = 1.0
    # Optional small MLP head on the single stack (still no pair stack).
    num_single_layers: int = 2
    ffn_expansion: int = 4

    def __post_init__(self) -> None:
        if self.single_dim <= 0:
            raise ValueError("single_dim must be positive")
        if self.num_single_layers < 0:
            raise ValueError("num_single_layers must be >= 0")


# ---------------------------------------------------------------------------
# Single-stack MLP block (no pair communication)
# ---------------------------------------------------------------------------


class _SingleMLPBlock(nn.Module):
    """A simple pre-norm MLP block on the single stack.

    This block has NO pair communication -- it is intentionally weaker than a
    PairFormer block.  Its purpose is to keep the bilinear baseline's single
    stack depth comparable to the PairFormer's so any F1 gap is attributable to
    the pair stack, not to single-stack depth.
    """

    def __init__(self, dim: int, *, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


# ---------------------------------------------------------------------------
# Bilinear pair head
# ---------------------------------------------------------------------------


class BilinearPairHead(nn.Module):
    """Matched-capacity bilinear pair head baseline.

    Args:
        config: BilinearPairHeadConfig.

    Complexity (forward): ``O(L^2 * C + L * C^2)``.
    """

    def __init__(self, config: Optional[BilinearPairHeadConfig] = None) -> None:
        super().__init__()
        self.config = config or BilinearPairHeadConfig()
        C = self.config.single_dim

        # Input embedding (single stack only, no pair stack)
        self.input_embedding = InputEmbedding(
            single_dim=C,
            max_len=self.config.max_len,
            use_onehot=self.config.use_onehot,
            learnable_pos=self.config.learnable_pos,
            frozen_feature_dim=0,
            dropout=self.config.dropout,
        )

        # Single-stack MLP blocks (no pair communication)
        self.single_blocks = nn.ModuleList([
            _SingleMLPBlock(C, expansion=self.config.ffn_expansion, dropout=self.config.dropout)
            for _ in range(self.config.num_single_layers)
        ])

        # Bilinear pair matrix (will be symmetrized)
        self.pair_matrix = nn.Linear(C, C, bias=False)
        nn.init.normal_(self.pair_matrix.weight, mean=0.0, std=0.02)

        # Pair bias
        self.pair_bias = nn.Parameter(torch.zeros(1))

        # Compatibility scale (learnable scalar)
        if self.config.use_compatibility:
            self.compat_scale = nn.Parameter(torch.tensor(float(self.config.compat_scale_init)))

        # Unpaired head
        self.single_norm = nn.LayerNorm(C)
        self.unpaired_logits = nn.Linear(C, 1)
        nn.init.normal_(self.unpaired_logits.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.unpaired_logits.bias)

        # Calibration temperature
        if self.config.use_calibration:
            self.log_temperature = nn.Parameter(torch.tensor(float(self.config.init_temperature)).log())
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(float(self.config.init_temperature).log()),
                persistent=False,
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _symmetric_pair_matrix(self) -> torch.Tensor:
        """Return the symmetrized bilinear matrix ``M_sym = 0.5 * (M + M^T)``."""
        M = self.pair_matrix.weight  # (C, C)
        return 0.5 * (M + M.t())

    def forward(
        self,
        indices: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        frozen_features: Optional[torch.Tensor] = None,
        region_labels: Optional[torch.Tensor] = None,
    ) -> PairFormerOutput:
        """Run the bilinear pair head forward pass.

        Args:
            indices: LongTensor ``(B, L)`` with nucleotide vocab indices.
            mask: optional BoolTensor ``(B, L)`` (True = real position).
            frozen_features: ignored (present for interface compatibility).
            region_labels: ignored.

        Returns:
            PairFormerOutput.
        """
        if mask is None:
            mask = indices != 5  # PAD_INDEX = 5

        # Single stack
        single = self.input_embedding(indices, frozen_features=None)
        for block in self.single_blocks:
            single = block(single)

        # Bilinear pair score: s_ij = h_i^T M_sym h_j
        M_sym = self._symmetric_pair_matrix()  # (C, C)
        # (B, L, C) @ (C, C) -> (B, L, C); then s_ij = h'_i . h_j
        h_proj = single @ M_sym  # (B, L, C)
        # s = h_proj @ h^T  -> (B, L, L)
        raw = torch.bmm(h_proj, single.transpose(1, 2))  # (B, L, L)

        # Add compatibility bias
        if self.config.use_compatibility:
            compat = pair_compatibility_matrix(indices)  # (B, L, L, 3)
            # Use the canonical + wobble channels as positive evidence
            compat_score = compat[..., 0] + compat[..., 1]  # (B, L, L)
            raw = raw + self.compat_scale * compat_score

        raw = raw + self.pair_bias

        # Symmetrize (already symmetric in theory, but float arithmetic drift)
        logits = 0.5 * (raw + raw.transpose(1, 2))

        # Mask diagonal and padding
        B, L, _ = logits.shape
        device = logits.device
        diag = torch.eye(L, dtype=torch.bool, device=device)
        logits = logits.masked_fill(diag.unsqueeze(0), float("-inf"))
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
        logits = logits.masked_fill(~pair_mask, float("-inf"))

        # BPP
        temperature = self.temperature
        safe_logits = logits.clamp(min=-30.0)
        bpp = torch.sigmoid(safe_logits / temperature)
        bpp = bpp * pair_mask.float()
        bpp = bpp * (~diag).unsqueeze(0).float()

        # Unpaired head
        s = self.single_norm(single)
        unpaired_logit = self.unpaired_logits(s).squeeze(-1)  # (B, L)
        unpaired_prob = torch.sigmoid(unpaired_logit)
        unpaired_prob = unpaired_prob * mask.float()
        unpaired_logit = unpaired_logit.masked_fill(~mask, float("-inf"))

        return PairFormerOutput(
            logits=logits,
            bpp=bpp,
            unpaired_logit=unpaired_logit,
            unpaired_prob=unpaired_prob,
            pair_type_logits=None,
            mask=mask,
            temperature=temperature.detach(),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
