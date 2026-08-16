"""Static PairFormer model for ReactFlow 2.0 RNA secondary structure prediction.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 332-452.

This module implements the compact PairFormer backbone that directly
predicts symmetric ``L x L`` base-pair logits/probabilities, along with
per-position unpaired probabilities and an optional pair-type head.

Architecture
------------
1. **Input embedding** (single): nucleotide + positional encoding.
2. **Pair initialization** (symmetric): outer product + distance + compatibility.
3. **PairFormer blocks** (8-12): each block applies
   - Triangle multiplicative update (pair stack)
   - Triangle attention (pair stack)
   - Outer product mean (single -> pair)
   - Pair-to-single attention (pair -> single)
   - Single row attention (single stack)
   - Pair transition (FFN)
   - Single transition (FFN)
4. **Output head**: symmetric pair logits + unpaired logits, with a learned
   calibration temperature and a final BPP = sigmoid(logits / temperature).

Symmetry guarantee
------------------
- Pair initialization is symmetric by construction (see SymmetricPairInit).
- Triangle updates and attention symmetrize internally.
- OuterProductMean symmetrizes its update.
- The output head produces ``logit_ij = 0.5 * (raw_ij + raw_ji)``.

Complexity
----------
- Time: ``O(L^3 * P + L^2 * C)`` per block.
- Memory: ``O(L^2 * P + L * C)`` per block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn

from ..backbones import (
    InputEmbedding,
    SymmetricPairInit,
    TriangleMultiplicativeUpdate,
    TriangleAttention,
    PairTransition,
    OuterProductMean,
    PairToSingleAttention,
    SingleRowAttention,
    SingleTransition,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PairFormerConfig:
    """Configuration for the compact StaticPairFormer.

    Defaults follow the spec: 8-12 blocks, single width 256, pair width 64-128.
    """

    # Dimensions
    single_dim: int = 256
    pair_dim: int = 64
    max_len: int = 512

    # Embedding
    use_onehot: bool = False
    learnable_pos: bool = True
    frozen_feature_dim: int = 0
    dropout: float = 0.0

    # Frozen feature pair fusion (C1-3: pair-aware fusion)
    # When True, frozen features are also used to initialize the pair stack
    # via OuterProductMean, in addition to being concatenated to the single
    # embedding.  When False (single_only), frozen features only affect the
    # single embedding.
    frozen_pair_fusion: bool = False

    # PairFormer block
    num_blocks: int = 8
    num_heads_pair: int = 4
    num_heads_single: int = 8
    triangle_hidden_dim: Optional[int] = None
    ffn_expansion: int = 4
    block_dropout: float = 0.0
    outer_product_mean_dim: int = 16  # small to keep OPM memory O(B*L*L*D*P) not O(B*L*L*D^2)

    # Output head
    num_pair_types: int = 0  # 0 disables pair-type head
    use_calibration: bool = True
    init_temperature: float = 1.0

    # Symmetry
    share_pair_init_projection: bool = True

    # Region labels (optional)
    num_region_labels: int = 0

    def __post_init__(self) -> None:
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.single_dim <= 0 or self.pair_dim <= 0:
            raise ValueError("single_dim and pair_dim must be positive")
        if self.single_dim % self.num_heads_single != 0:
            # Adjust head_dim; allowed, just warn via assertion
            pass


# ---------------------------------------------------------------------------
# PairFormer block
# ---------------------------------------------------------------------------


class PairFormerBlock(nn.Module):
    """One PairFormer block updating both single and pair stacks.

    Order of operations (per spec lines 371-376):
    1. Triangle multiplicative update (pair)
    2. Triangle attention (pair)
    3. Outer product mean (single -> pair)
    4. Pair transition (pair FFN)
    5. Pair-to-single attention (pair -> single)
    6. Single row attention (single)
    7. Single transition (single FFN)

    Args:
        config: PairFormerConfig.

    Complexity: ``O(L^3 * P + L^2 * C)``.
    """

    def __init__(self, config: PairFormerConfig) -> None:
        super().__init__()
        self.config = config
        C = config.single_dim
        P = config.pair_dim

        # Pair stack updates
        self.triangle_mult = TriangleMultiplicativeUpdate(
            P, hidden_dim=config.triangle_hidden_dim, dropout=config.block_dropout,
        )
        self.triangle_attn = TriangleAttention(
            P, num_heads=config.num_heads_pair, dropout=config.block_dropout,
        )
        self.outer_product_mean = OuterProductMean(C, P, projection_dim=config.outer_product_mean_dim)
        self.pair_transition = PairTransition(P, expansion=config.ffn_expansion, dropout=config.block_dropout)

        # Single stack updates
        self.pair_to_single = PairToSingleAttention(
            C, P, num_heads=config.num_heads_single, dropout=config.block_dropout,
        )
        self.single_attn = SingleRowAttention(
            C, num_heads=config.num_heads_single, dropout=config.block_dropout,
        )
        self.single_transition = SingleTransition(C, expansion=config.ffn_expansion, dropout=config.block_dropout)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one PairFormer block.

        Args:
            single: ``(B, L, C)``.
            pair: ``(B, L, L, P)``.
            mask: optional ``(B, L)``.

        Returns:
            (single', pair') with same shapes.
        """
        # Pair stack
        pair = self.triangle_mult(pair, mask=mask)
        pair = self.triangle_attn(pair, mask=mask)
        pair = self.outer_product_mean(single, pair, mask=mask)
        pair = self.pair_transition(pair)

        # Single stack
        single = self.pair_to_single(single, pair, mask=mask)
        single = self.single_attn(single, mask=mask)
        single = self.single_transition(single)

        return single, pair


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------


class PairOutputHead(nn.Module):
    """Symmetric pair logits + unpaired logits + optional pair-type logits.

    Formula:
        raw_ij = Linear(z_ij)
        logit_ij = 0.5 * (raw_ij + raw_ji)   # symmetrize
        logit_ii = -inf  (diagonal masked)
        unpaired_i = Linear(s_i)
        pair_type_ij = Linear(z_ij) -> num_pair_types logits (optional)
        BPP_ij = sigmoid(logit_ij / temperature)
        unpaired_prob_i = sigmoid(unpaired_i)

    Args:
        pair_dim: pair representation dimension.
        single_dim: single representation dimension.
        num_pair_types: 0 to disable pair-type head.
        use_calibration: learn a temperature parameter.

    Complexity: ``O(L^2 * P)``.
    """

    def __init__(
        self,
        pair_dim: int,
        single_dim: int,
        *,
        num_pair_types: int = 0,
        use_calibration: bool = True,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.single_dim = single_dim
        self.num_pair_types = num_pair_types
        self.use_calibration = use_calibration

        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_logits = nn.Linear(pair_dim, 1)
        nn.init.normal_(self.pair_logits.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pair_logits.bias)

        self.single_norm = nn.LayerNorm(single_dim)
        self.unpaired_logits = nn.Linear(single_dim, 1)
        nn.init.normal_(self.unpaired_logits.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.unpaired_logits.bias)

        if num_pair_types > 0:
            self.pair_type_logits = nn.Linear(pair_dim, num_pair_types)
            nn.init.normal_(self.pair_type_logits.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.pair_type_logits.bias)

        if use_calibration:
            # log-temperature parameter initialized so that exp(log_temp) = init_temperature
            # Use 1D tensor (numel=1) for FSDP compatibility (FSDP doesn't support scalars)
            self.log_temperature = nn.Parameter(torch.tensor([float(init_temperature)]).log())
        else:
            self.register_buffer("log_temperature", torch.tensor([float(init_temperature)]).log(), persistent=False)

    @property
    def temperature(self) -> torch.Tensor:
        """Return the calibration temperature (always positive)."""
        return self.log_temperature.exp()

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> "PairFormerOutput":
        """Compute the output head.

        Args:
            single: ``(B, L, C)``.
            pair: ``(B, L, L, P)``.
            mask: optional ``(B, L)`` real-position mask.

        Returns:
            PairFormerOutput with logits, bpp, unpaired_prob, mask.
        """
        B, L, _ = single.shape
        device = single.device

        # Pair logits (symmetric)
        z = self.pair_norm(pair)
        raw = self.pair_logits(z).squeeze(-1)  # (B, L, L)
        # Symmetrize
        logits = 0.5 * (raw + raw.transpose(1, 2))

        # Mask diagonal and padding
        diag = torch.eye(L, dtype=torch.bool, device=device)
        logits = logits.masked_fill(diag.unsqueeze(0), float("-inf"))
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, L, L)
            logits = logits.masked_fill(~pair_mask, float("-inf"))

        # BPP
        temperature = self.temperature
        # Use a numerically stable sigmoid of logits / temperature
        # When logits = -inf (masked), BPP should be 0
        safe_logits = logits.clamp(min=-30.0)
        bpp = torch.sigmoid(safe_logits / temperature)
        if mask is not None:
            bpp = bpp * pair_mask.float()
        bpp = bpp * (~diag).unsqueeze(0).float()

        # Unpaired logits
        s = self.single_norm(single)
        unpaired_logit = self.unpaired_logits(s).squeeze(-1)  # (B, L)
        unpaired_prob = torch.sigmoid(unpaired_logit)
        if mask is not None:
            unpaired_prob = unpaired_prob * mask.float()
            unpaired_logit = unpaired_logit.masked_fill(~mask, float("-inf"))

        # Pair-type logits (optional)
        pair_type_logits = None
        if self.num_pair_types > 0:
            pair_type_logits = self.pair_type_logits(z)  # (B, L, L, num_pair_types)
            # Symmetrize
            pair_type_logits = 0.5 * (pair_type_logits + pair_type_logits.transpose(1, 2))
            if mask is not None:
                pair_type_logits = pair_type_logits * pair_mask.unsqueeze(-1).float()
            pair_type_logits = pair_type_logits * (~diag).unsqueeze(0).unsqueeze(-1).float()

        return PairFormerOutput(
            logits=logits,
            bpp=bpp,
            unpaired_logit=unpaired_logit,
            unpaired_prob=unpaired_prob,
            pair_type_logits=pair_type_logits,
            mask=mask,
            temperature=temperature.detach(),
        )


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class PairFormerOutput:
    """Container for the StaticPairFormer output.

    Attributes:
        logits: pair logits ``(B, L, L)`` (symmetric, diagonal = -inf).
        bpp: base-pair probabilities ``(B, L, L)`` (sigmoid(logits/temp)).
        unpaired_logit: per-position unpaired logits ``(B, L)``.
        unpaired_prob: per-position unpaired probability ``(B, L)``.
        pair_type_logits: optional ``(B, L, L, num_pair_types)``.
        mask: optional ``(B, L)`` real-position mask.
        temperature: calibration temperature (scalar).
    """

    logits: torch.Tensor
    bpp: torch.Tensor
    unpaired_logit: torch.Tensor
    unpaired_prob: torch.Tensor
    pair_type_logits: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    temperature: torch.Tensor

    def symmetry_residual(self) -> torch.Tensor:
        """Return the symmetry residual ``||logit_ij - logit_ji||_2`` (should be ~0).

        Masked cells (diagonal, padding) have logits = -inf; the diff there
        would be ``(-inf) - (-inf) = NaN``, so we use ``torch.where`` to
        replace those entries with 0 before computing the residual.
        """
        diff = self.logits - self.logits.transpose(1, 2)
        # Ignore -inf diagonal / padding (diff there would be NaN)
        finite = torch.isfinite(diff)
        diff_safe = torch.where(finite, diff, torch.zeros_like(diff))
        return diff_safe.pow(2).sum(dim=[1, 2]).sqrt()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class StaticPairFormer(nn.Module):
    """Compact PairFormer for RNA secondary structure prediction.

    This is the main C1-2 model.  It takes a batch of RNA sequences and
    produces symmetric pair logits, BPP, and unpaired probabilities.

    Args:
        config: PairFormerConfig.

    Complexity (forward):
        - Time: ``O(num_blocks * (L^3 * P + L^2 * C))``.
        - Memory: ``O(num_blocks * (L^2 * P + L * C))``.
    """

    def __init__(self, config: Optional[PairFormerConfig] = None) -> None:
        super().__init__()
        self.config = config or PairFormerConfig()
        C = self.config.single_dim
        P = self.config.pair_dim
        self.gradient_checkpointing: bool = False

        # Input embedding
        self.input_embedding = InputEmbedding(
            single_dim=C,
            max_len=self.config.max_len,
            use_onehot=self.config.use_onehot,
            learnable_pos=self.config.learnable_pos,
            frozen_feature_dim=self.config.frozen_feature_dim,
            dropout=self.config.dropout,
        )

        # Optional region label embedding
        if self.config.num_region_labels > 0:
            self.region_embedding = nn.Embedding(self.config.num_region_labels, C)
            nn.init.normal_(self.region_embedding.weight, mean=0.0, std=0.02)

        # Pair initialization
        self.pair_init = SymmetricPairInit(
            single_dim=C,
            pair_dim=P,
            share_projection=self.config.share_pair_init_projection,
            use_distance_embedding=True,
            use_compatibility=True,
        )

        # Frozen feature pair fusion: when enabled, frozen features also
        # contribute to pair initialization via OuterProductMean.
        self.frozen_pair_fusion = self.config.frozen_pair_fusion and self.config.frozen_feature_dim > 0
        if self.frozen_pair_fusion:
            self.frozen_opm = OuterProductMean(
                self.config.frozen_feature_dim, P,
                projection_dim=min(self.config.frozen_feature_dim, 32),
            )

        # PairFormer blocks
        self.blocks = nn.ModuleList([
            PairFormerBlock(self.config) for _ in range(self.config.num_blocks)
        ])

        # Output head
        self.output_head = PairOutputHead(
            pair_dim=P,
            single_dim=C,
            num_pair_types=self.config.num_pair_types,
            use_calibration=self.config.use_calibration,
            init_temperature=self.config.init_temperature,
        )

    def forward(
        self,
        indices: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        frozen_features: Optional[torch.Tensor] = None,
        region_labels: Optional[torch.Tensor] = None,
    ) -> PairFormerOutput:
        """Run the PairFormer forward pass.

        Args:
            indices: LongTensor ``(B, L)`` with nucleotide vocab indices.
            mask: optional BoolTensor ``(B, L)`` (True = real position).
            frozen_features: optional FloatTensor ``(B, L, frozen_feature_dim)``.
            region_labels: optional LongTensor ``(B, L)`` with region indices.

        Returns:
            PairFormerOutput.
        """
        if mask is None:
            mask = indices != 5  # PAD_INDEX = 5

        # Single embedding
        single = self.input_embedding(indices, frozen_features=frozen_features)
        if region_labels is not None and self.config.num_region_labels > 0:
            single = single + self.region_embedding(region_labels)

        # Pair initialization
        pair = self.pair_init(single, indices, mask=mask)

        # Frozen feature pair fusion: add OPM(frozen_features) to pair init
        if self.frozen_pair_fusion and frozen_features is not None:
            pair = pair + self.frozen_opm(frozen_features, pair, mask=mask)

        # PairFormer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                import torch.utils.checkpoint as cp
                def _run_blk(s, p, m, blk=block):
                    return blk(s, p, mask=m)
                single, pair = cp.checkpoint(_run_blk, single, pair, mask, use_reentrant=False)
            else:
                single, pair = block(single, pair, mask=mask)

        # Output head
        output = self.output_head(single, pair, mask=mask)
        return output

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing to reduce memory at the cost of compute."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
