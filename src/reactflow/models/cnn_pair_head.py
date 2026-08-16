"""CNN and UNet pair head baselines for the C1-2 matched-capacity comparison.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 377-394 (matched-capacity
baselines).

These baselines use a 1D CNN over the sequence to produce single features,
then form an explicit pair representation and refine it with a 2D CNN
(``CNNPairHead``) or a UNet-style encoder-decoder over the pair matrix
(``UNetPairHead``).

Unlike the bilinear baseline, both CNN variants DO have an explicit pair
representation that gets refined -- but the refinement is purely local
(convolutional), with no triangle updates or outer-product-mean communication
between single and pair stacks.  They isolate the contribution of the
PairFormer's *triangle* operations from generic "have a pair stack" capacity.

Symmetry guarantee
------------------
- Pair initialization uses a symmetric outer-product-mean-style update.
- 2D convolutions over the pair matrix are symmetrized by averaging the
  output with its transpose after each conv block.
- The output logits are explicitly symmetrized: ``logit_ij = 0.5 * (s_ij + s_ji)``.
- The diagonal and padding positions are masked to ``-inf``.

Complexity
----------
- ``CNNPairHead``: time ``O(L^2 * P * K^2 * num_layers)``, memory ``O(L^2 * P)``.
- ``UNetPairHead``: time ``O(L^2 * P * K^2 * num_layers)``, memory ``O(L^2 * P)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ..backbones import InputEmbedding, pair_compatibility_matrix, RelativeDistanceEmbedding
from .static_pairformer import PairFormerOutput


def _largest_group_divisor(channels: int, max_groups: int = 8) -> int:
    """Return the largest divisor of ``channels`` that is ``<= max_groups``.

    GroupNorm requires ``num_channels % num_groups == 0``.  We pick the
    largest power-of-2 group count up to ``max_groups`` that divides
    ``channels``, falling back to 1 (instance norm) if none works.
    """
    for g in (8, 4, 2, 1):
        if g <= max_groups and channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CNNPairHeadConfig:
    """Configuration for the CNN pair head baseline."""

    single_dim: int = 256
    pair_dim: int = 64
    max_len: int = 512
    use_onehot: bool = False
    learnable_pos: bool = True
    dropout: float = 0.0

    # 1D CNN over sequence
    num_single_layers: int = 3
    single_kernel_size: int = 9

    # 2D CNN over pair matrix
    num_pair_layers: int = 4
    pair_kernel_size: int = 5
    pair_hidden_dim: int = 64

    use_compatibility: bool = True
    use_distance_embedding: bool = True
    use_calibration: bool = True
    init_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.single_dim <= 0 or self.pair_dim <= 0:
            raise ValueError("single_dim and pair_dim must be positive")
        if self.num_single_layers < 0 or self.num_pair_layers < 0:
            raise ValueError("layer counts must be >= 0")


@dataclass
class UNetPairHeadConfig:
    """Configuration for the UNet pair head baseline.

    The UNet has ``num_levels`` down/up-sampling stages, each halving the
    spatial resolution of the pair matrix.  At the bottom of the U, a bottleneck
    block operates at reduced resolution.  Skip connections concatenate
    encoder features into the decoder.
    """

    single_dim: int = 256
    pair_dim: int = 64
    max_len: int = 512
    use_onehot: bool = False
    learnable_pos: bool = True
    dropout: float = 0.0

    num_single_layers: int = 2
    single_kernel_size: int = 9

    # UNet pair stack
    num_levels: int = 3
    pair_kernel_size: int = 5
    pair_hidden_dim: int = 64

    use_compatibility: bool = True
    use_distance_embedding: bool = True
    use_calibration: bool = True
    init_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.single_dim <= 0 or self.pair_dim <= 0:
            raise ValueError("single_dim and pair_dim must be positive")
        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1")


# ---------------------------------------------------------------------------
# 1D CNN single block (symmetric padding)
# ---------------------------------------------------------------------------


class _SingleConvBlock(nn.Module):
    """1D conv + GELU + residual on the single stack.

    Pads symmetrically so that the output length equals the input length.
    """

    def __init__(self, dim: int, *, kernel_size: int = 9, dropout: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y = self.norm(x)
        y = y.transpose(1, 2)  # (B, C, L)
        y = self.conv(y)
        y = self.act(y)
        y = self.drop(y)
        y = y.transpose(1, 2)  # (B, L, C)
        return x + y


# ---------------------------------------------------------------------------
# 2D CNN pair block (symmetric by averaging with transpose)
# ---------------------------------------------------------------------------


class _PairConvBlock(nn.Module):
    """2D conv + GELU + residual on the pair stack, symmetrized.

    Input/output: (B, P, L, L).  After each conv we average the output with
    its transpose (over the last two spatial dims) so the pair representation
    remains symmetric.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        kernel_size: int = 5,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual and (in_dim == out_dim)
        pad = kernel_size // 2
        # GroupNorm requires num_groups to divide num_channels.
        num_groups = _largest_group_divisor(in_dim)
        self.norm = nn.GroupNorm(num_groups, in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, L, L)
        y = self.norm(x)
        y = self.conv(y)
        # Symmetrize: average with transpose over spatial dims
        y = 0.5 * (y + y.transpose(-1, -2))
        y = self.act(y)
        y = self.drop(y)
        if self.residual:
            return x + y
        return y


# ---------------------------------------------------------------------------
# Pair initialization (outer product mean style, symmetric)
# ---------------------------------------------------------------------------


class _PairInitConv(nn.Module):
    """Symmetric pair initialization from single features.

    Forms ``z_ij = Linear(a_i ⊗ b_j) + dist_emb + compat`` and symmetrizes.
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        *,
        use_distance_embedding: bool = True,
        use_compatibility: bool = True,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.use_distance_embedding = use_distance_embedding
        self.use_compatibility = use_compatibility

        # Project single to two smaller "factor" dims to keep outer product small
        self.factor_dim = max(8, int(single_dim ** 0.5))
        self.proj_a = nn.Linear(single_dim, self.factor_dim)
        self.proj_b = nn.Linear(single_dim, self.factor_dim)
        self.out_proj = nn.Linear(self.factor_dim * self.factor_dim, pair_dim)

        if use_distance_embedding:
            self.dist_emb = RelativeDistanceEmbedding(pair_dim)

        if use_compatibility:
            self.compat_proj = nn.Linear(3, pair_dim)

    def forward(
        self,
        single: torch.Tensor,
        indices: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(B, L, L, P)`` symmetric pair tensor."""
        B, L, C = single.shape
        a = self.proj_a(single)  # (B, L, F)
        b = self.proj_b(single)  # (B, L, F)
        # Outer product: (B, L, L, F*F)
        outer = torch.einsum("bid,bje->bijde", a, b)
        outer = outer.reshape(B, L, L, self.factor_dim * self.factor_dim)
        z = self.out_proj(outer)  # (B, L, L, P)

        # Symmetrize
        z = 0.5 * (z + z.transpose(1, 2))

        if self.use_distance_embedding:
            z = z + self.dist_emb(L, device=single.device)

        if self.use_compatibility:
            compat = pair_compatibility_matrix(indices)  # (B, L, L, 3)
            z = z + self.compat_proj(compat)

        # Zero out diagonal and padding
        device = single.device
        diag = torch.eye(L, dtype=torch.bool, device=device)
        z = z * (~diag).unsqueeze(0).unsqueeze(-1).float()
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            z = z * pair_mask.unsqueeze(-1).float()

        return z


# ---------------------------------------------------------------------------
# Output head (shared by CNN and UNet variants)
# ---------------------------------------------------------------------------


class _PairConvOutputHead(nn.Module):
    """Output head producing PairFormerOutput from a pair tensor.

    Args:
        pair_dim: pair representation channels.
        single_dim: single representation width.
        use_calibration: learn a temperature parameter.
    """

    def __init__(
        self,
        pair_dim: int,
        single_dim: int,
        *,
        use_calibration: bool = True,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_logits = nn.Linear(pair_dim, 1)
        nn.init.normal_(self.pair_logits.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pair_logits.bias)

        self.single_norm = nn.LayerNorm(single_dim)
        self.unpaired_logits = nn.Linear(single_dim, 1)
        nn.init.normal_(self.unpaired_logits.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.unpaired_logits.bias)

        if use_calibration:
            self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(float(init_temperature).log()),
                persistent=False,
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> PairFormerOutput:
        B, L, _ = single.shape
        device = single.device

        z = self.pair_norm(pair)
        raw = self.pair_logits(z).squeeze(-1)  # (B, L, L)
        logits = 0.5 * (raw + raw.transpose(1, 2))

        diag = torch.eye(L, dtype=torch.bool, device=device)
        logits = logits.masked_fill(diag.unsqueeze(0), float("-inf"))
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            logits = logits.masked_fill(~pair_mask, float("-inf"))
        else:
            pair_mask = (~diag).unsqueeze(0)

        temperature = self.temperature
        safe_logits = logits.clamp(min=-30.0)
        bpp = torch.sigmoid(safe_logits / temperature)
        bpp = bpp * pair_mask.float()
        bpp = bpp * (~diag).unsqueeze(0).float()

        s = self.single_norm(single)
        unpaired_logit = self.unpaired_logits(s).squeeze(-1)
        unpaired_prob = torch.sigmoid(unpaired_logit)
        if mask is not None:
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


# ---------------------------------------------------------------------------
# CNN pair head
# ---------------------------------------------------------------------------


class CNNPairHead(nn.Module):
    """Matched-capacity CNN pair head baseline.

    Pipeline:
    1. InputEmbedding -> single (B, L, C)
    2. N x _SingleConvBlock over single
    3. _PairInitConv -> pair (B, L, L, P)
    4. M x _PairConvBlock over pair (2D CNN, symmetrized)
    5. _PairConvOutputHead -> PairFormerOutput

    Args:
        config: CNNPairHeadConfig.

    Complexity (forward): ``O(L^2 * P * K^2 * num_pair_layers + L * C * K * num_single_layers)``.
    """

    def __init__(self, config: Optional[CNNPairHeadConfig] = None) -> None:
        super().__init__()
        self.config = config or CNNPairHeadConfig()
        C = self.config.single_dim
        P = self.config.pair_dim

        self.input_embedding = InputEmbedding(
            single_dim=C,
            max_len=self.config.max_len,
            use_onehot=self.config.use_onehot,
            learnable_pos=self.config.learnable_pos,
            frozen_feature_dim=0,
            dropout=self.config.dropout,
        )

        self.single_blocks = nn.ModuleList([
            _SingleConvBlock(C, kernel_size=self.config.single_kernel_size, dropout=self.config.dropout)
            for _ in range(self.config.num_single_layers)
        ])

        self.pair_init = _PairInitConv(
            single_dim=C,
            pair_dim=P,
            use_distance_embedding=self.config.use_distance_embedding,
            use_compatibility=self.config.use_compatibility,
            max_len=self.config.max_len,
        )

        pair_in = P
        self.pair_blocks = nn.ModuleList()
        for _ in range(self.config.num_pair_layers):
            self.pair_blocks.append(
                _PairConvBlock(
                    pair_in,
                    self.config.pair_hidden_dim,
                    kernel_size=self.config.pair_kernel_size,
                    dropout=self.config.dropout,
                    residual=(pair_in == self.config.pair_hidden_dim),
                )
            )
            pair_in = self.config.pair_hidden_dim

        # If final pair dim differs from config.pair_hidden_dim, nothing more to do;
        # output head takes pair_in as its pair_dim.
        self.output_head = _PairConvOutputHead(
            pair_dim=pair_in,
            single_dim=C,
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
        if mask is None:
            mask = indices != 5

        single = self.input_embedding(indices, frozen_features=None)
        for block in self.single_blocks:
            single = block(single)

        pair = self.pair_init(single, indices, mask=mask)  # (B, L, L, P)

        # Apply 2D conv blocks (expect (B, P_ch, L, L))
        x = pair.permute(0, 3, 1, 2)  # (B, P, L, L)
        for block in self.pair_blocks:
            x = block(x)
        pair_out = x.permute(0, 2, 3, 1)  # (B, L, L, P)

        return self.output_head(single, pair_out, mask=mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# UNet pair head
# ---------------------------------------------------------------------------


class _UNetEncBlock(nn.Module):
    """UNet encoder block: 2 conv layers + downsample."""

    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = _PairConvBlock(in_ch, out_ch, kernel_size=kernel_size, dropout=dropout, residual=False)
        self.conv2 = _PairConvBlock(out_ch, out_ch, kernel_size=kernel_size, dropout=dropout, residual=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        down = self.pool(x)
        return down, skip


class _UNetDecBlock(nn.Module):
    """UNet decoder block: upsample + concat skip + 2 conv layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, *, kernel_size: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv1 = _PairConvBlock(in_ch + skip_ch, out_ch, kernel_size=kernel_size, dropout=dropout, residual=False)
        self.conv2 = _PairConvBlock(out_ch, out_ch, kernel_size=kernel_size, dropout=dropout, residual=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]])
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetPairHead(nn.Module):
    """Matched-capacity UNet pair head baseline.

    Pipeline:
    1. InputEmbedding -> single (B, L, C)
    2. N x _SingleConvBlock over single
    3. _PairInitConv -> pair (B, L, L, P)
    4. UNet encoder-decoder over pair matrix
       - num_levels down/up stages, each halving spatial resolution
       - skip connections concatenate encoder features into decoder
       - all conv layers symmetrized by averaging with transpose
    5. _PairConvOutputHead -> PairFormerOutput

    The UNet forces the pair representation to pass through a bottleneck at
    reduced resolution, which tests whether long-range pair dependencies
    benefit from a multi-scale (rather than flat-triangle) inductive bias.

    Args:
        config: UNetPairHeadConfig.

    Complexity (forward): ``O(L^2 * P * K^2 * num_levels)`` (downsampled levels
    contribute geometrically less, so total is still ``O(L^2 * P * K^2 * num_levels)``).
    """

    def __init__(self, config: Optional[UNetPairHeadConfig] = None) -> None:
        super().__init__()
        self.config = config or UNetPairHeadConfig()
        C = self.config.single_dim
        P = self.config.pair_dim
        H = self.config.pair_hidden_dim

        self.input_embedding = InputEmbedding(
            single_dim=C,
            max_len=self.config.max_len,
            use_onehot=self.config.use_onehot,
            learnable_pos=self.config.learnable_pos,
            frozen_feature_dim=0,
            dropout=self.config.dropout,
        )

        self.single_blocks = nn.ModuleList([
            _SingleConvBlock(C, kernel_size=self.config.single_kernel_size, dropout=self.config.dropout)
            for _ in range(self.config.num_single_layers)
        ])

        self.pair_init = _PairInitConv(
            single_dim=C,
            pair_dim=P,
            use_distance_embedding=self.config.use_distance_embedding,
            use_compatibility=self.config.use_compatibility,
            max_len=self.config.max_len,
        )

        # Project pair_dim -> H channels for the UNet
        self.pair_in_proj = nn.Conv2d(P, H, kernel_size=1)

        # Encoder
        self.encoders = nn.ModuleList()
        channels = [H]
        for _ in range(self.config.num_levels):
            in_ch = channels[-1]
            out_ch = in_ch * 2
            self.encoders.append(_UNetEncBlock(in_ch, out_ch, kernel_size=self.config.pair_kernel_size, dropout=self.config.dropout))
            channels.append(out_ch)

        # Bottleneck
        self.bottleneck = _PairConvBlock(channels[-1], channels[-1], kernel_size=self.config.pair_kernel_size, dropout=self.config.dropout, residual=True)

        # Decoder.  Each decoder i corresponds to encoder level
        # (num_levels - 1 - i); the skip from that encoder has out_ch =
        # channels[num_levels - i] channels (not channels[num_levels - 1 - i]).
        # We halve the channel count at each decoder level so the final decoder
        # outputs H channels (matching pair_in_proj input and the output head).
        self.decoders = nn.ModuleList()
        prev_ch = channels[-1]  # bottleneck output channels
        for i in range(self.config.num_levels):
            # Skip from encoder level (num_levels - 1 - i) has out_ch =
            # channels[num_levels - i] channels.
            skip_ch = channels[self.config.num_levels - i]
            # Halve channels at each decoder level: out_ch = channels level below skip.
            out_ch = channels[self.config.num_levels - 1 - i]
            self.decoders.append(_UNetDecBlock(prev_ch, skip_ch, out_ch, kernel_size=self.config.pair_kernel_size, dropout=self.config.dropout))
            prev_ch = out_ch

        # Output head takes the final decoder output (channels = H)
        self.output_head = _PairConvOutputHead(
            pair_dim=H,
            single_dim=C,
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
        if mask is None:
            mask = indices != 5

        single = self.input_embedding(indices, frozen_features=None)
        for block in self.single_blocks:
            single = block(single)

        pair = self.pair_init(single, indices, mask=mask)  # (B, L, L, P)
        x = pair.permute(0, 3, 1, 2)  # (B, P, L, L)
        x = self.pair_in_proj(x)  # (B, H, L, L)

        # Encoder
        skips: List[torch.Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (reverse order)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        pair_out = x.permute(0, 2, 3, 1)  # (B, L, L, H)
        return self.output_head(single, pair_out, mask=mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
