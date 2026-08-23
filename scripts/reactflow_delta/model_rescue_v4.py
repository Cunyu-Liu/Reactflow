#!/usr/bin/env python3
"""Mutation-conditioned dual-tower models for Model Rescue v4.

The module contains no target-side inputs.  It predicts a full-construct
signed-delta mean from WT inputs, corrected mutation coordinates, and optional
frozen paired RNA-FM embeddings.  Residual calibration is a separate zero-mean
stage whose component locations are both the detached point mean.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v2 import gaussian_mixture_crps_torch
from scripts.reactflow_delta.run_p3_lrso_v3 import ALPHA, SCALE_FLOOR


PRIMARY_CANDIDATE = "v4_dual_tower_rnafm"
SCRATCH_CONTROL = "v4_dual_tower_scratch"
FOUNDATION_ONLY_CONTROL = "v4_rnafm_only"
CAPACITY_NULL = "v4_capacity_matched_sequence_null"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v4_prediction.v1"


@dataclass(frozen=True)
class V4ModelConfig:
    d_model: int = 512
    heads: int = 8
    wt_blocks: int = 5
    response_blocks: int = 5
    ff_dim: int = 2048
    pair_dim: int = 128
    pair_heads: int = 8
    pair_blocks: int = 5
    foundation_dim: int = 640
    dropout: float = 0.10
    max_relative_distance: int = 256
    pair_enabled: bool = True

    def validate(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.pair_enabled and self.pair_dim % self.pair_heads:
            raise ValueError("pair_dim must be divisible by pair_heads")
        if self.pair_enabled and self.pair_blocks != self.wt_blocks:
            raise ValueError("one pair block is required for each WT block")
        if self.foundation_dim < 0:
            raise ValueError("foundation_dim cannot be negative")

    @classmethod
    def primary(cls) -> "V4ModelConfig":
        return cls()

    @classmethod
    def scratch(cls) -> "V4ModelConfig":
        return replace(cls(), foundation_dim=0)

    @classmethod
    def capacity_null(cls) -> "V4ModelConfig":
        return replace(cls(), pair_enabled=False, pair_blocks=0, response_blocks=6)


def mutation_one_hot(
    refs: list[str], alts: list[str], device: torch.device
) -> torch.Tensor:
    if len(refs) != len(alts):
        raise ValueError("refs and alts must have the same length")
    ref_idx = torch.tensor([ALPHA.get(x.replace("T", "U"), 3) for x in refs], device=device)
    alt_idx = torch.tensor([ALPHA.get(x.replace("T", "U"), 3) for x in alts], device=device)
    result = torch.zeros(len(refs), 8, device=device)
    result.scatter_(1, ref_idx[:, None], 1.0)
    result.scatter_(1, alt_idx[:, None] + 4, 1.0)
    return result


class PairBiasedSequenceBlock(nn.Module):
    """Pre-norm self-attention with an optional learned bias from pair state."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        ff_dim: int,
        dropout: float,
        pair_dim: int | None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.pair_bias = nn.Linear(pair_dim, heads, bias=False) if pair_dim else None
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        pair: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, d_model = x.shape
        normalized = self.norm1(x)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if pair is not None:
            if self.pair_bias is None:
                raise ValueError("pair state supplied to a block without pair bias")
            if pair.ndim == 3:
                bias = self.pair_bias(pair).permute(2, 0, 1).unsqueeze(0)
            elif pair.ndim == 4:
                bias = self.pair_bias(pair).permute(0, 3, 1, 2)
            else:
                raise ValueError("pair state must have shape LxLxP or BxLxLxP")
            logits = logits + bias
        if valid_mask is not None:
            if valid_mask.ndim == 1:
                valid_mask = valid_mask.unsqueeze(0).expand(batch, -1)
            logits = logits.masked_fill(
                ~valid_mask[:, None, None, :], torch.finfo(logits.dtype).min
            )
        attention = self.attn_dropout(torch.softmax(logits, dim=-1))
        context = torch.matmul(attention, v).transpose(1, 2).reshape(
            batch, length, d_model
        )
        x = x + self.residual_dropout(self.out(context))
        return x + self.residual_dropout(self.ff(self.norm2(x)))


class PairAxialBlock(nn.Module):
    """Row attention, column attention, and transition over an RNA pair field."""

    def __init__(self, pair_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.row_norm = nn.LayerNorm(pair_dim)
        self.row_attention = nn.MultiheadAttention(
            pair_dim, heads, dropout=dropout, batch_first=True
        )
        self.column_norm = nn.LayerNorm(pair_dim)
        self.column_attention = nn.MultiheadAttention(
            pair_dim, heads, dropout=dropout, batch_first=True
        )
        self.transition_norm = nn.LayerNorm(pair_dim)
        self.transition = nn.Sequential(
            nn.Linear(pair_dim, 4 * pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * pair_dim, pair_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[0] != pair.shape[1]:
            raise ValueError("pair state must have shape LxLxP")
        row = self.row_norm(pair)
        row_update = self.row_attention(row, row, row, need_weights=False)[0]
        pair = pair + self.dropout(row_update)
        column = self.column_norm(pair.transpose(0, 1))
        column_update = self.column_attention(
            column, column, column, need_weights=False
        )[0].transpose(0, 1)
        pair = pair + self.dropout(column_update)
        return pair + self.dropout(self.transition(self.transition_norm(pair)))


class PairInitializer(nn.Module):
    def __init__(self, config: V4ModelConfig) -> None:
        super().__init__()
        pair_dim = config.pair_dim
        self.max_relative_distance = config.max_relative_distance
        self.left = nn.Linear(config.d_model, pair_dim, bias=False)
        self.right = nn.Linear(config.d_model, pair_dim, bias=False)
        self.base_pair = nn.Embedding(16, pair_dim)
        self.relative = nn.Embedding(2 * config.max_relative_distance + 1, pair_dim)
        self.reactivity_pair = nn.Linear(4, pair_dim, bias=False)
        self.foundation_q = (
            nn.Linear(config.foundation_dim, pair_dim, bias=False)
            if config.foundation_dim
            else None
        )
        self.foundation_k = (
            nn.Linear(config.foundation_dim, pair_dim, bias=False)
            if config.foundation_dim
            else None
        )
        self.norm = nn.LayerNorm(pair_dim)

    def forward(
        self,
        sequence_state: torch.Tensor,
        sequence_one_hot: torch.Tensor,
        wt_reactivity: torch.Tensor,
        wt_observed: torch.Tensor,
        wt_foundation: torch.Tensor | None,
    ) -> torch.Tensor:
        length = sequence_state.shape[0]
        device = sequence_state.device
        base = sequence_one_hot.argmax(-1)
        pair_index = base[:, None] * 4 + base[None, :]
        position = torch.arange(length, device=device)
        relative = (position[None, :] - position[:, None]).clamp(
            -self.max_relative_distance, self.max_relative_distance
        ) + self.max_relative_distance
        react_i = wt_reactivity[:, None].expand(length, length)
        react_j = wt_reactivity[None, :].expand(length, length)
        pair_features = torch.stack(
            [
                react_i,
                react_j,
                torch.abs(react_i - react_j),
                wt_observed[:, None].float() * wt_observed[None, :].float(),
            ],
            dim=-1,
        )
        pair = (
            self.left(sequence_state)[:, None, :]
            + self.right(sequence_state)[None, :, :]
            + self.base_pair(pair_index)
            + self.relative(relative)
            + self.reactivity_pair(pair_features)
        )
        if self.foundation_q is not None:
            if wt_foundation is None:
                raise ValueError("frozen WT foundation embedding is required")
            pair = pair + self.foundation_q(wt_foundation)[:, None, :] * self.foundation_k(
                wt_foundation
            )[None, :, :] / math.sqrt(pair.shape[-1])
        return self.norm(pair)


class MutationConditionedDualTower(nn.Module):
    """Full-construct signed-delta mean model for one WT construct at a time."""

    def __init__(self, config: V4ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or V4ModelConfig.primary()
        self.config.validate()
        config = self.config
        # sequence one-hot (4), react/error/observed/position (4), region one-hot (2)
        self.local_input = nn.Linear(10, config.d_model)
        self.foundation_input = (
            nn.Linear(config.foundation_dim, config.d_model, bias=False)
            if config.foundation_dim
            else None
        )
        self.input_norm = nn.LayerNorm(config.d_model)

        pair_dim = config.pair_dim if config.pair_enabled else None
        self.wt_blocks = nn.ModuleList(
            PairBiasedSequenceBlock(
                config.d_model,
                config.heads,
                config.ff_dim,
                config.dropout,
                pair_dim,
            )
            for _ in range(config.wt_blocks)
        )
        if config.pair_enabled:
            self.pair_initializer = PairInitializer(config)
            self.sequence_to_pair = nn.ModuleList(
                nn.Linear(config.d_model, 2 * config.pair_dim, bias=False)
                for _ in range(config.pair_blocks)
            )
            self.pair_blocks = nn.ModuleList(
                PairAxialBlock(config.pair_dim, config.pair_heads, config.dropout)
                for _ in range(config.pair_blocks)
            )
        else:
            self.pair_initializer = None
            self.sequence_to_pair = nn.ModuleList()
            self.pair_blocks = nn.ModuleList()

        response_input_dim = 2 * config.d_model + 9
        if config.foundation_dim:
            response_input_dim += config.foundation_dim
        if config.pair_enabled:
            response_input_dim += 2 * config.pair_dim
        self.response_input = nn.Sequential(
            nn.Linear(response_input_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.response_blocks = nn.ModuleList(
            PairBiasedSequenceBlock(
                config.d_model,
                config.heads,
                config.ff_dim,
                config.dropout,
                pair_dim,
            )
            for _ in range(config.response_blocks)
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.mean_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def _encode_wt(
        self,
        sequence_one_hot: torch.Tensor,
        wt_reactivity: torch.Tensor,
        wt_error: torch.Tensor,
        wt_observed: torch.Tensor,
        position: torch.Tensor,
        region_one_hot: torch.Tensor,
        wt_foundation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        local = torch.cat(
            [
                sequence_one_hot,
                wt_reactivity[:, None],
                wt_error[:, None],
                wt_observed.float()[:, None],
                position[:, None],
                region_one_hot,
            ],
            dim=-1,
        )
        sequence_state = self.local_input(local)
        if self.foundation_input is not None:
            if wt_foundation is None:
                raise ValueError("frozen WT foundation embedding is required")
            sequence_state = sequence_state + self.foundation_input(wt_foundation.detach())
        sequence_state = self.input_norm(sequence_state)
        pair = None
        if self.pair_initializer is not None:
            pair = self.pair_initializer(
                sequence_state,
                sequence_one_hot,
                wt_reactivity,
                wt_observed,
                wt_foundation.detach() if wt_foundation is not None else None,
            )
        for index, block in enumerate(self.wt_blocks):
            if pair is not None:
                update = self.sequence_to_pair[index](sequence_state)
                left, right = update.chunk(2, dim=-1)
                pair = self.pair_blocks[index](
                    pair + left[:, None, :] + right[None, :, :]
                )
            sequence_state = block(sequence_state.unsqueeze(0), pair).squeeze(0)
        return sequence_state, pair

    def forward_mean_and_features(
        self,
        sequence_one_hot: torch.Tensor,
        wt_reactivity: torch.Tensor,
        wt_error: torch.Tensor,
        wt_observed: torch.Tensor,
        position: torch.Tensor,
        region_one_hot: torch.Tensor,
        edit_idx: torch.Tensor,
        refs: list[str],
        alts: list[str],
        wt_foundation: torch.Tensor | None = None,
        mutant_foundation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = sequence_one_hot.shape[0]
        batch = edit_idx.shape[0]
        if len(refs) != batch or len(alts) != batch:
            raise ValueError("mutation identity count must equal edit_idx count")
        if torch.any(edit_idx < 0) or torch.any(edit_idx >= length):
            raise ValueError("corrected full mutation coordinate is outside construct")
        sequence_state, pair = self._encode_wt(
            sequence_one_hot,
            wt_reactivity,
            wt_error,
            wt_observed,
            position,
            region_one_hot,
            wt_foundation,
        )
        source = sequence_state[edit_idx]
        receiver = sequence_state.unsqueeze(0).expand(batch, -1, -1)
        source_expanded = source[:, None, :].expand(batch, length, -1)
        distance = (
            torch.arange(length, device=edit_idx.device)[None, :] - edit_idx[:, None]
        ).float() / max(length - 1, 1)
        mutation = mutation_one_hot(refs, alts, edit_idx.device)
        response_parts = [
            receiver,
            source_expanded,
            distance[..., None],
            mutation[:, None, :].expand(batch, length, -1),
        ]
        if self.config.foundation_dim:
            if wt_foundation is None or mutant_foundation is None:
                raise ValueError("paired frozen foundation embeddings are required")
            if mutant_foundation.shape != (batch, length, self.config.foundation_dim):
                raise ValueError("mutant foundation embedding shape is incorrect")
            response_parts.append(mutant_foundation.detach() - wt_foundation.detach()[None])
        if pair is not None:
            source_row = pair[edit_idx]
            source_column = pair.transpose(0, 1)[edit_idx]
            response_parts.extend([source_row, source_column])
        response = self.response_input(torch.cat(response_parts, dim=-1))
        for block in self.response_blocks:
            response = block(response, pair)
        features = self.output_norm(response)
        mean = self.mean_head(features).squeeze(-1)
        same = torch.tensor(
            [ref.replace("T", "U") == alt.replace("T", "U") for ref, alt in zip(refs, alts)],
            device=mean.device,
            dtype=torch.bool,
        )
        mean = mean.masked_fill(same[:, None], 0.0)
        return mean, features

    def forward_mean(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_mean_and_features(*args, **kwargs)[0]


class RNAFMOnlyMean(nn.Module):
    """Foundation-only attribution control with no pair or sequence tower."""

    def __init__(self, foundation_dim: int = 640, hidden: int = 512) -> None:
        super().__init__()
        self.foundation_dim = foundation_dim
        self.head = nn.Sequential(
            nn.Linear(3 * foundation_dim + 9, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward_mean_and_features(
        self,
        wt_foundation: torch.Tensor,
        mutant_foundation: torch.Tensor,
        edit_idx: torch.Tensor,
        refs: list[str],
        alts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = mutant_foundation.shape
        if width != self.foundation_dim or wt_foundation.shape != (length, width):
            raise ValueError("foundation-only embedding shape is incorrect")
        source = wt_foundation.detach()[edit_idx, None, :].expand(batch, length, -1)
        wt = wt_foundation.detach()[None].expand(batch, -1, -1)
        delta = mutant_foundation.detach() - wt
        distance = (
            torch.arange(length, device=edit_idx.device)[None, :] - edit_idx[:, None]
        ).float() / max(length - 1, 1)
        mutation = mutation_one_hot(refs, alts, edit_idx.device)
        features = torch.cat(
            [
                wt,
                source,
                delta,
                distance[..., None],
                mutation[:, None, :].expand(batch, length, -1),
            ],
            dim=-1,
        )
        mean = self.head(features).squeeze(-1)
        return mean, features

    def forward_mean(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_mean_and_features(*args, **kwargs)[0]


class ZeroMeanResidualCalibrator(nn.Module):
    """Conditional two-Gaussian residual with a fixed point mean."""

    def __init__(self, feature_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(
        self, point_mean: torch.Tensor, frozen_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.head(frozen_features.detach())
        narrow_weight = torch.sigmoid(raw[..., 0])
        narrow_scale = SCALE_FLOOR + torch.nn.functional.softplus(raw[..., 1])
        wide_scale = narrow_scale + torch.nn.functional.softplus(raw[..., 2])
        weights = torch.stack([narrow_weight, 1.0 - narrow_weight], dim=-1)
        detached_mean = point_mean.detach()
        locations = torch.stack([detached_mean, detached_mean], dim=-1)
        scales = torch.stack([narrow_scale, wide_scale], dim=-1)
        return weights, locations, scales


def freeze_mean_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def residual_crps(
    calibrator: ZeroMeanResidualCalibrator,
    point_mean: torch.Tensor,
    frozen_features: torch.Tensor,
    signed_delta_target: torch.Tensor,
    qualified_mask: torch.Tensor,
) -> torch.Tensor:
    weights, locations, scales = calibrator(point_mean, frozen_features)
    crps = gaussian_mixture_crps_torch(
        locations, scales, weights, signed_delta_target
    ).masked_fill(~qualified_mask, 0.0)
    counts = qualified_mask.float().sum(-1)
    valid = counts > 0
    if not bool(valid.any()):
        return crps.sum() * 0.0
    return (crps.sum(-1) / counts.clamp(min=1.0))[valid].mean()


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def capacity_ratio(candidate: nn.Module, null: nn.Module) -> float:
    candidate_count = trainable_parameter_count(candidate)
    null_count = trainable_parameter_count(null)
    return abs(candidate_count - null_count) / candidate_count
