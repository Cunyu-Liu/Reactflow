#!/usr/bin/env python3
"""Feature41-anchored point models for Model Rescue v11.

The primary and null have identical trainable state.  Their only difference is
the fixed, non-trainable multiplier applied to the outer-train feature41 point:

    primary = feature41 + neural_residual
    null    =             neural_residual

Residual calibration is deliberately not implemented here; v11 reuses the
frozen :class:`MedianAsymmetricResidual` family from model_rescue_v10.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from scripts.reactflow_delta.run_p3_lrso_v3 import ALPHA


PRIMARY_CANDIDATE = "v11_feature41_anchored_context_residual"
MATCHED_NULL = "v11_unanchored_context_null"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v11_prediction.v1"

CONTEXT_WIDTH = 192
ATTENTION_HEADS = 8
CONTEXT_BLOCKS = 4
FFN_WIDTH = 768
HEAD_WIDTH = 256
HEAD_LAYERS = 2
RELATIVE_DISTANCE_WINDOW = 256
DROPOUT = 0.1


def mutation_one_hot(
    refs: list[str], alts: list[str], device: torch.device
) -> torch.Tensor:
    if len(refs) != len(alts):
        raise ValueError("refs and alts must have equal length")
    ref_index = torch.tensor(
        [ALPHA.get(base.replace("T", "U"), 3) for base in refs], device=device
    )
    alt_index = torch.tensor(
        [ALPHA.get(base.replace("T", "U"), 3) for base in alts], device=device
    )
    encoded = torch.zeros(len(refs), 8, device=device)
    encoded.scatter_(1, ref_index[:, None], 1.0)
    encoded.scatter_(1, alt_index[:, None] + 4, 1.0)
    return encoded


class V11ContextBlock(nn.Module):
    """Pre-norm relative attention plus a position-wise GELU FFN."""

    def __init__(
        self,
        width: int = CONTEXT_WIDTH,
        heads: int = ATTENTION_HEADS,
        ffn_width: int = FFN_WIDTH,
        dropout: float = DROPOUT,
        relative_window: int = RELATIVE_DISTANCE_WINDOW,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("context width must be divisible by attention heads")
        self.heads = int(heads)
        self.head_width = int(width // heads)
        self.relative_window = int(relative_window)
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.relative_bias = nn.Parameter(
            torch.randn(heads, 2 * relative_window + 1) * 0.02
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_width, width),
        )
        self._relative_index_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _relative_index(self, length: int, device: torch.device) -> torch.Tensor:
        key = (int(length), str(device))
        cached = self._relative_index_cache.get(key)
        if cached is None:
            position = torch.arange(length, device=device)
            cached = (
                position[None, :]
                - position[:, None]
                + self.relative_window
            ).clamp(0, 2 * self.relative_window)
            self._relative_index_cache[key] = cached
        return cached

    def forward(self, state: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or observed.shape != state.shape[:2]:
            raise ValueError("V11 context state or observed mask has invalid shape")
        batch, length, width = state.shape
        normalized = self.attention_norm(state)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_width
        )
        relative = self.relative_bias[:, self._relative_index(length, state.device)]
        logits = logits + relative.unsqueeze(0)

        # Keep the diagonal valid so an unobserved query never has an all-masked
        # softmax row.  Unobserved outputs remain present but are masked by the
        # registered prediction status downstream.
        invalid_key = (~observed).unsqueeze(1).unsqueeze(2)
        diagonal = torch.eye(length, dtype=torch.bool, device=state.device)[
            None, None
        ]
        logits = logits.masked_fill(invalid_key & ~diagonal, float("-inf"))
        attention = self.attention_dropout(torch.softmax(logits, dim=-1))
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, length, width
        )
        state = state + self.residual_dropout(self.attention_output(context))
        return state + self.residual_dropout(self.ffn(self.ffn_norm(state)))


class V11PointModel(nn.Module):
    """Mutation-conditioned context residual with a fixed feature41 anchor."""

    def __init__(self, *, feature41_skip_multiplier: float) -> None:
        super().__init__()
        if feature41_skip_multiplier not in (0.0, 1.0):
            raise ValueError("V11 skip multiplier is frozen to zero or one")
        self.feature41_skip_multiplier = float(feature41_skip_multiplier)
        self.input_projection = nn.Linear(10, CONTEXT_WIDTH)
        self.input_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.blocks = nn.ModuleList(
            V11ContextBlock() for _ in range(CONTEXT_BLOCKS)
        )
        self.output_norm = nn.LayerNorm(CONTEXT_WIDTH)
        point_input_width = 2 * CONTEXT_WIDTH + 1 + 8 + 1
        self.residual_head = nn.Sequential(
            nn.Linear(point_input_width, HEAD_WIDTH),
            nn.GELU(),
            nn.LayerNorm(HEAD_WIDTH),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_WIDTH, HEAD_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_WIDTH, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def encode(self, context: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(context) != 6:
            raise ValueError("V11 context must contain six aligned tensors")
        sequence, reactivity, precision, observed, position, region = context
        length = sequence.shape[0]
        if sequence.shape != (length, 4) or region.shape != (length, 2):
            raise ValueError("V11 sequence or region tensor has invalid shape")
        for tensor in (reactivity, precision, observed, position):
            if tensor.shape != (length,):
                raise ValueError("V11 scalar context tensor has invalid shape")
        normalized_position = position / max(length - 1, 1)
        local = torch.cat(
            [
                sequence,
                reactivity[:, None],
                precision[:, None],
                observed[:, None],
                normalized_position[:, None],
                region,
            ],
            dim=-1,
        )
        state = self.input_norm(self.input_projection(local)).unsqueeze(0)
        observed_mask = observed.bool().unsqueeze(0)
        for block in self.blocks:
            state = block(state, observed_mask)
        return self.output_norm(state[0])

    def forward_point_and_features(
        self,
        hidden: torch.Tensor,
        edit_index: torch.Tensor,
        signed_distance: torch.Tensor,
        refs: list[str],
        alts: list[str],
        prediction_mask: torch.Tensor,
        feature41_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2 or hidden.shape[1] != CONTEXT_WIDTH:
            raise ValueError("V11 hidden representation has invalid shape")
        batch = edit_index.shape[0]
        length = hidden.shape[0]
        expected = (batch, length)
        if signed_distance.shape != expected or prediction_mask.shape != expected:
            raise ValueError("V11 distance or prediction mask has invalid shape")
        if feature41_point.shape != expected:
            raise ValueError("V11 feature41 point has invalid shape")
        if len(refs) != batch or len(alts) != batch:
            raise ValueError("V11 mutation identity count is misaligned")

        source = hidden[edit_index]
        source = source[:, None, :].expand(batch, length, -1)
        receiver = hidden[None, :, :].expand(batch, -1, -1)
        normalized_distance = signed_distance / max(length - 1, 1)
        mutation = mutation_one_hot(refs, alts, hidden.device)
        mutation = mutation[:, None, :].expand(batch, length, -1)
        features = torch.cat(
            [
                source,
                receiver,
                normalized_distance[..., None],
                mutation,
                feature41_point[..., None],
            ],
            dim=-1,
        )
        residual = self.residual_head(features).squeeze(-1)
        point = self.feature41_skip_multiplier * feature41_point + residual
        point = point.masked_fill(~prediction_mask, 0.0)
        same = torch.tensor(
            [
                ref.replace("T", "U") == alt.replace("T", "U")
                for ref, alt in zip(refs, alts)
            ],
            dtype=torch.bool,
            device=hidden.device,
        )
        point = point.masked_fill(same[:, None], 0.0)
        return point, features

    def forward_point(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_point_and_features(*args, **kwargs)[0]


def make_exact_matched_pair(
    *, seed: int, device: str | torch.device
) -> tuple[V11PointModel, V11PointModel]:
    torch.manual_seed(int(seed))
    primary = V11PointModel(feature41_skip_multiplier=1.0).to(device)
    null = copy.deepcopy(primary)
    null.feature41_skip_multiplier = 0.0
    assert_exact_trainable_match(primary, null)
    return primary, null


def assert_exact_trainable_match(
    primary: V11PointModel, null: V11PointModel
) -> None:
    primary_parameters = dict(primary.named_parameters())
    null_parameters = dict(null.named_parameters())
    if primary_parameters.keys() != null_parameters.keys():
        raise RuntimeError("V11 primary and null parameter names differ")
    for name in primary_parameters:
        left = primary_parameters[name]
        right = null_parameters[name]
        if left.shape != right.shape or not torch.equal(left.detach(), right.detach()):
            raise RuntimeError(f"V11 matched initialization differs at {name}")
    if primary.feature41_skip_multiplier != 1.0:
        raise RuntimeError("V11 primary feature41 anchor is not one")
    if null.feature41_skip_multiplier != 0.0:
        raise RuntimeError("V11 null feature41 anchor is not zero")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def target_delta(
    target: torch.Tensor,
    qualified_mask: torch.Tensor,
    wt_filled: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        qualified_mask,
        target - wt_filled[None, :],
        torch.zeros_like(target),
    )


def method_cell_balanced_l1(
    point: torch.Tensor,
    target: torch.Tensor,
    qualified_mask: torch.Tensor,
    wt_filled: torch.Tensor,
) -> torch.Tensor:
    """Position mean within mutant, then equal mutant mean within one cell."""
    delta = target_delta(target, qualified_mask, wt_filled)
    absolute = torch.abs(point - delta).masked_fill(~qualified_mask, 0.0)
    counts = qualified_mask.float().sum(-1)
    valid = counts > 0
    if not bool(valid.any()):
        return absolute.sum() * 0.0
    per_mutant = absolute.sum(-1) / counts.clamp(min=1.0)
    return per_mutant[valid].mean()


def freeze_point_model(model: V11PointModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
