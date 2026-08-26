#!/usr/bin/env python3
"""Outcome-blind cross-construct context proposed for post-V14 routing.

This module is implementation-only.  It does not authorize a scientific run.
The candidate and null share every parameter; their sole functional difference
is whether a focal construct can attend to the other WT constructs in its
puzzle set.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v11 import (
    V11ContextBlock,
    mutation_one_hot,
)


FULL_CROSS_CONSTRUCT = "FULL_CROSS_CONSTRUCT"
BLOCK_DIAGONAL_NULL = "BLOCK_DIAGONAL_NULL"
CONNECTIVITY_MODES = {FULL_CROSS_CONSTRUCT, BLOCK_DIAGONAL_NULL}

EXPECTED_CONSTRUCTS_PER_PUZZLE = 8
CONTEXT_WIDTH = 256
ATTENTION_HEADS = 8
BASE_FEATURE_WIDTH = 522
POINT_HEAD_WIDTH = 384
DROPOUT = 0.1


class OutcomeBlindWTEncoder(nn.Module):
    """Encode one WT construct without mutation outcomes or identity labels."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(11, CONTEXT_WIDTH)
        self.input_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.blocks = nn.ModuleList(
            V11ContextBlock(
                width=CONTEXT_WIDTH,
                heads=ATTENTION_HEADS,
                ffn_width=4 * CONTEXT_WIDTH,
                dropout=DROPOUT,
                relative_window=256,
            )
            for _ in range(6)
        )
        self.output_norm = nn.LayerNorm(CONTEXT_WIDTH)

    def forward(self, context: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(context) != 6:
            raise ValueError("WT context must contain six aligned tensors")
        sequence, reactivity, precision, observed, position, region = context
        length = sequence.shape[0]
        if sequence.shape != (length, 4) or region.shape != (length, 2):
            raise ValueError("WT sequence or region tensor has invalid shape")
        for tensor in (reactivity, precision, observed, position):
            if tensor.shape != (length,):
                raise ValueError("WT scalar context tensor has invalid shape")
        normalized_position = position / max(length - 1, 1)
        corruption_token = torch.zeros(
            length, 1, dtype=sequence.dtype, device=sequence.device
        )
        local = torch.cat(
            [
                sequence,
                reactivity[:, None],
                precision[:, None],
                observed[:, None],
                normalized_position[:, None],
                region,
                corruption_token,
            ],
            dim=-1,
        )
        state = self.input_norm(self.input_projection(local)).unsqueeze(0)
        attention_keys = observed.bool().unsqueeze(0)
        for block in self.blocks:
            state = block(state, attention_keys)
        return self.output_norm(state[0])


class PuzzleSetMetaContext(nn.Module):
    """Feature41-anchored point adapter with puzzle-set WT context.

    ``construct_hidden`` and ``construct_observed`` must be produced without
    mutant outcomes.  Construct order has no positional encoding, so the set
    mixer is permutation equivariant.  A real zero-observed construct is
    represented by the mean of all of its sequence-derived hidden states plus
    an explicit observed-fraction value of zero.
    """

    def __init__(self, *, connectivity: str) -> None:
        super().__init__()
        if connectivity not in CONNECTIVITY_MODES:
            raise ValueError(f"unsupported puzzle-set connectivity: {connectivity}")
        self.connectivity = connectivity
        self.construct_projection = nn.Linear(CONTEXT_WIDTH + 1, CONTEXT_WIDTH)
        self.attention_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.set_attention = nn.MultiheadAttention(
            CONTEXT_WIDTH,
            ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.ffn = nn.Sequential(
            nn.Linear(CONTEXT_WIDTH, 4 * CONTEXT_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(4 * CONTEXT_WIDTH, CONTEXT_WIDTH),
        )
        self.residual_dropout = nn.Dropout(DROPOUT)
        self.output_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.point_head = nn.Sequential(
            nn.Linear(BASE_FEATURE_WIDTH + CONTEXT_WIDTH, POINT_HEAD_WIDTH),
            nn.GELU(),
            nn.LayerNorm(POINT_HEAD_WIDTH),
            nn.Dropout(DROPOUT),
            nn.Linear(POINT_HEAD_WIDTH, POINT_HEAD_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(POINT_HEAD_WIDTH, 1),
        )
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)

    @staticmethod
    def _validate_construct(
        hidden: torch.Tensor, observed: torch.Tensor
    ) -> None:
        if hidden.ndim != 2 or hidden.shape[1] != CONTEXT_WIDTH:
            raise ValueError("puzzle-set hidden state has invalid shape")
        if observed.shape != (hidden.shape[0],) or observed.dtype != torch.bool:
            raise ValueError("puzzle-set observed mask has invalid shape or dtype")
        if hidden.shape[0] == 0:
            raise ValueError("puzzle-set construct cannot be empty")

    def pool_construct_tokens(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(construct_hidden) != EXPECTED_CONSTRUCTS_PER_PUZZLE or len(
            construct_observed
        ) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("puzzle-set context requires exactly eight constructs")
        pooled = []
        for hidden, observed in zip(construct_hidden, construct_observed):
            self._validate_construct(hidden, observed)
            if bool(observed.any()):
                summary = hidden[observed].mean(dim=0)
            else:
                # P20_Eterna is a registered, reachable zero-observed WT
                # construct. Its sequence-derived hidden states remain legal
                # outcome-blind inputs even though no reactivity target exists.
                summary = hidden.mean(dim=0)
            observed_fraction = observed.to(hidden.dtype).mean().reshape(1)
            pooled.append(torch.cat([summary, observed_fraction], dim=0))
        return self.construct_projection(torch.stack(pooled, dim=0))

    def mix_construct_tokens(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        tokens = self.pool_construct_tokens(
            construct_hidden, construct_observed
        ).unsqueeze(0)
        normalized = self.attention_norm(tokens)
        attention_mask = None
        if self.connectivity == BLOCK_DIAGONAL_NULL:
            attention_mask = ~torch.eye(
                EXPECTED_CONSTRUCTS_PER_PUZZLE,
                dtype=torch.bool,
                device=tokens.device,
            )
        attended, _weights = self.set_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        tokens = tokens + self.residual_dropout(attended)
        tokens = tokens + self.residual_dropout(self.ffn(self.ffn_norm(tokens)))
        return self.output_norm(tokens[0])

    def forward(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
        focal_construct_index: int,
        base_point_features: torch.Tensor,
        feature41_point: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= int(focal_construct_index) < EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("focal construct index is outside the puzzle set")
        if base_point_features.ndim != 3 or base_point_features.shape[-1] != (
            BASE_FEATURE_WIDTH
        ):
            raise ValueError("base point features have invalid shape")
        expected = base_point_features.shape[:2]
        if feature41_point.shape != expected or prediction_mask.shape != expected:
            raise ValueError("point anchor or prediction mask is misaligned")
        if prediction_mask.dtype != torch.bool:
            raise ValueError("prediction mask must be boolean")

        mixed = self.mix_construct_tokens(construct_hidden, construct_observed)
        focal = mixed[int(focal_construct_index)]
        expanded = focal.reshape(1, 1, -1).expand(*expected, -1)
        residual = self.point_head(
            torch.cat([base_point_features, expanded], dim=-1)
        ).squeeze(-1)
        point = (feature41_point + residual).masked_fill(~prediction_mask, 0.0)
        return point, residual, mixed


class PuzzleSetMetaContextPointModel(nn.Module):
    """Complete feature41-anchored point model for the proposed capability."""

    def __init__(self, *, connectivity: str) -> None:
        super().__init__()
        self.encoder = OutcomeBlindWTEncoder()
        self.meta_context = PuzzleSetMetaContext(connectivity=connectivity)

    @property
    def connectivity(self) -> str:
        return self.meta_context.connectivity

    @connectivity.setter
    def connectivity(self, value: str) -> None:
        if value not in CONNECTIVITY_MODES:
            raise ValueError(f"unsupported puzzle-set connectivity: {value}")
        self.meta_context.connectivity = value

    def encode_puzzle_set(
        self, contexts: Sequence[tuple[torch.Tensor, ...]]
    ) -> list[torch.Tensor]:
        if len(contexts) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("point model requires exactly eight WT contexts")
        return [self.encoder(context) for context in contexts]

    @staticmethod
    def base_point_features(
        focal_hidden: torch.Tensor,
        edit_index: torch.Tensor,
        signed_distance: torch.Tensor,
        refs: list[str],
        alts: list[str],
        feature41_point: torch.Tensor,
    ) -> torch.Tensor:
        if focal_hidden.ndim != 2 or focal_hidden.shape[1] != CONTEXT_WIDTH:
            raise ValueError("focal WT hidden state has invalid shape")
        batch = edit_index.shape[0]
        length = focal_hidden.shape[0]
        expected = (batch, length)
        if signed_distance.shape != expected or feature41_point.shape != expected:
            raise ValueError("distance or feature41 point is misaligned")
        if len(refs) != batch or len(alts) != batch:
            raise ValueError("mutation identity count is misaligned")
        if bool((edit_index < 0).any()) or bool((edit_index >= length).any()):
            raise ValueError("edit index is outside the focal construct")
        source = focal_hidden[edit_index][:, None, :].expand(batch, length, -1)
        receiver = focal_hidden[None, :, :].expand(batch, -1, -1)
        normalized_distance = signed_distance / max(length - 1, 1)
        mutation = mutation_one_hot(refs, alts, focal_hidden.device)
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
        if features.shape != (batch, length, BASE_FEATURE_WIDTH):
            raise RuntimeError("puzzle-set base point feature width changed")
        return features

    def forward(
        self,
        contexts: Sequence[tuple[torch.Tensor, ...]],
        focal_construct_index: int,
        edit_index: torch.Tensor,
        signed_distance: torch.Tensor,
        refs: list[str],
        alts: list[str],
        feature41_point: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode_puzzle_set(contexts)
        observed = [context[3].bool() for context in contexts]
        base_features = self.base_point_features(
            hidden[int(focal_construct_index)],
            edit_index,
            signed_distance,
            refs,
            alts,
            feature41_point,
        )
        point, residual, mixed = self.meta_context(
            hidden,
            observed,
            focal_construct_index,
            base_features,
            feature41_point,
            prediction_mask,
        )
        same = torch.tensor(
            [
                ref.replace("T", "U") == alt.replace("T", "U")
                for ref, alt in zip(refs, alts)
            ],
            dtype=torch.bool,
            device=point.device,
        )
        point = point.masked_fill(same[:, None], 0.0)
        return point, residual, mixed


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def make_exact_matched_pair(
    *, seed: int, device: str | torch.device = "cpu"
) -> tuple[PuzzleSetMetaContext, PuzzleSetMetaContext]:
    torch.manual_seed(int(seed))
    candidate = PuzzleSetMetaContext(
        connectivity=FULL_CROSS_CONSTRUCT
    ).to(device)
    null = copy.deepcopy(candidate)
    null.connectivity = BLOCK_DIAGONAL_NULL
    left = dict(candidate.named_parameters())
    right = dict(null.named_parameters())
    if left.keys() != right.keys():
        raise RuntimeError("puzzle-set candidate and null parameter names differ")
    for name in left:
        if not torch.equal(left[name].detach(), right[name].detach()):
            raise RuntimeError(
                f"puzzle-set candidate and null initialization differs at {name}"
            )
    if parameter_count(candidate) != parameter_count(null):
        raise RuntimeError("puzzle-set candidate and null parameter counts differ")
    return candidate, null


def make_exact_full_model_pair(
    *, seed: int, device: str | torch.device = "cpu"
) -> tuple[PuzzleSetMetaContextPointModel, PuzzleSetMetaContextPointModel]:
    torch.manual_seed(int(seed))
    candidate = PuzzleSetMetaContextPointModel(
        connectivity=FULL_CROSS_CONSTRUCT
    ).to(device)
    null = copy.deepcopy(candidate)
    null.connectivity = BLOCK_DIAGONAL_NULL
    left = dict(candidate.named_parameters())
    right = dict(null.named_parameters())
    if left.keys() != right.keys():
        raise RuntimeError("full puzzle-set candidate and null parameter names differ")
    for name in left:
        if not torch.equal(left[name].detach(), right[name].detach()):
            raise RuntimeError(
                f"full puzzle-set initialization differs at {name}"
            )
    if parameter_count(candidate) != parameter_count(null):
        raise RuntimeError("full puzzle-set candidate and null counts differ")
    return candidate, null
