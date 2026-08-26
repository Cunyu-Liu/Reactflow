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


FULL_CROSS_CONSTRUCT = "FULL_CROSS_CONSTRUCT"
BLOCK_DIAGONAL_NULL = "BLOCK_DIAGONAL_NULL"
CONNECTIVITY_MODES = {FULL_CROSS_CONSTRUCT, BLOCK_DIAGONAL_NULL}

EXPECTED_CONSTRUCTS_PER_PUZZLE = 8
CONTEXT_WIDTH = 256
ATTENTION_HEADS = 8
BASE_FEATURE_WIDTH = 522
POINT_HEAD_WIDTH = 384
DROPOUT = 0.1


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
