#!/usr/bin/env python3
"""Outcome-blind cross-construct context proposed for post-V14 routing.

This module is implementation-only.  It does not authorize a scientific run.
The candidate and null share every parameter and eight-token attention graph;
their sole functional difference is registered versus fixed wrong-position
alignment of the seven non-focal WT constructs.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v11 import (
    V11ContextBlock,
    method_cell_balanced_l1,
    mutation_one_hot,
)
from scripts.reactflow_delta.model_rescue_v14 import V14PointModel


FULL_CROSS_CONSTRUCT = "FULL_CROSS_CONSTRUCT"
POSITION_DERANGED_NULL = "POSITION_DERANGED_NULL"
POSITION_DERANGEMENT_SHIFT = 17
CONNECTIVITY_MODES = {FULL_CROSS_CONSTRUCT, POSITION_DERANGED_NULL}
POSITION_ALIGNED_OPERATOR = "POSITION_ALIGNED_CROSS_CONSTRUCT_CONSENSUS_V5"

EXPECTED_CONSTRUCTS_PER_PUZZLE = 8
CONTEXT_WIDTH = 256
ATTENTION_HEADS = 8
V14_LOCAL_FEATURE_WIDTH = 522
BASE_FEATURE_WIDTH = V14_LOCAL_FEATURE_WIDTH + 1
ALIGNMENT_STAT_WIDTH = 4
POINT_HEAD_WIDTH = 384
DROPOUT = 0.1
EXPECTED_TOTAL_PARAMETERS = 6_171_697
EXPECTED_TRAINABLE_PARAMETERS = 1_404_417
POINT_HEAD_WARMUP_EPOCHS = 1
POINT_HEAD_LR = 1e-3
POINT_CONTEXT_LR = 3e-4
POINT_GRADIENT_CLIP = 5.0


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

    def forward(
        self,
        context: tuple[torch.Tensor, ...],
        corruption_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(context) != 6:
            raise ValueError("WT context must contain six aligned tensors")
        sequence, reactivity, precision, observed, position, region = context
        length = sequence.shape[0]
        if sequence.shape != (length, 4) or region.shape != (length, 2):
            raise ValueError("WT sequence or region tensor has invalid shape")
        for tensor in (reactivity, precision, observed, position):
            if tensor.shape != (length,):
                raise ValueError("WT scalar context tensor has invalid shape")
        if corruption_mask is None:
            corruption_mask = torch.zeros(
                length, dtype=torch.bool, device=sequence.device
            )
        if corruption_mask.shape != (length,) or corruption_mask.dtype != torch.bool:
            raise ValueError("WT corruption mask has invalid shape or dtype")
        if bool((corruption_mask & ~observed.bool()).any()):
            raise ValueError("WT corruption can mask only observed positions")
        visible_reactivity = reactivity.masked_fill(corruption_mask, 0.0)
        visible_precision = precision.masked_fill(corruption_mask, 0.0)
        visible_observed = observed.masked_fill(corruption_mask, 0.0)
        normalized_position = position / max(length - 1, 1)
        local = torch.cat(
            [
                sequence,
                visible_reactivity[:, None],
                visible_precision[:, None],
                visible_observed[:, None],
                normalized_position[:, None],
                region,
                corruption_mask.to(sequence.dtype)[:, None],
            ],
            dim=-1,
        )
        state = self.input_norm(self.input_projection(local)).unsqueeze(0)
        attention_keys = (observed.bool() & ~corruption_mask).unsqueeze(0)
        for block in self.blocks:
            state = block(state, attention_keys)
        return self.output_norm(state[0])


V14_ENCODER_PREFIXES = (
    "input_projection.",
    "input_norm.",
    "blocks.",
    "output_norm.",
)


def load_frozen_v14_encoder(
    encoder: OutcomeBlindWTEncoder,
    v14_point_state: dict[str, torch.Tensor],
) -> None:
    """Import the exact V14 encoder subset and make it immutable.

    The source is one outer-fold V14 seed-0 candidate checkpoint. The
    pretraining decoder and V14 residual head are intentionally not consumers
    of the puzzle-set model.
    """

    expected = encoder.state_dict()
    imported = {
        name: value
        for name, value in v14_point_state.items()
        if name.startswith(V14_ENCODER_PREFIXES)
    }
    if set(imported) != set(expected):
        missing = sorted(set(expected) - set(imported))
        unexpected = sorted(set(imported) - set(expected))
        raise ValueError(
            "V14 encoder checkpoint does not match the puzzle-set encoder: "
            f"missing={missing} unexpected={unexpected}"
        )
    for name, value in imported.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"V14 encoder tensor shape changed at {name}")
    encoder.load_state_dict(imported, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def assert_v14_encoder_replay(
    encoder: OutcomeBlindWTEncoder,
    source: V14PointModel,
    context: tuple[torch.Tensor, ...],
) -> None:
    """Prove that the frozen P1 encoder exactly replays V14 without masking."""

    encoder.eval()
    source.eval()
    with torch.no_grad():
        observed = encoder(context)
        expected = source.encode(context, None)
    if not torch.equal(observed, expected):
        maximum = float(torch.max(torch.abs(observed - expected)).cpu())
        raise RuntimeError(f"puzzle-set V14 encoder replay differs by {maximum}")


class PuzzleSetMetaContext(nn.Module):
    """Parent-anchored point adapter with aligned puzzle-set WT context.

    ``construct_hidden`` and ``construct_observed`` must be produced without
    mutant outcomes. All eight constructs share the registered full-sequence
    coordinate frame, so cross-construct attention is applied independently at
    every aligned position. Construct order has no positional encoding and the
    mixer remains permutation equivariant. A real zero-observed construct keeps
    its sequence-derived hidden state plus an explicit observed value of zero.
    """

    def __init__(self, *, connectivity: str) -> None:
        super().__init__()
        if connectivity not in CONNECTIVITY_MODES:
            raise ValueError(f"unsupported puzzle-set connectivity: {connectivity}")
        self.connectivity = connectivity
        self.construct_projection = nn.Linear(CONTEXT_WIDTH + 1, CONTEXT_WIDTH)
        self.alignment_projection = nn.Linear(ALIGNMENT_STAT_WIDTH, CONTEXT_WIDTH)
        self.attention_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.set_attention = nn.MultiheadAttention(
            CONTEXT_WIDTH,
            ATTENTION_HEADS,
            dropout=0.0,
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
            nn.Linear(BASE_FEATURE_WIDTH + 2 * CONTEXT_WIDTH, POINT_HEAD_WIDTH),
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
    def _validate_construct(hidden: torch.Tensor, observed: torch.Tensor) -> None:
        if hidden.ndim != 2 or hidden.shape[1] != CONTEXT_WIDTH:
            raise ValueError("puzzle-set hidden state has invalid shape")
        if observed.shape != (hidden.shape[0],) or observed.dtype != torch.bool:
            raise ValueError("puzzle-set observed mask has invalid shape or dtype")
        if hidden.shape[0] == 0:
            raise ValueError("puzzle-set construct cannot be empty")

    def project_individual_construct_tokens(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
        construct_reactivity: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Project eight constructs without putting non-focal data in a query.

        Cross-construct statistics are represented by the separate summary token
        below.  Keeping each individual token self-only makes the focal query
        identical in the aligned candidate and position-deranged null.
        """

        if (
            len(construct_hidden) != EXPECTED_CONSTRUCTS_PER_PUZZLE
            or len(construct_observed) != EXPECTED_CONSTRUCTS_PER_PUZZLE
            or len(construct_reactivity) != EXPECTED_CONSTRUCTS_PER_PUZZLE
        ):
            raise ValueError("individual projection requires exactly eight constructs")
        lengths = {int(hidden.shape[0]) for hidden in construct_hidden}
        if len(lengths) != 1:
            raise ValueError("puzzle-set constructs do not share one coordinate frame")
        projected_inputs = []
        individual_statistics = []
        for hidden, observed, reactivity in zip(
            construct_hidden, construct_observed, construct_reactivity
        ):
            self._validate_construct(hidden, observed)
            if reactivity.shape != observed.shape:
                raise ValueError("puzzle-set WT reactivity is misaligned")
            finite_observed = observed & torch.isfinite(reactivity)
            safe_reactivity = torch.where(
                finite_observed, reactivity, torch.zeros_like(reactivity)
            )
            projected_inputs.append(
                torch.cat([hidden, observed.to(hidden.dtype)[:, None]], dim=-1)
            )
            individual_statistics.append(
                torch.stack(
                    [
                        safe_reactivity,
                        torch.zeros_like(reactivity),
                        finite_observed.to(reactivity.dtype),
                        torch.zeros_like(reactivity),
                    ],
                    dim=-1,
                )
            )
        return self.construct_projection(torch.stack(projected_inputs, dim=0)) + (
            self.alignment_projection(torch.stack(individual_statistics, dim=0))
        )

    def nonfocal_summary_token(
        self,
        individual_tokens: torch.Tensor,
        construct_observed: Sequence[torch.Tensor],
        construct_reactivity: Sequence[torch.Tensor],
        focal_construct_index: int,
    ) -> torch.Tensor:
        """Pool the seven non-focal tokens without a learned summary parameter."""

        if (
            individual_tokens.ndim != 3
            or individual_tokens.shape[0] != EXPECTED_CONSTRUCTS_PER_PUZZLE
            or individual_tokens.shape[2] != CONTEXT_WIDTH
        ):
            raise ValueError("individual puzzle-set tokens have invalid shape")
        if (
            len(construct_observed) != EXPECTED_CONSTRUCTS_PER_PUZZLE
            or len(construct_reactivity) != EXPECTED_CONSTRUCTS_PER_PUZZLE
        ):
            raise ValueError("nonfocal summary requires exactly eight constructs")
        if not 0 <= int(focal_construct_index) < EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("nonfocal summary focal construct is outside the set")
        nonfocal = [
            index
            for index in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
            if index != int(focal_construct_index)
        ]
        pooled = individual_tokens[nonfocal].mean(dim=0)
        values = torch.stack([construct_reactivity[index] for index in nonfocal], dim=0)
        observed = torch.stack(
            [construct_observed[index] for index in nonfocal], dim=0
        ).bool()
        observed = observed & torch.isfinite(values)
        safe = torch.where(observed, values, torch.zeros_like(values))
        counts = observed.sum(dim=0)
        denominator = counts.clamp_min(1).to(values.dtype)
        supported = counts > 0
        mean = torch.where(
            supported, safe.sum(dim=0) / denominator, torch.zeros_like(denominator)
        )
        variance = torch.where(
            supported,
            safe.square().sum(dim=0) / denominator - mean.square(),
            torch.zeros_like(mean),
        ).clamp_min(0.0)
        statistics = torch.stack(
            [
                mean,
                torch.sqrt(variance),
                counts.to(values.dtype) / float(EXPECTED_CONSTRUCTS_PER_PUZZLE - 1),
                torch.zeros_like(mean),
            ],
            dim=-1,
        )
        return pooled + self.alignment_projection(statistics)

    def zero_nonfocal_reference_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Build the projected seven-plus-one reference from raw zero inputs.

        The reference is defined before either learned projection.  It therefore
        contains the same learned biases as the observed path, which are removed
        only after both paths traverse the complete attention/FFN/norm block.
        """

        if hidden.ndim != 2 or hidden.shape[1] != CONTEXT_WIDTH:
            raise ValueError("zero-nonfocal reference has invalid hidden shape")
        length = int(hidden.shape[0])
        individual_input = torch.zeros(
            length,
            CONTEXT_WIDTH + 1,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        alignment_input = torch.zeros(
            length,
            ALIGNMENT_STAT_WIDTH,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        individual = self.construct_projection(individual_input) + (
            self.alignment_projection(alignment_input)
        )
        individual_tokens = individual[:, None, :].expand(-1, 7, -1)
        summary = individual_tokens.mean(dim=1) + self.alignment_projection(
            alignment_input
        )
        return torch.cat(
            [
                individual_tokens,
                summary[:, None, :],
            ],
            dim=1,
        )

    @staticmethod
    def _position_deranged_inputs(
        values: Sequence[torch.Tensor], focal_construct_index: int
    ) -> list[torch.Tensor]:
        if len(values) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("position derangement requires exactly eight constructs")
        if not 0 <= int(focal_construct_index) < EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("position derangement focal construct is outside the set")
        return [
            (
                value
                if index == int(focal_construct_index)
                else torch.roll(value, shifts=POSITION_DERANGEMENT_SHIFT, dims=0)
            )
            for index, value in enumerate(values)
        ]

    def _cross_block(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        normalized_query = self.attention_norm(query)
        normalized_key_value = self.attention_norm(key_value)
        attended, _weights = self.set_attention(
            normalized_query,
            normalized_key_value,
            normalized_key_value,
            need_weights=False,
        )
        cross = attended + self.residual_dropout(self.ffn(self.ffn_norm(attended)))
        return self.output_norm(cross)

    def paired_cross_block(
        self,
        query: torch.Tensor,
        actual_key_value: torch.Tensor,
        reference_key_value: torch.Tensor,
    ) -> torch.Tensor:
        """Remove every query-only or learned-constant cross contribution.

        Both paths reuse the identical focal query and, in training mode, the
        identical stochastic draw.  RNG advances as one cross-block forward.
        Consequently raw-zero non-focal inputs yield exact zero cross evidence,
        regardless of learned projection, attention, FFN or normalization bias.
        """

        if actual_key_value.shape != reference_key_value.shape:
            raise ValueError("paired cross-block K/V tensors have different shapes")
        if query.ndim != 3 or actual_key_value.ndim != 3:
            raise ValueError("paired cross-block tensors must be batched sequences")
        if query.shape[0] != actual_key_value.shape[0] or query.shape[2] != (
            actual_key_value.shape[2]
        ):
            raise ValueError("paired cross-block query and K/V are misaligned")
        if not self.training:
            return self._cross_block(query, actual_key_value) - self._cross_block(
                query, reference_key_value
            )

        cpu_before = torch.get_rng_state()
        cuda_before = (
            torch.cuda.get_rng_state(query.device)
            if query.device.type == "cuda"
            else None
        )
        observed = self._cross_block(query, actual_key_value)
        cpu_after = torch.get_rng_state()
        cuda_after = (
            torch.cuda.get_rng_state(query.device)
            if query.device.type == "cuda"
            else None
        )
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, query.device)
        try:
            baseline = self._cross_block(query, reference_key_value)
        finally:
            torch.set_rng_state(cpu_after)
            if cuda_after is not None:
                torch.cuda.set_rng_state(cuda_after, query.device)
        return observed - baseline

    def mix_construct_tokens(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
        construct_reactivity: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        # Every focal query attends to seven non-focal individual tokens and one
        # parameter-free non-focal summary. The focal token is never a K/V. The
        # candidate uses registered coordinates; the null keeps the query fixed
        # and circularly shifts all seven non-focal inputs, preserving identical
        # attention support, parameterization and compute.
        query_batches = []
        key_value_batches = []
        reference_key_value_batches = []
        for focal_index in range(EXPECTED_CONSTRUCTS_PER_PUZZLE):
            if self.connectivity == POSITION_DERANGED_NULL:
                hidden = self._position_deranged_inputs(construct_hidden, focal_index)
                observed = self._position_deranged_inputs(
                    construct_observed, focal_index
                )
                reactivity = self._position_deranged_inputs(
                    construct_reactivity, focal_index
                )
            else:
                hidden = list(construct_hidden)
                observed = list(construct_observed)
                reactivity = list(construct_reactivity)
            individual = self.project_individual_construct_tokens(
                hidden, observed, reactivity
            )
            summary = self.nonfocal_summary_token(
                individual, observed, reactivity, focal_index
            )
            nonfocal = [
                index
                for index in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
                if index != focal_index
            ]
            query_batches.append(individual[focal_index][:, None, :])
            key_value_batches.append(
                torch.cat(
                    [individual[nonfocal].permute(1, 0, 2), summary[:, None, :]],
                    dim=1,
                )
            )
            reference_key_value_batches.append(
                self.zero_nonfocal_reference_tokens(hidden[focal_index])
            )

        query = torch.cat(query_batches, dim=0)
        key_value = torch.cat(key_value_batches, dim=0)
        reference_key_value = torch.cat(reference_key_value_batches, dim=0)
        cross = self.paired_cross_block(query, key_value, reference_key_value)
        length = int(construct_hidden[0].shape[0])
        return cross.reshape(EXPECTED_CONSTRUCTS_PER_PUZZLE, length, CONTEXT_WIDTH)

    def paired_point_head(
        self, actual: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actual/reference with one shared dropout draw.

        The RNG is advanced exactly as one point-head forward. This makes a
        zero-cross reference cancel base-only behavior exactly in train mode.
        """

        if actual.shape != reference.shape:
            raise ValueError("paired point-head inputs have different shapes")
        if not self.training:
            return self.point_head(actual), self.point_head(reference)
        cpu_before = torch.get_rng_state()
        cuda_before = (
            torch.cuda.get_rng_state(actual.device)
            if actual.device.type == "cuda"
            else None
        )
        observed = self.point_head(actual)
        cpu_after = torch.get_rng_state()
        cuda_after = (
            torch.cuda.get_rng_state(actual.device)
            if actual.device.type == "cuda"
            else None
        )
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, actual.device)
        try:
            baseline = self.point_head(reference)
        finally:
            torch.set_rng_state(cpu_after)
            if cuda_after is not None:
                torch.cuda.set_rng_state(cuda_after, actual.device)
        return observed, baseline

    def forward(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
        construct_reactivity: Sequence[torch.Tensor],
        focal_construct_index: int,
        edit_index: torch.Tensor,
        base_point_features: torch.Tensor,
        parent_point: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= int(focal_construct_index) < EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("focal construct index is outside the puzzle set")
        if base_point_features.ndim != 3 or base_point_features.shape[-1] != (
            BASE_FEATURE_WIDTH
        ):
            raise ValueError("base point features have invalid shape")
        expected = base_point_features.shape[:2]
        if parent_point.shape != expected or prediction_mask.shape != expected:
            raise ValueError("point anchor or prediction mask is misaligned")
        if prediction_mask.dtype != torch.bool:
            raise ValueError("prediction mask must be boolean")

        mixed = self.mix_construct_tokens(
            construct_hidden, construct_observed, construct_reactivity
        )
        point, residual = self.point_from_mixed(
            mixed,
            focal_construct_index,
            edit_index,
            base_point_features,
            parent_point,
            prediction_mask,
        )
        return point, residual, mixed

    def point_from_mixed(
        self,
        mixed: torch.Tensor,
        focal_construct_index: int,
        edit_index: torch.Tensor,
        base_point_features: torch.Tensor,
        parent_point: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            mixed.ndim != 3
            or mixed.shape[0] != EXPECTED_CONSTRUCTS_PER_PUZZLE
            or (mixed.shape[2] != CONTEXT_WIDTH)
        ):
            raise ValueError("mixed aligned puzzle-set states have invalid shape")
        if not 0 <= int(focal_construct_index) < EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("focal construct index is outside the puzzle set")
        if base_point_features.ndim != 3 or base_point_features.shape[-1] != (
            BASE_FEATURE_WIDTH
        ):
            raise ValueError("base point features have invalid shape")
        expected = base_point_features.shape[:2]
        if parent_point.shape != expected or prediction_mask.shape != expected:
            raise ValueError("point anchor or prediction mask is misaligned")
        invalid_edit = (edit_index < 0) | (edit_index >= mixed.shape[1])
        if edit_index.shape != (expected[0],) or bool(invalid_edit.any()):
            raise ValueError("puzzle-set edit index is misaligned")
        if mixed.shape[1] != expected[1]:
            raise ValueError("mixed puzzle-set length differs from focal construct")
        if prediction_mask.dtype != torch.bool:
            raise ValueError("prediction mask must be boolean")
        focal = mixed[int(focal_construct_index)]
        source = focal[edit_index][:, None, :].expand(*expected, -1)
        receiver = focal[None, :, :].expand(expected[0], -1, -1)
        cross = torch.cat([source, receiver], dim=-1)
        actual = torch.cat([base_point_features, cross], dim=-1)
        reference = torch.cat([base_point_features, torch.zeros_like(cross)], dim=-1)
        observed, baseline = self.paired_point_head(actual, reference)
        residual = (observed - baseline).squeeze(-1)
        point = (parent_point + residual).masked_fill(~prediction_mask, 0.0)
        return point, residual


class PuzzleSetMetaContextPointModel(nn.Module):
    """V13-parent-anchored point model for the proposed puzzle-set capability."""

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
        parent_point: torch.Tensor,
    ) -> torch.Tensor:
        if focal_hidden.ndim != 2 or focal_hidden.shape[1] != CONTEXT_WIDTH:
            raise ValueError("focal WT hidden state has invalid shape")
        batch = edit_index.shape[0]
        length = focal_hidden.shape[0]
        expected = (batch, length)
        if (
            signed_distance.shape != expected
            or feature41_point.shape != expected
            or parent_point.shape != expected
        ):
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
                parent_point[..., None],
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
        parent_point: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode_puzzle_set(contexts)
        observed = [context[3].bool() for context in contexts]
        reactivity = [context[1] for context in contexts]
        mixed = self.meta_context.mix_construct_tokens(hidden, observed, reactivity)
        point, residual = self.forward_from_encoded(
            hidden,
            mixed,
            focal_construct_index,
            edit_index,
            signed_distance,
            refs,
            alts,
            feature41_point,
            parent_point,
            prediction_mask,
        )
        return point, residual, mixed

    def forward_from_encoded(
        self,
        hidden: Sequence[torch.Tensor],
        mixed: torch.Tensor,
        focal_construct_index: int,
        edit_index: torch.Tensor,
        signed_distance: torch.Tensor,
        refs: list[str],
        alts: list[str],
        feature41_point: torch.Tensor,
        parent_point: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(hidden) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("encoded puzzle set must contain eight constructs")
        base_features = self.base_point_features(
            hidden[int(focal_construct_index)],
            edit_index,
            signed_distance,
            refs,
            alts,
            feature41_point,
            parent_point,
        )
        point, residual = self.meta_context.point_from_mixed(
            mixed,
            focal_construct_index,
            edit_index,
            base_features,
            parent_point,
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
        return point, residual


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def make_exact_matched_pair(
    *, seed: int, device: str | torch.device = "cpu"
) -> tuple[PuzzleSetMetaContext, PuzzleSetMetaContext]:
    torch.manual_seed(int(seed))
    candidate = PuzzleSetMetaContext(connectivity=FULL_CROSS_CONSTRUCT).to(device)
    null = copy.deepcopy(candidate)
    null.connectivity = POSITION_DERANGED_NULL
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
    *,
    seed: int,
    v14_point_state: dict[str, torch.Tensor],
    device: str | torch.device = "cpu",
) -> tuple[PuzzleSetMetaContextPointModel, PuzzleSetMetaContextPointModel]:
    torch.manual_seed(int(seed))
    candidate = PuzzleSetMetaContextPointModel(connectivity=FULL_CROSS_CONSTRUCT).to(
        device
    )
    load_frozen_v14_encoder(candidate.encoder, v14_point_state)
    null = copy.deepcopy(candidate)
    null.connectivity = POSITION_DERANGED_NULL
    left = dict(candidate.named_parameters())
    right = dict(null.named_parameters())
    if left.keys() != right.keys():
        raise RuntimeError("full puzzle-set candidate and null parameter names differ")
    for name in left:
        if not torch.equal(left[name].detach(), right[name].detach()):
            raise RuntimeError(f"full puzzle-set initialization differs at {name}")
    if parameter_count(candidate) != parameter_count(null):
        raise RuntimeError("full puzzle-set candidate and null counts differ")
    if parameter_count(candidate, trainable_only=True) != parameter_count(
        null, trainable_only=True
    ):
        raise RuntimeError("full puzzle-set trainable counts differ")
    if any(parameter.requires_grad for parameter in candidate.encoder.parameters()):
        raise RuntimeError("puzzle-set V14 parent encoder is not frozen")
    return candidate, null


def _finite_gradients(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise RuntimeError(f"nonfinite puzzle-set point gradient: {name}")


def _puzzle_order(n_puzzles: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n_puzzles))
    random.Random(int(seed) * 100_003 + int(epoch)).shuffle(order)
    return order


def puzzle_balanced_point_loss(
    model: PuzzleSetMetaContextPointModel,
    puzzle_batch: dict,
) -> torch.Tensor:
    """Equal cell mean inside one puzzle after mutant/position balancing."""

    contexts = puzzle_batch.get("contexts")
    cells = puzzle_batch.get("cells")
    if not isinstance(contexts, Sequence) or len(contexts) != (
        EXPECTED_CONSTRUCTS_PER_PUZZLE
    ):
        raise ValueError("puzzle training batch requires eight WT contexts")
    if not isinstance(cells, Sequence) or not 1 <= len(cells) <= (
        EXPECTED_CONSTRUCTS_PER_PUZZLE
    ):
        raise ValueError("puzzle training batch requires qualified method cells")
    focal_indices = [int(cell["focal_construct_index"]) for cell in cells]
    if len(focal_indices) != len(set(focal_indices)) or not set(focal_indices) <= set(
        range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
    ):
        raise ValueError("puzzle training cells must use unique focal constructs")

    hidden = model.encode_puzzle_set(contexts)
    observed = [context[3].bool() for context in contexts]
    reactivity = [context[1] for context in contexts]
    mixed = model.meta_context.mix_construct_tokens(hidden, observed, reactivity)
    cell_losses = []
    for cell in cells:
        qualified = cell["qualified_mask"]
        if qualified.dtype != torch.bool or not bool(qualified.any()):
            raise ValueError("every puzzle training cell needs a qualified target")
        point, _residual = model.forward_from_encoded(
            hidden,
            mixed,
            int(cell["focal_construct_index"]),
            cell["edit_index"],
            cell["signed_distance"],
            cell["refs"],
            cell["alts"],
            cell["feature41_point"],
            cell["parent_point"],
            cell["prediction_mask"],
        )
        cell_losses.append(
            method_cell_balanced_l1(
                point,
                cell["target"],
                qualified,
                cell["wt"],
            )
        )
    return torch.stack(cell_losses).mean()


def fit_puzzle_set_point_model(
    model: PuzzleSetMetaContextPointModel,
    puzzle_batches: Sequence[dict],
    *,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    """Fit the paired point residual without overwriting cross pretraining.

    Epoch zero updates only the point head. The remaining epochs keep the head
    at its original learning rate while the context layers inherit the exact
    masked-WT pretraining learning rate. Target exposure and optimizer-step
    counts remain unchanged.
    """

    if not puzzle_batches:
        raise ValueError("puzzle-set point training requires outer-train puzzles")
    if epochs < 1:
        raise ValueError("puzzle-set point training requires a positive epoch count")
    torch.manual_seed(int(seed) + 1_500_000)
    model.train()
    model.encoder.eval()
    context_named = [
        (name, parameter)
        for name, parameter in model.meta_context.named_parameters()
        if not name.startswith("point_head.")
    ]
    context_parameters = [parameter for _name, parameter in context_named]
    head_parameters = list(model.meta_context.point_head.parameters())
    if (
        not context_parameters
        or not head_parameters
        or any(not parameter.requires_grad for parameter in context_parameters)
        or any(not parameter.requires_grad for parameter in head_parameters)
        or any(parameter.requires_grad for parameter in model.encoder.parameters())
    ):
        raise RuntimeError("puzzle-set trainable parameter boundary changed")
    context_before_warmup = {
        name: parameter.detach().clone() for name, parameter in context_named
    }
    for parameter in context_parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    optimizer = torch.optim.Adam(
        [
            {"params": head_parameters, "lr": POINT_HEAD_LR},
            {"params": context_parameters, "lr": POINT_CONTEXT_LR},
        ],
        weight_decay=0.0,
    )
    history: list[float] = []
    optimizer_steps = 0
    head_update_steps = 0
    context_update_steps = 0
    warmup_context_unchanged = False
    try:
        for epoch in range(int(epochs)):
            if epoch == POINT_HEAD_WARMUP_EPOCHS:
                warmup_context_unchanged = all(
                    torch.equal(context_before_warmup[name], parameter.detach())
                    for name, parameter in context_named
                )
                if not warmup_context_unchanged:
                    raise RuntimeError(
                        "puzzle-set point warmup changed frozen context layers"
                    )
                for parameter in context_parameters:
                    parameter.requires_grad_(True)
            losses = []
            for index in _puzzle_order(len(puzzle_batches), seed, epoch):
                optimizer.zero_grad(set_to_none=True)
                loss = puzzle_balanced_point_loss(model, puzzle_batches[index])
                loss.backward()
                _finite_gradients(model)
                joint = epoch >= POINT_HEAD_WARMUP_EPOCHS
                if not joint and any(
                    parameter.grad is not None for parameter in context_parameters
                ):
                    raise RuntimeError("puzzle-set warmup produced a context gradient")
                active_parameters = head_parameters + (
                    context_parameters if joint else []
                )
                torch.nn.utils.clip_grad_norm_(active_parameters, POINT_GRADIENT_CLIP)
                optimizer.step()
                optimizer_steps += 1
                head_update_steps += 1
                context_update_steps += int(joint)
                losses.append(float(loss.detach().cpu()))
            history.append(float(np.mean(losses)))
        if not warmup_context_unchanged:
            warmup_context_unchanged = all(
                torch.equal(context_before_warmup[name], parameter.detach())
                for name, parameter in context_named
            )
    finally:
        for parameter in context_parameters:
            parameter.requires_grad_(True)
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("puzzle-set point history is incomplete or nonfinite")
    expected_steps = int(epochs) * len(puzzle_batches)
    expected_context_steps = max(int(epochs) - POINT_HEAD_WARMUP_EPOCHS, 0) * len(
        puzzle_batches
    )
    if (
        optimizer_steps != expected_steps
        or head_update_steps != expected_steps
        or context_update_steps != expected_context_steps
    ):
        raise RuntimeError("puzzle-set point update accounting changed")
    return {
        "history": history,
        "optimizer_steps": optimizer_steps,
        "head_update_steps": head_update_steps,
        "context_update_steps": context_update_steps,
        "target_exposures_per_available_cell": int(epochs),
        "head_only_warmup_epochs": POINT_HEAD_WARMUP_EPOCHS,
        "head_learning_rate": POINT_HEAD_LR,
        "context_learning_rate": POINT_CONTEXT_LR,
        "gradient_clip": POINT_GRADIENT_CLIP,
        "warmup_context_unchanged": warmup_context_unchanged,
        "best_epoch_selection_performed": False,
    }
