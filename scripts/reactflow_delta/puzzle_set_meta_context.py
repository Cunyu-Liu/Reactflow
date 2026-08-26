#!/usr/bin/env python3
"""Outcome-blind cross-construct context proposed for post-V14 routing.

This module is implementation-only.  It does not authorize a scientific run.
The candidate and null share every parameter; their sole functional difference
is whether a focal construct can attend to the other WT constructs in its
puzzle set.
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
BLOCK_DIAGONAL_NULL = "BLOCK_DIAGONAL_NULL"
CONNECTIVITY_MODES = {FULL_CROSS_CONSTRUCT, BLOCK_DIAGONAL_NULL}
POSITION_ALIGNED_OPERATOR = "POSITION_ALIGNED_CROSS_CONSTRUCT_V1"

EXPECTED_CONSTRUCTS_PER_PUZZLE = 8
CONTEXT_WIDTH = 256
ATTENTION_HEADS = 8
V14_LOCAL_FEATURE_WIDTH = 522
BASE_FEATURE_WIDTH = V14_LOCAL_FEATURE_WIDTH + 1
POINT_HEAD_WIDTH = 384
DROPOUT = 0.1
EXPECTED_TOTAL_PARAMETERS = 6_170_417
EXPECTED_TRAINABLE_PARAMETERS = 1_403_137


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
    def _validate_construct(
        hidden: torch.Tensor, observed: torch.Tensor
    ) -> None:
        if hidden.ndim != 2 or hidden.shape[1] != CONTEXT_WIDTH:
            raise ValueError("puzzle-set hidden state has invalid shape")
        if observed.shape != (hidden.shape[0],) or observed.dtype != torch.bool:
            raise ValueError("puzzle-set observed mask has invalid shape or dtype")
        if hidden.shape[0] == 0:
            raise ValueError("puzzle-set construct cannot be empty")

    def align_construct_tokens(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(construct_hidden) != EXPECTED_CONSTRUCTS_PER_PUZZLE or len(
            construct_observed
        ) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("puzzle-set context requires exactly eight constructs")
        lengths = {int(hidden.shape[0]) for hidden in construct_hidden}
        if len(lengths) != 1:
            raise ValueError("puzzle-set constructs do not share one coordinate frame")
        aligned = []
        for hidden, observed in zip(construct_hidden, construct_observed):
            self._validate_construct(hidden, observed)
            aligned.append(
                torch.cat(
                    [hidden, observed.to(hidden.dtype)[:, None]], dim=-1
                )
            )
        return self.construct_projection(torch.stack(aligned, dim=0))

    def mix_construct_tokens(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        # MultiheadAttention treats full-sequence position as the batch axis
        # and the eight constructs as the unordered token axis. Information
        # therefore crosses constructs only at the same registered coordinate.
        tokens = self.align_construct_tokens(
            construct_hidden, construct_observed
        ).permute(1, 0, 2)
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
        return self.output_norm(tokens).permute(1, 0, 2)

    def forward(
        self,
        construct_hidden: Sequence[torch.Tensor],
        construct_observed: Sequence[torch.Tensor],
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

        mixed = self.mix_construct_tokens(construct_hidden, construct_observed)
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
        if mixed.ndim != 3 or mixed.shape[0] != EXPECTED_CONSTRUCTS_PER_PUZZLE or (
            mixed.shape[2] != CONTEXT_WIDTH
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
        residual = self.point_head(
            torch.cat([base_point_features, source, receiver], dim=-1)
        ).squeeze(-1)
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
        mixed = self.meta_context.mix_construct_tokens(hidden, observed)
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
    *,
    seed: int,
    v14_point_state: dict[str, torch.Tensor],
    device: str | torch.device = "cpu",
) -> tuple[PuzzleSetMetaContextPointModel, PuzzleSetMetaContextPointModel]:
    torch.manual_seed(int(seed))
    candidate = PuzzleSetMetaContextPointModel(
        connectivity=FULL_CROSS_CONSTRUCT
    ).to(device)
    load_frozen_v14_encoder(candidate.encoder, v14_point_state)
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
    mixed = model.meta_context.mix_construct_tokens(hidden, observed)
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
) -> list[float]:
    """Visit every puzzle once per epoch under an exactly balanced objective."""

    if not puzzle_batches:
        raise ValueError("puzzle-set point training requires outer-train puzzles")
    if epochs < 1:
        raise ValueError("puzzle-set point training requires a positive epoch count")
    torch.manual_seed(int(seed) + 1_500_000)
    model.train()
    model.encoder.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(
        parameter.requires_grad for parameter in model.encoder.parameters()
    ):
        raise RuntimeError("puzzle-set trainable parameter boundary changed")
    optimizer = torch.optim.Adam(trainable, lr=1e-3, weight_decay=0.0)
    history: list[float] = []
    for epoch in range(int(epochs)):
        losses = []
        for index in _puzzle_order(len(puzzle_batches), seed, epoch):
            optimizer.zero_grad()
            loss = puzzle_balanced_point_loss(model, puzzle_batches[index])
            loss.backward()
            _finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("puzzle-set point history is incomplete or nonfinite")
    return history
