#!/usr/bin/env python3
"""Feature41-anchored shared WT/exact-mutant re-encoding for V13.

The primary and nested null have identical trainable state.  The only forward
path difference is the sequence supplied to the shared second encoder pass:
the primary receives the registered exact-mutant sequence, while the null
receives an identically shaped WT replay.  Neither model accepts a held target,
target error, qualified target mask, method identifier, or puzzle identifier.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v11 import (
    ATTENTION_HEADS,
    CONTEXT_BLOCKS,
    CONTEXT_WIDTH,
    DROPOUT,
    FFN_WIDTH,
    HEAD_WIDTH,
    V11ContextBlock,
    method_cell_balanced_l1,
    mutation_one_hot,
)
from scripts.reactflow_delta.run_p3_lrso_v3 import ALPHA


PRIMARY_CANDIDATE = "v13_feature41_anchored_exact_mutant_contrast"
MATCHED_NULL = "v13_feature41_anchored_wt_replay_null"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v13_prediction.v1"
SECOND_PASS_EXACT = "exact_mutant"
SECOND_PASS_WT_REPLAY = "wt_replay"
SecondPassMode = Literal["exact_mutant", "wt_replay"]
MUTANT_MICROBATCH = 64
EXPECTED_POINT_PARAMETERS = 2_064_737


def _canonical_base(value: str) -> str:
    return value.upper().replace("T", "U")


def _base_index(value: str) -> int:
    canonical = _canonical_base(value)
    if canonical not in ALPHA:
        raise ValueError(f"unsupported nucleotide {value!r}")
    return int(ALPHA[canonical])


def build_second_pass_sequences(
    wt_sequence: torch.Tensor,
    edit_index: torch.Tensor,
    refs: list[str],
    alts: list[str],
    *,
    mode: SecondPassMode,
) -> torch.Tensor:
    """Return BxLx4 WT replays or exact single-nucleotide mutant sequences."""

    if wt_sequence.ndim != 2 or wt_sequence.shape[1] != 4:
        raise ValueError("WT sequence must have shape Lx4")
    if edit_index.ndim != 1 or len(refs) != len(edit_index) or len(alts) != len(
        edit_index
    ):
        raise ValueError("mutation coordinates and identities are misaligned")
    length = wt_sequence.shape[0]
    if torch.any(edit_index < 0) or torch.any(edit_index >= length):
        raise ValueError("corrected mutation coordinate is outside the construct")
    if mode not in (SECOND_PASS_EXACT, SECOND_PASS_WT_REPLAY):
        raise ValueError(f"unknown V13 second-pass mode {mode!r}")

    output = wt_sequence.unsqueeze(0).expand(len(edit_index), -1, -1).clone()
    observed_ref = wt_sequence.argmax(dim=-1)[edit_index].tolist()
    expected_ref = [_base_index(value) for value in refs]
    if list(map(int, observed_ref)) != expected_ref:
        raise ValueError("registered ref does not match the corrected WT sequence")
    if mode == SECOND_PASS_EXACT:
        alt_index = torch.tensor(
            [_base_index(value) for value in alts],
            device=wt_sequence.device,
            dtype=torch.long,
        )
        output[torch.arange(len(edit_index), device=wt_sequence.device), edit_index] = 0
        output[
            torch.arange(len(edit_index), device=wt_sequence.device),
            edit_index,
            alt_index,
        ] = 1
    return output


class V13PointModel(nn.Module):
    """Compact shared counterfactual encoder with a fixed feature41 anchor."""

    def __init__(self, *, second_pass_mode: SecondPassMode) -> None:
        super().__init__()
        if second_pass_mode not in (SECOND_PASS_EXACT, SECOND_PASS_WT_REPLAY):
            raise ValueError("V13 second-pass mode is not frozen")
        self.second_pass_mode: SecondPassMode = second_pass_mode
        self.input_projection = nn.Linear(10, CONTEXT_WIDTH)
        self.input_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.blocks = nn.ModuleList(
            V11ContextBlock() for _ in range(CONTEXT_BLOCKS)
        )
        self.output_norm = nn.LayerNorm(CONTEXT_WIDTH)
        point_input_width = 4 * CONTEXT_WIDTH + 1 + 8 + 1
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

    def encode_sequence_batch(
        self,
        sequence: torch.Tensor,
        wt_context: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Encode one sequence or a sequence batch with shared WT measurements."""

        if len(wt_context) != 6:
            raise ValueError("V13 context must contain six aligned tensors")
        wt_sequence, reactivity, precision, observed, position, region = wt_context
        if wt_sequence.ndim != 2 or wt_sequence.shape[1] != 4:
            raise ValueError("V13 WT context sequence must have shape Lx4")
        length = wt_sequence.shape[0]
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 3 or sequence.shape[1:] != (length, 4):
            raise ValueError("V13 encoded sequence must have shape Lx4 or BxLx4")
        if region.shape != (length, 2):
            raise ValueError("V13 region tensor has invalid shape")
        for tensor in (reactivity, precision, observed, position):
            if tensor.shape != (length,):
                raise ValueError("V13 scalar context tensor has invalid shape")

        batch = sequence.shape[0]
        normalized_position = position / max(length - 1, 1)
        fixed = torch.cat(
            [
                reactivity[:, None],
                precision[:, None],
                observed[:, None],
                normalized_position[:, None],
                region,
            ],
            dim=-1,
        ).unsqueeze(0).expand(batch, -1, -1)
        local = torch.cat([sequence, fixed], dim=-1)
        state = self.input_norm(self.input_projection(local))
        observed_mask = observed.bool().unsqueeze(0).expand(batch, -1)
        for block in self.blocks:
            state = block(state, observed_mask)
        return self.output_norm(state)

    def encode_wt(self, wt_context: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.encode_sequence_batch(wt_context[0], wt_context)[0]

    def encode_second_pass(
        self,
        wt_context: tuple[torch.Tensor, ...],
        edit_index: torch.Tensor,
        refs: list[str],
        alts: list[str],
        *,
        microbatch: int = MUTANT_MICROBATCH,
    ) -> torch.Tensor:
        if microbatch <= 0:
            raise ValueError("V13 mutant microbatch must be positive")
        sequences = build_second_pass_sequences(
            wt_context[0],
            edit_index,
            refs,
            alts,
            mode=self.second_pass_mode,
        )
        encoded = []
        for start in range(0, len(sequences), microbatch):
            encoded.append(
                self.encode_sequence_batch(sequences[start : start + microbatch], wt_context)
            )
        return torch.cat(encoded, dim=0)

    def encode_paired_passes(
        self,
        wt_context: tuple[torch.Tensor, ...],
        edit_index: torch.Tensor,
        refs: list[str],
        alts: list[str],
        *,
        microbatch: int = MUTANT_MICROBATCH,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode matched WT/second-pass rows in the same batched operations."""

        if microbatch <= 0:
            raise ValueError("V13 mutant microbatch must be positive")
        second_sequences = build_second_pass_sequences(
            wt_context[0],
            edit_index,
            refs,
            alts,
            mode=self.second_pass_mode,
        )
        wt_rows = []
        second_rows = []
        for start in range(0, len(second_sequences), microbatch):
            second = second_sequences[start : start + microbatch]
            wt = wt_context[0].unsqueeze(0).expand(len(second), -1, -1)
            paired = self.encode_sequence_batch(torch.cat([wt, second], dim=0), wt_context)
            wt_encoded, second_encoded = paired.split(len(second), dim=0)
            wt_rows.append(wt_encoded)
            second_rows.append(second_encoded)
        return torch.cat(wt_rows, dim=0), torch.cat(second_rows, dim=0)

    def forward_point_and_features(
        self,
        wt_context: tuple[torch.Tensor, ...],
        edit_index: torch.Tensor,
        signed_distance: torch.Tensor,
        refs: list[str],
        alts: list[str],
        prediction_mask: torch.Tensor,
        feature41_point: torch.Tensor,
        *,
        microbatch: int = MUTANT_MICROBATCH,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = edit_index.shape[0]
        length = wt_context[0].shape[0]
        expected = (batch, length)
        if signed_distance.shape != expected or prediction_mask.shape != expected:
            raise ValueError("V13 distance or prediction mask has invalid shape")
        if feature41_point.shape != expected:
            raise ValueError("V13 feature41 point has invalid shape")
        if len(refs) != batch or len(alts) != batch:
            raise ValueError("V13 mutation identities are misaligned")

        wt_hidden, second_hidden = self.encode_paired_passes(
            wt_context,
            edit_index,
            refs,
            alts,
            microbatch=microbatch,
        )
        row = torch.arange(batch, device=edit_index.device)
        wt_source = wt_hidden[row, edit_index]
        second_source = second_hidden[
            row, edit_index
        ]
        delta_source = second_source - wt_source
        delta_receiver = second_hidden - wt_hidden
        normalized_distance = signed_distance / max(length - 1, 1)
        mutation = mutation_one_hot(refs, alts, wt_hidden.device)
        features = torch.cat(
            [
                wt_source[:, None, :].expand(batch, length, -1),
                wt_hidden,
                delta_source[:, None, :].expand(batch, length, -1),
                delta_receiver,
                normalized_distance[..., None],
                mutation[:, None, :].expand(batch, length, -1),
                feature41_point[..., None],
            ],
            dim=-1,
        )
        residual = self.residual_head(features).squeeze(-1)
        point = feature41_point + residual
        point = point.masked_fill(~prediction_mask, 0.0)
        same = torch.tensor(
            [_canonical_base(ref) == _canonical_base(alt) for ref, alt in zip(refs, alts)],
            dtype=torch.bool,
            device=point.device,
        )
        point = point.masked_fill(same[:, None], 0.0)
        return point, features

    def forward_point(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_point_and_features(*args, **kwargs)[0]


def make_exact_matched_pair(
    *, seed: int, device: str | torch.device
) -> tuple[V13PointModel, V13PointModel]:
    torch.manual_seed(int(seed))
    candidate = V13PointModel(second_pass_mode=SECOND_PASS_EXACT).to(device)
    null = copy.deepcopy(candidate)
    null.second_pass_mode = SECOND_PASS_WT_REPLAY
    assert_exact_trainable_match(candidate, null)
    return candidate, null


def assert_exact_trainable_match(
    candidate: V13PointModel, null: V13PointModel
) -> None:
    left = dict(candidate.named_parameters())
    right = dict(null.named_parameters())
    if left.keys() != right.keys():
        raise RuntimeError("V13 candidate/null parameter names differ")
    for name in left:
        if left[name].shape != right[name].shape:
            raise RuntimeError(f"V13 candidate/null parameter shape differs: {name}")
        if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
            raise RuntimeError(f"V13 candidate/null initial state differs: {name}")
    if candidate.second_pass_mode != SECOND_PASS_EXACT:
        raise RuntimeError("V13 candidate exact-mutant input is not frozen")
    if null.second_pass_mode != SECOND_PASS_WT_REPLAY:
        raise RuntimeError("V13 null WT-replay input is not frozen")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def freeze_point_model(model: V13PointModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


__all__ = [
    "PRIMARY_CANDIDATE",
    "MATCHED_NULL",
    "PREDICTION_SCHEMA",
    "SECOND_PASS_EXACT",
    "SECOND_PASS_WT_REPLAY",
    "MUTANT_MICROBATCH",
    "EXPECTED_POINT_PARAMETERS",
    "V13PointModel",
    "build_second_pass_sequences",
    "make_exact_matched_pair",
    "assert_exact_trainable_match",
    "trainable_parameter_count",
    "freeze_point_model",
    "method_cell_balanced_l1",
]
