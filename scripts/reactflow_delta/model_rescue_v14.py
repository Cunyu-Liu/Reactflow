#!/usr/bin/env python3
"""Masked-WT-profile pretraining model for Model Rescue v14."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v11 import (
    V11ContextBlock,
    method_cell_balanced_l1,
    mutation_one_hot,
)


PRETRAINED_CANDIDATE = "v14_masked_wt_profile_pretrained_feature41_anchor"
FROM_SCRATCH_NULL = "v14_from_scratch_feature41_anchor"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v14_prediction.v1"
FOLD_SCHEMA = "reactflow_delta.model_rescue_v14_fold.v1"

INPUT_CHANNELS = 11
CONTEXT_WIDTH = 256
ATTENTION_HEADS = 8
CONTEXT_BLOCKS = 6
FFN_WIDTH = 1024
HEAD_WIDTH = 384
HEAD_LAYERS = 2
RELATIVE_DISTANCE_WINDOW = 256
DROPOUT = 0.1
MASK_FRACTION = 0.40

EXPECTED_ENCODER_PARAMETERS = 4_767_280
EXPECTED_DECODER_PARAMETERS = 769
EXPECTED_RESIDUAL_HEAD_PARAMETERS = 349_825
EXPECTED_TOTAL_PARAMETERS = 5_117_874
EXPECTED_DOWNSTREAM_PARAMETERS = 5_117_105


class V14PointModel(nn.Module):
    """Feature41-anchored residual model with a task-matched WT decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(INPUT_CHANNELS, CONTEXT_WIDTH)
        self.input_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.blocks = nn.ModuleList(
            V11ContextBlock(
                width=CONTEXT_WIDTH,
                heads=ATTENTION_HEADS,
                ffn_width=FFN_WIDTH,
                dropout=DROPOUT,
                relative_window=RELATIVE_DISTANCE_WINDOW,
            )
            for _ in range(CONTEXT_BLOCKS)
        )
        self.output_norm = nn.LayerNorm(CONTEXT_WIDTH)
        self.pretraining_decoder = nn.Sequential(
            nn.LayerNorm(CONTEXT_WIDTH),
            nn.Linear(CONTEXT_WIDTH, 1),
        )
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

    def encode(
        self,
        context: tuple[torch.Tensor, ...],
        corruption_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(context) != 6:
            raise ValueError("V14 context must contain six aligned tensors")
        sequence, reactivity, precision, observed, position, region = context
        length = sequence.shape[0]
        if sequence.shape != (length, 4) or region.shape != (length, 2):
            raise ValueError("V14 sequence or region tensor has invalid shape")
        for tensor in (reactivity, precision, observed, position):
            if tensor.shape != (length,):
                raise ValueError("V14 scalar context tensor has invalid shape")
        if corruption_mask is None:
            corruption_mask = torch.zeros(
                length, dtype=torch.bool, device=sequence.device
            )
        if corruption_mask.shape != (length,) or corruption_mask.dtype != torch.bool:
            raise ValueError("V14 corruption mask has invalid shape or dtype")
        if bool((corruption_mask & ~observed.bool()).any()):
            raise ValueError("V14 can mask only WT-observed positions")

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
        if local.shape != (length, INPUT_CHANNELS):
            raise RuntimeError("V14 input channel construction changed")
        state = self.input_norm(self.input_projection(local)).unsqueeze(0)
        attention_keys = (observed.bool() & ~corruption_mask).unsqueeze(0)
        for block in self.blocks:
            state = block(state, attention_keys)
        return self.output_norm(state[0])

    def reconstruct_masked_wt(
        self,
        context: tuple[torch.Tensor, ...],
        corruption_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.pretraining_decoder(
            self.encode(context, corruption_mask)
        ).squeeze(-1)

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
            raise ValueError("V14 hidden representation has invalid shape")
        batch = edit_index.shape[0]
        length = hidden.shape[0]
        expected = (batch, length)
        if signed_distance.shape != expected or prediction_mask.shape != expected:
            raise ValueError("V14 distance or prediction mask has invalid shape")
        if feature41_point.shape != expected:
            raise ValueError("V14 feature41 point has invalid shape")
        if len(refs) != batch or len(alts) != batch:
            raise ValueError("V14 mutation identity count is misaligned")

        source = hidden[edit_index][:, None, :].expand(batch, length, -1)
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
        point = (feature41_point + residual).masked_fill(~prediction_mask, 0.0)
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


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def encoder_parameters(model: V14PointModel) -> Iterable[nn.Parameter]:
    yield from model.input_projection.parameters()
    yield from model.input_norm.parameters()
    yield from model.blocks.parameters()
    yield from model.output_norm.parameters()


def pretraining_parameters(model: V14PointModel) -> list[nn.Parameter]:
    return list(encoder_parameters(model)) + list(model.pretraining_decoder.parameters())


def downstream_parameters(model: V14PointModel) -> list[nn.Parameter]:
    return list(encoder_parameters(model)) + list(model.residual_head.parameters())


def freeze_pretraining_decoder(model: V14PointModel) -> None:
    model.pretraining_decoder.eval()
    for parameter in model.pretraining_decoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def freeze_point_model(model: V14PointModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def module_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def assert_snapshot_equal(
    expected: dict[str, torch.Tensor], module: nn.Module, label: str
) -> None:
    actual = module.state_dict()
    if expected.keys() != actual.keys():
        raise RuntimeError(f"V14 {label} state names changed")
    for name, value in expected.items():
        if not torch.equal(value, actual[name].detach().cpu()):
            raise RuntimeError(f"V14 {label} changed at {name}")


def assert_exact_parameter_contract(model: V14PointModel) -> None:
    encoder = (
        parameter_count(model.input_projection)
        + parameter_count(model.input_norm)
        + parameter_count(model.blocks)
        + parameter_count(model.output_norm)
    )
    observed = {
        "encoder": encoder,
        "decoder": parameter_count(model.pretraining_decoder),
        "residual": parameter_count(model.residual_head),
        "total": parameter_count(model),
    }
    expected = {
        "encoder": EXPECTED_ENCODER_PARAMETERS,
        "decoder": EXPECTED_DECODER_PARAMETERS,
        "residual": EXPECTED_RESIDUAL_HEAD_PARAMETERS,
        "total": EXPECTED_TOTAL_PARAMETERS,
    }
    if observed != expected:
        raise RuntimeError(f"V14 parameter contract changed: {observed}")
    if sum(parameter.numel() for parameter in downstream_parameters(model)) != (
        EXPECTED_DOWNSTREAM_PARAMETERS
    ):
        raise RuntimeError("V14 downstream parameter count changed")


def make_exact_matched_pair(
    *, seed: int, device: str | torch.device
) -> tuple[V14PointModel, V14PointModel]:
    torch.manual_seed(int(seed))
    candidate = V14PointModel().to(device)
    null = copy.deepcopy(candidate)
    assert_exact_parameter_contract(candidate)
    assert_exact_parameter_contract(null)
    assert_exact_initial_match(candidate, null)
    return candidate, null


def assert_exact_initial_match(
    candidate: V14PointModel, null: V14PointModel
) -> None:
    left = dict(candidate.named_parameters())
    right = dict(null.named_parameters())
    if left.keys() != right.keys():
        raise RuntimeError("V14 candidate/null parameter names differ")
    for name in left:
        if left[name].shape != right[name].shape or not torch.equal(
            left[name].detach(), right[name].detach()
        ):
            raise RuntimeError(f"V14 common initialization differs at {name}")


def deterministic_corruption_mask(
    observed: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    construct_index: int,
) -> torch.Tensor:
    if observed.ndim != 1:
        raise ValueError("V14 observed mask must be one-dimensional")
    observed_index = torch.nonzero(
        observed.detach().cpu().bool(), as_tuple=False
    ).flatten()
    n_observed = int(len(observed_index))
    if n_observed < 2:
        raise ValueError("V14 masked reconstruction needs at least two observed WT positions")
    n_masked = min(
        n_observed - 1,
        max(1, int(math.floor(MASK_FRACTION * n_observed + 0.5))),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(seed) * 1_000_003 + int(epoch) * 10_007 + int(construct_index) * 101 + 14
    )
    selected = observed_index[
        torch.randperm(n_observed, generator=generator)[:n_masked]
    ]
    mask = torch.zeros(len(observed), dtype=torch.bool)
    mask[selected] = True
    return mask.to(observed.device)


def finite_gradients(module: nn.Module, stage: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite V14 gradient in {stage}: {name}")


def pretrain_wt_encoder(
    model: V14PointModel,
    contexts: dict[str, tuple[torch.Tensor, ...]],
    *,
    epochs: int,
    seed: int,
) -> list[float]:
    if not contexts:
        raise ValueError("V14 pretraining requires outer-train WT constructs")
    construct_ids = sorted(contexts)
    residual_before = module_snapshot(model.residual_head)
    model.train()
    optimizer = torch.optim.AdamW(
        pretraining_parameters(model), lr=3e-4, weight_decay=0.01
    )
    history: list[float] = []
    for epoch in range(epochs):
        order = list(range(len(construct_ids)))
        random.Random(seed * 100_003 + epoch).shuffle(order)
        losses = []
        for construct_index in order:
            context = contexts[construct_ids[construct_index]]
            corruption = deterministic_corruption_mask(
                context[3],
                seed=seed,
                epoch=epoch,
                construct_index=construct_index,
            )
            prediction = model.reconstruct_masked_wt(context, corruption)
            loss = torch.abs(prediction[corruption] - context[1][corruption]).mean()
            optimizer.zero_grad()
            loss.backward()
            finite_gradients(model, "pretraining")
            if any(parameter.grad is not None for parameter in model.residual_head.parameters()):
                raise RuntimeError("V14 pretraining produced residual-head gradients")
            torch.nn.utils.clip_grad_norm_(pretraining_parameters(model), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("V14 pretraining history is incomplete or nonfinite")
    assert_snapshot_equal(residual_before, model.residual_head, "residual head during pretraining")
    return history


def fit_point_model(
    model: V14PointModel,
    cells: list[dict],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    *,
    epochs: int,
    seed: int,
) -> list[float]:
    torch.manual_seed(seed + 1_400_000)
    freeze_pretraining_decoder(model)
    model.train()
    model.pretraining_decoder.eval()
    optimizer = torch.optim.Adam(
        downstream_parameters(model), lr=1e-3, weight_decay=0.0
    )
    history: list[float] = []
    for epoch in range(epochs):
        order = list(range(len(cells)))
        random.Random(seed * 100_003 + epoch).shuffle(order)
        losses = []
        for index in order:
            cell = cells[index]
            hidden = model.encode(context_cache[cell["construct_id"]])
            point = model.forward_point(
                hidden,
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
                cell["feature41_point"],
            )
            loss = method_cell_balanced_l1(
                point,
                cell["target"],
                cell["qualified_mask"],
                cell["wt"],
            )
            optimizer.zero_grad()
            loss.backward()
            finite_gradients(model, "point")
            if any(
                parameter.grad is not None
                for parameter in model.pretraining_decoder.parameters()
            ):
                raise RuntimeError("V14 downstream training produced decoder gradients")
            torch.nn.utils.clip_grad_norm_(downstream_parameters(model), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("V14 point history is incomplete or nonfinite")
    return history
