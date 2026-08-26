#!/usr/bin/env python3
"""Outer-train-only masked-WT pretraining for puzzle-set context layers."""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v14 import (
    MASK_FRACTION as V14_MASK_FRACTION,
    deterministic_corruption_mask,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    CONTEXT_WIDTH,
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    PuzzleSetMetaContextPointModel,
    parameter_count,
)


EXPECTED_DECODER_PARAMETERS = 769
EXPECTED_CONTEXT_PRETRAINING_PARAMETERS = 857_600
EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS = 858_369
PRETRAINING_MASK_FRACTION = V14_MASK_FRACTION


class PuzzleSetWTDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.LayerNorm(CONTEXT_WIDTH),
            nn.Linear(CONTEXT_WIDTH, 1),
        )

    def forward(self, focal_mixed: torch.Tensor) -> torch.Tensor:
        if focal_mixed.ndim != 2 or focal_mixed.shape[1] != CONTEXT_WIDTH:
            raise ValueError("puzzle-set WT decoder input is misaligned")
        return self.decoder(focal_mixed).squeeze(-1)


def make_exact_decoder_pair(
    *, seed: int, device: str | torch.device
) -> tuple[PuzzleSetWTDecoder, PuzzleSetWTDecoder]:
    torch.manual_seed(int(seed) + 1_600_000)
    candidate = PuzzleSetWTDecoder().to(device)
    null = copy.deepcopy(candidate)
    if (
        parameter_count(candidate) != EXPECTED_DECODER_PARAMETERS
        or parameter_count(null) != EXPECTED_DECODER_PARAMETERS
    ):
        raise RuntimeError("puzzle-set WT decoder parameter count changed")
    for left, right in zip(candidate.parameters(), null.parameters()):
        if not torch.equal(left.detach(), right.detach()):
            raise RuntimeError("puzzle-set WT decoder initialization differs")
    return candidate, null


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _assert_snapshot(
    expected: dict[str, torch.Tensor], module: nn.Module, label: str
) -> None:
    actual = module.state_dict()
    if expected.keys() != actual.keys():
        raise RuntimeError(f"puzzle-set {label} state names changed")
    for name, value in expected.items():
        if not torch.equal(value, actual[name].detach().cpu()):
            raise RuntimeError(f"puzzle-set pretraining changed frozen {label}: {name}")


def context_pretraining_parameters(
    model: PuzzleSetMetaContextPointModel,
) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.meta_context.named_parameters()
        if not name.startswith("point_head.")
    ]


def freeze_pretraining_decoder(decoder: PuzzleSetWTDecoder) -> None:
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _visible_puzzle_inputs(
    contexts: Sequence[tuple[torch.Tensor, ...]],
    *,
    focal_index: int,
    corruption_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    reactivity = [context[1] for context in contexts]
    observed = [context[3].bool() for context in contexts]
    reactivity[focal_index] = reactivity[focal_index].masked_fill(corruption_mask, 0.0)
    observed[focal_index] = observed[focal_index] & ~corruption_mask
    return reactivity, observed


def _unmasked_hidden_cache(
    model: PuzzleSetMetaContextPointModel,
    batches: Sequence[dict[str, Any]],
) -> list[list[torch.Tensor]]:
    model.encoder.eval()
    output = []
    with torch.no_grad():
        for batch in batches:
            output.append(
                [model.encoder(context).detach() for context in batch["contexts"]]
            )
    return output


def puzzle_set_wt_reconstruction_loss(
    model: PuzzleSetMetaContextPointModel,
    decoder: PuzzleSetWTDecoder,
    contexts: Sequence[tuple[torch.Tensor, ...]],
    unmasked_hidden: Sequence[torch.Tensor],
    *,
    puzzle_index: int,
    epoch: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    """Mask one focal WT at a time and reconstruct it from visible inputs."""

    if (
        len(contexts) != EXPECTED_CONSTRUCTS_PER_PUZZLE
        or len(unmasked_hidden) != EXPECTED_CONSTRUCTS_PER_PUZZLE
    ):
        raise ValueError("puzzle-set pretraining requires eight WT constructs")
    losses = []
    eligible = 0
    for focal_index, context in enumerate(contexts):
        if int(context[3].bool().sum()) < 2:
            continue
        corruption = deterministic_corruption_mask(
            context[3],
            seed=int(seed),
            epoch=int(epoch),
            construct_index=(
                int(puzzle_index) * EXPECTED_CONSTRUCTS_PER_PUZZLE + focal_index
            ),
        )
        with torch.no_grad():
            focal_hidden = model.encoder(context, corruption).detach()
        hidden = list(unmasked_hidden)
        hidden[focal_index] = focal_hidden
        reactivity, observed = _visible_puzzle_inputs(
            contexts,
            focal_index=focal_index,
            corruption_mask=corruption,
        )
        mixed = model.meta_context.mix_construct_tokens(hidden, observed, reactivity)
        prediction = decoder(mixed[focal_index])
        losses.append(torch.abs(prediction[corruption] - context[1][corruption]).mean())
        eligible += 1
    if not losses:
        raise ValueError("puzzle-set pretraining puzzle has no eligible WT target")
    return torch.stack(losses).mean(), eligible


def fit_puzzle_set_wt_pretraining(
    model: PuzzleSetMetaContextPointModel,
    decoder: PuzzleSetWTDecoder,
    puzzle_batches: Sequence[dict[str, Any]],
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Train context layers only; encoder and parent-anchored point stay fixed."""

    if epochs < 1 or not puzzle_batches:
        raise ValueError("puzzle-set WT pretraining requires batches and epochs")
    for batch in puzzle_batches:
        if set(batch) != {"puzzle", "contexts"}:
            raise ValueError("puzzle-set WT pretraining accepts WT contexts only")
        if len(batch["contexts"]) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError("puzzle-set WT pretraining puzzle is incomplete")
    encoder_before = _snapshot(model.encoder)
    point_before = _snapshot(model.meta_context.point_head)
    context_before = {
        name: value.detach().cpu().clone()
        for name, value in model.meta_context.state_dict().items()
        if not name.startswith("point_head.")
    }
    point_requires = [
        parameter.requires_grad
        for parameter in model.meta_context.point_head.parameters()
    ]
    for parameter in model.meta_context.point_head.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("puzzle-set pretraining encoder boundary changed")
    context_trainable = context_pretraining_parameters(model)
    decoder_trainable = list(decoder.parameters())
    trainable = context_trainable + decoder_trainable
    if not trainable or any(not parameter.requires_grad for parameter in trainable):
        raise RuntimeError("puzzle-set pretraining parameter boundary changed")
    if (
        sum(parameter.numel() for parameter in context_trainable)
        != EXPECTED_CONTEXT_PRETRAINING_PARAMETERS
        or sum(parameter.numel() for parameter in decoder_trainable)
        != EXPECTED_DECODER_PARAMETERS
        or sum(parameter.numel() for parameter in trainable)
        != EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("puzzle-set pretraining trainable count changed")

    torch.manual_seed(int(seed) + 1_600_000)
    model.meta_context.train()
    model.meta_context.point_head.eval()
    decoder.train()
    hidden_cache = _unmasked_hidden_cache(model, puzzle_batches)
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.01)
    history: list[float] = []
    eligible_counts: set[int] = set()
    try:
        for epoch in range(int(epochs)):
            order = list(range(len(puzzle_batches)))
            random.Random(int(seed) * 100_003 + int(epoch)).shuffle(order)
            losses = []
            for puzzle_index in order:
                loss, eligible = puzzle_set_wt_reconstruction_loss(
                    model,
                    decoder,
                    puzzle_batches[puzzle_index]["contexts"],
                    hidden_cache[puzzle_index],
                    puzzle_index=puzzle_index,
                    epoch=epoch,
                    seed=seed,
                )
                optimizer.zero_grad()
                loss.backward()
                for name, parameter in list(
                    model.meta_context.named_parameters()
                ) + list(decoder.named_parameters()):
                    if parameter.grad is not None and not bool(
                        torch.isfinite(parameter.grad).all()
                    ):
                        raise RuntimeError(
                            f"nonfinite puzzle-set WT pretraining gradient in {name}"
                        )
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                eligible_counts.add(int(eligible))
            history.append(float(np.mean(losses)))
    finally:
        for parameter, required in zip(
            model.meta_context.point_head.parameters(), point_requires
        ):
            parameter.requires_grad_(required)
            parameter.grad = None
    if len(history) != int(epochs) or not np.isfinite(history).all():
        raise RuntimeError("puzzle-set WT pretraining history is invalid")
    _assert_snapshot(encoder_before, model.encoder, "encoder")
    _assert_snapshot(point_before, model.meta_context.point_head, "point head")
    changed = any(
        not torch.equal(value, model.meta_context.state_dict()[name].detach().cpu())
        for name, value in context_before.items()
    )
    if not changed:
        raise RuntimeError("puzzle-set WT pretraining did not change context layers")
    if any(parameter.grad is not None for parameter in model.encoder.parameters()):
        raise RuntimeError("puzzle-set WT pretraining produced encoder gradients")
    freeze_pretraining_decoder(decoder)
    return {
        "history": history,
        "optimizer_steps": int(epochs) * len(puzzle_batches),
        "trainable_parameter_count": EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS,
        "eligible_construct_counts": sorted(eligible_counts),
        "mask_fraction": float(PRETRAINING_MASK_FRACTION),
        "context_layers_changed": True,
        "encoder_changed": False,
        "point_head_changed": False,
        "decoder_frozen_downstream": all(
            not parameter.requires_grad for parameter in decoder.parameters()
        ),
        "mutant_outcome_used": False,
    }
