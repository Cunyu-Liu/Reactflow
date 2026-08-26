#!/usr/bin/env python3
"""Outer-train-only retention diagnostic for puzzle-set WT context learning."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.reactflow_delta.puzzle_set_meta_context import (
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
)
from scripts.reactflow_delta.puzzle_set_meta_context_pretraining import (
    PuzzleSetWTDecoder,
    puzzle_set_wt_reconstruction_loss,
)


RETENTION_DIAGNOSTIC_EPOCH = 200
RETENTION_SCHEMA = "reactflow_delta.puzzle_set_context_retention.proposed.v1"


def snapshot_context_for_retention(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture only the set context state; the point head is not diagnostic input."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.meta_context.state_dict().items()
        if not name.startswith("point_head.")
    }


def _load_context_snapshot(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    state = model.meta_context.state_dict()
    expected = {name for name in state if not name.startswith("point_head.")}
    if set(snapshot) != expected:
        raise ValueError("retention context snapshot has an inexact state universe")
    for name in expected:
        source = snapshot[name]
        target = state[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"retention context snapshot is incompatible at {name}")
        state[name] = source.to(device=target.device)
    model.meta_context.load_state_dict(state, strict=True)


def _validated_outer_train_batches(
    puzzle_batches: Sequence[dict[str, Any]], *, held_puzzle: str
) -> list[dict[str, Any]]:
    if not puzzle_batches or not str(held_puzzle):
        raise ValueError(
            "retention diagnostic requires outer-train puzzles and held ID"
        )
    observed = set()
    validated = []
    for batch in puzzle_batches:
        if not isinstance(batch, dict) or set(batch) != {"puzzle", "contexts"}:
            raise ValueError(
                "retention diagnostic accepts outer-train {puzzle, contexts} only"
            )
        puzzle = str(batch["puzzle"])
        if not puzzle or puzzle == str(held_puzzle):
            raise ValueError("retention diagnostic cannot include the held puzzle")
        if puzzle in observed:
            raise ValueError("retention diagnostic puzzle IDs must be unique")
        contexts = batch["contexts"]
        if not isinstance(contexts, Sequence) or len(contexts) != (
            EXPECTED_CONSTRUCTS_PER_PUZZLE
        ):
            raise ValueError("retention diagnostic requires eight WT contexts")
        observed.add(puzzle)
        validated.append(batch)
    return sorted(validated, key=lambda batch: str(batch["puzzle"]))


def _evaluate_state(
    model: nn.Module,
    decoder: PuzzleSetWTDecoder,
    puzzle_batches: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    output = {}
    with torch.no_grad():
        for puzzle_index, batch in enumerate(puzzle_batches):
            contexts = batch["contexts"]
            unmasked_hidden = [model.encoder(context).detach() for context in contexts]
            loss, eligible = puzzle_set_wt_reconstruction_loss(
                model,
                decoder,
                contexts,
                unmasked_hidden,
                puzzle_index=puzzle_index,
                epoch=RETENTION_DIAGNOSTIC_EPOCH,
                seed=int(seed),
            )
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise RuntimeError("retention diagnostic produced nonfinite WT L1")
            output[str(batch["puzzle"])] = {
                "wt_reconstruction_l1": value,
                "eligible_constructs": int(eligible),
            }
    return output


def _module_observation(module: nn.Module) -> dict[str, Any]:
    return {
        "state": {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        },
        "gradients": [
            None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for parameter in module.parameters()
        ],
        "modes": [submodule.training for submodule in module.modules()],
    }


def _assert_module_observation(
    module: nn.Module, expected: dict[str, Any], label: str
) -> None:
    if [submodule.training for submodule in module.modules()] != expected["modes"]:
        raise RuntimeError(f"retention diagnostic changed {label} mode")
    actual_state = module.state_dict()
    for name, value in expected["state"].items():
        if not torch.equal(value, actual_state[name].detach().cpu()):
            raise RuntimeError(f"retention diagnostic changed {label} state at {name}")
    for parameter, gradient in zip(module.parameters(), expected["gradients"]):
        if gradient is None:
            if parameter.grad is not None:
                raise RuntimeError(f"retention diagnostic created {label} gradients")
        elif parameter.grad is None or not torch.equal(
            gradient, parameter.grad.detach().cpu()
        ):
            raise RuntimeError(f"retention diagnostic changed {label} gradients")


def evaluate_context_retention(
    *,
    arm: str,
    post_point_model: nn.Module,
    final_frozen_decoder: PuzzleSetWTDecoder,
    initial_context_snapshot: dict[str, torch.Tensor],
    post_pretraining_context_snapshot: dict[str, torch.Tensor],
    puzzle_batches: Sequence[dict[str, Any]],
    held_puzzle: str,
    seed: int,
    training_epochs: int,
) -> dict[str, Any]:
    """Evaluate three context states without targets, selection, or mutation."""

    if arm not in {"candidate", "null"}:
        raise ValueError("retention diagnostic arm must be candidate or null")
    if not 1 <= int(training_epochs) <= RETENTION_DIAGNOSTIC_EPOCH:
        raise ValueError("retention diagnostic must use a post-training mask epoch")
    batches = _validated_outer_train_batches(
        puzzle_batches, held_puzzle=str(held_puzzle)
    )
    if any(parameter.requires_grad for parameter in final_frozen_decoder.parameters()):
        raise ValueError("retention diagnostic requires one final frozen decoder")

    model_before = _module_observation(post_point_model)
    decoder_before = _module_observation(final_frozen_decoder)
    diagnostic_model = copy.deepcopy(post_point_model)
    post_point_context_snapshot = snapshot_context_for_retention(post_point_model)

    decoder_modes = [module.training for module in final_frozen_decoder.modules()]
    try:
        final_frozen_decoder.eval()
        _load_context_snapshot(diagnostic_model, initial_context_snapshot)
        initial = _evaluate_state(
            diagnostic_model, final_frozen_decoder, batches, seed=int(seed)
        )
        _load_context_snapshot(diagnostic_model, post_pretraining_context_snapshot)
        pretraining = _evaluate_state(
            diagnostic_model, final_frozen_decoder, batches, seed=int(seed)
        )
        _load_context_snapshot(diagnostic_model, post_point_context_snapshot)
        point = _evaluate_state(
            diagnostic_model, final_frozen_decoder, batches, seed=int(seed)
        )
    finally:
        for module, training in zip(final_frozen_decoder.modules(), decoder_modes):
            module.training = training

    _assert_module_observation(post_point_model, model_before, "post-point model")
    _assert_module_observation(final_frozen_decoder, decoder_before, "decoder")

    puzzle_ids = [str(batch["puzzle"]) for batch in batches]
    per_puzzle = []
    for puzzle in puzzle_ids:
        counts = {
            initial[puzzle]["eligible_constructs"],
            pretraining[puzzle]["eligible_constructs"],
            point[puzzle]["eligible_constructs"],
        }
        if len(counts) != 1:
            raise RuntimeError("retention diagnostic state universes differ")
        per_puzzle.append(
            {
                "puzzle": puzzle,
                "eligible_constructs": counts.pop(),
                "initial_context_l1": initial[puzzle]["wt_reconstruction_l1"],
                "post_pretraining_l1": pretraining[puzzle]["wt_reconstruction_l1"],
                "post_point_l1": point[puzzle]["wt_reconstruction_l1"],
            }
        )

    means = {
        "initial_context_l1": float(
            np.mean([row["initial_context_l1"] for row in per_puzzle])
        ),
        "post_pretraining_l1": float(
            np.mean([row["post_pretraining_l1"] for row in per_puzzle])
        ),
        "post_point_l1": float(np.mean([row["post_point_l1"] for row in per_puzzle])),
    }
    pretraining_gain = means["initial_context_l1"] - means["post_pretraining_l1"]
    retained_fraction = (
        None
        if abs(pretraining_gain) <= 1.0e-12
        else float(
            (means["initial_context_l1"] - means["post_point_l1"]) / pretraining_gain
        )
    )
    pretraining_established = bool(pretraining_gain > 0.0)
    retention_positive = bool(
        pretraining_established
        and retained_fraction is not None
        and retained_fraction > 0.0
    )
    return {
        "schema_version": RETENTION_SCHEMA,
        "arm": arm,
        "evidence_status": "OUTER_TRAIN_WT_RETENTION_DIAGNOSTIC_ONLY",
        "diagnostic_epoch": RETENTION_DIAGNOSTIC_EPOCH,
        "training_mask_epochs": [0, int(training_epochs) - 1],
        "diagnostic_mask_disjoint_from_training": True,
        "held_puzzle": str(held_puzzle),
        "outer_train_puzzle_ids": puzzle_ids,
        "per_puzzle": per_puzzle,
        "mean": means,
        "retained_fraction": retained_fraction,
        "pretraining_established": pretraining_established,
        "retention_positive": retention_positive,
        "same_final_frozen_decoder": True,
        "mutant_outcome_used": False,
        "held_puzzle_accessed": False,
        "checkpoint_selection_performed": False,
        "learning_rate_selection_performed": False,
    }
