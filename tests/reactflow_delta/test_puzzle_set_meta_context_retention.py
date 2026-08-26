from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from scripts.reactflow_delta.puzzle_set_meta_context_retention import (
    RETENTION_DIAGNOSTIC_EPOCH,
    evaluate_context_retention,
    snapshot_context_for_retention,
)


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(
        self,
        context: tuple[torch.Tensor, ...],
        corruption_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("retention encoder must run in eval and no_grad")
        reactivity = context[1]
        if corruption_mask is not None:
            reactivity = reactivity.masked_fill(corruption_mask, 0.0)
        hidden = torch.zeros(
            len(reactivity), 256, dtype=reactivity.dtype, device=reactivity.device
        )
        hidden[:, 0] = reactivity * self.scale
        return hidden


class _ToyMetaContext(nn.Module):
    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.point_head = nn.Linear(1, 1)

    def mix_construct_tokens(
        self,
        construct_hidden: list[torch.Tensor],
        construct_observed: list[torch.Tensor],
        construct_reactivity: list[torch.Tensor],
    ) -> torch.Tensor:
        del construct_observed, construct_reactivity
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("retention meta-context must run in eval and no_grad")
        hidden = torch.stack(construct_hidden)
        total = hidden.sum(dim=0)
        nonfocal_mean = torch.stack(
            [
                (total - hidden[focal]) / (len(construct_hidden) - 1)
                for focal in range(len(construct_hidden))
            ]
        )
        return self.gain * nonfocal_mean


class _ToyPointModel(nn.Module):
    def __init__(self, gain: float) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()
        self.meta_context = _ToyMetaContext(gain)


class _ToyFrozenDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, mixed: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("retention decoder must run in eval and no_grad")
        return mixed[:, 0] * self.scale


def _context(*, offset: float, length: int = 10) -> tuple[torch.Tensor, ...]:
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(0.2 + offset, 1.1 + offset, length)
    precision = torch.ones(length)
    observed = torch.ones(length)
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed, position, region


def _outer_train_batches() -> list[dict[str, object]]:
    return [
        {
            "puzzle": puzzle,
            "contexts": [_context(offset=offset) for _ in range(8)],
        }
        for puzzle, offset in (("P02", 0.1), ("P01", 0.0))
    ]


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _gradients(module: nn.Module) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in module.parameters()
    ]


def _modes(module: nn.Module) -> list[bool]:
    return [submodule.training for submodule in module.modules()]


def _context_snapshot_at(model: _ToyPointModel, gain: float) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        model.meta_context.gain.fill_(gain)
    return snapshot_context_for_retention(model)


def test_retention_uses_epoch_200_one_decoder_and_three_context_states() -> None:
    model = _ToyPointModel(gain=0.5)
    initial = _context_snapshot_at(model, 0.0)
    post_pretraining = _context_snapshot_at(model, 1.0)
    with torch.no_grad():
        model.meta_context.gain.fill_(0.5)
    decoder = _ToyFrozenDecoder()

    result = evaluate_context_retention(
        arm="candidate",
        post_point_model=model,
        final_frozen_decoder=decoder,
        initial_context_snapshot=initial,
        post_pretraining_context_snapshot=post_pretraining,
        puzzle_batches=_outer_train_batches(),
        held_puzzle="P20",
        seed=13,
        training_epochs=200,
    )

    assert result["diagnostic_epoch"] == RETENTION_DIAGNOSTIC_EPOCH == 200
    assert result["training_mask_epochs"] == [0, 199]
    assert result["diagnostic_mask_disjoint_from_training"] is True
    assert result["outer_train_puzzle_ids"] == ["P01", "P02"]
    assert [row["eligible_constructs"] for row in result["per_puzzle"]] == [8, 8]
    assert result["mean"]["post_pretraining_l1"] == pytest.approx(0.0, abs=1e-7)
    assert result["mean"]["post_point_l1"] == pytest.approx(
        0.5 * result["mean"]["initial_context_l1"], abs=1e-7
    )
    assert result["arm"] == "candidate"
    assert result["retained_fraction"] == pytest.approx(0.5, abs=1e-7)
    assert result["pretraining_established"] is True
    assert result["retention_positive"] is True
    assert result["same_final_frozen_decoder"] is True
    assert result["mutant_outcome_used"] is False
    assert result["held_puzzle_accessed"] is False
    assert result["checkpoint_selection_performed"] is False
    assert result["learning_rate_selection_performed"] is False


def test_retention_is_deterministic_and_preserves_state_gradients_and_modes() -> None:
    model = _ToyPointModel(gain=0.5)
    initial = _context_snapshot_at(model, 0.0)
    post_pretraining = _context_snapshot_at(model, 1.0)
    with torch.no_grad():
        model.meta_context.gain.fill_(0.5)
    decoder = _ToyFrozenDecoder()

    model.train()
    model.encoder.eval()
    model.meta_context.point_head.eval()
    decoder.train()
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1))
    for index, parameter in enumerate(decoder.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 11))
    model_state = _state(model)
    model_gradients = _gradients(model)
    model_modes = _modes(model)
    decoder_state = _state(decoder)
    decoder_gradients = _gradients(decoder)
    decoder_modes = _modes(decoder)

    arguments = {
        "arm": "candidate",
        "post_point_model": model,
        "final_frozen_decoder": decoder,
        "initial_context_snapshot": initial,
        "post_pretraining_context_snapshot": post_pretraining,
        "puzzle_batches": _outer_train_batches(),
        "held_puzzle": "P20",
        "seed": 17,
        "training_epochs": 200,
    }
    first = evaluate_context_retention(**arguments)
    second = evaluate_context_retention(**arguments)

    assert first == second
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in model_state.items()
    )
    assert all(
        expected is not None and actual is not None and torch.equal(expected, actual)
        for expected, actual in zip(model_gradients, _gradients(model))
    )
    assert _modes(model) == model_modes
    assert all(
        torch.equal(value, decoder.state_dict()[name])
        for name, value in decoder_state.items()
    )
    assert all(
        expected is not None and actual is not None and torch.equal(expected, actual)
        for expected, actual in zip(decoder_gradients, _gradients(decoder))
    )
    assert _modes(decoder) == decoder_modes


@pytest.mark.parametrize("forbidden", ["cells", "target"])
def test_retention_rejects_any_non_context_outer_train_payload(forbidden: str) -> None:
    model = _ToyPointModel(gain=0.5)
    snapshot = snapshot_context_for_retention(model)
    invalid = copy.deepcopy(_outer_train_batches())
    invalid[0][forbidden] = torch.ones(1)

    with pytest.raises(ValueError, match=r"outer-train \{puzzle, contexts\} only"):
        evaluate_context_retention(
            arm="candidate",
            post_point_model=model,
            final_frozen_decoder=_ToyFrozenDecoder(),
            initial_context_snapshot=snapshot,
            post_pretraining_context_snapshot=snapshot,
            puzzle_batches=invalid,
            held_puzzle="P20",
            seed=1,
            training_epochs=200,
        )


def test_retention_rejects_held_puzzle_and_unfrozen_decoder() -> None:
    model = _ToyPointModel(gain=0.5)
    snapshot = snapshot_context_for_retention(model)
    decoder = _ToyFrozenDecoder()

    with pytest.raises(ValueError, match="cannot include the held puzzle"):
        evaluate_context_retention(
            arm="candidate",
            post_point_model=model,
            final_frozen_decoder=decoder,
            initial_context_snapshot=snapshot,
            post_pretraining_context_snapshot=snapshot,
            puzzle_batches=_outer_train_batches(),
            held_puzzle="P01",
            seed=1,
            training_epochs=200,
        )

    decoder.scale.requires_grad_(True)
    with pytest.raises(ValueError, match="requires one final frozen decoder"):
        evaluate_context_retention(
            arm="candidate",
            post_point_model=model,
            final_frozen_decoder=decoder,
            initial_context_snapshot=snapshot,
            post_pretraining_context_snapshot=snapshot,
            puzzle_batches=_outer_train_batches(),
            held_puzzle="P20",
            seed=1,
            training_epochs=200,
        )


def test_retention_snapshot_is_context_only_and_exact() -> None:
    model = _ToyPointModel(gain=0.5)
    snapshot = snapshot_context_for_retention(model)
    assert set(snapshot) == {"gain"}

    with pytest.raises(ValueError, match="inexact state universe"):
        evaluate_context_retention(
            arm="candidate",
            post_point_model=model,
            final_frozen_decoder=_ToyFrozenDecoder(),
            initial_context_snapshot={},
            post_pretraining_context_snapshot=snapshot,
            puzzle_batches=_outer_train_batches(),
            held_puzzle="P20",
            seed=1,
            training_epochs=200,
        )


def test_retention_records_actual_smoke_epoch_range_and_requires_disjoint_mask() -> (
    None
):
    model = _ToyPointModel(gain=0.5)
    initial = _context_snapshot_at(model, 0.0)
    post_pretraining = _context_snapshot_at(model, 1.0)
    with torch.no_grad():
        model.meta_context.gain.fill_(0.5)
    result = evaluate_context_retention(
        arm="null",
        post_point_model=model,
        final_frozen_decoder=_ToyFrozenDecoder(),
        initial_context_snapshot=initial,
        post_pretraining_context_snapshot=post_pretraining,
        puzzle_batches=_outer_train_batches(),
        held_puzzle="P20",
        seed=1,
        training_epochs=3,
    )
    assert result["arm"] == "null"
    assert result["training_mask_epochs"] == [0, 2]

    with pytest.raises(ValueError, match="post-training mask epoch"):
        evaluate_context_retention(
            arm="candidate",
            post_point_model=model,
            final_frozen_decoder=_ToyFrozenDecoder(),
            initial_context_snapshot=initial,
            post_pretraining_context_snapshot=post_pretraining,
            puzzle_batches=_outer_train_batches(),
            held_puzzle="P20",
            seed=1,
            training_epochs=201,
        )
