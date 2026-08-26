from __future__ import annotations

import inspect

import torch

from scripts.reactflow_delta.model_rescue_v14 import V14PointModel
from scripts.reactflow_delta.puzzle_set_meta_context import (
    FULL_CROSS_CONSTRUCT,
    make_exact_full_model_pair,
)
from scripts.reactflow_delta.puzzle_set_meta_context_pretraining import (
    EXPECTED_DECODER_PARAMETERS,
    EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS,
    _visible_puzzle_inputs,
    fit_puzzle_set_wt_pretraining,
    make_exact_decoder_pair,
)


def _context(length: int = 6, *, observed: bool = True):
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.ones(length)
    observed_mask = torch.full((length,), float(observed))
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed_mask, position, region


def _full_pair(seed: int = 1700):
    torch.manual_seed(1701)
    source = V14PointModel()
    return make_exact_full_model_pair(seed=seed, v14_point_state=source.state_dict())


def _batches(*, zero_observed: bool = False):
    contexts = [_context() for _index in range(8)]
    if zero_observed:
        contexts[0] = _context(observed=False)
    return [{"puzzle": "synthetic", "contexts": contexts}]


def _snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def test_pretraining_decoder_pair_is_exact_and_parameter_matched() -> None:
    candidate, null = make_exact_decoder_pair(seed=3, device="cpu")
    assert sum(parameter.numel() for parameter in candidate.parameters()) == (
        EXPECTED_DECODER_PARAMETERS
    )
    assert sum(parameter.numel() for parameter in null.parameters()) == (
        EXPECTED_DECODER_PARAMETERS
    )
    for left, right in zip(candidate.parameters(), null.parameters()):
        assert torch.equal(left, right)


def test_masked_wt_pretraining_changes_context_only_and_replays_parent() -> None:
    candidate, _null = _full_pair()
    decoder, _null_decoder = make_exact_decoder_pair(seed=4, device="cpu")
    encoder_before = _snapshot(candidate.encoder)
    point_before = _snapshot(candidate.meta_context.point_head)
    context_before = {
        name: value.detach().clone()
        for name, value in candidate.meta_context.state_dict().items()
        if not name.startswith("point_head.")
    }
    result = fit_puzzle_set_wt_pretraining(
        candidate, decoder, _batches(), epochs=2, seed=5
    )
    assert len(result["history"]) == 2
    assert result["optimizer_steps"] == 2
    assert (
        result["trainable_parameter_count"] == EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS
    )
    assert result["eligible_construct_counts"] == [8]
    assert result["mask_fraction"] == 0.4
    assert result["context_layers_changed"] is True
    assert result["encoder_changed"] is False
    assert result["point_head_changed"] is False
    assert result["decoder_frozen_downstream"] is True
    assert result["mutant_outcome_used"] is False
    assert all(
        torch.equal(value, candidate.encoder.state_dict()[name])
        for name, value in encoder_before.items()
    )
    assert all(
        torch.equal(value, candidate.meta_context.point_head.state_dict()[name])
        for name, value in point_before.items()
    )
    assert any(
        not torch.equal(value, candidate.meta_context.state_dict()[name])
        for name, value in context_before.items()
    )
    assert all(not parameter.requires_grad for parameter in decoder.parameters())

    candidate.eval()
    contexts = _batches()[0]["contexts"]
    edit = torch.tensor([1])
    parent = torch.full((1, 6), 0.25)
    with torch.no_grad():
        point = candidate(
            contexts,
            0,
            edit,
            torch.arange(6)[None, :].float() - edit[:, None],
            ["A"],
            ["G"],
            torch.zeros(1, 6),
            parent,
            torch.ones(1, 6, dtype=torch.bool),
        )[0]
    assert torch.equal(point, parent)


def test_pretraining_is_deterministic_under_identical_connectivity() -> None:
    first, second = _full_pair(seed=1702)
    second.connectivity = FULL_CROSS_CONSTRUCT
    first_decoder, second_decoder = make_exact_decoder_pair(seed=6, device="cpu")
    first_result = fit_puzzle_set_wt_pretraining(
        first, first_decoder, _batches(zero_observed=True), epochs=1, seed=7
    )
    second_result = fit_puzzle_set_wt_pretraining(
        second, second_decoder, _batches(zero_observed=True), epochs=1, seed=7
    )
    assert first_result["history"] == second_result["history"]
    assert first_result["eligible_construct_counts"] == [7]
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(first_decoder.parameters(), second_decoder.parameters()):
        assert torch.equal(left, right)


def test_pretraining_interface_cannot_accept_mutant_target_cells() -> None:
    signature = inspect.signature(fit_puzzle_set_wt_pretraining)
    for forbidden in ("target", "mutant", "held_target", "target_mask"):
        assert forbidden not in signature.parameters
    candidate, _null = _full_pair(seed=1703)
    decoder, _ = make_exact_decoder_pair(seed=8, device="cpu")
    invalid = _batches()[0] | {"cells": [{"target": torch.ones(1)}]}
    try:
        fit_puzzle_set_wt_pretraining(candidate, decoder, [invalid], epochs=1, seed=9)
    except ValueError as error:
        assert "WT contexts only" in str(error)
    else:
        raise AssertionError("pretraining accepted mutant outcome cells")


def test_focal_corruption_is_excluded_from_aligned_and_deranged_statistics() -> None:
    candidate, null = _full_pair(seed=1704)
    contexts = _batches()[0]["contexts"]
    corruption = torch.tensor([False, True, False, True, False, False])
    reactivity, observed = _visible_puzzle_inputs(
        contexts, focal_index=0, corruption_mask=corruption
    )
    assert torch.equal(reactivity[0][corruption], torch.zeros(2))
    assert torch.equal(observed[0][corruption], torch.zeros(2, dtype=torch.bool))
    candidate_stats = candidate.meta_context.construct_alignment_statistics(
        reactivity, observed
    )
    null_reactivity = null.meta_context._position_deranged_inputs(reactivity, 0)
    null_observed = null.meta_context._position_deranged_inputs(observed, 0)
    null_stats = null.meta_context.construct_alignment_statistics(
        null_reactivity, null_observed
    )
    assert torch.equal(candidate_stats[0, corruption, 2], torch.ones(2))
    assert torch.equal(null_stats[0, corruption, 2], torch.ones(2))
    assert torch.equal(candidate_stats[0, corruption, 3], torch.zeros(2))
    assert torch.equal(null_stats[0, corruption, 3], torch.zeros(2))
    assert not torch.equal(
        candidate_stats[0, corruption, 0], null_stats[0, corruption, 0]
    )
