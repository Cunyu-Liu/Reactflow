from __future__ import annotations

import inspect

import torch

from scripts.reactflow_delta.model_rescue_v14 import V14PointModel
from scripts.reactflow_delta.puzzle_set_meta_context import (
    BASE_FEATURE_WIDTH,
    CONTEXT_WIDTH,
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    FULL_CROSS_CONSTRUCT,
    POINT_CONTEXT_LR,
    POINT_GRADIENT_CLIP,
    POINT_HEAD_LR,
    POINT_HEAD_WARMUP_EPOCHS,
    POSITION_DERANGED_NULL,
    POSITION_DERANGEMENT_SHIFT,
    PuzzleSetMetaContextPointModel,
    PuzzleSetMetaContext,
    assert_v14_encoder_replay,
    fit_puzzle_set_point_model,
    make_exact_full_model_pair,
    make_exact_matched_pair,
    parameter_count,
    puzzle_balanced_point_loss,
)


def _v14_source(seed: int = 1400) -> V14PointModel:
    torch.manual_seed(seed)
    return V14PointModel()


def _full_pair(seed: int):
    source = _v14_source()
    return make_exact_full_model_pair(seed=seed, v14_point_state=source.state_dict())


def _set_inputs(*, length: int = 7, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(1401)
    hidden = [
        torch.randn(
            length,
            CONTEXT_WIDTH,
            generator=generator,
            requires_grad=requires_grad,
        )
        for index in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
    ]
    observed = [torch.ones(len(value), dtype=torch.bool) for value in hidden]
    return hidden, observed


def _reactivity(hidden):
    return [
        torch.linspace(-1.0, 1.0, len(value)) + index / 10.0
        for index, value in enumerate(hidden)
    ]


def _context(length: int, *, observed: bool = True):
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.ones(length)
    observed_mask = torch.full((length,), float(observed))
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed_mask, position, region


def _training_batch():
    contexts = [_context(4) for _ in range(8)]
    cells = []
    for focal in range(8):
        edit = torch.tensor([focal % 4])
        distance = torch.arange(4)[None, :] - edit[:, None]
        cells.append(
            {
                "focal_construct_index": focal,
                "edit_index": edit,
                "signed_distance": distance.float(),
                "refs": ["A"],
                "alts": ["G"],
                "feature41_point": torch.zeros(1, 4),
                "parent_point": torch.zeros(1, 4),
                "prediction_mask": torch.ones(1, 4, dtype=torch.bool),
                "target": torch.full((1, 4), float(focal + 1) / 10.0),
                "qualified_mask": torch.ones(1, 4, dtype=torch.bool),
                "wt": torch.zeros(4),
            }
        )
    return {"puzzle": "synthetic-puzzle", "contexts": contexts, "cells": cells}


def test_puzzle_set_candidate_and_null_are_exact_parameter_matches() -> None:
    candidate, null = make_exact_matched_pair(seed=14)
    assert candidate.connectivity == FULL_CROSS_CONSTRUCT
    assert null.connectivity == POSITION_DERANGED_NULL
    assert parameter_count(candidate) == parameter_count(null)
    assert candidate.set_attention.dropout == null.set_attention.dropout == 0.0
    for left, right in zip(candidate.parameters(), null.parameters()):
        assert torch.equal(left, right)


def test_puzzle_set_forward_is_target_and_identity_free() -> None:
    signature = inspect.signature(PuzzleSetMetaContext.forward)
    for forbidden in (
        "target",
        "target_error",
        "qualified_mask",
        "method_id",
        "puzzle_id",
        "dataset_id",
    ):
        assert forbidden not in signature.parameters


def test_puzzle_set_handles_the_registered_zero_observed_construct() -> None:
    model = PuzzleSetMetaContext(connectivity=FULL_CROSS_CONSTRUCT).eval()
    hidden, observed = _set_inputs()
    observed[0] = torch.zeros_like(observed[0])
    tokens = model.project_individual_construct_tokens(
        hidden, observed, _reactivity(hidden)
    )
    assert tokens.shape == (EXPECTED_CONSTRUCTS_PER_PUZZLE, 7, CONTEXT_WIDTH)
    assert torch.isfinite(tokens).all()


def test_full_context_is_permutation_equivariant() -> None:
    hidden, observed = _set_inputs()
    reactivity = _reactivity(hidden)
    permutation = [3, 0, 7, 2, 6, 1, 5, 4]
    for connectivity in (FULL_CROSS_CONSTRUCT, POSITION_DERANGED_NULL):
        model = PuzzleSetMetaContext(connectivity=connectivity).eval()
        original = model.mix_construct_tokens(hidden, observed, reactivity)
        permuted = model.mix_construct_tokens(
            [hidden[index] for index in permutation],
            [observed[index] for index in permutation],
            [reactivity[index] for index in permutation],
        )
        for new_index, old_index in enumerate(permutation):
            assert torch.allclose(
                permuted[new_index], original[old_index], atol=3e-6, rtol=0.0
            )


def test_position_aligned_mixer_rejects_incompatible_coordinate_frames() -> None:
    model = PuzzleSetMetaContext(connectivity=FULL_CROSS_CONSTRUCT)
    hidden, observed = _set_inputs()
    hidden[-1] = hidden[-1][:-1]
    observed[-1] = observed[-1][:-1]
    try:
        model.mix_construct_tokens(hidden, observed, _reactivity(hidden))
    except ValueError as error:
        assert "coordinate frame" in str(error)
    else:
        raise AssertionError("position-aware mixer accepted unequal construct lengths")


def test_candidate_and_deranged_null_use_the_same_eight_token_attention_support() -> (
    None
):
    candidate, null = make_exact_matched_pair(seed=5)
    candidate.eval()
    null.eval()
    hidden, observed = _set_inputs()
    reactivity = _reactivity(hidden)

    captured = {}

    def capture(label):
        def hook(_module, args, _output):
            captured.setdefault(label, []).append(
                (args[0].shape, args[1].shape, args[2].shape)
            )

        return hook

    candidate_handle = candidate.set_attention.register_forward_hook(
        capture("candidate")
    )
    null_handle = null.set_attention.register_forward_hook(capture("null"))
    try:
        candidate.mix_construct_tokens(hidden, observed, reactivity)
        null.mix_construct_tokens(hidden, observed, reactivity)
    finally:
        candidate_handle.remove()
        null_handle.remove()

    assert len(captured["candidate"]) == len(captured["null"]) == 2
    assert captured["candidate"] == captured["null"]
    for query, key, value in captured["candidate"]:
        assert query == (8 * len(hidden[0]), 1, CONTEXT_WIDTH)
        assert key == value == (8 * len(hidden[0]), 8, CONTEXT_WIDTH)
    assert candidate.set_attention.dropout == null.set_attention.dropout == 0.0


def test_attention_kv_is_seven_nonfocal_tokens_plus_parameter_free_summary() -> None:
    candidate, null = make_exact_matched_pair(seed=501)
    candidate.eval()
    null.eval()
    hidden, observed = _set_inputs(length=23)
    reactivity = _reactivity(hidden)
    captured = {}

    def capture(label):
        def hook(_module, args, _output):
            captured.setdefault(label, []).append(
                tuple(value.detach().clone() for value in args[:3])
            )

        return hook

    candidate_handle = candidate.set_attention.register_forward_hook(
        capture("candidate")
    )
    null_handle = null.set_attention.register_forward_hook(capture("null"))
    try:
        candidate.mix_construct_tokens(hidden, observed, reactivity)
        null.mix_construct_tokens(hidden, observed, reactivity)
    finally:
        candidate_handle.remove()
        null_handle.remove()

    assert len(captured["candidate"]) == len(captured["null"]) == 2
    candidate_query, candidate_key, candidate_value = captured["candidate"][0]
    candidate_reference_query, candidate_reference_key, _ = captured["candidate"][1]
    null_query, null_key, null_value = captured["null"][0]
    null_reference_query, null_reference_key, _ = captured["null"][1]
    assert torch.equal(candidate_query, null_query)
    assert torch.equal(candidate_query, candidate_reference_query)
    assert torch.equal(null_query, null_reference_query)
    assert torch.equal(candidate_key, candidate_value)
    assert torch.equal(null_key, null_value)
    assert torch.equal(candidate_reference_key, null_reference_key)

    individual = candidate.project_individual_construct_tokens(
        hidden, observed, reactivity
    )
    summary = candidate.nonfocal_summary_token(
        individual, observed, reactivity, focal_construct_index=0
    )
    expected_query = candidate.attention_norm(individual[0])
    expected_key = candidate.attention_norm(
        torch.cat([individual[1:].permute(1, 0, 2), summary[:, None, :]], dim=1)
    )
    length = len(hidden[0])
    assert torch.equal(candidate_query[:length, 0], expected_query)
    assert torch.equal(candidate_key[:length], expected_key)
    assert not torch.equal(candidate_key, null_key)


def test_zero_nonfocal_cross_reference_cancels_bias_and_focal_query() -> None:
    generator = torch.Generator().manual_seed(1502)
    for connectivity in (FULL_CROSS_CONSTRUCT, POSITION_DERANGED_NULL):
        model = PuzzleSetMetaContext(connectivity=connectivity)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("bias"):
                    parameter.copy_(torch.randn(parameter.shape, generator=generator))
        for training in (False, True):
            model.train(training)
            for focal in range(EXPECTED_CONSTRUCTS_PER_PUZZLE):
                hidden = [
                    torch.zeros(5, CONTEXT_WIDTH)
                    for _ in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
                ]
                observed = [
                    torch.zeros(5, dtype=torch.bool)
                    for _ in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
                ]
                reactivity = [
                    torch.zeros(5) for _ in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
                ]
                hidden[focal] = torch.randn(5, CONTEXT_WIDTH, generator=generator)
                observed[focal] = torch.ones(5, dtype=torch.bool)
                reactivity[focal] = torch.randn(5, generator=generator)
                first = model.mix_construct_tokens(hidden, observed, reactivity)[focal]
                hidden[focal] = torch.randn(5, CONTEXT_WIDTH, generator=generator)
                reactivity[focal] = torch.randn(5, generator=generator)
                second = model.mix_construct_tokens(hidden, observed, reactivity)[focal]
                assert torch.equal(first, torch.zeros_like(first))
                assert torch.equal(second, torch.zeros_like(second))


def test_paired_cross_block_reuses_dropout_and_advances_rng_once() -> None:
    model = PuzzleSetMetaContext(connectivity=FULL_CROSS_CONSTRUCT).train()
    generator = torch.Generator().manual_seed(1503)
    query = torch.randn(6, 1, CONTEXT_WIDTH, generator=generator)
    key_value = torch.randn(6, 8, CONTEXT_WIDTH, generator=generator)

    torch.manual_seed(1504)
    rng_before = torch.get_rng_state()
    observed = model.paired_cross_block(query, key_value, key_value)
    observed_next = torch.rand(8)
    assert torch.equal(observed, torch.zeros_like(observed))

    torch.set_rng_state(rng_before)
    model._cross_block(query, key_value)
    expected_next = torch.rand(8)
    assert torch.equal(observed_next, expected_next)


def test_candidate_and_deranged_null_have_finite_nonzero_qkv_gradients() -> None:
    candidate, null = make_exact_matched_pair(seed=6)
    hidden, observed = _set_inputs()
    reactivity = _reactivity(hidden)
    for model in (candidate, null):
        model.eval()
        model.zero_grad(set_to_none=True)
        model.mix_construct_tokens(
            hidden, observed, reactivity
        ).square().mean().backward()
        gradient = model.set_attention.in_proj_weight.grad
        assert gradient is not None
        query, key, value = gradient.chunk(3, dim=0)
        for block in (query, key, value):
            assert torch.isfinite(block).all()
            assert float(block.abs().sum()) > 1e-9


def test_candidate_focal_context_uses_other_constructs() -> None:
    candidate, _null = make_exact_matched_pair(seed=5)
    candidate.eval()
    hidden, observed = _set_inputs(requires_grad=True)
    focal = (
        candidate.mix_construct_tokens(hidden, observed, _reactivity(hidden))[0, 3]
        .square()
        .sum()
    )
    gradient = torch.autograd.grad(focal, hidden[1], allow_unused=True)[0]
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_cross_construct_attention_is_position_aligned() -> None:
    candidate, _null = make_exact_matched_pair(seed=6)
    candidate.eval()
    hidden, observed = _set_inputs(requires_grad=True)
    focal = (
        candidate.mix_construct_tokens(hidden, observed, _reactivity(hidden))[0, 3]
        .square()
        .sum()
    )
    gradient = torch.autograd.grad(focal, hidden[1], allow_unused=True)[0]
    assert gradient is not None
    assert float(gradient[3].abs().sum()) > 0.0
    off_position = torch.cat([gradient[:3], gradient[4:]], dim=0)
    assert torch.equal(off_position, torch.zeros_like(off_position))


def test_zero_initialized_adapter_replays_parent_in_both_arms() -> None:
    candidate, null = make_exact_matched_pair(seed=22)
    candidate.eval()
    null.eval()
    hidden, observed = _set_inputs()
    generator = torch.Generator().manual_seed(77)
    base = torch.randn(3, 7, BASE_FEATURE_WIDTH, generator=generator)
    parent = torch.randn(3, 7, generator=generator)
    edit = torch.tensor([1, 3, 5])
    mask = torch.ones(3, 7, dtype=torch.bool)
    candidate_point, candidate_residual, _ = candidate(
        hidden, observed, _reactivity(hidden), 0, edit, base, parent, mask
    )
    null_point, null_residual, _ = null(
        hidden, observed, _reactivity(hidden), 0, edit, base, parent, mask
    )
    assert torch.equal(candidate_residual, torch.zeros_like(candidate_residual))
    assert torch.equal(null_residual, torch.zeros_like(null_residual))
    assert torch.equal(candidate_point, parent)
    assert torch.equal(null_point, parent)


def test_zero_cross_exactly_cancels_base_shortcut_and_replays_rng() -> None:
    model = PuzzleSetMetaContext(connectivity=FULL_CROSS_CONSTRUCT).train()
    generator = torch.Generator().manual_seed(502)
    with torch.no_grad():
        model.point_head[-1].weight.copy_(
            torch.randn(
                model.point_head[-1].weight.shape,
                generator=generator,
            )
        )
        model.point_head[-1].bias.copy_(
            torch.randn(model.point_head[-1].bias.shape, generator=generator)
        )
    batch = 2
    length = 7
    mixed = torch.zeros(EXPECTED_CONSTRUCTS_PER_PUZZLE, length, CONTEXT_WIDTH)
    edit = torch.tensor([1, 4])
    base = torch.randn(batch, length, BASE_FEATURE_WIDTH, generator=generator)
    parent = torch.randn(batch, length, generator=generator)
    mask = torch.ones(batch, length, dtype=torch.bool)

    torch.manual_seed(503)
    rng_before = torch.get_rng_state()
    point, residual = model.point_from_mixed(mixed, 0, edit, base, parent, mask)
    observed_next = torch.rand(8)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(point, parent)

    torch.set_rng_state(rng_before)
    zero_cross = torch.zeros(batch, length, 2 * CONTEXT_WIDTH)
    model.point_head(torch.cat([base, zero_cross], dim=-1))
    expected_next = torch.rand(8)
    assert torch.equal(observed_next, expected_next)

    changed_base = torch.randn(batch, length, BASE_FEATURE_WIDTH, generator=generator)
    changed_parent = torch.randn(batch, length, generator=generator)
    changed_point, changed_residual = model.point_from_mixed(
        mixed, 0, edit, changed_base, changed_parent, mask
    )
    assert torch.equal(changed_residual, torch.zeros_like(changed_residual))
    assert torch.equal(changed_point, changed_parent)


def test_full_point_models_are_exact_matches_and_target_free() -> None:
    candidate, null = _full_pair(31)
    assert candidate.connectivity == FULL_CROSS_CONSTRUCT
    assert null.connectivity == POSITION_DERANGED_NULL
    assert (
        parameter_count(candidate)
        == parameter_count(null)
        == (EXPECTED_TOTAL_PARAMETERS)
    )
    assert (
        parameter_count(candidate, trainable_only=True)
        == parameter_count(null, trainable_only=True)
        == EXPECTED_TRAINABLE_PARAMETERS
    )
    for left, right in zip(candidate.parameters(), null.parameters()):
        assert torch.equal(left, right)
    signature = inspect.signature(PuzzleSetMetaContextPointModel.forward)
    for forbidden in (
        "target",
        "target_error",
        "qualified_mask",
        "method_id",
        "puzzle_id",
    ):
        assert forbidden not in signature.parameters


def test_frozen_puzzle_encoder_exactly_replays_v14_source() -> None:
    source = _v14_source(seed=1414).eval()
    candidate, null = make_exact_full_model_pair(
        seed=32, v14_point_state=source.state_dict()
    )
    context = _context(9)
    assert_v14_encoder_replay(candidate.encoder, source, context)
    assert_v14_encoder_replay(null.encoder, source, context)
    assert all(
        not parameter.requires_grad for parameter in candidate.encoder.parameters()
    )
    assert all(not parameter.requires_grad for parameter in null.encoder.parameters())


def test_full_point_model_replays_parent_and_handles_zero_observed_p20() -> None:
    candidate, _null = _full_pair(41)
    candidate.eval()
    contexts = [_context(9) for _ in range(8)]
    contexts[0] = _context(9, observed=False)
    edit = torch.tensor([2, 4])
    distance = torch.arange(9)[None, :] - edit[:, None]
    feature41 = torch.randn(2, 9, generator=torch.Generator().manual_seed(6))
    parent = torch.randn(2, 9, generator=torch.Generator().manual_seed(7))
    prediction_mask = torch.ones(2, 9, dtype=torch.bool)
    point, residual, mixed = candidate(
        contexts,
        0,
        edit,
        distance.float(),
        ["A", "C"],
        ["G", "U"],
        feature41,
        parent,
        prediction_mask,
    )
    assert torch.equal(point, parent)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert mixed.shape == (8, 9, CONTEXT_WIDTH)
    assert torch.isfinite(mixed).all()


def test_full_point_model_is_equivariant_to_construct_order() -> None:
    candidate, _null = _full_pair(51)
    candidate.eval()
    contexts = [_context(8) for _index in range(8)]
    edit = torch.tensor([2])
    distance = torch.arange(8)[None, :] - edit[:, None]
    feature41 = torch.randn(1, 8, generator=torch.Generator().manual_seed(8))
    parent = torch.randn(1, 8, generator=torch.Generator().manual_seed(9))
    mask = torch.ones(1, 8, dtype=torch.bool)
    original = candidate(
        contexts,
        0,
        edit,
        distance.float(),
        ["A"],
        ["G"],
        feature41,
        parent,
        mask,
    )[2][0]
    permutation = [3, 2, 0, 7, 1, 6, 5, 4]
    new_focal = permutation.index(0)
    permuted_context = [contexts[index] for index in permutation]
    repeated = candidate(
        permuted_context,
        new_focal,
        edit,
        distance.float(),
        ["A"],
        ["G"],
        feature41,
        parent,
        mask,
    )[2][new_focal]
    assert torch.allclose(original, repeated, atol=3e-6, rtol=0.0)


def test_puzzle_balanced_loss_uses_all_eight_cells_equally() -> None:
    candidate, _null = _full_pair(61)
    candidate.eval()
    loss = puzzle_balanced_point_loss(candidate, _training_batch())
    # At zero initialization point=feature41=0. Each single-mutant cell has
    # constant loss 0.1, ..., 0.8; equal cell mean is therefore 0.45.
    assert torch.allclose(loss, torch.tensor(0.45), atol=1e-7, rtol=0.0)


def test_puzzle_training_replays_identically_under_same_connectivity() -> None:
    for connectivity in (FULL_CROSS_CONSTRUCT, POSITION_DERANGED_NULL):
        first, second = _full_pair(71)
        first.connectivity = connectivity
        second.connectivity = connectivity
        encoder_before = {
            name: value.detach().clone()
            for name, value in first.encoder.state_dict().items()
        }
        training = _training_batch()
        parent_before = [
            cell["parent_point"].detach().clone() for cell in training["cells"]
        ]
        first_training = fit_puzzle_set_point_model(first, [training], epochs=1, seed=9)
        second_training = fit_puzzle_set_point_model(
            second, [_training_batch()], epochs=1, seed=9
        )
        assert first_training == second_training
        assert len(first_training["history"]) == 1
        assert first_training["head_update_steps"] == 1
        assert first_training["context_update_steps"] == 0
        assert first_training["warmup_context_unchanged"] is True
        for left, right in zip(first.parameters(), second.parameters()):
            assert torch.equal(left, right)
        for name, value in first.encoder.state_dict().items():
            assert torch.equal(value, encoder_before[name])
        assert all(parameter.grad is None for parameter in first.encoder.parameters())
        for expected, cell in zip(parent_before, training["cells"]):
            assert torch.equal(expected, cell["parent_point"])


def test_point_warmup_freezes_context_then_uses_discriminative_learning_rates() -> None:
    warmup, _null = _full_pair(72)
    warmup_context_before = {
        name: value.detach().clone()
        for name, value in warmup.meta_context.state_dict().items()
        if not name.startswith("point_head.")
    }
    warmup_head_before = {
        name: value.detach().clone()
        for name, value in warmup.meta_context.point_head.state_dict().items()
    }
    warmup_training = fit_puzzle_set_point_model(
        warmup, [_training_batch()], epochs=1, seed=10
    )
    assert all(
        torch.equal(value, warmup.meta_context.state_dict()[name])
        for name, value in warmup_context_before.items()
    )
    assert any(
        not torch.equal(value, warmup.meta_context.point_head.state_dict()[name])
        for name, value in warmup_head_before.items()
    )
    assert warmup_training == {
        "history": warmup_training["history"],
        "optimizer_steps": 1,
        "head_update_steps": 1,
        "context_update_steps": 0,
        "target_exposures_per_available_cell": 1,
        "head_only_warmup_epochs": POINT_HEAD_WARMUP_EPOCHS,
        "head_learning_rate": POINT_HEAD_LR,
        "context_learning_rate": POINT_CONTEXT_LR,
        "gradient_clip": POINT_GRADIENT_CLIP,
        "warmup_context_unchanged": True,
        "best_epoch_selection_performed": False,
    }

    joint, _null = _full_pair(73)
    joint_context_before = {
        name: value.detach().clone()
        for name, value in joint.meta_context.state_dict().items()
        if not name.startswith("point_head.")
    }
    joint_training = fit_puzzle_set_point_model(
        joint, [_training_batch()], epochs=2, seed=11
    )
    assert any(
        not torch.equal(value, joint.meta_context.state_dict()[name])
        for name, value in joint_context_before.items()
    )
    assert joint_training["optimizer_steps"] == 2
    assert joint_training["head_update_steps"] == 2
    assert joint_training["context_update_steps"] == 1
    assert joint_training["target_exposures_per_available_cell"] == 2
    assert joint_training["head_learning_rate"] == POINT_HEAD_LR
    assert joint_training["context_learning_rate"] == POINT_CONTEXT_LR


def test_deranged_null_summary_uses_the_same_shifted_nonfocal_set() -> None:
    _candidate, null = make_exact_matched_pair(seed=90)
    length = 23
    hidden, _ = _set_inputs(length=length)
    reactivity = [
        torch.arange(length, dtype=torch.float32) + 100.0 * index
        for index in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
    ]
    observed = [
        torch.ones(length, dtype=torch.bool)
        for _index in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
    ]
    focal = 0
    shifted_hidden = null._position_deranged_inputs(hidden, focal)
    shifted_reactivity = null._position_deranged_inputs(reactivity, focal)
    shifted_observed = null._position_deranged_inputs(observed, focal)
    individual = null.project_individual_construct_tokens(
        shifted_hidden, shifted_observed, shifted_reactivity
    )
    captured = []

    def capture(_module, args, _output):
        captured.append(args[0].detach().clone())

    handle = null.alignment_projection.register_forward_hook(capture)
    try:
        null.nonfocal_summary_token(
            individual, shifted_observed, shifted_reactivity, focal
        )
    finally:
        handle.remove()
    assert len(captured) == 1
    statistics = captured[0]
    receiver = 3
    source = (receiver - POSITION_DERANGEMENT_SHIFT) % length
    expected_mean = torch.stack(
        [reactivity[index][source] for index in range(1, 8)]
    ).mean()
    assert torch.allclose(statistics[receiver, 0], expected_mean, atol=1e-7, rtol=0.0)
    assert not torch.equal(
        statistics[receiver, 0],
        torch.stack(reactivity[1:])[:, receiver].mean(),
    )


def test_correct_coordinate_nonfocal_counterfactual_only_affects_candidate() -> None:
    candidate, null = make_exact_matched_pair(seed=91)
    candidate.eval()
    null.eval()
    hidden, observed = _set_inputs(length=23)
    reactivity = _reactivity(hidden)
    perturbed_hidden = list(hidden)
    perturbed_reactivity = list(reactivity)
    receiver = 3
    nonfocal = 1
    perturbed_hidden[nonfocal] = hidden[nonfocal].clone()
    perturbed_hidden[nonfocal][receiver] += 5.0
    perturbed_reactivity[nonfocal] = reactivity[nonfocal].clone()
    perturbed_reactivity[nonfocal][receiver] += 5.0

    with torch.no_grad():
        candidate_original = candidate.mix_construct_tokens(
            hidden, observed, reactivity
        )
        candidate_perturbed = candidate.mix_construct_tokens(
            perturbed_hidden, observed, perturbed_reactivity
        )
        null_original = null.mix_construct_tokens(hidden, observed, reactivity)
        null_perturbed = null.mix_construct_tokens(
            perturbed_hidden, observed, perturbed_reactivity
        )

    candidate_change = candidate_perturbed - candidate_original
    null_change = null_perturbed - null_original
    assert float(candidate_change[0, receiver].abs().max()) > 1e-6
    assert torch.equal(null_change[0, receiver], torch.zeros(CONTEXT_WIDTH))
    shifted_receiver = (receiver + POSITION_DERANGEMENT_SHIFT) % len(hidden[0])
    assert float(null_change[0, shifted_receiver].abs().max()) > 1e-6


def test_cross_construct_path_receives_gradient_after_zero_init_bootstrap() -> None:
    candidate, _null = _full_pair(92)
    fit_puzzle_set_point_model(candidate, [_training_batch()], epochs=1, seed=19)
    candidate.zero_grad(set_to_none=True)
    torch.manual_seed(1901)
    loss = puzzle_balanced_point_loss(candidate, _training_batch())
    loss.backward()

    parameters = dict(candidate.named_parameters())
    for name in (
        "meta_context.construct_projection.weight",
        "meta_context.alignment_projection.weight",
        "meta_context.set_attention.in_proj_weight",
        "meta_context.point_head.0.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in candidate.encoder.parameters())


def test_puzzle_training_rejects_duplicate_or_missing_focal_cells() -> None:
    candidate, _null = _full_pair(81)
    batch = _training_batch()
    batch["cells"][-1]["focal_construct_index"] = 0
    try:
        puzzle_balanced_point_loss(candidate, batch)
    except ValueError as error:
        assert "unique focal constructs" in str(error)
    else:
        raise AssertionError("puzzle-set training accepted an unbalanced cell set")


def test_puzzle_loss_keeps_eight_contexts_with_seven_supervised_cells() -> None:
    candidate, _null = _full_pair(82)
    candidate.eval()
    batch = _training_batch()
    batch["cells"] = batch["cells"][1:]
    loss = puzzle_balanced_point_loss(candidate, batch)
    # The zero-initialized model predicts zero. Available cell losses are
    # 0.2, ..., 0.8 and are averaged without inventing the absent cell target.
    assert torch.allclose(loss, torch.tensor(0.5), atol=1e-7, rtol=0.0)
