from __future__ import annotations

import copy
import inspect

import torch

from scripts.reactflow_delta.model_rescue_v14 import V14PointModel
from scripts.reactflow_delta.puzzle_set_meta_context import (
    BASE_FEATURE_WIDTH,
    BLOCK_DIAGONAL_NULL,
    CONTEXT_WIDTH,
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    FULL_CROSS_CONSTRUCT,
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
    return make_exact_full_model_pair(
        seed=seed, v14_point_state=source.state_dict()
    )


def _set_inputs(*, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(1401)
    hidden = [
        torch.randn(
            7,
            CONTEXT_WIDTH,
            generator=generator,
            requires_grad=requires_grad,
        )
        for index in range(EXPECTED_CONSTRUCTS_PER_PUZZLE)
    ]
    observed = [
        torch.ones(len(value), dtype=torch.bool) for value in hidden
    ]
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
    assert null.connectivity == BLOCK_DIAGONAL_NULL
    assert parameter_count(candidate) == parameter_count(null)
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
    tokens = model.align_construct_tokens(hidden, observed, _reactivity(hidden))
    assert tokens.shape == (EXPECTED_CONSTRUCTS_PER_PUZZLE, 7, CONTEXT_WIDTH)
    assert torch.isfinite(tokens).all()


def test_alignment_statistics_are_leave_one_construct_and_missing_aware() -> None:
    candidate, null = make_exact_matched_pair(seed=1402)
    reactivity = [torch.full((4,), float(index)) for index in range(8)]
    observed = [torch.ones(4, dtype=torch.bool) for _index in range(8)]
    candidate_stats = candidate.construct_alignment_statistics(
        reactivity, observed
    )
    null_stats = null.construct_alignment_statistics(reactivity, observed)
    assert torch.allclose(
        candidate_stats[0, :, 0], torch.full((4,), 4.0), atol=1e-7, rtol=0.0
    )
    assert torch.allclose(
        candidate_stats[0, :, 1], torch.full((4,), 2.0), atol=1e-7, rtol=0.0
    )
    assert torch.equal(candidate_stats[0, :, 2], torch.ones(4))
    assert torch.allclose(
        candidate_stats[0, :, 3], torch.full((4,), -4.0), atol=1e-7, rtol=0.0
    )
    assert torch.equal(null_stats[:, :, 0], torch.stack(reactivity))
    assert torch.equal(null_stats[:, :, 1], torch.zeros(8, 4))
    assert torch.equal(null_stats[:, :, 2], torch.ones(8, 4))
    assert torch.equal(null_stats[:, :, 3], torch.zeros(8, 4))

    observed[0] = torch.zeros(4, dtype=torch.bool)
    candidate_missing = candidate.construct_alignment_statistics(
        reactivity, observed
    )
    null_missing = null.construct_alignment_statistics(reactivity, observed)
    assert torch.equal(candidate_missing[0, :, 0], torch.full((4,), 4.0))
    assert torch.equal(candidate_missing[0, :, 2], torch.ones(4))
    assert torch.equal(candidate_missing[0, :, 3], torch.zeros(4))
    assert torch.equal(null_missing[0], torch.zeros(4, 4))


def test_full_context_is_permutation_equivariant() -> None:
    model = PuzzleSetMetaContext(connectivity=FULL_CROSS_CONSTRUCT).eval()
    hidden, observed = _set_inputs()
    reactivity = _reactivity(hidden)
    original = model.mix_construct_tokens(hidden, observed, reactivity)
    permutation = [3, 0, 7, 2, 6, 1, 5, 4]
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


def test_block_diagonal_null_disconnects_nonfocal_constructs() -> None:
    _candidate, null = make_exact_matched_pair(seed=5)
    null.eval()
    hidden, observed = _set_inputs(requires_grad=True)
    focal = null.mix_construct_tokens(
        hidden, observed, _reactivity(hidden)
    )[0, 3].sum()
    gradient = torch.autograd.grad(focal, hidden[1], allow_unused=True)[0]
    assert gradient is not None
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_candidate_focal_context_uses_other_constructs() -> None:
    candidate, _null = make_exact_matched_pair(seed=5)
    candidate.eval()
    hidden, observed = _set_inputs(requires_grad=True)
    focal = candidate.mix_construct_tokens(
        hidden, observed, _reactivity(hidden)
    )[0, 3].square().sum()
    gradient = torch.autograd.grad(focal, hidden[1], allow_unused=True)[0]
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_cross_construct_attention_is_position_aligned() -> None:
    candidate, _null = make_exact_matched_pair(seed=6)
    candidate.eval()
    hidden, observed = _set_inputs(requires_grad=True)
    focal = candidate.mix_construct_tokens(
        hidden, observed, _reactivity(hidden)
    )[0, 3].square().sum()
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


def test_full_point_models_are_exact_matches_and_target_free() -> None:
    candidate, null = _full_pair(31)
    assert candidate.connectivity == FULL_CROSS_CONSTRUCT
    assert null.connectivity == BLOCK_DIAGONAL_NULL
    assert parameter_count(candidate) == parameter_count(null) == (
        EXPECTED_TOTAL_PARAMETERS
    )
    assert parameter_count(candidate, trainable_only=True) == parameter_count(
        null, trainable_only=True
    ) == EXPECTED_TRAINABLE_PARAMETERS
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
    assert all(not parameter.requires_grad for parameter in candidate.encoder.parameters())
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
    first, second = _full_pair(71)
    second.connectivity = FULL_CROSS_CONSTRUCT
    encoder_before = {
        name: value.detach().clone()
        for name, value in first.encoder.state_dict().items()
    }
    training = _training_batch()
    parent_before = [
        cell["parent_point"].detach().clone() for cell in training["cells"]
    ]
    first_history = fit_puzzle_set_point_model(
        first, [training], epochs=1, seed=9
    )
    second_history = fit_puzzle_set_point_model(
        second, [_training_batch()], epochs=1, seed=9
    )
    assert first_history == second_history
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
    for name, value in first.encoder.state_dict().items():
        assert torch.equal(value, encoder_before[name])
    assert all(parameter.grad is None for parameter in first.encoder.parameters())
    for expected, cell in zip(parent_before, training["cells"]):
        assert torch.equal(expected, cell["parent_point"])


def test_trained_candidate_uses_nonfocal_context_but_matched_null_cannot() -> None:
    candidate, null = _full_pair(91)
    fit_puzzle_set_point_model(
        candidate, [_training_batch()], epochs=2, seed=17
    )
    fit_puzzle_set_point_model(null, [_training_batch()], epochs=2, seed=17)
    candidate.eval()
    null.eval()

    original = [_context(4) for _ in range(8)]
    perturbed = copy.deepcopy(original)
    nonfocal = list(perturbed[1])
    nonfocal[1] = nonfocal[1] + torch.tensor([5.0, -4.0, 3.0, -2.0])
    perturbed[1] = tuple(nonfocal)
    edit = torch.tensor([0])
    distance = torch.arange(4)[None, :].float()

    def focal_prediction(model, contexts):
        with torch.no_grad():
            return model(
                contexts,
                0,
                edit,
                distance,
                ["A"],
                ["G"],
                torch.zeros(1, 4),
                torch.zeros(1, 4),
                torch.ones(1, 4, dtype=torch.bool),
            )[0]

    candidate_original = focal_prediction(candidate, original)
    candidate_perturbed = focal_prediction(candidate, perturbed)
    null_original = focal_prediction(null, original)
    null_perturbed = focal_prediction(null, perturbed)
    assert float((candidate_original - candidate_perturbed).abs().max()) > 1e-6
    assert torch.equal(null_original, null_perturbed)


def test_cross_construct_path_receives_gradient_after_zero_init_bootstrap() -> None:
    candidate, _null = _full_pair(92)
    fit_puzzle_set_point_model(
        candidate, [_training_batch()], epochs=1, seed=19
    )
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
