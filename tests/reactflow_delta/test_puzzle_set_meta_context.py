from __future__ import annotations

import inspect

import torch

from scripts.reactflow_delta.puzzle_set_meta_context import (
    BASE_FEATURE_WIDTH,
    BLOCK_DIAGONAL_NULL,
    CONTEXT_WIDTH,
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    FULL_CROSS_CONSTRUCT,
    PuzzleSetMetaContextPointModel,
    PuzzleSetMetaContext,
    make_exact_full_model_pair,
    make_exact_matched_pair,
    parameter_count,
)


def _set_inputs(*, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(1401)
    hidden = [
        torch.randn(
            5 + index,
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


def _context(length: int, *, observed: bool = True):
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.ones(length)
    observed_mask = torch.full((length,), float(observed))
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed_mask, position, region


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
    tokens = model.pool_construct_tokens(hidden, observed)
    assert tokens.shape == (EXPECTED_CONSTRUCTS_PER_PUZZLE, CONTEXT_WIDTH)
    assert torch.isfinite(tokens).all()


def test_full_context_is_permutation_equivariant() -> None:
    model = PuzzleSetMetaContext(connectivity=FULL_CROSS_CONSTRUCT).eval()
    hidden, observed = _set_inputs()
    original = model.mix_construct_tokens(hidden, observed)
    permutation = [3, 0, 7, 2, 6, 1, 5, 4]
    permuted = model.mix_construct_tokens(
        [hidden[index] for index in permutation],
        [observed[index] for index in permutation],
    )
    for new_index, old_index in enumerate(permutation):
        assert torch.allclose(
            permuted[new_index], original[old_index], atol=3e-6, rtol=0.0
        )


def test_block_diagonal_null_disconnects_nonfocal_constructs() -> None:
    _candidate, null = make_exact_matched_pair(seed=5)
    null.eval()
    hidden, observed = _set_inputs(requires_grad=True)
    focal = null.mix_construct_tokens(hidden, observed)[0].sum()
    gradient = torch.autograd.grad(focal, hidden[1], allow_unused=True)[0]
    assert gradient is not None
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_candidate_focal_context_uses_other_constructs() -> None:
    candidate, _null = make_exact_matched_pair(seed=5)
    candidate.eval()
    hidden, observed = _set_inputs(requires_grad=True)
    focal = candidate.mix_construct_tokens(hidden, observed)[0].sum()
    gradient = torch.autograd.grad(focal, hidden[1], allow_unused=True)[0]
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_zero_initialized_adapter_replays_feature41_in_both_arms() -> None:
    candidate, null = make_exact_matched_pair(seed=22)
    candidate.eval()
    null.eval()
    hidden, observed = _set_inputs()
    generator = torch.Generator().manual_seed(77)
    base = torch.randn(3, 11, BASE_FEATURE_WIDTH, generator=generator)
    feature41 = torch.randn(3, 11, generator=generator)
    mask = torch.ones(3, 11, dtype=torch.bool)
    candidate_point, candidate_residual, _ = candidate(
        hidden, observed, 0, base, feature41, mask
    )
    null_point, null_residual, _ = null(
        hidden, observed, 0, base, feature41, mask
    )
    assert torch.equal(candidate_residual, torch.zeros_like(candidate_residual))
    assert torch.equal(null_residual, torch.zeros_like(null_residual))
    assert torch.equal(candidate_point, feature41)
    assert torch.equal(null_point, feature41)


def test_full_point_models_are_exact_matches_and_target_free() -> None:
    candidate, null = make_exact_full_model_pair(seed=31)
    assert candidate.connectivity == FULL_CROSS_CONSTRUCT
    assert null.connectivity == BLOCK_DIAGONAL_NULL
    assert parameter_count(candidate) == parameter_count(null)
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


def test_full_point_model_replays_feature41_and_handles_zero_observed_p20() -> None:
    candidate, _null = make_exact_full_model_pair(seed=41)
    candidate.eval()
    contexts = [_context(9) for _ in range(8)]
    contexts[0] = _context(9, observed=False)
    edit = torch.tensor([2, 4])
    distance = torch.arange(9)[None, :] - edit[:, None]
    feature41 = torch.randn(2, 9, generator=torch.Generator().manual_seed(6))
    prediction_mask = torch.ones(2, 9, dtype=torch.bool)
    point, residual, mixed = candidate(
        contexts,
        0,
        edit,
        distance.float(),
        ["A", "C"],
        ["G", "U"],
        feature41,
        prediction_mask,
    )
    assert torch.equal(point, feature41)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert mixed.shape == (8, CONTEXT_WIDTH)
    assert torch.isfinite(mixed).all()


def test_full_point_model_is_equivariant_to_construct_order() -> None:
    candidate, _null = make_exact_full_model_pair(seed=51)
    candidate.eval()
    contexts = [_context(8 + (index % 2)) for index in range(8)]
    edit = torch.tensor([2])
    distance = torch.arange(8)[None, :] - edit[:, None]
    feature41 = torch.randn(1, 8, generator=torch.Generator().manual_seed(8))
    mask = torch.ones(1, 8, dtype=torch.bool)
    original = candidate(
        contexts,
        0,
        edit,
        distance.float(),
        ["A"],
        ["G"],
        feature41,
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
        mask,
    )[2][new_focal]
    assert torch.allclose(original, repeated, atol=3e-6, rtol=0.0)
