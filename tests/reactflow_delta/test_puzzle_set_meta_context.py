from __future__ import annotations

import inspect

import torch

from scripts.reactflow_delta.puzzle_set_meta_context import (
    BASE_FEATURE_WIDTH,
    BLOCK_DIAGONAL_NULL,
    CONTEXT_WIDTH,
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    FULL_CROSS_CONSTRUCT,
    PuzzleSetMetaContext,
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
