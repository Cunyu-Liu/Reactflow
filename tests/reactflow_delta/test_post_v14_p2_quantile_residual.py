from __future__ import annotations

from dataclasses import fields, replace
import random

import numpy as np
import pytest
import torch

import scripts.reactflow_delta.post_v14_p2_quantile_residual as p2_core
from scripts.reactflow_delta.model_rescue_v10 import mixture_cdf_at_point
from scripts.reactflow_delta.post_v14_p2_quantile_residual import (
    ADAM_LEARNING_RATE,
    ADAM_WEIGHT_DECAY,
    CANDIDATE_HIDDEN_WIDTH,
    DIRECT_FEATURE_WIDTH,
    EXPECTED_PARAMETER_COUNT,
    FEATURE41_WIDTH,
    GAP_FLOOR,
    GRADIENT_CLIP_NORM,
    INITIAL_GRID_REPLAY_ATOL,
    INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0,
    INITIAL_GRID_REPLAY_RTOL,
    INPUT_WIDTH,
    MEDIAN_INDEX,
    N_GAPS,
    POINT_REPLAY_ATOL,
    POINT_REPLAY_RTOL,
    PUZZLE_ORDER_SEED_MULTIPLIER,
    QUADRATURE_WEIGHTS,
    TAUS,
    V10_INITIAL_NARROW_SCALE_RAW_TARGET,
    V10_INITIAL_WIDE_GAP_RAW_TARGET,
    MonotoneQuantileResidual,
    P2OuterTrainRows,
    PairedFitResult,
    assert_initial_grid_replay,
    build_grid_matched_models,
    equal_puzzle_hierarchy_mean,
    finite_atom_scientific_crps,
    fit_paired_quantile_and_v10,
    initialize_candidate_from_registered_grid,
    inverse_softplus,
    parameter_count,
    registered_v10_initial_grid,
    weighted_expected_absolute_delta,
    weighted_pinball_training_surrogate,
)


def test_frozen_grid_masses_architecture_and_parameter_match() -> None:
    expected_taus = np.asarray(
        [
            0.025,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
            0.975,
        ],
        dtype=np.float64,
    )
    expected_weights = np.asarray(
        [
            0.0375,
            0.0375,
            0.075,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.075,
            0.0375,
            0.0375,
        ],
        dtype=np.float64,
    )
    taus = np.asarray(TAUS)
    weights = np.asarray(QUADRATURE_WEIGHTS)
    assert np.array_equal(taus, expected_taus)
    assert np.array_equal(weights, expected_weights)
    assert bool((np.diff(taus) > 0.0).all())
    assert np.isclose(weights.sum(), 1.0, atol=1.0e-15, rtol=0.0)
    assert np.isclose(
        weights[:MEDIAN_INDEX].sum(), 0.45, atol=1.0e-15, rtol=0.0
    )
    assert weights[MEDIAN_INDEX] == 0.10
    assert np.isclose(
        weights[MEDIAN_INDEX + 1 :].sum(), 0.45, atol=1.0e-15, rtol=0.0
    )

    candidate, comparator = build_grid_matched_models(seed=7)
    assert candidate.input_layer.in_features == INPUT_WIDTH == 244
    assert candidate.input_layer.out_features == CANDIDATE_HIDDEN_WIDTH == 248
    assert candidate.output_layer.in_features == CANDIDATE_HIDDEN_WIDTH
    assert candidate.output_layer.out_features == N_GAPS == 12
    assert parameter_count(candidate) == EXPECTED_PARAMETER_COUNT == 63_748
    assert parameter_count(comparator) == EXPECTED_PARAMETER_COUNT


def test_candidate_is_strictly_monotone_with_exact_detached_median() -> None:
    assert POINT_REPLAY_ATOL == 1.0e-7
    torch.manual_seed(11)
    candidate = MonotoneQuantileResidual()
    standardized = torch.randn(6, INPUT_WIDTH, dtype=torch.float32)
    point = torch.linspace(-0.3, 0.4, 6, dtype=torch.float64, requires_grad=True)
    quantiles = candidate(point, standardized)
    assert quantiles.dtype == torch.float64
    assert torch.equal(quantiles[:, MEDIAN_INDEX], point.detach())
    assert bool((torch.diff(quantiles, dim=-1) > 0.0).all())
    quantiles.sum().backward()
    assert point.grad is None
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in candidate.parameters()
    )
    assert np.allclose(
        quantiles[:, MEDIAN_INDEX].detach().numpy(),
        point.detach().numpy(),
        atol=POINT_REPLAY_ATOL,
        rtol=POINT_REPLAY_RTOL,
    )

    with pytest.raises(ValueError, match="width"):
        candidate(torch.zeros(2), torch.zeros(2, INPUT_WIDTH - 1))
    bad = torch.zeros(2, INPUT_WIDTH)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        candidate(torch.zeros(2), bad)


def test_training_surrogate_scientific_crps_and_weighted_abs_are_distinct() -> None:
    atom_values = torch.as_tensor(
        np.linspace(-1.3, 1.1, 13)[None, :], dtype=torch.float64
    )
    target = torch.tensor([0.37], dtype=torch.float64)
    taus = np.asarray(TAUS)
    weights = np.asarray(QUADRATURE_WEIGHTS)
    atoms = atom_values.numpy()[0]
    residual = float(target[0]) - atoms
    expected_pinball = 2.0 * np.sum(
        weights * residual * (taus - (residual < 0.0).astype(np.float64))
    )
    expected_crps = np.sum(weights * np.abs(float(target[0]) - atoms)) - 0.5 * np.sum(
        weights[:, None] * weights[None, :] * np.abs(atoms[:, None] - atoms[None, :])
    )
    expected_absolute = np.sum(weights * np.abs(atoms))

    surrogate = weighted_pinball_training_surrogate(target, atom_values)
    scientific = finite_atom_scientific_crps(target, atom_values)
    absolute = weighted_expected_absolute_delta(atom_values)
    assert float(surrogate[0]) == pytest.approx(expected_pinball, abs=1.0e-14)
    assert float(scientific[0]) == pytest.approx(expected_crps, abs=1.0e-14)
    assert float(absolute[0]) == pytest.approx(expected_absolute, abs=1.0e-14)
    assert not torch.equal(surrogate, scientific)


def test_equal_puzzle_hierarchy_has_all_four_frozen_levels() -> None:
    values = torch.tensor([1.0, 3.0, 9.0, 5.0, 2.0, 6.0], dtype=torch.float64)
    mutant = torch.tensor([0, 0, 1, 2, 3, 3])
    method_cell = torch.tensor([0, 0, 0, 1, 2, 2])
    puzzle = torch.tensor([0, 0, 0, 0, 1, 1])
    # puzzle0: mean(mean([1,3]), 9) and 5 -> mean(5.5, 5) = 5.25
    # puzzle1: mean([2,6]) = 4; equal-puzzle result = mean(5.25, 4)
    result = equal_puzzle_hierarchy_mean(values, mutant, method_cell, puzzle)
    assert float(result) == pytest.approx(4.625, abs=0.0)


def test_registered_v10_initialization_and_grid_replay_are_exactly_frozen() -> None:
    assert V10_INITIAL_NARROW_SCALE_RAW_TARGET == 0.08
    assert V10_INITIAL_WIDE_GAP_RAW_TARGET == 0.20
    candidate, comparator = build_grid_matched_models(seed=17)
    assert torch.count_nonzero(candidate.output_layer.weight) == 0
    assert torch.count_nonzero(comparator.output_layer.weight) == 0
    expected_comparator_bias = torch.tensor(
        [
            0.0,
            inverse_softplus(V10_INITIAL_NARROW_SCALE_RAW_TARGET),
            inverse_softplus(V10_INITIAL_WIDE_GAP_RAW_TARGET),
            0.0,
        ],
        dtype=torch.float32,
    )
    assert torch.equal(comparator.output_layer.bias.detach(), expected_comparator_bias)
    # Resetting both constructors to the same seed gives the overlapping input
    # weight rows the same initialization despite the two hidden widths.
    assert torch.equal(
        candidate.input_layer.weight.detach(),
        comparator.input_layer.weight.detach()[:CANDIDATE_HIDDEN_WIDTH],
    )

    generator = torch.Generator().manual_seed(29)
    standardized = torch.randn(5, INPUT_WIDTH, generator=generator)
    point = torch.tensor([-1.7, -0.2, 0.0, 0.45, 2.1], dtype=torch.float64)
    with torch.no_grad():
        candidate_grid = candidate(point, standardized).numpy()
    registered_grid = registered_v10_initial_grid(comparator, point, standardized)
    assert_initial_grid_replay(candidate_grid, registered_grid)
    assert np.array_equal(candidate_grid[:, MEDIAN_INDEX], point.numpy())

    # The atom CDF includes its 0.10 median mass, while the continuous matched
    # V10 replay has exactly 0.50 CDF at the point: grid replay is not equality
    # of the two complete predictive distributions.
    atom_cdf_at_point = np.sum(
        np.asarray(QUADRATURE_WEIGHTS)[None, :]
        * (candidate_grid <= point.numpy()[:, None]),
        axis=1,
    )
    with torch.no_grad():
        mix_weights, mix_locations, mix_scales = comparator(point, standardized)
        gaussian_cdf_at_point = mixture_cdf_at_point(
            point, mix_weights, mix_locations, mix_scales
        ).numpy()
    assert np.array_equal(atom_cdf_at_point, np.full(5, 0.55))
    assert np.allclose(gaussian_cdf_at_point, 0.5, atol=3.0e-6, rtol=0.0)
    assert not np.array_equal(atom_cdf_at_point, gaussian_cdf_at_point)


def test_grid_replay_rejects_gap_floor_tolerance_drift_and_nonzero_rtol() -> None:
    assert GAP_FLOOR == 1.0e-4
    assert INITIAL_GRID_REPLAY_ATOL == 1.0e-6
    candidate = MonotoneQuantileResidual()
    invalid_grid = np.arange(13, dtype=np.float64)
    invalid_grid[7:] -= 1.0 - 1.0e-4
    with pytest.raises(ValueError, match="strictly greater"):
        initialize_candidate_from_registered_grid(candidate, invalid_grid)

    reference = np.tile(
        np.linspace(-1.0, 1.0, 13, dtype=np.float64)[None, :], (3, 1)
    )
    assert_initial_grid_replay(reference.copy(), reference)
    within_tolerance = reference.copy()
    within_tolerance[1, 4] += 5.0e-7
    assert_initial_grid_replay(within_tolerance, reference)
    bad = reference.copy()
    bad[2, 4] += 1.01e-6
    with pytest.raises(RuntimeError, match="does not replay"):
        assert_initial_grid_replay(bad, reference)
    with pytest.raises(ValueError, match=INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0):
        assert_initial_grid_replay(reference, reference, rtol=1.0e-12)
    assert INITIAL_GRID_REPLAY_RTOL == 0.0


def _synthetic_outer_train_rows() -> tuple[P2OuterTrainRows, np.ndarray]:
    generator = np.random.default_rng(41)
    puzzle = np.repeat(np.asarray(["p0", "p1", "p2"], dtype=object), 4)
    method_cell = np.asarray(
        [f"{puzzle_name}-m{method}" for puzzle_name in ("p0", "p1", "p2") for method in (0, 0, 1, 1)],
        dtype=object,
    )
    mutant = np.asarray(
        [f"{puzzle_name}-u{mutant_id}" for puzzle_name in ("p0", "p1", "p2") for mutant_id in range(4)],
        dtype=object,
    )
    frozen_point = np.linspace(-0.25, 0.30, len(puzzle), dtype=np.float64)
    rows = P2OuterTrainRows(
        feature41=generator.normal(size=(len(puzzle), FEATURE41_WIDTH)),
        direct_features=generator.normal(size=(len(puzzle), DIRECT_FEATURE_WIDTH)),
        target_delta=frozen_point + generator.normal(scale=0.15, size=len(puzzle)),
        puzzle=puzzle,
        method_cell=method_cell,
        mutant=mutant,
    )
    return rows, frozen_point


def test_minimal_paired_fit_uses_one_rows_stats_seed_order_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, frozen_point = _synthetic_outer_train_rows()
    point_snapshot = frozen_point.copy()
    seed = 23

    build_calls: list[tuple[int, str]] = []
    candidate_batches: list[tuple[np.ndarray, np.ndarray]] = []
    comparator_batches: list[tuple[np.ndarray, np.ndarray]] = []
    pinball_calls: list[tuple[np.ndarray, torch.Tensor]] = []
    comparator_targets: list[np.ndarray] = []
    hierarchy_calls: list[
        tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    real_build = p2_core.build_grid_matched_models
    real_pinball = p2_core.weighted_pinball_training_surrogate
    real_comparator_crps = p2_core.gaussian_mixture_crps_torch
    real_hierarchy = p2_core.equal_puzzle_hierarchy_mean

    def as_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().copy()

    def capture_build(
        build_seed: int, device: str | torch.device = "cpu"
    ) -> tuple[MonotoneQuantileResidual, torch.nn.Module]:
        build_calls.append((build_seed, str(device)))
        candidate, comparator = real_build(build_seed, device)
        candidate_forward = candidate.forward
        comparator_forward = comparator.forward

        def capture_candidate_forward(
            point: torch.Tensor, standardized: torch.Tensor
        ) -> torch.Tensor:
            if candidate.training:
                candidate_batches.append((as_numpy(point), as_numpy(standardized)))
            return candidate_forward(point, standardized)

        def capture_comparator_forward(
            point: torch.Tensor, standardized: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if comparator.training:
                comparator_batches.append((as_numpy(point), as_numpy(standardized)))
            return comparator_forward(point, standardized)

        monkeypatch.setattr(candidate, "forward", capture_candidate_forward)
        monkeypatch.setattr(comparator, "forward", capture_comparator_forward)
        return candidate, comparator

    def capture_pinball(
        target: torch.Tensor, quantiles: torch.Tensor
    ) -> torch.Tensor:
        pinball = real_pinball(target, quantiles)
        pinball_calls.append((as_numpy(target), pinball))
        return pinball

    def capture_comparator_crps(
        locations: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        comparator_targets.append(as_numpy(target))
        return real_comparator_crps(locations, scales, weights, target)

    def capture_hierarchy(
        position_values: torch.Tensor,
        mutant: torch.Tensor,
        method_cell: torch.Tensor,
        puzzle: torch.Tensor,
    ) -> torch.Tensor:
        hierarchy_calls.append(
            (
                position_values,
                as_numpy(mutant),
                as_numpy(method_cell),
                as_numpy(puzzle),
            )
        )
        return real_hierarchy(position_values, mutant, method_cell, puzzle)

    monkeypatch.setattr(p2_core, "build_grid_matched_models", capture_build)
    monkeypatch.setattr(
        p2_core, "weighted_pinball_training_surrogate", capture_pinball
    )
    monkeypatch.setattr(p2_core, "gaussian_mixture_crps_torch", capture_comparator_crps)
    monkeypatch.setattr(p2_core, "equal_puzzle_hierarchy_mean", capture_hierarchy)

    result = fit_paired_quantile_and_v10(
        rows, frozen_point, seed=seed, epochs=1, device="cpu"
    )
    raw_input = np.concatenate(
        [
            rows.feature41,
            frozen_point[:, None],
            np.abs(frozen_point)[:, None],
            rows.direct_features,
        ],
        axis=1,
    )
    expected_scale = raw_input.std(axis=0)
    expected_scale = np.where(expected_scale >= 1.0e-6, expected_scale, 1.0)
    expected_order = ["p0", "p1", "p2"]
    random.Random(seed * PUZZLE_ORDER_SEED_MULTIPLIER).shuffle(expected_order)
    expected_standardized = result.standardizer.transform_numpy(raw_input).astype(
        np.float32
    )
    expected_batches = [
        np.flatnonzero(np.asarray(rows.puzzle) == puzzle_name)
        for puzzle_name in expected_order
    ]

    assert np.array_equal(frozen_point, point_snapshot)
    assert np.array_equal(result.standardizer.mean, raw_input.mean(axis=0))
    assert np.array_equal(result.standardizer.scale, expected_scale)
    assert result.seed == seed
    assert result.epochs == 1
    assert result.puzzle_orders == (tuple(expected_order),)
    assert PUZZLE_ORDER_SEED_MULTIPLIER == 100_003
    assert build_calls == [(seed, "cpu")]
    assert len(candidate_batches) == len(comparator_batches) == len(expected_batches)
    assert len(pinball_calls) == len(comparator_targets) == len(expected_batches)
    assert len(hierarchy_calls) == 2 * len(expected_batches)
    hierarchy_labels = (rows.mutant, rows.method_cell, rows.puzzle)
    for batch_index, row_indices in enumerate(expected_batches):
        candidate_point, candidate_input = candidate_batches[batch_index]
        comparator_point, comparator_input = comparator_batches[batch_index]
        expected_point = point_snapshot[row_indices]
        expected_target = rows.target_delta[row_indices]
        assert np.array_equal(candidate_point, expected_point)
        assert np.array_equal(comparator_point, expected_point)
        assert np.array_equal(candidate_input, expected_standardized[row_indices])
        assert np.array_equal(comparator_input, expected_standardized[row_indices])
        assert np.array_equal(candidate_input, comparator_input)
        assert np.array_equal(pinball_calls[batch_index][0], expected_target)
        assert np.array_equal(comparator_targets[batch_index], expected_target)

        candidate_hierarchy = hierarchy_calls[2 * batch_index]
        comparator_hierarchy = hierarchy_calls[2 * batch_index + 1]
        assert candidate_hierarchy[0] is pinball_calls[batch_index][1]
        for candidate_codes, comparator_codes, labels in zip(
            candidate_hierarchy[1:], comparator_hierarchy[1:], hierarchy_labels
        ):
            labels_in_batch = np.asarray(labels)[row_indices]
            expected_same_group = labels_in_batch[:, None] == labels_in_batch[None, :]
            assert np.array_equal(candidate_codes, comparator_codes)
            assert np.array_equal(
                candidate_codes[:, None] == candidate_codes[None, :],
                expected_same_group,
            )

    assert len(result.candidate_history) == len(result.comparator_history) == 1
    assert np.isfinite(result.candidate_history).all()
    assert np.isfinite(result.comparator_history).all()
    assert np.array_equal(result.candidate_quantiles[:, MEDIAN_INDEX], point_snapshot)
    assert np.allclose(
        result.candidate_quantiles[:, MEDIAN_INDEX],
        point_snapshot,
        atol=POINT_REPLAY_ATOL,
        rtol=POINT_REPLAY_RTOL,
    )
    assert bool((np.diff(result.candidate_quantiles, axis=1) > 0.0).all())
    assert result.candidate_quantiles.shape == (len(frozen_point), 13)
    assert result.v10_replay_weights.shape == (len(frozen_point), 2)
    assert result.v10_replay_locations.shape == (len(frozen_point), 2)
    assert result.v10_replay_scales.shape == (len(frozen_point), 2)
    assert result.candidate_expected_absolute_delta.shape == (len(frozen_point),)
    assert result.v10_replay_expected_absolute_delta.shape == (len(frozen_point),)
    for returned_array in (
        result.candidate_quantiles,
        result.candidate_expected_absolute_delta,
        result.v10_replay_weights,
        result.v10_replay_locations,
        result.v10_replay_scales,
        result.v10_replay_expected_absolute_delta,
    ):
        assert np.isfinite(returned_array).all()
    assert ADAM_LEARNING_RATE == 1.0e-3
    assert ADAM_WEIGHT_DECAY == 0.0
    assert GRADIENT_CLIP_NORM == 5.0
    result_fields = {field.name for field in fields(PairedFitResult)}
    assert not any("score" in name or "verdict" in name for name in result_fields)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        pytest.param("feature41", np.nan, id="feature41-nan"),
        pytest.param("feature41", np.inf, id="feature41-inf"),
        pytest.param("direct_features", np.nan, id="direct-features-nan"),
        pytest.param("direct_features", np.inf, id="direct-features-inf"),
        pytest.param("target_delta", np.nan, id="target-delta-nan"),
        pytest.param("target_delta", np.inf, id="target-delta-inf"),
        pytest.param("frozen_point", np.nan, id="frozen-point-nan"),
        pytest.param("frozen_point", np.inf, id="frozen-point-inf"),
    ],
)
def test_paired_fit_rejects_nonfinite_inputs(
    field_name: str, bad_value: float
) -> None:
    rows, frozen_point = _synthetic_outer_train_rows()
    bad_rows = rows
    bad_point = frozen_point.copy()
    if field_name == "frozen_point":
        bad_point[0] = bad_value
    else:
        bad_array = np.asarray(getattr(rows, field_name)).copy()
        bad_array.reshape(-1)[0] = bad_value
        bad_rows = replace(rows, **{field_name: bad_array})
    with pytest.raises(ValueError, match="nonfinite"):
        fit_paired_quantile_and_v10(
            bad_rows, bad_point, seed=1, epochs=1
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "feature41",
        "direct_features",
        "target_delta",
        "puzzle",
        "method_cell",
        "mutant",
        "frozen_point",
    ],
)
def test_paired_fit_rejects_unaligned_outer_train_rows(field_name: str) -> None:
    rows, frozen_point = _synthetic_outer_train_rows()
    bad_rows = rows
    bad_point = frozen_point
    if field_name == "frozen_point":
        bad_point = frozen_point[:-1]
    else:
        bad_rows = replace(rows, **{field_name: getattr(rows, field_name)[:-1]})
    with pytest.raises(ValueError, match="not aligned"):
        fit_paired_quantile_and_v10(bad_rows, bad_point, seed=1, epochs=1)
