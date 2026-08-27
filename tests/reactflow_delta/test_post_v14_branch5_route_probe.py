from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.reactflow_delta.post_v14_branch5_route_probe import (
    CudaProbeRidgeStats,
    MATCHED_NULL_SHIFT,
    PROBE_FEATURE_WIDTH,
    RAW_SUMMARY_WIDTH,
    ProbeRidgeStats,
    fit_probe_ridge,
    fit_probe_ridge_cuda,
    nonfocal_linear_summary,
    predict_probe_ridge,
    puzzle_method_balanced_weights,
    source_receiver_features,
)


def _contexts(length: int = 23) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.zeros(8, length, 256, dtype=torch.float64)
    reactivity = torch.zeros(8, length, dtype=torch.float64)
    observed = torch.ones(8, length, dtype=torch.bool)
    for construct in range(8):
        position = torch.arange(length, dtype=torch.float64)
        hidden[construct, :, 0] = 100 * construct + position
        hidden[construct, :, 1] = construct
        reactivity[construct] = 10 * construct + position
    return hidden, reactivity, observed


def test_summary_excludes_focal_and_is_nonfocal_permutation_invariant() -> None:
    hidden, reactivity, observed = _contexts()
    expected = nonfocal_linear_summary(
        hidden, reactivity, observed, focal_index=3, shift=0
    )
    hidden[3] = 1e9
    reactivity[3] = -1e9
    observed[3] = False
    assert torch.equal(
        expected,
        nonfocal_linear_summary(hidden, reactivity, observed, focal_index=3, shift=0),
    )

    permutation = torch.tensor([6, 1, 5, 3, 0, 7, 2, 4])
    permuted_focal = int(torch.nonzero(permutation == 3).item())
    actual = nonfocal_linear_summary(
        hidden[permutation],
        reactivity[permutation],
        observed[permutation],
        focal_index=permuted_focal,
        shift=0,
    )
    assert torch.allclose(expected, actual, atol=1e-12, rtol=0.0)


def test_shift17_matches_fixed_roll_direction_and_missing_support_is_zero() -> None:
    hidden, reactivity, observed = _contexts(length=23)
    observed[:, 4] = False
    aligned = nonfocal_linear_summary(
        hidden, reactivity, observed, focal_index=0, shift=0
    )
    shifted = nonfocal_linear_summary(
        hidden,
        reactivity,
        observed,
        focal_index=0,
        shift=MATCHED_NULL_SHIFT,
    )
    assert torch.equal(shifted, torch.roll(aligned, shifts=17, dims=0))
    assert torch.equal(aligned[4, 256:], torch.zeros(4, dtype=aligned.dtype))


def test_source_receiver_feature_layout_is_exact() -> None:
    summary = torch.arange(7 * RAW_SUMMARY_WIDTH, dtype=torch.float64).reshape(
        7, RAW_SUMMARY_WIDTH
    )
    edit = torch.tensor([1, 5], dtype=torch.int64)
    features = source_receiver_features(summary, edit)
    assert features.shape == (2, 7, PROBE_FEATURE_WIDTH)
    assert torch.equal(features[0, 6, :RAW_SUMMARY_WIDTH], summary[1])
    assert torch.equal(features[0, 6, RAW_SUMMARY_WIDTH:], summary[6])
    assert torch.equal(features[1, 2, :RAW_SUMMARY_WIDTH], summary[5])


def test_hierarchical_weights_equalize_puzzles_cells_mutants_and_positions() -> None:
    masks = [
        [
            np.asarray([[1, 1, 0], [0, 0, 1]], dtype=bool),
            np.asarray([[1, 0], [1, 1]], dtype=bool),
        ],
        [
            np.asarray([[1, 0, 1]], dtype=bool),
            np.asarray([[1], [1], [1]], dtype=bool),
        ],
    ]
    weights = puzzle_method_balanced_weights(masks)
    qualified_rows = sum(int(mask.sum()) for puzzle in masks for mask in puzzle)
    assert sum(
        float(value.sum()) for puzzle in weights for value in puzzle
    ) == pytest.approx(qualified_rows)
    for puzzle in weights:
        assert sum(float(value.sum()) for value in puzzle) == pytest.approx(
            qualified_rows / 2
        )
        assert [float(value.sum()) for value in puzzle] == pytest.approx(
            [qualified_rows / 4, qualified_rows / 4]
        )
    nonzero = np.concatenate(
        [value[value > 0] for puzzle in weights for value in puzzle]
    )
    assert float(nonzero.mean()) == pytest.approx(1.0)


def test_ridge_has_unpenalized_mean_and_frozen_alpha() -> None:
    x = np.asarray(
        [[-2.0, 0.0, 1.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 2.0], [2.0, 2.0, -1.0]]
    )
    y = 3.0 + x @ np.asarray([2.0, -1.0, 0.5])
    stats = ProbeRidgeStats.zeros(width=3)
    stats.add_rows(x, y, np.ones(len(x)))
    model = fit_probe_ridge(stats)
    standardized = (x - model["mean_x"]) / model["scale_x"]
    prediction = model["mean_y"] + standardized @ model["coefficient"]
    assert float(np.mean(prediction)) == pytest.approx(float(np.mean(y)))
    assert model["alpha"] == 1.0
    with pytest.raises(ValueError, match="alpha=1"):
        fit_probe_ridge(stats, alpha=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")
def test_cuda_ridge_matches_the_cpu_reference_and_solves_on_gpu(
    monkeypatch,
) -> None:
    x = np.asarray(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 2.0],
            [2.0, 2.0, -1.0],
            [0.5, -0.25, 0.75],
        ],
        dtype=np.float64,
    )
    y = 3.0 + x @ np.asarray([2.0, -1.0, 0.5])
    weight = np.asarray([0.5, 1.0, 2.0, 1.5, 0.75], dtype=np.float64)

    reference_stats = ProbeRidgeStats.zeros(width=3)
    reference_stats.add_rows(x, y, weight)
    reference = fit_probe_ridge(reference_stats)

    device = torch.device("cuda:0")
    cuda_stats = CudaProbeRidgeStats.zeros(device=device, width=3)
    cuda_stats.add_rows(
        torch.tensor(x, dtype=torch.float64, device=device),
        torch.tensor(y, dtype=torch.float64, device=device),
        torch.tensor(weight, dtype=torch.float64, device=device),
    )
    solve_devices = []
    real_solve = torch.linalg.solve

    def checked_solve(matrix, rhs):
        solve_devices.append((matrix.device, rhs.device, matrix.dtype, rhs.dtype))
        return real_solve(matrix, rhs)

    monkeypatch.setattr(torch.linalg, "solve", checked_solve)
    actual = fit_probe_ridge_cuda(cuda_stats)
    assert solve_devices == [(device, device, torch.float64, torch.float64)]
    for name in ("mean_x", "scale_x", "mean_y", "coefficient"):
        assert isinstance(actual[name], torch.Tensor)
        assert actual[name].is_cuda
        assert actual[name].dtype == torch.float64
        np.testing.assert_allclose(
            actual[name].detach().cpu().numpy(),
            np.asarray(reference[name]),
            atol=1e-11,
            rtol=1e-11,
        )
    expected_prediction = (
        reference["mean_y"]
        + ((x - reference["mean_x"]) / reference["scale_x"]) @ reference["coefficient"]
    )
    actual_prediction = (
        actual["mean_y"]
        + (
            (torch.tensor(x, dtype=torch.float64, device=device) - actual["mean_x"])
            / actual["scale_x"]
        )
        @ actual["coefficient"]
    )
    np.testing.assert_allclose(
        actual_prediction.detach().cpu().numpy(),
        expected_prediction,
        atol=1e-11,
        rtol=1e-11,
    )


def test_prediction_requires_the_frozen_520d_feature_map() -> None:
    model = {
        "mean_x": np.zeros(PROBE_FEATURE_WIDTH),
        "scale_x": np.ones(PROBE_FEATURE_WIDTH),
        "mean_y": 0.25,
        "coefficient": np.zeros(PROBE_FEATURE_WIDTH),
        "alpha": 1.0,
    }
    assert np.array_equal(
        predict_probe_ridge(model, np.zeros((3, PROBE_FEATURE_WIDTH))),
        np.full(3, 0.25),
    )
