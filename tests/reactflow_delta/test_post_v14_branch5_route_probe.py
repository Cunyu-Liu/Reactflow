from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.reactflow_delta.post_v14_branch5_route_probe import (
    MATCHED_NULL_SHIFT,
    PROBE_FEATURE_WIDTH,
    RAW_SUMMARY_WIDTH,
    ProbeRidgeStats,
    fit_probe_ridge,
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
