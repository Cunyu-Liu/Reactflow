from __future__ import annotations

import numpy as np

from scripts.reactflow_delta.probe_model_rescue_v3_blend import (
    mixture_mean,
    select_alpha_from_curves,
    shifted_residual_locations,
)


def test_alpha_selection_excludes_held_puzzle() -> None:
    grid = np.asarray([0.0, 0.5, 1.0])
    curves = {
        ("P01", "A"): np.asarray([1000.0, 1000.0, 0.0]),
        ("P02", "A"): np.asarray([0.0, 1.0, 2.0]),
        ("P03", "A"): np.asarray([0.0, 1.0, 2.0]),
        ("P02", "B"): np.asarray([2.0, 1.0, 0.0]),
    }
    assert (
        select_alpha_from_curves(
            curves, grid, excluded_puzzle="P01", method="A"
        )
        == 0.0
    )
    curves[("P01", "A")] = np.asarray([0.0, 0.0, 1000000.0])
    assert (
        select_alpha_from_curves(
            curves, grid, excluded_puzzle="P01", method="A"
        )
        == 0.0
    )


def test_translated_zero_mean_residual_has_requested_point_mean() -> None:
    old_mean = np.asarray([1.0, -2.0])
    new_mean = np.asarray([2.5, 3.0])
    weights = np.asarray([[0.2, 0.8], [0.7, 0.3]])
    # Both components share a location, as required by Model Rescue v2.
    locations = np.repeat(old_mean[:, None], 2, axis=1)
    shifted = shifted_residual_locations(locations, old_mean, new_mean)
    np.testing.assert_allclose(mixture_mean(shifted, weights), new_mean)
    np.testing.assert_allclose(shifted[:, 1] - shifted[:, 0], 0.0)
