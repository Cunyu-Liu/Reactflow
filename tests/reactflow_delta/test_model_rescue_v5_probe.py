from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.reactflow_delta.model_rescue_v5_probe import (
    BASELINE_FEATURE_NAMES,
    WeightedRidgeStats,
    baseline_features,
    cell_position_weights,
    fit_weighted_standardized_ridge,
    predict_weighted_ridge,
)


@dataclass
class _Record:
    full_pos: int = 2
    ref: str = "G"
    alt: str = "A"


@dataclass
class _Construct:
    sequence: str
    wt_reactivity: np.ndarray
    wt_error: np.ndarray
    wt_observed: np.ndarray
    region_map: np.ndarray


def test_baseline_feature_shape_and_corrected_distance() -> None:
    construct = _Construct(
        sequence="ACGUAC",
        wt_reactivity=np.array([0.1, 0.2, 0.3, 0.4, np.nan, 0.6]),
        wt_error=np.full(6, 0.1),
        wt_observed=np.array([True, True, True, True, False, True]),
        region_map=np.array(["other_assay_region", "design_region"] * 3),
    )
    features = baseline_features(construct, _Record(), np.array([0, 2, 5]))
    assert features.shape == (3, len(BASELINE_FEATURE_NAMES))
    assert np.allclose(features[:, 0], np.array([-2, 0, 3]) / 5)
    assert np.array_equal(features[:, 5], np.array([0, 1, 0]))
    assert np.isfinite(features).all()


def test_cell_position_weights_balance_mutants_before_positions() -> None:
    first, second = cell_position_weights(np.array([2, 4]))
    assert np.isclose(first.sum(), 0.5)
    assert np.isclose(second.sum(), 0.5)
    duplicated_positions = cell_position_weights(np.array([4, 4]))
    assert np.isclose(duplicated_positions[0].sum(), 0.5)
    assert np.isclose(duplicated_positions[1].sum(), 0.5)


def test_weighted_sufficient_statistics_match_direct_linear_signal() -> None:
    x = np.column_stack([np.arange(20, dtype=float), np.arange(20, dtype=float) ** 2])
    y = np.column_stack([2.0 * x[:, 0] - 1.0, 0.5 * x[:, 1] + 3.0])
    weight = np.linspace(1.0, 2.0, len(x))
    stats = WeightedRidgeStats.zeros(2, 2)
    stats.add_rows(x[:8], y[:8], weight[:8])
    stats.add_rows(x[8:], y[8:], weight[8:])
    model = fit_weighted_standardized_ridge(stats, alpha=1e-10)
    prediction = predict_weighted_ridge(model, x)
    assert np.allclose(prediction, y, atol=1e-6)


def test_zero_variance_feature_is_finite() -> None:
    x = np.column_stack([np.arange(10, dtype=float), np.ones(10)])
    y = np.column_stack([x[:, 0], np.abs(x[:, 0])])
    stats = WeightedRidgeStats.zeros(2, 2)
    stats.add_rows(x, y, np.ones(10))
    model = fit_weighted_standardized_ridge(stats)
    assert np.isfinite(predict_weighted_ridge(model, x)).all()
