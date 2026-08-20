from __future__ import annotations

import numpy as np

from scripts.reactflow_delta.run_model_rescue_structure_probe_v1 import (
    _all_pair_graph_distance,
    _fit_standardized_ridge,
    _fit_ridge_from_statistics,
    _mfe_pairs,
    _predict_standardized_ridge,
    _ridge_sufficient_statistics,
    _sequence_features,
    resolve_structure_gate,
)


def test_mfe_graph_distance_uses_backbone_and_base_pair_shortcut():
    pairs = _mfe_pairs("((...))")
    assert pairs == [(1, 5), (0, 6)]
    distance = _all_pair_graph_distance(7, pairs)
    assert distance[0, 6] == 1
    assert distance[1, 5] == 1
    assert np.array_equal(distance, distance.T)


def test_train_only_standardized_ridge_learns_simple_signal():
    x = np.arange(20, dtype=float)[:, None]
    y = 2.0 * x[:, 0] - 1.0
    model = _fit_standardized_ridge(x, y, alpha=1e-8)
    pred = _predict_standardized_ridge(model, np.array([[20.0], [21.0]]))
    assert np.allclose(pred, [39.0, 41.0], atol=1e-5)


def test_sequence_features_follow_qualified_receiver_subset_length():
    receivers = np.array([0, 3, 7])
    x = _sequence_features(10, 2, receivers, "A", "G", np.array([False, True, False]))
    assert x.shape == (3, 15)


def test_sufficient_statistics_fit_matches_direct_standardized_ridge():
    x = np.column_stack([np.arange(30, dtype=float), np.arange(30, dtype=float) ** 2])
    y = np.column_stack([2.0 * x[:, 0] - 1.0, 0.5 * x[:, 1] + 3.0])
    from_stats = _fit_ridge_from_statistics(_ridge_sufficient_statistics(x, y), alpha=1.0)
    for target in range(2):
        direct = _fit_standardized_ridge(x, y[:, target], alpha=1.0)
        pred_direct = _predict_standardized_ridge(direct, x)
        pred_stats = _predict_standardized_ridge(from_stats, x)[:, target]
        assert np.allclose(pred_direct, pred_stats)


def test_structure_gate_requires_ci_consistency_and_other_target_guardrail():
    signed = {"mean_gain": 0.01, "ci95": [0.001, 0.02], "positive_puzzles": 14}
    magnitude = {"mean_gain": -0.0001, "ci95": [-0.002, 0.001], "positive_puzzles": 8}
    passed = resolve_structure_gate(signed, magnitude, signed_baseline=0.2, magnitude_baseline=0.1)
    assert passed["status"] == "STRUCT_DELTA_ELIGIBLE_M2"

    magnitude_bad = dict(magnitude, mean_gain=-0.002)
    failed = resolve_structure_gate(signed, magnitude_bad, signed_baseline=0.2, magnitude_baseline=0.1)
    assert failed["status"] == "STRUCT_DELTA_EXCLUDED_M2"
