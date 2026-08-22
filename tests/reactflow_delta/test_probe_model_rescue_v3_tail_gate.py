from __future__ import annotations

import numpy as np

from scripts.reactflow_delta.probe_model_rescue_v3_tail_gate import (
    apply_gate,
    fit_convex_l1_alpha,
    fit_two_bin_gate,
)


def test_exact_l1_alpha_recovers_convex_target() -> None:
    b1 = np.asarray([0.0, 0.0, 0.0])
    mean = np.asarray([2.0, 2.0, 2.0])
    target = np.asarray([0.5, 0.5, 1.5])
    alpha = fit_convex_l1_alpha(target, b1, mean, np.ones(3))
    assert alpha == 0.25


def test_two_bin_gate_uses_only_prediction_feature_for_application() -> None:
    rows = {
        "target": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "b1": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "mean": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "weight": np.ones(4),
    }
    spec = fit_two_bin_gate(rows, feature="b1_magnitude")
    alpha = apply_gate(spec, rows["b1"], rows["mean"])
    assert np.all((0.0 <= alpha) & (alpha <= 1.0))
    # Same predictions imply the same gate output; targets are not an application input.
    np.testing.assert_array_equal(alpha, apply_gate(spec, rows["b1"], rows["mean"]))
