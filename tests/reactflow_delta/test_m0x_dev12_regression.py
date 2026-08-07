#!/usr/bin/env python3
"""EPRO_DEV_12 regression-head unit tests (pure, no remote data)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reactflow_delta.m0x_epro_dev12_regression import (  # noqa: E402
    DeltaMagnitudeRegressor, _pool_delta_target, _standardized_delta_target,
)


# ---------------------------------------------------------------------------
# Model: linear head (no sigmoid) -> signed delta, magnitude = |value|
# ---------------------------------------------------------------------------
def test_regressor_output_is_unbounded_signed_delta():
    torch.manual_seed(0)
    m = DeltaMagnitudeRegressor(feat_dim=8, hidden=16, layers=2, dropout=0.0)
    x = torch.randn(64, 8)
    out = m(x)
    assert out.shape == (64,)
    # Linear head -> output NOT bounded to [0,1]; can exceed sigmoid range.
    params = list(m.parameters())
    # make weights large enough to push output outside [0,1]
    with torch.no_grad():
        for p in params:
            p.mul_(5.0)
    out2 = m(x)
    assert float(out2.min()) < -1.0 or float(out2.max()) > 1.0, (
        "linear head must be able to produce unbounded delta magnitudes")


def test_regressor_forward_shape_matches_feat_dim():
    m = DeltaMagnitudeRegressor(feat_dim=42, hidden=256, layers=3, dropout=0.1)
    x = torch.randn(10, 42)
    assert m(x).shape == (10,)


# ---------------------------------------------------------------------------
# Pooling: eligible-position signed delta targets
# ---------------------------------------------------------------------------
def test_pool_delta_target_only_eligible_and_preserves_sign():
    recs = [
        # mask [T,F,T]; delta [1.0, 99.0(ignored), -2.0]
        {"mask": [True, False, True], "delta": [1.0, 99.0, -2.0],
         "features": np.zeros((3, 2), dtype=np.float32)},
        # mask [F,T]; delta [99.0(ignored), 3.5]
        {"mask": [False, True], "delta": [99.0, 3.5],
         "features": np.zeros((2, 2), dtype=np.float32)},
    ]
    X, y = _pool_delta_target(recs)
    assert y.tolist() == [1.0, -2.0, 3.5]  # signed preserved, ineligible dropped
    assert X.shape == (3, 2)


def test_pool_delta_target_empty_mask_yields_empty():
    X, y = _pool_delta_target(
        [{"mask": [False, False], "delta": [1.0, 2.0],
          "features": np.zeros((2, 2), dtype=np.float32)}])
    assert len(y) == 0 and X.shape[0] == 0


# ---------------------------------------------------------------------------
# Scale-standardized target (delta/scale) so |delta_r_hat| matches burden truth
# ---------------------------------------------------------------------------
class _FakePair:
    def __init__(self, delta):
        self.delta = delta


def test_standardized_target_divides_by_pair_scale():
    out = _standardized_delta_target(_FakePair([1.0, -2.0, 4.0]),
                                     [True, True, True], 2.0)
    assert out == [0.5, -1.0, 2.0]


def test_standardized_target_zeroes_nonfinite_positions():
    out = _standardized_delta_target(
        _FakePair([float("nan"), 3.0, float("inf"), -1.0]),
        [True, True, True, True], 1.0)
    assert out == [0.0, 3.0, 0.0, -1.0]


def test_standardized_target_respects_mask_length():
    # helper returns one element per mask position (masked positions still get a
    # value; pooling drops them, so length must match mask).
    out = _standardized_delta_target(_FakePair([6.0, -6.0]), [True, False], 2.0)
    assert out == [3.0, -3.0]


# ---------------------------------------------------------------------------
# MAE training sanity on a tiny synthetic regression task
# ---------------------------------------------------------------------------
def test_regressor_fits_tiny_linear_task():
    torch.manual_seed(20260804)
    np.random.seed(20260804)
    m = DeltaMagnitudeRegressor(feat_dim=4, hidden=32, layers=2, dropout=0.0)
    X = torch.randn(200, 4)
    w = torch.randn(4)
    y = X @ w + torch.randn(200) * 0.05

    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        loss = (m(X) - y).abs().mean()
        loss.backward()
        opt.step()
    final_mae = float((m(X) - y).abs().mean())
    assert final_mae < 0.2, f"regressor failed to fit linear task, MAE={final_mae:.4f}"