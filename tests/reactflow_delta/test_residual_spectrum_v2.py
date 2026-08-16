#!/usr/bin/env python3
"""Tests for residual_spectrum_v2 — residual-learning spectrum model with
per-position median prior and diagnostics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import residual_spectrum_v2 as rs


def _make_data(n=64, WINDOW=21):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 16)).astype(np.float32)
    # targets: mostly a per-column base plus a small feature-driven signal
    base = rng.normal(size=WINDOW) * 0.5
    Wx = rng.normal(size=(16, WINDOW))
    signal = X.astype(np.float64) @ Wx * 0.3
    Y = (base[None, :] + signal).astype(np.float32)
    W = np.ones((n, WINDOW), dtype=np.float32)
    return X, Y, W


def test_prior_matches_weighted_median():
    X, Y, W = _make_data(n=65)
    prior, cnt = rs.per_position_prior(Y, W)
    assert prior.shape == (21,)
    assert cnt.shape == (21,)
    assert (cnt == 65).all()
    # odd count + unweighted all-ones -> prior equals the plain median of each column
    for k in range(21):
        assert prior[k] == pytest.approx(float(np.median(Y[:, k])), rel=1e-5)


def test_prior_ignores_zero_weight():
    X, Y, W = _make_data()
    W[:, 5] = 0.0
    prior, cnt = rs.per_position_prior(Y, W)
    assert cnt[5] == 0
    assert prior[5] == 0.0
    assert cnt[0] == 64


def test_prior_distribution_returns_schema():
    X, Y, W = _make_data()
    prior, _ = rs.per_position_prior(Y, W)
    pd = rs.prior_distribution(Y, W, prior)
    assert len(pd) == 21
    d0 = pd[0]
    assert set(d0) >= {"k", "prior", "count", "mean", "std", "abs_mean", "p90_abs"}
    assert d0["count"] == 64
    assert d0["abs_mean"] >= 0.0


def test_model_init_predicts_zero():
    X, Y, W = _make_data(n=8)
    torch.manual_seed(0)
    m = rs.ResidualSpectrumMLP(16, 21, hidden=32, seed=0)
    m.eval()
    with torch.no_grad():
        out = m(torch.from_numpy(X))
    # zero-initialized final layer -> output ~0
    assert np.abs(out.numpy()).max() < 1e-6


def test_train_residual_no_crash_and_returns_log():
    X, Y, W = _make_data(n=64)
    prior, _ = rs.per_position_prior(Y, W)
    device = torch.device("cpu")
    model, log = rs.train_residual(X, Y, W, prior, epochs=3, bs=32, seed=0,
                                   device=device)
    assert "prior" in log
    assert "learning_curve" in log
    assert len(log["learning_curve"]) == 3
    assert "final" in log
    assert "mae_model_train" in log["final"]
    assert "mae_prior_train" in log["final"]
    assert "frac_delta_gt_0" in log["final"]
    # residuals defined
    assert log["final"]["resid_abs_mean"] >= 0.0


def test_learning_curve_shrinks_loss():
    X, Y, W = _make_data(n=128)
    prior, _ = rs.per_position_prior(Y, W)
    device = torch.device("cpu")
    _, log = rs.train_residual(X, Y, W, prior, epochs=10, bs=32, seed=0, device=device)
    lc = log["learning_curve"]
    # with clear signal, loss should generally decrease
    assert lc[-1]["loss"] < lc[0]["loss"] + 1e-6


def test_resid_pen_controls_delta_magnitude():
    X, Y, W = _make_data(n=256)
    prior, _ = rs.per_position_prior(Y, W)
    device = torch.device("cpu")
    _, log_hi = rs.train_residual(X, Y, W, prior, epochs=10, bs=64, seed=0,
                                  device=device, resid_pen=10.0)
    _, log_lo = rs.train_residual(X, Y, W, prior, epochs=10, bs=64, seed=0,
                                  device=device, resid_pen=0.0)
    # higher penalty -> smaller learned residuals (nearer to prior)
    assert log_hi["final"]["delta_abs_mean"] <= log_lo["final"]["delta_abs_mean"] + 1e-6


def test_predict_delta_shape_and_no_grad():
    X, Y, W = _make_data(n=64)
    prior, _ = rs.per_position_prior(Y, W)
    device = torch.device("cpu")
    model, _ = rs.train_residual(X, Y, W, prior, epochs=2, bs=32, seed=0, device=device)
    Xte = np.random.default_rng(1).normal(size=(10, 16)).astype(np.float32)
    delta = rs.predict_delta(model, Xte, device)
    assert delta.shape == (10, 21)
    # prediction = prior + delta has finite values
    pred = prior + delta
    assert np.isfinite(pred).all()