#!/usr/bin/env python3
"""test_residual_spectrum_v3 — unit tests for the position-aware residual spectrum model.

Verifies:
  * fail-safe zero-init: untrained model outputs delta ~ 0, so pred == prior (baseline)
  * position-aware structure: distinct decoder heads + non-trivial position embeddings
  * training reduces train MAE below the sequence-free prior baseline (residual shrinks)
  * predict_delta shape/behavior on held-out features
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import residual_spectrum_v3 as rv3  # noqa: E402
import residual_spectrum_v2 as rv2  # noqa: E402

W = 21


def _mk_data(n=256, in_dim=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    # target spectrum = position-dependent baseline + noise
    Y = np.tile(np.linspace(-0.3, 0.3, W), (n, 1)).astype(np.float32) \
        + rng.normal(scale=0.05, size=(n, W)).astype(np.float32)
    Wt = np.ones((n, W), dtype=np.float32)
    return X, Y, Wt


def test_untrained_model_outputs_zero_delta():
    X, Y, Wt = _mk_data(seed=1)
    model = rv3.PositionAwareResidualMLP(X.shape[1], W, seed=0)
    model.eval()
    with torch.no_grad():
        delta = model(torch.from_numpy(X))
    assert delta.shape == (X.shape[0], W)
    # fail-safe: untrained delta is exactly 0 (== prior == baseline)
    assert torch.abs(delta).max().item() < 1e-6


def test_position_aware_heads_and_embeddings():
    X, _, _ = _mk_data(seed=2)
    model = rv3.PositionAwareResidualMLP(X.shape[1], W, seed=3)
    assert len(model.heads) == W
    # position embeddings should be distinct across positions (position-aware)
    emb = model.pos_emb.detach().numpy()
    assert emb.shape == (W, model.pos_emb.shape[1])
    diffs = [np.linalg.norm(emb[i] - emb[j]) for i in range(W) for j in range(i + 1, W)]
    assert any(d > 1e-4 for d in diffs)


def test_training_reduces_mae_vs_prior():
    X, Y, Wt = _mk_data(n=512, seed=4)
    prior, _ = rv2.per_position_prior(Y, Wt)
    # untrained residual model must equal prior MAE at init
    model, log = rv3.train_posaware(
        X, Y, Wt, prior, epochs=40, bs=128, lr=1e-3, resid_pen=1e-3,
        hidden=64, head_hidden=16, dropout=0.1, seed=0, device=torch.device("cpu"))
    f = log["final"]
    assert f["mae_model_train"] < f["mae_prior_train"], \
        f"model {f['mae_model_train']:.4f} should beat prior {f['mae_prior_train']:.4f}"
    # delta should be non-degenerate (model moved off the prior)
    assert f["frac_delta_gt_0"] > 0.01


def test_predict_delta_shape():
    X, Y, Wt = _mk_data(n=128, seed=5)
    prior, _ = rv2.per_position_prior(Y, Wt)
    model, _ = rv3.train_posaware(
        X, Y, Wt, prior, epochs=5, bs=64, lr=1e-3, resid_pen=1e-3,
        hidden=32, head_hidden=8, dropout=0.0, seed=1, device=torch.device("cpu"))
    Xte = np.random.default_rng(9).normal(size=(7, X.shape[1])).astype(np.float32)
    d = rv3.predict_delta(model, Xte)
    assert d.shape == (7, W)
    assert d.dtype == np.float32
    assert np.isfinite(d).all()
