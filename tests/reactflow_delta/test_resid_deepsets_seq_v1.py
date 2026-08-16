#!/usr/bin/env python3
"""test_resid_deepsets_seq_v1 — unit tests for the DeepSets residual-spectrum module.

Covers: per-position prior correctness, pos/glob splitting, global-seq concatenation,
zero-init (delta=0 == prior at init), residual training reducing train MAE vs prior
without exploding delta, and predict_resid_sets returning prior+delta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import resid_deepsets_seq_v1 as rds


def _mk_data(n=64, W=21, pos_dim=7, glob_dim=12, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, W, pos_dim)).astype(np.float32)
    glob = rng.normal(size=(n, glob_dim)).astype(np.float32)
    # target with a strong per-position mean that the prior captures
    prior_true = np.linspace(-0.5, 0.5, W).astype(np.float32)
    Y = np.tile(prior_true, (n, 1)) + rng.normal(scale=0.05, size=(n, W)).astype(np.float32)
    Wm = np.ones((n, W), dtype=np.float32)
    return pos, glob, Y, Wm, prior_true


def test_per_position_prior_weighted_median():
    Y = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]], dtype=np.float64)
    W = np.ones((3, 2), dtype=np.float64)
    prior, cnt = rds.per_position_prior(Y, W)
    # median of [1,2,3] = 2 ; median of [5,6,7] = 6
    assert np.allclose(prior, [2.0, 6.0])
    assert np.all(cnt == 3)


def test_per_position_prior_masked_positions_zero():
    Y = np.zeros((3, 5), dtype=np.float64)
    W = np.zeros((3, 5), dtype=np.float64)
    W[:, 0] = 1.0
    Y[:, 0] = [10.0, 20.0, 30.0]
    prior, cnt = rds.per_position_prior(Y, W)
    assert cnt[0] == 3
    assert np.median([10.0, 20.0, 30.0]) == pytest.approx(prior[0])
    assert cnt[1] == 0 and prior[1] == 0.0


def test_split_pos_glob_roundtrip():
    n, W, pos_dim, glob_dim = 8, 21, 7, 11
    X = np.arange(n * (W * pos_dim + glob_dim), dtype=np.float32).reshape(n, -1)
    pos, glob = rds.split_pos_glob(X, W, pos_dim)
    assert pos.shape == (n, W, pos_dim)
    assert glob.shape == (n, glob_dim)
    assert np.array_equal(pos.reshape(n, -1), X[:, :W * pos_dim])
    assert np.array_equal(glob, X[:, W * pos_dim:])


def test_concat_glob_seq():
    base = np.zeros((3, 4), dtype=np.float32)
    seq = np.ones((3, 5), dtype=np.float32)
    out = rds.concat_glob_seq(base, seq)
    assert out.shape == (3, 9)
    assert np.all(out[:, :4] == 0.0)
    assert np.all(out[:, 4:] == 1.0)


def test_deepsets_resid_zero_init_equals_prior():
    pos, glob, Y, Wm, prior_true = _mk_data()
    n, W = pos.shape[0], pos.shape[1]
    glob_dim = glob.shape[1]
    model = rds.DeepSetsResidSpectrum(pos.shape[2], 16, glob_dim, W, seed=0)
    model.eval()
    with torch.no_grad():
        delta = model(torch.from_numpy(pos), torch.from_numpy(glob))
    # zero-initialised final layer => delta should be ~0
    assert float(delta.abs().max()) < 1e-6
    # predict_resid_sets at init == prior broadcast
    pred = rds.predict_resid_sets(model, pos, glob, prior_true, torch.device("cpu"))
    assert pred.shape == (n, W)
    assert np.allclose(pred, prior_true, atol=1e-6)


def test_train_reduces_train_mae_vs_prior_without_delta_explosion():
    pos, glob, Y, Wm, prior_true = _mk_data(n=96)
    n, W = pos.shape[0], pos.shape[1]
    glob_dim = glob.shape[1]
    model = rds.DeepSetsResidSpectrum(pos.shape[2], 16, glob_dim, W, seed=1)
    device = torch.device("cpu")
    model, log = rds.train_resid_sets(
        model, pos, glob, Y, Wm, prior_true, device, seed=1,
        epochs=40, bs=32, lr=1e-3, resid_pen=1e-3)
    fin = log["final"]
    # residual learning must never be worse than the prior on train (delta>=0 init,
    # and the model is free to stay at delta=0); it should match-or-improve it.
    assert fin["mae_model_train"] <= fin["mae_prior_train"]
    # residual should not explode
    assert fin["delta_abs_mean"] < 1.0
    assert fin["delta_abs_max"] < 2.0
    # learning curve recorded
    assert len(log["learning_curve"]) >= 30
