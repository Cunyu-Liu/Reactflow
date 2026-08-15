#!/usr/bin/env python3
"""test_residual_spectrum_v6 — unit tests for the Student-t robust-head model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import residual_spectrum_v6 as rv6  # noqa: E402
import residual_spectrum_v2 as rv2  # noqa: E402

W = 21
POS_DIM = 7
TAIL_DIM = 13


def _mk_data(n=256, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, W, POS_DIM)).astype(np.float32)
    glob = rng.normal(size=(n, TAIL_DIM)).astype(np.float32)
    Y = np.tile(np.linspace(-0.3, 0.3, W), (n, 1)).astype(np.float32) \
        + rng.normal(scale=0.05, size=(n, W)).astype(np.float32)
    Wm = np.ones((n, W), dtype=np.float32)
    return pos, glob, Y, Wm


def test_untrained_model_zero_delta_but_scale_positive():
    pos, _, _, _ = _mk_data(seed=1)
    model = rv6.PositionAwareAttentionStudentT(POS_DIM, TAIL_DIM, W, seed=0)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(pos.reshape(pos.shape[0], -1))
        x = torch.cat([x, torch.zeros(pos.shape[0], TAIL_DIM)], dim=1)
        delta, log_scale = model(x)
    assert delta.shape == (pos.shape[0], W)
    assert torch.abs(delta).max().item() < 1e-6      # fail-safe zero delta
    scale = torch.exp(log_scale)
    assert scale.min().item() > 1e-4                  # positive scale at init
    assert torch.isfinite(log_scale).all()


def test_student_t_nll_is_finite_and_reduces_for_good_loc():
    rng = np.random.default_rng(2)
    resid = torch.tensor(rng.normal(size=(8, W)).astype(np.float32))
    log_scale = torch.full((8, W), np.log(0.3), dtype=torch.float32)
    w = torch.ones(8, W)
    nll = rv6._student_t_nll(resid, log_scale, nu=4.0, w=w)
    assert torch.isfinite(nll)
    # residual 0 (perfect loc) should give lower NLL than large residual
    resid0 = torch.zeros_like(resid)
    nll0 = rv6._student_t_nll(resid0, log_scale, nu=4.0, w=w)
    assert nll0.item() < nll.item()


def test_training_reduces_mae_vs_prior():
    pos, glob, Y, Wm = _mk_data(n=512, seed=4)
    prior, _ = rv2.per_position_prior(Y, Wm)
    model, log = rv6.train_posaware_student_t(
        pos, glob, Y, Wm, prior, POS_DIM, TAIL_DIM, W,
        epochs=40, bs=128, lr=1e-3, resid_pen=1e-3,
        hidden=64, head_hidden=16, nhead=4, nlayers=1, dropout=0.1,
        nu=4.0, scale0=0.3, seed=0, device=torch.device("cpu"))
    f = log["final"]
    assert f["mae_model_train"] < f["mae_prior_train"], \
        f"model {f['mae_model_train']:.4f} should beat prior {f['mae_prior_train']:.4f}"
    assert f["frac_delta_gt_0"] > 0.01
    assert f["scale_mean"] > 0.0


def test_predict_shape():
    pos, glob, Y, Wm = _mk_data(n=128, seed=6)
    prior, _ = rv2.per_position_prior(Y, Wm)
    model, _ = rv6.train_posaware_student_t(
        pos, glob, Y, Wm, prior, POS_DIM, TAIL_DIM, W,
        epochs=5, bs=64, lr=1e-3, resid_pen=1e-3,
        hidden=32, head_hidden=8, nhead=2, nlayers=1, dropout=0.0,
        nu=4.0, seed=1, device=torch.device("cpu"))
    pte = np.random.default_rng(9).normal(size=(7, W, POS_DIM)).astype(np.float32)
    gte = np.random.default_rng(10).normal(size=(7, TAIL_DIM)).astype(np.float32)
    d = rv6.predict_posaware_student_t(model, pte, gte)
    assert d.shape == (7, W)
    assert d.dtype == np.float32
    assert np.isfinite(d).all()
