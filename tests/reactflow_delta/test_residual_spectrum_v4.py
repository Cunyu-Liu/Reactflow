#!/usr/bin/env python3
"""test_residual_spectrum_v4 — unit tests for the position-aware SELF-ATTENTION
residual spectrum model.

Verifies:
  * fail-safe zero-init: untrained model outputs delta ~ 0 (pred == prior)
  * position-aware structure: distinct per-position decoder heads
  * attention is active: the attention stage changes representations (attention
    outputs are not all-identical across positions when input differs)
  * training reduces train MAE below the sequence-free prior baseline
  * predict_posaware_attn shape/behavior on held-out (pos, glob)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import residual_spectrum_v4 as rv4  # noqa: E402
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


def test_untrained_model_outputs_zero_delta():
    pos, _, _, _ = _mk_data(seed=1)
    model = rv4.PositionAwareAttentionResidMLP(POS_DIM, TAIL_DIM, W, seed=0)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(pos.reshape(pos.shape[0], -1))
        x = torch.cat([x, torch.zeros(pos.shape[0], TAIL_DIM)], dim=1)
        delta = model(x)
    assert delta.shape == (pos.shape[0], W)
    assert torch.abs(delta).max().item() < 1e-6  # fail-safe zero delta at init


def test_position_aware_heads():
    model = rv4.PositionAwareAttentionResidMLP(POS_DIM, TAIL_DIM, W, seed=3)
    # per-position decoder weights must be DISTINCT across positions (W1[0] != W1[10])
    w0 = model.W1[0].detach().numpy()
    w10 = model.W1[10].detach().numpy()
    assert not np.allclose(w0, w10)
    # final decoder layer zero-init (fail-safe)
    assert np.abs(model.W2.detach().numpy()).max() < 1e-6


def test_attention_changes_representations():
    """With distinct position inputs, the attention stage output must differ
    across positions (i.e. attention is not degenerate / not all-collapsed)."""
    pos, _, _, _ = _mk_data(n=4, seed=5)
    model = rv4.PositionAwareAttentionResidMLP(POS_DIM, TAIL_DIM, W, seed=0)
    model.eval()
    with torch.no_grad():
        pt = torch.from_numpy(pos)
        h = model.pos_enc(pt)
        pe = rv4._pos_encoding(W, model.hidden, pt.device).unsqueeze(0)
        h = h + pe
        h_attn = model.attn(h)
    # attention output varies across positions (not identical everywhere)
    flat = h_attn[0]
    diffs = [torch.norm(flat[i] - flat[j]).item()
             for i in range(W) for j in range(i + 1, W)]
    assert max(diffs) > 1e-3


def test_training_reduces_mae_vs_prior():
    pos, glob, Y, Wm = _mk_data(n=512, seed=4)
    prior, _ = rv2.per_position_prior(Y, Wm)
    model, log = rv4.train_posaware_attn2(
        pos, glob, Y, Wm, prior, POS_DIM, TAIL_DIM, W,
        epochs=40, bs=128, lr=1e-3, resid_pen=1e-3,
        hidden=64, head_hidden=16, nhead=4, nlayers=1, dropout=0.1,
        seed=0, device=torch.device("cpu"))
    f = log["final"]
    assert f["mae_model_train"] < f["mae_prior_train"], \
        f"model {f['mae_model_train']:.4f} should beat prior {f['mae_prior_train']:.4f}"
    assert f["frac_delta_gt_0"] > 0.01


def test_predict_posaware_attn_shape():
    pos, glob, Y, Wm = _mk_data(n=128, seed=6)
    prior, _ = rv2.per_position_prior(Y, Wm)
    model, _ = rv4.train_posaware_attn2(
        pos, glob, Y, Wm, prior, POS_DIM, TAIL_DIM, W,
        epochs=5, bs=64, lr=1e-3, resid_pen=1e-3,
        hidden=32, head_hidden=8, nhead=2, nlayers=1, dropout=0.0,
        seed=1, device=torch.device("cpu"))
    pte = np.random.default_rng(9).normal(size=(7, W, POS_DIM)).astype(np.float32)
    gte = np.random.default_rng(10).normal(size=(7, TAIL_DIM)).astype(np.float32)
    d = rv4.predict_posaware_attn(model, pte, gte)
    assert d.shape == (7, W)
    assert d.dtype == np.float32
    assert np.isfinite(d).all()
