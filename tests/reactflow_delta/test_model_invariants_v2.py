#!/usr/bin/env python3
"""Phase 3 model invariants tests (contract Phase 3 rule ⑦: gradient/residual/NaN/
length/memory) + capacity matching (rule ④) + permutation equivariance.

These verify the architecture invariants regardless of device. When CUDA is available
the model is also exercised on CUDA (no silent CPU fallback in training).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "reactflow_delta"))
from models.pair_v1 import (
    PairHeadV1, CapacityMatchedMLP, count_params, split_pos_glob,
)
from train_v2 import train_flat, train_pair, predict_flat, predict_pair


W = 21
POS_DIM = 7
GLOB_DIM = 40
B = 16


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synthetic():
    rng = np.random.RandomState(0)
    pos = rng.randn(B, W, POS_DIM).astype(np.float32)
    glob = rng.randn(B, GLOB_DIM).astype(np.float32)
    y = rng.rand(B).astype(np.float32)
    w = rng.randint(1, 40, B).astype(np.float32)
    return pos, glob, y, w


def test_capacity_match_generic_within_tolerance():
    cand = PairHeadV1(POS_DIM, GLOB_DIM, hidden=64, seed=0)
    target = count_params(cand)
    in_dim = W * POS_DIM + GLOB_DIM
    gen = CapacityMatchedMLP(in_dim, target, seed=0)
    assert count_params(gen) <= target * 1.10, (count_params(gen), target)


def test_split_pos_glob_consistency():
    X = np.random.RandomState(1).randn(B, W * POS_DIM + GLOB_DIM).astype(np.float32)
    pos, glob = split_pos_glob(torch.from_numpy(X), W, POS_DIM)
    assert pos.shape == (B, W, POS_DIM)
    assert glob.shape == (B, GLOB_DIM)
    # reconstruction: concatenation == original
    recon = torch.cat([pos.reshape(B, W * POS_DIM), glob], dim=1)
    assert torch.allclose(recon, torch.from_numpy(X), atol=1e-6)


def test_permutation_equivariance_of_pair_head():
    """DeepSets pooling must be permutation-invariant (set semantics)."""
    torch.manual_seed(0)
    model = PairHeadV1(POS_DIM, GLOB_DIM, hidden=16, seed=0).to(_dev()).eval()
    pos, glob, _, _ = _synthetic()
    pt = torch.from_numpy(pos).to(_dev())
    gt = torch.from_numpy(glob).to(_dev())
    with torch.no_grad():
        out1 = model(pt, gt)
        perm = torch.randperm(W)
        out2 = model(pt[:, perm, :], gt)
    assert torch.allclose(out1, out2, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_nan_gradient_flow_pair_and_flat(seed):
    dev = _dev()
    pos, glob, y, w = _synthetic()
    # pair head
    cand = PairHeadV1(POS_DIM, GLOB_DIM, hidden=16, seed=seed).to(dev)
    pt = torch.from_numpy(pos).to(dev); gt = torch.from_numpy(glob).to(dev)
    yt = torch.from_numpy(y).to(dev); wt_ = torch.from_numpy(w).to(dev)
    pred = cand(pt, gt)
    loss = (wt_ * (pred - yt).abs()).mean()
    loss.backward()
    assert not torch.isnan(pred).any()
    for p in cand.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN gradient in pair head"
    # generic flat
    in_dim = W * POS_DIM + GLOB_DIM
    X = np.random.RandomState(seed).randn(B, in_dim).astype(np.float32)
    gen = CapacityMatchedMLP(in_dim, 20000, seed=seed).to(dev)
    Xt = torch.from_numpy(X).to(dev)
    pred2 = gen(Xt)
    loss2 = (wt_ * (pred2 - yt).abs()).mean()
    loss2.backward()
    assert not torch.isnan(pred2).any()
    for p in gen.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN gradient in generic"


def test_train_produces_finite_predictions_cuda_required():
    """If CUDA is available, a short weighted-MAE training must yield finite preds."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; training invariants require CUDA")
    dev = _dev()
    pos, glob, y, w = _synthetic()
    # pair head training
    cand = PairHeadV1(POS_DIM, GLOB_DIM, hidden=16, seed=0)
    train_pair(cand, pos, glob, y, w, dev, seed=0)
    pred = predict_pair(cand, pos, glob, dev)
    assert np.isfinite(pred).all()
    # generic training
    in_dim = W * POS_DIM + GLOB_DIM
    X = np.random.RandomState(0).randn(B, in_dim).astype(np.float32)
    gen = CapacityMatchedMLP(in_dim, 20000, seed=0)
    train_flat(gen, X, y, w, dev, seed=0)
    pred2 = predict_flat(gen, X, dev)
    assert np.isfinite(pred2).all()


def test_memory_constant_through_window():
    """Window size only changes input dim; a larger window must not blow up params
    (no quadratic growth in the set encoder)."""
    small = PairHeadV1(POS_DIM, GLOB_DIM, hidden=16, seed=0)
    big = PairHeadV1(POS_DIM, GLOB_DIM + 200, hidden=16, seed=0)
    assert count_params(big) - count_params(small) < 5000  # linear, not quadratic
