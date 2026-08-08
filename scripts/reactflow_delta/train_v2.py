#!/usr/bin/env python3
"""Phase 3 training helpers (weighted-MAE regression, CUDA-required)."""
from __future__ import annotations

import numpy as np
import torch

EPOCHS = 30
BS = 128
LR = 1e-3


def _assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")


def train_flat(model, X, y, w, device, seed=0):
    """Weighted-MAE training for a flat-input model (generic MLP / linear head)."""
    _assert_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    wt_ = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = Xt.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            pred = model(Xt[idx])
            loss = (wt_[idx] * (pred - yt[idx]).abs()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_pair(model, pos, glob, y, w, device, seed=0):
    """Weighted-MAE training for a set-input model (PairHeadV1)."""
    _assert_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    wt_ = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = pos.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = np.random.permutation(n)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            pb = torch.from_numpy(pos[idx]).to(device)
            gb = torch.from_numpy(glob[idx]).to(device)
            yb, wb = yt[idx], wt_[idx]
            pred = model(pb, gb)
            loss = (wb * (pred - yb).abs()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_flat(model, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        return model(Xt).cpu().numpy()


def predict_pair(model, pos, glob, device):
    model.eval()
    with torch.no_grad():
        pt = torch.from_numpy(np.asarray(pos, dtype=np.float32)).to(device)
        gt = torch.from_numpy(np.asarray(glob, dtype=np.float32)).to(device)
        return model(pt, gt).cpu().numpy()
