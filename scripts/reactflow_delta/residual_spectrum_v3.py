#!/usr/bin/env python3
"""residual_spectrum_v3 — POSITION-AWARE residual spectrum model.

Why this is a stronger per-position model than residual_spectrum_v2
-------------------------------------------------------------------
v2 (ResidualSpectrumMLP) maps the whole local-window feature vector to all WINDOW
positions through ONE shared output head (in_dim -> 128 -> 64 -> WINDOW).  Every
window position shares the same output weights, so it cannot give position-specific
response patterns beyond what the (position-aligned) input already encodes.

v3 decouples the shared feature ENCODING from per-position DECODING:

    shared trunk :  X (local-window features)  ->  h   (compact representation)
    per-position :  concat(h, pos_emb[k])  ->  delta[k]   (k = 0..WINDOW-1)

Each window position k has its OWN decoder head plus a learned position embedding.
This lets the central edit site (k ~ half) learn a different reaction mapping than
the window flanks, matching the observed skill profile (peak at the edit site).

Fail-safe residual property is preserved: every per-position decoder's final layer
is zero-initialized, so delta = 0 (== prior == sequence-free baseline) at init; the
model can only help on held-out data, never catastrophically drift.

Regularization (needed because per-position heads add capacity):
  * Dropout in the shared trunk and each head.
  * L2 residual penalty mean(delta^2) (same as v2).

Loss = weighted MAE(residual) + resid_pen * mean(delta^2).
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

DEFAULT_EPOCHS = 30
DEFAULT_BS = 128
DEFAULT_LR = 1e-3
DEFAULT_RESID_PEN = 1e-2
DEFAULT_DROPOUT = 0.1
DEFAULT_HEAD_HIDDEN = 32


class PositionAwareResidualMLP(nn.Module):
    """Shared-trunk + per-position-decoder residual model.

    Args:
        in_dim : input feature dim.
        out_dim: output window length (= WINDOW positions).
        hidden : trunk hidden dim.
        head_hidden: per-position decoder hidden dim (0 => single linear head).
        dropout: dropout probability in trunk + heads.
        seed   : torch manual seed.
    """
    def __init__(self, in_dim, out_dim, hidden=128, head_hidden=DEFAULT_HEAD_HIDDEN,
                 dropout=DEFAULT_DROPOUT, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.npos = out_dim
        trunk = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            trunk.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk)
        # learned position embeddings, one per window position.
        # Small random init is safe: each head's FINAL layer is zero-init below, so
        # delta stays 0 at init regardless of the embedding, preserving the fail-safe
        # residual property, while embeddings are distinct from the start (position-aware).
        self.pos_emb = nn.Parameter(torch.zeros(out_dim, hidden))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.heads = nn.ModuleList()
        for _ in range(out_dim):
            layers = []
            if head_hidden > 0:
                layers += [nn.Linear(hidden * 2, head_hidden), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(head_hidden, 1))
            else:
                layers.append(nn.Linear(hidden * 2, 1))
            head = nn.Sequential(*layers)
            # zero-init final layer so initial delta = 0 (== prior == baseline)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads.append(head)

    def forward(self, x):
        h = self.trunk(x)                      # (B, hidden)
        B = h.shape[0]
        outs = []
        for k in range(self.npos):
            hk = torch.cat([h, self.pos_emb[k].expand(B, -1)], dim=1)  # (B, 2*hidden)
            outs.append(self.heads[k](hk))     # (B, 1)
        return torch.cat(outs, dim=1)          # (B, npos)


def train_posaware(
    X, Y, W, prior,
    epochs=DEFAULT_EPOCHS, bs=DEFAULT_BS, lr=DEFAULT_LR,
    resid_pen=DEFAULT_RESID_PEN, hidden=128, head_hidden=DEFAULT_HEAD_HIDDEN,
    dropout=DEFAULT_DROPOUT, seed=0, device=None,
):
    """Train a position-aware residual model.

    Loss = weighted MAE(residual) + resid_pen * mean(delta^2).

    Returns (model, log_dict) with per-epoch learning curve + final residual stats.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)

    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    model = PositionAwareResidualMLP(in_dim, out_dim, hidden=hidden,
                                     head_hidden=head_hidden, dropout=dropout, seed=seed)
    model = model.to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Wt = torch.from_numpy(W).to(device)
    prior_t = torch.from_numpy(prior).to(device)   # (WINDOW,)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xt.shape[0]
    curve = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        ep_delta_abs = []
        n_b = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb, wb = Xt[idx], Yt[idx], Wt[idx]
            if wb.mean() <= 0.0:
                continue
            delta = model(xb)                       # (B, WINDOW)
            pred = prior_t + delta
            resid = yb - pred
            mae = (wb * resid.abs()).sum() / wb.sum()
            reg = resid_pen * (delta ** 2).mean()
            loss = mae + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu())
            ep_delta_abs.append(delta.detach().cpu().abs().numpy())
            n_b += 1
        dabs = np.concatenate(ep_delta_abs) if ep_delta_abs else np.zeros((0, out_dim))
        curve.append({
            "epoch": ep,
            "loss": ep_loss / max(n_b, 1),
            "delta_abs_mean": float(dabs.mean()) if dabs.size else 0.0,
            "delta_abs_p90": float(np.percentile(dabs, 90)) if dabs.size else 0.0,
            "delta_abs_max": float(dabs.max()) if dabs.size else 0.0,
        })

    model.eval()
    with torch.no_grad():
        delta_all = model(Xt).cpu().numpy()          # (n, WINDOW)
    pred_all = prior + delta_all
    resid_all = Y - pred_all
    wsel = W > 0
    delta_sel = delta_all[wsel]
    resid_sel = resid_all[wsel]
    y_sel = Y[wsel]
    col_sel = np.argwhere(wsel)[:, 1]
    prior_sel = prior[col_sel]

    log = OrderedDict()
    log["learning_curve"] = curve
    log["final"] = {
        "delta_abs_mean": float(np.abs(delta_sel).mean()) if wsel.any() else 0.0,
        "delta_abs_p90": float(np.percentile(np.abs(delta_sel), 90)) if wsel.any() else 0.0,
        "delta_abs_max": float(np.abs(delta_sel).max()) if wsel.any() else 0.0,
        "delta_signed_mean": float(delta_sel.mean()) if wsel.any() else 0.0,
        "frac_delta_gt_0": float((np.abs(delta_sel) > 1e-6).mean()) if wsel.any() else 0.0,
        "resid_abs_mean": float(np.abs(resid_sel).mean()) if wsel.any() else 0.0,
        "mae_model_train": float(np.abs(resid_sel).mean()) if wsel.any() else 0.0,
        "mae_prior_train": float(np.abs(y_sel - prior_sel).mean()) if wsel.any() else 0.0,
    }
    return model, log


def predict_delta(model, X, device=None):
    """Predict residual deltas (no grad) for held-out features."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        return model(Xt).cpu().numpy().astype(np.float32)


