#!/usr/bin/env python3
"""residual_spectrum_v6 — POSITION-AWARE SELF-ATTENTION residual spectrum model
with a ROBUST STUDENT-t NEGATIVE-LOG-LIKELIHOOD head.

Motivation
----------
v4/v5 use weighted-MAE(residual) as the training loss.  The M2 response spectra
are noisy and heavy-tailed, and per-project memory the robust (Student-t /
right-censored) head was where the real method contribution came from.  This
variant replaces the MAE loss with a per-position Student-t NLL, which down-weights
outliers and learns an explicit per-position scale:

  delta_loc[k] = mu_head(x)[k]          (residual location, = prior + delta)
  log_scale[k] = s_head(x)[k]           (per-position log scale, unconstrained)
  nu           = fixed degrees of freedom (>=2)

  loss = -mean_w[ log StudentT( y[k]; loc=prior[k]+delta[k], scale[k], nu ) ]

The prediction used at evaluation time is still the location (prior + delta), so
WMAE skill / deviation-detection remain directly comparable to v4/v5.

Fail-safe residual property (same as v4): the location head's FINAL layer is
zero-initialised, so delta=0 (== prior) at init.  The scale head is initialised to
a small positive scale so the model starts at the baseline likelihood and can only
improve.

The positional-encoding cache fix from v4 is preserved.
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
DEFAULT_NHEAD = 4
DEFAULT_NLAYERS = 1
DEFAULT_NU = 4.0


def _pos_encoding(W: int, hidden: int, device) -> torch.Tensor:
    pe = torch.zeros(W, hidden, device=device)
    for k in range(W):
        for i in range(0, hidden, 2):
            pe[k, i] = math.sin(k / (10000.0 ** (i / hidden)))
            if i + 1 < hidden:
                pe[k, i + 1] = math.cos(k / (10000.0 ** (i / hidden)))
    return pe


def split_pos_glob(X, W, pos_dim):
    """Split flat (n, W*pos_dim + glob_dim) into (pos (n,W,pos_dim), glob (n,glob))."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    pos = X[:, :W * pos_dim].reshape(n, W, pos_dim)
    glob = X[:, W * pos_dim:]
    return pos, glob


def _student_t_nll(resid, log_scale, nu, w):
    """Weighted Student-t NLL (natural log).

    resid      : (B, W) residual y - loc
    log_scale  : (B, W) log scale
    nu         : float df
    w          : (B, W) non-negative weights
    Returns weighted mean NLL (scalar tensor).
    """
    scale = torch.exp(log_scale).clamp_min(1e-6)
    # Student-t log density:
    # log p = logGamma((nu+1)/2) - logGamma(nu/2) - 0.5*log(nu*pi)
    #         - log(scale) - ((nu+1)/2)*log(1 + (resid/scale)^2/nu)
    z2 = (resid / scale) ** 2
    lg1 = torch.lgamma(torch.tensor((nu + 1.0) / 2.0, device=resid.device))
    lg2 = torch.lgamma(torch.tensor(nu / 2.0, device=resid.device))
    const = lg1 - lg2 - 0.5 * math.log(nu * math.pi)
    nll = -(const - log_scale - ((nu + 1.0) / 2.0) * torch.log1p(z2 / nu))
    return (w * nll).sum() / w.sum().clamp_min(1.0)


class PositionAwareAttentionStudentT(nn.Module):
    """Per-position encoder + self-attention + per-position (loc, log_scale) heads.

    Location head final layer zero-init => delta=0 (fail-safe, == prior).
    Scale head init to log(scale0) so the model starts at a sensible likelihood.
    """
    def __init__(self, pos_dim, tail_dim, out_dim, hidden=128,
                 head_hidden=DEFAULT_HEAD_HIDDEN, nhead=DEFAULT_NHEAD,
                 nlayers=DEFAULT_NLAYERS, dropout=DEFAULT_DROPOUT,
                 scale0=0.3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.npos = out_dim
        self.hidden = hidden
        self.head_hidden = head_hidden
        self.pos_enc = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=dropout, activation="relu", batch_first=True)
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.tail_enc = nn.Sequential(
            nn.Linear(tail_dim, hidden), nn.ReLU(),
        )
        self.register_buffer("pe_cache", _pos_encoding(out_dim, hidden, torch.device("cpu")))

        in3 = hidden * 3
        # location head (W, in3, 1) zero-init
        self.W_loc = nn.Parameter(torch.empty(out_dim, in3, 1))
        self.b_loc = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.zeros_(self.W_loc)
        nn.init.zeros_(self.b_loc)
        # scale head (W, in3, 1), init to log(scale0)
        self.W_scale = nn.Parameter(torch.empty(out_dim, in3, 1))
        self.b_scale = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.zeros_(self.W_scale)
        nn.init.constant_(self.b_scale, math.log(max(scale0, 1e-3)))

    def forward(self, x):
        """x: flat feature tensor (B, in_dim).  Returns (delta (B,W), log_scale (B,W))."""
        B = x.shape[0]
        npos, pos_dim = self.npos, self.pos_enc[0].in_features
        tail_dim = self.tail_enc[0].in_features
        pos = x[:, :npos * pos_dim].reshape(B, npos, pos_dim)
        glob = x[:, npos * pos_dim:]
        h = self.pos_enc(pos)                                   # (B, W, hidden)
        pe = self.pe_cache.to(x.device).unsqueeze(0)
        h = h + pe
        h = self.attn(h)                                        # (B, W, hidden)
        pooled = h.mean(dim=1, keepdim=True)
        t = self.tail_enc(glob).unsqueeze(1)
        z = torch.cat([h, pooled.expand(B, npos, self.hidden), t.expand(B, npos, self.hidden)],
                      dim=-1)                                   # (B, W, 3*hidden)
        delta = torch.einsum("bwi,wio->bwo", z, self.W_loc) + self.b_loc.unsqueeze(0)
        log_scale = torch.einsum("bwi,wio->bwo", z, self.W_scale) + self.b_scale.unsqueeze(0)
        return delta.squeeze(-1), log_scale.squeeze(-1)


def train_posaware_student_t(
    pos_tr, glob_tr, Y, Wm, prior, pos_dim, tail_dim, out_dim,
    epochs=DEFAULT_EPOCHS, bs=DEFAULT_BS, lr=DEFAULT_LR,
    resid_pen=DEFAULT_RESID_PEN, hidden=128, head_hidden=DEFAULT_HEAD_HIDDEN,
    nhead=DEFAULT_NHEAD, nlayers=DEFAULT_NLAYERS, dropout=DEFAULT_DROPOUT,
    nu=DEFAULT_NU, scale0=0.3, seed=0, device=None, fast=False,
):
    """Train the Student-t attention model.

    Loss = weighted Student-t NLL + resid_pen * mean(delta^2).
    Returns (model, log).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = np.asarray(Y, dtype=np.float32)
    Wm = np.asarray(Wm, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)

    model = PositionAwareAttentionStudentT(
        pos_dim, tail_dim, out_dim, hidden=hidden, head_hidden=head_hidden,
        nhead=nhead, nlayers=nlayers, dropout=dropout, scale0=scale0, seed=seed)
    model = model.to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    pt = torch.from_numpy(np.asarray(pos_tr, dtype=np.float32)).to(device)
    gt = torch.from_numpy(np.asarray(glob_tr, dtype=np.float32)).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Wt = torch.from_numpy(Wm).to(device)
    prior_t = torch.from_numpy(prior).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = pt.shape[0]
    curve = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        ep_dabs = []
        ep_loss_gpu = None
        ep_dabs_gpu = []
        n_b = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pb, gb, yb, wb = pt[idx], gt[idx], Yt[idx], Wt[idx]
            if wb.mean() <= 0.0:
                continue
            xb = torch.cat([pb.reshape(pb.shape[0], -1), gb], dim=1)
            delta, log_scale = model(xb)
            pred = prior_t + delta
            resid = yb - pred
            nll = _student_t_nll(resid, log_scale, nu, wb)
            reg = resid_pen * (delta ** 2).mean()
            loss = nll + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            if fast:
                ep_loss_gpu = loss.detach() if ep_loss_gpu is None else ep_loss_gpu + loss.detach()
                ep_dabs_gpu.append(delta.detach().abs())
            else:
                ep_loss += float(loss.detach().cpu())
                ep_dabs.append(delta.detach().cpu().abs().numpy())
            n_b += 1
        if fast:
            ep_loss = float(ep_loss_gpu.cpu()) if ep_loss_gpu is not None else 0.0
            dabs = torch.cat(ep_dabs_gpu).cpu().numpy() if ep_dabs_gpu \
                else np.zeros((0, out_dim), dtype=np.float32)
        else:
            dabs = np.concatenate(ep_dabs) if ep_dabs else np.zeros((0, out_dim))
        curve.append({
            "epoch": ep,
            "loss": ep_loss / max(n_b, 1),
            "delta_abs_mean": float(dabs.mean()) if dabs.size else 0.0,
            "delta_abs_p90": float(np.percentile(dabs, 90)) if dabs.size else 0.0,
            "delta_abs_max": float(dabs.max()) if dabs.size else 0.0,
        })

    model.eval()
    with torch.no_grad():
        x_all = torch.cat([pt.reshape(n, -1), gt], dim=1)
        delta_all, log_scale_all = model(x_all)
        delta_all = delta_all.cpu().numpy()
        log_scale_all = log_scale_all.cpu().numpy()
    pred_all = prior + delta_all
    resid_all = Y - pred_all
    wsel = Wm > 0
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
        "scale_mean": float(np.exp(log_scale_all[wsel]).mean()) if wsel.any() else 0.0,
    }
    return model, log


def predict_posaware_student_t(model, pos_te, glob_te, device=None):
    """Predict residual deltas for held-out (pos, glob); returns (n, W) np.float32."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pt = torch.from_numpy(np.asarray(pos_te, dtype=np.float32)).to(device)
        gt = torch.from_numpy(np.asarray(glob_te, dtype=np.float32)).to(device)
        B = pt.shape[0]
        x = torch.cat([pt.reshape(B, -1), gt], dim=1)
        delta, _ = model(x)
        return delta.cpu().numpy().astype(np.float32)
