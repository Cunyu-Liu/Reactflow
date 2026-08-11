#!/usr/bin/env python3
"""residual_spectrum_v2 — residual-learning spectrum model with per-position median
prior and detailed training diagnostics.

Why this design
---------------
The v1 spectrum MLP regressed the full scale-invariant spectrum directly and came out
*significantly WORSE* than the sequence-free per-position train median baseline
(skill ~ -0.18, permutation p = 0.005).  The permutation test confirmed the target IS
identifiable (p != 1.0), so the failure is the MODEL, not the target.  Diagnosis: a
WINDOW-dim head with ~(in_dim x 128 x 64 x 21) parameters can easily overfit the small
train-changer pool and drift its predictions far from the strong, stable per-position
median prior on held-out publications.

This module implements RESIDUAL LEARNING around that prior:

    pred[k] = prior[k] + delta[k]

where prior[k] is the train-changer weighted median at window position k (the same
sequence-free baseline that v1 could not beat), and delta[k] is what the model learns
from sequence/condition features.  Benefits:

  * The model is INITIALIZED to predict delta = 0 (== prior == the baseline), so its
    worst case on held-out data is the baseline itself.  It can only help, never
    catastrophically hurt, unless it generalizes (which is exactly what we want to
    test).
  * A residual-regularization term (L2 on delta) actively shrinks learned residuals
    toward 0, guarding against overfitting the small train pool.

Detailed diagnostics (the user's explicit request)
--------------------------------------------------
Every fold writes a structured log capturing:
  * PRIOR DISTRIBUTION : per-window-position prior[k] (mean/std/median/count of the
    train-weighted median and the underlying train residual distribution).  This
    tells us how strong/informative the sequence-free prior is per position, and
    where in the window signal concentrates.
  * RESIDUAL LEARNING CURVE : per-epoch train loss (weighted MAE on residual), plus
    stats of the learned |delta| distribution (mean, p90, max), the fraction of
    positions where the model moved away from the prior (|delta| > 0), and the
    signed mean.  This directly shows whether the model is drifting from the prior
    (overfitting) or staying near 0 (not learning).

The train function is pure-python/numpy+torch and returns (model, log_dict) so it is
unit-testable.  The prior computation is pure and fold-invariant.
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
DEFAULT_RESID_PEN = 1e-2  # L2 penalty coefficient on learned residuals


def per_position_prior(Y, W):
    """Per-window-position weighted median prior from train changers.

    Args:
        Y : (n, WINDOW) train target spectrum values.
        W : (n, WINDOW) train per-position weights (0/1).

    Returns:
        np.ndarray (WINDOW,) prior[k] = weighted median of Y[:, k] over W[:, k]==1,
        with per-position support (count of nonzero-weight train samples).
    """
    Y = np.asarray(Y, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, WINDOW = Y.shape
    prior = np.zeros(WINDOW, dtype=np.float64)
    cnt = np.zeros(WINDOW, dtype=np.int64)
    for k in range(WINDOW):
        y = Y[:, k]
        w = W[:, k]
        sel = w > 0
        cnt[k] = int(sel.sum())
        if cnt[k] == 0:
            prior[k] = 0.0
            continue
        yy = y[sel]
        ww = w[sel]
        order = np.argsort(yy)
        yy, ww = yy[order], ww[order]
        cw = np.cumsum(ww)
        idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
        idx = min(idx, len(yy) - 1)
        prior[k] = float(yy[idx])
    return prior, cnt


def prior_distribution(Y, W, prior):
    """Summarize the train residual distribution around the prior, per position.

    Returns a list of dicts (one per window position) with:
      prior        : the prior value
      count        : # train samples with weight>0 at this position
      mean         : weighted mean of Y at this position
      std          : weighted std of Y at this position
      abs_mean     : mean |Y - prior|  (typical deviation from prior)
      p90_abs      : 90th percentile of |Y - prior|
    """
    Y = np.asarray(Y, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, WINDOW = Y.shape
    out = []
    for k in range(WINDOW):
        sel = W[:, k] > 0
        cnt = int(sel.sum())
        d = {"k": k, "prior": float(prior[k]), "count": cnt}
        if cnt == 0:
            d.update({"mean": None, "std": None, "abs_mean": None, "p90_abs": None})
            out.append(d)
            continue
        y = Y[sel, k]
        w = W[sel, k]
        tot = w.sum()
        mean = float((w * y).sum() / tot)
        var = float((w * (y - mean) ** 2).sum() / tot)
        dev = np.abs(y - prior[k])
        d.update({
            "mean": mean,
            "std": math.sqrt(max(var, 0.0)),
            "abs_mean": float((w * dev).sum() / tot),
            "p90_abs": float(np.percentile(dev, 90)),
        })
        out.append(d)
    return out


class ResidualSpectrumMLP(nn.Module):
    """Maps features to a WINDOW-dim residual delta; final pred = prior + delta.

    The final linear layer bias is initialized to zeros so the network starts by
    predicting delta ~ 0 (== the prior == baseline), by construction.
    """
    def __init__(self, in_dim, out_dim, hidden=128, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        # init final layer to ~0 so initial prediction == prior (baseline)
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


def train_residual(
    X, Y, W, prior,
    epochs=DEFAULT_EPOCHS, bs=DEFAULT_BS, lr=DEFAULT_LR,
    resid_pen=DEFAULT_RESID_PEN, hidden=128, seed=0, device=None,
):
    """Train a residual-spectrum model.

    Loss = weighted MAE(residual) + resid_pen * mean(delta^2).

    Returns (model, log_dict) where log_dict is a structured training log with the
    per-epoch learning curve and final residual statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)

    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    model = ResidualSpectrumMLP(in_dim, out_dim, hidden=hidden, seed=seed)
    model = model.to(device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Wt = torch.from_numpy(W).to(device)
    prior_t = torch.from_numpy(prior).to(device)  # (WINDOW,)

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
            xb = Xt[idx]
            yb = Yt[idx]
            wb = Wt[idx]
            wh = wb.mean()
            if wh <= 0.0:
                continue
            delta = model(xb)                      # (B, WINDOW)
            pred = prior_t + delta                 # (B, WINDOW)
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

    # final residual statistics on the FULL train set (post-training)
    model.eval()
    with torch.no_grad():
        delta_all = model(Xt).cpu().numpy()              # (n, WINDOW)
    pred_all = prior + delta_all
    resid_all = Y - pred_all
    wsel = W > 0
    delta_sel = delta_all[wsel]
    resid_sel = resid_all[wsel]
    y_sel = Y[wsel]
    # prior value aligned to each selected (i,k)
    col_sel = np.argwhere(wsel)[:, 1]
    prior_sel = prior[col_sel]

    log = OrderedDict()
    log["prior"] = prior.tolist()
    log["prior_distribution"] = prior_distribution(Y, W, prior)
    log["learning_curve"] = curve
    log["final"] = {
        "delta_abs_mean": float(np.abs(delta_sel).mean()) if wsel.any() else 0.0,
        "delta_abs_p90": float(np.percentile(np.abs(delta_sel), 90)) if wsel.any() else 0.0,
        "delta_abs_max": float(np.abs(delta_sel).max()) if wsel.any() else 0.0,
        "delta_signed_mean": float(delta_sel.mean()) if wsel.any() else 0.0,
        "frac_delta_gt_0": float((np.abs(delta_sel) > 1e-6).mean()) if wsel.any() else 0.0,
        "resid_abs_mean": float(np.abs(resid_sel).mean()) if wsel.any() else 0.0,
        # model MAE vs baseline(prior) MAE on TRAIN (diagnostic of in-sample fit)
        "mae_model_train": float(np.abs(resid_sel).mean()) if wsel.any() else 0.0,
        "mae_prior_train": float(np.abs(y_sel - prior_sel).mean()) if wsel.any() else 0.0,
    }
    return model, log


def predict_delta(model, X, device=None):
    """Predict residual deltas (no grad) for held-out features.

    Args:
        model : trained ResidualSpectrumMLP.
        X     : (m, in_dim) float32 features.
        device: torch device.

    Returns:
        np.ndarray (m, WINDOW) float32 deltas (NOT added to prior).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        return model(Xt).cpu().numpy().astype(np.float32)