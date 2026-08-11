#!/usr/bin/env python3
"""resid_deepsets_seq_v1 — DeepSets residual-learning full-spectrum model with
STRICT-legal global sequence features.

Combines the three methodological fixes established by the preceding experiments:

  1. FULL per-position response spectrum target (rss.pair_response_spectrum).  This
     broke the p=1.0 degeneracy observed in the scalar-magnitude collapse (v1
     spectrum permutation p=0.005 => the target IS identifiable).
  2. RESIDUAL learning around the per-window-position train median prior (v2).
     The model is initialised at delta=0 (== prior == the sequence-free baseline),
     and a residual penalty shrinks learned deltas toward 0, so it can only help
     if the features carry transferable signal.  v2 proved that with the stock
     LOCAL-only 63-D features the learned deltas overfit train changers and did NOT
     transfer to held-out publications (skill < 0).
  3. POSITION-AWARE DeepSets whose global branch is augmented by STRICT-legal
     FULL-SEQUENCE features (k-mer composition + ViennaRNA WT folding) from
     global_seq_features_v1.  This supplies the global sequence/structural context
     that the failing local-only features lacked -- the single most-plausible
     untested lever for cross-publication transfer.

pred[k] = prior[k] + delta[k]

where prior[k] is the train-changer weighted median at window position k, and delta
comes from a set-based encoder (per-position MLP + sinusoidal positional encoding,
sum-pooled to a global context) feeding a per-position decoder, with the global
branch = [pair-level tail features | global sequence features].  The final decoder
layer is zero-initialised so the model starts at delta=0 (== prior), and an L2
residual penalty shrinks learned deltas toward 0.

Pure torch/numpy; the train function returns (model, log) and is unit-testable.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def pos_encoding(W: int, hidden: int, device) -> torch.Tensor:
    """Sinusoidal positional encoding, shape (W, hidden), on ``device``."""
    pe = torch.zeros(W, hidden, device=device)
    for k in range(W):
        for i in range(0, hidden, 2):
            pe[k, i] = math.sin(k / (10000.0 ** (i / hidden)))
            if i + 1 < hidden:
                pe[k, i + 1] = math.cos(k / (10000.0 ** (i / hidden)))
    return pe


def per_position_prior(Y, W):
    """Per-window-position weighted-median prior from train changers.

    Returns (prior[np.ndarray (WINDOW,)], cnt[np.ndarray (WINDOW,) int64]).
    """
    Y = np.asarray(Y, dtype=np.float64)
    Wm = np.asarray(W, dtype=np.float64)
    n, WINDOW = Y.shape
    prior = np.zeros(WINDOW, dtype=np.float64)
    cnt = np.zeros(WINDOW, dtype=np.int64)
    for k in range(WINDOW):
        y = Y[:, k]
        w = Wm[:, k]
        sel = w > 0
        cnt[k] = int(sel.sum())
        if cnt[k] == 0:
            prior[k] = 0.0
            continue
        yy, ww = y[sel], w[sel]
        order = np.argsort(yy)
        yy, ww = yy[order], ww[order]
        cw = np.cumsum(ww)
        idx = min(int(np.searchsorted(cw, 0.5 * cw[-1])), len(yy) - 1)
        prior[k] = float(yy[idx])
    return prior, cnt


def split_pos_glob(X, W, pos_dim):
    """Split flat (n, W*pos_dim + glob_dim) into (pos, glob)."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    pos = X[:, :W * pos_dim].reshape(n, W, pos_dim)
    glob = X[:, W * pos_dim:]
    return pos, glob


def concat_glob_seq(glob_base, seq_global):
    """Append global sequence features to the pair-level tail features.

    glob_base : (n, glob_base_dim) — tail of stock pair feature (ref/alt, edit
                position, condition one-hots).
    seq_global: (n, GLOBAL_SEQ_DIM) — STRICT-legal k-mer+ViennaRNA features.
    """
    return np.concatenate([np.asarray(glob_base, dtype=np.float32),
                           np.asarray(seq_global, dtype=np.float32)], axis=1)


class DeepSetsResidSpectrum(nn.Module):
    """Position-aware DeepSets regressing a WINDOW-dim residual delta.

    pred[k] = prior[k] + delta[k].  The final decoder layer is zero-initialised so
    the network starts at delta=0 (== prior == baseline) by construction.

    The decoder is applied as a single batched Linear over the (B, W, 2*hidden)
    tensor (instead of a 21-iteration Python loop) so training is fast.
    """
    def __init__(self, pos_dim, hidden, glob_dim, out_dim, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.phi = nn.Sequential(
            nn.Linear(pos_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.rho_global = nn.Sequential(
            nn.Linear(hidden + glob_dim, hidden), nn.ReLU(),
        )
        # applied to the (B, W, 2*hidden) stacked decoder input
        self.rho_out = nn.Sequential(
            nn.Linear(hidden + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # initialise final layer to 0 so initial prediction == prior (baseline)
        last = self.rho_out[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, pos_set, glob):
        B, W, _ = pos_set.shape
        pe = pos_encoding(W, self.hidden, pos_set.device)          # (W, hidden)
        pe = pe.unsqueeze(0).expand(B, W, self.hidden)             # (B, W, hidden)
        x = torch.cat([pos_set, pe], dim=-1)                       # (B, W, pos_dim+hidden)
        e = self.phi(x)                                            # (B, W, hidden)
        pooled = e.sum(dim=1)                                      # (B, hidden)
        g = self.rho_global(torch.cat([pooled, glob], dim=1))      # (B, hidden)
        # stack [g | e_k] for all positions at once -> (B, W, 2*hidden)
        g_b = g.unsqueeze(1).expand(B, W, self.hidden)             # (B, W, hidden)
        z = torch.cat([g_b, e], dim=-1)                            # (B, W, 2*hidden)
        return self.rho_out(z).squeeze(-1)                         # (B, W) residual delta


def train_resid_sets(model, pos, glob, Y, Wm, prior, device, seed=0,
                     epochs=30, bs=128, lr=1e-3, resid_pen=1e-2):
    """Masked weighted-MAE residual training; returns (model, log_dict).

    Loss = weighted-MAE(prior+delta, Y) over eligible positions
         + resid_pen * mean|delta|   (shrink learned residuals toward 0).

    log_dict records per-epoch loss and |delta| stats plus final residuals.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    pt = torch.from_numpy(np.asarray(pos, dtype=np.float32)).to(device)
    gt = torch.from_numpy(np.asarray(glob, dtype=np.float32)).to(device)
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(device)
    Wt = torch.from_numpy(np.asarray(Wm, dtype=np.float32)).to(device)
    prior_t = torch.from_numpy(np.asarray(prior, dtype=np.float32)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = pt.shape[0]
    model.train()
    learning_curve = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        ep_dm = 0.0
        ep_dp90 = 0.0
        ep_dmax = 0.0
        ep_dfrac = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            delta = model(pt[idx], gt[idx])
            pred = prior_t.unsqueeze(0) + delta
            w = Wt[idx]
            wh = w.mean()
            if wh <= 0.0:
                continue
            mae = (w * (pred - Yt[idx]).abs()).sum() / w.sum()
            reg = resid_pen * delta.abs().mean()
            loss = mae + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            dabs = delta.detach().abs()
            ep_loss += float(mae.detach())
            ep_dm += float(dabs.mean())
            ep_dp90 += float(torch.quantile(dabs, 0.9))
            ep_dmax += float(dabs.max())
            ep_dfrac += float((dabs > 0).float().mean())
            nb += 1
        if nb == 0:
            break
        learning_curve.append({
            "epoch": ep, "loss": ep_loss / nb, "delta_abs_mean": ep_dm / nb,
            "delta_abs_p90": ep_dp90 / nb, "delta_abs_max": ep_dmax / nb,
            "frac_delta_gt_0": ep_dfrac / nb,
        })

    # final residual stats on train
    model.eval()
    with torch.no_grad():
        delta_all = model(pt, gt).detach().abs()
        dabs = delta_all
        # fraction moved off prior per position, pooled
        frac_gt0 = float((dabs > 0).float().mean())
        # train model-vs-prior MAE
        pred_all = prior_t.unsqueeze(0) + model(pt, gt).detach()
        den = Wt.sum()
        mae_model = float((Wt * (pred_all - Yt).abs()).sum() / den) if den > 0 else float("nan")
        mae_prior = float((Wt * (prior_t.unsqueeze(0) - Yt).abs()).sum() / den) if den > 0 else float("nan")
    log = {
        "final": {
            "delta_abs_mean": float(dabs.mean()),
            "delta_abs_p90": float(torch.quantile(dabs, 0.9)),
            "delta_abs_max": float(dabs.max()),
            "frac_delta_gt_0": frac_gt0,
            "mae_model_train": mae_model,
            "mae_prior_train": mae_prior,
        },
        "learning_curve": learning_curve,
    }
    return model, log


def predict_resid_sets(model, pos, glob, prior, device):
    """Return pred = prior + delta for held-out (pos, glob)."""
    model.eval()
    with torch.no_grad():
        pt = torch.from_numpy(np.asarray(pos, dtype=np.float32)).to(device)
        gt = torch.from_numpy(np.asarray(glob, dtype=np.float32)).to(device)
        delta = model(pt, gt)
        prior_t = torch.from_numpy(np.asarray(prior, dtype=np.float32)).to(device)
        return (prior_t.unsqueeze(0) + delta).cpu().numpy().astype(np.float32)
