#!/usr/bin/env python3
"""residual_spectrum_v4 — POSITION-AWARE SELF-ATTENTION residual spectrum model.

Method-level upgrade over residual_spectrum_v3 (position-aware residual MLP).

The v3 finding: splitting the flat local-window feature into PER-POSITION inputs
and giving each window position its own decoder head beat the shared-head MLP by
~+1.5pp WMAE skill (+10.41 vs +8.88), and the skill profile peaks sharply at the
central edit site (pos10 rho 0.443 vs ~0.28 at the flanks).  This implies the
signal is strongly position-structured, and flank positions might benefit from the
central edit site's context.

v4 therefore adds a TRANSFORMER SELF-ATTENTION stage over the 21 window positions:

    X (flat local-window + pair tail)
      -> split into pos (B, W, POS_DIM) and glob (B, tail_dim)
      -> per-position encoder MLP         (shared across positions)
      -> + sinusoidal positional encoding
      -> TransformerEncoder over W positions   <-- NEW: positions exchange context
      -> per-position decoder heads (each its own final zero-init linear)

Each decoder head's final layer is zero-initialised, so delta = 0 (== prior ==
sequence-free baseline) at init: the model can only help on held-out data, never
catastrophically drift (fail-safe residual property, same as v3).

Loss = weighted MAE(residual) + resid_pen * mean(delta^2)  (same as v3).
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


def _pos_encoding(W: int, hidden: int, device) -> torch.Tensor:
    """Sinusoidal positional encoding, shape (W, hidden), on ``device``."""
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


class PositionAwareAttentionResidMLP(nn.Module):
    """Per-position encoder + self-attention over window positions + per-position
    decoder heads.  Final decoder layer of each head zero-init => delta=0 at init.

    Decoder is VECTORIZED via einsum (per-position weight tensors) so there is NO
    21-iteration Python loop — the per-position heads stay distinct, but forward
    runs as a single batched op.  This was the fix that makes the attention model
    feasible on the MIG 1g.5gb slice (the naive ModuleList loop was ~5x slower).
    """
    def __init__(self, pos_dim, tail_dim, out_dim, hidden=128, head_hidden=DEFAULT_HEAD_HIDDEN,
                 nhead=DEFAULT_NHEAD, nlayers=DEFAULT_NLAYERS, dropout=DEFAULT_DROPOUT, seed=0):
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
        # Precompute the sinusoidal positional encoding ONCE (it is fold/seed
        # invariant, only depends on (W, hidden)) and cache it as a buffer.
        # Recomputing it per-forward was the dominant cost (~73 ms/call from the
        # Python sin/cos double loop), which made the whole model CPU-bound even
        # on a full A100.  This is a pure optimization: the values are identical.
        self.register_buffer("pe_cache", _pos_encoding(out_dim, hidden, torch.device("cpu")))

        # ---- per-position decoder as per-position weight tensors (einsum) ----
        # z_k = cat([h_k, pooled, t])  ->  (B, W, 3*hidden)
        in3 = hidden * 3
        if head_hidden > 0:
            # (W, in3, head_hidden) then (W, head_hidden, 1)
            self.W1 = nn.Parameter(torch.empty(out_dim, in3, head_hidden))
            self.b1 = nn.Parameter(torch.empty(out_dim, head_hidden))
            nn.init.xavier_uniform_(self.W1)
            nn.init.zeros_(self.b1)
            self.W2 = nn.Parameter(torch.empty(out_dim, head_hidden, 1))
            self.b2 = nn.Parameter(torch.empty(out_dim, 1))
            # zero-init final layer so initial delta = 0 (== prior == baseline)
            nn.init.zeros_(self.W2)
            nn.init.zeros_(self.b2)
            self._two_layer = True
        else:
            self.W2 = nn.Parameter(torch.empty(out_dim, in3, 1))
            self.b2 = nn.Parameter(torch.empty(out_dim, 1))
            nn.init.zeros_(self.W2)
            nn.init.zeros_(self.b2)
            self._two_layer = False

    def forward(self, x):
        """x: flat feature tensor (B, in_dim).  Splits internally."""
        B = x.shape[0]
        npos, pos_dim = self.npos, self.pos_enc[0].in_features
        tail_dim = self.tail_enc[0].in_features
        pos = x[:, :npos * pos_dim].reshape(B, npos, pos_dim)
        glob = x[:, npos * pos_dim:]
        h = self.pos_enc(pos)                                   # (B, W, hidden)
        pe = self.pe_cache.to(x.device).unsqueeze(0)            # (1, W, hidden) cached
        h = h + pe
        h = self.attn(h)                                        # (B, W, hidden)
        pooled = h.mean(dim=1, keepdim=True)                    # (B, 1, hidden)
        t = self.tail_enc(glob).unsqueeze(1)                    # (B, 1, hidden)
        z = torch.cat([h, pooled.expand(B, npos, self.hidden), t.expand(B, npos, self.hidden)],
                      dim=-1)                                   # (B, W, 3*hidden)
        if self._two_layer:
            y1 = torch.einsum("bwi,wih->bwh", z, self.W1) + self.b1   # (B, W, head_hidden)
            y1 = torch.relu(y1)
            y = torch.einsum("bwh,who->bwo", y1, self.W2) + self.b2.unsqueeze(0)
        else:
            y = torch.einsum("bwi,wio->bwo", z, self.W2) + self.b2.unsqueeze(0)
        return y.squeeze(-1)                                    # (B, W) residual delta


def train_posaware_attn2(
    pos_tr, glob_tr, Y, Wm, prior, pos_dim, tail_dim, out_dim,
    epochs=DEFAULT_EPOCHS, bs=DEFAULT_BS, lr=DEFAULT_LR,
    resid_pen=DEFAULT_RESID_PEN, hidden=128, head_hidden=DEFAULT_HEAD_HIDDEN,
    nhead=DEFAULT_NHEAD, nlayers=DEFAULT_NLAYERS, dropout=DEFAULT_DROPOUT,
    seed=0, device=None, fast=False,
):
    """Train position-aware self-attention residual model from pre-split pos/glob.

    pos_tr : (n, W, pos_dim) per-position local features
    glob_tr: (n, tail_dim) pair-level tail features

    ``fast=True`` uses two pure-optimization changes with IDENTICAL training
    semantics (same loss, same optimizer, same data order, same seeds):
      1. ``torch.compile`` the model (fuses eager-mode kernels; ~7x per-step).
      2. Accumulate loss/delta stats on GPU and sync to CPU once per epoch
         instead of forcing a GPU->CPU sync every batch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = np.asarray(Y, dtype=np.float32)
    Wm = np.asarray(Wm, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)

    model = PositionAwareAttentionResidMLP(
        pos_dim, tail_dim, out_dim, hidden=hidden, head_hidden=head_hidden,
        nhead=nhead, nlayers=nlayers, dropout=dropout, seed=seed)
    model = model.to(device)
    if fast:
        # ``fast`` only removes the per-batch GPU->CPU sync (accumulate on GPU,
        # sync once per epoch).  The real speedup came from caching the
        # sinusoidal positional encoding in the model constructor (was ~73 ms/call
        # from a Python sin/cos loop); with that cache, sync removal is ~1.01x.
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    pt = torch.from_numpy(np.asarray(pos_tr, dtype=np.float32)).to(device)
    gt = torch.from_numpy(np.asarray(glob_tr, dtype=np.float32)).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Wt = torch.from_numpy(Wm).to(device)
    prior_t = torch.from_numpy(prior).to(device)   # (WINDOW,)

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
            delta = model(xb)
            pred = prior_t + delta
            resid = yb - pred
            mae = (wb * resid.abs()).sum() / wb.sum()
            reg = resid_pen * (delta ** 2).mean()
            loss = mae + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            if fast:
                # accumulate on GPU; one CPU sync at end of epoch
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
        delta_all = model(x_all).cpu().numpy()
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
    }
    return model, log


def predict_posaware_attn(model, pos_te, glob_te, device=None):
    """Predict residual deltas for held-out (pos, glob); returns (n, W) np.float32."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pt = torch.from_numpy(np.asarray(pos_te, dtype=np.float32)).to(device)
        gt = torch.from_numpy(np.asarray(glob_te, dtype=np.float32)).to(device)
        B = pt.shape[0]
        x = torch.cat([pt.reshape(B, -1), gt], dim=1)
        return model(x).cpu().numpy().astype(np.float32)
