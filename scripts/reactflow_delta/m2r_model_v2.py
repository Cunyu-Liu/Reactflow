#!/usr/bin/env python3
"""m2r_model_v2.py — symmetric pair encoder for the M2R rescue_factor task.

GBDT (flat 213-d features) reaches +24.8% skill.  A symmetric pair encoder that
treats the (i,j) pair symmetrically may capture pairing mechanics better:

  pos_i = MLP(window_i)         # per-site context (WT + singles at site i)
  pos_j = MLP(window_j)         # per-site context (WT + singles at site j)
  h = [pos_i + pos_j, |pos_i - pos_j|, pos_i * pos_j, pos_i, pos_j, glob]
  out = MLP(h)

Key idea: the *difference* and *product* of the two site embeddings encode
whether the pair is complementary / whether disruption is symmetric — the
mechanism behind rescue.

Fail-safe: residual learning is not applicable here (target is rescue_factor,
not a deviation), so we use a plain regression with a median-anchored init.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEFAULT_EPOCHS = 40
DEFAULT_BS = 128
DEFAULT_LR = 1e-3


def build_pair_windows(s: "M2RPair", W=7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (win_i, win_j, glob) where win_i is the concatenated per-position
    features in a window around site i (WT react, WT err, singleA, singleB at
    each window offset), and glob is pair-level features (structure depth at
    i,j, bases, pairing flags, edit distance).

    This uses ONLY WT + single-mutant profiles (legal, non-circular).
    """
    import m2r_features_v1 as m2rf

    def window(arr, center):
        return np.array([m2rf._nan_to(arr[center + k]) if 0 <= center + k < len(arr) else 0.0
                         for k in range(-W, W + 1)])

    i, j = s.editA_seq_pos, s.editB_seq_pos
    # per-site: 4 arrays x 15 window positions
    win_i = np.concatenate([
        window(s.wt_reactivity, i), window(s.wt_error, i),
        window(s.singleA_reactivity, i), window(s.singleA_error, i),
        window(s.singleB_reactivity, i), window(s.singleB_error, i),
    ])
    win_j = np.concatenate([
        window(s.wt_reactivity, j), window(s.wt_error, j),
        window(s.singleA_reactivity, j), window(s.singleA_error, j),
        window(s.singleB_reactivity, j), window(s.singleB_error, j),
    ])
    # glob: structure depth/paired at i,j + bases + pairing + distances
    tgt = s.target_structure
    if len(tgt) < max(i, j) + 1:
        tgt = tgt + "." * (max(i, j) + 1 - len(tgt))
    pa, dp = m2rf.dot_to_depth(tgt)
    base_i = s.sequence[i] if i < len(s.sequence) else "N"
    base_j = s.sequence[j] if j < len(s.sequence) else "N"
    oh_i = np.array([1.0 if base_i == b else 0.0 for b in "ACGU"])
    oh_j = np.array([1.0 if base_j == b else 0.0 for b in "ACGU"])
    n = len(s.sequence)
    glob = np.concatenate([
        np.array([pa[i], pa[j], dp[i], dp[j],
                  float(abs(i - j) / max(n, 1)),
                  float(i / max(n, 1)), float(j / max(n, 1))]),
        oh_i, oh_j,
        np.array([1.0 if (base_i, base_j) in m2rf.WC_PAIRS else 0.0,
                  1.0 if (base_i, base_j) in m2rf.WOBBLE else 0.0]),
    ])
    return win_i.astype(np.float32), win_j.astype(np.float32), glob.astype(np.float32)


class SymmetricPairRescue(nn.Module):
    def __init__(self, site_dim, glob_dim, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.site_enc = nn.Sequential(
            nn.Linear(site_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.glob_enc = nn.Sequential(
            nn.Linear(glob_dim, hidden), nn.ReLU(),
        )
        # input: [e_i, e_j, e_i+e_j, e_i-e_j, e_i*e_j, glob_enc]
        in_dim = 5 * hidden + hidden
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, wi, wj, glob):
        e_i = self.site_enc(wi)
        e_j = self.site_enc(wj)
        g = self.glob_enc(glob)
        h = torch.cat([e_i, e_j, e_i + e_j, e_i - e_j, e_i * e_j, g], dim=-1)
        return self.head(h).squeeze(-1)


def train_symmetric(samples, keys, device=None, epochs=DEFAULT_EPOCHS,
                    bs=DEFAULT_BS, lr=DEFAULT_LR, seed=0, fast=False):
    """Train on a subset of samples (train fold), return (model, log)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_i = np.stack([build_pair_windows(s)[0] for s in samples])
    w_j = np.stack([build_pair_windows(s)[1] for s in samples])
    g = np.stack([build_pair_windows(s)[2] for s in samples])
    y = np.array([s.rescue_factor for s in samples], dtype=np.float32)

    model = SymmetricPairRescue(w_i.shape[1], g.shape[1], seed=seed).to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    Wi = torch.from_numpy(w_i).to(device)
    Wj = torch.from_numpy(w_j).to(device)
    G = torch.from_numpy(g).to(device)
    Yt = torch.from_numpy(y).to(device)
    n = len(samples)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    model.train()
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_b = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Wi[idx], Wj[idx], G[idx]), Yt[idx])
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu())
            n_b += 1
        curve.append({"epoch": ep, "loss": ep_loss / max(n_b, 1)})
    model.eval()
    with torch.no_grad():
        pred = model(Wi, Wj, G).cpu().numpy()
    log = {"learning_curve": curve,
           "final_mae": float(np.mean(np.abs(y - pred)))}
    return model, log


def predict_symmetric(model, samples, device=None):
    if device is None:
        device = next(model.parameters()).device
    w_i = np.stack([build_pair_windows(s)[0] for s in samples])
    w_j = np.stack([build_pair_windows(s)[1] for s in samples])
    g = np.stack([build_pair_windows(s)[2] for s in samples])
    model.eval()
    with torch.no_grad():
        Wi = torch.from_numpy(w_i).to(device)
        Wj = torch.from_numpy(w_j).to(device)
        G = torch.from_numpy(g).to(device)
        return model(Wi, Wj, G).cpu().numpy()
