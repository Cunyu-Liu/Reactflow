#!/usr/bin/env python3
"""m2r_deepsets_v1.py — full-profile attention model for M2R rescue_factor.

Motivation (m2r_noise_floor_v1.py): measurement noise allows R2 ~0.82-0.99, but
the window-feature GBDT reaches only R2 0.37, and GBDT cannot use global
full-profile features (tested: 0.2507 vs 0.2517).  The rescue_factor is defined
from FULL-profile RMSD over the design region, so a model that sees the full
WT + singleA + singleB reactivity/error profiles (LEGAL: no double-mutant
profile) and learns the global aggregation via self-attention is the natural
method-level lever.

Architecture: per-position MLP encoder -> +sinusoidal pos encoding -> Transformer
encoder over full-profile positions -> global mean-pool -> MLP head -> scalar
rescue prediction.  Per-position features: [wt_react, wt_err, sA_react, sA_err,
sB_react, sB_err] (6 dims), plus site indicator flags for i and j.

Run in design-level LOO (exchangeable unit = (puzzle, method)), GPU-only.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r

SEED = 20260817
MAXLEN = 400


def _prof(p, n):
    a = np.full(n, np.nan, dtype=np.float64)
    for k, v in enumerate(p):
        if k < n:
            a[k] = v
    return a


def _pos_encoding(L, d_model, device="cpu"):
    pe = torch.zeros(L, d_model)
    pos = torch.arange(L, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(device)


class FullProfileAttention(nn.Module):
    """Transformer encoder over full-profile per-position features -> scalar."""
    def __init__(self, pos_dim, hidden=128, nhead=4, nlayers=2, dropout=0.1,
                 head_hidden=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.pos_enc = nn.Sequential(nn.Linear(pos_dim, hidden), nn.ReLU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=dropout, activation="relu", batch_first=True)
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1))
        # zero-init final to start near 0 (same scale as mean target)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, mask):
        """x: (B, L, pos_dim); mask: (B, L) bool valid (True=valid)."""
        B, L = x.shape[0], x.shape[1]
        h = self.pos_enc(x)                              # (B,L,hidden)
        pe = self._pe.to(x.device).unsqueeze(0)          # (1,L,hidden)
        h = h + pe
        # transformer with src_key_padding_mask (True = pad/ignored)
        pad = ~mask
        h = self.attn(h, src_key_padding_mask=pad)       # (B,L,hidden)
        # masked mean pool
        w = mask.unsqueeze(-1).float()                   # (B,L,1)
        denom = w.sum(dim=1).clamp(min=1.0)              # (B,1)
        pooled = (h * w).sum(dim=1) / denom              # (B,hidden)
        return self.head(pooled).squeeze(-1)             # (B,)


def build_dataset(samples, maxlen=MAXLEN):
    """Build per-sample (pos_features, mask, y, keys).  pos_dim = 8.
    [wt_react, wt_err, sA_react, sA_err, sB_react, sB_err, is_i, is_j]
    """
    X, M, y, keys = [], [], [], []
    for s in samples:
        n = len(s.wt_reactivity)
        n = min(n, maxlen)
        wt_r = _prof(s.wt_reactivity, n); wt_e = _prof(s.wt_error, n)
        a_r = _prof(s.singleA_reactivity, n); a_e = _prof(s.singleA_error, n)
        b_r = _prof(s.singleB_reactivity, n); b_e = _prof(s.singleB_error, n)
        feats = np.stack([wt_r, wt_e, a_r, a_e, b_r, b_e], axis=1)  # (n,6)
        ok = np.isfinite(feats).all(axis=1)
        # NaN -> 0: mask already tells the model to ignore invalid positions,
        # but NaN would still poison the position encoder through the forward.
        feats = np.where(np.isfinite(feats), feats, 0.0)
        # fixed clip (leak-free, monotonic): reactivity to [-3,10], error to
        # [0,10].  Outlier error values (max ~2865) otherwise dominate the
        # position-encoder gradient scale and stall training.
        feats[:, 0] = np.clip(feats[:, 0], -3.0, 10.0)   # wt_react
        feats[:, 2] = np.clip(feats[:, 2], -3.0, 10.0)   # sA_react
        feats[:, 4] = np.clip(feats[:, 4], -3.0, 10.0)   # sB_react
        feats[:, 1] = np.clip(feats[:, 1], 0.0, 10.0)    # wt_err
        feats[:, 3] = np.clip(feats[:, 3], 0.0, 10.0)    # sA_err
        feats[:, 5] = np.clip(feats[:, 5], 0.0, 10.0)    # sB_err
        # site indicators (0-indexed full-seq positions; may be >= n)
        si = np.zeros(n, dtype=np.float64); sj = np.zeros(n, dtype=np.float64)
        if s.editA_seq_pos < n:
            si[s.editA_seq_pos] = 1.0
        if s.editB_seq_pos < n:
            sj[s.editB_seq_pos] = 1.0
        feats = np.concatenate([feats, si[:, None], sj[:, None]], axis=1)  # (n,8)
        X.append(feats)
        M.append(ok)
        y.append(s.rescue_factor)
        keys.append(s.design_id)
    # pad to maxlen
    L = max(min(len(x), maxlen) for x in X)
    Xp = np.zeros((len(X), L, 8), dtype=np.float32)
    Mp = np.zeros((len(X), L), dtype=bool)
    for i, (x, m) in enumerate(zip(X, M)):
        l = min(len(x), L)
        Xp[i, :l] = x[:l]
        Mp[i, :l] = m[:l]
    return torch.from_numpy(Xp), torch.from_numpy(Mp), np.array(y), np.array(keys, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-device", default="1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--head-hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    args = ap.parse_args()

    os_env = __import__("os")
    os_env.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    free_mem, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    print(f"[m2r_ds] GPU OK: cuda_visible={args.cuda_device} name={gpu_name} "
          f"free={free_mem/1e9:.1f}GB", flush=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, M, y, keys = build_dataset(samples, args.maxlen)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)
    print(f"[m2r_ds] n_samples={len(y)} n_designs={n_des} L={X.shape[1]} "
          f"pos_dim={X.shape[2]}", flush=True)

    y_med = float(np.median(y))
    mae_bl = float(np.mean(np.abs(y - y_med)))
    y_c = y - y_med   # residual target: zero-init head -> predict median at init

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds = np.zeros(len(y))
    t0 = time.time()
    for fi, held in enumerate(des_list):
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = y_med
            continue
        xt, mt, yt = X[m], M[m], torch.from_numpy(y_c[m].astype(np.float32))
        xv, mv = X[~m], M[~m]
        model = FullProfileAttention(
            X.shape[2], hidden=args.hidden, nhead=args.nhead, nlayers=args.nlayers,
            dropout=args.dropout, head_hidden=args.head_hidden, seed=SEED).to(device)
        model._pe = _pos_encoding(X.shape[1], args.hidden, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lossf = nn.MSELoss()
        n_t = len(yt)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n_t, generator=torch.Generator().manual_seed(SEED))
            for i in range(0, n_t, args.bs):
                idx = perm[i:i + args.bs]
                bx = xt[idx].to(device); bm = mt[idx].to(device); by = yt[idx].to(device)
                opt.zero_grad()
                pr = model(bx, bm)
                loss = lossf(pr, by)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            xvv = xv.to(device); mvv = mv.to(device)
            p = model(xvv, mvv).cpu().numpy()
        preds[~m] = y_med + p
        if fi % 30 == 0:
            print(f"[m2r_ds] fold {fi}/{n_des} held={held} elapsed={time.time()-t0:.0f}s",
                  flush=True)

    wall = round(time.time() - t0, 1)
    mae = float(np.mean(np.abs(y - preds)))
    skill = 1.0 - mae / mae_bl
    r2 = 1.0 - float(np.sum((y - preds) ** 2)) / float(np.sum((y - y.mean()) ** 2))

    report = {
        "schema": "reactflow_delta.m2r_deepsets.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": int(len(y)), "n_designs": n_des,
        "model": "full-profile Transformer (8 pos dims)",
        "config": {"epochs": args.epochs, "bs": args.bs, "lr": args.lr,
                   "hidden": args.hidden, "nhead": args.nhead,
                   "nlayers": args.nlayers, "head_hidden": args.head_hidden,
                   "dropout": args.dropout, "weight_decay": args.weight_decay},
        "baseline_mae": mae_bl,
        "result": {"mae": mae, "skill": skill, "r2": r2},
        "wall_seconds": wall,
    }
    np.savez(out / "m2r_deepsets_oof.npz", preds=preds, y=y, keys=keys)
    (out / "m2r_deepsets_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[m2r_ds] DONE -> {out}")
    print(f"  skill={skill:+.4f} R2={r2:.4f} MAE={mae:.4f} wall={wall}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
