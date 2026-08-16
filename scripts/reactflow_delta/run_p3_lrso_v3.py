#!/usr/bin/env python3
"""run_p3_lrso_v3: spec-compliant plain RFD-LRSO vs fold-specific B*=Direct*
(contract 10.2, 9.1, 12.5, 7.4, 14.1).

v3 SUPERSEDES v1/v2 which did NOT follow the frozen spec:
  * v1/v2 detached the WT context encoder to a fixed random feature map during
    training (contract 10.2 requires a TRAINABLE 2-block attention encoder).
  * v1/v2 filled missing mutant target reactivity with 0.0 and then computed the
    observation mask from the already-filled array, so every position was
    treated as observable and the model was trained to predict 0 at unobserved
    positions (contract 7.4 / 14.1.3: missing != 0; target mask is
    training/evaluator-side mask; unobservable = TARGET_UNOBSERVABLE, never 0).
  * v1/v2 trained a single seed for 6-12 fixed epochs with NO inner validation
    (contract 10.2: epochs <= 200 with inner CRPS early stopping; inner 4-fold
    puzzle-grouped split; lr {3e-4,1e-3}, wd {0,1e-4}, likelihood
    {Gaussian,Student-t} chosen by inner validation; contract 9.1: fixed
    five-seed {0..4} equal-weight deployment mixture).

v3 per outer fold (held puzzle fully excluded from fit/tuning/early-stop):
  - inner 4-fold puzzle-grouped validation (fold.inner_groups) selects
    {lr, wd, likelihood} AND the early-stopped epoch count by inner OOF CRPS;
  - train the final model on the full outer-train for exactly that epoch count,
    for each of the fixed seeds {0,1,2,3,4};
  - B*=Direct* = reg_direct (ridge on direct chemistry template), outer-train fit;
  - held-puzzle full-construct predictive Gaussian mixture (5 seeds) CRPS vs B*;
  - D_p^P3 = L_B* - L_LRSO ; 20-puzzle two-sided 95% t CI (positive => LRSO better).

Outcome-blind: held outcomes only in the evaluator. Missing targets never enter
the loss (masked); records with no target-qualified position are
TARGET_UNOBSERVABLE for training but still produce full-construct held
predictions. WT missing positions are excluded from attention, training loss and
held scoring; the WT reactivity numeric input is mean-filled (NOT 0) and a
WT-observed binary token is appended (contract WT tokens include "WT observed
mask"). Predictive scale uses a positive parameterization (softplus + train-only
floor, contract 10.2.1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as _st

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4, _grouped_folds
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian, mixture_crps
from scripts.reactflow_delta.p2_learnability import (
    d_p_p2, leave_one_puzzle_influence, puzzle_level_ci20, studentized_sign_flip,
)
from scripts.reactflow_delta.lrso_v1 import RFDLRSO


def _crps_gaussian_vec(loc: np.ndarray, scale: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Gaussian CRPS (same exact closed form as evaluator_crps_v1)."""
    m = y - loc
    s = np.maximum(scale, 1e-12)
    e_xy = s * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * (m / s) ** 2) \
        + m * (2.0 * _st.norm.cdf(m / s) - 1.0)
    return e_xy - s / np.sqrt(np.pi)


def _exp_abs_norm_vec(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Vectorized E|W| for W ~ N(m, s^2)."""
    s = np.maximum(s, 1e-12)
    return s * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * (m / s) ** 2) \
        + m * (2.0 * _st.norm.cdf(m / s) - 1.0)


def _mixture_crps_vec(locs: list[np.ndarray], scales: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Vectorized equal-weight Gaussian-mixture CRPS over arrays of positions.
    locs[k], scales[k], y are 1-d arrays of the same length."""
    n = len(locs)
    w = 1.0 / n
    e_xy = np.zeros_like(y, dtype=float)
    for k in range(n):
        e_xy += w * _exp_abs_norm_vec(y - locs[k], scales[k])
    e_xx = np.zeros_like(y, dtype=float)
    for i in range(n):
        for j in range(n):
            s_ij = np.sqrt(scales[i] ** 2 + scales[j] ** 2)
            e_xx += w * w * _exp_abs_norm_vec(locs[i] - locs[j], s_ij)
    return e_xy - 0.5 * e_xx

ALPHA = {"A": 0, "C": 1, "G": 2, "U": 3}
SEEDS = [0, 1, 2, 3, 4]
LR_GRID = [3e-4, 1e-3]
WD_GRID = [0.0, 1e-4]
LIKELIHOODS = ["gaussian", "student_t"]
MAX_EPOCHS = 200
PATIENCE = 20
SCALE_FLOOR = 1e-3
STUDENT_DF = 5.0  # fixed df>2 (finite variance); inner-selected vs Gaussian


class ScaleHead(nn.Module):
    """Per-position positive predictive scale from the encoded WT context.
    Positive parameterization: softplus + train-only floor. Its input H depends
    on the trainable encoder, so the scale is trained end-to-end."""

    def __init__(self, d: int = 96, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return SCALE_FLOOR + torch.nn.functional.softplus(self.mlp(H).squeeze(-1))


class LRSOv3(nn.Module):
    """Spec-compliant LRSO: TRAINABLE WT encoder + operator + scale head."""

    def __init__(self, k_rank: int, d: int = 96, heads: int = 4, hidden: int = 64,
                 likelihood: str = "gaussian") -> None:
        super().__init__()
        self.k_rank = k_rank
        self.likelihood = likelihood
        self.encoder = RFDLRSO(k_rank=max(k_rank, 1), d=d, heads=heads).encoder
        self.wt_obs_proj = nn.Linear(1, d)
        self.ctx_norm = nn.LayerNorm(d)
        self.src = nn.Sequential(nn.Linear(d + 8, hidden), nn.ReLU(), nn.Linear(hidden, max(k_rank, 1)))
        self.recv = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, max(k_rank, 1)))
        self.gmod = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, max(k_rank, 1)))
        self.bdirect = nn.Sequential(nn.Linear(d + d + 1 + 8, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
        self.scale_head = ScaleHead(d)

    def encode(self, ctx) -> torch.Tensor:
        """Trainable WT context encoding (no detach). ctx is the tensor tuple
        returned by _wt_ctx_tensors; returns (L, d)."""
        seq, react, prec, obs_token, pos, region = ctx
        x = (self.encoder.seq_emb(seq[None]) + self.encoder.react_emb(react[None].unsqueeze(-1))
             + self.encoder.prec_emb(prec[None].unsqueeze(-1))
             + self.encoder.pos_emb(pos[None].unsqueeze(-1))
             + self.encoder.region_emb(region[None])
             + self.wt_obs_proj(obs_token[None].unsqueeze(-1)))
        mask = obs_token.bool()[None]
        for blk in self.encoder.blocks:
            x = blk(x, mask)
        return x[0]

    def forward_op(self, H, edit_idx, dists, refs, alts, masks):
        """H: (L,d); returns delta (B,L) and scale (L)."""
        device = H.device
        B = edit_idx.shape[0]
        L = H.shape[0]
        Hn = self.ctx_norm(H)
        hp = Hn[edit_idx]
        ref_idx = torch.tensor([ALPHA.get(x, 3) for x in refs], device=device)
        alt_idx = torch.tensor([ALPHA.get(x, 3) for x in alts], device=device)
        ra = torch.zeros(B, 8, device=device)
        ra.scatter_(1, ref_idx[:, None], 1.0)
        ra.scatter_(1, alt_idx[:, None] + 4, 1.0)
        src = self.src(torch.cat([hp, ra], dim=-1))            # (B, k)
        recv = self.recv(Hn)                                    # (L, k)
        g = self.gmod(dists.unsqueeze(-1))                      # (B, L, k)
        hp_e = hp.unsqueeze(1).expand(B, L, -1)
        H_e = Hn.unsqueeze(0).expand(B, -1, -1)
        ra_e = ra.unsqueeze(1).expand(B, L, -1)
        bd = self.bdirect(torch.cat([hp_e, H_e, dists.unsqueeze(-1), ra_e], dim=-1)).squeeze(-1)
        if self.k_rank == 0:
            lrso = torch.zeros_like(bd)
        else:
            lrso = (src.unsqueeze(1) * recv.unsqueeze(0) * g).sum(-1)
        delta = bd + lrso
        delta = delta.masked_fill(~masks, 0.0)
        # ref==alt => mean forced 0 (vectorized, no Python loop)
        same = torch.tensor([r == a for r, a in zip(refs, alts)], device=device, dtype=torch.bool)
        delta[same] = 0.0
        scale = self.scale_head(Hn)  # (L,) positive
        return delta, scale


def _wt_ctx_tensors(univ, construct_id: str, device: str):
    """WT context tensors. Missing WT reactivity mean-filled (NOT 0); a
    WT-observed binary token is appended; attention mask excludes missing WT."""
    c = univ.get_construct(construct_id)
    L = len(c.wt_reactivity)
    seq = np.zeros((L, 4), dtype=np.float32)
    for i, base in enumerate(c.sequence):
        seq[i, ALPHA.get(base, 3)] = 1.0
    obs = c.wt_observed.astype(bool)
    react = c.wt_reactivity.astype(np.float32)
    fill = float(np.nanmean(react[obs])) if obs.any() else 0.0
    react = np.where(obs, react, fill).astype(np.float32)
    err = c.wt_error.astype(np.float32)
    prec = np.where(np.isfinite(err) & (err > 0) & obs,
                    -np.log(np.maximum(err, 1e-6)), 0.0).astype(np.float32)
    obs_token = obs.astype(np.float32)
    pos = np.arange(L, dtype=np.float32)
    region = np.stack([(c.region_map == "design_region").astype(np.float32),
                       (c.region_map == "other_assay_region").astype(np.float32)], axis=-1)
    return (torch.tensor(seq, device=device), torch.tensor(react, device=device),
            torch.tensor(prec, device=device), torch.tensor(obs_token, device=device),
            torch.tensor(pos, device=device), torch.tensor(region, device=device))


def _target_matrix(univ, recs):
    """Raw (B, L) mutant target reactivity (NaN preserved) + (B, L) WT-obs bool."""
    first = univ.get_construct(recs[0].construct_id)
    L = len(first.wt_reactivity)
    wt_obs = np.tile(first.wt_observed.astype(bool), (len(recs), 1))
    tmat = np.full((len(recs), L), np.nan, dtype=np.float32)
    for bi, r in enumerate(recs):
        tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        if tprof is not None:
            tmat[bi] = tprof.astype(np.float32)
    return tmat, wt_obs


def _qualified_mask(tmat: np.ndarray, wt_obs: np.ndarray) -> np.ndarray:
    """Target-qualified mask = mutant target observed AND WT observed. Computed
    from the RAW array BEFORE any 0-fill, so missing never equals 0."""
    return (~np.isnan(tmat)) & wt_obs


def _wt_filled(univ, cid) -> np.ndarray:
    c = univ.get_construct(cid)
    obs = c.wt_observed.astype(bool)
    fill = float(np.nanmean(c.wt_reactivity[obs])) if obs.any() else 0.0
    return np.where(obs, c.wt_reactivity.astype(np.float32), fill)


def _maybe_compile(model):
    """Optionally wrap model with torch.compile for engineering speedup."""
    if _COMPILE_FLAG:
        try:
            return torch.compile(model)
        except Exception:
            return model
    return model


_COMPILE_FLAG = False


def _nll_macro_tensors(model, H, edit_idx, dists, refs, alts, tmat_t, masks_t, wt_filled_t):
    """Per-mutant macro masked NLL over PRE-CONVERTED tensors (training path;
    avoids numpy->tensor copies every epoch). Missing targets (mask False) are
    excluded; never fitted to 0.

    IMPORTANT: NaN at masked positions is removed from the autograd graph via
    torch.where BEFORE the arithmetic. torch.masked_fill backward leaves
    grad_input = 0 at masked positions, but `0 * NaN(local grad)` = NaN, so a
    NaN kept in the graph poisons ALL gradients. Replacing masked y with a
    finite value keeps every tensor NaN-free."""
    delta, scale = model.forward_op(H, edit_idx, dists, refs, alts, masks_t)
    pred = wt_filled_t[None, :] + delta
    y = torch.where(masks_t, tmat_t, torch.zeros_like(tmat_t))  # NaN removed
    if model.likelihood == "gaussian":
        sigma = scale.clamp(min=SCALE_FLOOR).expand(pred.shape[0], -1)
        nll_pos = 0.5 * ((y - pred) / sigma) ** 2 + torch.log(sigma) + 0.5 * np.log(2.0 * np.pi)
    else:
        df = torch.tensor(STUDENT_DF, device=H.device)
        sigma = scale.clamp(min=SCALE_FLOOR).expand(pred.shape[0], -1)
        z = (y - pred) / sigma
        nll_pos = (torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2)
                   - 0.5 * torch.log(df * torch.pi) - torch.log(sigma)
                   - ((df + 1) / 2) * torch.log1p(z * z / df))
        nll_pos = -nll_pos
    nll_pos = nll_pos.masked_fill(~masks_t, 0.0)
    denom = masks_t.float().sum(-1).clamp(min=1.0)
    return torch.mean(nll_pos.sum(-1) / denom)


def _nll_macro(model, H, edit_idx, dists, refs, alts, tmat, wt_obs, wt_filled):
    """Public masked NLL entry (converts numpy -> tensors). Kept for tests."""
    device = H.device
    masks = torch.tensor(_qualified_mask(tmat, wt_obs), device=device)
    tmat_t = torch.tensor(tmat, device=device)
    wt_filled_t = torch.tensor(wt_filled, device=device)
    return _nll_macro_tensors(model, H, edit_idx, dists, refs, alts,
                              tmat_t, masks, wt_filled_t)


def _make_train_batches(univ, records, ctx_cache, device):
    """Pre-compute per-construct cached tensors ONCE (reused every epoch)."""
    by = {}
    for r in records:
        by.setdefault(r.construct_id, []).append(r)
    batches = []
    for cid, recs in by.items():
        tmat, wt_obs = _target_matrix(univ, recs)
        if (~np.isnan(tmat) & wt_obs).sum() == 0:
            continue  # TARGET_UNOBSERVABLE for training (still predicted at inference)
        L = tmat.shape[1]
        edit_idx = torch.tensor([r.pos for r in recs], device=device)
        dists = (torch.arange(L, device=device)[None, :] - edit_idx[:, None]).float()
        masks_t = torch.tensor(_qualified_mask(tmat, wt_obs), device=device)
        tmat_t = torch.tensor(tmat, device=device)
        wt_filled_t = torch.tensor(_wt_filled(univ, cid), device=device)
        batches.append((cid, edit_idx, dists, [r.ref for r in recs],
                        [r.alt for r in recs], tmat_t, masks_t, wt_filled_t))
    return batches


def _fit_epochs(model, univ, train_records, ctx_cache, device, cfg, epochs):
    """Train the model for exactly `epochs` epochs over outer-train records
    (per-construct batches; equal construct weight; per-mutant macro NLL)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    batches = _make_train_batches(univ, train_records, ctx_cache, device)
    for _ in range(epochs):
        for (cid, edit_idx, dists, refs, alts, tmat_t, masks_t, wt_filled_t) in batches:
            H = model.encode(ctx_cache[cid])
            loss = _nll_macro_tensors(model, H, edit_idx, dists, refs, alts,
                                      tmat_t, masks_t, wt_filled_t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def _crps_constructs(model, univ, by_construct, ctx_cache, device):
    """Mean held/inner CRPS over records, scored on target-qualified AND
    WT-observed positions at the model-predicted per-position scale (vectorized)."""
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for cid, recs in by_construct.items():
            if not recs:
                continue
            tmat, wt_obs = _target_matrix(univ, recs)
            obs = wt_obs[0]
            edit_idx = torch.tensor([r.pos for r in recs], device=device)
            dists = (torch.arange(tmat.shape[1], device=device)[None, :] - edit_idx[:, None]).float()
            refs = [r.ref for r in recs]; alts = [r.alt for r in recs]
            H = model.encode(ctx_cache[cid])
            masks = torch.tensor(_qualified_mask(tmat, wt_obs), device=device)
            delta, scale = model.forward_op(H, edit_idx, dists, refs, alts, masks)
            pred = torch.tensor(_wt_filled(univ, cid), device=device)[None, :] + delta
            pred_np = pred.cpu().numpy(); scale_np = scale.cpu().numpy()
            for bi in range(len(recs)):
                tprof = tmat[bi]
                q = np.where(~np.isnan(tprof) & obs)[0]
                if q.size == 0:
                    continue
                total += float(np.mean(_crps_gaussian_vec(pred_np[bi][q], scale_np[q], tprof[q])))
                n += 1
    return total / n if n else float("nan")


def _select_inner_cfg(univ, train_records, inner_groups, ctx_cache, device, k_rank,
                      grid, max_epochs, patience, seed=0):
    """Inner 4-fold puzzle-grouped selection of {lr,wd,likelihood} + early-stopped
    epoch count by inner OOF CRPS. Returns (best_cfg, best_epoch, best_score,
    cfg_scores)."""
    records_by_puzzle: dict[str, list] = {}
    for r in train_records:
        records_by_puzzle.setdefault(r.puzzle, []).append(r)
    best_cfg = None; best_score = float("inf"); best_epoch = 0
    cfg_scores = []
    for cfg in grid:
        ep_hist = []; scores = []
        for val_group in inner_groups:
            val_puzzles = set(val_group)
            val_records = [r for p in val_puzzles for r in records_by_puzzle.get(p, [])]
            tr_records = [r for r in train_records if r.puzzle not in val_puzzles]
            torch.manual_seed(seed)  # seed BEFORE construction => reproducible init
            model = _maybe_compile(LRSOv3(k_rank=k_rank, likelihood=cfg["likelihood"]).to(device))
            best_ep, best_val = _early_stop_fit(model, univ, tr_records, val_records,
                                                ctx_cache, device, cfg, max_epochs, patience,
                                                seed=seed)
            ep_hist.append(best_ep)
            scores.append(best_val)
        finite = [s for s in scores if np.isfinite(s)]
        mean_score = float(np.mean(finite)) if finite else float("nan")
        cfg_scores.append({"cfg": cfg, "inner_crps": mean_score,
                           "per_inner_fold_crps": scores,
                           "best_epochs": ep_hist})
        if np.isfinite(mean_score) and mean_score < best_score:
            best_score = mean_score; best_cfg = cfg
            best_epoch = int(np.mean(ep_hist)) if ep_hist else 0
    return best_cfg, best_epoch, best_score, cfg_scores


def _epoch_select_fixed_cfg(univ, train_records, inner_groups, ctx_cache, device,
                            k_rank, cfg, max_epochs, patience, seed=0):
    """Per-fold inner 4-fold puzzle-grouped early-stopped epoch selection for a
    FIXED hyperparameter cfg. Returns (mean_best_epoch, mean_best_val_crps)."""
    records_by_puzzle: dict[str, list] = {}
    for r in train_records:
        records_by_puzzle.setdefault(r.puzzle, []).append(r)
    ep_hist = []; score_hist = []
    for val_group in inner_groups:
        val_puzzles = set(val_group)
        val_records = [r for p in val_puzzles for r in records_by_puzzle.get(p, [])]
        tr_records = [r for r in train_records if r.puzzle not in val_puzzles]
        torch.manual_seed(seed)
        model = _maybe_compile(LRSOv3(k_rank=k_rank, likelihood=cfg["likelihood"]).to(device))
        best_ep, best_val = _early_stop_fit(model, univ, tr_records, val_records,
                                            ctx_cache, device, cfg, max_epochs, patience,
                                            seed=seed)
        ep_hist.append(best_ep); score_hist.append(best_val)
    finite = [s for s in score_hist if np.isfinite(s)]
    mean_val = float(np.mean(finite)) if finite else float("nan")
    mean_ep = int(np.mean(ep_hist)) if ep_hist else 0
    return mean_ep, mean_val


def _select_hp_once(univ, all_records, puzzles, device, ref_rank, grid,
                    hp_max_epochs, hp_patience, seed=0):
    """One-time development-level hyperparameter selection (contract 10.2
    frozen search space; selected by puzzle-grouped inner validation, exactly
    as §8.1 selects B*/Direct* once before folding). Uses a modest ranking
    epoch budget for the grid scan; the FINAL model per fold is trained with
    per-fold inner early stopping at the full budget. Returns best cfg."""
    inner = _grouped_folds(puzzles, 4, seed=seed)
    records_by_puzzle: dict[str, list] = {}
    for r in all_records:
        records_by_puzzle.setdefault(r.puzzle, []).append(r)
    best_cfg = None; best_score = float("inf")
    for cfg in grid:
        scores = []
        for val_group in inner:
            val_puzzles = set(val_group)
            val_records = [r for p in val_puzzles for r in records_by_puzzle.get(p, [])]
            tr_records = [r for r in all_records if r.puzzle not in val_puzzles]
            torch.manual_seed(seed)
            model = _maybe_compile(LRSOv3(k_rank=ref_rank, likelihood=cfg["likelihood"]).to(device))
            _ep, best_val = _early_stop_fit(model, univ, tr_records, val_records,
                                            ctx_cache_once(univ, tr_records + val_records, device),
                                            device, cfg, hp_max_epochs, hp_patience, seed=seed)
            scores.append(best_val)
        finite = [s for s in scores if np.isfinite(s)]
        mean_score = float(np.mean(finite)) if finite else float("nan")
        print(f"[hp-once] cfg={cfg} inner_crps={mean_score:.5f}", flush=True)
        if np.isfinite(mean_score) and mean_score < best_score:
            best_score = mean_score; best_cfg = cfg
    return best_cfg, best_score


def ctx_cache_once(univ, records, device):
    return {cid: _wt_ctx_tensors(univ, cid, device)
            for cid in sorted({r.construct_id for r in records})}


def _early_stop_fit(model, univ, train_records, val_records, ctx_cache, device,
                    cfg, max_epochs, patience, seed=0):
    """Train with inner-CRPS early stopping (max_epochs, patience). Returns
    (best_epoch, best_val_crps); model left at best state."""
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    batches = _make_train_batches(univ, train_records, ctx_cache, device)
    val_by = {}
    for r in val_records:
        val_by.setdefault(r.construct_id, []).append(r)
    best_val = float("inf"); best_epoch = 0; best_state = None
    for epoch in range(1, max_epochs + 1):
        for (cid, edit_idx, dists, refs, alts, tmat_t, masks_t, wt_filled_t) in batches:
            H = model.encode(ctx_cache[cid])
            loss = _nll_macro_tensors(model, H, edit_idx, dists, refs, alts,
                                      tmat_t, masks_t, wt_filled_t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_crps = _crps_constructs(model, univ, val_by, ctx_cache, device)
        if np.isfinite(val_crps) and val_crps < best_val - 1e-12:
            best_val = val_crps; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_val


def _mixture_held_crps(models, univ, held_by, ctx_cache, device):
    """Five-seed equal-weight Gaussian-mixture CRPS (contract 9.1): score the
    mixture CDF directly, NOT the mean of per-seed CRPS (vectorized)."""
    if len(models) != len(SEEDS):
        raise ValueError("deployment requires exactly 5 seeds")
    models = [m.eval() for m in models]
    total = 0.0; n = 0
    with torch.no_grad():
        for cid, recs in held_by.items():
            if not recs:
                continue
            tmat, wt_obs = _target_matrix(univ, recs)
            obs = wt_obs[0]
            edit_idx = torch.tensor([r.pos for r in recs], device=device)
            dists = (torch.arange(tmat.shape[1], device=device)[None, :] - edit_idx[:, None]).float()
            refs = [r.ref for r in recs]; alts = [r.alt for r in recs]
            ctx = ctx_cache[cid]
            masks = torch.tensor(_qualified_mask(tmat, wt_obs), device=device)
            preds = []; scales = []
            for m in models:
                H = m.encode(ctx)
                delta, scale = m.forward_op(H, edit_idx, dists, refs, alts, masks)
                pred = torch.tensor(_wt_filled(univ, cid), device=device)[None, :] + delta
                preds.append(pred.cpu().numpy()); scales.append(scale.cpu().numpy())
            for bi in range(len(recs)):
                tprof = tmat[bi]
                q = np.where(~np.isnan(tprof) & obs)[0]
                if q.size == 0:
                    continue
                locs = [p[bi][q] for p in preds]
                scs = [s[q] for s in scales]
                total += float(np.mean(_mixture_crps_vec(locs, scs, tprof[q])))
                n += 1
    return total / n if n else float("nan")


def _fit_ridge_bstar(univ, records):
    """B* = reg_direct (ridge on direct chemistry template), outer-train fit.
    Missing target positions excluded by mask (never 0-filled into loss)."""
    feats, targets = [], []
    for r in records:
        c = univ.get_construct(r.construct_id)
        tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        if tprof is None:
            continue
        obs = c.wt_observed.astype(bool)
        we = float(c.wt_reactivity[r.pos]) if obs[r.pos] else 0.0
        nz = obs & ~np.isnan(tprof)
        for i in np.where(nz)[0]:
            feats.append(_feat(we, float(c.wt_reactivity[i]), i - r.pos, r.ref, r.alt))
            targets.append(float(tprof[i]))
    X = np.array(feats); y = np.array(targets)
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    lam = 1e-1
    try:
        coef = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    if not np.all(np.isfinite(coef)):
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ coef
    scale = max(float(np.std(resid)) if resid.size else 1e-3, 1e-3)
    return coef, scale


def _feat(wt_e, wt_r, dist, ref, alt):
    r = np.zeros(4); a = np.zeros(4)
    r[ALPHA.get(ref, 3)] = 1.0; a[ALPHA.get(alt, 3)] = 1.0
    return np.concatenate([[wt_e, wt_r, dist, np.tanh(dist)], r, a]).astype(np.float32)


def _bstar_held_crps(univ, held_records, coef):
    total = 0.0; n = 0
    for r in held_records:
        c = univ.get_construct(r.construct_id)
        tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        if tprof is None:
            continue
        obs = c.wt_observed.astype(bool)
        we = float(c.wt_reactivity[r.pos]) if obs[r.pos] else 0.0
        nz = obs & ~np.isnan(tprof)
        idx = np.where(nz)[0]
        prof = np.full(len(c.wt_reactivity), np.nan)
        for i in idx:
            f = _feat(we, float(c.wt_reactivity[i]), i - r.pos, r.ref, r.alt)
            prof[i] = float(np.dot(coef, np.concatenate([[1.0], f])))
        q = np.where(~np.isnan(tprof) & ~np.isnan(prof))[0]
        if q.size == 0:
            continue
        total += float(np.nanmean([crps_gaussian(prof[i], 0.3, tprof[i]) for i in q]))
        n += 1
    return total / n if n else float("nan")


def _run_fold_lrso(univ, fold, all_records, ctx_cache, device, ranks,
                   grid, max_epochs, patience, fixed_cfg=None):
    """Spec-compliant P3 for one outer fold. If fixed_cfg is given (from the
    one-time hyperparameter selection), per-fold inner early stopping selects
    the epoch with that cfg; otherwise per-fold cfg+epoch selection runs.
    Returns per-rank held mixture CRPS."""
    held = fold.held_puzzle
    train_records = [r for r in all_records if r.puzzle in set(fold.train_puzzles)]
    held_records = [r for r in all_records if r.puzzle == held]
    held_by = {}
    for r in held_records:
        held_by.setdefault(r.construct_id, []).append(r)
    ref_rank = min(ranks)
    if fixed_cfg is not None:
        cfg = fixed_cfg
        epochs, _v = _epoch_select_fixed_cfg(univ, train_records, fold.inner_groups,
                                             ctx_cache, device, ref_rank, cfg,
                                             max_epochs, patience)
        cfg_scores = None
    else:
        cfg, epochs, _score, cfg_scores = _select_inner_cfg(
            univ, train_records, fold.inner_groups, ctx_cache, device, ref_rank, grid,
            max_epochs, patience)
    out = {}
    for k in ranks:
        use_cfg = cfg
        use_epochs = epochs
        if use_cfg is None:
            out[k] = {"held_crps": float("nan"), "cfg": None, "epochs": 0,
                      "mixture": False, "note": "no finite inner cfg"}
            continue
        models = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            m = _maybe_compile(LRSOv3(k_rank=k, likelihood=use_cfg["likelihood"]).to(device))
            _fit_epochs(m, univ, train_records, ctx_cache, device, use_cfg, max(use_epochs, 1))
            models.append(m)
        held_crps = _mixture_held_crps(models, univ, held_by, ctx_cache, device)
        out[k] = {"held_crps": held_crps, "cfg": use_cfg, "epochs": use_epochs,
                  "inner_cfg_scores": cfg_scores, "mixture": True}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spec-compliant P3 LRSO (contract 10.2/9.1/12.5)")
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank", default="2,4,8")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--hp-selection", choices=["once", "per_fold"], default="once",
                    help="hyperparameter selection scheme (default once: development-level "
                         "inner-validation, consistent with §8.1 B* selection)")
    ap.add_argument("--hp-max-epochs", type=int, default=40,
                    help="epoch budget for the one-time hyperparameter RANKING scan "
                         "(final models still use per-fold inner early stopping up to "
                         "--max-epochs)")
    ap.add_argument("--hp-patience", type=int, default=10)
    ap.add_argument("--no-inner-select", action="store_true",
                    help="skip hyperparameter/likelihood inner selection; use the "
                         "2026-08-15 HP-selected defaults {lr=1e-3, wd=0, student_t} "
                         "(frozen search space, selected by dev-level inner validation)")
    ap.add_argument("--compile", action="store_true",
                    help="wrap the model with torch.compile (pure engineering "
                         "speedup; no semantic change)")
    args = ap.parse_args(argv)
    global _COMPILE_FLAG
    _COMPILE_FLAG = args.compile
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    ranks = [int(x) for x in args.rank.split(",")]
    print("device", device, "ranks", ranks, "max_epochs", args.max_epochs,
          "patience", args.patience, "hp_selection", args.hp_selection,
          "compile", args.compile, flush=True)

    grid = [{"lr": lr, "wd": wd, "likelihood": lik}
            for lr in LR_GRID for wd in WD_GRID for lik in LIKELIHOODS]
    if args.no_inner_select:
        # HP-selected 2026-08-15: dev-level inner validation over the frozen grid
        # chose lr=1e-3, wd=0, Student-t (inner CRPS 0.18982 best of 8 configs).
        grid = [{"lr": 1e-3, "wd": 0.0, "likelihood": "student_t"}]

    univ = M2Universe(Path(args.m2_csv)); univ.build()
    puzzles = sorted(set(r.puzzle for r in univ.get_records()))
    split = build_split_v4(puzzles)
    all_records = univ.get_records()

    fixed_cfg = None
    if args.hp_selection == "once" and not args.no_inner_select:
        fixed_cfg, hp_score = _select_hp_once(univ, all_records, puzzles, device,
                                              min(ranks), grid,
                                              args.hp_max_epochs, args.hp_patience)
        print("HP_SELECTED", fixed_cfg, "inner_crps", hp_score, flush=True)

    rank_held = {k: {} for k in ranks}
    rank_dp = {k: {} for k in ranks}
    rank_cfg = {k: {} for k in ranks}
    b_held = {}

    import time
    for fold in split["folds"]:
        t0 = time.time()
        held = fold.held_puzzle
        train_records = [r for r in all_records if r.puzzle in set(fold.train_puzzles)]
        held_records = [r for r in all_records if r.puzzle == held]
        ctx_cache = {}
        for cid in sorted({r.construct_id for r in train_records + held_records}):
            ctx_cache[cid] = _wt_ctx_tensors(univ, cid, device)

        coef_b, _ = _fit_ridge_bstar(univ, train_records)
        b_held[held] = _bstar_held_crps(univ, held_records, coef_b)
        print(f"[{time.strftime('%H:%M:%S')}] fold {fold.outer_fold} ({held}) "
              f"B*_held_crps={b_held[held]:.4f}", flush=True)

        fold_out = _run_fold_lrso(univ, fold, all_records, ctx_cache, device, ranks,
                                  grid, args.max_epochs, args.patience, fixed_cfg)
        for k in ranks:
            rank_held[k][held] = fold_out[k]["held_crps"]
            rank_cfg[k][held] = fold_out[k]["cfg"]
            rank_dp[k][held] = d_p_p2(b_held[held], rank_held[k][held])
            print(f"[{time.strftime('%H:%M:%S')}] fold {fold.outer_fold} rank {k} "
                  f"L_LRSO={rank_held[k][held]:.4f} D_p^P3={rank_dp[k][held]:.4f} "
                  f"cfg={fold_out[k]['cfg']}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] fold {fold.outer_fold} done in "
              f"{time.time()-t0:.0f}s", flush=True)

    result = {"schema_version": "reactflow_delta.p3_lrso_v3.v1",
              "device": device, "ranks": ranks,
              "implementation": ("spec_compliant: trainable WT encoder; missing!=0 masked NLL; "
                                 "inner 4-fold puzzle-grouped validation + early stop (max 200); "
                                 "lr/wd/likelihood inner-selected; five-seed Gaussian mixture"),
              "hp_selection": args.hp_selection,
              "hp_selected_cfg": fixed_cfg,
              "b_star_held_crps": b_held, "rank_held_crps": rank_held,
              "rank_d_p3": rank_dp, "rank_cfg": rank_cfg,
              "seeds": SEEDS, "max_epochs": args.max_epochs, "patience": args.patience,
              "supersedes": ("p3_lrso_v2 (v1/v2 INVALID: detached encoder; missing target "
                             "0-filled then treated observable; fixed 6-12 epochs; single "
                             "seed; no inner validation)")}
    for k in ranks:
        effects = [rank_dp[k][f.held_puzzle] for f in split["folds"]]
        result[f"ci_rank_{k}"] = puzzle_level_ci20(effects)
        result[f"sign_rank_{k}"] = studentized_sign_flip(effects)
        result[f"lop_rank_{k}"] = leave_one_puzzle_influence(effects,
                                                             [f.held_puzzle for f in split["folds"]])
    result["verdict"] = {str(k): ("NO_INCREMENTAL_LRSO_SKILL" if not result[f"ci_rank_{k}"].get("ci_low_gt_0")
                                  else "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT") for k in ranks}
    (out / "p3_lrso_v3_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k in ("device", "ranks", "verdict",
                     "ci_rank_2", "ci_rank_4", "ci_rank_8")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
