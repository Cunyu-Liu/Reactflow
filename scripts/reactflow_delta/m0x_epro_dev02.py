#!/usr/bin/env python3
"""EPRO_DEV_02 (M0-X): pretraining arms — Arm A (from-scratch) vs Arm B (static).

Per contract §16, compares two controlled arms sharing the same data/split/
mask/capacity/budget/evaluator/selection/test:
  * Arm A: from-scratch EPRO-Small (reuses the EPRO_DEV_01 from-scratch run,
    identical protocol/config, run epro_dev_01_20260804).
  * Arm B: static pretraining on the frozen M2SL5 (Ribonanza pre-competition,
    CC0, RMDB) static structure/reactivity profiles, then fine-tune the full
    EPRO-Small on the frozen D2-X train split.

The static pretraining task is per-position regression of the static chemical
reactivity profile from WT-only thermo features (no Delta outcome is used).
The pretrained encoder is the only transfer; the remaining EPRO-Small operator
modules are trained from scratch on the Delta task.  Arm B must complete an
exposure-ledger audit (§16.2): the M2SL5 static sequences are disjoint from the
D2-X test/validation studies (M2SL5 is a SL5 SARS-CoV-2 / M2-seq construct set,
not in the D2-X publication split), and NO Delta outcome label is used during
pretraining.

GPU required (CUDA_VISIBLE_DEVICES=1, fallback=0).  Test split is SEALED and
NEVER read.  Same dev definitions as EPRO_DEV_01 (changer detection estimand A,
study-macro AUPRC, cluster bootstrap CI, group-aware permutation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- sys.path so pending modules are importable -----------------------------
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_HERE))  # b0x_* modules
sys.path.insert(0, str(Path.cwd() / "src"))  # reactflow.delta (script run)
if (_HERE.parents[2] / "src").exists():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))  # reactflow.delta (module load)

from b0x_data import load_pairs, split_groups  # noqa: E402
from b0x_baselines import run_baseline, _pair_scale  # noqa: E402
from b0x_evaluate import (  # noqa: E402
    cluster_ci,
    group_aware_permutation,
    per_pair_loss,
    pooled_skill,
)
from reactflow.delta.baselines import build_mutant_sequences  # noqa: E402
from reactflow.delta.model import EPROConfig, EPROModel  # noqa: E402
from reactflow.delta.rdat import parse_rdat  # noqa: E402
from reactflow.delta.thermo_state import compute_wt_thermo_state  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (identical to EPRO_DEV_01)
# ---------------------------------------------------------------------------
SEED = 20260804
TEMPERATURE = 37.0
CHANGER_TOL = 0.05
CONTACT_BPP_THRESHOLD = 0.05
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_02_20260804"
ITERATION_ID = "EPRO_DEV_02"
HYPOTHESIS_ID = "m0x_h02_pretrain_arm"

EPRO_CONFIG = dict(
    model_type="epro_lite",
    latent_dim=32,
    hidden_dim=128,
    n_encoder_layers=3,
    local_window=3,
    rho_max=0.95,
    neumann_iter=20,
    switch_enabled=False,
    dropout=0.0,
)

# Pretraining hyperparameters (Arm B encoder-only warm start).
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3
PRETRAIN_MAX_PAIRS = 2000  # cap on static M2SL5 sequences used for pretraining
PRETRAIN_SEQ_LEN = 139  # M2SL5 profile length


# ---------------------------------------------------------------------------
# Feature pipeline (identical to EPRO_DEV_01)
# ---------------------------------------------------------------------------
class PairData:
    __slots__ = ("pair_id", "parent", "study", "split", "features", "delta_thermo",
                 "edges", "edge_features", "edit_pos", "delta_true", "mask",
                 "pair_scale", "n")

    def __init__(self, pair_id, parent, study, split, features, delta_thermo,
                 edges, edge_features, edit_pos, delta_true, mask, pair_scale, n):
        self.pair_id = pair_id
        self.parent = parent
        self.study = study
        self.split = split
        self.features = features
        self.delta_thermo = delta_thermo
        self.edges = edges
        self.edge_features = edge_features
        self.edit_pos = edit_pos
        self.delta_true = delta_true
        self.mask = mask
        self.pair_scale = pair_scale
        self.n = n


def _build_features_datasets(train: list, val: list) -> tuple[list[PairData], list[PairData]]:
    """Compute WT thermo + delta_thermo + edges for all pairs, with caching."""
    wt_cache: dict[str, dict] = {}
    mut_cache: dict[tuple, tuple] = {}

    def wt_thermo(seq: str) -> dict:
        key = hashlib.sha256(seq.encode("ascii")).hexdigest()
        st = wt_cache.get(key)
        if st is None:
            st = compute_wt_thermo_state(seq, temperature=TEMPERATURE)
            wt_cache[key] = st
        return st

    def mutant_thermo(pair) -> tuple:
        key = (pair.parent, pair.mutation_pos, pair.ref_allele)
        m = mut_cache.get(key)
        if m is not None:
            return m
        wt_seq = pair.seq
        mut_seqs = build_mutant_sequences(wt_seq, pair.mutation_pos + 1, pair.ref_allele)
        n = len(wt_seq)
        n_alts = len(mut_seqs)
        su = np.zeros(n, dtype=np.float64)
        se = np.zeros(n, dtype=np.float64)
        sb = np.zeros(n, dtype=np.float64)
        smf = 0.0
        spf = 0.0
        for ms in mut_seqs:
            st = compute_wt_thermo_state(ms, temperature=TEMPERATURE)
            su += np.asarray(st["unpaired_prob"], dtype=np.float64)
            se += np.asarray(st["positional_entropy_bits"], dtype=np.float64)
            sb += (1.0 - np.asarray(st["unpaired_prob"], dtype=np.float64))
            smf += float(st["mfe_energy_kcal_mol"])
            spf += float(st["pf_energy_kcal_mol"])
        m = (su / n_alts, se / n_alts, sb / n_alts, smf / n_alts, spf / n_alts)
        mut_cache[key] = m
        return m

    def build(pair) -> PairData:
        wt = wt_thermo(pair.seq)
        n = len(pair.mask)
        seq_len = float(len(pair.seq))

        wt_f = np.zeros((n, 5), dtype=np.float32)
        wt_f[:, 0] = np.asarray(wt["unpaired_prob"], dtype=np.float32)[:n]
        wt_f[:, 1] = np.asarray(wt["positional_entropy_bits"], dtype=np.float32)[:n]
        wt_f[:, 2] = (1.0 - np.asarray(wt["unpaired_prob"], dtype=np.float32))[:n]
        seq_pos = np.arange(n, dtype=np.float32)
        wt_f[:, 3] = seq_pos / seq_len
        wt_f[:, 4] = np.abs(seq_pos - pair.mutation_pos) / seq_len

        mu, me, mb, mf, pf = mutant_thermo(pair)
        wt_mfe = float(wt["mfe_energy_kcal_mol"])
        wt_pf = float(wt["pf_energy_kcal_mol"])
        dt = np.stack([
            np.asarray(mu, dtype=np.float32)[:n] - wt_f[:, 0],
            np.asarray(me, dtype=np.float32)[:n] - wt_f[:, 1],
            np.asarray(mb, dtype=np.float32)[:n] - wt_f[:, 2],
            np.full(n, float(mf) - wt_mfe, dtype=np.float32),
            np.full(n, float(pf) - wt_pf, dtype=np.float32),
        ], axis=1)

        features = np.concatenate([wt_f, dt], axis=1).astype(np.float32)  # (n, 10)

        edges_list: list[tuple[int, int]] = []
        edge_feats: list[list[float]] = []
        for i in range(n - 1):
            edges_list.append((i, i + 1))
            edges_list.append((i + 1, i))
            edge_feats.append([0.0, 1.0, 0.0])
            edge_feats.append([0.0, 1.0, 0.0])
        bpp = wt["bpp"]
        for i in range(n):
            for j in range(n):
                if j != i and bpp[i][j] > CONTACT_BPP_THRESHOLD:
                    edges_list.append((i, j))
                    edge_feats.append([float(bpp[i][j]), float(abs(i - j)), float(bpp[i][j])])
        edges = torch.tensor(edges_list, dtype=torch.long).T
        edge_features = torch.tensor(edge_feats, dtype=torch.float32)

        delta_true = torch.tensor(pair.delta, dtype=torch.float32)
        mask = torch.tensor(pair.mask, dtype=torch.bool)
        scale = _pair_scale(pair)

        return PairData(
            pair_id=pair.pair_id, parent=pair.parent, study=pair.study, split=pair.split,
            features=torch.tensor(features, dtype=torch.float32),
            delta_thermo=torch.tensor(dt, dtype=torch.float32),
            edges=edges, edge_features=edge_features,
            edit_pos=pair.mutation_pos, delta_true=delta_true, mask=mask,
            pair_scale=scale, n=n,
        )

    train_ds = [build(p) for p in train]
    val_ds = [build(p) for p in val]
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Static pretraining (Arm B) — encoder-only warm start on M2SL5 reactivity
# ---------------------------------------------------------------------------
class StaticPretrainHead(nn.Module):
    """Per-position regression head from the EPRO encoder latent to reactivity."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)


def _build_static_pretrain_dataset(rdat_paths: list[Path], max_pairs: int) -> list[dict]:
    """Parse M2SL5 static reactivity + build WT-only features (no delta target).

    Returns a list of dicts with:
      - features: (n, 10) WT thermo features (delta columns zeroed, no edit)
      - target:   (n,) normalized static reactivity (NaN -> 0, masked)
      - mask:     (n,) finite-reactivity mask
      - seq_len:  int
    """
    out: list[dict] = []
    wt_cache: dict[str, dict] = {}

    def wt_thermo(seq: str) -> dict:
        key = hashlib.sha256(seq.encode("ascii")).hexdigest()
        st = wt_cache.get(key)
        if st is None:
            st = compute_wt_thermo_state(seq, temperature=TEMPERATURE)
            wt_cache[key] = st
        return st

    for path in rdat_paths:
        doc = parse_rdat(path)
        for prof in doc["profiles"]:
            seq = prof.get("profile_sequence")
            if not seq:
                continue
            seq = seq.upper().replace("T", "U")
            n = len(seq)
            if n == 0:
                continue
            try:
                wt = wt_thermo(seq)
            except Exception:
                continue
            seq_len = float(n)
            wt_f = np.zeros((n, 5), dtype=np.float32)
            wt_f[:, 0] = np.asarray(wt["unpaired_prob"], dtype=np.float32)[:n]
            wt_f[:, 1] = np.asarray(wt["positional_entropy_bits"], dtype=np.float32)[:n]
            wt_f[:, 2] = (1.0 - np.asarray(wt["unpaired_prob"], dtype=np.float32))[:n]
            wt_f[:, 3] = np.arange(n, dtype=np.float32) / seq_len
            wt_f[:, 4] = 0.0  # no edit during static pretraining

            features = np.concatenate([wt_f, np.zeros((n, 5), dtype=np.float32)],
                                      axis=1).astype(np.float32)  # (n,10)

            raw = np.asarray(prof["reactivity"], dtype=np.float64)
            if len(raw) != n:
                raw = raw[:n] if len(raw) >= n else np.concatenate(
                    [raw, np.full(n - len(raw), np.nan)])
            # Normalize reactivity to [0,1] via robust min-max (NaNs preserved).
            finite = raw[np.isfinite(raw)]
            if finite.size == 0:
                target = np.zeros(n, dtype=np.float32)
                mask = np.zeros(n, dtype=bool)
            else:
                lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
                span = float(hi - lo) if hi > lo else 1.0
                target = ((raw - lo) / span).astype(np.float32)
                target = np.where(np.isfinite(target), target, 0.0)
                mask = np.isfinite(raw)
            out.append({
                "features": torch.tensor(features, dtype=torch.float32),
                "target": torch.tensor(target, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "seq_len": n,
            })
            if len(out) >= max_pairs:
                return out
    return out


def _pretrain_encoder(encoder: nn.Module, static_ds: list[dict], device, epochs,
                      lr) -> dict:
    """Train the EPRO encoder to predict static reactivity (MAE, no delta)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    encoder.to(device)
    head = StaticPretrainHead(encoder.latent_dim, hidden_dim=64).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    history = {"epochs": []}
    t0 = time.time()
    for epoch in range(epochs):
        order = np.random.permutation(len(static_ds))
        encoder.train()
        head.train()
        tot = 0.0
        nb = 0
        for idx in order:
            rec = static_ds[idx]
            feats = rec["features"].to(device).unsqueeze(0)  # (1,n,10)
            target = rec["target"].to(device)
            mask = rec["mask"].to(device)
            z = encoder(feats).squeeze(0)  # (n, latent)
            pred = head(z)
            valid = mask
            if valid.sum() == 0:
                continue
            loss = (pred - target).abs()[valid].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        rec = {"epoch": epoch, "pretrain_loss": tot / max(nb, 1)}
        history["epochs"].append(rec)
        print(f"[pretrain] epoch {epoch+1}/{epochs} loss={rec['pretrain_loss']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    history["total_elapsed_s"] = time.time() - t0
    return history


# ---------------------------------------------------------------------------
# Models / training (identical to EPRO_DEV_01)
# ---------------------------------------------------------------------------
class GenericModel(nn.Module):
    def __init__(self, feat_dim=10, hidden=124, conv_layers=2, kernel=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        convs = []
        for _ in range(conv_layers):
            convs.append(nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2))
            convs.append(nn.GELU())
        self.convs = nn.Sequential(*convs)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: dict) -> dict:
        x = batch["features"]
        x = self.inp(x)
        x = x.unsqueeze(0).transpose(1, 2)
        x = self.convs(x)
        x = x.transpose(1, 2).squeeze(0)
        out = self.head(x).squeeze(-1)
        return {"delta_r_hat": out}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _make_batch(pd: PairData, device) -> dict:
    return {
        "features": pd.features.to(device),
        "delta_thermo": pd.delta_thermo.to(device),
        "edit_pos": pd.edit_pos,
        "edges": pd.edges.to(device),
        "edge_features": pd.edge_features.to(device),
        "mask": pd.mask.to(device),
    }


def _train(model, train_ds, val_ds, val_pairs, device, epochs, eval_every, patience,
           lr, weight_decay, grad_clip, wt_ref_preds, tag):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_sc = -1e18
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = {"epochs": []}

    def val_skill():
        preds = _predict(model, val_ds, device)
        sk = pooled_skill(val_pairs, preds, wt_ref_preds)
        return sk["skill_wmae"], preds

    t0 = time.time()
    for epoch in range(epochs):
        order = np.random.permutation(len(train_ds))
        model.train()
        tot = 0.0
        nb = 0
        for idx in order:
            pd = train_ds[idx]
            batch = _make_batch(pd, device)
            opt.zero_grad()
            out = model(batch)
            mu = out["delta_r_hat"]
            target = pd.delta_true.to(device)
            target = torch.where(torch.isnan(target), torch.zeros_like(target), target)
            valid = pd.mask.to(device) & ~torch.isnan(pd.delta_true.to(device))
            if valid.sum() == 0:
                continue
            loss = (mu - target).abs()[valid].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss.item())
            nb += 1
        epoch_loss = tot / max(nb, 1)

        rec = {"epoch": epoch, "train_loss": epoch_loss}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            sc, preds = val_skill()
            rec["val_skill_wmae"] = sc
            if sc > best_sc:
                best_sc = sc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            elapsed = time.time() - t0
            print(f"[{tag}] epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} "
                  f"val_skill={sc:.4f} best={best_sc:.4f}@{best_epoch} ({elapsed:.0f}s)", flush=True)
            if no_improve >= patience:
                print(f"[{tag}] early stop at epoch {epoch} (best {best_sc:.4f}@{best_epoch})", flush=True)
                break
        history["epochs"].append(rec)

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_skill_wmae"] = best_sc
    history["best_epoch"] = best_epoch
    history["total_elapsed_s"] = time.time() - t0
    return model, history


def _predict(model, ds, device) -> dict[str, np.ndarray]:
    model.eval()
    preds = {}
    with torch.no_grad():
        for pd in ds:
            batch = _make_batch(pd, device)
            out = model(batch)
            preds[pd.pair_id] = out["delta_r_hat"].detach().cpu().numpy().astype(np.float32)
    return preds


# ---------------------------------------------------------------------------
# Dev metrics (estimand A), identical to EPRO_DEV_01
# ---------------------------------------------------------------------------
def _eligible_arrays(pd: PairData, pred: np.ndarray):
    t = np.array([float(pd.delta_true[i]) for i in range(pd.n) if pd.mask[i]], dtype=np.float64)
    pr = np.array([float(pred[i]) for i in range(pd.n) if pd.mask[i]], dtype=np.float64)
    if len(pr) != len(t):
        pr = pr[: len(t)] if len(pr) >= len(t) else np.concatenate([pr, np.zeros(len(t) - len(pr))])
    return t, pr


def _changer_labels(ds, preds):
    out = []
    for pd in ds:
        t, pr = _eligible_arrays(pd, preds[pd.pair_id])
        label = (np.abs(t) > CHANGER_TOL * pd.pair_scale).astype(np.float64)
        score = np.abs(pr).astype(np.float64)
        out.append({"study": pd.study, "parent": pd.parent, "label": label, "score": score})
    return out


def _average_precision(y_true, score):
    y_true = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, 1.0)
    npos = y.sum()
    if npos == 0:
        return 0.0
    rec = tp / npos
    ap = np.sum((rec - np.concatenate([[0.0], rec[:-1]])) * prec)
    return float(ap)


def _study_macro_auprc(changed_records):
    by_study = defaultdict(list)
    for r in changed_records:
        by_study[r["study"]].append(r)
    scores = []
    for study, recs in by_study.items():
        y = np.concatenate([r["label"] for r in recs])
        s = np.concatenate([r["score"] for r in recs])
        scores.append(_average_precision(y, s))
    return float(np.mean(scores)) if scores else float("nan")


def _fit_logistic(score, y):
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = float(score.mean())
    sd = float(score.std()) + 1e-6
    x = (score - mu) / sd
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    lr = 0.1
    n = X.shape[0]
    for _ in range(3000):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        w -= lr * (X.T @ (p - y)) / n
    return w, (mu, sd)


def _apply_logistic(w, stats, score):
    score = np.asarray(score, dtype=np.float64)
    x = (score - stats[0]) / stats[1]
    return 1.0 / (1.0 + np.exp(-(w[0] + w[1] * x)))


def _brier_logloss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    brier = float(np.mean((y - p) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return brier, ll


def _auprc_gain_bootstrap(ds_a, changed_a, ds_b, changed_b, n_boot=1000, seed=SEED):
    def clusters(changed):
        cl = defaultdict(list)
        for i, r in enumerate(changed):
            cl[(r["study"], r["parent"])].append(i)
        return list(cl.items())

    cl_a = clusters(changed_a)
    cl_b = clusters(changed_b)
    assert len(cl_a) == len(cl_b)
    rng = random.Random(seed)
    real = _study_macro_auprc(changed_a) - _study_macro_auprc(changed_b)
    diffs = []
    for _ in range(n_boot):
        if len(cl_a) == 0:
            break
        sel = [rng.choice(cl_a) for _ in range(len(cl_a))]
        sub_a = [changed_a[i] for _, idxs in sel for i in idxs]
        sub_b = [changed_b[i] for _, idxs in sel for i in idxs]
        diffs.append(_study_macro_auprc(sub_a) - _study_macro_auprc(sub_b))
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"point": real, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": n_boot}
    return {
        "point": real,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_boot": n_boot,
    }


def _dev_metrics(ds, preds, train_ds, train_preds):
    changed = _changer_labels(ds, preds)
    changed_train = _changer_labels(train_ds, train_preds)
    auprc = _study_macro_auprc(changed)

    ts = np.concatenate([r["score"] for r in changed_train])
    ty = np.concatenate([r["label"] for r in changed_train])
    w, stats = _fit_logistic(ts, ty)
    vs = np.concatenate([r["score"] for r in changed])
    vy = np.concatenate([r["label"] for r in changed])
    prob = _apply_logistic(w, stats, vs)
    brier, ll = _brier_logloss(vy, prob)
    return {"study_macro_auprc": auprc, "brier": brier, "log_loss": ll,
            "n_changer": int(vy.sum()), "n_positions": int(len(vy))}


# ---------------------------------------------------------------------------
# Exposure-ledger audit for Arm B (§16.2)
# ---------------------------------------------------------------------------
def _exposure_audit(rdat_paths: list[Path], val_pairs: list, train_pairs: list) -> dict:
    """Check that M2SL5 static sequences are disjoint from D2-X train/val/test."""
    static_seqs = set()
    for path in rdat_paths:
        doc = parse_rdat(path)
        for prof in doc["profiles"]:
            seq = prof.get("profile_sequence")
            if seq:
                static_seqs.add(seq.upper().replace("T", "U"))
    # Collect all Delta sequences (train + val, and test is sealed so use only
    # what is available; pretraining uses only M2SL5 which is outside D2-X).
    delta_seqs = set()
    for p in list(train_pairs) + list(val_pairs):
        if p.seq:
            delta_seqs.add(p.seq.upper().replace("T", "U"))
    overlap = static_seqs & delta_seqs
    return {
        "static_sequence_count": len(static_seqs),
        "delta_sequence_count": len(delta_seqs),
        "exact_overlap_count": len(overlap),
        "disjoint": len(overlap) == 0,
        "delta_outcome_used_during_pretraining": False,
        "pretraining_target": "static chemical reactivity (no Delta outcome)",
        "source": "M2SL5 (Ribonanza pre-competition, CC0, RMDB)",
        "test_sealed": True,
        "note": "overlap is exact-sequence identity; leakage-resistant D2-X split is publication-level",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--rdat-2a3", type=Path, required=True)
    ap.add_argument("--rdat-dms", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    ap.add_argument("--pretrain-max-pairs", type=int, default=PRETRAIN_MAX_PAIRS)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CUDA guard (fallback=0) ---
    if args.device == "cuda":
        if not torch.cuda.is_available():
            evidence = {"error": "CUDA unavailable; GPU required (fallback=0)",
                        "cuda_available": torch.cuda.is_available()}
            (out_dir / "gpu_failure_evidence.json").write_text(
                json.dumps(evidence, indent=2), encoding="utf-8")
            print("FATAL: CUDA unavailable. GPU required (fallback=0).", file=sys.stderr)
            return 2
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}", flush=True)
    device = torch.device(args.device)

    # --- Data ---
    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"train", "validation"})
    groups = split_groups(pairs)
    train = groups.get("train", [])
    val = groups.get("validation", [])
    if args.tiny > 0:
        train = train[: args.tiny]
    print(f"[data] train={len(train)} validation={len(val)}", flush=True)

    # --- Baseline reference (P2 paired + wt_only), re-run fresh ---
    print("[baseline] running wt_only + p2_paired on train+val...", flush=True)
    wt_res = run_baseline("wt_only", train, val, device=args.device)
    p2_val = run_baseline("p2_paired", train, val, device=args.device,
                          hidden=64, epochs=20, lr=1e-3, seed=0)
    p2_train = run_baseline("p2_paired", train, train, device=args.device,
                            hidden=64, epochs=20, lr=1e-3, seed=0)
    wt_ref_preds = wt_res.predictions
    p2_val_preds = p2_val.predictions
    p2_train_preds = p2_train.predictions
    print(f"[baseline] p2 status={p2_val.status} params={p2_val.param_count}", flush=True)

    # --- Build features ---
    print("[features] building thermo/delta_thermo/edges...", flush=True)
    t0 = time.time()
    train_ds, val_ds = _build_features_datasets(train, val)
    print(f"[features] done in {time.time()-t0:.0f}s (train={len(train_ds)} val={len(val_ds)})", flush=True)

    # --- Arm B: static pretraining on M2SL5 ---
    print("[pretrain] building static M2SL5 dataset...", flush=True)
    rdat_paths = [args.rdat_2a3, args.rdat_dms]
    static_ds = _build_static_pretrain_dataset(rdat_paths, args.pretrain_max_pairs)
    print(f"[pretrain] static sequences={len(static_ds)}", flush=True)
    exposure = _exposure_audit(rdat_paths, val, train)
    print(f"[exposure] disjoint={exposure['disjoint']} "
          f"static={exposure['static_sequence_count']} "
          f"delta={exposure['delta_sequence_count']} overlap={exposure['exact_overlap_count']}",
          flush=True)
    if not exposure["disjoint"]:
        raise RuntimeError("pretraining overlap with Delta data detected; aborting")

    # --- Two arms: Arm A (from-scratch) and Arm B (pretrained encoder) ---
    epro_cfg = EPROConfig(**EPRO_CONFIG)
    epro_params = None
    pretrain_hist = None
    for tag, warm_start in (("arm_a_fromscratch", False), ("arm_b_pretrained", True)):
        model = EPROModel(epro_cfg)
        if epro_params is None:
            epro_params = model.param_count()
            print(f"[model] EPRO-Small params={epro_params:,}", flush=True)
            if not (50000 <= epro_params <= 250000):
                raise RuntimeError(f"EPRO-Small params {epro_params} outside [50k,250k]")
        if warm_start:
            # Pretrain the encoder ON the fresh model in-place, then capture its
            # weights and rebuild a full model with the pretrained encoder.
            print(f"[pretrain] training encoder for {tag}...", flush=True)
            pretrain_hist = _pretrain_encoder(
                model.encoder, static_ds, device, args.pretrain_epochs, args.lr)
            pretrained_encoder_state = {
                k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
            model = EPROModel(epro_cfg)
            model.encoder.load_state_dict(pretrained_encoder_state)
        print(f"[train] {tag}", flush=True)
        model, hist = _train(model, train_ds, val_ds, val, device, args.epochs,
                             args.eval_every, args.patience, args.lr,
                             args.weight_decay, args.grad_clip, wt_ref_preds, tag)
        val_preds = _predict(model, val_ds, device)
        train_preds = _predict(model, train_ds, device)
        if tag == "arm_a_fromscratch":
            a_hist, a_val_preds, a_train_preds = hist, val_preds, train_preds
        else:
            b_hist, b_val_preds, b_train_preds = hist, val_preds, train_preds

    # --- Frozen evaluator metrics (vs wt_only ref) ---
    def frozen_metrics(preds):
        sk = pooled_skill(val, preds, wt_ref_preds)
        ci = cluster_ci(val, preds, wt_ref_preds, n_boot=args.n_boot, seed=SEED)
        perm = group_aware_permutation(val, "x", preds, wt_ref_preds,
                                       n_perm=args.n_perm, seed=SEED)
        losses = [per_pair_loss(p, preds[p.pair_id]) for p in val
                  if p.pair_id in preds]
        return {
            "skill_wmae": sk["skill_wmae"],
            "skill_mae": sk["skill_mae"],
            "wmae_mean": float(np.mean([l["wmae"] for l in losses])) if losses else None,
            "mae_mean": float(np.mean([l["mae"] for l in losses])) if losses else None,
            "cluster_ci_vs_wt": ci,
            "permutation_vs_wt": perm,
        }

    a_frozen = frozen_metrics(a_val_preds)
    b_frozen = frozen_metrics(b_val_preds)
    p2_frozen = frozen_metrics(p2_val_preds)
    wt_frozen = frozen_metrics(wt_ref_preds)

    # cluster CI + permutation vs P2 baseline
    a_ci_vs_p2 = cluster_ci(val, a_val_preds, p2_val_preds, n_boot=args.n_boot, seed=SEED)
    b_ci_vs_p2 = cluster_ci(val, b_val_preds, p2_val_preds, n_boot=args.n_boot, seed=SEED)
    b_perm_vs_p2 = group_aware_permutation(val, "armB", b_val_preds,
                                           p2_val_preds, n_perm=args.n_perm, seed=SEED)

    # Dev metrics (AUPRC/Brier/log-loss)
    a_dev = _dev_metrics(val_ds, a_val_preds, train_ds, a_train_preds)
    b_dev = _dev_metrics(val_ds, b_val_preds, train_ds, b_train_preds)
    p2_dev = _dev_metrics(val_ds, p2_val_preds, train_ds, p2_train_preds)
    wt_dev = _dev_metrics(val_ds, wt_ref_preds, train_ds,
                          {p.pair_id: np.zeros(len(p.mask), dtype=np.float32) for p in train})

    # AUPRC gain + cluster CI (Arm B vs Arm A; Arm B vs P2)
    a_changed = _changer_labels(val_ds, a_val_preds)
    b_changed = _changer_labels(val_ds, b_val_preds)
    p2_changed = _changer_labels(val_ds, p2_val_preds)
    b_auprc_gain = _auprc_gain_bootstrap(val_ds, b_changed, val_ds, p2_changed,
                                         n_boot=args.n_boot, seed=SEED)
    b_vs_a_gain = _auprc_gain_bootstrap(val_ds, b_changed, val_ds, a_changed,
                                        n_boot=args.n_boot, seed=SEED)

    table = {
        "arm_a_fromscratch": {
            "param_count": epro_params,
            "study_macro_auprc": a_dev["study_macro_auprc"],
            "brier": a_dev["brier"],
            "log_loss": a_dev["log_loss"],
            "cluster_ci_low_vs_p2": a_ci_vs_p2["ci_low"],
            "cluster_ci_low_vs_wt": a_frozen["cluster_ci_vs_wt"]["ci_low"],
            "skill_wmae": a_frozen["skill_wmae"],
        },
        "arm_b_pretrained": {
            "param_count": epro_params,
            "study_macro_auprc": b_dev["study_macro_auprc"],
            "brier": b_dev["brier"],
            "log_loss": b_dev["log_loss"],
            "cluster_ci_low_vs_p2": b_ci_vs_p2["ci_low"],
            "cluster_ci_low_vs_wt": b_frozen["cluster_ci_vs_wt"]["ci_low"],
            "skill_wmae": b_frozen["skill_wmae"],
            "auprc_gain_vs_p2": b_auprc_gain,
            "auprc_gain_vs_arm_a": b_vs_a_gain,
        },
        "p2_paired_baseline": {
            "param_count": p2_val.param_count,
            "study_macro_auprc": p2_dev["study_macro_auprc"],
            "brier": p2_dev["brier"],
            "log_loss": p2_dev["log_loss"],
            "cluster_ci_low_vs_wt": p2_frozen["cluster_ci_vs_wt"]["ci_low"],
            "skill_wmae": p2_frozen["skill_wmae"],
        },
        "wt_only": {
            "param_count": 31,
            "study_macro_auprc": wt_dev["study_macro_auprc"],
            "brier": wt_dev["brier"],
            "log_loss": wt_dev["log_loss"],
            "skill_wmae": wt_frozen["skill_wmae"],
        },
    }

    # --- PASS determination ---
    auprc_gain_ci_low = b_auprc_gain["ci_low"]
    pass_ci_positive = (not math.isnan(auprc_gain_ci_low)) and auprc_gain_ci_low > 0
    pass_beats_arm_a = (b_dev["study_macro_auprc"] > a_dev["study_macro_auprc"])
    pass_ok = pass_ci_positive and pass_beats_arm_a

    manifest = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "iteration_id": ITERATION_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "warning": "NOT_FINAL",
        "data": {
            "train_pairs": len(train),
            "validation_pairs": len(val),
            "test_sealed": True,
            "test_accessed": False,
        },
        "models": {
            "epro_small": {"config": EPRO_CONFIG, "param_count": epro_params},
        },
        "pretraining": {
            "arm_b_source": "M2SL5 (Ribonanza pre-competition, CC0, RMDB)",
            "static_sequences": len(static_ds),
            "epochs": args.pretrain_epochs,
            "lr": args.lr,
            "exposure": exposure,
        },
        "training": {
            "epochs_budget": args.epochs,
            "eval_every": args.eval_every,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "optimizer": "Adam",
            "loss": "MAE on eligible positions",
            "selection_metric": "validation pooled skill_wmae vs wt_only",
            "device": args.device,
            "gpu_name": gpu_name if args.device == "cuda" else None,
            "arm_a_history": a_hist,
            "arm_b_history": b_hist,
            "arm_b_pretrain_history": pretrain_hist,
        },
        "dev_definitions": {
            "changer_tol": CHANGER_TOL,
            "changer_definition": "|delta_true| > CHANGER_TOL * pair_scale(90th pct |WT|)",
            "score": "|delta_r_hat|",
            "calibration": "logistic fit on train per model",
            "study_macro_auprc": "mean over studies of per-study AP over eligible positions",
            "auprc_gain_ci": "cluster bootstrap (study->parent), seed " + str(SEED),
        },
        "evaluation": {
            "arm_a_fromscratch": a_frozen,
            "arm_b_pretrained": b_frozen,
            "p2_paired_baseline": p2_frozen,
            "wt_only": wt_frozen,
            "arm_b_vs_p2": {
                "cluster_ci": b_ci_vs_p2,
                "permutation": b_perm_vs_p2,
                "auprc_gain": b_auprc_gain,
            },
            "arm_b_vs_arm_a": {
                "auprc_gain": b_vs_a_gain,
            },
        },
        "comparison_table": table,
        "pass": {
            "pass": pass_ok,
            "arm_b_auprc_gain_ci_low_vs_p2": auprc_gain_ci_low,
            "pass_auprc_gain_ci_positive": pass_ci_positive,
            "arm_b_auprc": b_dev["study_macro_auprc"],
            "arm_a_auprc": a_dev["study_macro_auprc"],
            "pass_beats_arm_a": pass_beats_arm_a,
            "note": "pass = Arm B AUPRC-gain cluster CI low > 0 vs P2 AND Arm B study-macro AUPRC > Arm A",
        },
    }

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str),
                             encoding="utf-8")

    np.savez_compressed(str(out_dir / "predictions.npz"),
                        arm_a=dict(a_val_preds), arm_b=dict(b_val_preds),
                        p2=dict(p2_val_preds), wt=dict(wt_ref_preds))

    print("\n=== COMPARISON TABLE ===", flush=True)
    for name, row in table.items():
        print(f"{name:22s} params={row['param_count']:>7,} auprc={row['study_macro_auprc']:.4f} "
              f"brier={row['brier']:.4f} logloss={row['log_loss']:.4f} "
              f"ci_low_vs_p2={row.get('cluster_ci_low_vs_p2')} skill={row['skill_wmae']:.4f}", flush=True)
    print(f"\nPASS = {pass_ok} (auprc_gain_ci_low={auprc_gain_ci_low:.4f}, "
          f"beats_arm_a={pass_beats_arm_a})", flush=True)
    print(f"exposure: disjoint={exposure['disjoint']}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())