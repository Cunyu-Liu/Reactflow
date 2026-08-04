#!/usr/bin/env python3
"""EPRO_DEV_01 (M0-X): from-scratch EPRO-Small vs capacity-matched generic.

Trains an EPRO-Small model (50k-250k params) and a capacity-matched generic
model from scratch on the frozen D2-X train split and evaluates both on the
frozen validation split using the frozen B0-X evaluator.  The test split is
SEALED and NEVER read.  GPU is required (CUDA_VISIBLE_DEVICES=1, fallback=0).

Dev definitions (documented, frozen):
  * Changer label (estimand A): position i is a "changer" if
    |delta_true[i]| > CHANGER_TOL * pair_scale, where pair_scale is the 90th
    percentile of |WT reactivity| over the pair's eligible positions (the same
    scale used by the B0-X WMAE weights).  Scale-relative because the WT
    reactivity scale differs strongly across studies (train |delta| mean 0.85
    vs validation 51.2).
  * Model score for ranking = |delta_r_hat| (predicted effect magnitude).
  * Calibration: a shared logistic P(changer) = sigmoid(a*score+b) procedure fit
    on the TRAIN split per model, then applied to validation.
  * study-macro AUPRC = mean over studies of per-study average-precision over
    eligible positions (study equal weight).
  * AUPRC gain CI = cluster bootstrap over (study, parent) clusters (seed fixed).

Primary pass criterion (per preregistration m0x_h01_frozen_fromscratch):
  study-macro AUPRC gain over P2 baseline with cluster CI lower bound > 0 AND
  study-macro AUPRC > matched generic study-macro AUPRC.
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

# --- sys.path so `b0x_*` and `reactflow.delta` are importable ---------------
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
from reactflow.delta.thermo_state import compute_wt_thermo_state  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------
SEED = 20260804
TEMPERATURE = 37.0
CHANGER_TOL = 0.05
CONTACT_BPP_THRESHOLD = 0.05
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_01_20260804"
ITERATION_ID = "EPRO_DEV_01"
HYPOTHESIS_ID = "m0x_h01_frozen_fromscratch"

# EPRO-Small config (verified: 93,613 params, in [50k, 250k]).
# neumann_iter=20 keeps the Neumann solver cheap enough for a 93k-param dev
# model on 3516 pairs (~77s/epoch on GPU 1); the parameter count is unchanged
# by the solver depth.
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

# Generic model config (verified: 93,993 params, within +-5% of EPRO-Small).
GENERIC_CONFIG = dict(
    feat_dim=10,
    hidden=124,
    conv_layers=2,
    kernel=3,
)


# ---------------------------------------------------------------------------
# Feature pipeline
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
            un = np.asarray(st["unpaired_prob"], dtype=np.float64)
            su += un
            se += np.asarray(st["positional_entropy_bits"], dtype=np.float64)
            sb += (1.0 - un)
            smf += float(st["mfe_energy_kcal_mol"])
            spf += float(st["pf_energy_kcal_mol"])
        m = (su / n_alts, se / n_alts, sb / n_alts, smf / n_alts, spf / n_alts)
        mut_cache[key] = m
        return m

    def build(pair) -> PairData:
        wt = wt_thermo(pair.seq)
        n = len(pair.mask)  # aligned reactivity-array length (may differ from len(seq))
        seq_len = float(len(pair.seq))

        # WT features (n, 5) — array index i maps to sequence position i (B0-X
        # convention); only the first n sequence positions are within the array.
        wt_f = np.zeros((n, 5), dtype=np.float32)
        wt_f[:, 0] = np.asarray(wt["unpaired_prob"], dtype=np.float32)[:n]
        wt_f[:, 1] = np.asarray(wt["positional_entropy_bits"], dtype=np.float32)[:n]
        wt_f[:, 2] = (1.0 - np.asarray(wt["unpaired_prob"], dtype=np.float32))[:n]
        seq_pos = np.arange(n, dtype=np.float32)
        wt_f[:, 3] = seq_pos / seq_len
        wt_f[:, 4] = np.abs(seq_pos - pair.mutation_pos) / seq_len

        # delta_thermo (n, 5) = mutant_mean - wt (sliced to array length)
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

        # Edges: sequence-adjacent + contact edges from BPP (first n positions).
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
        edges = torch.tensor(edges_list, dtype=torch.long).T  # (2, n_edges)
        edge_features = torch.tensor(edge_feats, dtype=torch.float32)  # (n_edges, 3)

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
# Models
# ---------------------------------------------------------------------------
class GenericModel(nn.Module):
    """Capacity-matched generic: per-position MLP + gated 1D convs over positions.

    Takes the same input features (per-position, feat_dim) and outputs the same
    per-position delta.  No operator/thermo structure; a plain sequence model.
    """

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
        x = batch["features"]  # (n, feat)
        x = self.inp(x)
        x = x.unsqueeze(0).transpose(1, 2)  # (1, hidden, n)
        x = self.convs(x)
        x = x.transpose(1, 2).squeeze(0)  # (n, hidden)
        out = self.head(x).squeeze(-1)  # (n,)
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
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
# Dev metrics (estimand A: changer detection/ranking)
# ---------------------------------------------------------------------------
def _eligible_arrays(pd: PairData, pred: np.ndarray):
    t = np.array([float(pd.delta_true[i]) for i in range(pd.n) if pd.mask[i]], dtype=np.float64)
    pr = np.array([float(pred[i]) for i in range(pd.n) if pd.mask[i]], dtype=np.float64)
    if len(pr) != len(t):
        pr = pr[: len(t)] if len(pr) >= len(t) else np.concatenate([pr, np.zeros(len(t) - len(pr))])
    return t, pr


def _changer_labels(ds, preds):
    """Return per-pair (label, score) on eligible positions (scale-relative)."""
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
    """Cluster bootstrap (study->parent) of the study-macro AUPRC difference."""
    # Build (study, parent) -> list of record indices.
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
    """Compute AUPRC/Brier/log-loss for a model on validation (calibrated on train)."""
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
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tiny", type=int, default=0,
                    help="if >0, use only this many train pairs (sanity check)")
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

    # --- Baseline reference predictions (P2 paired + wt_only), re-run fresh ---
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

    # --- Build models ---
    epro_cfg = EPROConfig(**EPRO_CONFIG)
    epro = EPROModel(epro_cfg)
    epro_params = epro.param_count()
    generic = GenericModel(**GENERIC_CONFIG)
    generic_params = generic.param_count()
    print(f"[model] EPRO-Small params={epro_params:,}  generic params={generic_params:,}", flush=True)
    if not (50000 <= epro_params <= 250000):
        raise RuntimeError(f"EPRO-Small params {epro_params} outside [50k,250k]")
    ratio = abs(epro_params - generic_params) / epro_params
    if ratio > 0.05:
        raise RuntimeError(f"generic params {generic_params} not within ±5% of EPRO {epro_params}")

    # --- Train EPRO-Small ---
    print("[train] EPRO-Small", flush=True)
    epro, hist_epro = _train(epro, train_ds, val_ds, val, device, args.epochs,
                             args.eval_every, args.patience, args.lr,
                             args.weight_decay, args.grad_clip, wt_ref_preds, "epro")
    epro_val_preds = _predict(epro, val_ds, device)
    epro_train_preds = _predict(epro, train_ds, device)

    # --- Train generic ---
    print("[train] generic", flush=True)
    generic, hist_generic = _train(generic, train_ds, val_ds, val, device, args.epochs,
                                   args.eval_every, args.patience, args.lr,
                                   args.weight_decay, args.grad_clip, wt_ref_preds, "generic")
    generic_val_preds = _predict(generic, val_ds, device)
    generic_train_preds = _predict(generic, train_ds, device)

    # --- Frozen evaluator metrics (vs wt_only ref) ---
    # NOTE: the frozen evaluator operates on original Pair objects (uses .delta,
    # .mask, .wt_reactivity), so we pass `val` (list[Pair]) with predictions keyed
    # by pair_id.
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

    epro_frozen = frozen_metrics(epro_val_preds)
    generic_frozen = frozen_metrics(generic_val_preds)
    p2_frozen = frozen_metrics(p2_val_preds)
    wt_frozen = frozen_metrics(wt_ref_preds)

    # cluster CI + permutation vs P2 baseline (strongest baseline)
    epro_ci_vs_p2 = cluster_ci(val, epro_val_preds, p2_val_preds,
                               n_boot=args.n_boot, seed=SEED)
    generic_ci_vs_p2 = cluster_ci(val, generic_val_preds, p2_val_preds,
                                  n_boot=args.n_boot, seed=SEED)
    epro_perm_vs_p2 = group_aware_permutation(val, "epro", epro_val_preds,
                                              p2_val_preds, n_perm=args.n_perm, seed=SEED)

    # --- Dev metrics (AUPRC/Brier/log-loss) ---
    epro_dev = _dev_metrics(val_ds, epro_val_preds, train_ds, epro_train_preds)
    generic_dev = _dev_metrics(val_ds, generic_val_preds, train_ds, generic_train_preds)
    p2_dev = _dev_metrics(val_ds, p2_val_preds, train_ds, p2_train_preds)
    wt_dev = _dev_metrics(val_ds, wt_ref_preds, train_ds,
                          {p.pair_id: np.zeros(len(p.mask), dtype=np.float32) for p in train})

    # AUPRC gain + cluster CI (EPRO vs P2; generic vs P2)
    epro_changed = _changer_labels(val_ds, epro_val_preds)
    p2_changed = _changer_labels(val_ds, p2_val_preds)
    generic_changed = _changer_labels(val_ds, generic_val_preds)
    epro_auprc_gain = _auprc_gain_bootstrap(val_ds, epro_changed, val_ds, p2_changed,
                                            n_boot=args.n_boot, seed=SEED)
    generic_auprc_gain = _auprc_gain_bootstrap(val_ds, generic_changed, val_ds, p2_changed,
                                               n_boot=args.n_boot, seed=SEED)

    # --- Horizontal comparison table ---
    table = {
        "epro_small": {
            "param_count": epro_params,
            "study_macro_auprc": epro_dev["study_macro_auprc"],
            "brier": epro_dev["brier"],
            "log_loss": epro_dev["log_loss"],
            "cluster_ci_low_vs_p2": epro_ci_vs_p2["ci_low"],
            "cluster_ci_low_vs_wt": epro_frozen["cluster_ci_vs_wt"]["ci_low"],
            "skill_wmae": epro_frozen["skill_wmae"],
            "auprc_gain_vs_p2": epro_auprc_gain,
        },
        "matched_generic": {
            "param_count": generic_params,
            "study_macro_auprc": generic_dev["study_macro_auprc"],
            "brier": generic_dev["brier"],
            "log_loss": generic_dev["log_loss"],
            "cluster_ci_low_vs_p2": generic_ci_vs_p2["ci_low"],
            "cluster_ci_low_vs_wt": generic_frozen["cluster_ci_vs_wt"]["ci_low"],
            "skill_wmae": generic_frozen["skill_wmae"],
            "auprc_gain_vs_p2": generic_auprc_gain,
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
    auprc_gain_ci_low = epro_auprc_gain["ci_low"]
    pass_ci_positive = (not math.isnan(auprc_gain_ci_low)) and auprc_gain_ci_low > 0
    pass_beats_generic = (epro_dev["study_macro_auprc"] > generic_dev["study_macro_auprc"])
    pass_ok = pass_ci_positive and pass_beats_generic

    manifest = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "iteration_id": ITERATION_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "warning": "NOT_FINAL",  # dev iteration; finalizer/gate not run here
        "data": {
            "train_pairs": len(train),
            "validation_pairs": len(val),
            "test_sealed": True,
            "test_accessed": False,
        },
        "models": {
            "epro_small": {"config": EPRO_CONFIG, "param_count": epro_params},
            "matched_generic": {"config": GENERIC_CONFIG, "param_count": generic_params},
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
            "epro_history": hist_epro,
            "generic_history": hist_generic,
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
            "epro_small": epro_frozen,
            "matched_generic": generic_frozen,
            "p2_paired_baseline": p2_frozen,
            "wt_only": wt_frozen,
            "epro_vs_p2": {
                "cluster_ci": epro_ci_vs_p2,
                "permutation": epro_perm_vs_p2,
                "auprc_gain": epro_auprc_gain,
            },
            "generic_vs_p2": {
                "cluster_ci": generic_ci_vs_p2,
                "auprc_gain": generic_auprc_gain,
            },
        },
        "comparison_table": table,
        "pass": {
            "pass": pass_ok,
            "epro_auprc_gain_ci_low_vs_p2": auprc_gain_ci_low,
            "pass_auprc_gain_ci_positive": pass_ci_positive,
            "epro_auprc": epro_dev["study_macro_auprc"],
            "generic_auprc": generic_dev["study_macro_auprc"],
            "pass_beats_matched_generic": pass_beats_generic,
            "note": "pass = AUPRC-gain cluster CI low > 0 vs P2 AND study-macro AUPRC > generic",
        },
    }

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str),
                             encoding="utf-8")

    # Save predictions for audit.
    np.savez_compressed(str(out_dir / "predictions.npz"),
                        epro=dict(epro_val_preds), generic=dict(generic_val_preds),
                        p2=dict(p2_val_preds), wt=dict(wt_ref_preds))

    print("\n=== COMPARISON TABLE ===", flush=True)
    for name, row in table.items():
        print(f"{name:22s} params={row['param_count']:>7,} auprc={row['study_macro_auprc']:.4f} "
              f"brier={row['brier']:.4f} logloss={row['log_loss']:.4f} "
              f"ci_low_vs_p2={row.get('cluster_ci_low_vs_p2')} skill={row['skill_wmae']:.4f}", flush=True)
    print(f"\nPASS = {pass_ok} (auprc_gain_ci_low={auprc_gain_ci_low:.4f}, "
          f"beats_generic={pass_beats_generic})", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())