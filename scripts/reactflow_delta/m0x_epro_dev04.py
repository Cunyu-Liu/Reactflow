#!/usr/bin/env python3
"""EPRO_DEV_04 (M0-X, epoch-13 scope continuation): direct changer classifier.

Diagnosis of EPRO_DEV_01/02/03: the EPRO delta-regression GNN collapses to
near-constant delta predictions (loss flat at ~0.832, |delta| score ~constant),
so changer-detection study-macro AUPRC (~0.42) is far below the P2 paired
baseline (0.6936) and the wt_only ridge baseline (0.6748).  Both baselines train
per-position models on the B0-X feature vector (WT reactivity, distance to
mutation, local 5-base context one-hot, ref/alt one-hot, is-mutation) and do
NOT collapse.

EPRO_DEV_04 therefore abandons delta regression and trains a DIRECT per-position
changer classifier:
  * Target:  changer := |delta_true| > CHANGER_TOL * pair_scale  (binary).
  * Features: B0-X per-position features (31) + delta_thermo (5) = 36.
  * Model:   a moderate-capacity MLP (single-card trainable) with a sigmoid
             head -> P(changer) per position.
  * Loss:    focal loss (gamma>=0) with positive-class weighting, which is
             robust to the extreme chang/non-changer imbalance and directly
             ranks changers (the AUPRC objective).
  * Score:   P(changer).  AUPRC is calibration-invariant, so the score is used
             directly for study-macro AUPRC; multi-layer calibration
             (Platt -> temperature -> isotonic) is reported as a secondary
             calibration metric with a robust PAVA.

Same frozen dev definitions (estimand A, study-macro AUPRC, cluster-bootstrap
CI, group-aware permutation), same frozen seed 20260804, test SEALED, GPU
required (fallback=0).  Exposure ledger: no pretraining in this iteration, so
no exposure audit is needed; test is never read.
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

# --- sys.path so pending modules are importable ---
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_HERE))  # b0x_* modules
sys.path.insert(0, str(Path.cwd() / "src"))  # reactflow.delta (script run)
if (_HERE.parents[2] / "src").exists():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))  # reactflow.delta (module load)

from b0x_baselines import run_baseline, _pair_scale, _build_features as p2_features  # noqa: E402
from b0x_data import load_pairs, split_groups  # noqa: E402
from b0x_evaluate import (  # noqa: E402
    cluster_ci,
    group_aware_permutation,
    per_pair_loss,
    pooled_skill,
)
from reactflow.delta.baselines import build_mutant_sequences  # noqa: E402
from reactflow.delta.thermo_state import compute_wt_thermo_state  # noqa: E402

# --- Frozen constants ---
SEED = 20260804
TEMPERATURE = 37.0
CHANGER_TOL = 0.05
CONTACT_BPP_THRESHOLD = 0.05
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_04_20260805"
ITERATION_ID = "EPRO_DEV_04"
HYPOTHESIS_ID = "m0x_h04_changer_classifier"


# ---------------------------------------------------------------------------
# Feature pipeline: B0-X features + delta_thermo (mutant-WT structural deltas)
# ---------------------------------------------------------------------------
def _delta_thermo_features(pair, wt_cache, mut_cache) -> np.ndarray:
    """Per-position (n) x 5 delta-thermo features = mutant avg - WT state."""
    wt = wt_cache.get(pair.pair_id)
    if wt is None:
        wt = compute_wt_thermo_state(pair.seq, temperature=TEMPERATURE)
        wt_cache[pair.pair_id] = wt
    n = len(pair.mask)          # aligned length (mask / evaluation grid)
    n_seq = len(pair.seq)       # full sequence length (mutants may be longer)

    wt_f = np.zeros((n, 3), dtype=np.float32)
    wt_f[:, 0] = np.asarray(wt["unpaired_prob"], dtype=np.float32)[:n]
    wt_f[:, 1] = np.asarray(wt["positional_entropy_bits"], dtype=np.float32)[:n]
    wt_f[:, 2] = (1.0 - np.asarray(wt["unpaired_prob"], dtype=np.float32))[:n]

    key = (pair.parent, pair.mutation_pos, pair.ref_allele)
    m = mut_cache.get(key)
    if m is None:
        mut_seqs = build_mutant_sequences(pair.seq, pair.mutation_pos + 1, pair.ref_allele)
        n_alts = max(len(mut_seqs), 1)
        su = np.zeros(n_seq, dtype=np.float64)
        se = np.zeros(n_seq, dtype=np.float64)
        sb = np.zeros(n_seq, dtype=np.float64)
        smf = 0.0
        spf = 0.0
        for ms in mut_seqs:
            st = compute_wt_thermo_state(ms, temperature=TEMPERATURE)
            Lm = len(ms)
            su[:min(Lm, n_seq)] += np.asarray(st["unpaired_prob"],
                                              dtype=np.float64)[:min(Lm, n_seq)]
            se[:min(Lm, n_seq)] += np.asarray(st["positional_entropy_bits"],
                                              dtype=np.float64)[:min(Lm, n_seq)]
            sb[:min(Lm, n_seq)] += (1.0 - np.asarray(st["unpaired_prob"],
                                                     dtype=np.float64))[:min(Lm, n_seq)]
            smf += float(st["mfe_energy_kcal_mol"])
            spf += float(st["pf_energy_kcal_mol"])
        m = (su[:n] / n_alts, se[:n] / n_alts, sb[:n] / n_alts, smf / n_alts, spf / n_alts)
        mut_cache[key] = m

    wt_mfe = float(wt["mfe_energy_kcal_mol"])
    wt_pf = float(wt["pf_energy_kcal_mol"])
    mu, me, mb, mf, pf = m
    dt = np.stack([
        np.asarray(mu, dtype=np.float32)[:n] - wt_f[:, 0],
        np.asarray(me, dtype=np.float32)[:n] - wt_f[:, 1],
        np.asarray(mb, dtype=np.float32)[:n] - wt_f[:, 2],
        np.full(n, float(mf) - wt_mfe, dtype=np.float32),
        np.full(n, float(pf) - wt_pf, dtype=np.float32),
    ], axis=1)
    return dt.astype(np.float32)


def _build_pair_records(pairs: list, wt_cache, mut_cache) -> list[dict]:
    """Per-pair record: features (n,36), changer labels (n,), mask, study, parent."""
    recs = []
    for p in pairs:
        f_b0x = p2_features(p)                      # (n, 31)
        f_dt = _delta_thermo_features(p, wt_cache, mut_cache)  # (n, 5)
        feats = np.concatenate([f_b0x, f_dt], axis=1)  # (n, 36)
        mask = [bool(p.mask[i]) for i in range(len(p.mask))]
        scale = _pair_scale(p)
        delta = p.delta
        label = np.array([1.0 if (mask[i] and math.isfinite(float(delta[i]))
                                  and abs(float(delta[i])) > CHANGER_TOL * scale)
                          else 0.0 for i in range(len(p.mask))], dtype=np.float64)
        recs.append({
            "pair_id": p.pair_id, "study": p.study, "parent": p.parent,
            "features": feats, "label": label, "mask": mask,
        })
    return recs


def _pool(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pool eligible-position features and labels across records."""
    Xs, ys = [], []
    for r in recs:
        for i in range(len(r["mask"])):
            if r["mask"][i]:
                Xs.append(r["features"][i])
                ys.append(r["label"][i])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float64))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ChangerClassifier(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        blocks = [nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(layers - 1):
            blocks.append(nn.Linear(hidden, hidden))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):  # (N, F) -> logits (N,)
        return self.net(x).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _focal_loss(logits, y, alpha_pos: float, gamma: float) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p = torch.clamp(p, 1e-7, 1 - 1e-7)
    ce_pos = -torch.log(p)          # y=1
    ce_neg = -torch.log(1 - p)      # y=0
    pt = torch.where(y == 1, p, 1 - p)
    alpha_t = torch.where(y == 1, torch.full_like(y, alpha_pos),
                          torch.full_like(y, 1 - alpha_pos))
    loss = alpha_t * (1 - pt) ** gamma * torch.where(y == 1, ce_pos, ce_neg)
    return loss.mean()


# ---------------------------------------------------------------------------
# Dev metrics (estimand A) + robust calibration
# ---------------------------------------------------------------------------
def _changer_records(recs: list[dict], score: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for r in recs:
        s = score[r["pair_id"]]
        label = r["label"]
        out.append({"study": r["study"], "parent": r["parent"],
                    "label": label, "score": s})
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


# --- Robust PAVA isotonic (no zero-weight division) ---
def _isotonic_pava(y, x):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    n = len(ys)
    wts = np.ones(n)
    # Standard PAVA merge loop (block index vectors).
    while True:
        moved = False
        i = 0
        while i < n - 1:
            if wts[i] > 0 and wts[i + 1] > 0 and ys[i] > ys[i + 1]:
                w = wts[i] + wts[i + 1]
                ys[i] = (wts[i] * ys[i] + wts[i + 1] * ys[i + 1]) / w
                wts[i] = w
                wts[i + 1] = 0.0
                moved = True
            i += 1
        if not moved:
            break
    # Build monotone step function on sorted xs.
    xx = xs[wts > 0]
    yy = ys[wts > 0]
    if len(xx) < 2:
        return np.full_like(x, ys[0] if n else 0.0)
    return np.interp(x, xx, np.maximum.accumulate(yy))


def _brier_logloss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    brier = float(np.mean((y - p) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return brier, ll


def _calibration(train_score, train_y, val_score, val_y):
    """Platt(logistic) -> temperature scaling -> isotonic PAVA, fitted on train."""
    ts = np.asarray(train_score, dtype=np.float64)
    ty = np.asarray(train_y, dtype=np.float64)
    vs = np.asarray(val_score, dtype=np.float64)
    vy = np.asarray(val_y, dtype=np.float64)

    # Platt via gradient on standardized score.
    mu, sd = float(ts.mean()), float(ts.std()) + 1e-6
    x = (ts - mu) / sd
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    lr = 0.1
    for _ in range(3000):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        w -= lr * (X.T @ (p - ty)) / len(ty)
    vx = (vs - mu) / sd
    p_platt = 1.0 / (1.0 + np.exp(-(w[0] + w[1] * vx)))

    # Temperature scaling on logit.
    logits = np.log(np.clip(p_platt, 1e-9, 1 - 1e-9)
                    / (1 - np.clip(p_platt, 1e-9, 1 - 1e-9)))
    T = 1.0
    for _ in range(2000):
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        grad = np.mean((p - vy) * logits / (T * T))
        T -= 1.0 * grad
        T = max(T, 1e-3)
    p_temp = 1.0 / (1.0 + np.exp(-(logits / T)))

    p_iso = _isotonic_pava(vy, p_temp)

    out = {}
    for name, p in (("platt", p_platt), ("temperature_scaled", p_temp),
                    ("isotonic_pava", p_iso)):
        b, ll = _brier_logloss(vy, p)
        out[name] = {"brier": b, "log_loss": ll}
    out["temperature_scaled"]["temperature"] = T
    return out


def _auprc_gain_bootstrap(changed_a, changed_b, n_boot=1000, seed=SEED):
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
    if cl_a:
        for _ in range(n_boot):
            sel = [rng.choice(cl_a) for _ in range(len(cl_a))]
            sub_a = [changed_a[i] for _, idxs in sel for i in idxs]
            sub_b = [changed_b[i] for _, idxs in sel for i in idxs]
            diffs.append(_study_macro_auprc(sub_a) - _study_macro_auprc(sub_b))
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"point": real, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": n_boot}
    return {"point": real, "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _train_classifier(model, X, y, val_recs, device, epochs, batch_size, lr,
                      weight_decay, alpha_pos, gamma, eval_every, patience):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device).float()
    n = Xt.shape[0]
    best_auprc = -1e18
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = {"epochs": []}
    t0 = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        tot = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logits = model(Xt[idx])
            loss = _focal_loss(logits, yt[idx], alpha_pos, gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        epoch_loss = tot / max(nb, 1)

        rec = {"epoch": epoch, "train_loss": epoch_loss}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_score = _predict(model, val_recs, device)
            changed = _changer_records(val_recs, val_score)
            auprc = _study_macro_auprc(changed)
            rec["val_study_macro_auprc"] = auprc
            if auprc > best_auprc:
                best_auprc = auprc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(f"[train] epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} "
                  f"val_auprc={auprc:.4f} best={best_auprc:.4f}@{best_epoch} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if no_improve >= patience:
                print(f"[train] early stop at epoch {epoch} "
                      f"(best {best_auprc:.4f}@{best_epoch})", flush=True)
                break
        history["epochs"].append(rec)

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_study_macro_auprc"] = best_auprc
    history["best_epoch"] = best_epoch
    history["total_elapsed_s"] = time.time() - t0
    return model, history


def _predict(model, recs, device) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    with torch.no_grad():
        for r in recs:
            x = torch.tensor(r["features"], device=device)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            out[r["pair_id"]] = p
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--focal-alpha-pos", type=float, default=0.0,
                    help="0 => auto from prevalence")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            evidence = {"error": "CUDA unavailable; GPU required (fallback=0)",
                        "cuda_available": torch.cuda.is_available()}
            (out_dir / "gpu_failure_evidence.json").write_text(
                json.dumps(evidence, indent=2), encoding="utf-8")
            print("FATAL: CUDA unavailable. GPU required (fallback=0).",
                  file=sys.stderr)
            return 2
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu_name} free={free/1e9:.1f}GB total={total/1e9:.1f}GB",
              flush=True)
    device = torch.device(args.device)

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"train", "validation"})
    groups = split_groups(pairs)
    train = groups.get("train", [])
    val = groups.get("validation", [])
    if args.tiny > 0:
        train = train[: args.tiny]
    print(f"[data] train={len(train)} validation={len(val)} (test SEALED)",
          flush=True)

    # Baselines (fresh).
    print("[baseline] running wt_only + p2_paired on train+val...", flush=True)
    wt_res = run_baseline("wt_only", train, val, device=args.device)
    p2_val = run_baseline("p2_paired", train, val, device=args.device,
                          hidden=64, epochs=20, lr=1e-3, seed=0)
    p2_train = run_baseline("p2_paired", train, train, device=args.device,
                            hidden=64, epochs=20, lr=1e-3, seed=0)
    wt_ref_preds = wt_res.predictions
    p2_val_preds = p2_val.predictions
    print(f"[baseline] p2 status={p2_val.status} params={p2_val.param_count}",
          flush=True)

    # Build records + pooled train set.
    print("[features] building B0-X + delta_thermo features...", flush=True)
    wt_cache, mut_cache = {}, {}
    t0 = time.time()
    train_recs = _build_pair_records(train, wt_cache, mut_cache)
    val_recs = _build_pair_records(val, wt_cache, mut_cache)
    print(f"[features] done in {time.time()-t0:.0f}s", flush=True)
    X, y = _pool(train_recs)
    pos_frac = float(y.mean())
    alpha_pos = args.focal_alpha_pos if args.focal_alpha_pos > 0 else max(0.05, 1 - pos_frac)
    print(f"[data] pooled train positions={X.shape[0]} changer_frac={pos_frac:.4f} "
          f"alpha_pos(auto)={alpha_pos:.3f}", flush=True)

    # Train classifier.
    model = ChangerClassifier(feat_dim=X.shape[1], hidden=args.hidden,
                              layers=args.layers, dropout=args.dropout)
    print(f"[model] ChangerClassifier params={model.param_count():,}", flush=True)
    model, hist = _train_classifier(model, X, y, val_recs, device, args.epochs,
                                    args.batch_size, args.lr, args.weight_decay,
                                    alpha_pos, args.focal_gamma, args.eval_every,
                                    args.patience)

    val_score = _predict(model, val_recs, device)
    train_score = _predict(model, train_recs, device)

    # Changer records for AUPRC.
    cls_changed = _changer_records(val_recs, val_score)
    p2_changed = _changer_records(val_recs, {p.pair_id: np.abs(p2_val_preds[p.pair_id])
                                              for p in val})
    wt_changed = _changer_records(val_recs, {p.pair_id: np.abs(wt_ref_preds[p.pair_id])
                                              for p in val})

    # Dev metrics.
    cls_auprc = _study_macro_auprc(cls_changed)
    p2_auprc = _study_macro_auprc(p2_changed)
    wt_auprc = _study_macro_auprc(wt_changed)

    # Calibration on pooled val scores vs p2/wt.
    val_pool_scores = np.concatenate([r["score"] for r in cls_changed])
    val_pool_y = np.concatenate([r["label"] for r in cls_changed])
    tr_pool_scores = np.concatenate([r["score"] for r in _changer_records(
        train_recs, train_score)])
    tr_pool_y = np.concatenate([r["label"] for r in _changer_records(
        train_recs, train_score)])
    calib = _calibration(tr_pool_scores, tr_pool_y, val_pool_scores, val_pool_y)

    # AUPRC gain + CI vs P2 and vs wt_only.
    gain_vs_p2 = _auprc_gain_bootstrap(cls_changed, p2_changed,
                                       n_boot=args.n_boot, seed=SEED)
    gain_vs_wt = _auprc_gain_bootstrap(cls_changed, wt_changed,
                                       n_boot=args.n_boot, seed=SEED)

    # Frozen evaluator metrics (delta skill) for completeness.
    def frozen_metrics(preds):
        sk = pooled_skill(val, preds, wt_ref_preds)
        ci = cluster_ci(val, preds, wt_ref_preds, n_boot=args.n_boot, seed=SEED)
        perm = group_aware_permutation(val, "x", preds, wt_ref_preds,
                                       n_perm=args.n_perm, seed=SEED)
        return {"skill_wmae": sk["skill_wmae"], "cluster_ci_vs_wt": ci,
                "permutation_vs_wt": perm}

    # PASS: classifier AUPRC-gain cluster-CI low > 0 vs P2.
    ci_low = gain_vs_p2["ci_low"]
    pass_ok = (not math.isnan(ci_low)) and ci_low > 0

    table = {
        "changer_classifier": {
            "param_count": model.param_count(),
            "study_macro_auprc": cls_auprc,
            "calibration_best_brier": calib["isotonic_pava"]["brier"],
            "calibration_best_logloss": calib["isotonic_pava"]["log_loss"],
            "auprc_gain_vs_p2": gain_vs_p2,
            "auprc_gain_vs_wt": gain_vs_wt,
        },
        "p2_paired_baseline": {
            "param_count": p2_val.param_count,
            "study_macro_auprc": p2_auprc,
        },
        "wt_only": {
            "param_count": wt_res.param_count,
            "study_macro_auprc": wt_auprc,
        },
    }

    manifest = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "iteration_id": ITERATION_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "authority_amendment": "reactflow_delta_v4_m0x_epro_scope_20260805 (epoch 13)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "warning": "NOT_FINAL",
        "data": {"train_pairs": len(train), "validation_pairs": len(val),
                 "test_sealed": True, "test_accessed": False,
                 "pooled_train_positions": int(X.shape[0]),
                 "changer_frac": pos_frac},
        "model": {"changer_classifier": {
            "config": {"hidden": args.hidden, "layers": args.layers,
                       "dropout": args.dropout, "focal_gamma": args.focal_gamma,
                       "focal_alpha_pos": alpha_pos},
            "param_count": model.param_count(), "feat_dim": int(X.shape[1])}},
        "training": {"epochs_budget": args.epochs, "batch_size": args.batch_size,
                     "lr": args.lr, "weight_decay": args.weight_decay,
                     "loss": "focal loss (direct changer classification)",
                     "device": args.device,
                     "gpu_name": gpu_name if args.device == "cuda" else None,
                     "history": hist},
        "dev_definitions": {
            "changer_tol": CHANGER_TOL,
            "changer_definition": "|delta_true| > CHANGER_TOL * pair_scale",
            "score": "P(changer) (sigmoid output)",
            "loss": "focal loss, alpha_pos=" + f"{alpha_pos:.3f}",
            "study_macro_auprc": "mean over studies of per-study AP over eligible positions",
            "auprc_gain_ci": "cluster bootstrap (study->parent), seed " + str(SEED),
        },
        "evaluation": {
            "changer_classifier": {"study_macro_auprc": cls_auprc,
                                   "calibration": calib,
                                   "auprc_gain_vs_p2": gain_vs_p2,
                                   "auprc_gain_vs_wt": gain_vs_wt},
            "p2_paired_baseline": {"study_macro_auprc": p2_auprc},
            "wt_only": {"study_macro_auprc": wt_auprc},
        },
        "comparison_table": table,
        "pass": {"pass": pass_ok, "auprc_gain_ci_low_vs_p2": ci_low,
                 "note": "pass = classifier AUPRC-gain cluster CI low > 0 vs P2 baseline"},
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    np.savez_compressed(str(out_dir / "predictions.npz"),
                        classifier=dict(val_score),
                        p2=dict(p2_val_preds), wt=dict(wt_ref_preds))

    print("\n=== COMPARISON TABLE ===", flush=True)
    print(f"changer_classifier params={model.param_count():,} "
          f"auprc={cls_auprc:.4f} brier={calib['isotonic_pava']['brier']:.4f} "
          f"logloss={calib['isotonic_pava']['log_loss']:.4f}", flush=True)
    print(f"p2_paired_baseline params={p2_val.param_count:,} auprc={p2_auprc:.4f}",
          flush=True)
    print(f"wt_only            params={wt_res.param_count} auprc={wt_auprc:.4f}",
          flush=True)
    print(f"\nPASS = {pass_ok} (auprc_gain_ci_low={ci_low:.4f})", flush=True)
    print(f"manifest: {out_dir / 'run_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())