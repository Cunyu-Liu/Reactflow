#!/usr/bin/env python3
"""EPRO_DEV_05 (M0-X, epoch-13 scope continuation): EPRO-structured changer classifier.

Objective (per user directive 2026-08-05):
  * Use the PARTNER EPRO structure (mechanistic encoder -> forcing ->
    susceptibility -> response) as a structure-aware backbone, NOT a flat MLP.
  * Larger capacity, trained for MANY epochs with a proper LR schedule, tracking
    convergence and generalization gap (train vs val AUPRC).
  * Fix the calibration degradation of DEV_03/04: single-method probability
    calibration (isotonic PAVA / Platt / temperature) fit on train, reported
    with Brier / log-loss / ECE on val.  The old redundant cascade
    (Platt -> temperature -> PAVA) is dropped (see m0x_calibration.py).
  * Compare against PUBLISHED-class baselines on the same changer-detection
    task, not just our internal p2_paired / wt_only:
       - vienna_physics:  ViennaRNA (Turner rules) in-silico mutagenesis,
         score = |Delta(unpaired_prob)| per position (classical thermodynamic
         proxy for structure change).
       - p2_paired:  our trained per-position pairwise baseline.
       - wt_only  :  ridge on WT-only features.
       - mlp_changer (DEV_04 reference): flat MLP changer classifier.

Design (synthesis of DEV_03's EPRO backbone + DEV_04's direct classifier):
  * EPROModel(backbone) yields per-position latent response h, WT latent z_w and
    endpoint delta.  These are concatenated with the base+delta_thermo features
    and fed to a per-position MLP head trained with focal loss on the changer
    label |delta_true| > CHANGER_TOL * pair_scale.
  * Trained end-to-end so the backbone learns structure-discriminative latents.
  * LR = warmup + cosine decay; grad clip; early stop on validation study-macro
    AUPRC (the primary, calibration-invariant dev metric).

Same frozen dev definitions (estimand A, study-macro AUPRC, cluster bootstrap CI,
group-aware permutation), same frozen seed 20260804, test SEALED, GPU required
(fallback=0).  No static pretraining in this iteration (exposure ledger clean).
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(_HERE))  # b0x_* modules, m0x_* modules
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
from reactflow.delta.model import EPROConfig, EPROModel  # noqa: E402
from m0x_epro_dev03 import _build_features_datasets, _make_batch  # noqa: E402
from m0x_calibration import fit_and_report  # noqa: E402

# --- Frozen constants (identical to DEV_01..04) ---
SEED = 20260804
TEMPERATURE = 37.0
CHANGER_TOL = 0.05
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_05_20260805"
ITERATION_ID = "EPRO_DEV_05"
HYPOTHESIS_ID = "m0x_h05_epro_changer_classifier"


def _focal_loss(logits, y, alpha_pos: float, gamma: float) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p = torch.clamp(p, 1e-7, 1 - 1e-7)
    ce_pos = -torch.log(p)
    ce_neg = -torch.log(1 - p)
    pt = torch.where(y == 1, p, 1 - p)
    alpha_t = torch.where(y == 1, torch.full_like(y, alpha_pos),
                          torch.full_like(y, 1 - alpha_pos))
    loss = alpha_t * (1 - pt) ** gamma * torch.where(y == 1, ce_pos, ce_neg)
    return loss.mean()


class EPROChangerClassifier(nn.Module):
    """EPRO backbone + per-position changer head (focal loss)."""

    def __init__(self, epro_cfg, head_hidden: int = 384, head_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = EPROModel(epro_cfg)
        latent = epro_cfg.latent_dim
        feat_dim = 10 + 5 + latent * 3  # base(10) + delta_thermo(5) + h + z_w + delta
        blocks = [nn.Linear(feat_dim, head_hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(head_layers - 1):
            blocks += [nn.Linear(head_hidden, head_hidden), nn.GELU(),
                       nn.Dropout(dropout)]
        blocks.append(nn.Linear(head_hidden, 1))
        self.head = nn.Sequential(*blocks)

    def forward(self, batch):
        out = self.backbone(batch)          # per-position latents
        h = out["h"]                        # (n, latent)
        z = out["z_w"]                      # (n, latent)
        d = out["delta"]                    # (n, latent)
        base = batch["features"]            # (n, 10)
        dt = batch["delta_thermo"]          # (n, 5)
        x = torch.cat([base, dt, h, z, d], dim=-1)  # (n, feat_dim)
        return self.head(x).squeeze(-1)     # (n,) logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Changer labels / metrics (reuse frozen definitions) ---
def _changer_from_pair(pd) -> tuple[np.ndarray, np.ndarray]:
    """Return (eligible_mask_idx_out, label) as (n,) arrays over full length."""
    n = pd.n
    label = np.zeros(n, dtype=np.float64)
    elig = np.zeros(n, dtype=bool)
    for i in range(n):
        if pd.mask[i]:
            d = float(pd.delta_true[i])
            if math.isfinite(d):
                elig[i] = True
                label[i] = 1.0 if abs(d) > CHANGER_TOL * pd.pair_scale else 0.0
    return label, elig


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


def _to_changed_records(ds, scores: dict) -> list[dict]:
    out = []
    for pd in ds:
        label, elig = _changer_from_pair(pd)
        s = np.asarray(scores[pd.pair_id], dtype=np.float64)
        m = elig
        out.append({"study": pd.study, "parent": pd.parent,
                    "label": label[m], "score": s[m]})
    return out


def _pooled_xy(changed) -> tuple[np.ndarray, np.ndarray]:
    ys = np.concatenate([r["label"] for r in changed])
    ss = np.concatenate([r["score"] for r in changed])
    return ys, ss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _train(model, train_ds, val_ds, device, epochs, lr, weight_decay, grad_clip,
           alpha_pos, gamma, warmup, eval_every, patience):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_ds)

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)

    best_auprc = -1e18
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = {"epochs": []}
    t0 = time.time()
    step = 0

    for epoch in range(epochs):
        order = np.random.permutation(len(train_ds))
        model.train()
        tot = 0.0
        nb = 0
        for idx in order:
            pd = train_ds[idx]
            batch = _make_batch(pd, device)
            opt.zero_grad()
            logits = model(batch)                       # (n,)
            label, elig = _changer_from_pair(pd)
            y = torch.tensor(label, device=device, dtype=torch.float32)
            em = torch.tensor(elig, device=device, dtype=torch.bool)
            if em.sum() == 0:
                continue
            loss = _focal_loss(logits[em], y[em], alpha_pos, gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            step += 1
            tot += float(loss.item())
            nb += 1
        epoch_loss = tot / max(nb, 1)

        rec = {"epoch": epoch, "train_loss": epoch_loss}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_scores = _predict(model, val_ds, device)
            tr_scores = _predict(model, train_ds, device)
            val_ch = _to_changed_records(val_ds, val_scores)
            tr_ch = _to_changed_records(train_ds, tr_scores)
            val_auprc = _study_macro_auprc(val_ch)
            tr_auprc = _study_macro_auprc(tr_ch)
            rec["val_study_macro_auprc"] = val_auprc
            rec["train_study_macro_auprc"] = tr_auprc
            rec["gen_gap"] = tr_auprc - val_auprc
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(f"[train] epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} "
                  f"val_auprc={val_auprc:.4f} train_auprc={tr_auprc:.4f} "
                  f"gap={rec['gen_gap']:.4f} best={best_auprc:.4f}@{best_epoch} "
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


def _predict(model, ds, device) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    with torch.no_grad():
        for pd in ds:
            batch = _make_batch(pd, device)
            logits = model(batch)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            out[pd.pair_id] = p
    return out


# ---------------------------------------------------------------------------
# Published baselines
# ---------------------------------------------------------------------------
def _vienna_physics_scores(ds) -> dict[str, np.ndarray]:
    """ViennaRNA (Turner-rules) in-silico mutagenesis changer score.

    delta_thermo[:, 0] = Delta(unpaired_prob) per position (WT vs mutant-
    averaged structural state).  |Delta(unpaired_prob)| is the classical
    thermodynamic proxy for positional structure change; serve it directly as
    the changer score (no training).
    """
    out = {}
    for pd in ds:
        dt = pd.delta_thermo.cpu().numpy() if hasattr(pd.delta_thermo, "cpu") \
            else pd.delta_thermo        # (n, 5)
        score = np.abs(dt[:, 0]).astype(np.float32)
        out[pd.pair_id] = score
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    # EPRO capacity (epoch-13 scope expansion: largest single-card-trainable).
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=768)
    ap.add_argument("--n-encoder-layers", type=int, default=4)
    ap.add_argument("--local-window", type=int, default=7)
    ap.add_argument("--rho-max", type=float, default=0.95)
    ap.add_argument("--neumann-iter", type=int, default=30)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head-hidden", type=int, default=384)
    ap.add_argument("--head-layers", type=int, default=2)
    # Training.
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha-pos", type=float, default=0.0)
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

    # Internal baselines (fresh).
    print("[baseline] running wt_only + p2_paired...", flush=True)
    wt_res = run_baseline("wt_only", train, val, device=args.device)
    p2_val = run_baseline("p2_paired", train, val, device=args.device,
                          hidden=64, epochs=20, lr=1e-3, seed=0)
    wt_ref_preds = wt_res.predictions
    p2_val_preds = p2_val.predictions
    print(f"[baseline] p2 status={p2_val.status} params={p2_val.param_count}",
          flush=True)

    # Build PairData (reuse DEV_03 builder).
    print("[features] building thermo/delta_thermo/edges...", flush=True)
    t0 = time.time()
    train_ds, val_ds = _build_features_datasets(train, val)
    print(f"[features] done in {time.time()-t0:.0f}s "
          f"(train={len(train_ds)} val={len(val_ds)})", flush=True)

    # Published-class baseline: ViennaRNA physics changer score.
    vienna_val = _vienna_physics_scores(val_ds)
    vienna_tr = _vienna_physics_scores(train_ds)

    # Compute alpha_pos from pooled changers.
    n_pos = 0
    n_pos_pos = 0
    for pd in train_ds:
        label, elig = _changer_from_pair(pd)
        n_pos += int(elig.sum())
        n_pos_pos += int(np.sum(label[elig] > 0.5))
    pos_frac = n_pos_pos / max(n_pos, 1)
    alpha_pos = args.focal_alpha_pos if args.focal_alpha_pos > 0 else max(0.05, 1 - pos_frac)

    # Build model.
    epro_cfg = EPROConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_encoder_layers=args.n_encoder_layers,
        local_window=args.local_window,
        rho_max=args.rho_max,
        neumann_iter=args.neumann_iter,
        switch_enabled=False,
        dropout=args.dropout,
    )
    model = EPROChangerClassifier(epro_cfg, head_hidden=args.head_hidden,
                                  head_layers=args.head_layers,
                                  dropout=args.dropout)
    print(f"[model] EPROChangerClassifier params={model.param_count():,} "
          f"alpha_pos={alpha_pos:.3f} changer_frac={pos_frac:.4f}", flush=True)

    # Train for many epochs with convergence tracking.
    model, hist = _train(model, train_ds, val_ds, device, args.epochs,
                         args.lr, args.weight_decay, args.grad_clip,
                         alpha_pos, args.focal_gamma, args.warmup,
                         args.eval_every, args.patience)

    val_scores = _predict(model, val_ds, device)
    tr_scores = _predict(model, train_ds, device)

    # Changer records for all models.
    cls_val = _to_changed_records(val_ds, val_scores)
    cls_tr = _to_changed_records(train_ds, tr_scores)
    p2_val_ch = _to_changed_records(val_ds, {p.pair_id: np.abs(p2_val_preds[p.pair_id])
                                              for p in val})
    wt_val_ch = _to_changed_records(val_ds, {p.pair_id: np.abs(wt_ref_preds[p.pair_id])
                                              for p in val})
    vienna_val_ch = _to_changed_records(val_ds, vienna_val)

    auprc = {
        "epro_changer": _study_macro_auprc(cls_val),
        "p2_paired_baseline": _study_macro_auprc(p2_val_ch),
        "wt_only": _study_macro_auprc(wt_val_ch),
        "vienna_physics": _study_macro_auprc(vienna_val_ch),
    }

    # Calibration (single-method, fit on train, report on val).
    tr_y, tr_s = _pooled_xy(cls_tr)
    val_y, val_s = _pooled_xy(cls_val)
    calib_report, calib_probs_val, calib_method = fit_and_report(tr_s, tr_y, val_s, val_y)
    # Calibrated AUPRC: re-score eligible positions with calibrated probs.
    cls_val_cal = []
    k = 0
    for r in cls_val:
        npos = len(r["score"])
        cls_val_cal.append({"study": r["study"], "parent": r["parent"],
                            "label": r["label"], "score": calib_probs_val[k:k + npos]})
        k += npos
    auprc["epro_changer_calibrated"] = _study_macro_auprc(cls_val_cal)

    # AUPRC gain + cluster CI vs P2 and vs published ViennaRNA.
    gain_vs_p2 = _auprc_gain_bootstrap(cls_val, p2_val_ch, n_boot=args.n_boot, seed=SEED)
    gain_vs_vienna = _auprc_gain_bootstrap(cls_val, vienna_val_ch,
                                           n_boot=args.n_boot, seed=SEED)
    gain_vs_wt = _auprc_gain_bootstrap(cls_val, wt_val_ch, n_boot=args.n_boot, seed=SEED)

    ci_low = gain_vs_p2["ci_low"]
    pass_ok = (not math.isnan(ci_low)) and ci_low > 0

    table = {
        "epro_changer_classifier": {
            "param_count": model.param_count(),
            "study_macro_auprc": auprc["epro_changer"],
            "study_macro_auprc_calibrated": auprc["epro_changer_calibrated"],
            "calibration_selected_method": calib_method,
            "calibration": {kk: {kkk: vv for kkk, vv in vv.items() if kkk != "reliability"}
                            for kk, vv in calib_report.items() if isinstance(vv, dict)},
            "auprc_gain_vs_p2": gain_vs_p2,
            "auprc_gain_vs_vienna": gain_vs_vienna,
            "auprc_gain_vs_wt": gain_vs_wt,
        },
        "p2_paired_baseline": {"param_count": p2_val.param_count,
                               "study_macro_auprc": auprc["p2_paired_baseline"]},
        "wt_only": {"param_count": wt_res.param_count,
                    "study_macro_auprc": auprc["wt_only"]},
        "vienna_physics_published": {"param_count": 0,
                                     "study_macro_auprc": auprc["vienna_physics"],
                                     "note": "ViennaRNA Turner-rules in-silico mutagenesis, |Delta(unpaired)|"},
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
                 "pooled_train_eligible_positions": n_pos,
                 "changer_frac": pos_frac},
        "model": {"epro_changer_classifier": {
            "config": {"latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
                       "n_encoder_layers": args.n_encoder_layers,
                       "local_window": args.local_window, "rho_max": args.rho_max,
                       "neumann_iter": args.neumann_iter, "dropout": args.dropout,
                       "head_hidden": args.head_hidden, "head_layers": args.head_layers,
                       "focal_gamma": args.focal_gamma, "focal_alpha_pos": alpha_pos},
            "param_count": model.param_count()}},
        "training": {"epochs_budget": args.epochs, "lr": args.lr,
                     "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
                     "warmup": args.warmup, "loss": "focal loss (changer classification)",
                     "device": args.device,
                     "gpu_name": gpu_name if args.device == "cuda" else None,
                     "history": hist},
        "dev_definitions": {
            "changer_tol": CHANGER_TOL,
            "changer_definition": "|delta_true| > CHANGER_TOL * pair_scale",
            "score": "P(changer) (sigmoid output)",
            "study_macro_auprc": "mean over studies of per-study AP over eligible positions",
            "auprc_gain_ci": "cluster bootstrap (study->parent), seed " + str(SEED),
            "calibration": "single-method fit on train, Brier/logloss/ECE on val",
        },
        "evaluation": {
            "epro_changer_classifier": {"study_macro_auprc": auprc["epro_changer"],
                                        "calibration": calib_report,
                                        "auprc_gain_vs_p2": gain_vs_p2,
                                        "auprc_gain_vs_vienna": gain_vs_vienna,
                                        "auprc_gain_vs_wt": gain_vs_wt},
            "p2_paired_baseline": {"study_macro_auprc": auprc["p2_paired_baseline"]},
            "wt_only": {"study_macro_auprc": auprc["wt_only"]},
            "vienna_physics_published": {"study_macro_auprc": auprc["vienna_physics"]},
        },
        "comparison_table": table,
        "pass": {"pass": pass_ok, "auprc_gain_ci_low_vs_p2": ci_low,
                 "note": "pass = EPRO-changer AUPRC-gain cluster CI low > 0 vs P2 baseline"},
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    np.savez_compressed(str(out_dir / "predictions.npz"),
                        epro_changer=dict(val_scores),
                        vienna=dict(vienna_val),
                        p2=dict(p2_val_preds), wt=dict(wt_ref_preds))

    print("\n=== COMPARISON TABLE ===", flush=True)
    print(f"epro_changer      params={model.param_count():,} auprc={auprc['epro_changer']:.4f} "
          f"calibrated={auprc['epro_changer_calibrated']:.4f}", flush=True)
    print(f"vienna_physics(P) params=0 auprc={auprc['vienna_physics']:.4f}", flush=True)
    print(f"p2_paired         params={p2_val.param_count:,} auprc={auprc['p2_paired_baseline']:.4f}", flush=True)
    print(f"wt_only           params={wt_res.param_count} auprc={auprc['wt_only']:.4f}", flush=True)
    print(f"\ncalibration selected={calib_method}", flush=True)
    print(f"PASS = {pass_ok} (auprc_gain_ci_low_vs_p2={ci_low:.4f})", flush=True)
    print(f"manifest: {out_dir / 'run_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())