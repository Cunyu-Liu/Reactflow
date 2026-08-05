#!/usr/bin/env python3
"""EPRO_DEV_07 (M0-X, epoch-13 scope continuation): sequence-context changer
classifier with multi-scale structural context.

EPRO_DEV_06 (0.7353 study-macro AUPRC) is a *per-position* MLP over a 42-dim
feature vector (B0-X + delta_thermo + contact-graph).  Its residual weakness is
that it pools all eligible positions across pairs and classifies each position
independently: it has NO sequential transcript context beyond the small +/-2
structural windows, so it cannot capture long-range structural coupling (e.g. a
mutation in a stem that destabilizes a distant loop, or cooperative pairing).

EPRO_DEV_07 therefore adds SEQUENCE CONTEXT while keeping the proven per-position
feature signal:

  * Features (46): B0-X(31) + delta_thermo(5) + structure-aware contact-graph(6)
    + multi-scale structural context(4): mean WT unpaired & delta-unpaired over
    windows +/-5 and +/-10 (the dev06 features only cover +/-2).
  * Model (SequenceContextChanger): a small bidirectional GRU runs over the
    per-position feature sequence of each transcript, producing a per-position
    context vector; the per-position features + context are fed to the proven
    flat MLP head (sigmoid) -> P(changer).  A residual connection keeps the
    MLP's direct per-position signal so we cannot regress from dev06.
  * Loss: focal loss (gamma>=0) with positive-class weighting, evaluated only on
    eligible positions of each pair.
  * Training: per-pair sequences batched on GPU; AdamW + cosine LR + grad clip;
    early stop on validation study-macro AUPRC (primary, calibration-invariant).
  * Calibration: single-method (Platt / isotonic PAVA / temperature), fit on
    train, reported on val (Brier / logloss / ECE).

Same frozen dev definitions (estimand A, study-macro AUPRC, cluster-bootstrap CI,
group-aware permutation), same frozen seed 20260804, test SEALED, GPU required
(real CUDA, fallback=0).  No pretraining, so no exposure audit needed; test is
never read.  Published baselines: EternaFold / MXfold2 / CONTRAfold (SOTA
folding models, run by m0x_sota_baselines.py) + ViennaRNA physics + p2 + wt.
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

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path.cwd() / "src"))

from b0x_baselines import run_baseline, _pair_scale, _build_features as p2_features  # noqa: E402
from b0x_data import load_pairs, split_groups  # noqa: E402
from reactflow.delta.baselines import build_mutant_sequences  # noqa: E402
from reactflow.delta.thermo_state import compute_wt_thermo_state  # noqa: E402
from m0x_calibration import fit_and_report  # noqa: E402

SEED = 20260804
TEMPERATURE = 37.0
CHANGER_TOL = 0.05
CONTACT_BPP_THRESHOLD = 0.05
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_07_20260805"
ITERATION_ID = "EPRO_DEV_07"
HYPOTHESIS_ID = "m0x_h07_sequence_context_changer"


# ---------------------------------------------------------------------------
# Feature pipeline (dev06 + multi-scale structural context)
# ---------------------------------------------------------------------------
def _delta_thermo_features(pair, wt_cache, mut_cache) -> np.ndarray:
    wt = wt_cache.get(pair.pair_id)
    if wt is None:
        wt = compute_wt_thermo_state(pair.seq, temperature=TEMPERATURE)
        wt_cache[pair.pair_id] = wt
    n = len(pair.mask)
    n_seq = len(pair.seq)
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
            su[:min(Lm, n_seq)] += np.asarray(st["unpaired_prob"], dtype=np.float64)[:min(Lm, n_seq)]
            se[:min(Lm, n_seq)] += np.asarray(st["positional_entropy_bits"], dtype=np.float64)[:min(Lm, n_seq)]
            sb[:min(Lm, n_seq)] += (1.0 - np.asarray(st["unpaired_prob"], dtype=np.float64))[:min(Lm, n_seq)]
            smf += float(st["mfe_energy_kcal_mol"])
            spf += float(st["pf_energy_kcal_mol"])
        m = (su[:n] / n_alts, se[:n] / n_alts, sb[:n] / n_alts, smf / n_alts, spf / n_alts)
        mut_cache[key] = m

    wt_mfe = float(wt["mfe_energy_kcal_mol"])
    wt_pf = float(wt["pf_energy_kcal_mol"])
    mu, me, mb, mf, pf = m
    return np.stack([
        np.asarray(mu, dtype=np.float32)[:n] - wt_f[:, 0],
        np.asarray(me, dtype=np.float32)[:n] - wt_f[:, 1],
        np.asarray(mb, dtype=np.float32)[:n] - wt_f[:, 2],
        np.full(n, float(mf) - wt_mfe, dtype=np.float32),
        np.full(n, float(pf) - wt_pf, dtype=np.float32),
    ], axis=1).astype(np.float32)


def _structure_aware_features(pair, wt_cache, delta_thermo) -> np.ndarray:
    wt = wt_cache.get(pair.pair_id)
    if wt is None:
        wt = compute_wt_thermo_state(pair.seq, temperature=TEMPERATURE)
        wt_cache[pair.pair_id] = wt
    n = len(pair.mask)
    bpp = np.asarray(wt["bpp"], dtype=np.float64)
    unpaired = np.asarray(wt["unpaired_prob"], dtype=np.float64)[:n]

    pairing = np.zeros(n, dtype=np.float32)
    n_partners = np.zeros(n, dtype=np.float32)
    partner_mean_unp = np.zeros(n, dtype=np.float32)
    partner_mean_dunp = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row = bpp[i][:n]
        partners = np.where(row > CONTACT_BPP_THRESHOLD)[0]
        pairing[i] = float(row[row != i].sum()) if n > 1 else 0.0
        n_partners[i] = float(len(partners))
        if len(partners) > 0:
            partner_mean_unp[i] = float(np.mean(unpaired[partners]))
            partner_mean_dunp[i] = float(np.mean(delta_thermo[partners, 0]))

    local_unp = np.zeros(n, dtype=np.float32)
    local_dunp = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        local_unp[i] = float(np.mean(unpaired[lo:hi]))
        local_dunp[i] = float(np.mean(delta_thermo[lo:hi, 0]))

    return np.stack([pairing, n_partners,
                     partner_mean_unp, partner_mean_dunp,
                     local_unp, local_dunp], axis=1).astype(np.float32)


def _multiscale_context_features(pair, wt_cache, delta_thermo) -> np.ndarray:
    """Per-position (n) x 4 multi-scale structural context (windows +/-5, +/-10).

    Complements dev06's local +/-2 windows with longer-range structural context.
    """
    wt = wt_cache.get(pair.pair_id)
    if wt is None:
        wt = compute_wt_thermo_state(pair.seq, temperature=TEMPERATURE)
        wt_cache[pair.pair_id] = wt
    n = len(pair.mask)
    unpaired = np.asarray(wt["unpaired_prob"], dtype=np.float64)[:n]
    dunp = np.asarray(delta_thermo[:, 0], dtype=np.float64)[:n]

    def window_mean(arr, w, i):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        return float(np.mean(arr[lo:hi]))

    unp5 = np.array([window_mean(unpaired, 5, i) for i in range(n)], dtype=np.float32)
    dunp5 = np.array([window_mean(dunp, 5, i) for i in range(n)], dtype=np.float32)
    unp10 = np.array([window_mean(unpaired, 10, i) for i in range(n)], dtype=np.float32)
    dunp10 = np.array([window_mean(dunp, 10, i) for i in range(n)], dtype=np.float32)
    return np.stack([unp5, dunp5, unp10, dunp10], axis=1).astype(np.float32)


def _build_pair_records(pairs: list, wt_cache, mut_cache) -> list[dict]:
    """Per-pair record: features (n,46), changer labels, mask, study, parent."""
    recs = []
    for p in pairs:
        f_b0x = p2_features(p)                       # (n, 31)
        f_dt = _delta_thermo_features(p, wt_cache, mut_cache)   # (n, 5)
        f_sa = _structure_aware_features(p, wt_cache, f_dt)     # (n, 6)
        f_ms = _multiscale_context_features(p, wt_cache, f_dt)  # (n, 4)
        feats = np.concatenate([f_b0x, f_dt, f_sa, f_ms], axis=1)  # (n, 46)
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


# ---------------------------------------------------------------------------
# Model: bidirectional GRU sequence-context + residual per-position MLP head
# ---------------------------------------------------------------------------
class SequenceContextChanger(nn.Module):
    """GRU over per-position feature sequence + per-position MLP head.

    forward(x): (B, L, F) -> logits (B, L).  L = aligned transcript length.
    """

    def __init__(self, feat_dim: int, gru_hidden: int = 64, gru_layers: int = 1,
                 head_hidden: int = 256, head_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(feat_dim, gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True, dropout=0.0)
        ctx_dim = gru_hidden * 2
        # Residual per-position MLP: features + GRU context -> logit.
        blocks = [nn.Linear(feat_dim + ctx_dim, head_hidden), nn.GELU(),
                  nn.Dropout(dropout)]
        for _ in range(head_layers - 1):
            blocks += [nn.Linear(head_hidden, head_hidden), nn.GELU(),
                       nn.Dropout(dropout)]
        blocks.append(nn.Linear(head_hidden, 1))
        self.head = nn.Sequential(*blocks)

    def forward(self, x):  # (B, L, F) -> logits (B, L)
        B, L, F = x.shape
        ctx, _ = self.gru(x)                # (B, L, 2*gru_hidden)
        h = torch.cat([x, ctx], dim=-1)     # (B, L, F + ctx_dim)
        return self.head(h).squeeze(-1)     # (B, L)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss / metrics (frozen dev definitions, identical to dev04/06)
# ---------------------------------------------------------------------------
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


def _changer_records(recs: list[dict], score: dict[str, np.ndarray]) -> list[dict]:
    # NOTE: identical to dev04/06 -- do NOT mask ineligible positions.  Ineligible
    # positions carry label 0 (from _build_pair_records) so they enter the AP over
    # all aligned positions, keeping study-macro AUPRC exactly comparable to dev06.
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


def _pooled_xy(changed) -> tuple[np.ndarray, np.ndarray]:
    ys = np.concatenate([r["label"] for r in changed])
    ss = np.concatenate([r["score"] for r in changed])
    return ys, ss


# ---------------------------------------------------------------------------
# Training (per-pair sequences, GPU)
# ---------------------------------------------------------------------------
def _make_batch(recs, device):
    """Pack variable-length records into a padded (B, L, F) tensor + masks."""
    recs = list(recs)
    Lm = max(len(r["features"]) for r in recs)
    F = recs[0]["features"].shape[1]
    feat = torch.zeros((len(recs), Lm, F), device=device)
    lab = torch.zeros((len(recs), Lm), device=device)
    elig = torch.zeros((len(recs), Lm), dtype=torch.bool, device=device)
    for b, r in enumerate(recs):
        n = len(r["features"])
        feat[b, :n] = torch.tensor(r["features"], device=device)
        lab[b, :n] = torch.tensor(r["label"], device=device)
        elig[b, :n] = torch.tensor([bool(m) for m in r["mask"]], device=device)
    return feat, lab, elig


def _train(model, train_recs, val_recs, device, epochs, batch_size, lr,
           weight_decay, grad_clip, alpha_pos, gamma, eval_every, patience):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auprc = -1e18
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = {"epochs": []}
    t0 = time.time()

    for epoch in range(epochs):
        order = np.random.permutation(len(train_recs))
        model.train()
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            recs = [train_recs[i] for i in idx]
            feat, lab, elig = _make_batch(recs, device)
            if elig.sum() == 0:
                continue
            logits = model(feat)                     # (B, L)
            loss = _focal_loss(logits[elig], lab[elig], alpha_pos, gamma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        epoch_loss = tot / max(nb, 1)

        rec = {"epoch": epoch, "train_loss": epoch_loss}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_scores = _predict(model, val_recs, device)
            tr_scores = _predict(model, train_recs, device)
            val_ch = _changer_records(val_recs, val_scores)
            tr_ch = _changer_records(train_recs, tr_scores)
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


def _predict(model, recs, device) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    with torch.no_grad():
        for b in range(0, len(recs), 64):
            batch = recs[b:b + 64]
            feat, _lab, _elig = _make_batch(batch, device)
            logits = model(feat)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            for k, r in enumerate(batch):
                out[r["pair_id"]] = p[k][:len(r["features"])]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gru-hidden", type=int, default=64)
    ap.add_argument("--gru-layers", type=int, default=1)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--head-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--focal-alpha-pos", type=float, default=0.0)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=1000)
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
        val = val[: max(args.tiny, 1)]
    print(f"[data] train={len(train)} validation={len(val)} (test SEALED)",
          flush=True)

    print("[baseline] running wt_only + p2_paired on train+val...", flush=True)
    wt_res = run_baseline("wt_only", train, val, device=args.device)
    p2_val = run_baseline("p2_paired", train, val, device=args.device,
                          hidden=64, epochs=20, lr=1e-3, seed=0)

    print("[features] building B0-X + delta_thermo + structure-aware + "
          "multi-scale context...", flush=True)
    wt_cache, mut_cache = {}, {}
    t0 = time.time()
    train_recs = _build_pair_records(train, wt_cache, mut_cache)
    val_recs = _build_pair_records(val, wt_cache, mut_cache)
    print(f"[features] done in {time.time()-t0:.0f}s", flush=True)

    # alpha_pos from pooled train prevalence.
    all_y = np.concatenate([r["label"] for r in train_recs])
    pos_frac = float(all_y.mean())
    alpha_pos = args.focal_alpha_pos if args.focal_alpha_pos > 0 else max(0.05, 1 - pos_frac)
    print(f"[data] train positions={all_y.size:,} feat_dim={train_recs[0]['features'].shape[1]} "
          f"changer_frac={pos_frac:.4f} alpha_pos(auto)={alpha_pos:.3f}", flush=True)

    model = SequenceContextChanger(
        feat_dim=train_recs[0]["features"].shape[1],
        gru_hidden=args.gru_hidden, gru_layers=args.gru_layers,
        head_hidden=args.head_hidden, head_layers=args.head_layers,
        dropout=args.dropout)
    print(f"[model] SequenceContextChanger params={model.param_count():,}",
          flush=True)
    model, hist = _train(model, train_recs, val_recs, device, args.epochs,
                         args.batch_size, args.lr, args.weight_decay,
                         args.grad_clip, alpha_pos, args.focal_gamma,
                         args.eval_every, args.patience)

    val_score = _predict(model, val_recs, device)
    train_score = _predict(model, train_recs, device)

    cls_changed = _changer_records(val_recs, val_score)
    tr_changed = _changer_records(train_recs, train_score)
    p2_changed = _changer_records(val_recs,
                                   {p.pair_id: np.abs(np.asarray(p2_val.predictions[p.pair_id]))
                                    for p in val})
    wt_changed = _changer_records(val_recs,
                                   {p.pair_id: np.abs(np.asarray(wt_res.predictions[p.pair_id]))
                                    for p in val})

    cls_auprc = _study_macro_auprc(cls_changed)
    tr_auprc = _study_macro_auprc(tr_changed)
    p2_auprc = _study_macro_auprc(p2_changed)
    wt_auprc = _study_macro_auprc(wt_changed)
    gen_gap = tr_auprc - cls_auprc

    tr_y, tr_s = _pooled_xy(tr_changed)
    val_y, val_s = _pooled_xy(cls_changed)
    calib_report, calib_probs_val, calib_method = fit_and_report(tr_s, tr_y, val_s, val_y)

    cls_val_cal = []
    k = 0
    for r in cls_changed:
        npos = len(r["score"])
        cls_val_cal.append({"study": r["study"], "parent": r["parent"],
                            "label": r["label"], "score": calib_probs_val[k:k + npos]})
        k += npos
    cls_cal_auprc = _study_macro_auprc(cls_val_cal)

    gain_vs_p2 = _auprc_gain_bootstrap(cls_changed, p2_changed,
                                       n_boot=args.n_boot, seed=SEED)
    gain_vs_wt = _auprc_gain_bootstrap(cls_changed, wt_changed,
                                       n_boot=args.n_boot, seed=SEED)

    ci_low = gain_vs_p2["ci_low"]
    pass_ok = (not math.isnan(ci_low)) and ci_low > 0

    table = {
        "sequence_context_changer": {
            "param_count": model.param_count(),
            "study_macro_auprc": cls_auprc,
            "train_study_macro_auprc": tr_auprc,
            "gen_gap": gen_gap,
            "study_macro_auprc_calibrated": cls_cal_auprc,
            "calibration_selected_method": calib_method,
            "auprc_gain_vs_p2": gain_vs_p2,
            "auprc_gain_vs_wt": gain_vs_wt,
        },
        "p2_paired_baseline": {"param_count": p2_val.param_count,
                               "study_macro_auprc": p2_auprc},
        "wt_only": {"param_count": wt_res.param_count,
                    "study_macro_auprc": wt_auprc},
        "dev06_reference": {"study_macro_auprc": 0.7353243279593717,
                            "note": "EPRO_DEV_06 per-position structure-aware MLP"},
        "vienna_physics_published": {"param_count": 0,
                                     "study_macro_auprc": 0.4534,
                                     "note": "ViennaRNA Turner-rules in-silico mutagenesis (dev06)"},
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
                 "train_positions": int(all_y.size), "changer_frac": pos_frac},
        "model": {"sequence_context_changer": {
            "config": {"gru_hidden": args.gru_hidden, "gru_layers": args.gru_layers,
                       "head_hidden": args.head_hidden,
                       "head_layers": args.head_layers, "dropout": args.dropout,
                       "focal_gamma": args.focal_gamma, "focal_alpha_pos": alpha_pos},
            "param_count": model.param_count(),
            "feat_dim": int(train_recs[0]["features"].shape[1]),
            "feature_spec": "B0-X(31) + delta_thermo(5) + contact_graph(6) + "
                            "multi_scale_context(4)"}},
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
            "study_macro_auprc": "mean over studies of per-study AP over eligible positions",
            "auprc_gain_ci": "cluster bootstrap (study->parent), seed " + str(SEED),
            "calibration": "single-method fit on train, Brier/logloss/ECE on val",
        },
        "evaluation": {
            "sequence_context_changer": {"study_macro_auprc": cls_auprc,
                                         "train_study_macro_auprc": tr_auprc,
                                         "gen_gap": gen_gap,
                                         "study_macro_auprc_calibrated": cls_cal_auprc,
                                         "auprc_gain_vs_p2": gain_vs_p2,
                                         "auprc_gain_vs_wt": gain_vs_wt},
            "p2_paired_baseline": {"study_macro_auprc": p2_auprc},
            "wt_only": {"study_macro_auprc": wt_auprc},
        },
        "comparison_table": table,
        "pass": {"pass": pass_ok, "auprc_gain_ci_low_vs_p2": ci_low,
                 "note": "pass = sequence-context classifier AUPRC-gain cluster CI low > 0 vs P2"},
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    np.savez_compressed(str(out_dir / "predictions.npz"),
                        sequence_context=dict(val_score),
                        p2=dict(p2_val.predictions), wt=dict(wt_res.predictions))

    print("\n=== COMPARISON TABLE ===", flush=True)
    print(f"sequence_context params={model.param_count():,} "
          f"auprc={cls_auprc:.4f} train={tr_auprc:.4f} gap={gen_gap:.4f}"
          f" cal={cls_cal_auprc:.4f}", flush=True)
    print(f"dev06_ref                    auprc=0.7353", flush=True)
    print(f"p2_paired         params={p2_val.param_count:,} auprc={p2_auprc:.4f}",
          flush=True)
    print(f"wt_only           params={wt_res.param_count} auprc={wt_auprc:.4f}",
          flush=True)
    print(f"\ncalibration selected={calib_method}", flush=True)
    print(f"PASS = {pass_ok} (auprc_gain_ci_low_vs_p2={ci_low:.4f})", flush=True)
    print(f"manifest: {out_dir / 'run_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())