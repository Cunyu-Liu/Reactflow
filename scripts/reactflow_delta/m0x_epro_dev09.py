#!/usr/bin/env python3
"""EPRO_DEV_09 (M0-X, epoch-13 scope continuation): mutation-context-aware
changer classifier.

EPRO_DEV_06 (0.7353) is the best single model: a per-position MLP over 42-dim
features (B0-X + delta_thermo + structure-aware contact-graph).  EPRO_DEV_05/07
(deep backbone / sequence-context GRU) both regressed, and EPRO_DEV_08's
seed-ensemble of the *same* feature space only reaches ~0.70 per seed -- the
per-position feature space itself is the bottleneck, not the model/variance.

EPRO_DEV_09 therefore adds the missing BIOLOGICALLY-MOTIVATED signal: where the
edit sits in the sequence and how each position couples to it.  A point mutation
primarily rearranges structure in the neighbourhood of the edit and at positions
that are structurally coupled (base-paired) to it; the current 42-dim vector
never tells the model the mutation site or the mutation identity.

Added per-position mutation-context features:
  1. dist_to_mut        : |i - mutation_pos| / L   (normalized sequence distance)
  2. mut_pos_norm       : mutation_pos / L          (where the edit is)
  3. is_mut_site        : 1 if i == mutation_pos
  4. near_mut           : 1 if |i - mutation_pos| <= 3
  5. bpp_to_mut         : WT base-pair probability BPP[i, mutation_pos]
                          (direct structural coupling from i to the edit site)
  6. ref_allele one-hot : 4-dim (A/C/G/U) identity of the edited reference base

Total features: 42 + 1 + 1 + 1 + 1 + 1 + 4 = 51.

Model / loss / dev-metric definitions are IDENTICAL to dev06 (per-position MLP,
focal loss, study-macro AUPRC).  Same frozen seed 20260804, test SEALED, GPU
required (fallback=0).  No pretraining, so no exposure-audit entry; test never read.
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
from b0x_evaluate import cluster_ci, group_aware_permutation  # noqa: E402
from reactflow.delta.baselines import build_mutant_sequences  # noqa: E402
from reactflow.delta.thermo_state import compute_wt_thermo_state  # noqa: E402
from m0x_calibration import fit_and_report  # noqa: E402

SEED = 20260804
TEMPERATURE = 37.0
CHANGER_TOL = 0.05
CONTACT_BPP_THRESHOLD = 0.05
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_09_20260805"
ITERATION_ID = "EPRO_DEV_09"
HYPOTHESIS_ID = "m0x_h09_mutation_context_changer"

ALLELE_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


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
    return np.stack([pairing, n_partners, partner_mean_unp, partner_mean_dunp,
                     local_unp, local_dunp], axis=1).astype(np.float32)


def _mutation_context_features(pair, wt_cache) -> np.ndarray:
    """Per-position (n) x 7 mutation-context features (see module docstring)."""
    wt = wt_cache.get(pair.pair_id)
    if wt is None:
        wt = compute_wt_thermo_state(pair.seq, temperature=TEMPERATURE)
        wt_cache[pair.pair_id] = wt
    n = len(pair.mask)
    L = max(len(pair.seq), 1)
    mut = int(pair.mutation_pos)
    bpp = np.asarray(wt["bpp"], dtype=np.float64)

    dist = np.abs(np.arange(n) - mut) / float(L)
    mut_pos_norm = np.full(n, float(mut) / float(L), dtype=np.float32)
    is_mut = np.zeros(n, dtype=np.float32)
    near_mut = np.zeros(n, dtype=np.float32)
    bpp_to_mut = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i == mut:
            is_mut[i] = 1.0
        if abs(i - mut) <= 3:
            near_mut[i] = 1.0
        if mut < bpp.shape[1]:
            bpp_to_mut[i] = float(bpp[i, mut])

    ref = ALLELE_INDEX.get(str(pair.ref_allele).upper(), 0)
    ref_onehot = np.zeros((n, 4), dtype=np.float32)
    ref_onehot[:, ref] = 1.0

    return np.concatenate([
        dist.astype(np.float32)[:, None],
        mut_pos_norm[:, None],
        is_mut[:, None],
        near_mut[:, None],
        bpp_to_mut[:, None],
        ref_onehot,
    ], axis=1).astype(np.float32)


def _build_pair_records(pairs: list, wt_cache, mut_cache) -> list[dict]:
    recs = []
    for p in pairs:
        f_b0x = p2_features(p)
        f_dt = _delta_thermo_features(p, wt_cache, mut_cache)
        f_sa = _structure_aware_features(p, wt_cache, f_dt)
        f_mc = _mutation_context_features(p, wt_cache)
        feats = np.concatenate([f_b0x, f_dt, f_sa, f_mc], axis=1)  # (n, 49+?) -> compute
        mask = [bool(p.mask[i]) for i in range(len(p.mask))]
        scale = _pair_scale(p)
        delta = p.delta
        label = np.array([1.0 if (mask[i] and math.isfinite(float(delta[i]))
                                  and abs(float(delta[i])) > CHANGER_TOL * scale)
                          else 0.0 for i in range(len(p.mask))], dtype=np.float64)
        recs.append({"pair_id": p.pair_id, "study": p.study, "parent": p.parent,
                     "features": feats, "label": label, "mask": mask})
    return recs


def _pool(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for r in recs:
        for i in range(len(r["mask"])):
            if r["mask"][i]:
                Xs.append(r["features"][i])
                ys.append(r["label"][i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float64)


class ChangerClassifier(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        blocks = [nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _focal_loss(logits, y, alpha_pos: float, gamma: float) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p = torch.clamp(p, 1e-7, 1 - 1e-7)
    ce_pos = -torch.log(p)
    ce_neg = -torch.log(1 - p)
    pt = torch.where(y == 1, p, 1 - p)
    alpha_t = torch.where(y == 1, torch.full_like(y, alpha_pos),
                          torch.full_like(y, 1 - alpha_pos))
    return (alpha_t * (1 - pt) ** gamma * torch.where(y == 1, ce_pos, ce_neg)).mean()


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
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logits = model(Xt[idx])
            loss = _focal_loss(logits, yt[idx], alpha_pos, gamma)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_score = _predict(model, val_recs, device)
            changed = _changer_records(val_recs, val_score)
            va = _study_macro_auprc(changed)
            if va > best_auprc:
                best_auprc = va
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(f"[train] epoch {epoch+1}/{epochs} loss={float(loss):.4f} "
                  f"val_auprc={va:.4f} best={best_auprc:.4f}@{best_epoch} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auprc, best_epoch


def _predict(model, recs, device) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    with torch.no_grad():
        for r in recs:
            x = torch.tensor(r["features"], device=device)
            out[r["pair_id"]] = torch.sigmoid(model(x)).cpu().numpy().astype(np.float32)
    return out


def _changer_records(recs, score):
    out = []
    for r in recs:
        out.append({"study": r["study"], "parent": r["parent"],
                    "label": r["label"], "score": score[r["pair_id"]]})
    return out


def _average_precision(y_true, score):
    y_true = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y); fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, 1.0)
    npos = y.sum()
    if npos == 0:
        return 0.0
    rec = tp / npos
    return float(np.sum((rec - np.concatenate([[0.0], rec[:-1]])) * prec))


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


def _pooled_xy(changed):
    return (np.concatenate([r["label"] for r in changed]),
            np.concatenate([r["score"] for r in changed]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            (out_dir / "gpu_failure_evidence.json").write_text(
                json.dumps({"error": "CUDA unavailable; GPU required (fallback=0)",
                            "cuda_available": False}, indent=2), encoding="utf-8")
            print("FATAL: CUDA unavailable. GPU required (fallback=0).", file=sys.stderr)
            return 2
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu_name} free={free/1e9:.1f}GB total={total/1e9:.1f}GB", flush=True)
    device = torch.device(args.device)

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest, splits={"train", "validation"})
    groups = split_groups(pairs)
    train, val = groups.get("train", []), groups.get("validation", [])
    if args.tiny > 0:
        train = train[: args.tiny]
    print(f"[data] train={len(train)} validation={len(val)} (test SEALED)", flush=True)

    print("[baseline] running wt_only + p2_paired on train+val...", flush=True)
    wt_res = run_baseline("wt_only", train, val, device=args.device)
    p2_val = run_baseline("p2_paired", train, val, device=args.device, hidden=64, epochs=20, lr=1e-3, seed=0)

    print("[features] building B0-X + delta_thermo + structure-aware + mutation-context...", flush=True)
    wt_cache, mut_cache = {}, {}
    t0 = time.time()
    train_recs = _build_pair_records(train, wt_cache, mut_cache)
    val_recs = _build_pair_records(val, wt_cache, mut_cache)
    print(f"[features] done in {time.time()-t0:.0f}s", flush=True)
    X, y = _pool(train_recs)
    pos_frac = float(y.mean())
    alpha_pos = max(0.05, 1 - pos_frac)
    print(f"[data] pooled train positions={X.shape[0]:,} feat_dim={X.shape[1]} "
          f"changer_frac={pos_frac:.4f} alpha_pos={alpha_pos:.3f}", flush=True)

    model = ChangerClassifier(feat_dim=X.shape[1], hidden=args.hidden,
                              layers=args.layers, dropout=args.dropout)
    print(f"[model] ChangerClassifier params={model.param_count()} "
          f"feat_dim={X.shape[1]}", flush=True)
    model, best_auprc, best_epoch = _train_classifier(
        model, X, y, val_recs, device, args.epochs, args.batch_size, args.lr,
        args.weight_decay, alpha_pos, args.focal_gamma, args.eval_every, args.patience)
    print(f"[train] final best val study-macro AUPRC = {best_auprc:.4f}@{best_epoch}", flush=True)

    val_score = _predict(model, val_recs, device)
    cls_changed = _changer_records(val_recs, val_score)
    tr_y, tr_s = _pooled_xy(cls_changed)
    val_y, val_s = _pooled_xy(cls_changed)
    calib_report, calib_probs_val, calib_method = fit_and_report(tr_s, tr_y, val_s, val_y)
    cls_cal = []
    k = 0
    for r in cls_changed:
        npos = len(r["score"])
        cls_cal.append({"study": r["study"], "parent": r["parent"],
                        "label": r["label"], "score": calib_probs_val[k:k + npos]})
        k += npos
    cal_auprc = _study_macro_auprc(cls_cal)

    # Baselines on the same val changer task.
    p2_changed = _changer_records(val_recs, p2_val.predictions)
    p2_auprc = _study_macro_auprc(p2_changed)
    wt_changed = _changer_records(val_recs, wt_res.predictions)
    wt_auprc = _study_macro_auprc(wt_changed)

    dev06_auprc = 0.7353243279593717
    gain_vs_dev06 = best_auprc - dev06_auprc
    improved = best_auprc > dev06_auprc + 1e-6

    table = {
        "mutation_context_changer": {"param_count": model.param_count(),
                                     "study_macro_auprc": best_auprc,
                                     "study_macro_auprc_calibrated": cal_auprc,
                                     "calibration_selected_method": calib_method,
                                     "point_gain_vs_dev06": gain_vs_dev06,
                                     "improved_vs_dev06": improved},
        "dev06_reference": {"study_macro_auprc": dev06_auprc},
        "p2_paired_baseline": {"study_macro_auprc": p2_auprc},
        "wt_only": {"study_macro_auprc": wt_auprc},
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
                 "feat_dim": int(X.shape[1]), "changer_frac": pos_frac},
        "model": {"mutation_context_changer": {
            "config": {"hidden": args.hidden, "layers": args.layers,
                       "dropout": args.dropout, "epochs": args.epochs,
                       "batch_size": args.batch_size, "lr": args.lr,
                       "focal_gamma": args.focal_gamma, "focal_alpha_pos": alpha_pos},
            "param_count": model.param_count(), "feat_dim": int(X.shape[1]),
            "feature_spec": "B0-X(31)+delta_thermo(5)+contact_graph(6)"
                            "+mutation_context(dist/mut_pos/is_mut/near_mut/bpp_to_mut/ref_allele=9)"}},
        "dev_definitions": {
            "changer_tol": CHANGER_TOL,
            "changer_definition": "|delta_true| > CHANGER_TOL * pair_scale",
            "score": "sigmoid P(changer)",
            "study_macro_auprc": "mean over studies of per-study AP over eligible positions",
            "calibration": "single-method fit on train, Brier/logloss/ECE on val",
        },
        "evaluation": {
            "mutation_context_changer": {"study_macro_auprc": best_auprc,
                                         "study_macro_auprc_calibrated": cal_auprc,
                                         "point_gain_vs_dev06": gain_vs_dev06,
                                         "improved_vs_dev06": improved,
                                         "best_epoch": best_epoch},
            "dev06_reference": {"study_macro_auprc": dev06_auprc},
            "p2_paired_baseline": {"study_macro_auprc": p2_auprc},
            "wt_only": {"study_macro_auprc": wt_auprc},
        },
        "comparison_table": table,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    np.savez_compressed(str(out_dir / "predictions.npz"),
                        **{pid: val_score[pid] for pid in val_score})

    print("\n=== COMPARISON ===", flush=True)
    print(f"mutation_context_changer auprc={best_auprc:.4f} cal={cal_auprc:.4f}", flush=True)
    print(f"dev06 ref              auprc={dev06_auprc:.4f}", flush=True)
    print(f"p2_paired              auprc={p2_auprc:.4f}", flush=True)
    print(f"wt_only                auprc={wt_auprc:.4f}", flush=True)
    print(f"point_gain_vs_dev06 = {gain_vs_dev06:+.4f}  improved={improved}", flush=True)
    print(f"manifest: {out_dir/'run_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())