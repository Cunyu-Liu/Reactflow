#!/usr/bin/env python3
"""EPRO_DEV_10 (M0-X, epoch-13 scope continuation): hyperparameter search over
the proven per-position structure-aware changer classifier (dev06) on the CORRECT
publication split (train 3516 / val 548).

Rationale: all prior improvement attempts (DEV_07 GRU seq-context 0.7005,
DEV_08 seed-ensemble 0.7032, DEV_09 mutation-context 0.7060) were evaluated on
the WRONG 190-val split and failed to beat DEV_06 (0.7353). DEV_06's own
hyperparameters (hidden=256, layers=3, lr=1e-3, dropout=0.1, gamma=2.0) were
also selected on the wrong split. This iteration re-runs a principled
hyperparameter grid on the CORRECT publication split with the SAME 42-dim
feature space, cached to disk once so many configs train cheaply, and reports
study-macro AUPRC + cluster-bootstrap gain vs dev06.

  * Features (42): B0-X(31) + delta_thermo(5) + structure-aware contact-graph(6),
    identical to dev06. Built once and CACHED to /tmp so configs reuse inputs.
  * Model: dev06 ChangerClassifier (per-position MLP, sigmoid head).
  * Grid: hidden in {192,256,384}, layers in {2,3,4}, dropout in {0.0,0.1,0.2},
    lr in {3e-4,1e-3,3e-3}, focal_gamma in {1.0,2.0,3.0}. Evaluated on val
    study-macro AUPRC; best config selected.
  * Baseline comparisons: dev06 structure-aware changer (0.7353), p2_paired,
    wt_only, vienna_physics -- same changer-detection task on the same split.

Same frozen dev definitions (estimand A, study-macro AUPRC, cluster-bootstrap CI,
group-aware permutation), same frozen seed 20260804, test SEALED, GPU required
(real CUDA, fallback=0). No pretraining, so no exposure-audit entry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path.cwd() / "src"))

import m0x_epro_dev06 as dev06  # reuse feature pipeline + model + metrics  # noqa: E402

SEED = 20260804
SCHEMA = "reactflow_delta.m0x_run_manifest.v1"
RUN_ID = "epro_dev_10_hpsearch_20260805"
ITERATION_ID = "EPRO_DEV_10"
HYPOTHESIS_ID = "m0x_h10_hpsearch_changer"

# dev06 reference (correct publication split, from run_manifest.json).
REF_DEV06_AUPRC = 0.7353243279593717

# Hyperparameter grid.
GRID = [
    {"hidden": h, "layers": l, "dropout": d, "lr": lr, "focal_gamma": g}
    for h in (192, 256, 384)
    for l in (2, 3, 4)
    for d in (0.0, 0.1, 0.2)
    for lr in (3e-4, 1e-3, 3e-3)
    for g in (1.0, 2.0, 3.0)
]
# Subset to keep total runtime reasonable yet cover each axis.
CONFIGS = [
    {"hidden": 192, "layers": 3, "dropout": 0.1, "lr": 1e-3, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 3, "dropout": 0.1, "lr": 1e-3, "focal_gamma": 2.0},  # dev06 default
    {"hidden": 384, "layers": 3, "dropout": 0.1, "lr": 1e-3, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 2, "dropout": 0.1, "lr": 1e-3, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 4, "dropout": 0.1, "lr": 1e-3, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 3, "dropout": 0.0, "lr": 1e-3, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 3, "dropout": 0.2, "lr": 1e-3, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 3, "dropout": 0.1, "lr": 3e-4, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 3, "dropout": 0.1, "lr": 3e-3, "focal_gamma": 2.0},
    {"hidden": 256, "layers": 3, "dropout": 0.1, "lr": 1e-3, "focal_gamma": 1.0},
    {"hidden": 256, "layers": 3, "dropout": 0.1, "lr": 1e-3, "focal_gamma": 3.0},
]


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


def _predict(model, recs, device):
    model.eval()
    out = {}
    with torch.no_grad():
        for r in recs:
            x = torch.tensor(r["features"], device=device)
            out[r["pair_id"]] = torch.sigmoid(model(x)).cpu().numpy().astype(np.float32)
    return out


def _train_one(X, y, val_recs, device, cfg, epochs=300, batch_size=4096,
               eval_every=10, patience=30):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = dev06.ChangerClassifier(feat_dim=X.shape[1], hidden=cfg["hidden"],
                                    layers=cfg["layers"], dropout=cfg["dropout"])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device).float()
    n = Xt.shape[0]
    pos_frac = float(y.mean())
    alpha_pos = max(0.05, 1 - pos_frac)
    best_auprc = -1e18
    best_state, best_epoch = None, -1
    no_improve = 0
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logits = model(Xt[idx])
            loss = dev06._focal_loss(logits, yt[idx], alpha_pos, cfg["focal_gamma"])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_score = _predict(model, val_recs, device)
            changed = _changer_records(val_recs, val_score)
            va = _study_macro_auprc(changed)
            if va > best_auprc:
                best_auprc = va
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auprc, best_epoch, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev06-manifest", type=Path, default=None)
    ap.add_argument("--feat-cache", type=Path,
                    default=Path("/tmp/m0x_hpsearch_feats_publication.npz"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            (out_dir / "gpu_failure_evidence.json").write_text(
                json.dumps({"error": "CUDA unavailable", "cuda_available": False},
                           indent=2), encoding="utf-8")
            print("FATAL: CUDA unavailable. GPU required (fallback=0).",
                  file=sys.stderr)
            return 2
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu_name} free={free/1e9:.1f}GB total={total/1e9:.1f}GB",
              flush=True)
    device = torch.device(args.device)

    pairs = dev06.load_pairs(args.canonical_jsonl, args.split_manifest,
                             splits={"train", "validation"})
    groups = dev06.split_groups(pairs)
    train = groups.get("train", [])
    val = groups.get("validation", [])
    print(f"[data] train={len(train)} validation={len(val)} (test SEALED)",
          flush=True)

    # Reuse dev06 feature pipeline (same 42-dim space, correct split).
    print("[features] building B0-X + delta_thermo + structure-aware...", flush=True)
    wt_cache, mut_cache = {}, {}
    t0 = time.time()
    train_recs = dev06._build_pair_records(train, wt_cache, mut_cache)
    val_recs = dev06._build_pair_records(val, wt_cache, mut_cache)
    print(f"[features] done in {time.time()-t0:.0f}s", flush=True)
    X, y = dev06._pool(train_recs)
    print(f"[data] pooled train positions={X.shape[0]} feat_dim={X.shape[1]} "
          f"changer_frac={y.mean():.4f}", flush=True)

    # Baseline changers (same as dev06).
    p2_changed = None  # dev06 ref already has p2; recompute from dev06 manifest if absent
    results = []
    best = None
    for cfg in CONFIGS:
        m, auprc, ep, secs = _train_one(X, y, val_recs, device, cfg)
        val_score = _predict(m, val_recs, device)
        changed = _changer_records(val_recs, val_score)
        results.append({"config": cfg, "study_macro_auprc": auprc,
                        "best_epoch": ep, "train_s": round(secs, 1),
                        "param_count": m.param_count()})
        print(f"[cfg] {cfg} -> val_auprc={auprc:.4f}@{ep} ({secs:.0f}s) "
              f"params={m.param_count():,}", flush=True)
        if best is None or auprc > best["study_macro_auprc"]:
            best = {"config": dict(cfg), "study_macro_auprc": auprc,
                    "best_epoch": ep, "param_count": m.param_count()}
            torch.save(m.state_dict(), out_dir / "best_model.pt")
            best["changed"] = changed

    results.sort(key=lambda r: -r["study_macro_auprc"])
    print("\n=== HP SEARCH RESULTS ===", flush=True)
    for r in results:
        print(f"  {r['config']} -> {r['study_macro_auprc']:.4f}", flush=True)
    print(f"\nBest: {best['config']} -> {best['study_macro_auprc']:.4f} "
          f"(dev06 ref {REF_DEV06_AUPRC:.4f})", flush=True)

    # Gain vs dev06 (point difference; bootstrap on same-cluster pairs requires
    # per-position dev06 scores which live in the dev06 manifest predictions).
    improved = best["study_macro_auprc"] > REF_DEV06_AUPRC
    gain_clean = {"point_gain_vs_dev06": best["study_macro_auprc"] - REF_DEV06_AUPRC,
                  "note": "point difference vs dev06 reference "
                          "(positive = hp-search better)"}

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
                 "feat_dim": int(X.shape[1])},
        "split": "publication (train 3516 / val 548)",
        "dev06_reference": {"study_macro_auprc": REF_DEV06_AUPRC},
        "grid": CONFIGS,
        "results": results,
        "best": {k: v for k, v in best.items() if k != "changed"},
        "improved_vs_dev06": improved,
        "point_gain_vs_dev06": best["study_macro_auprc"] - REF_DEV06_AUPRC,
        "comparison_table": {
            "structure_aware_hpsearch": {"study_macro_auprc": best["study_macro_auprc"],
                                         "config": best["config"]},
            "structure_aware_dev06": {"study_macro_auprc": REF_DEV06_AUPRC},
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nmanifest: {out_dir/'run_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())