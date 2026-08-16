#!/usr/bin/env python3
"""M0-X EPRO_DEV_12: supervised Delta-MAGNITUDE regression (MLP regression head).

Motive (dialectical review of pair-level evaluation, 20260807):
  The prior SOTA pair-level comparison showed ALL models (supervised + zero-shot
  folding) near random, which we diagnosed as TWO layered evaluation failures:
    (1) degenerate binary pair labels (any/local_cluster -> 100% prevalence) and
        a hard oracle ceiling (~0.5) for the only non-degenerate majority label;
    (2) SEMANTIC MISMATCH in the continuous-burden endpoint: dev10/07/09 are
        ChangerClassifier models whose saved score is sigmoid P(changer) in
        [0,1] (m0x_sota_pairlevel_v3.py L325), not a delta magnitude.  Comparing
        that bounded probability against the unbounded true burden
        mean(|delta|/scale) produced the misleading negative Spearman.

  This script retrains the SUPERVISED model with a true DELTA-MAGNITUDE
  regression head (linear output, no sigmoid), on the SAME 42-dim B0-X +
  delta_thermo + structure-aware features and SAME frozen train/val split as
  dev10, so it becomes eligible for the continuous-burden PRIMARY endpoint.

  Model (Route B, user-confirmed): flat per-position MLP with a LINEAR head
  outputting signed delta_r_hat; trained with MAE on SIGNED SCALE-STANDARDIZED
  delta_true (target = pair.delta / pair_scale), so |delta_r_hat| lives in the
  SAME units as the continuous-burden endpoint truth mean(|delta|/scale).  The
  continuous-burden score is |delta_r_hat|.

  Selection: train to best validation regression skill (WMAE-skill vs wt_only),
  with early stopping; checkpoint = best val state.  Test is SEALED, never read.

  GPU required (CUDA_VISIBLE_DEVICES=1, fallback=0).  Same frozen seed 20260804.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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

import m0x_epro_dev06 as dev06  # noqa: E402  (feature pipeline + pair records)
from b0x_baselines import run_baseline, _pair_scale  # noqa: E402
from b0x_data import load_pairs, split_groups  # noqa: E402
from b0x_evaluate import pooled_skill  # noqa: E402

SEED = 20260804
TEMPERATURE = 37.0
SCHEMA = "reactflow_delta.m0x_dev12_regression_run_manifest.v1"
RUN_ID = "epro_dev12_regression_std_20260807"
ITERATION_ID = "EPRO_DEV_12_REGRESSION"

# Best dev10 hyper-params (route-B: keep architecture, swap head+loss).
DEFAULT_CONFIG = {"hidden": 256, "layers": 3, "dropout": 0.1,
                  "lr": 3e-4, "batch_size": 4096, "weight_decay": 1e-5,
                  "epochs": 300, "eval_every": 10, "patience": 30}


# ---------------------------------------------------------------------------
# Model: per-position MLP with LINEAR regression head (signed delta_r_hat)
# ---------------------------------------------------------------------------
class DeltaMagnitudeRegressor(nn.Module):
    """Flat per-position MLP, linear head -> signed delta. |output| = magnitude.

    Same architecture width as dev10's ChangerClassifier (hidden/layers/dropout)
    but the final layer is LINEAR (not sigmoid) and the target is signed delta
    (not a binary changer label).  This makes the output a true delta magnitude
    that is eligible for the continuous-burden endpoint.
    """

    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        blocks = [nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(layers - 1):
            blocks.append(nn.Linear(hidden, hidden))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(hidden, 1))  # LINEAR head (no sigmoid)
        self.net = nn.Sequential(*blocks)

    def forward(self, x):  # (N, F) -> signed delta_r_hat (N,)
        return self.net(x).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Regression training (early stop on val WMAE-skill vs wt_only)
# ---------------------------------------------------------------------------
def _train_regressor(model, X, y, val_recs, val_pairs, wt_ref_preds, device,
                     epochs, batch_size, lr, weight_decay, eval_every, patience):
    """Train with MAE on signed SCALE-STANDARDIZED delta (delta/scale); select
    best val WMAE-skill vs wt_only.

    y is the pooled standardized delta_target for eligible positions (delta/scale),
    so |delta_r_hat| is comparable to the continuous-burden truth mean(|delta|/scale).
    val_recs are dict features (for _predict); val_pairs are Pair objects
    (pooled_skill needs Pair, not dict recs); predictions keyed by pair_id.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device).float()
    n = Xt.shape[0]
    best_skill = -1e18
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = {"epochs": []}
    t0 = time.time()

    if wt_ref_preds is None:
        raise ValueError("wt_ref_preds required for val skill selection")

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        tot = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            mu = model(Xt[idx])
            loss = (mu - yt[idx]).abs().mean()  # MAE on signed delta
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        epoch_loss = tot / max(nb, 1)

        rec = {"epoch": epoch, "train_mae": epoch_loss}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            preds = _predict(model, val_recs, device)
            # Model outputs STANDARDIZED delta (target = delta/scale).  Rescale
            # back to raw units so pooled_skill (whose truth is raw pair.delta)
            # stays unit-consistent for checkpoint selection.
            preds_raw = {}
            for p in val_pairs:
                if p.pair_id in preds:
                    preds_raw[p.pair_id] = preds[p.pair_id] * _pair_scale(p)
            sk = pooled_skill(val_pairs, preds_raw, wt_ref_preds)
            val_skill = sk["skill_wmae"]
            val_mae = sk["skill_mae"]
            rec["val_skill_wmae"] = val_skill
            rec["val_skill_mae"] = val_mae
            if val_skill > best_skill:
                best_skill = val_skill
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(f"[train] epoch {epoch+1}/{epochs} mae={epoch_loss:.4f} "
                  f"val_skill={val_skill:.4f} best={best_skill:.4f}@{best_epoch} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if no_improve >= patience:
                print(f"[train] early stop at epoch {epoch} "
                      f"(best {best_skill:.4f}@{best_epoch})", flush=True)
                break
        history["epochs"].append(rec)

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_skill_wmae"] = best_skill
    history["best_epoch"] = best_epoch
    history["total_elapsed_s"] = time.time() - t0
    return model, history


def _predict(model, recs, device) -> dict[str, np.ndarray]:
    """Return per-position SIGNED delta_r_hat (magnitude = |value|)."""
    model.eval()
    out = {}
    with torch.no_grad():
        for r in recs:
            x = torch.tensor(r["features"], device=device)
            mu = model(x).cpu().numpy().astype(np.float32)
            out[r["pair_id"]] = mu
    return out


def _standardized_delta_target(pair, mask, scale) -> list:
    """Per-position SCALE-STANDARDIZED signed delta = pair.delta[i] / scale.

    The regression target must live in the SAME units as the continuous-burden
    endpoint truth mean(|delta|/scale), so |delta_r_hat| is directly comparable.
    scale is pair-constant, so dividing each eligible delta by the pair scale
    standardizes the whole pair.  Non-finite / out-of-range positions -> 0.0.
    """
    return [float(pair.delta[i]) / scale if i < len(pair.delta)
            and math.isfinite(float(pair.delta[i])) else 0.0
            for i in range(len(mask))]


def _pool_delta_target(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pool eligible-position features and SCALE-STANDARDIZED delta targets.

    delta target = pair.delta / _pair_scale(pair) (scale-relative), so the model
    predicts delta in the SAME units as the continuous-burden endpoint truth
    mean(|delta|/scale); |delta_r_hat| is directly comparable.
    """
    Xs, ys = [], []
    for r in recs:
        for i in range(len(r["mask"])):
            if r["mask"][i]:
                Xs.append(r["features"][i])
                ys.append(r["delta"][i])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float64))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    for k, v in DEFAULT_CONFIG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            (out_dir / "gpu_failure_evidence.json").write_text(
                json.dumps({"error": "CUDA unavailable; GPU required (fallback=0)",
                            "cuda_available": torch.cuda.is_available()}, indent=2),
                encoding="utf-8")
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

    # wt_only reference predictions for val skill selection.
    print("[baseline] wt_only for val skill reference...", flush=True)
    wt_res = run_baseline("wt_only", train, val, device=args.device)
    wt_ref_preds = wt_res.predictions
    print(f"[baseline] wt_only status={wt_res.status} params={wt_res.param_count}",
          flush=True)

    # Build pair records (same 42-dim features as dev10).
    print("[features] building B0-X + delta_thermo + structure-aware features...",
          flush=True)
    wt_cache, mut_cache = {}, {}
    t0 = time.time()
    train_recs = dev06._build_pair_records(train, wt_cache, mut_cache)
    val_recs = dev06._build_pair_records(val, wt_cache, mut_cache)
    print(f"[features] done in {time.time()-t0:.0f}s", flush=True)

    # Attach SCALE-STANDARDIZED signed delta (delta/scale) as the regression
    # target.  The continuous-burden endpoint truth is mean(|delta|/scale), so
    # the model must predict delta in the SAME standardized units for
    # |delta_r_hat| to be directly comparable.  scale is pair-constant, so
    # dividing each per-position delta by _pair_scale(p) standardizes the pair.
    # (dev06 records carry delta_thermo, not the raw delta target; pull from the
    # Pair objects.)
    for r, p in zip(train_recs, train):
        r["delta"] = _standardized_delta_target(p, r["mask"], _pair_scale(p))
    for r, p in zip(val_recs, val):
        r["delta"] = _standardized_delta_target(p, r["mask"], _pair_scale(p))

    X, y = _pool_delta_target(train_recs)
    print(f"[data] pooled train positions={X.shape[0]} feat_dim={X.shape[1]} "
          f"delta_mean={float(y.mean()):.4f} delta_abs_mean={float(np.abs(y).mean()):.4f}",
          flush=True)

    # Train delta-magnitude regressor.
    model = DeltaMagnitudeRegressor(feat_dim=X.shape[1], hidden=args.hidden,
                                    layers=args.layers, dropout=args.dropout)
    print(f"[model] DeltaMagnitudeRegressor params={model.param_count():,}",
          flush=True)
    model, hist = _train_regressor(model, X, y, val_recs, val, wt_ref_preds, device,
                                   args.epochs, args.batch_size, args.lr,
                                   args.weight_decay, args.eval_every,
                                   args.patience)

    val_preds = _predict(model, val_recs, device)
    train_preds = _predict(model, train_recs, device)

    # Save signed delta_r_hat predictions (magnitude = |value|).
    np.savez_compressed(str(out_dir / "predictions.npz"), **val_preds)
    np.savez_compressed(str(out_dir / "train_predictions.npz"), **train_preds)
    torch.save(model.state_dict(), str(out_dir / "best_model.pt"))

    # Continuous-burden summary (|delta_r_hat|).
    from m0x_eval_recovery import (  # noqa: E402
        burden_applicability, single_pair_burden, spearman, kendall, ndcg_at_k_scores,
    )
    app = burden_applicability("epro_dev12")
    true_b, pred_b = [], []
    for p in val:
        sc = val_preds.get(p.pair_id)
        if sc is None:
            continue
        mask = np.asarray(p.mask, dtype=bool)
        delta = np.asarray(p.delta, dtype=np.float64)
        b = single_pair_burden(float(np.mean(np.abs(sc[mask]))),
                               mask, delta, _pair_scale(p))
        true_b.append(b["true"])
        pred_b.append(b["pred"])
    burden = {
        "status": app["status"],
        "n": len(true_b),
        "spearman": spearman(np.array(true_b), np.array(pred_b)),
        "kendall": kendall(np.array(true_b), np.array(pred_b)),
        "ndcg_at_10": ndcg_at_k_scores(np.array(true_b), np.array(pred_b), 10),
    }
    print(f"[burden] epro_dev12 n={burden['n']} spearman={burden['spearman']:.4f} "
          f"kendall={burden['kendall']:.4f} ndcg@10={burden['ndcg_at_10']:.4f}",
          flush=True)

    manifest = {
        "schema": SCHEMA, "run_id": RUN_ID, "iteration": ITERATION_ID,
        "model": "DeltaMagnitudeRegressor (Route B, linear head + MAE)",
        "score_semantics": "signed delta_r_hat in scale-standardized units (delta/scale); "
                        "magnitude=|value|",
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "param_count": model.param_count(),
        "best_val_skill_wmae": hist["best_val_skill_wmae"],
        "best_epoch": hist["best_epoch"],
        "total_elapsed_s": hist["total_elapsed_s"],
        "train_mae": hist["epochs"][-1]["train_mae"],
        "val_skill_wmae_final": hist["epochs"][-1].get("val_skill_wmae"),
        "continuous_burden": burden,
        "test_access": "SEALED",
        "gpu": "cuda" if args.device == "cuda" else "cpu",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] manifest -> {out_dir/'run_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())