#!/usr/bin/env python3
"""run_m2r_v1.py — M2R rescue_factor prediction: multi-model LOO evaluation.

Models:
  1. Baseline: median rescue_factor (sequence-free).
  2. Ridge regression (linear, closed-form, fast).
  3. GBDT (LightGBM, non-linear).
  4. MLP (small neural net, GPU).

Exchangeable unit: (puzzle, method) design — leave-one-design-out.
Output: horizontal comparison table + pooled metrics + per-design breakdown.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf

SEED = 20260816


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _skill(mae_model, mae_baseline):
    return 1.0 - mae_model / mae_baseline if mae_baseline > 0 else 0.0


def train_ridge(Xtr, ytr, alpha=1.0):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha).fit(Xtr, ytr)


def train_gbdt(Xtr, ytr, n_estimators=100, max_depth=3):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=SEED, verbose=-1, n_jobs=8).fit(Xtr, ytr)


def train_mlp(Xtr, ytr, device=None):
    """Small MLP: 2 hidden layers, 64 ReLU, batch norm, dropout."""
    import torch, torch.nn as nn
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, d = Xtr.shape
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(d, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, 1),
    ).to(device)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(ytr, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    model.train()
    bs = 256
    for ep in range(30):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
    return model


def predict_ridge(model, Xte):
    return model.predict(Xte)


def predict_gbdt(model, Xte):
    return model.predict(Xte)


def predict_mlp(model, Xte, device=None):
    import torch
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.ascontiguousarray(Xte, dtype=np.float32)).to(device)
        return model(Xt).squeeze(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-device", default="2")
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = None
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[m2r] GPU OK: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("[m2r] CUDA not available, MLP will use CPU", flush=True)
    except ImportError:
        print("[m2r] torch not available", flush=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = m2r.build_all_pair_samples(designs)
    samples = [s for s in samples if s.rescue_factor is not None]
    print(f"[m2r] n_samples={len(samples)} n_designs={len(designs)}", flush=True)

    X, y, keys, feat_names = m2rf.build_all(samples)
    keys = np.array(keys)
    n_feats = X.shape[1]
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)
    print(f"[m2r] X={X.shape} n_designs={n_des}", flush=True)

    # baseline
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[m2r] baseline MAE={mae_bl:.4f} (median={y_med:.4f})", flush=True)

    # ---- full LOO ----
    ridge_preds = np.zeros(len(y))
    gbdt_preds = np.zeros(len(y))
    mlp_preds = np.zeros(len(y))
    mlp_ok = not args.skip_mlp and device is not None

    t0 = time.time()
    for fi, held in enumerate(des_list):
        m = keys != held
        if m.sum() <= 10:
            ridge_preds[~m] = y_med
            gbdt_preds[~m] = y_med
            if mlp_ok:
                mlp_preds[~m] = y_med
            continue
        Xtr, ytr = X[m], y[m]
        Xte = X[~m]

        if fi % 20 == 0:
            print(f"[m2r] fold {fi}/{n_des} held={held} ntr={m.sum()} nte={(~m).sum()}", flush=True)

        fold_t0 = time.time()
        # Ridge
        r = train_ridge(Xtr, ytr)
        ridge_preds[~m] = predict_ridge(r, Xte)

        # GBDT
        try:
            g = train_gbdt(Xtr, ytr)
            gbdt_preds[~m] = predict_gbdt(g, Xte)
        except Exception:
            gbdt_preds[~m] = y_med

        # MLP
        if mlp_ok:
            try:
                m_mlp = train_mlp(Xtr, ytr, device=device)
                mlp_preds[~m] = predict_mlp(m_mlp, Xte, device=device)
            except Exception:
                mlp_preds[~m] = y_med
        if fi % 20 == 0:
            print(f"[m2r] fold {fi} done in {time.time()-fold_t0:.1f}s", flush=True)

    wall = round(time.time() - t0, 1)

    # ---- pooled metrics ----
    models = {
        "ridge": ridge_preds,
        "gbdt": gbdt_preds,
    }
    if mlp_ok:
        models["mlp"] = mlp_preds

    # also compute per-design exclusion sensitivity (LOO robustness)
    all_preds = models
    per_design_sensitivity = {}
    for mn, preds in all_preds.items():
        pds = []
        for held in des_list:
            m = keys != held
            ma = _mae(y[m], preds[m])
            sk = _skill(ma, mae_bl) if mae_bl * (1 - 1e-9) > 0 else 0.0
            pds.append({"excluded_design": held, "pooled_mae": ma, "pooled_skill": sk})
        per_design_sensitivity[mn] = pds

    mae_baseline = mae_bl
    report = {
        "schema": "reactflow_delta.m2r_loo_v1",
        "dataset": "OpenKnot_M2R",
        "exchangeable_unit": "design",
        "n_designs": n_des, "n_samples": int(len(y)), "n_features": n_feats,
        "baseline": {"median": float(y_med), "mae": float(mae_bl)},
        "models": {},
        "per_design_sensitivity": per_design_sensitivity,
        "wall_seconds": wall,
    }
    for mn, preds in all_preds.items():
        mae = _mae(y, preds)
        sk = _skill(mae, mae_bl)
        r2 = _r2(y, preds)
        # per-design
        dskills = []
        for held in des_list:
            m = keys != held
            if m.sum() > 0:
                dskills.append(_skill(_mae(y[m], preds[m]), mae_bl))
        das = np.array(dskills)
        report["models"][mn] = {
            "mae": float(mae), "skill": float(sk), "r2": float(r2),
            "per_design_mean_skill": float(das.mean()),
            "per_design_median_skill": float(np.median(das)),
            "per_design_pct_positive": float((das > 0).mean()),
            "per_design_min_skill": float(das.min()) if len(das) else None,
            "per_design_max_skill": float(das.max()) if len(das) else None,
        }

    # ---- final table ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_loo_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== M2R rescue_factor LOO (horizontal comparison) ===")
    print(f"  baseline MAE={mae_bl:.4f}")
    for mn in sorted(report["models"].keys()):
        r = report["models"][mn]
        print(f"  {mn:8s} MAE={r['mae']:.4f} skill={r['skill']:+.4f} "
              f"R2={r['r2']:.4f} pct+={r['per_design_pct_positive']:.3f} "
              f"design_skill={r['per_design_mean_skill']:+.4f} "
              f"[{r['per_design_min_skill']:+.4f},{r['per_design_max_skill']:+.4f}]")
    print(f"wall={wall:.0f}s DONE -> {out / 'm2r_loo_report.json'}")


if __name__ == "__main__":
    sys.exit(main())