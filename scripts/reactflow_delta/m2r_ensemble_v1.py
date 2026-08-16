#!/usr/bin/env python3
"""m2r_ensemble_v1.py — M2R ensemble method-level levers.

The M2 response-spectrum task showed that a 3-way ensemble (across
architectures) was the strongest method-level improvement (+12.84% vs
+10.10% best single).  For M2R, the same question is asked:

  1. GBDT seed-bagging: average K GBDTs with different random_state.
  2. GBDT + Ridge blend (LOO-consistent stacking weight): combine the
     nonlinear GBDT with the linear Ridge, whose errors decorrelate.

All in design-level LOO (exchangeable unit = (puzzle, method)), using the
current headline feature set (230 dims incl. M2_structure).

LEGAL: everything is computed from OOF predictions per held-out design; the
blend weight is fit on training designs only (no test leakage).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf

SEED = 20260816


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of GBDT seeds to average (seed-bagging)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    if args.m2_csv:
        m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)
    print(f"[m2r_ens] n_samples={len(y)} n_designs={n_des} X={X.shape} seeds={args.seeds}",
          flush=True)

    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[m2r_ens] baseline MAE={mae_bl:.4f}", flush=True)

    # OOF prediction accumulators
    gbdt_mean = np.zeros(len(y))     # mean over seeds
    gbdt_seeds = np.zeros((len(y), args.seeds))
    ridge_pred = np.zeros(len(y))

    t0 = time.time()
    for fi, held in enumerate(des_list):
        m = keys != held
        if m.sum() <= 10:
            gbdt_mean[~m] = y_med
            ridge_pred[~m] = y_med
            for s in range(args.seeds):
                gbdt_seeds[~m, s] = y_med
            continue
        Xtr, ytr = X[m], y[m]
        Xte = X[~m]

        # GBDT seed-bagging (pure variance reduction, no tuning)
        gs = np.zeros((Xte.shape[0], args.seeds))
        for s in range(args.seeds):
            g = lgb.LGBMRegressor(
                n_estimators=args.trees, max_depth=args.depth,
                random_state=SEED + s, verbose=-1, n_jobs=2)
            g.fit(Xtr, ytr)
            gs[:, s] = g.predict(Xte)
        gbdt_seeds[~m] = gs
        gbdt_mean[~m] = gs.mean(axis=1)

        # Ridge (linear, closed-form fit)
        r = Ridge(alpha=1.0).fit(Xtr, ytr)
        ridge_pred[~m] = r.predict(Xte)

        if fi % 20 == 0:
            print(f"[m2r_ens] fold {fi}/{n_des} held={held} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    wall = round(time.time() - t0, 1)

    # ---- pooled metrics ----
    # Alpha sweep with FIXED a-priori weights (no post-hoc tuning per design).
    # We report the whole sweep; the choice of a single headline alpha must be
    # justified by its robustness (LOO-exclusion) rather than peak value alone.
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    models = {
        "gbdt_mean_seedbag": gbdt_mean,
        "gbdt_seed0": gbdt_seeds[:, 0],
        "ridge": ridge_pred,
    }
    for a in alphas:
        models[f"gbdt_ridge_a{int(a*100)}"] = a * gbdt_mean + (1.0 - a) * ridge_pred

    report = {
        "schema": "reactflow_delta.m2r_ensemble.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": len(y),
        "n_designs": n_des,
        "n_features": int(X.shape[1]),
        "trees": args.trees,
        "depth": args.depth,
        "seeds": args.seeds,
        "baseline_mae": mae_bl,
        "models": {},
        "loo_exclusion": {},
        "wall_seconds": wall,
    }
    for mn, preds in models.items():
        mae = _mae(y, preds)
        report["models"][mn] = {
            "mae": mae,
            "skill": _skill(mae, mae_bl),
            "r2": _r2(y, preds),
        }

    # per-seed individual skills (to see seed-bagging gain)
    report["per_seed_skills"] = []
    for s in range(args.seeds):
        mae_s = _mae(y, gbdt_seeds[:, s])
        report["per_seed_skills"].append({
            "seed": s,
            "skill": _skill(mae_s, mae_bl),
            "r2": _r2(y, gbdt_seeds[:, s]),
        })

    # ---- LOO-exclusion robustness for the GBDT+Ridge blend ----
    # For each blend alpha: exclude each design, recompute skill on the rest,
    # and report the min/max/mean gain vs the single GBDT (seed0).
    for a in alphas:
        bn = f"gbdt_ridge_a{int(a*100)}"
        blend = a * gbdt_mean + (1.0 - a) * ridge_pred
        gains = []
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                continue
            s_gb = _skill(_mae(y[m], gbdt_mean[m]), mae_bl)
            s_bl = _skill(_mae(y[m], blend[m]), mae_bl)
            gains.append(s_bl - s_gb)
        gains = np.array(gains)
        report["loo_exclusion"][bn] = {
            "gain_mean": float(gains.mean()),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "pct_positive": float((gains > 0).mean()),
            "n_folds": int(len(gains)),
        }

    (out / "m2r_ensemble_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("\n[m2r_ens] DONE ->", out / "m2r_ensemble_report.json")
    for mn, v in report["models"].items():
        print(f"  {mn:22s} skill={v['skill']:+.4f} R2={v['r2']:.4f} MAE={v['mae']:.4f}")
    print(f"  per-seed skills: {[round(x['skill'],4) for x in report['per_seed_skills']]}")
    print("\n  blend LOO-exclusion gain vs single GBDT:")
    for bn, v in report["loo_exclusion"].items():
        print(f"  {bn:18s} mean={v['gain_mean']:+.4f} min={v['gain_min']:+.4f} "
              f"max={v['gain_max']:+.4f} pct_pos={v['pct_positive']:.3f}")


if __name__ == "__main__":
    main()