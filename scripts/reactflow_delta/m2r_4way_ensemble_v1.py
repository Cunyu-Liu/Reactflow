#!/usr/bin/env python3
"""m2r_4way_ensemble_v1.py — architecture-decorrelated 4-way ensemble for M2R.

MOTIVATION (method-level, not data-level):
The strong 3-way (L1+L2 LightGBM + Ridge, 300-tr base) is the current headline
(+28.11% / R2 0.387).  The 3-way gains came from TWO decorrelation levers:
objective decorrelation (L1 vs L2) and capacity (100 -> 300 trees).  A third,
untested lever is ARCHITECTURE decorrelation: XGBoost's boosting/greedy-splitting
inductive bias differs from LightGBM's leaf-wise growth.  The ceiling audit
showed XGB and LGB reach similar oracle R2 (0.729 vs 0.736), but on LEGAL
features their error structures may decorrelate — exactly the property that made
L1+L2 objective decorrelation work.

Method (design-level LOO, full 236-dim stack = 230 M2R feats incl. M2_structure
+ 6 M2-transfer):
    * L1-LGB   (objective="l1",           300 trees, depth 6, lr 0.05)
    * L2-LGB   (objective="regression",   300 trees, depth 6, lr 0.05)
    * XGB-L2   (reg:squarederror,         300 trees, depth 6, lr 0.05)
    * Ridge    (alpha=1.0)
    * 4-way blend with FIXED A-PRIORI weights (0.45, 0.25, 0.20, 0.10)
      ("trust the L1 skill leader most, add L2 + architecture-decorrelated XGB,
       small Ridge") — chosen from the same a-priori logic as the 3-way, NOT
       tuned on the held-out labels.
    * the full weight plateau is reported to show the choice is not a knife-edge

Audits:
    * LOO-exclusion robustness of the 4-way gain vs the current strong-3-way
      headline (100%-positive check over 159 leave-one-design-out folds)
    * per-design (unpooled) gain
    * saves OOF for later permtest / puzzle-level audit
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_transfer_v1 as tr

SEED = 20260817
# FIXED A-PRIORI headline weights (chosen inside a wide plateau, see report)
W_L1, W_L2, W_X, W_R = 0.45, 0.25, 0.20, 0.10


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fourway_blend(l1, l2, xg, ridge, w1=W_L1, w2=W_L2, wx=W_X):
    """Weighted 4-way blend.  w_ridge = 1 - w1 - w2 - wx."""
    wr = 1.0 - w1 - w2 - wx
    return w1 * l1 + w2 * l2 + wx * xg + wr * ridge


def weight_plateau(l1, l2, xg, rg, y, mae_bl):
    """Skill over a coarse weight grid to show the plateau (not knife-edge)."""
    grid = {}
    for w1 in np.arange(0.0, 1.001, 0.1):
        for w2 in np.arange(0.0, 1.001 - w1, 0.1):
            for wx in np.arange(0.0, 1.001 - w1 - w2, 0.1):
                p = fourway_blend(l1, l2, xg, rg, w1, w2, wx)
                s = _skill(_mae(y, p), mae_bl)
                grid[f"{w1:.1f}_{w2:.1f}_{wx:.1f}"] = {
                    "w1": float(w1), "w2": float(w2), "wx": float(wx),
                    "wr": float(1.0 - w1 - w2 - wx), "skill": s}
    return grid


def run_design_level(X, X_tr, y, keys, args) -> dict:
    """Design-level LOO: L1-LGB, L2-LGB, XGB-L2, Ridge, 4-way blend + audits."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    des_list = sorted(set(keys.tolist()))
    X_comb = np.concatenate([X, X_tr], axis=1)
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge

    def loo_lgb(obj):
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
                min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                objective=obj, random_state=SEED, verbose=-1, n_jobs=2)
            g.fit(X_comb[m], y[m])
            preds[~m] = g.predict(X_comb[~m])
        return preds

    def loo_xgb():
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=SEED, n_jobs=2, verbosity=0)
            g.fit(X_comb[m], y[m])
            preds[~m] = g.predict(X_comb[~m])
        return preds

    def loo_ridge():
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            r = Ridge(alpha=1.0).fit(X_comb[m], y[m])
            preds[~m] = r.predict(X_comb[~m])
        return preds

    t0 = time.time()
    p_l1 = loo_lgb("l1")
    print(f"[4way] L1-LGB skill={_skill(_mae(y, p_l1), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p_l2 = loo_lgb("regression")
    print(f"[4way] L2-LGB skill={_skill(_mae(y, p_l2), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p_xg = loo_xgb()
    print(f"[4way] XGB-L2 skill={_skill(_mae(y, p_xg), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p_rg = loo_ridge()
    print(f"[4way] Ridge skill={_skill(_mae(y, p_rg), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    blend = fourway_blend(p_l1, p_l2, p_xg, p_rg, W_L1, W_L2, W_X)

    # ---- previous strong-3-way headline (L1+L2 LGB + Ridge, 0.6/0.3/0.1) ----
    prev_3way = 0.6 * p_l1 + 0.3 * p_l2 + 0.1 * p_rg

    results = {
        "l1_lgb": {"mae": _mae(y, p_l1), "skill": _skill(_mae(y, p_l1), mae_bl),
                   "r2": _r2(y, p_l1)},
        "l2_lgb": {"mae": _mae(y, p_l2), "skill": _skill(_mae(y, p_l2), mae_bl),
                   "r2": _r2(y, p_l2)},
        "xgb_l2": {"mae": _mae(y, p_xg), "skill": _skill(_mae(y, p_xg), mae_bl),
                   "r2": _r2(y, p_xg)},
        "ridge": {"mae": _mae(y, p_rg), "skill": _skill(_mae(y, p_rg), mae_bl),
                  "r2": _r2(y, p_rg)},
        "prev_strong_3way": {
            "mae": _mae(y, prev_3way), "skill": _skill(_mae(y, prev_3way), mae_bl),
            "r2": _r2(y, prev_3way)},
        "fourway_blend_a_priori": {
            "mae": _mae(y, blend), "skill": _skill(_mae(y, blend), mae_bl),
            "r2": _r2(y, blend), "w1": W_L1, "w2": W_L2, "wx": W_X,
            "wr": 1.0 - W_L1 - W_L2 - W_X},
    }

    # ---- LOO-exclusion gain (4-way vs prev strong-3-way) ----
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            continue
        s_new = _skill(_mae(y[m], blend[m]), mae_bl)
        s_prev = _skill(_mae(y[m], prev_3way[m]), mae_bl)
        gains.append(s_new - s_prev)
    gains = np.array(gains)
    loo = {"gain_mean_pp": float(gains.mean() * 100),
           "gain_min_pp": float(gains.min() * 100),
           "gain_max_pp": float(gains.max() * 100),
           "pct_positive": float((gains > 0).mean()),
           "n_folds": int(len(gains))}

    # per-design (unpooled) gain
    dg = []
    for held in des_list:
        m = keys == held
        if m.sum() == 0:
            continue
        dg.append(_skill(_mae(y[m], blend[m]), y_med) -
                  _skill(_mae(y[m], prev_3way[m]), y_med))
    dg = np.array(dg)
    per_design = {"gain_mean_pp": float(dg.mean() * 100),
                  "pct_positive": float((dg > 0).mean()),
                  "n_designs": int(len(dg))}

    plateau = weight_plateau(p_l1, p_l2, p_xg, p_rg, y, mae_bl)

    np.savez(out / "m2r_4way_oof.npz",
             l1=p_l1, l2=p_l2, xg=p_xg, ridge=p_rg,
             blend=blend, prev_3way=prev_3way, y=y, keys=keys)

    report = {
        "schema": "reactflow_delta.m2r_4way_ensemble.v1",
        "dataset": "OpenKnot_M2R",
        "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "n_features": int(X_comb.shape[1]),
        "seed": SEED,
        "headline_weights": {"w1": W_L1, "w2": W_L2, "wx": W_X,
                             "wr": 1.0 - W_L1 - W_L2 - W_X},
        "baseline_mae": mae_bl,
        "results": results,
        "loo_exclusion_vs_strong_3way": loo,
        "per_design_vs_strong_3way": per_design,
        "plateau": plateau,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_4way_ensemble_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n=== M2R 4-way ensemble (design-level LOO, {X_comb.shape[1]}-dim) ===")
    for k, v in results.items():
        print(f"  {k:24s} MAE={v['mae']:.4f} skill={v['skill']:+.4f} R2={v['r2']:.4f}")
    g = loo
    print(f"  4-way vs strong-3-way LOO-exclusion gain: mean={g['gain_mean_pp']:+.2f}pp "
          f"range=[{g['gain_min_pp']:+.2f},{g['gain_max_pp']:+.2f}]pp "
          f"pct_pos={g['pct_positive']:.3f}")
    print(f"  per-design: mean={per_design['gain_mean_pp']:+.2f}pp "
          f"pct_pos={per_design['pct_positive']:.3f}")
    print(f"  wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_4way_ensemble_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred", required=True,
                    help="M2 keyed_predictions jsonl (design-level OOF)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)

    m2_oof = tr.load_m2_oof(args.m2_pred)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    print(f"[4way] n_samples={len(y)} X={X.shape} X_tr={X_tr.shape}",
          flush=True)

    run_design_level(X, X_tr, y, keys, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
