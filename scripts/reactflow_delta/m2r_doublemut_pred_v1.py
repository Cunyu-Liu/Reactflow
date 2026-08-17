#!/usr/bin/env python3
"""m2r_doublemut_pred_v1.py — leak-free auxiliary predictor of the double-mutant
RMSD numerator (rD) as a legal feature for the M2R rescue model.

MOTIVATION (method-level, tests the ceiling claim):
The rescue factor is exactly  rescue = 1 - rD/sqrt(rA^2 + rB^2)  over the design
region, where rD = RMSD(double-mutant, WT), rA/rB = RMSD(single-mutant, WT).
The ceiling audit showed legal features saturate near R2 0.40 while the oracle
(knowing rD) reaches R2 0.73-0.96.  Two questions:
  Q1 (this script): is rD ITSELF partly predictable from legal inputs?  If yes,
     the double-mutant effect is not fully irreducible and a leak-free
     prediction of rD should transfer real signal into the rescue model.
  Q2: does adding that OOF prediction (rD_pred) as a feature improve the
     v1+v2 strong 3-way headline (+28.91% / R2 0.3975)?

Method (leak-free, exchangeable unit = design):
  * rD target computed per sample over the design region (same formula scope
    as the rescue factor; training-set labels only).
  * Auxiliary strong GBDT (300 trees, depth 6) trained design-level LOO on the
    LEGAL v1+v2+transfer features -> OOF rD_pred (never sees held-out labels).
  * rD_pred appended to the legal stack; strong 3-way (L1-LGB + L2-LGB +
    Ridge, 0.6/0.3/0.1) re-run at design-level LOO.
  * Reports rD predictability (corr/R2/MAE) + the rescue-skill gain with
    LOO-exclusion robustness.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_features_v2 as m2rf2
import m2r_transfer_v1 as tr

SEED = 20260817
W1, W2, W3 = 0.6, 0.3, 0.1
N_TREES, DEPTH = 300, 6


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _prof(p):
    return np.array([x if x is not None and np.isfinite(x) else np.nan
                     for x in p], dtype=np.float64)


def _region_mask(n, sub_start, sub_end):
    m = np.zeros(n, dtype=bool)
    lo = max(sub_start - 1, 0) if sub_start is not None else 0
    hi = sub_end if sub_end is not None else n
    m[lo:hi] = True
    return m


def rmsd_double_wt(s):
    """design-region RMSD(double, WT) = rD (training-only label)."""
    n = len(s.wt_reactivity)
    mask = _region_mask(n, s.sub_start, s.sub_end)
    wt = _prof(s.wt_reactivity)
    rd = _prof(s.double_reactivity)
    m = np.isfinite(wt) & np.isfinite(rd) & mask
    if m.sum() < 3:
        return np.nan
    return float(np.sqrt(np.mean((wt[m] - rd[m]) ** 2)))


def _loo_lgb(X, y, keys, des_list, obj, seed=SEED):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        g = lgb.LGBMRegressor(
            n_estimators=N_TREES, max_depth=DEPTH, num_leaves=31,
            learning_rate=0.05, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, objective=obj, random_state=seed, verbose=-1,
            n_jobs=2)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    return preds


def _loo_ridge(X, y, keys, des_list):
    from sklearn.linear_model import Ridge
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        r = Ridge(alpha=1.0).fit(X[m], y[m])
        preds[~m] = r.predict(X[~m])
    return preds


def run_lever(X1, X2, X_tr, rD, y, keys, args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    des_list = sorted(set(keys.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    X_base = np.concatenate([X1, X2, X_tr], axis=1)   # v1+v2+transfer (258)
    t0 = time.time()

    # ---- Q1: leak-free OOF prediction of rD from legal features ----
    rD_fin = np.where(np.isfinite(rD), rD, np.nan)
    valid = np.isfinite(rD_fin)
    rD_pred = np.full(len(y), np.nan)
    rD_pred[valid] = _loo_lgb(X_base[valid], rD_fin[valid],
                              np.array(keys)[valid],
                              sorted(set(np.array(keys)[valid].tolist())),
                              "regression")
    corr = float(np.corrcoef(rD_fin[valid], rD_pred[valid])[0, 1])
    r2_rd = _r2(rD_fin[valid], rD_pred[valid])
    print(f"[rdp] rD predictability: corr={corr:.4f} R2={r2_rd:.4f} "
          f"n_valid={valid.sum()} wall={time.time()-t0:.0f}s", flush=True)
    # normalize rD_pred to a stable feature
    rD_pred_std = (rD_pred - np.nanmean(rD_pred)) / (np.nanstd(rD_pred) + 1e-9)
    rD_pred_std = np.where(np.isfinite(rD_pred_std), rD_pred_std, 0.0)

    X_aug = np.concatenate([X_base, rD_pred_std[:, None]], axis=1)  # 259

    # ---- Q2: strong 3-way on base vs base+rD_pred ----
    p_l1 = _loo_lgb(X_base, y, keys, des_list, "l1")
    p_l2 = _loo_lgb(X_base, y, keys, des_list, "regression")
    p_r = _loo_ridge(X_base, y, keys, des_list)
    blend_base = W1 * p_l1 + W2 * p_l2 + W3 * p_r
    print(f"[rdp] base 3-way skill={_skill(_mae(y, blend_base), mae_bl):+.4f} "
          f"R2={_r2(y, blend_base):.4f} wall={time.time()-t0:.0f}s", flush=True)

    a_l1 = _loo_lgb(X_aug, y, keys, des_list, "l1")
    a_l2 = _loo_lgb(X_aug, y, keys, des_list, "regression")
    a_r = _loo_ridge(X_aug, y, keys, des_list)
    blend_aug = W1 * a_l1 + W2 * a_l2 + W3 * a_r
    print(f"[rdp] aug 3-way skill={_skill(_mae(y, blend_aug), mae_bl):+.4f} "
          f"R2={_r2(y, blend_aug):.4f} wall={time.time()-t0:.0f}s", flush=True)

    # LOO-exclusion gain (aug vs base)
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        gains.append(_skill(_mae(y[m], blend_aug[m]), mae_bl) -
                     _skill(_mae(y[m], blend_base[m]), mae_bl))
    gains = np.array(gains)

    report = {
        "schema": "reactflow_delta.m2r_doublemut_pred.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "n_features": {"base": int(X_base.shape[1]), "aug": int(X_aug.shape[1])},
        "seed": SEED, "baseline_mae": mae_bl,
        "rd_predictability": {
            "corr": corr, "r2": r2_rd, "n_valid": int(valid.sum()),
            "method": "strong GBDT design-level LOO on legal v1+v2+transfer",
        },
        "results": {
            "base_3way": {"mae": _mae(y, blend_base),
                          "skill": _skill(_mae(y, blend_base), mae_bl),
                          "r2": _r2(y, blend_base)},
            "aug_rD_3way": {"mae": _mae(y, blend_aug),
                            "skill": _skill(_mae(y, blend_aug), mae_bl),
                            "r2": _r2(y, blend_aug)},
        },
        "rD_gain": {
            "pooled_gain_pp": float((_skill(_mae(y, blend_aug), mae_bl) -
                                     _skill(_mae(y, blend_base), mae_bl)) * 100),
            "r2_gain": float(_r2(y, blend_aug) - _r2(y, blend_base)),
            "loo_exclusion": {
                "gain_mean_pp": float(gains.mean() * 100),
                "gain_min_pp": float(gains.min() * 100),
                "gain_max_pp": float(gains.max() * 100),
                "pct_positive": float((gains > 0).mean()),
                "n_folds": int(len(gains)),
            },
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_doublemut_pred_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_doublemut_pred_oof.npz",
             blend_base=blend_base, blend_aug=blend_aug,
             rD_pred=rD_pred, rD=rD, y=y, keys=keys)
    g = report["rD_gain"]
    print(f"\n=== M2R rD auxiliary-predictor lever (design-level LOO) ===")
    print(f"rD predictability: corr={corr:.4f} R2={r2_rd:.4f}")
    print(f"base 3-way: skill={report['results']['base_3way']['skill']:+.4f} "
          f"R2={report['results']['base_3way']['r2']:.4f}")
    print(f"aug  rD   : skill={report['results']['aug_rD_3way']['skill']:+.4f} "
          f"R2={report['results']['aug_rD_3way']['r2']:.4f}")
    print(f"gain: {g['pooled_gain_pp']:+.2f}pp (R2 {g['r2_gain']:+.4f})")
    loo = g["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_doublemut_pred_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X1, y, keys, _ = m2rf.build_all(samples)
    X2, _ = m2rf2.build_all_v2(samples)
    keys = np.array(keys)
    rD = np.array([rmsd_double_wt(s) for s in samples], dtype=np.float64)

    m2_oof = tr.load_m2_oof(args.m2_pred)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    print(f"[rdp] n={len(y)} X1={X1.shape} X2={X2.shape} X_tr={X_tr.shape} "
          f"rD_finite={np.isfinite(rD).mean():.3f}", flush=True)

    run_lever(X1, X2, X_tr, rD, y, keys, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
