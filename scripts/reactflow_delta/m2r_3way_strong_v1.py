#!/usr/bin/env python3
"""m2r_3way_strong_v1.py — strong-base 3-way ensemble on the full 236-dim stack.

MOTIVATION (method-level, from m2r_ceiling_audit_v1.py):
The ceiling audit showed the default GBDT (100 trees, depth 3) leaves a small
but real capacity gain on the table: on legal features, strong GBDT (300 trees,
depth 6, lr 0.05) reached R2 0.3614 vs 0.3539 default (+0.0075).  This script
tests whether upgrading BOTH base GBDTs in the cross-objective 3-way ensemble
(L1+L2 GBDT + Ridge) on the FULL 236-dim stack (230 M2R feats incl. M2_structure
+ 6 M2-transfer) gives a real headline gain over the current 100-tree 3-way.

Cells (design-level LOO, exchangeable unit = design):
  * default 3-way (L1/L2 @ 100 trees, depth 3)  -> reproduces +26.59% / R2 0.370
  * strong  3-way (L1/L2 @ 300 trees, depth 6)  -> the method-level candidate
Weights stay FIXED A-PRIORI at (0.6, 0.3, 0.1) for both (no selection).
LOO-exclusion gain (strong vs default 3-way) is reported over 159 folds.
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
W1, W2, W3 = 0.6, 0.3, 0.1


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _loo_gbdt(X, y, keys, des_list, obj, strong: bool):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        if strong:
            g = lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
                min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                objective=obj, random_state=SEED, verbose=-1, n_jobs=2)
        else:
            g = lgb.LGBMRegressor(n_estimators=100, max_depth=3,
                                  objective=obj, random_state=SEED,
                                  verbose=-1, n_jobs=2)
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


def _loo_exclusion_gain(y, keys, des_list, p_new, p_old, mae_bl):
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            continue
        s_new = _skill(_mae(y[m], p_new[m]), mae_bl)
        s_old = _skill(_mae(y[m], p_old[m]), mae_bl)
        gains.append(s_new - s_old)
    gains = np.array(gains)
    return {"gain_mean_pp": float(gains.mean() * 100),
            "gain_min_pp": float(gains.min() * 100),
            "gain_max_pp": float(gains.max() * 100),
            "pct_positive": float((gains > 0).mean()),
            "n_folds": int(len(gains))}


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
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))

    m2_oof = tr.load_m2_oof(args.m2_pred)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    X_full = np.concatenate([X, X_tr], axis=1)
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[strong3way] n={len(y)} designs={len(des_list)} X={X_full.shape}",
          flush=True)

    t0 = time.time()
    # default 3-way (reproduce headline)
    p_l1_d = _loo_gbdt(X_full, y, keys, des_list, "l1", False)
    p_l2_d = _loo_gbdt(X_full, y, keys, des_list, "regression", False)
    p_r_d = _loo_ridge(X_full, y, keys, des_list)
    blend_d = W1 * p_l1_d + W2 * p_l2_d + W3 * p_r_d
    print(f"[strong3way] default 3-way skill="
          f"{_skill(_mae(y, blend_d), mae_bl):+.4f} R2={_r2(y, blend_d):.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    # strong 3-way (method candidate)
    p_l1_s = _loo_gbdt(X_full, y, keys, des_list, "l1", True)
    p_l2_s = _loo_gbdt(X_full, y, keys, des_list, "regression", True)
    p_r_s = _loo_ridge(X_full, y, keys, des_list)
    blend_s = W1 * p_l1_s + W2 * p_l2_s + W3 * p_r_s
    print(f"[strong3way] strong 3-way skill="
          f"{_skill(_mae(y, blend_s), mae_bl):+.4f} R2={_r2(y, blend_s):.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    loo_gain = _loo_exclusion_gain(y, keys, des_list, blend_s, blend_d, mae_bl)
    report = {
        "schema": "reactflow_delta.m2r_3way_strong.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "n_features": int(X_full.shape[1]), "seed": SEED,
        "weights": {"w1": W1, "w2": W2, "w3": W3},
        "default_3way": {
            "l1": {"mae": _mae(y, p_l1_d), "skill": _skill(_mae(y, p_l1_d), mae_bl),
                   "r2": _r2(y, p_l1_d)},
            "l2": {"mae": _mae(y, p_l2_d), "skill": _skill(_mae(y, p_l2_d), mae_bl),
                   "r2": _r2(y, p_l2_d)},
            "ridge": {"mae": _mae(y, p_r_d), "skill": _skill(_mae(y, p_r_d), mae_bl),
                      "r2": _r2(y, p_r_d)},
            "blend": {"mae": _mae(y, blend_d), "skill": _skill(_mae(y, blend_d), mae_bl),
                      "r2": _r2(y, blend_d)},
        },
        "strong_3way": {
            "l1": {"mae": _mae(y, p_l1_s), "skill": _skill(_mae(y, p_l1_s), mae_bl),
                   "r2": _r2(y, p_l1_s)},
            "l2": {"mae": _mae(y, p_l2_s), "skill": _skill(_mae(y, p_l2_s), mae_bl),
                   "r2": _r2(y, p_l2_s)},
            "ridge": {"mae": _mae(y, p_r_s), "skill": _skill(_mae(y, p_r_s), mae_bl),
                      "r2": _r2(y, p_r_s)},
            "blend": {"mae": _mae(y, blend_s), "skill": _skill(_mae(y, blend_s), mae_bl),
                      "r2": _r2(y, blend_s)},
        },
        "strong_vs_default_loo_exclusion": loo_gain,
        "headline": {
            "default": {"mae": _mae(y, blend_d),
                        "skill": _skill(_mae(y, blend_d), mae_bl),
                        "r2": _r2(y, blend_d)},
            "strong": {"mae": _mae(y, blend_s),
                       "skill": _skill(_mae(y, blend_s), mae_bl),
                       "r2": _r2(y, blend_s)},
        },
    }
    (out / "m2r_3way_strong_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_3way_strong_oof.npz",
             blend_d=blend_d, blend_s=blend_s, y=y, keys=keys)
    g = loo_gain
    print(f"\n=== strong-vs-default 3-way ===")
    print(f"default: skill={report['headline']['default']['skill']:+.4f} "
          f"R2={report['headline']['default']['r2']:.4f}")
    print(f"strong : skill={report['headline']['strong']['skill']:+.4f} "
          f"R2={report['headline']['strong']['r2']:.4f}")
    print(f"LOO-exclusion gain: mean={g['gain_mean_pp']:+.2f}pp "
          f"range=[{g['gain_min_pp']:+.2f},{g['gain_max_pp']:+.2f}]pp "
          f"pct_pos={g['pct_positive']:.3f}")
    print(f"DONE -> {out / 'm2r_3way_strong_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
