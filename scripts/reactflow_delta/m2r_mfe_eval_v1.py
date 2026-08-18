#!/usr/bin/env python3
"""m2r_mfe_eval_v1.py — screen the ViennaRNA MFE legal feature modality.

Adds the thermodynamic (sequence-folding) features from m2r_mfe_features_v1 to
the v1+v2+transfer stack and measures the strong-3way gain at design-level LOO:

  * base   = v1+v2+transfer (258 dims) -> strong 3-way (single seed)
  * mfe_aug = base + MFE (292 dims)    -> strong 3-way
  * mfe_only = MFE (34 dims) alone      -> strong 3-way (is thermodynamics
    predictive by itself?  scientific diagnostic)

If mfe_aug shows a real gain, we run the full multi-seed + puzzle audit.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_features_v2 as m2rf2
import m2r_mfe_features_v1 as mfe
import m2r_transfer_v1 as tr

SEED = 20260817
W1, W2, W3 = 0.6, 0.3, 0.1
CFG = dict(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
           min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _loo_lgb(X, y, keys, des_list, obj):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        g = lgb.LGBMRegressor(objective=obj, random_state=SEED, verbose=-1,
                              n_jobs=2, **CFG)
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


def _threeway(X, y, keys, des_list):
    p_l1 = _loo_lgb(X, y, keys, des_list, "l1")
    p_l2 = _loo_lgb(X, y, keys, des_list, "regression")
    p_r = _loo_ridge(X, y, keys, des_list)
    return W1 * p_l1 + W2 * p_l2 + W3 * p_r


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
    XM, _ = mfe.build_all_mfe(samples)
    keys = np.array(keys)

    m2_oof = tr.load_m2_oof(args.m2_pred)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)

    X_base = np.concatenate([X1, X2, X_tr], axis=1)
    X_mfe = np.concatenate([X1, X2, X_tr, XM], axis=1)
    des_list = sorted(set(keys.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[mfe] n={len(y)} base={X_base.shape} mfe_aug={X_mfe.shape} "
          f"mfe_only={XM.shape}", flush=True)

    t0 = time.time()
    b = _threeway(X_base, y, keys, des_list)
    print(f"[mfe] base 3-way skill={_skill(_mae(y, b), mae_bl):+.4f} "
          f"R2={_r2(y, b):.4f} wall={time.time()-t0:.0f}s", flush=True)
    a = _threeway(X_mfe, y, keys, des_list)
    print(f"[mfe] mfe_aug 3-way skill={_skill(_mae(y, a), mae_bl):+.4f} "
          f"R2={_r2(y, a):.4f} wall={time.time()-t0:.0f}s", flush=True)
    m = _threeway(XM, y, keys, des_list)
    print(f"[mfe] mfe_only 3-way skill={_skill(_mae(y, m), mae_bl):+.4f} "
          f"R2={_r2(y, m):.4f} wall={time.time()-t0:.0f}s", flush=True)

    # LOO-exclusion gain (mfe_aug vs base)
    gains = []
    for held in des_list:
        mm = keys != held
        if mm.sum() < 10:
            continue
        gains.append(_skill(_mae(y[mm], a[mm]), mae_bl) -
                     _skill(_mae(y[mm], b[mm]), mae_bl))
    gains = np.array(gains)

    report = {
        "schema": "reactflow_delta.m2r_mfe_eval.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "n_features": {"base": int(X_base.shape[1]),
                       "mfe_aug": int(X_mfe.shape[1]),
                       "mfe_only": int(XM.shape[1])},
        "seed": SEED, "cfg": CFG,
        "headline_weights": {"w1": W1, "w2": W2, "w3": W3},
        "baseline_mae": mae_bl,
        "mfe_feature_names": mfe.mfe_feature_names(),
        "results": {
            "base_3way": {"mae": _mae(y, b), "skill": _skill(_mae(y, b), mae_bl),
                          "r2": _r2(y, b)},
            "mfe_aug_3way": {"mae": _mae(y, a), "skill": _skill(_mae(y, a), mae_bl),
                             "r2": _r2(y, a)},
            "mfe_only_3way": {"mae": _mae(y, m), "skill": _skill(_mae(y, m), mae_bl),
                              "r2": _r2(y, m)},
        },
        "mfe_aug_gain": {
            "pooled_gain_pp": float((_skill(_mae(y, a), mae_bl) -
                                     _skill(_mae(y, b), mae_bl)) * 100),
            "r2_gain": float(_r2(y, a) - _r2(y, b)),
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
    (out / "m2r_mfe_eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_mfe_eval_oof.npz", base=b, mfe_aug=a, mfe_only=m,
             y=y, keys=keys)
    print(f"\n=== M2R MFE thermodynamic features ===")
    print(f"base 3-way : skill={report['results']['base_3way']['skill']:+.4f} "
          f"R2={report['results']['base_3way']['r2']:.4f}")
    print(f"mfe_aug    : skill={report['results']['mfe_aug_3way']['skill']:+.4f} "
          f"R2={report['results']['mfe_aug_3way']['r2']:.4f}")
    print(f"mfe_only   : skill={report['results']['mfe_only_3way']['skill']:+.4f} "
          f"R2={report['results']['mfe_only_3way']['r2']:.4f}")
    g = report["mfe_aug_gain"]
    print(f"mfe_aug gain: {g['pooled_gain_pp']:+.2f}pp (R2 {g['r2_gain']:+.4f})")
    loo = g["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_mfe_eval_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
