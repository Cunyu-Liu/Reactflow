#!/usr/bin/env python3
"""m2r_stack_soup_v1.py — config-soup BONUS lever (reuses saved base OOF).

Loads the base OOF columns saved by m2r_stack_v1.py (l1_avg, l2_avg, ridge,
y, keys) and computes the CFG_B columns (500 trees / depth 8 / 127 leaves)
to form a config-soup blend (0.5 x CFG_A + 0.5 x CFG_B per objective).  Tests
whether additional capacity beyond the 300/depth6 strong base still helps —
runs LAST so it can never block the main stacking / residual analysis.
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

W1, W2, W3 = 0.6, 0.3, 0.1
SEEDS = [20260817, 20260818, 20260819, 20260820, 20260821]
CFG_B = dict(n_estimators=500, max_depth=8, num_leaves=127, learning_rate=0.03,
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


def _loo_lgb_cfg(X, y, keys, des_list, obj, cfg, seed):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        g = lgb.LGBMRegressor(objective=obj, random_state=seed, verbose=-1,
                              n_jobs=2, **cfg)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-npz", required=True)
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.base_npz)
    l1_avg = z["l1_avg"]; l2_avg = z["l2_avg"]
    p_r = z["ridge"]; y = z["y"]; keys = np.asarray(z["keys"])
    des_list = sorted(set(keys.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X1, _, _, _ = m2rf.build_all(samples)
    X2, _ = m2rf2.build_all_v2(samples)
    m2_oof = tr.load_m2_oof(args.m2_pred)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    X = np.concatenate([X1, X2, X_tr], axis=1)
    print(f"[soup] n={len(y)} X={X.shape}", flush=True)

    t0 = time.time()
    l1B = np.stack([_loo_lgb_cfg(X, y, keys, des_list, "l1", CFG_B, s)
                    for s in SEEDS], axis=1)
    l2B = np.stack([_loo_lgb_cfg(X, y, keys, des_list, "regression", CFG_B, s)
                    for s in SEEDS], axis=1)
    l1_b = l1B.mean(axis=1); l2_b = l2B.mean(axis=1)
    l1_soup = 0.5 * l1_avg + 0.5 * l1_b
    l2_soup = 0.5 * l2_avg + 0.5 * l2_b
    blend_fixed = W1 * l1_avg + W2 * l2_avg + W3 * p_r
    blend_soup = W1 * l1_soup + W2 * l2_soup + W3 * p_r
    blend_B = W1 * l1_b + W2 * l2_b + W3 * p_r

    def _blend_loo(bA, bB, name):
        gains = []
        for held in des_list:
            m = keys != held
            if m.sum() < 10:
                continue
            gains.append(_skill(_mae(y[m], bB[m]), mae_bl) -
                         _skill(_mae(y[m], bA[m]), mae_bl))
        gains = np.array(gains)
        return {
            "pooled_gain_pp": float((_skill(_mae(y, bB), mae_bl) -
                                     _skill(_mae(y, bA), mae_bl)) * 100),
            "r2_gain": float(_r2(y, bB) - _r2(y, bA)),
            "loo_exclusion": {
                "gain_mean_pp": float(gains.mean() * 100),
                "gain_min_pp": float(gains.min() * 100),
                "gain_max_pp": float(gains.max() * 100),
                "pct_positive": float((gains > 0).mean()),
                "n_folds": int(len(gains)),
            },
        }

    report = {
        "schema": "reactflow_delta.m2r_stack_soup.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "k_seeds": len(SEEDS), "seeds": SEEDS,
        "cfg_B": CFG_B,
        "baseline_mae": mae_bl,
        "results": {
            "l1_cfgA": _skill(_mae(y, l1_avg), mae_bl),
            "l1_cfgB": _skill(_mae(y, l1_b), mae_bl),
            "l2_cfgA": _skill(_mae(y, l2_avg), mae_bl),
            "l2_cfgB": _skill(_mae(y, l2_b), mae_bl),
            "fixed_3way": {"mae": _mae(y, blend_fixed),
                           "skill": _skill(_mae(y, blend_fixed), mae_bl),
                           "r2": _r2(y, blend_fixed)},
            "cfgB_only_3way": {"mae": _mae(y, blend_B),
                               "skill": _skill(_mae(y, blend_B), mae_bl),
                               "r2": _r2(y, blend_B)},
            "soup_3way": {"mae": _mae(y, blend_soup),
                          "skill": _skill(_mae(y, blend_soup), mae_bl),
                          "r2": _r2(y, blend_soup)},
        },
        "soup_vs_fixed": _blend_loo(blend_fixed, blend_soup, "soup"),
        "cfgB_vs_fixed": _blend_loo(blend_fixed, blend_B, "cfgB"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_stack_soup_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_stack_soup_oof.npz",
             l1_avg=l1_avg, l2_avg=l2_avg, l1_b=l1_b, l2_b=l2_b,
             blend_fixed=blend_fixed, blend_soup=blend_soup, y=y, keys=keys)
    print(f"\n=== M2R config-soup (CFG_A + CFG_B) ===")
    print(f"l1: cfgA={report['results']['l1_cfgA']:+.4f} "
          f"cfgB={report['results']['l1_cfgB']:+.4f}")
    print(f"l2: cfgA={report['results']['l2_cfgA']:+.4f} "
          f"cfgB={report['results']['l2_cfgB']:+.4f}")
    print(f"fixed : skill={report['results']['fixed_3way']['skill']:+.4f} "
          f"R2={report['results']['fixed_3way']['r2']:.4f}")
    print(f"cfgB  : skill={report['results']['cfgB_only_3way']['skill']:+.4f} "
          f"R2={report['results']['cfgB_only_3way']['r2']:.4f}")
    print(f"soup  : skill={report['results']['soup_3way']['skill']:+.4f} "
          f"R2={report['results']['soup_3way']['r2']:.4f}")
    sv = report["soup_vs_fixed"]
    print(f"soup gain: {sv['pooled_gain_pp']:+.2f}pp "
          f"LOO=[{sv['loo_exclusion']['gain_min_pp']:+.2f},"
          f"{sv['loo_exclusion']['gain_max_pp']:+.2f}] "
          f"pct_pos={sv['loo_exclusion']['pct_positive']:.3f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_stack_soup_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
