#!/usr/bin/env python3
"""m2r_multiseed_v1.py — multi-seed averaging of the v1+v2 strong 3-way.

MOTIVATION (method-level, variance reduction):
The current headline (strong 3-way + v2, +28.91% / R2 0.3975) trains each
base GBDT with a single random_state.  GBDTs with subsample/colsample
stochasticity have seed-level variance; averaging OOF predictions over K seeds
reduces that variance and typically gives a small, robust, method-level gain
— orthogonal to every feature and ensemble lever already closed.

Method (design-level LOO, exchangeable unit = design):
  * strong L1-LGB and L2-LGB (300 tr, depth 6) over K seeds
  * Ridge (deterministic)
  * averaged seed OOF -> 3-way blend (0.6/0.3/0.1)
  * compares vs single-seed (K=1) baseline (+28.91%)
  * LOO-exclusion gain + per-design, saves OOF for permtest
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

SEED0 = 20260817
W1, W2, W3 = 0.6, 0.3, 0.1
N_TREES, DEPTH = 300, 6
SEEDS = [20260817, 20260818, 20260819, 20260820, 20260821]


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _loo_lgb_seed(X, y, keys, des_list, obj, seed):
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


def run_multiseed(X, y, keys, args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    des_list = sorted(set(keys.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    t0 = time.time()

    p_r = _loo_ridge(X, y, keys, des_list)

    l1_seeds = []
    l2_seeds = []
    for k, seed in enumerate(SEEDS):
        p = _loo_lgb_seed(X, y, keys, des_list, "l1", seed)
        l1_seeds.append(p)
        print(f"[ms] L1 seed {seed} skill={_skill(_mae(y, p), mae_bl):+.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)
        p = _loo_lgb_seed(X, y, keys, des_list, "regression", seed)
        l2_seeds.append(p)
        print(f"[ms] L2 seed {seed} skill={_skill(_mae(y, p), mae_bl):+.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)

    l1_seeds = np.array(l1_seeds)
    l2_seeds = np.array(l2_seeds)

    # single-seed baseline (K=1, headline seed)
    blend_1 = W1 * l1_seeds[0] + W2 * l2_seeds[0] + W3 * p_r

    # multi-seed: average each objective over K seeds, then 3-way blend
    l1_avg = l1_seeds.mean(axis=0)
    l2_avg = l2_seeds.mean(axis=0)
    blend_K = W1 * l1_avg + W2 * l2_avg + W3 * p_r

    # also: per-design and LOO-exclusion gain (multi-seed vs single-seed)
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        gains.append(_skill(_mae(y[m], blend_K[m]), mae_bl) -
                     _skill(_mae(y[m], blend_1[m]), mae_bl))
    gains = np.array(gains)

    per_seed_skill = {"l1": [_skill(_mae(y, p), mae_bl) for p in l1_seeds],
                      "l2": [_skill(_mae(y, p), mae_bl) for p in l2_seeds]}

    report = {
        "schema": "reactflow_delta.m2r_multiseed.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "n_features": int(X.shape[1]), "k_seeds": len(SEEDS), "seeds": SEEDS,
        "baseline_mae": mae_bl,
        "per_seed_skill": per_seed_skill,
        "results": {
            "single_seed_3way": {"mae": _mae(y, blend_1),
                                 "skill": _skill(_mae(y, blend_1), mae_bl),
                                 "r2": _r2(y, blend_1)},
            "multiseed_3way": {"mae": _mae(y, blend_K),
                               "skill": _skill(_mae(y, blend_K), mae_bl),
                               "r2": _r2(y, blend_K)},
        },
        "multiseed_gain": {
            "pooled_gain_pp": float((_skill(_mae(y, blend_K), mae_bl) -
                                     _skill(_mae(y, blend_1), mae_bl)) * 100),
            "r2_gain": float(_r2(y, blend_K) - _r2(y, blend_1)),
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
    (out / "m2r_multiseed_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_multiseed_oof.npz",
             blend_1=blend_1, blend_K=blend_K, y=y, keys=keys)
    g = report["multiseed_gain"]
    print(f"\n=== M2R multi-seed averaging (design-level LOO, K={len(SEEDS)}) ===")
    print(f"single-seed: skill={report['results']['single_seed_3way']['skill']:+.4f} "
          f"R2={report['results']['single_seed_3way']['r2']:.4f}")
    print(f"multi-seed : skill={report['results']['multiseed_3way']['skill']:+.4f} "
          f"R2={report['results']['multiseed_3way']['r2']:.4f}")
    print(f"gain: {g['pooled_gain_pp']:+.2f}pp (R2 {g['r2_gain']:+.4f})")
    loo = g["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> {out / 'm2r_multiseed_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-seeds", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global SEEDS
    SEEDS = SEEDS[:args.k_seeds]

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X1, y, keys, _ = m2rf.build_all(samples)
    X2, _ = m2rf2.build_all_v2(samples)
    keys = np.array(keys)
    m2_oof = tr.load_m2_oof(args.m2_pred)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    X = np.concatenate([X1, X2, X_tr], axis=1)
    print(f"[ms] n={len(y)} X={X.shape} K={len(SEEDS)}", flush=True)

    run_multiseed(X, y, keys, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
