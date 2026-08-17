#!/usr/bin/env python3
"""m2r_4way_puzzle_v1.py — puzzle-level leak-free architecture-decorrelated
4-way ensemble (L1-LGB + L2-LGB + XGB-L2 + Ridge).

Audits the 4-way a-priori blend (0.45/0.25/0.20/0.10) at the PUZZLE level:
train on 19 puzzles -> predict the held-out puzzle, with PUZZLE-LEVEL M2 OOF
transfer features (leak-free, from run_response_spectrum_m2_attn_puzzle_v1).
Confirms the 4-way gain over the strong 3-way (0.6/0.3/0.1) survives complete
puzzle holdout.

Also reports: puzzle-block permutation p for the 4-way, per-puzzle 4-way-vs-
strong-3way gain (100%-positive check over 20 puzzles).
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
W1, W2, WX, WR = 0.45, 0.25, 0.20, 0.10
W3_1, W3_2, W3_3 = 0.6, 0.3, 0.1


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    return 1.0 - float(np.sum((y - p) ** 2)) / float(np.sum((y - y.mean()) ** 2))


def _loo_lgb(X, y, pz, puzzles, obj):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in puzzles:
        m = pz != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        g = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective=obj, random_state=SEED, verbose=-1, n_jobs=2)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    return preds


def _loo_xgb(X, y, pz, puzzles):
    from xgboost import XGBRegressor
    preds = np.zeros(len(y))
    for held in puzzles:
        m = pz != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        g = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=2, verbosity=0)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    return preds


def _loo_ridge(X, y, pz, puzzles):
    from sklearn.linear_model import Ridge
    preds = np.zeros(len(y))
    for held in puzzles:
        m = pz != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        r = Ridge(alpha=1.0).fit(X[m], y[m])
        preds[~m] = r.predict(X[~m])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred-puzzle", required=True,
                    help="PUZZLE-level M2 keyed_predictions jsonl (leak-free)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    sample_puzzles = np.array([s.puzzle for s in samples])
    puzzles = sorted(set(sample_puzzles.tolist()))

    m2_oof = tr.load_m2_oof(args.m2_pred_puzzle)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    X_comb = np.concatenate([X, X_tr], axis=1)
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[4waypz] n={len(y)} puzzles={len(puzzles)} X={X_comb.shape} "
          f"transfer_nonzero={(np.abs(X_tr).sum(axis=1) > 0).mean():.3f}",
          flush=True)

    t0 = time.time()
    ps_l1 = _loo_lgb(X_comb, y, sample_puzzles, puzzles, "l1")
    print(f"[4waypz] L1-LGB skill={_skill(_mae(y, ps_l1), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    ps_l2 = _loo_lgb(X_comb, y, sample_puzzles, puzzles, "regression")
    print(f"[4waypz] L2-LGB skill={_skill(_mae(y, ps_l2), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    ps_xg = _loo_xgb(X_comb, y, sample_puzzles, puzzles)
    print(f"[4waypz] XGB-L2 skill={_skill(_mae(y, ps_xg), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    ps_r = _loo_ridge(X_comb, y, sample_puzzles, puzzles)
    print(f"[4waypz] Ridge skill={_skill(_mae(y, ps_r), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    blend_4 = W1 * ps_l1 + W2 * ps_l2 + WX * ps_xg + WR * ps_r
    blend_3 = W3_1 * ps_l1 + W3_2 * ps_l2 + W3_3 * ps_r
    print(f"[4waypz] strong3 skill={_skill(_mae(y, blend_3), mae_bl):+.4f} "
          f"R2={_r2(y, blend_3):.4f}", flush=True)
    print(f"[4waypz] fourway  skill={_skill(_mae(y, blend_4), mae_bl):+.4f} "
          f"R2={_r2(y, blend_4):.4f} wall={time.time()-t0:.0f}s", flush=True)

    # per-puzzle gain (4-way vs strong 3-way)
    pp = {}
    gains = []
    for p in puzzles:
        m = sample_puzzles == p
        if m.sum() == 0:
            continue
        s_3 = _skill(_mae(y[m], blend_3[m]), y_med)
        s_4 = _skill(_mae(y[m], blend_4[m]), y_med)
        pp[p] = {"n": int(m.sum()), "strong3_skill": float(s_3),
                 "fourway_skill": float(s_4), "gain_pp": float((s_4 - s_3) * 100)}
        gains.append(s_4 - s_3)
    gains = np.array(gains)

    # puzzle-block permutation on pooled skill (shuffle y within puzzles)
    rng = np.random.default_rng(SEED)
    skill_4_pool = _skill(_mae(y, blend_4), mae_bl)
    cnt = 0
    for _ in range(args.n_perm):
        perm_y = np.empty_like(y)
        for p in puzzles:
            m = sample_puzzles == p
            perm_y[m] = rng.permutation(y[m])
        sk = _skill(_mae(perm_y, blend_4), mae_bl)
        if sk >= skill_4_pool:
            cnt += 1
    perm_p = (cnt + 1) / (args.n_perm + 1)

    report = {
        "schema": "reactflow_delta.m2r_4way_puzzle.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "puzzle",
        "n_samples": int(len(y)), "n_puzzles": len(puzzles),
        "n_features": int(X_comb.shape[1]), "seed": SEED,
        "headline_weights": {"w1": W1, "w2": W2, "wx": WX, "wr": WR},
        "results": {
            "strong_3way": {"mae": _mae(y, blend_3), "skill": _skill(_mae(y, blend_3), mae_bl),
                            "r2": _r2(y, blend_3)},
            "fourway_a_priori": {"mae": _mae(y, blend_4), "skill": _skill(_mae(y, blend_4), mae_bl),
                                 "r2": _r2(y, blend_4)},
        },
        "per_puzzle": pp,
        "per_puzzle_gain_vs_strong_3way": {
            "gain_mean_pp": float(gains.mean() * 100),
            "gain_min_pp": float(gains.min() * 100),
            "gain_max_pp": float(gains.max() * 100),
            "pct_positive": float((gains > 0).mean()),
            "n_puzzles": int(len(gains)),
        },
        "permutation_p": float(perm_p),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_4way_puzzle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_4way_puzzle_oof.npz",
             blend_3=blend_3, blend_4=blend_4, y=y, puzzles=sample_puzzles)
    g = report["per_puzzle_gain_vs_strong_3way"]
    print(f"\n=== architecture-decorrelated 4-way puzzle-level ===")
    print(f"strong3: skill={report['results']['strong_3way']['skill']:+.4f} "
          f"R2={report['results']['strong_3way']['r2']:.4f}")
    print(f"fourway: skill={report['results']['fourway_a_priori']['skill']:+.4f} "
          f"R2={report['results']['fourway_a_priori']['r2']:.4f}")
    print(f"per-puzzle gain: mean={g['gain_mean_pp']:+.2f}pp "
          f"range=[{g['gain_min_pp']:+.2f},{g['gain_max_pp']:+.2f}]pp "
          f"pct_pos={g['pct_positive']:.3f}")
    print(f"perm p = {perm_p:.4f}")
    print(f"DONE -> {out / 'm2r_4way_puzzle_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
