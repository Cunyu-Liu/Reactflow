#!/usr/bin/env python3
"""m2r_features_v2_puzzle_v1.py — puzzle-level leak-free v1+v2 strong 3-way.

Audits the v2 feature-group gain (cross-mutant disruption magnitudes + stem
context) at the PUZZLE level: train on 19 puzzles -> predict the held-out
puzzle, with PUZZLE-LEVEL M2 OOF transfer features (leak-free, from
run_response_spectrum_m2_attn_puzzle_v1).  Confirms the v1+v2 strong-3-way gain
(+0.80pp at design level) survives complete puzzle holdout.

Also reports: puzzle-block permutation p for the v1+v2 strong 3-way, per-puzzle
v2-vs-v1 gain (100%-positive check over 20 puzzles).
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
    X1, y, keys, _ = m2rf.build_all(samples)
    X2, _ = m2rf2.build_all_v2(samples)
    sample_puzzles = np.array([s.puzzle for s in samples])
    puzzles = sorted(set(sample_puzzles.tolist()))

    m2_oof = tr.load_m2_oof(args.m2_pred_puzzle)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    X1c = np.concatenate([X1, X_tr], axis=1)
    X12c = np.concatenate([X1, X2, X_tr], axis=1)
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[v2pz] n={len(y)} puzzles={len(puzzles)} X1={X1c.shape} "
          f"X12={X12c.shape} transfer_nonzero={(np.abs(X_tr).sum(axis=1) > 0).mean():.3f}",
          flush=True)

    t0 = time.time()
    # v1 (reproduce +27.42% at puzzle level)
    p1_l1 = _loo_lgb(X1c, y, sample_puzzles, puzzles, "l1")
    p1_l2 = _loo_lgb(X1c, y, sample_puzzles, puzzles, "regression")
    p1_r = _loo_ridge(X1c, y, sample_puzzles, puzzles)
    blend1 = W1 * p1_l1 + W2 * p1_l2 + W3 * p1_r
    print(f"[v2pz] v1 skill={_skill(_mae(y, blend1), mae_bl):+.4f} "
          f"R2={_r2(y, blend1):.4f} wall={time.time()-t0:.0f}s", flush=True)

    # v1+v2
    p2_l1 = _loo_lgb(X12c, y, sample_puzzles, puzzles, "l1")
    p2_l2 = _loo_lgb(X12c, y, sample_puzzles, puzzles, "regression")
    p2_r = _loo_ridge(X12c, y, sample_puzzles, puzzles)
    blend2 = W1 * p2_l1 + W2 * p2_l2 + W3 * p2_r
    print(f"[v2pz] v1+v2 skill={_skill(_mae(y, blend2), mae_bl):+.4f} "
          f"R2={_r2(y, blend2):.4f} wall={time.time()-t0:.0f}s", flush=True)

    # per-puzzle gain (v1+v2 vs v1)
    pp = {}
    gains = []
    for p in puzzles:
        m = sample_puzzles == p
        if m.sum() == 0:
            continue
        s_1 = _skill(_mae(y[m], blend1[m]), y_med)
        s_2 = _skill(_mae(y[m], blend2[m]), y_med)
        pp[p] = {"n": int(m.sum()), "v1_skill": float(s_1),
                 "v1v2_skill": float(s_2), "gain_pp": float((s_2 - s_1) * 100)}
        gains.append(s_2 - s_1)
    gains = np.array(gains)

    # puzzle-block permutation on pooled skill (shuffle y within puzzles)
    rng = np.random.default_rng(SEED)
    skill_2_pool = _skill(_mae(y, blend2), mae_bl)
    cnt = 0
    for _ in range(args.n_perm):
        perm_y = np.empty_like(y)
        for p in puzzles:
            m = sample_puzzles == p
            perm_y[m] = rng.permutation(y[m])
        sk = _skill(_mae(perm_y, blend2), mae_bl)
        if sk >= skill_2_pool:
            cnt += 1
    perm_p = (cnt + 1) / (args.n_perm + 1)

    report = {
        "schema": "reactflow_delta.m2r_features_v2_puzzle.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "puzzle",
        "n_samples": int(len(y)), "n_puzzles": len(puzzles),
        "n_features": {"v1": int(X1c.shape[1]), "v1_v2": int(X12c.shape[1])},
        "seed": SEED,
        "headline_weights": {"w1": W1, "w2": W2, "w3": W3},
        "results": {
            "v1_3way": {"mae": _mae(y, blend1), "skill": _skill(_mae(y, blend1), mae_bl),
                        "r2": _r2(y, blend1)},
            "v1_v2_3way": {"mae": _mae(y, blend2), "skill": _skill(_mae(y, blend2), mae_bl),
                           "r2": _r2(y, blend2)},
        },
        "per_puzzle": pp,
        "per_puzzle_gain_vs_v1": {
            "gain_mean_pp": float(gains.mean() * 100),
            "gain_min_pp": float(gains.min() * 100),
            "gain_max_pp": float(gains.max() * 100),
            "pct_positive": float((gains > 0).mean()),
            "n_puzzles": int(len(gains)),
        },
        "permutation_p": float(perm_p),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_features_v2_puzzle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_features_v2_puzzle_oof.npz",
             blend1=blend1, blend2=blend2, y=y, puzzles=sample_puzzles)
    g = report["per_puzzle_gain_vs_v1"]
    print(f"\n=== M2R v2 feature-group puzzle-level ===")
    print(f"v1:    skill={report['results']['v1_3way']['skill']:+.4f} "
          f"R2={report['results']['v1_3way']['r2']:.4f}")
    print(f"v1+v2: skill={report['results']['v1_v2_3way']['skill']:+.4f} "
          f"R2={report['results']['v1_v2_3way']['r2']:.4f}")
    print(f"per-puzzle gain: mean={g['gain_mean_pp']:+.2f}pp "
          f"range=[{g['gain_min_pp']:+.2f},{g['gain_max_pp']:+.2f}]pp "
          f"pct_pos={g['pct_positive']:.3f}")
    print(f"perm p = {perm_p:.4f}")
    print(f"DONE -> {out / 'm2r_features_v2_puzzle_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
