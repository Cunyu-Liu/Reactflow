#!/usr/bin/env python3
"""m2r_multiseed_puzzle_v1.py — puzzle-level multi-seed averaging (leak-free).

Audits the multi-seed variance-reduction lever at the PUZZLE level: train on
19 puzzles -> predict the held-out puzzle, with PUZZLE-LEVEL M2 OOF transfer
features (leak-free).  Confirms the design-level multi-seed gain
(+0.31pp, perm p=0.014, 100% LOO-exclusion positive) survives complete puzzle
holdout.

Method (puzzle-level LOO, exchangeable unit = puzzle):
  * strong L1-LGB and L2-LGB (300 tr, depth 6) over K seeds (puzzle-level LOO)
  * Ridge (deterministic)
  * averaged seed OOF -> 3-way blend (0.6/0.3/0.1)
  * compares vs single-seed (K=1) baseline at puzzle level
  * per-puzzle gain + puzzle-block permutation p
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


def _loo_lgb_seed(X, y, pz, puzzles, obj, seed):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in puzzles:
        m = pz != held
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


def run_puzzle_multiseed(X, y, pz, args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    puzzles = sorted(set(pz.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    t0 = time.time()

    p_r = _loo_ridge(X, y, pz, puzzles)
    l1_seeds = []
    l2_seeds = []
    for k, seed in enumerate(SEEDS):
        p = _loo_lgb_seed(X, y, pz, puzzles, "l1", seed)
        l1_seeds.append(p)
        print(f"[mspz] L1 seed {seed} skill={_skill(_mae(y, p), mae_bl):+.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)
        p = _loo_lgb_seed(X, y, pz, puzzles, "regression", seed)
        l2_seeds.append(p)
        print(f"[mspz] L2 seed {seed} skill={_skill(_mae(y, p), mae_bl):+.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)

    l1_seeds = np.array(l1_seeds)
    l2_seeds = np.array(l2_seeds)

    blend_1 = W1 * l1_seeds[0] + W2 * l2_seeds[0] + W3 * p_r
    l1_avg = l1_seeds.mean(axis=0)
    l2_avg = l2_seeds.mean(axis=0)
    blend_K = W1 * l1_avg + W2 * l2_avg + W3 * p_r

    # per-puzzle gain (multi-seed vs single-seed)
    pp = {}
    gains = []
    for p in puzzles:
        m = pz == p
        if m.sum() == 0:
            continue
        s_1 = _skill(_mae(y[m], blend_1[m]), y_med)
        s_K = _skill(_mae(y[m], blend_K[m]), y_med)
        pp[p] = {"n": int(m.sum()), "single_skill": float(s_1),
                 "multiseed_skill": float(s_K),
                 "gain_pp": float((s_K - s_1) * 100)}
        gains.append(s_K - s_1)
    gains = np.array(gains)

    # puzzle-block permutation on the multi-seed pooled skill
    rng = np.random.default_rng(SEED0)
    skill_K_pool = _skill(_mae(y, blend_K), mae_bl)
    cnt = 0
    for _ in range(args.n_perm):
        perm_y = np.empty_like(y)
        for p in puzzles:
            m = pz == p
            perm_y[m] = rng.permutation(y[m])
        sk = _skill(_mae(perm_y, blend_K), mae_bl)
        if sk >= skill_K_pool:
            cnt += 1
    perm_p = (cnt + 1) / (args.n_perm + 1)

    report = {
        "schema": "reactflow_delta.m2r_multiseed_puzzle.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "puzzle",
        "n_samples": int(len(y)), "n_puzzles": len(puzzles),
        "n_features": int(X.shape[1]), "k_seeds": len(SEEDS), "seeds": SEEDS,
        "headline_weights": {"w1": W1, "w2": W2, "w3": W3},
        "baseline_mae": mae_bl,
        "per_seed_skill": {
            "l1": [_skill(_mae(y, p), mae_bl) for p in l1_seeds],
            "l2": [_skill(_mae(y, p), mae_bl) for p in l2_seeds],
        },
        "results": {
            "single_seed_3way": {"mae": _mae(y, blend_1),
                                 "skill": _skill(_mae(y, blend_1), mae_bl),
                                 "r2": _r2(y, blend_1)},
            "multiseed_3way": {"mae": _mae(y, blend_K),
                               "skill": _skill(_mae(y, blend_K), mae_bl),
                               "r2": _r2(y, blend_K)},
        },
        "per_puzzle": pp,
        "multiseed_gain": {
            "pooled_gain_pp": float((_skill(_mae(y, blend_K), mae_bl) -
                                     _skill(_mae(y, blend_1), mae_bl)) * 100),
            "r2_gain": float(_r2(y, blend_K) - _r2(y, blend_1)),
            "per_puzzle_mean_pp": float(gains.mean() * 100),
            "per_puzzle_min_pp": float(gains.min() * 100),
            "per_puzzle_max_pp": float(gains.max() * 100),
            "per_puzzle_pct_positive": float((gains > 0).mean()),
            "n_puzzles": int(len(gains)),
        },
        "permutation_p": float(perm_p),
        "n_perm": args.n_perm,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_multiseed_puzzle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_multiseed_puzzle_oof.npz",
             blend_1=blend_1, blend_K=blend_K, y=y, puzzles=pz)
    g = report["multiseed_gain"]
    print(f"\n=== M2R multi-seed averaging, PUZZLE-level LOO (K={len(SEEDS)}) ===")
    print(f"single-seed: skill={report['results']['single_seed_3way']['skill']:+.4f} "
          f"R2={report['results']['single_seed_3way']['r2']:.4f}")
    print(f"multi-seed : skill={report['results']['multiseed_3way']['skill']:+.4f} "
          f"R2={report['results']['multiseed_3way']['r2']:.4f}")
    print(f"gain: {g['pooled_gain_pp']:+.2f}pp (R2 {g['r2_gain']:+.4f})")
    print(f"per-puzzle: mean={g['per_puzzle_mean_pp']:+.2f}pp "
          f"range=[{g['per_puzzle_min_pp']:+.2f},{g['per_puzzle_max_pp']:+.2f}]pp "
          f"pct_pos={g['per_puzzle_pct_positive']:.3f}")
    print(f"perm p = {perm_p:.4f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_multiseed_puzzle_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred-puzzle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=500)
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
    pz = np.array([s.puzzle for s in samples])

    m2_oof = tr.load_m2_oof(args.m2_pred_puzzle)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    X = np.concatenate([X1, X2, X_tr], axis=1)
    print(f"[mspz] n={len(y)} X={X.shape} puzzles={len(set(pz.tolist()))} "
          f"K={len(SEEDS)}", flush=True)

    run_puzzle_multiseed(X, y, pz, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
