#!/usr/bin/env python3
"""m2r_mfe_puzzle_v1.py — puzzle-level leak-free multi-seed 3-way + MFE features.

Confirms the MFE thermodynamic feature gain at the PUZZLE level (train on 19
puzzles -> predict the held-out puzzle, with puzzle-level M2 transfer), and
compares vs the saved non-MFE puzzle-level multi-seed baseline.
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

SEED0 = 20260817
W1, W2, W3 = 0.6, 0.3, 0.1
SEEDS = [20260817, 20260818, 20260819, 20260820, 20260821]
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


def _loo_lgb_seed(X, y, pz, puzzles, obj, seed):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in puzzles:
        m = pz != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        g = lgb.LGBMRegressor(objective=obj, random_state=seed, verbose=-1,
                              n_jobs=2, **CFG)
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


def run_puzzle_mfe(X, y, pz, args) -> dict:
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
        print(f"[mfepz] L1 seed {seed} skill={_skill(_mae(y, p), mae_bl):+.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)
        p = _loo_lgb_seed(X, y, pz, puzzles, "regression", seed)
        l2_seeds.append(p)
        print(f"[mfepz] L2 seed {seed} skill={_skill(_mae(y, p), mae_bl):+.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)

    l1_seeds = np.array(l1_seeds)
    l2_seeds = np.array(l2_seeds)
    blend_1 = W1 * l1_seeds[0] + W2 * l2_seeds[0] + W3 * p_r
    l1_avg = l1_seeds.mean(axis=0)
    l2_avg = l2_seeds.mean(axis=0)
    blend_K = W1 * l1_avg + W2 * l2_avg + W3 * p_r

    # ---- compare vs saved non-MFE puzzle multi-seed baseline ----
    base_npz = np.load(args.base_npz)
    base_blend = np.asarray(base_npz["blend_K"], dtype=np.float64)
    base_y = np.asarray(base_npz["y"], dtype=np.float64)
    if len(base_blend) != len(y) or np.abs(base_y - y).max() > 1e-9:
        raise SystemExit("[mfepz] baseline npz y mismatch")
    base_skill = _skill(_mae(y, base_blend), mae_bl)

    # per-puzzle gain
    gains = []
    for p in puzzles:
        m = pz == p
        if m.sum() == 0:
            continue
        gains.append(_skill(_mae(y[m], blend_K[m]), y_med) -
                     _skill(_mae(y[m], base_blend[m]), y_med))
    gains = np.array(gains)

    # puzzle-block permutation on the MFE multi-seed pooled skill
    rng = np.random.default_rng(SEED0)
    skill_K = _skill(_mae(y, blend_K), mae_bl)
    cnt = 0
    for _ in range(args.n_perm):
        perm_y = np.empty_like(y)
        for p in puzzles:
            m = pz == p
            perm_y[m] = rng.permutation(y[m])
        if _skill(_mae(perm_y, blend_K), mae_bl) >= skill_K:
            cnt += 1
    perm_p = (cnt + 1) / (args.n_perm + 1)

    report = {
        "schema": "reactflow_delta.m2r_mfe_puzzle.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "puzzle",
        "n_samples": int(len(y)), "n_puzzles": len(puzzles),
        "n_features": int(X.shape[1]),
        "k_seeds": len(SEEDS), "seeds": SEEDS,
        "baseline_mae": mae_bl,
        "results": {
            "mfe_multiseed_3way": {"mae": _mae(y, blend_K),
                                   "skill": skill_K, "r2": _r2(y, blend_K)},
            "nonmfe_multiseed_3way": {"mae": _mae(y, base_blend),
                                      "skill": base_skill,
                                      "r2": _r2(y, base_blend)},
        },
        "mfe_gain_vs_nonmfe": {
            "pooled_gain_pp": float((skill_K - base_skill) * 100),
            "r2_gain": float(_r2(y, blend_K) - _r2(y, base_blend)),
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
    (out / "m2r_mfe_puzzle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_mfe_puzzle_oof.npz",
             blend_1=blend_1, blend_K=blend_K, y=y, puzzles=pz)
    g = report["mfe_gain_vs_nonmfe"]
    print(f"\n=== M2R MFE multi-seed, PUZZLE-level LOO (K={len(SEEDS)}) ===")
    print(f"non-MFE multi-seed: skill={base_skill:+.4f} "
          f"R2={_r2(y, base_blend):.4f}")
    print(f"MFE multi-seed    : skill={skill_K:+.4f} R2={_r2(y, blend_K):.4f}")
    print(f"gain: {g['pooled_gain_pp']:+.2f}pp (R2 {g['r2_gain']:+.4f})")
    print(f"per-puzzle: mean={g['per_puzzle_mean_pp']:+.2f}pp "
          f"range=[{g['per_puzzle_min_pp']:+.2f},{g['per_puzzle_max_pp']:+.2f}]pp "
          f"pct_pos={g['per_puzzle_pct_positive']:.3f}")
    print(f"perm p = {perm_p:.4f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_mfe_puzzle_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred-puzzle", required=True)
    ap.add_argument("--base-npz", required=True,
                    help="m2r_multiseed_puzzle_oof.npz (non-MFE baseline)")
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
    XM, _ = mfe.build_all_mfe(samples)
    pz = np.array([s.puzzle for s in samples])

    m2_oof = tr.load_m2_oof(args.m2_pred_puzzle)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    X = np.concatenate([X1, X2, X_tr, XM], axis=1)
    print(f"[mfepz] n={len(y)} X={X.shape} puzzles={len(set(pz.tolist()))} "
          f"K={len(SEEDS)}", flush=True)

    run_puzzle_mfe(X, y, pz, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
