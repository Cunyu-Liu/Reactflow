#!/usr/bin/env python3
"""m2r_3way_puzzle_v1.py — puzzle-level leak-free 3-way ensemble for M2R.

Audits the 3-way ensemble (L1-GBDT + L2-GBDT + Ridge, weights a-priori
0.6/0.3/0.1) at the PUZZLE level: train on 19 puzzles -> predict all 8 designs
of the held-out puzzle, using PUZZLE-LEVEL M2 OOF transfer features (leak-free,
from run_response_spectrum_m2_attn_puzzle_v1.py).  This confirms the cross-
objective ensemble gain generalizes to completely unseen puzzles.

Also reports the puzzle-block permutation p for the 3-way blend and the
per-puzzle 3-way-vs-prev-headline gain (100%-positive check over 20 puzzles).
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
    return 1.0 - float(np.sum((y - p) ** 2)) / float(np.sum((y - y.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred-puzzle", required=True,
                    help="PUZZLE-level M2 keyed_predictions jsonl (leak-free)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    sample_puzzles = np.array([s.puzzle for s in samples])
    puzzles = sorted(set(sample_puzzles.tolist()))
    print(f"[3waypz] n_samples={len(y)} n_puzzles={len(puzzles)} X={X.shape}",
          flush=True)

    # ---- puzzle-level M2 OOF transfer features (leak-free) ----
    m2_oof = tr.load_m2_oof(args.m2_pred_puzzle)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    print(f"[3waypz] X_tr={X_tr.shape} nonzero_frac="
          f"{(np.abs(X_tr).sum(axis=1) > 0).mean():.3f}", flush=True)
    X_comb = np.concatenate([X, X_tr], axis=1)

    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    def loo_pz(obj, **kw):
        preds = np.zeros(len(y))
        for held in puzzles:
            m = sample_puzzles != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=args.depth,
                                  random_state=SEED, verbose=-1, n_jobs=2,
                                  objective=obj, **kw)
            g.fit(X_comb[m], y[m])
            preds[~m] = g.predict(X_comb[~m])
        return preds

    def loo_ridge():
        preds = np.zeros(len(y))
        for held in puzzles:
            m = sample_puzzles != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            r = Ridge(alpha=1.0).fit(X_comb[m], y[m])
            preds[~m] = r.predict(X_comb[~m])
        return preds

    t0 = time.time()
    p_l1 = loo_pz("l1")
    print(f"[3waypz] L1 skill={_skill(_mae(y, p_l1), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p_l2 = loo_pz("regression")
    print(f"[3waypz] L2 skill={_skill(_mae(y, p_l2), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p_ridge = loo_ridge()
    print(f"[3waypz] Ridge skill={_skill(_mae(y, p_ridge), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    blend = W1 * p_l1 + W2 * p_l2 + W3 * p_ridge
    prev_blend = 0.80 * p_l1 + 0.20 * p_ridge

    results = {
        "l1_gbdt": {"mae": _mae(y, p_l1), "skill": _skill(_mae(y, p_l1), mae_bl),
                    "r2": _r2(y, p_l1)},
        "l2_gbdt": {"mae": _mae(y, p_l2), "skill": _skill(_mae(y, p_l2), mae_bl),
                    "r2": _r2(y, p_l2)},
        "ridge": {"mae": _mae(y, p_ridge), "skill": _skill(_mae(y, p_ridge), mae_bl),
                  "r2": _r2(y, p_ridge)},
        "prev_headline_l1_ridge_a80": {
            "mae": _mae(y, prev_blend), "skill": _skill(_mae(y, prev_blend), mae_bl),
            "r2": _r2(y, prev_blend)},
        "threeway_blend_a_priori": {
            "mae": _mae(y, blend), "skill": _skill(_mae(y, blend), mae_bl),
            "r2": _r2(y, blend), "w1": W1, "w2": W2, "w3": W3},
    }

    # ---- per-puzzle gain: 3-way vs prev headline ----
    per_puzzle = {}
    for held in puzzles:
        m = sample_puzzles == held
        if m.sum() == 0:
            continue
        s3 = _skill(_mae(y[m], blend[m]), mae_bl)
        s1 = _skill(_mae(y[m], prev_blend[m]), mae_bl)
        per_puzzle[held] = {"n": int(m.sum()),
                            "prev_skill": float(s1),
                            "threeway_skill": float(s3),
                            "gain_pp": float(s3 - s1)}

    gains = np.array([v["gain_pp"] for v in per_puzzle.values()])

    # ---- puzzle-block permutation p for the 3-way blend ----
    dam = np.array([y[sample_puzzles == pz].mean() for pz in puzzles])
    dpm = np.array([blend[sample_puzzles == pz].mean() for pz in puzzles])
    mae_bl_d = float(np.mean(np.abs(dam - np.median(dam))))
    skill_d = 1.0 - float(np.mean(np.abs(dam - dpm))) / mae_bl_d if mae_bl_d > 0 else 0.0
    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(args.n_perm):
        sk = 1.0 - float(np.mean(np.abs(dam - dpm[rng.permutation(len(puzzles))]))) / mae_bl_d if mae_bl_d > 0 else 0.0
        if sk >= skill_d:
            cnt += 1
    perm_p = (cnt + 1) / (args.n_perm + 1)

    np.savez(out / "m2r_3way_puzzle_oof.npz",
             l1=p_l1, l2=p_l2, ridge=p_ridge, blend=blend,
             prev_blend=prev_blend, y=y, sample_puzzles=sample_puzzles)

    report = {
        "schema": "reactflow_delta.m2r_3way_puzzle.v1",
        "dataset": "OpenKnot_M2R",
        "exchangeable_unit": "puzzle",
        "n_samples": int(len(y)), "n_puzzles": len(puzzles),
        "n_features": int(X_comb.shape[1]),
        "trees": args.trees, "depth": args.depth,
        "seed": SEED, "headline_weights": {"w1": W1, "w2": W2, "w3": W3},
        "baseline_mae": mae_bl,
        "results": results,
        "per_puzzle_gain_vs_prev": {
            "gain_mean_pp": float(gains.mean() * 100),
            "gain_min_pp": float(gains.min() * 100),
            "gain_max_pp": float(gains.max() * 100),
            "pct_positive": float((gains > 0).mean()),
            "n_puzzles": int(len(gains)),
        },
        "permutation_p": float(perm_p),
        "per_puzzle": per_puzzle,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_3way_puzzle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== M2R 3-way ensemble (puzzle-level LOO, leak-free) ===")
    for k, v in results.items():
        print(f"  {k:30s} skill={v['skill']:+.4f} R2={v['r2']:.4f} MAE={v['mae']:.4f}")
    g = report["per_puzzle_gain_vs_prev"]
    print(f"  3way-vs-prev per-puzzle gain: mean={g['gain_mean_pp']:+.2f}pp "
          f"range=[{g['gain_min_pp']:+.2f},{g['gain_max_pp']:+.2f}]pp "
          f"pct_pos={g['pct_positive']:.3f} perm_p={perm_p:.4f}")
    print(f"  wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_3way_puzzle_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
