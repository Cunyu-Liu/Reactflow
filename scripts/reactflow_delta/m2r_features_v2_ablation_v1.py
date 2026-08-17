#!/usr/bin/env python3
"""m2r_features_v2_ablation_v1.py — does the v2 legal feature group (cross-
mutant disruption overlap + structural stem context) improve the M2R model?

MOTIVATION:
The 4-way ensemble (XGB architecture decorrelation) closed at +0.07pp
(design-level, perm p=0.256).  The ceiling audit shows legal features saturate
at R2 ~0.39 while the double-mutant oracle reaches R2 0.73 — the residual gap is
the double-mutant INTERACTION effect, which the v1 feature set (independent
single-mutant windows) barely captures.  v2 adds:
  A. cross-mutant disruption overlap (dA vs dB correlation + spatial overlap)
  B. legal disruption magnitudes rA, rB, sqrt(rA^2+rB^2) (the rescue denominator)
  C. target-structure stem context (stem length, stem position, depth)
  D. M2_structure cross context (both sites paired in independent experiment)

Method (design-level LOO, exchangeable unit = design):
  * strong GBDT (300 trees, depth 6, lr 0.05) L1 and L2 on v1-only vs v1+v2
  * Ridge on v1-only vs v1+v2
  * 3-way blend (0.6/0.3/0.1) on both
  * per-group v2 attribution (A/B/C/D separately added to v1)
  * LOO-exclusion gain (v1+v2 3-way vs v1 3-way), 100%-positive check

Nothing uses the double-mutant reactivity profile (all features legal).
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
# v2 feature order (see m2r_features_v2.build_v2_features):
#   A: 3 (design overlap) + 4 (window overlap) = 7
#   B: 5 (disruption magnitudes)
#   C: 8 (target-structure stem context)
#   D: 2 (M2_structure cross context)
V2_GROUPS = {"A": 7, "B": 5, "C": 8, "D": 2}


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
        g = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective=obj, random_state=SEED, verbose=-1, n_jobs=2)
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


def run_ablation(X1, X2, X_tr, y, keys, args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    des_list = sorted(set(keys.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    X1t = np.concatenate([X1, X_tr], axis=1)      # v1 + transfer (236)
    X12t = np.concatenate([X1, X2, X_tr], axis=1)  # v1 + v2 + transfer (260)
    n_v1 = X1.shape[1]; n_v2 = X2.shape[1]

    def _groups():
        # v2 group column ranges
        cols = []
        start = 0
        for g, c in V2_GROUPS.items():
            cols.append((g, np.arange(start, start + c)))
            start += c
        return cols

    t0 = time.time()

    # ---- v1-only strong-3way (baseline, should reproduce ~+0.2811) ----
    p1_l1 = _loo_lgb(X1t, y, keys, des_list, "l1")
    print(f"[abla] v1 L1 skill={_skill(_mae(y, p1_l1), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p1_l2 = _loo_lgb(X1t, y, keys, des_list, "regression")
    print(f"[abla] v1 L2 skill={_skill(_mae(y, p1_l2), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p1_r = _loo_ridge(X1t, y, keys, des_list)
    blend1 = W1 * p1_l1 + W2 * p1_l2 + W3 * p1_r
    print(f"[abla] v1 3-way skill={_skill(_mae(y, blend1), mae_bl):+.4f} "
          f"R2={_r2(y, blend1):.4f} wall={time.time()-t0:.0f}s", flush=True)

    # ---- v1+v2 full ----
    p2_l1 = _loo_lgb(X12t, y, keys, des_list, "l1")
    print(f"[abla] v1+v2 L1 skill={_skill(_mae(y, p2_l1), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p2_l2 = _loo_lgb(X12t, y, keys, des_list, "regression")
    print(f"[abla] v1+v2 L2 skill={_skill(_mae(y, p2_l2), mae_bl):+.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    p2_r = _loo_ridge(X12t, y, keys, des_list)
    blend2 = W1 * p2_l1 + W2 * p2_l2 + W3 * p2_r
    print(f"[abla] v1+v2 3-way skill={_skill(_mae(y, blend2), mae_bl):+.4f} "
          f"R2={_r2(y, blend2):.4f} wall={time.time()-t0:.0f}s", flush=True)

    # ---- per-group attribution: v1 + [group g] only ----
    group_res = {}
    for g, cols in _groups():
        Xg = np.concatenate([X1, X2[:, cols], X_tr], axis=1)
        pg_l1 = _loo_lgb(Xg, y, keys, des_list, "l1")
        group_res[g] = {
            "skill": _skill(_mae(y, pg_l1), mae_bl),
            "r2": _r2(y, pg_l1),
            "delta_pp_vs_v1": (_skill(_mae(y, pg_l1), mae_bl) -
                               _skill(_mae(y, p1_l1), mae_bl)) * 100,
            "n_features": int(Xg.shape[1]),
        }
        print(f"[abla] v1+[{g}] L1 skill={group_res[g]['skill']:+.4f} "
              f"({group_res[g]['delta_pp_vs_v1']:+.2f}pp) "
              f"wall={time.time()-t0:.0f}s", flush=True)

    # ---- LOO-exclusion gain (v1+v2 3-way vs v1 3-way) ----
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        gains.append(_skill(_mae(y[m], blend2[m]), mae_bl) -
                     _skill(_mae(y[m], blend1[m]), mae_bl))
    gains = np.array(gains)
    loo = {"gain_mean_pp": float(gains.mean() * 100),
           "gain_min_pp": float(gains.min() * 100),
           "gain_max_pp": float(gains.max() * 100),
           "pct_positive": float((gains > 0).mean()),
           "n_folds": int(len(gains))}

    results = {
        "v1_3way": {"mae": _mae(y, blend1), "skill": _skill(_mae(y, blend1), mae_bl),
                    "r2": _r2(y, blend1)},
        "v1_v2_3way": {"mae": _mae(y, blend2), "skill": _skill(_mae(y, blend2), mae_bl),
                       "r2": _r2(y, blend2)},
        "v1_l1": {"skill": _skill(_mae(y, p1_l1), mae_bl)},
        "v1_v2_l1": {"skill": _skill(_mae(y, p2_l1), mae_bl)},
        "v1_v2_ridge": {"skill": _skill(_mae(y, p2_r), mae_bl)},
    }

    report = {
        "schema": "reactflow_delta.m2r_features_v2_ablation.v1",
        "dataset": "OpenKnot_M2R",
        "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "n_features": {"v1": int(X1t.shape[1]), "v1_v2": int(X12t.shape[1]),
                       "v2_group": n_v2},
        "seed": SEED, "baseline_mae": mae_bl,
        "results": results,
        "per_group_attribution": group_res,
        "loo_exclusion_v2_gain": loo,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_features_v2_ablation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_features_v2_ablation_oof.npz",
             blend1=blend1, blend2=blend2, y=y, keys=keys)
    print("\n=== M2R features v2 ablation (design-level LOO) ===")
    for k, v in results.items():
        print(f"  {k:14s} skill={v['skill']:+.4f} R2={v.get('r2', float('nan')):.4f}")
    print("  per-group (v1 + [g], L1):")
    for g, v in group_res.items():
        print(f"    [{g}] {v['delta_pp_vs_v1']:+.2f}pp skill={v['skill']:+.4f}")
    print(f"  LOO-exclusion v2 gain: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"  wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_features_v2_ablation_report.json'}")
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

    m2_oof = tr.load_m2_oof(args.m2_pred)
    m2_design_key = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
    print(f"[abla] n={len(y)} X1={X1.shape} X2={X2.shape} X_tr={X_tr.shape}",
          flush=True)

    run_ablation(X1, X2, X_tr, y, keys, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
