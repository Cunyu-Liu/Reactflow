#!/usr/bin/env python3
"""m2r_robust_objective_v1.py — robust loss GBDT for the heavy-tailed rescue_factor.

The headline metric is MAE skill, but every previous GBDT optimizes L2 (MSE).
The rescue_factor target is heavy-tailed (min -2.57, 1% qtl -0.25, mean 0.41,
std 0.25), so an L1 / Huber / Fair objective directly optimizes the reported
metric and down-weights tail influence.  This is a pure method-level lever
(same features, same LOO protocol, different training objective).

Tests, in design-level LOO on the 230-dim feature stack (incl. M2_structure):
  * regression (L2, baseline)
  * l1 (MAE objective)
  * huber delta in {0.5, 1.0, 2.0}
  * fair c in {0.5, 1.0, 2.0}
Then, for the best robust config, adds M2-transfer features + Ridge blend
(a=0.80) to build the full-stack robust variant.
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


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred", default=None,
                    help="M2 keyed_predictions jsonl (for transfer features)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    print(f"[m2r_rob] n_samples={len(y)} n_designs={len(des_list)} X={X.shape}",
          flush=True)

    # optional transfer features
    X_tr = None
    if args.m2_pred:
        m2_oof = tr.load_m2_oof(args.m2_pred)
        m2_design_key = {}
        for did in m2_oof:
            parts = did.split("_")
            if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
                m2_design_key[(parts[2], "_".join(parts[3:]))] = did
        X_tr = tr.build_transfer_features(samples, m2_oof, m2_design_key)
        print(f"[m2r_rob] transfer feats {X_tr.shape}", flush=True)

    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    def loo(Xu, objective, **kw):
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = lgb.LGBMRegressor(
                n_estimators=args.trees, max_depth=args.depth,
                random_state=SEED, verbose=-1, n_jobs=2,
                objective=objective, **kw)
            g.fit(Xu[m], y[m])
            preds[~m] = g.predict(Xu[~m])
        return preds

    def loo_ridge(Xu):
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            r = Ridge(alpha=1.0).fit(Xu[m], y[m])
            preds[~m] = r.predict(Xu[~m])
        return preds

    configs = [
        ("l2", "regression", {}),
        ("l1", "l1", {}),
        ("huber_d0.5", "huber", {"alpha": 0.5}),
        ("huber_d1.0", "huber", {"alpha": 1.0}),
        ("huber_d2.0", "huber", {"alpha": 2.0}),
        ("fair_c0.5", "fair", {"c": 0.5}),
        ("fair_c1.0", "fair", {"c": 1.0}),
        ("fair_c2.0", "fair", {"c": 2.0}),
    ]

    results = {}
    t0 = time.time()
    preds_by_name = {}
    for name, obj, kw in configs:
        p = loo(X, obj, **kw)
        preds_by_name[name] = p
        results[name] = {
            "objective": obj, "params": kw,
            "mae": _mae(y, p), "skill": _skill(_mae(y, p), mae_bl),
            "r2": _r2(y, p),
        }
        print(f"[m2r_rob] {name:12s} skill={results[name]['skill']:+.4f} "
              f"R2={results[name]['r2']:.4f} MAE={results[name]['mae']:.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)

    # ---- full-stack robust variant: best robust objective + transfer + blend ----
    oof_comb = None
    if X_tr is not None:
        X_comb = np.concatenate([X, X_tr], axis=1)
        # find best robust config by skill (excluding l2 to be honest: pick best
        # robust, then compare against l2 full-stack)
        best_robust = max((c[0] for c in configs if c[0] != "l2"),
                          key=lambda n: results[n]["skill"])
        p_comb = loo(X_comb, results[best_robust]["objective"],
                     **results[best_robust]["params"])
        ridge_comb = loo_ridge(X_comb)
        blend = 0.80 * p_comb + 0.20 * ridge_comb
        results[f"fullstack_{best_robust}_blend"] = {
            "objective": results[best_robust]["objective"],
            "params": results[best_robust]["params"],
            "transfer": True, "blend": True,
            "mae": _mae(y, blend), "skill": _skill(_mae(y, blend), mae_bl),
            "r2": _r2(y, blend),
        }
        # l2 full-stack blend for honest comparison
        p_l2 = loo(X_comb, "regression")
        blend_l2 = 0.80 * p_l2 + 0.20 * ridge_comb
        results["fullstack_l2_blend"] = {
            "objective": "regression", "transfer": True, "blend": True,
            "mae": _mae(y, blend_l2), "skill": _skill(_mae(y, blend_l2), mae_bl),
            "r2": _r2(y, blend_l2),
        }
        print(f"[m2r_rob] fullstack_{best_robust}_blend skill="
              f"{results[f'fullstack_{best_robust}_blend']['skill']:+.4f}")
        print(f"[m2r_rob] fullstack_l2_blend     skill="
              f"{results['fullstack_l2_blend']['skill']:+.4f}")
        # save OOF for permtest
        oof_comb = {
            "l2": preds_by_name["l2"], "l1": preds_by_name["l1"],
            "best_robust_comb": p_comb, "best_robust_blend": blend,
            "l2_comb": p_l2, "l2_blend": blend_l2,
            "best_robust_name": best_robust,
            "y": y, "keys": keys,
        }
        np.savez(out / "m2r_robust_oof.npz", **oof_comb)

    report = {
        "schema": "reactflow_delta.m2r_robust_objective.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": len(y), "n_designs": len(des_list),
        "baseline_mae": mae_bl,
        "target_tail": {"min": float(y.min()), "p01": float(np.percentile(y, 1)),
                        "p05": float(np.percentile(y, 5)), "max": float(y.max())},
        "results": results,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_robust_objective_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[m2r_rob] DONE -> {out / 'm2r_robust_objective_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
