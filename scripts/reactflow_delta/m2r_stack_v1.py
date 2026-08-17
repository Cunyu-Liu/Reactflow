#!/usr/bin/env python3
"""m2r_stack_v1.py — stacking meta-learner + residual boosting on the multi-seed
strong 3-way OOF columns (method-level levers beyond fixed 0.6/0.3/0.1 weights).

MOTIVATION (method-level, improves the model itself, not the data):
The current headline blends L1-LGB / L2-LGB / Ridge with FIXED a-priori
weights 0.6/0.3/0.1 (chosen from a wide, flat plateau).  Two principled
method-level upgrades are tested on the SAME multi-seed OOF columns:

  (a) STACKING / learned blend weights: a leak-free nested meta-learner
      (NNLS / Ridge / Ridge+quadratic) fit per held-out design on the other
      designs' OOF columns {l1_avg, l2_avg, ridge} -> predicts that design's
      blend.  This lets the data choose the combination (and its interactions)
      instead of a fixed weight vector.

  (b) RESIDUAL BOOSTING: fit a second-stage GBDT on the OOF residual
      (y - blend_fixed) using the LEGAL features (design-level LOO, leak-free),
      then add a damped residual correction back.  If the 3-way blend leaves
      learnable structure on the table, this captures it.

  (c) CONFIG-SOUP (bonus): average L1 over 5 seeds AND over a second capacity
      config (300 trees/depth6 vs 500 trees/depth8) to further decorrelate.

Method (design-level LOO, exchangeable unit = design; everything leak-free):
  * multi-seed OOF columns: L1 (K=5), L2 (K=5), Ridge  [one pass, reused]
  * stacking / residual / soup all evaluated at design-level LOO
  * pooled gain + LOO-exclusion + paired design-block permutation
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
SEEDS = [20260817, 20260818, 20260819, 20260820, 20260821]
CFG_A = dict(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
             min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
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


def _loo_meta(M, y, keys, des_list, kind):
    """Nested design-level LOO stacking meta-learner on OOF columns M (n,k)."""
    from sklearn.linear_model import Ridge
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            preds[~m] = np.median(y)
            continue
        Xtr, ytr = M[m], y[m]
        if kind == "nnls":
            from scipy.optimize import nnls
            w, _ = nnls(Xtr, ytr)
            preds[~m] = M[~m] @ w
        elif kind == "ridge":
            r = Ridge(alpha=1.0).fit(Xtr, ytr)
            preds[~m] = r.predict(M[~m])
        elif kind == "ridge_q":
            Qtr = _quad(M[m])
            Qte = _quad(M[~m])
            r = Ridge(alpha=1.0).fit(Qtr, ytr)
            preds[~m] = r.predict(Qte)
    return preds


def _quad(M):
    n, k = M.shape
    parts = [M]
    for a in range(k):
        for b in range(a, k):
            parts.append((M[:, a] * M[:, b])[:, None])
    return np.concatenate(parts, axis=1)


def run_stack(X, y, keys, args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    des_list = sorted(set(keys.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    t0 = time.time()

    p_r = _loo_ridge(X, y, keys, des_list)
    l1A = np.stack([_loo_lgb_cfg(X, y, keys, des_list, "l1", CFG_A, s)
                    for s in SEEDS], axis=1)
    l2A = np.stack([_loo_lgb_cfg(X, y, keys, des_list, "regression", CFG_A, s)
                    for s in SEEDS], axis=1)
    l1_avg = l1A.mean(axis=1)
    l2_avg = l2A.mean(axis=1)
    # EARLY CHECKPOINT: save base OOF columns now so slow bonus levers
    # (config-soup) can never block the main stacking / residual analysis.
    np.savez(out / "m2r_stack_base_oof.npz",
             l1_avg=l1_avg, l2_avg=l2_avg, ridge=p_r, y=y, keys=keys)
    print(f"[stk] base columns done + saved wall={time.time()-t0:.0f}s",
          flush=True)

    # ---- baseline: fixed 3-way blend ----
    blend_fixed = W1 * l1_avg + W2 * l2_avg + W3 * p_r
    base_skill = _skill(_mae(y, blend_fixed), mae_bl)
    base_r2 = _r2(y, blend_fixed)
    print(f"[stk] fixed 3-way skill={base_skill:+.4f} R2={base_r2:.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    # ---- (a) stacking meta-learners ----
    M = np.stack([l1_avg, l2_avg, p_r], axis=1)  # (n, 3)
    stack_results = {}
    for kind in ["nnls", "ridge", "ridge_q"]:
        sp = _loo_meta(M, y, keys, des_list, kind)
        stack_results[kind] = {"mae": _mae(y, sp),
                               "skill": _skill(_mae(y, sp), mae_bl),
                               "r2": _r2(y, sp)}
        print(f"[stk] stack {kind:8s} skill={stack_results[kind]['skill']:+.4f} "
              f"R2={stack_results[kind]['r2']:.4f}", flush=True)

    # ---- (b) residual boosting on legal features ----
    resid = y - blend_fixed
    resid_pred = _loo_lgb_cfg(X, resid, keys, des_list, "regression", CFG_A,
                              SEED0)
    boost = {}
    for gamma in [0.2, 0.5, 1.0]:
        b = blend_fixed + gamma * resid_pred
        boost[f"g{gamma:.1f}"] = {"mae": _mae(y, b),
                                  "skill": _skill(_mae(y, b), mae_bl),
                                  "r2": _r2(y, b)}
        print(f"[stk] residual boost g={gamma:.1f} "
              f"skill={boost[f'g{gamma:.1f}']['skill']:+.4f} "
              f"R2={boost[f'g{gamma:.1f}']['r2']:.4f}", flush=True)

    # config-soup (CFG_B) — BONUS, runs LAST so it can never block the above
    soup = {}
    if args.soup:
        l1B = np.stack([_loo_lgb_cfg(X, y, keys, des_list, "l1", CFG_B, s)
                        for s in SEEDS], axis=1)
        l2B = np.stack([_loo_lgb_cfg(X, y, keys, des_list, "regression", CFG_B, s)
                        for s in SEEDS], axis=1)
        l1_soup = 0.5 * l1_avg + 0.5 * l1B.mean(axis=1)
        l2_soup = 0.5 * l2_avg + 0.5 * l2B.mean(axis=1)
        blend_soup = W1 * l1_soup + W2 * l2_soup + W3 * p_r
        soup = {
            "l1_cfgA_skill": _skill(_mae(y, l1_avg), mae_bl),
            "l1_cfgB_skill": _skill(_mae(y, l1B.mean(axis=1)), mae_bl),
            "l2_cfgA_skill": _skill(_mae(y, l2_avg), mae_bl),
            "l2_cfgB_skill": _skill(_mae(y, l2B.mean(axis=1)), mae_bl),
            "soup_blend": {"mae": _mae(y, blend_soup),
                           "skill": _skill(_mae(y, blend_soup), mae_bl),
                           "r2": _r2(y, blend_soup)},
        }
        print(f"[stk] config-soup skill={soup['soup_blend']['skill']:+.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)

    # ---- LOO-exclusion + perm test for the best candidate ----
    cand_names = ["nnls", "ridge", "ridge_q", "g0.5", "g1.0"]
    cand_preds = {
        "nnls": _loo_meta(M, y, keys, des_list, "nnls"),
        "ridge": _loo_meta(M, y, keys, des_list, "ridge"),
        "ridge_q": _loo_meta(M, y, keys, des_list, "ridge_q"),
        "g0.5": blend_fixed + 0.5 * resid_pred,
        "g1.0": blend_fixed + 1.0 * resid_pred,
    }
    best_name = max(cand_names, key=lambda k: cand_preds[k] is not None
                    and _skill(_mae(y, cand_preds[k]), mae_bl))
    best_skill = _skill(_mae(y, cand_preds[best_name]), mae_bl)

    per_cand = {}
    rng = np.random.default_rng(SEED0)
    for cn in cand_names:
        gains = []
        for held in des_list:
            m = keys != held
            if m.sum() < 10:
                continue
            gains.append(_skill(_mae(y[m], cand_preds[cn][m]), mae_bl) -
                         _skill(_mae(y[m], blend_fixed[m]), mae_bl))
        gains = np.array(gains)
        # paired design-block permutation on per-design deltas
        d1, d2 = [], []
        for d in des_list:
            m = keys == d
            if m.sum() == 0:
                continue
            d1.append(_skill(_mae(y[m], blend_fixed[m]), y_med))
            d2.append(_skill(_mae(y[m], cand_preds[cn][m]), y_med))
        d1, d2 = np.array(d1), np.array(d2)
        dg = d2 - d1
        mean_delta = float(dg.mean())
        cnt = 0
        for _ in range(args.n_perm):
            swap = rng.random(len(d1)) < 0.5
            pd_ = np.where(swap, dg, -dg)
            if pd_.mean() >= mean_delta:
                cnt += 1
        perm_p = (cnt + 1) / (args.n_perm + 1)
        per_cand[cn] = {
            "pooled_gain_pp": float((_skill(_mae(y, cand_preds[cn]), mae_bl) -
                                     base_skill) * 100),
            "loo_exclusion": {
                "gain_mean_pp": float(gains.mean() * 100),
                "gain_min_pp": float(gains.min() * 100),
                "gain_max_pp": float(gains.max() * 100),
                "pct_positive": float((gains > 0).mean()),
                "n_folds": int(len(gains)),
            },
            "per_design_mean_pp": float(mean_delta * 100),
            "permutation_p": float(perm_p),
        }
        print(f"[stk] {cn:8s} gain={per_cand[cn]['pooled_gain_pp']:+.2f}pp "
              f"p={per_cand[cn]['permutation_p']:.4f} "
              f"pct_pos={per_cand[cn]['loo_exclusion']['pct_positive']:.3f}",
              flush=True)

    report = {
        "schema": "reactflow_delta.m2r_stack.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "n_features": int(X.shape[1]),
        "seeds": SEEDS, "k_seeds": len(SEEDS),
        "cfg_A": CFG_A, "cfg_B": CFG_B,
        "baseline_mae": mae_bl,
        "headline_weights": {"w1": W1, "w2": W2, "w3": W3},
        "results": {
            "fixed_3way": {"mae": _mae(y, blend_fixed), "skill": base_skill,
                           "r2": base_r2},
            "stacking": stack_results,
            "residual_boost": boost,
        },
        "config_soup": soup if args.soup else None,
        "candidate_gains": per_cand,
        "best_candidate": best_name,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "m2r_stack_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_stack_oof.npz",
             l1_avg=l1_avg, l2_avg=l2_avg, ridge=p_r,
             blend_fixed=blend_fixed, resid_pred=resid_pred,
             y=y, keys=keys)
    print(f"\n=== M2R stacking / residual-boost / config-soup ===")
    print(f"fixed 3-way : skill={base_skill:+.4f} R2={base_r2:.4f}")
    for k, v in stack_results.items():
        print(f"  stack {k:8s}: skill={v['skill']:+.4f} R2={v['r2']:.4f}")
    for k, v in boost.items():
        print(f"  boost {k}  : skill={v['skill']:+.4f} R2={v['r2']:.4f}")
    print(f"best candidate: {best_name} skill={best_skill:+.4f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2r_stack_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--soup", action="store_true",
                    help="also run the CFG_B config-soup (extra ~20 min)")
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
    X = np.concatenate([X1, X2, X_tr], axis=1)
    print(f"[stk] n={len(y)} X={X.shape} K={len(SEEDS)}", flush=True)

    run_stack(X, y, keys, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
