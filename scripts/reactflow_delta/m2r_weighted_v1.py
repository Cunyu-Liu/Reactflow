#!/usr/bin/env python3
"""m2r_weighted_v1.py — inverse-variance sample weighting for the heavy-tail.

The noise-floor analysis (m2r_noise_floor_v1.py) shows per-sample
measurement-noise sigma varies widely (median 0.024, mean 0.107 — heavy tail).
Samples whose target is dominated by measurement noise are less learnable, so
down-weighting them (1/sigma^2) should let the model focus on reliable samples
and improve the MAE metric.

LEGAL (non-circular) weighting: sigma is propagated from the WT + singleA +
singleB reactivity-error channels ONLY — the double-mutant error is excluded,
since the double-mutant profile is the circular information that defines
rescue.  Weights are a fixed function of the legal error profiles, computed
per-sample with no use of the target value.

Tests, in design-level LOO with the L1 objective (current best):
  * unweighted L1 (baseline for this script)
  * 1/sigma^2 weighted L1
  * 1/(sigma^2 + eps) clipped-weighted L1  (avoid extreme weight on tiny sigma)
  * inverse-sigma weighted L1  (milder)
Also reports the per-design gain of the best weighting vs unweighted L1.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_noise_floor_v1 as nf

SEED = 20260817
N_MC = 200


def _prof(p, n):
    a = np.full(n, np.nan, dtype=np.float64)
    for k, v in enumerate(p):
        if k < n:
            a[k] = v
    return a


def per_sample_sigma_legal(samples, n_mc=N_MC, seed=SEED):
    """Propagate WT+singleA+singleB errors (NOT double) through the rescue
    formula to get a legal per-sample measurement-noise sigma."""
    rng = np.random.default_rng(seed)
    sigmas = np.zeros(len(samples))
    for idx, s in enumerate(samples):
        n = len(s.wt_reactivity)
        mask = nf._design_mask(n, s.sub_start, s.sub_end)
        wt = _prof(s.wt_reactivity, n); ra = _prof(s.singleA_reactivity, n)
        rb = _prof(s.singleB_reactivity, n); rd = _prof(s.double_reactivity, n)
        we = _prof(s.wt_error, n); ae = _prof(s.singleA_error, n)
        be = _prof(s.singleB_error, n)
        draws = []
        for _ in range(n_mc):
            wtp = np.where(np.isfinite(wt), wt + rng.normal(0, 1, n) * np.where(np.isfinite(we), we, 0.0), np.nan)
            rap = np.where(np.isfinite(ra), ra + rng.normal(0, 1, n) * np.where(np.isfinite(ae), ae, 0.0), np.nan)
            rbp = np.where(np.isfinite(rb), rb + rng.normal(0, 1, n) * np.where(np.isfinite(be), be, 0.0), np.nan)
            v = nf.rescue_from_profiles(wtp, rap, rbp, rd, mask)
            if np.isfinite(v):
                draws.append(v)
        sigmas[idx] = float(np.std(draws)) if len(draws) >= 30 else np.nan
    # fill NaNs with median (samples with too few valid profiles get median weight)
    med = float(np.nanmedian(sigmas))
    sigmas = np.where(np.isfinite(sigmas), sigmas, med)
    return sigmas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-mc", type=int, default=N_MC)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    print(f"[m2r_w] n_samples={len(y)} n_designs={len(des_list)} X={X.shape}",
          flush=True)

    # legal per-sample sigma (Monte Carlo error propagation, no double error)
    t0 = time.time()
    sig = per_sample_sigma_legal(samples, args.n_mc)
    print(f"[m2r_w] per-sample sigma: median={np.median(sig):.4f} "
          f"mean={np.mean(sig):.4f} p95={np.percentile(sig,95):.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)

    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    import lightgbm as lgb

    def loo_weighted(weight_fn, objective="l1"):
        preds = np.zeros(len(y))
        for held in des_list:
            m = keys != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            w = weight_fn(sig[m])
            g = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=args.depth,
                                  random_state=SEED, verbose=-1, n_jobs=2,
                                  objective=objective)
            g.fit(X[m], y[m], sample_weight=w)
            preds[~m] = g.predict(X[~m])
        return preds

    def _skill(p):
        return 1.0 - np.mean(np.abs(y - p)) / mae_bl

    def _skill_sub(p, m):
        return 1.0 - np.mean(np.abs(y[m] - p[m])) / mae_bl

    def _r2(p):
        return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    weightings = {
        "unweighted": lambda s: None,
        "inv_var": lambda s: 1.0 / (s ** 2 + 1e-6),
        "inv_var_clip": lambda s: np.clip(1.0 / (s ** 2 + 1e-6), 1e-3, 50.0),
        "inv_sigma": lambda s: 1.0 / (s + 1e-3),
    }
    results = {}
    preds_by = {}
    for name, wfn in weightings.items():
        p = loo_weighted(wfn)
        preds_by[name] = p
        results[name] = {"skill": _skill(p), "r2": _r2(p),
                         "mae": float(np.mean(np.abs(y - p)))}
        print(f"[m2r_w] {name:14s} skill={results[name]['skill']:+.4f} "
              f"R2={results[name]['r2']:.4f} MAE={results[name]['mae']:.4f} "
              f"wall={time.time()-t0:.0f}s", flush=True)

    # per-design gain of best weighting vs unweighted L1
    base = preds_by["unweighted"]
    best_name = max((n for n in weightings if n != "unweighted"),
                    key=lambda n: results[n]["skill"])
    best = preds_by[best_name]
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        s_b = _skill_sub(base, m); s_w = _skill_sub(best, m)
        gains.append(s_w - s_b)
    gains = np.array(gains)
    report = {
        "schema": "reactflow_delta.m2r_weighted.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": len(y), "n_designs": len(des_list),
        "objective": "l1", "trees": args.trees, "depth": args.depth,
        "baseline_mae": mae_bl,
        "sigma_legal": {"median": float(np.median(sig)), "mean": float(np.mean(sig)),
                        "p95": float(np.percentile(sig, 95)),
                        "corr_with_rescue_abs": float(
                            np.corrcoef(sig, np.abs(y - np.median(y)))[0, 1])},
        "results": results,
        "best_weighting": best_name,
        "best_vs_unweighted_loo": {
            "gain_mean_pp": float(gains.mean() * 100),
            "gain_min_pp": float(gains.min() * 100),
            "gain_max_pp": float(gains.max() * 100),
            "pct_positive": float((gains > 0).mean()),
            "n_folds": int(len(gains)),
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    np.savez(out / "m2r_weighted_oof.npz",
             **{k: v for k, v in preds_by.items()}, y=y, keys=keys)
    (out / "m2r_weighted_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[m2r_w] best weighting: {best_name} "
          f"skill={results[best_name]['skill']:+.4f}")
    g = report["best_vs_unweighted_loo"]
    print(f"  best-vs-unweighted LOO gain: mean={g['gain_mean_pp']:+.2f}pp "
          f"range=[{g['gain_min_pp']:+.2f},{g['gain_max_pp']:+.2f}]pp "
          f"pct_pos={g['pct_positive']:.3f}")
    print(f"\n  DONE -> {out / 'm2r_weighted_report.json'}")


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


if __name__ == "__main__":
    sys.exit(main())
