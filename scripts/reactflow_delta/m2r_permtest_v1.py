#!/usr/bin/env python3
"""m2r_permtest_v1.py — fast permutation test + bootstrap CI for the M2R GBDT
(default config, already completed in run_m2r_v1).  Uses the default GBDT
(100 trees, max_depth=3) which completed in ~40 min.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf

SEED = 20260816


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    n_des = len(des_list)

    import lightgbm as lgb
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    # ---- default GBDT LOO (same as run_m2r_v1) ----
    gbdt_preds = np.zeros(len(y))
    t0 = time.time()
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            gbdt_preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(n_estimators=100, max_depth=3,
                              random_state=SEED, verbose=-1, n_jobs=8)
        g.fit(X[m], y[m])
        gbdt_preds[~m] = g.predict(X[~m])
    wall = round(time.time() - t0, 1)

    mae_gbdt = _mae(y, gbdt_preds)
    skill_gbdt = 1.0 - mae_gbdt / mae_bl
    r2_gbdt = 1.0 - np.sum((y - gbdt_preds) ** 2) / np.sum((y - y.mean()) ** 2)

    # ---- design-block permutation test ----
    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(args.n_perm):
        perm = rng.permutation(n_des)
        perm_preds = np.zeros(len(y))
        for i, held in enumerate(des_list):
            m = keys == des_list[perm[i]]
            perm_preds[m] = gbdt_preds[m]
        sk = 1.0 - _mae(y, perm_preds) / mae_bl
        if sk >= skill_gbdt:
            cnt += 1
    perm_p = (cnt + 1) / (args.n_perm + 1)

    # ---- design-block bootstrap CI ----
    boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n_des, size=n_des)
        sel = np.zeros(len(y), dtype=bool)
        for i in idx:
            sel |= keys == des_list[i]
        if sel.sum() < 10:
            continue
        mae_b = _mae(y[sel], np.full(sel.sum(), y_med))
        mae_m = _mae(y[sel], gbdt_preds[sel])
        if mae_b > 0:
            boot.append(1.0 - mae_m / mae_b)
    boot = np.array(boot)
    ci_low = float(np.percentile(boot, 2.5)) if len(boot) else None
    ci_high = float(np.percentile(boot, 97.5)) if len(boot) else None

    # ---- per-design exclusion ----
    excl = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        mae_b = _mae(y[m], np.full(m.sum(), y_med))
        mae_m = _mae(y[m], gbdt_preds[m])
        excl.append(1.0 - mae_m / mae_b)
    excl = np.array(excl)

    # ---- per-design skill ----
    dskills = []
    for held in des_list:
        m = keys == held
        if m.sum() > 0:
            mae_m = _mae(y[m], gbdt_preds[m])
            mae_b = _mae(y[m], np.full(m.sum(), y_med))
            dskills.append(1.0 - mae_m / mae_b)
    dskills = np.array(dskills)

    report = {
        "schema": "reactflow_delta.m2r_permtest.v1",
        "dataset": "OpenKnot_M2R", "n_designs": n_des, "n_samples": int(len(y)),
        "baseline_mae": float(mae_bl),
        "gbdt_default": {
            "mae": float(mae_gbdt), "skill": float(skill_gbdt), "r2": float(r2_gbdt),
            "permutation_p": float(perm_p), "n_perm": args.n_perm,
            "ci_low": ci_low, "ci_high": ci_high, "n_boot": args.n_boot,
            "per_design_skill_mean": float(dskills.mean()),
            "per_design_skill_median": float(np.median(dskills)),
            "per_design_skill_pct_positive": float((dskills > 0).mean()),
            "per_design_skill_min": float(dskills.min()),
            "per_design_skill_max": float(dskills.max()),
            "loo_exclusion_min": float(excl.min()),
            "loo_exclusion_max": float(excl.max()),
            "wall_seconds": wall,
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_permtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2R GBDT default + stats ===")
    print(f"skill={skill_gbdt:+.4f} R2={r2_gbdt:.4f} perm_p={perm_p:.4f}")
    print(f"CI=({ci_low:.4f},{ci_high:.4f})")
    print(f"per-design: mean={dskills.mean():+.4f} median={np.median(dskills):+.4f} "
          f"pct+={(dskills>0).mean():.3f} range=[{dskills.min():+.4f},{dskills.max():+.4f}]")
    print(f"LOO exclusion: [{excl.min():+.4f},{excl.max():+.4f}]")
    print(f"wall={wall}s DONE -> {out / 'm2r_permtest.json'}")


if __name__ == "__main__":
    sys.exit(main())