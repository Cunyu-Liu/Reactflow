#!/usr/bin/env python3
"""m2r_permtest_v3.py — permutation test + bootstrap CI for M2R GBDT.

Server is heavily loaded (load avg 170+), so this uses a SMALL GBDT
(30 trees, max_depth=3, n_jobs=2) that completes fast even under load.
The permutation and bootstrap operate on OOF predictions (no re-training),
so the result is statistically valid for the trained model.
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
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--trees", type=int, default=30)
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

    gbdt_preds = np.zeros(len(y))
    t0 = time.time()
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            gbdt_preds[~m] = y_med
            continue
        g = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=3,
                              random_state=SEED, verbose=-1, n_jobs=2)
        g.fit(X[m], y[m])
        gbdt_preds[~m] = g.predict(X[~m])
    wall = round(time.time() - t0, 1)

    mae_gbdt = _mae(y, gbdt_preds)
    skill_gbdt = 1.0 - mae_gbdt / mae_bl
    r2_gbdt = 1.0 - np.sum((y - gbdt_preds) ** 2) / np.sum((y - y.mean()) ** 2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"m2r_gbdt_oof_t{args.trees}.npz",
             preds=gbdt_preds, y=y, keys=keys)

    rng = np.random.default_rng(SEED)
    # per-design mean predictions: valid block permutation test
    design_masks = {d: keys == d for d in des_list}
    d_mean_y = {}
    d_mean_p = {}
    for d in des_list:
        m = design_masks[d]
        d_mean_y[d] = float(y[m].mean())
        d_mean_p[d] = float(gbdt_preds[m].mean())
    dam = np.array([d_mean_y[d] for d in des_list])
    dpm = np.array([d_mean_p[d] for d in des_list])
    mae_bl_d = float(np.mean(np.abs(dam - np.median(dam))))
    skill_d_real = 1.0 - float(np.mean(np.abs(dam - dpm))) / mae_bl_d if mae_bl_d > 0 else 0.0

    cnt = 0
    for _ in range(args.n_perm):
        perm = rng.permutation(n_des)
        dpm_perm = dpm[perm]
        sk = 1.0 - float(np.mean(np.abs(dam - dpm_perm))) / mae_bl_d if mae_bl_d > 0 else 0.0
        if sk >= skill_d_real:
            cnt += 1
    perm_p = (cnt + 1) / (args.n_perm + 1)

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

    excl = []
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        mae_b = _mae(y[m], np.full(m.sum(), y_med))
        mae_m = _mae(y[m], gbdt_preds[m])
        excl.append(1.0 - mae_m / mae_b)
    excl = np.array(excl)

    dskills = []
    for held in des_list:
        m = keys == held
        if m.sum() > 0:
            mae_m = _mae(y[m], gbdt_preds[m])
            mae_b = _mae(y[m], np.full(m.sum(), y_med))
            dskills.append(1.0 - mae_m / mae_b)
    dskills = np.array(dskills)

    report = {
        "schema": "reactflow_delta.m2r_permtest.v3",
        "dataset": "OpenKnot_M2R", "n_designs": n_des, "n_samples": int(len(y)),
        "trees": args.trees, "baseline_mae": float(mae_bl),
        "gbdt": {
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
    (out / "m2r_permtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2R GBDT + stats ===")
    print(f"skill={skill_gbdt:+.4f} R2={r2_gbdt:.4f} perm_p={perm_p:.4f}")
    print(f"CI=({ci_low:.4f},{ci_high:.4f})")
    print(f"per-design: mean={dskills.mean():+.4f} pct+={(dskills>0).mean():.3f} "
          f"range=[{dskills.min():+.4f},{dskills.max():+.4f}]")
    print(f"LOO exclusion: [{excl.min():+.4f},{excl.max():+.4f}]")
    print(f"wall={wall}s DONE -> {out / 'm2r_permtest.json'}")


if __name__ == "__main__":
    sys.exit(main())