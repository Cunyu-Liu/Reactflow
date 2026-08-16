#!/usr/bin/env python3
"""m2r_learnability_v1.py — quick learnability diagnostic for the M2R rescue_factor
prediction task, BEFORE committing to a full model build.

Question: can the rescue_factor (a continuous measure of base-pair support) be
predicted from the available features (WT reactivity, target structure, sequence
context) at the two pair sites (i,j)?

We test:
  1. Simple baseline: median rescue_factor (sequence-free).
  2. Ridge regression on WT reactivity at i and j + edit distance + structure
     depth at i and j — fast, CPU-only, no GPU needed.
  3. GBDT (LightGBM) on the same features — captures non-linear interactions.
  4. LOO by design (exchangeable unit = design, same as M2).

If any model achieves positive skill, the task is learnable and worth pursuing
with a full model.  If even the best model is at or below baseline, the task
may be noise-dominated (like the response-spectrum).

Output: /mnt/.../m2r_learnability.json
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import residual_spectrum_v2 as rsv2


def _dot_to_depth(structure):
    """Map dot-bracket to per-position (paired, depth)."""
    n = len(structure)
    paired = np.zeros(n, dtype=np.float64)
    depth = np.zeros(n, dtype=np.float64)
    stack = []
    openers = "([{"
    closers = ")]}"
    for i, ch in enumerate(structure):
        if ch in openers:
            stack.append(ch)
            paired[i] = 1.0
            depth[i] = len(stack)
        elif ch in closers:
            paired[i] = 1.0
            depth[i] = len(stack)
            if stack:
                stack.pop()
        else:
            depth[i] = len(stack)
    return paired, depth


def _nan_to(v, default=0.0):
    return default if v is None or not np.isfinite(v) else float(v)


def _elig(W, m):
    mask = [1 if (a is not None and b is not None and np.isfinite(a) and np.isfinite(b))
            else 0 for a, b in zip(W, m)]
    return np.array(mask, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = m2r.build_all_pair_samples(designs)
    print(f"n_samples={len(samples)} n_designs={len(designs)}", flush=True)

    # --- build per-pair feature vectors ---
    # features: [react_i, react_j, err_i, err_j, sigA_i, sigA_j, sigB_i, sigB_j,
    #            dbl_i, dbl_j, dbl_err_i, dbl_err_j,
    #            str_paired_i, str_paired_j, str_depth_i, str_depth_j,
    #            edit_dist, mutA, mutB_design_norm]
    X, y, design_keys = [], [], []
    for s in samples:
        if s.rescue_factor is None:
            continue
        seq = s.sequence
        sub_start = s.sub_start
        i, j = s.editA_seq_pos, s.editB_seq_pos
        # target structure at i,j
        tgt = s.target_structure
        if len(tgt) < max(i, j) + 1:
            # pad if needed
            tgt = tgt + "." * (max(i, j) + 1 - len(tgt))
        str_pa, str_dp = _dot_to_depth(tgt)

        def _safe(arr, idx):
            return _nan_to(arr[idx]) if idx < len(arr) else 0.0

        react_i = _safe(s.wt_reactivity, i)
        react_j = _safe(s.wt_reactivity, j)
        err_i = _safe(s.wt_error, i)
        err_j = _safe(s.wt_error, j)
        sigA_i = _safe(s.singleA_reactivity, i)
        sigA_j = _safe(s.singleA_reactivity, j)
        sigB_i = _safe(s.singleB_reactivity, i)
        sigB_j = _safe(s.singleB_reactivity, j)
        dbl_i = _safe(s.double_reactivity, i)
        dbl_j = _safe(s.double_reactivity, j)
        dbl_err_i = _safe(s.double_error, i)
        dbl_err_j = _safe(s.double_error, j)

        # compute the difference signals: single vs WT, double vs single
        # delta_singleA at i: how much does singleA change reactivity at i?
        # delta_double at i: how much does double restore reactivity at i?
        delta_si = sigA_i - react_i if np.isfinite(sigA_i) and np.isfinite(react_i) else 0.0
        delta_sj = sigB_j - react_j if np.isfinite(sigB_j) and np.isfinite(react_j) else 0.0
        delta_di = dbl_i - sigA_i if np.isfinite(dbl_i) and np.isfinite(sigA_i) else 0.0
        delta_dj = dbl_j - sigB_j if np.isfinite(dbl_j) and np.isfinite(sigB_j) else 0.0
        # rescue signal: did double bring reactivity back toward WT?
        rescue_i = abs(dbl_i - react_i) - abs(sigA_i - react_i) if np.isfinite(dbl_i) and np.isfinite(react_i) and np.isfinite(sigA_i) else 0.0
        rescue_j = abs(dbl_j - react_j) - abs(sigB_j - react_j) if np.isfinite(dbl_j) and np.isfinite(react_j) and np.isfinite(sigB_j) else 0.0

        edit_dist = abs(j - i)
        design_norm = len(seq)

        feats = np.array([
            react_i, react_j, err_i, err_j,
            sigA_i, sigA_j, sigB_i, sigB_j,
            dbl_i, dbl_j, dbl_err_i, dbl_err_j,
            str_pa[i] if i < len(str_pa) else 0.0,
            str_pa[j] if j < len(str_pa) else 0.0,
            str_dp[i] if i < len(str_dp) else 0.0,
            str_dp[j] if j < len(str_dp) else 0.0,
            edit_dist / max(design_norm, 1),
            delta_si, delta_sj, delta_di, delta_dj,
            rescue_i, rescue_j,
            s.mutA / max(design_norm, 1),
            s.mutB / max(design_norm, 1),
        ], dtype=np.float64)

        X.append(feats)
        y.append(s.rescue_factor)
        design_keys.append(s.design_id)

    X = np.array(X)
    y = np.array(y)
    design_keys = np.array(design_keys)
    designs_list = sorted(set(design_keys.tolist()))
    n_designs = len(designs_list)
    n_feats = X.shape[1]
    print(f"n_samples={len(y)} n_designs={n_designs} n_feats={n_feats}", flush=True)

    # --- baseline WMAE ---
    baseline_mae = float(np.mean(np.abs(y - np.median(y))))
    print(f"baseline_mae={baseline_mae:.4f} (median predictor)", flush=True)

    # --- 1. Ridge regression, LOO by design ---
    ridge_skills = []
    for held in designs_list:
        m = design_keys != held
        if m.sum() <= 10:
            continue
        Xtr, ytr = X[m], y[m]
        Xte, yte = X[~m], y[~m]
        # ridge closed form
        lam = 1.0
        try:
            beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(n_feats), Xtr.T @ ytr)
            yp = Xte @ beta
        except np.linalg.LinAlgError:
            continue
        mae = float(np.mean(np.abs(yte - yp)))
        bw = float(np.mean(np.abs(yte - np.median(ytr))))
        if bw > 0:
            ridge_skills.append(1.0 - mae / bw)
    ridge_skills = np.array(ridge_skills)
    print(f"ridge LOO: mean_skill={ridge_skills.mean():+.4f} "
          f"median={np.median(ridge_skills):+.4f} "
          f"pct_positive={(ridge_skills > 0).mean():.3f} "
          f"min={ridge_skills.min():+.4f} max={ridge_skills.max():+.4f}",
          flush=True)

    # --- 2. GBDT (LightGBM), LOO by design ---
    try:
        import lightgbm as lgb
        gbdt_skills = []
        for held in designs_list:
            m = design_keys != held
            if m.sum() <= 10:
                continue
            Xtr, ytr = X[m], y[m]
            Xte, yte = X[~m], y[~m]
            model = lgb.LGBMRegressor(n_estimators=200, max_depth=3,
                                      random_state=20260816, verbose=-1)
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)
            mae = float(np.mean(np.abs(yte - yp)))
            bw = float(np.mean(np.abs(yte - np.median(ytr))))
            if bw > 0:
                gbdt_skills.append(1.0 - mae / bw)
        gbdt_skills = np.array(gbdt_skills)
        print(f"GBDT LOO:  mean_skill={gbdt_skills.mean():+.4f} "
              f"median={np.median(gbdt_skills):+.4f} "
              f"pct_positive={(gbdt_skills > 0).mean():.3f} "
              f"min={gbdt_skills.min():+.4f} max={gbdt_skills.max():+.4f}",
              flush=True)
    except ImportError:
        print("lightgbm not available, skipping GBDT", flush=True)
        gbdt_skills = None

    # --- 3. In-sample performance (upper bound) ---
    # ridge in-sample
    lam = 1.0
    beta = np.linalg.solve(X.T @ X + lam * np.eye(n_feats), X.T @ y)
    yp_ins = X @ beta
    mae_ridge_ins = float(np.mean(np.abs(y - yp_ins)))
    skill_ridge_ins = 1.0 - mae_ridge_ins / baseline_mae
    print(f"ridge in-sample: skill={skill_ridge_ins:+.4f}", flush=True)

    # --- 4. Feature importance (GBDT) ---
    feat_names = [
        "react_i", "react_j", "err_i", "err_j",
        "sigA_i", "sigA_j", "sigB_i", "sigB_j",
        "dbl_i", "dbl_j", "dbl_err_i", "dbl_err_j",
        "str_pa_i", "str_pa_j", "str_dp_i", "str_dp_j",
        "edit_dist", "delta_si", "delta_sj", "delta_di", "delta_dj",
        "rescue_i", "rescue_j", "mutA", "mutB",
    ]
    feat_imp = {}
    if gbdt_skills is not None and len(gbdt_skills) > 0:
        # fit on all data for importance
        gbdt_all = lgb.LGBMRegressor(n_estimators=200, max_depth=3,
                                      random_state=20260816, verbose=-1)
        gbdt_all.fit(X, y)
        imp = gbdt_all.feature_importances_
        for name, v in sorted(zip(feat_names, imp), key=lambda x: -x[1]):
            feat_imp[name] = int(v)

    report = {
        "schema": "reactflow_delta.m2r_learnability.v1",
        "dataset": "OpenKnot_M2R", "n_samples": int(len(y)),
        "n_designs": n_designs, "n_feats": n_feats,
        "baseline_mae": float(baseline_mae),
        "ridge_loo": {
            "mean_skill": float(ridge_skills.mean()),
            "median_skill": float(np.median(ridge_skills)),
            "pct_positive": float((ridge_skills > 0).mean()),
            "min_skill": float(ridge_skills.min()),
            "max_skill": float(ridge_skills.max()),
            "n_folds": int(len(ridge_skills)),
        },
        "gbdt_loo": {
            "mean_skill": float(gbdt_skills.mean()) if gbdt_skills is not None else None,
            "median_skill": float(np.median(gbdt_skills)) if gbdt_skills is not None else None,
            "pct_positive": float((gbdt_skills > 0).mean()) if gbdt_skills is not None else None,
            "min_skill": float(gbdt_skills.min()) if gbdt_skills is not None else None,
            "max_skill": float(gbdt_skills.max()) if gbdt_skills is not None else None,
            "n_folds": int(len(gbdt_skills)) if gbdt_skills is not None else 0,
        } if gbdt_skills is not None else None,
        "ridge_in_sample": {"skill": float(skill_ridge_ins)},
        "feature_importance": feat_imp,
        "feature_names": feat_names,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2r_learnability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nDONE -> {out / 'm2r_learnability.json'}")


if __name__ == "__main__":
    sys.exit(main())