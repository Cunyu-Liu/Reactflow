#!/usr/bin/env python3
"""m2r_transfer_v1.py — cross-task transfer: M2 model predictions as M2R features.

Motivation: M2 and M2R are related OpenKnot experiments on the SAME
(puzzle, method) designs.  The M2 response-spectrum model predicts the
single-mutant SHAPE response profile (a 21-nt window around each edit site).
We use the M2 model's learned predictions (mu-ensemble over 5 seeds) as
TRANSFER FEATURES for the M2R rescue_factor task:

  For an M2R pair (i, j):
    m2_center_A : M2 predicted response at the single-A edit site i
    m2_center_B : M2 predicted response at the single-B edit site j
    m2_A_at_j   : M2 prediction from the A-window at the position of site j
    m2_B_at_i   : M2 prediction from the B-window at the position of site i
    m2_maxabs_A : max |M2 pred| over the A-window
    m2_maxabs_B : max |M2 pred| over the B-window

LEAK-FREE: both M2 and M2R use the SAME design partition (leave-one-design-out,
exchangeable unit = (puzzle, method)).  The M2 prediction for a held-out design
was produced by an M2 model trained on all OTHER designs, so transfer features
carry no label information about the held-out design.  We map each M2R sample's
design_id back to the M2 prediction file using the (puzzle, method) design key
and only read the OOF (held-out) predictions.

Run in design-level LOO for M2R, comparing:
  * existing features (230 dims, incl. M2_structure)
  * existing + 6 M2-transfer features
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf

SEED = 20260816
M2_VARIANT = "wmae_resid_attn_spectrum"
N_SEEDS = 5
W = 21          # M2 window length (odd)
CENTER = W // 2  # 10 -> the edit site is at window center


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def load_m2_oof(m2_pred_path: str) -> dict:
    """Load M2 mu-ensemble OOF predictions.

    Returns dict: design_id -> {mutA (1-indexed): mu-ensemble prediction array
    (len 21)}.  design_id format = "OK7a_M2_{puzzle}_{method}" (from pair_id,
    which is "{design_id}:{mutA}").
    """
    acc = defaultdict(lambda: defaultdict(list))  # design_id -> mutA -> [seeds]
    with open(m2_pred_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["model_variant"] != M2_VARIANT:
                continue
            pid = r["pair_id"]
            design_id, mutA_str = pid.rsplit(":", 1)
            mutA = int(mutA_str)
            arr = np.array(r["raw_prediction"], dtype=np.float64)
            if len(arr) != W:
                continue
            acc[design_id][mutA].append(arr)
    # average over seeds
    out = {}
    for did, mm in acc.items():
        out[did] = {ma: np.mean(seeds, axis=0) for ma, seeds in mm.items()}
    return out


def build_transfer_features(
    samples: list,
    m2_oof: dict,
    m2_design_key: dict,
) -> np.ndarray:
    """Build M2 transfer features for each M2R pair sample.

    m2_design_key maps (puzzle, method) -> M2 design_id ("OK7a_M2_...").
    For each sample with edit sites i,j (full-seq 0-indexed) and pair sites
    mutA/mutB (1-indexed design positions), we fetch the M2 prediction for the
    design at mutA (window centered on site i) and mutB (window centered on j).

    The window position of site j within the A-window (centered on i) is
      p_j = CENTER + (editB_seq_pos - editA_seq_pos).
    Returns (n_samples, 6) array.
    """
    n = len(samples)
    out = np.zeros((n, 6), dtype=np.float64)
    for idx, s in enumerate(samples):
        m2id = m2_design_key.get((s.puzzle, s.method))
        if m2id is None:
            continue
        mm = m2_oof.get(m2id)
        if mm is None:
            continue
        predA = mm.get(s.mutA)   # window for single-A (center = site i)
        predB = mm.get(s.mutB)   # window for single-B (center = site j)
        if predA is not None:
            out[idx, 0] = predA[CENTER]                     # response at site i
            p_j = CENTER + (s.editB_seq_pos - s.editA_seq_pos)
            if 0 <= p_j < W:
                out[idx, 2] = predA[p_j]                     # A-window at site j
            out[idx, 4] = float(np.abs(predA).max())
        if predB is not None:
            out[idx, 1] = predB[CENTER]                     # response at site j
            p_i = CENTER + (s.editA_seq_pos - s.editB_seq_pos)
            if 0 <= p_i < W:
                out[idx, 3] = predB[p_i]                     # B-window at site i
            out[idx, 5] = float(np.abs(predB).max())
    return out


def _loo_gbdt(X, y, keys, n_estimators=100, max_depth=3):
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        import lightgbm as lgb
        g = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=SEED, verbose=-1, n_jobs=2)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred", required=True,
                    help="M2 keyed_predictions jsonl (attn variant)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load M2R ----
    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, names = m2rf.build_all(samples)
    keys = np.array(keys)
    des_list = sorted(set(keys.tolist()))
    print(f"[m2r_tr] n_samples={len(y)} n_designs={len(des_list)} "
          f"X={X.shape}", flush=True)

    # ---- load M2 OOF predictions ----
    print(f"[m2r_tr] loading M2 OOF preds from {args.m2_pred} ...", flush=True)
    m2_oof = load_m2_oof(args.m2_pred)
    print(f"[m2r_tr] M2 designs with preds: {len(m2_oof)}", flush=True)
    # map (puzzle, method) -> M2 design_id
    m2_design_key = {}
    for did in m2_oof:
        # "OK7a_M2_{puzzle}_{method}"
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            puzzle = parts[2]
            method = "_".join(parts[3:])
            m2_design_key[(puzzle, method)] = did
    print(f"[m2r_tr] mapped {len(m2_design_key)} designs", flush=True)

    # ---- build transfer features ----
    X_tr = build_transfer_features(samples, m2_oof, m2_design_key)
    print(f"[m2r_tr] X_tr={X_tr.shape}", flush=True)
    nz = (np.abs(X_tr).sum(axis=1) > 0).mean()
    print(f"[m2r_tr] transfer non-zero fraction: {nz:.3f}", flush=True)

    # ---- baseline ----
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[m2r_tr] baseline MAE={mae_bl:.4f}", flush=True)

    # ---- LOO: existing only ----
    t0 = time.time()
    pred_ex = _loo_gbdt(X, y, keys, args.trees, args.depth)
    wall_ex = time.time() - t0
    skill_ex = _skill(_mae(y, pred_ex), mae_bl)
    r2_ex = _r2(y, pred_ex)
    print(f"[m2r_tr] existing: skill={skill_ex:+.4f} R2={r2_ex:.4f} wall={wall_ex:.0f}s",
          flush=True)

    # ---- LOO: existing + transfer ----
    X_comb = np.concatenate([X, X_tr], axis=1)
    t0 = time.time()
    pred_comb = _loo_gbdt(X_comb, y, keys, args.trees, args.depth)
    wall_comb = time.time() - t0
    skill_comb = _skill(_mae(y, pred_comb), mae_bl)
    r2_comb = _r2(y, pred_comb)
    print(f"[m2r_tr] existing+transfer: skill={skill_comb:+.4f} R2={r2_comb:.4f} "
          f"wall={wall_comb:.0f}s", flush=True)

    # ---- Ridge LOO on combined features (for blend) ----
    from sklearn.linear_model import Ridge
    ridge_pred = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            ridge_pred[~m] = np.median(y)
            continue
        r = Ridge(alpha=1.0).fit(X_comb[m], y[m])
        ridge_pred[~m] = r.predict(X_comb[~m])
    mae_ridge = _mae(y, ridge_pred)
    skill_ridge = _skill(mae_ridge, mae_bl)
    r2_ridge = _r2(y, ridge_pred)
    print(f"[m2r_tr] ridge(combined): skill={skill_ridge:+.4f} R2={r2_ridge:.4f}",
          flush=True)

    # ---- GBDT+Ridge blend on combined features (a=0.80, a priori) ----
    blend_pred = 0.80 * pred_comb + 0.20 * ridge_pred
    skill_blend = _skill(_mae(y, blend_pred), mae_bl)
    r2_blend = _r2(y, blend_pred)
    print(f"[m2r_tr] blend(0.80) combined: skill={skill_blend:+.4f} R2={r2_blend:.4f}",
          flush=True)

    # ---- save OOF preds for audit ----
    np.savez(out / "m2r_transfer_oof.npz",
             pred_ex=pred_ex, pred_comb=pred_comb,
             ridge_comb=ridge_pred, blend_comb=blend_pred,
             y=y, keys=keys)

    # ---- LOO-exclusion robustness of the transfer gain ----
    gains = []
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            continue
        s_ex = _skill(_mae(y[m], pred_ex[m]), mae_bl)
        s_co = _skill(_mae(y[m], pred_comb[m]), mae_bl)
        gains.append(s_co - s_ex)
    gains = np.array(gains)

    report = {
        "schema": "reactflow_delta.m2r_transfer.v1",
        "dataset": "OpenKnot_M2R",
        "source_task": "M2 response-spectrum (attn v5)",
        "n_samples": len(y),
        "n_designs": len(des_list),
        "n_features_existing": int(X.shape[1]),
        "n_features_transfer": int(X_tr.shape[1]),
        "trees": args.trees,
        "depth": args.depth,
        "baseline_mae": mae_bl,
        "existing_only": {"skill": float(skill_ex), "r2": float(r2_ex)},
        "existing_plus_transfer": {"skill": float(skill_comb), "r2": float(r2_comb)},
        "ridge_combined": {"skill": float(skill_ridge), "r2": float(r2_ridge)},
        "gbdt_ridge_blend_combined_a80": {
            "skill": float(skill_blend), "r2": float(r2_blend)},
        "gain_from_transfer": {
            "skill_delta": float(skill_comb - skill_ex),
            "r2_delta": float(r2_comb - r2_ex),
        },
        "gain_from_transfer_blend": {
            "skill_delta": float(skill_blend - skill_ex),
            "r2_delta": float(r2_blend - r2_ex),
        },
        "loo_exclusion_gain": {
            "mean": float(gains.mean()),
            "min": float(gains.min()),
            "max": float(gains.max()),
            "pct_positive": float((gains > 0).mean()),
            "n_folds": int(len(gains)),
        },
        "transfer_nonzero_frac": float(nz),
        "wall_seconds": round(wall_ex + wall_comb, 1),
    }
    (out / "m2r_transfer_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("\n[m2r_tr] DONE ->", out / "m2r_transfer_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()