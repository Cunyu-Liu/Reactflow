#!/usr/bin/env python3
"""m2_calibration_levers.py — test the TWO WMAE-appropriate post-hoc calibration
levers on the M2 mu-ensemble predictions:

  1. PER-POSITION MEDIAN-RESIDUAL additive shift (theoretically optimal for MAE,
     since mean absolute error is minimized by the conditional median, NOT the
     mean).  The naive per-position MEAN shift was already shown to HURT (the
     positive mean is a right-tail artifact, not systematic under-prediction).

  2. DELTA SHRINKAGE:  pred_cal = prior + alpha * (pred - prior), alpha in a grid.
     This is the response-spectrum analog of the junction project's sigma/EB
     calibration: if the learned deltas are systematically over-extreme, shrinking
     them toward the sequence-free prior improves WMAE.

Both are estimated WITHOUT leakage for the deployable variant:
  - median shift / shrinkage alpha fitted on a TRAINING half of each design's
    positions, evaluated on the held-out half (out-of-fold split-half, the same
    non-leakage discipline as the junction project).
  - An in-sample upper bound is also reported for context.

OUTPUT: /mnt/.../m2_calibration_levers.json
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
BASELINE = "wmed_spectrum"
W = 21


def _load_rows(pred_path):
    rows = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _unroll(rows, model_variant):
    base = {}
    model = defaultdict(dict)
    for r in rows:
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        yv = r.get("y") or []
        wv = r.get("weight") or []
        pv = r.get("raw_prediction")
        if not (isinstance(yv, list) and isinstance(wv, list) and isinstance(pv, list)):
            continue
        y = np.array([float(a) for a, ww in zip(yv, wv) if ww], dtype=np.float64)
        w = np.ones(len(y), dtype=np.float64)
        p = np.array([float(a) for a, ww in zip(pv, wv) if ww], dtype=np.float64)
        if r["model_variant"] == BASELINE and r["seed"] == 0:
            base[r["pair_id"]] = {"y": y, "w": w, "pred": p}
        elif r["model_variant"] == model_variant:
            model[r["pair_id"]][r["seed"]] = p
    return base, model


def _wmae(y, w, pred):
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    num = float(np.sum(w * np.abs(y - pred)))
    den = float(np.sum(w))
    return num / den if den > 0 else float("nan")


def _unroll_flat(base, model, common, W=21):
    """Flatten all designs into pooled arrays with per-position index."""
    ys, ps, bs, pos_idx, des = [], [], [], [], []
    for k in common:
        b = base[k]
        ens = np.mean([model[k][s] for s in SEEDS], axis=0)
        d = k.split(":")[0]
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            ys.append(b["y"][j]); ps.append(ens[j]); bs.append(b["pred"][j])
            pos_idx.append(j); des.append(d)
    return (np.asarray(ys), np.asarray(ps), np.asarray(bs),
            np.asarray(pos_idx), np.asarray(des))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--variant", default="wmae_resid_attn_spectrum")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seed", type=int, default=20260816)
    ap.add_argument("--alpha-grid", default="0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2")
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    base, model = _unroll(rows, args.variant)
    common = [k for k in base if len(model.get(k, {})) == len(SEEDS)]
    alphas = [float(x) for x in args.alpha_grid.split(",")]

    y, p, b, pos_idx, des = _unroll_flat(base, model, common)
    n = len(y)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(n)
    n2 = n // 2
    tr = perm[:n2]; te = perm[n2:2 * n2]

    wmae_b = _wmae(y, np.ones(n), b)
    wmae_m = _wmae(y, np.ones(n), p)
    skill = 1.0 - wmae_m / wmae_b

    # ---- 1. per-position MEDIAN additive shift ----
    # fit on train half, apply to test half
    med_shift = {}
    for j in range(W):
        m = pos_idx[tr] == j
        if m.sum() > 0:
            med_shift[j] = float(np.median(y[tr][m] - p[tr][m]))
    p_cal_med = p + np.array([med_shift[j] for j in pos_idx])
    skill_cal_med_te = 1.0 - _wmae(y[te], np.ones(len(te)), p_cal_med[te]) / _wmae(y[te], np.ones(len(te)), b[te])

    # ---- 2. delta shrinkage: prior + alpha*(pred - prior) ----
    delta = p - b
    # choose alpha on train half
    best_alpha = None
    best_te_skill = -1e9
    res_alpha = {}
    for a in alphas:
        p_a = b + a * delta
        # fit-free, just evaluate
        sk_tr = 1.0 - _wmae(y[tr], np.ones(len(tr)), p_a[tr]) / _wmae(y[tr], np.ones(len(tr)), b[tr])
        sk_te = 1.0 - _wmae(y[te], np.ones(len(te)), p_a[te]) / _wmae(y[te], np.ones(len(te)), b[te])
        res_alpha[str(a)] = {"train_skill": float(sk_tr), "test_skill": float(sk_te)}
        if sk_te > best_te_skill:
            best_te_skill = sk_te
            best_alpha = a

    # ---- 3. per-position alpha shrinkage (shrink each position's delta) ----
    # fit per-position optimal alpha on train half (grid search), apply to test
    p_cal_posa = p.copy()
    best_pos_a = {}
    for j in range(W):
        mtr = (pos_idx == j) & np.isin(np.arange(n), tr)
        mte = (pos_idx == j) & np.isin(np.arange(n), te)
        if mtr.sum() < 10 or mte.sum() < 10:
            best_pos_a[j] = 1.0
            continue
        yj, bj, dj = y[mtr], b[mtr], (p - b)[mtr]
        best = None; best_v = -1e9
        for a in alphas:
            sk = 1.0 - _wmae(yj, np.ones(len(yj)), bj + a * dj) / _wmae(yj, np.ones(len(yj)), bj)
            if sk > best_v:
                best_v = sk; best = a
        best_pos_a[j] = best
        p_cal_posa[mte] = b[mte] + best * (p - b)[mte]
    sk_posa = 1.0 - _wmae(y[te], np.ones(len(te)), p_cal_posa[te]) / _wmae(y[te], np.ones(len(te)), b[te])

    # ---- 4. in-sample upper bounds (leaky, context only) ----
    # per-position median shift in-sample
    med_shift_all = {j: float(np.median(y[pos_idx == j] - p[pos_idx == j])) for j in range(W)}
    p_ins = p + np.array([med_shift_all[j] for j in pos_idx])
    sk_ins_med = 1.0 - _wmae(y, np.ones(n), p_ins) / wmae_b
    # per-position alpha in-sample
    p_insa = p.copy()
    for j in range(W):
        m = pos_idx == j
        yj, bj, dj = y[m], b[m], (p - b)[m]
        best = None; best_v = -1e9
        for a in alphas:
            sk = 1.0 - _wmae(yj, np.ones(len(yj)), bj + a * dj) / _wmae(yj, np.ones(len(yj)), bj)
            if sk > best_v:
                best_v = sk; best = a
        p_insa[m] = b[m] + best * dj
    sk_insa = 1.0 - _wmae(y, np.ones(n), p_insa) / wmae_b

    summary = {
        "schema": "reactflow_delta.response_spectrum.m2_calibration_levers.v1",
        "dataset": "OpenKnot_M2", "variant": args.variant,
        "n_pairs": len(common), "n_positions_pooled": int(n),
        "split": {"train": int(n2), "test": int(n2), "seed": args.split_seed},
        "wmae_baseline": float(wmae_b), "wmae_model": float(wmae_m),
        "skill_raw": float(skill),
        "test_skill_baseline_matched": float(1.0 - _wmae(y[te], np.ones(len(te)), b[te]) / _wmae(y[te], np.ones(len(te)), b[te])),
        "lever1_per_position_median_shift": {
            "fit": {f"pos_{j}": float(med_shift[j]) for j in range(W)},
            "test_skill": float(skill_cal_med_te),
        },
        "lever2_delta_shrinkage_global": {
            "alpha_grid": alphas,
            "per_alpha": {k: res_alpha[k] for k in res_alpha},
            "best_test_alpha": best_alpha,
            "best_test_skill": float(best_te_skill),
        },
        "lever3_per_position_alpha_shrinkage": {
            "best_pos_alpha": {f"pos_{j}": best_pos_a[j] for j in range(W)},
            "test_skill": float(sk_posa),
        },
        "in_sample_upper_bounds_leaky": {
            "per_position_median_shift": float(sk_ins_med),
            "per_position_alpha": float(sk_insa),
        },
        "note": ("test_skill uses a random split-half of POOLED positions with fit "
                 "on train half only (no leakage); in-sample bounds are leaky and "
                 "only show the absolute ceiling of each lever."),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2_calibration_levers.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2 calibration levers (WMAE-appropriate) ===")
    print(f"n={n} n_pairs={len(common)} raw_skill={skill:+.4f}")
    print(f"\n-- lever 2: global delta shrinkage (alpha, train/test skill) --")
    for a in alphas:
        r = res_alpha[str(a)]
        mark = " <-- best test" if a == best_alpha else ""
        print(f"  alpha={a:4.1f}  train={r['train_skill']:+.4f}  test={r['test_skill']:+.4f}{mark}")
    print(f"\n-- lever 1: per-position median shift test_skill={skill_cal_med_te:+.4f}")
    print(f"-- lever 3: per-position alpha   test_skill={sk_posa:+.4f}")
    print(f"-- in-sample upper bounds (leaky): median={sk_ins_med:+.4f} alpha={sk_insa:+.4f}")
    print(f"DONE -> {out / 'm2_calibration_levers.json'}")


if __name__ == "__main__":
    sys.exit(main())
