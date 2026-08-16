#!/usr/bin/env python3
"""m2_bias_diagnostic.py — diagnose systematic bias structure of the M2
mu-ensemble predictions, to evaluate whether POST-HOC CALIBRATION (the only
statistically-significant method lever in the junction project) can improve the
WMAE skill of the response-spectrum models.

We analyze the v5 (2-layer attention, best single model) predictions:

  resid = y - pred          (signed per-position residual, pooled)
  adt   = |y - prior|       (true deviation from sequence-free prior)

Diagnostics:
  1. PER-POSITION bias      : mean(resid) and |resid| at each of the W=21 window
                              positions.  If some positions are systematically
                              over/under-shot, a per-position additive or
                              multiplicative correction helps.
  2. PER-DESIGN bias        : mean(resid) per (puzzle x method) design.  Large
                              spread => a per-design additive shift (estimated
                              without leakage) helps.
  3. SPLIT-HALF stability   : for each design/position with >= N observations,
                              correlate the bias estimate across two random
                              halves.  rho>>0 => stable, real signal (calibratable);
                              rho~0  => noise (no headroom).
  4. BIAS vs skill headroom : how much WMAE would drop if we removed the
                              estimated bias exactly (upper bound for calibration).

OUTPUT: /mnt/.../m2_bias_diagnostic.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--variant", default="wmae_resid_attn_spectrum")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-per-group", type=int, default=30,
                    help="min obs per design/position for a stable bias estimate")
    ap.add_argument("--split-seed", type=int, default=20260816)
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    base, model = _unroll(rows, args.variant)
    common = [k for k in base if len(model.get(k, {})) == len(SEEDS)]

    # --- pool per-position observations ---
    # pos_bias[pos] = array of signed residuals (for WMAE-relevant positions)
    pos_resid = [[] for _ in range(W)]
    pos_abs = [[] for _ in range(W)]
    # design-level
    design_resid = defaultdict(list)
    # (design, position) cells for context-level
    cell_resid = defaultdict(list)

    y_all, p_all, b_all = [], [], []
    for k in common:
        b = base[k]
        ens = np.mean([model[k][s] for s in SEEDS], axis=0)
        if len(b["y"]) != len(ens):
            continue
        d = k.split(":")[0]
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            r = b["y"][j] - ens[j]
            pos_resid[j].append(r)
            pos_abs[j].append(abs(r))
            design_resid[d].append(r)
            cell_resid[(d, j)].append(r)
            y_all.append(b["y"][j]); p_all.append(ens[j]); b_all.append(b["pred"][j])

    y_all = np.asarray(y_all); p_all = np.asarray(p_all); b_all = np.asarray(b_all)
    resid_all = y_all - p_all

    def wmae(y, p):
        return float(np.mean(np.abs(y - p)))

    wmae_baseline = wmae(y_all, b_all)
    wmae_model = wmae(y_all, p_all)
    skill = 1.0 - wmae_model / wmae_baseline

    # --- 1. per-position bias ---
    pos = []
    for j in range(W):
        r = np.asarray(pos_resid[j])
        if len(r) == 0:
            continue
        pos.append({
            "position": j,
            "n": int(len(r)),
            "mean_resid": float(r.mean()),
            "std_resid": float(r.std()),
            "mean_abs_resid": float(np.abs(r).mean()),
            "rmse_resid": float(np.sqrt(np.mean(r ** 2))),
        })

    # --- 2. per-design bias + split-half stability ---
    d_res = []
    d_stable = []
    rng = np.random.default_rng(args.split_seed)
    for d, rs in sorted(design_resid.items()):
        r = np.asarray(rs)
        rec = {"design": d, "n": int(len(r)), "mean_resid": float(r.mean()),
               "std": float(r.std())}
        d_res.append(rec)
        if len(r) >= args.min_per_group * 2:
            perm = rng.permutation(len(r))
            n2 = len(r) // 2
            h1 = r[perm[:n2]]; h2 = r[perm[n2:2*n2]]
            if h1.std() > 0 and h2.std() > 0:
                d_stable.append((d, float(np.corrcoef(h1, h2)[0, 1])))
    d_means = np.array([x["mean_resid"] for x in d_res])
    d_std = float(d_means.std())
    d_split = [s for _, s in d_stable]
    d_split_rho = float(np.median(d_split)) if d_split else None
    d_split_n = len(d_split)

    # --- 3. per-(design,position) cell bias + split-half ---
    c_res = []
    c_stable = []
    for (d, j), rs in sorted(cell_resid.items()):
        r = np.asarray(rs)
        rec = {"design": d, "position": j, "n": int(len(r)),
               "mean_resid": float(r.mean())}
        c_res.append(rec)
        if len(r) >= args.min_per_group * 2:
            perm = rng.permutation(len(r))
            n2 = len(r) // 2
            h1 = r[perm[:n2]]; h2 = r[perm[n2:2*n2]]
            if h1.std() > 0 and h2.std() > 0:
                c_stable.append((d, j, float(np.corrcoef(h1, h2)[0, 1])))
    c_means = np.array([x["mean_resid"] for x in c_res])
    c_std = float(c_means.std())
    c_split = [s for _, _, s in c_stable]
    c_split_rho = float(np.median(c_split)) if c_split else None
    c_split_n = len(c_split)

    # --- 4. calibration headroom (remove estimated biases exactly) ---
    # 4a. per-position additive
    pos_shift = {x["position"]: x["mean_resid"] for x in pos}
    p_cal_pos = p_all.copy()
    idx = 0
    # rebuild position indices to apply shift (we stored in pooled order)
    # re-iterate
    p_cal_pos = []
    for k in common:
        b = base[k]
        ens = np.mean([model[k][s] for s in SEEDS], axis=0)
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            p_cal_pos.append(ens[j] + pos_shift[j])
    p_cal_pos = np.asarray(p_cal_pos)
    skill_cal_pos = 1.0 - wmae(y_all, p_cal_pos) / wmae_baseline

    # 4b. per-design additive (in-sample upper bound, NOT a deployable method)
    d_shift = {x["design"]: x["mean_resid"] for x in d_res}
    p_cal_d = []
    for k in common:
        b = base[k]
        ens = np.mean([model[k][s] for s in SEEDS], axis=0)
        d = k.split(":")[0]
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            p_cal_d.append(ens[j] + d_shift[d])
    p_cal_d = np.asarray(p_cal_d)
    skill_cal_d = 1.0 - wmae(y_all, p_cal_d) / wmae_baseline

    # 4c. per-(design,position) additive (in-sample upper bound)
    c_shift = {(x["design"], x["position"]): x["mean_resid"] for x in c_res}
    p_cal_c = []
    for k in common:
        b = base[k]
        ens = np.mean([model[k][s] for s in SEEDS], axis=0)
        d = k.split(":")[0]
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            p_cal_c.append(ens[j] + c_shift[(d, j)])
    p_cal_c = np.asarray(p_cal_c)
    skill_cal_c = 1.0 - wmae(y_all, p_cal_c) / wmae_baseline

    summary = {
        "schema": "reactflow_delta.response_spectrum.m2_bias_diagnostic.v1",
        "dataset": "OpenKnot_M2", "variant": args.variant,
        "exchangeable_unit": "puzzle_x_method_design",
        "n_pairs": len(common), "n_positions_pooled": int(len(y_all)),
        "wmae_baseline": float(wmae_baseline), "wmae_model": float(wmae_model),
        "skill": float(skill),
        "per_position_bias": pos,
        "per_design_bias": {
            "n_designs": len(d_res), "mean": float(d_means.mean()),
            "std_of_means": d_std, "abs_mean_of_means": float(np.abs(d_means).mean()),
            "split_half_median_rho": d_split_rho, "split_half_n_stable": d_split_n,
            "max_abs": float(np.abs(d_means).max()),
        },
        "per_cell_bias": {
            "n_cells": len(c_res), "mean": float(c_means.mean()),
            "std_of_means": c_std, "abs_mean_of_means": float(np.abs(c_means).mean()),
            "split_half_median_rho": c_split_rho, "split_half_n_stable": c_split_n,
            "max_abs": float(np.abs(c_means).max()),
        },
        "calibration_headroom_in_sample_upper_bound": {
            "per_position_additive": skill_cal_pos,
            "per_design_additive": skill_cal_d,
            "per_cell_additive": skill_cal_c,
        },
        "note": ("in-sample upper bounds show how much skill a PERFECT (leaky) "
                 "additive calibration could add; the real method must estimate "
                 "shifts without leakage (out-of-fold / split-half)"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2_bias_diagnostic.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2 bias diagnostic ===")
    print(f"n_pairs={len(common)} n_positions={len(y_all)} skill={skill:+.4f}")
    print(f"wmae baseline={wmae_baseline:.4f} model={wmae_model:.4f}")
    print("\n-- per-position bias (first 21) --")
    print(f"  {'pos':>4} {'n':>7} {'mean_resid':>11} {'abs_resid':>10} {'rmse':>9}")
    for x in pos:
        print(f"  {x['position']:>4} {x['n']:>7} {x['mean_resid']:>+11.4f} "
              f"{x['mean_abs_resid']:>10.4f} {x['rmse_resid']:>9.4f}")
    print("\n-- per-design bias --")
    print(f"  std_of_means={d_std:.4f} abs_mean={np.abs(d_means).mean():.4f} "
          f"max_abs={np.abs(d_means).max():.4f} split_half_rho={d_split_rho} (n={d_split_n})")
    print("\n-- per-(design,pos) cell bias --")
    print(f"  std_of_means={c_std:.4f} abs_mean={np.abs(c_means).mean():.4f} "
          f"max_abs={np.abs(c_means).max():.4f} split_half_rho={c_split_rho} (n={c_split_n})")
    print("\n-- calibration headroom (in-sample upper bound) --")
    print(f"  per_position_additive: skill={skill_cal_pos:+.4f} (gain {skill_cal_pos-skill:+.4f})")
    print(f"  per_design_additive  : skill={skill_cal_d:+.4f} (gain {skill_cal_d-skill:+.4f})")
    print(f"  per_cell_additive    : skill={skill_cal_c:+.4f} (gain {skill_cal_c-skill:+.4f})")
    print(f"DONE -> {out / 'm2_bias_diagnostic.json'}")


if __name__ == "__main__":
    sys.exit(main())
