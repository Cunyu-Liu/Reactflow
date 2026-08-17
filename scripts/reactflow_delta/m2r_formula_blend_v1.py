#!/usr/bin/env python3
"""m2r_formula_blend_v1.py — physics-constrained 4th blend member (formula lever).

MOTIVATION (method-level, refines the rD finding):
The rD auxiliary test (m2r_doublemut_pred_v1.py) showed the double-mutant RMSD
numerator rD is ~49% predictable from LEGAL inputs (corr 0.70), but feeding
rD_pred as a 259th FEATURE hurt the strong 3-way (-0.40pp).  A cleaner
hypothesis: rather than injecting rD_pred into the tree-split space, build a
SEPARATE physics-constrained predictor using the exact rescue formula

    rescue = 1 - rD / sqrt(rA^2 + rB^2)

with rD replaced by its leak-free OOF prediction rD_pred and the denominator
sqrt(rA^2+rB^2) computed EXACTLY from the single-mutant RMSDs (legal).  This
"formula member" f = 1 - rD_pred/rnorm is a fixed functional form (no training,
no leak), structurally different from what the GBDT/Ridge blend computes, so if
the rD signal that the 3-way misses is real, blending at the OUTPUT level should
capture it via error decorrelation — the same mechanism that made L1+L2+Ridge
(and cross-objective) ensembles pay off.

Method (leak-free, exchangeable unit = design):
  * reuses m2r_doublemut_pred_oof.npz: blend_base (v1+v2 strong 3-way OOF),
    rD_pred (design-level LOO OOF from legal features), y, keys
  * recomputes rA/rB/rnorm + rescue_exact from the CSV (cheap, no training)
    and VERIFIES npz ordering against the exact rescue formula (corr ~ 1.0)
  * formula member f = clip(1 - rD_pred/rnorm, -1, 1)
  * standalone skill/R2 of f, correlation with blend_base (decorrelation),
  * blend sweep: (1-w)*blend_base + w*f  for w in [0, .05, .1, .15, .2, .25]
  * LOO-exclusion gain at the a-priori headline weight w=0.10
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _prof(p):
    return np.array([x if x is not None and np.isfinite(x) else np.nan
                     for x in p], dtype=np.float64)


def _region_mask(n, sub_start, sub_end):
    m = np.zeros(n, dtype=bool)
    lo = max(sub_start - 1, 0) if sub_start is not None else 0
    hi = sub_end if sub_end is not None else n
    m[lo:hi] = True
    return m


def _rmsd(a, b, mask):
    m = np.isfinite(a) & np.isfinite(b) & mask
    if m.sum() < 3:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def per_sample_geometry(s) -> dict:
    n = len(s.wt_reactivity)
    mask = _region_mask(n, s.sub_start, s.sub_end)
    wt = _prof(s.wt_reactivity)
    ra = _prof(s.singleA_reactivity)
    rb = _prof(s.singleB_reactivity)
    rd = _prof(s.double_reactivity)
    rA = _rmsd(wt, ra, mask)
    rB = _rmsd(wt, rb, mask)
    rD = _rmsd(wt, rd, mask)
    rnorm = np.sqrt(rA ** 2 + rB ** 2)
    rescue_exact = (1.0 - rD / rnorm) if (np.isfinite(rnorm) and rnorm > 1e-9) else np.nan
    return {"rA": rA, "rB": rB, "rD": rD, "rnorm": rnorm,
            "rescue_exact": rescue_exact}


def run_formula_blend(npz: str, m2r_csv: str, args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    z = np.load(npz, allow_pickle=True)
    blend_base = np.asarray(z["blend_base"], dtype=np.float64)
    rD_pred = np.asarray(z["rD_pred"], dtype=np.float64)
    y = np.asarray(z["y"], dtype=np.float64)
    keys = np.asarray(z["keys"])

    designs, meta = m2r.parse_m2r_csv(m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    if len(samples) != len(y):
        raise SystemExit(f"[fb] sample count mismatch: {len(samples)} vs {len(y)}")

    geo = [per_sample_geometry(s) for s in samples]
    rA = np.array([g["rA"] for g in geo])
    rB = np.array([g["rB"] for g in geo])
    rD_true = np.array([g["rD"] for g in geo])
    rnorm = np.array([g["rnorm"] for g in geo])
    rescue_exact = np.array([g["rescue_exact"] for g in geo])

    fin = np.isfinite(rescue_exact) & np.isfinite(y)
    # ordering/alignment verification: recomputed exact rescue vs npz target
    align_corr = float(np.corrcoef(rescue_exact[fin], y[fin])[0, 1])
    print(f"[fb] alignment check: n={len(y)} exact-vs-npz corr={align_corr:.6f}",
          flush=True)
    if align_corr < 0.999:
        raise SystemExit(f"[fb] alignment check failed corr={align_corr:.6f}")

    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))

    # ---- formula member: leak-free, fixed functional form ----
    f = np.where(np.isfinite(rnorm) & (rnorm > 1e-9),
                 1.0 - rD_pred / rnorm, np.median(y))
    f = np.clip(f, -1.0, 1.0)

    dec = float(np.corrcoef(blend_base, f)[0, 1])
    res = {
        "formula_member": {
            "mae": _mae(y, f),
            "skill": _skill(_mae(y, f), mae_bl),
            "r2": _r2(y, f),
        },
        "decorrelation_corr_blend_vs_formula": dec,
        "align_corr_exact_vs_npz": align_corr,
    }

    # ---- blend sweep (all members are design-level OOF: no leak) ----
    best = None
    sweep = {}
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        b = (1.0 - w) * blend_base + w * f
        mae = _mae(y, b); sk = _skill(mae, mae_bl); r2 = _r2(y, b)
        sweep[f"w{w:.2f}"] = {"mae": mae, "skill": sk, "r2": r2,
                              "gain_pp": (sk - res["decorrelation_corr_blend_vs_formula"] * 0) }
        sweep[f"w{w:.2f}"].pop("gain_pp")
        if best is None or sk > best[1]:
            best = (w, sk)

    # headline: fixed a-priori w = 0.10
    W_HEAD = 0.10
    blend_head = (1.0 - W_HEAD) * blend_base + W_HEAD * f
    mae_head = _mae(y, blend_head)
    skill_head = _skill(mae_head, mae_bl)
    r2_head = _r2(y, blend_head)

    # LOO-exclusion gain (formula blend vs base) at a-priori w
    gains = []
    des_list = sorted(set(keys.tolist()))
    for held in des_list:
        m = keys != held
        if m.sum() < 10:
            continue
        gains.append(_skill(_mae(y[m], blend_head[m]), mae_bl) -
                     _skill(_mae(y[m], blend_base[m]), mae_bl))
    gains = np.array(gains)

    report = {
        "schema": "reactflow_delta.m2r_formula_blend.v1",
        "dataset": "OpenKnot_M2R", "exchangeable_unit": "design",
        "n_samples": int(len(y)), "n_designs": len(des_list),
        "baseline_mae": mae_bl,
        "method": ("physics-constrained 4th member f = clip(1 - rD_pred/rnorm, -1, 1); "
                   "rD_pred leak-free design-level OOF; rnorm exact from single-mutant RMSDs"),
        "headline_weight_w": W_HEAD,
        "results": {
            "blend_base": {"mae": _mae(y, blend_base),
                           "skill": _skill(_mae(y, blend_base), mae_bl),
                           "r2": _r2(y, blend_base)},
            "formula_blend": {"mae": mae_head, "skill": skill_head, "r2": r2_head},
        },
        "formula_member": res["formula_member"],
        "decorrelation_corr": dec,
        "align_corr_exact_vs_npz": align_corr,
        "sweep": sweep,
        "best_w": best[0],
    }
    # pooled gain vs base
    g = float((_skill(mae_head, mae_bl) - _skill(_mae(y, blend_base), mae_bl)) * 100)
    report["formula_blend_gain"] = {
        "pooled_gain_pp": g,
        "r2_gain": float(r2_head - _r2(y, blend_base)),
        "loo_exclusion": {
            "gain_mean_pp": float(gains.mean() * 100),
            "gain_min_pp": float(gains.min() * 100),
            "gain_max_pp": float(gains.max() * 100),
            "pct_positive": float((gains > 0).mean()),
            "n_folds": int(len(gains)),
        },
    }
    report["results"]["formula_blend"]["pooled_gain_pp_vs_base"] = g

    (out / "m2r_formula_blend_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_formula_blend_oof.npz",
             blend_base=blend_base, formula_member=f, blend_head=blend_head,
             rD_pred=rD_pred, rnorm=rnorm, y=y, keys=keys)

    print(f"\n=== M2R physics-constrained formula blend (w={W_HEAD}) ===")
    print(f"align corr (exact vs npz) = {align_corr:.6f}")
    print(f"formula member alone : skill={res['formula_member']['skill']:+.4f} "
          f"R2={res['formula_member']['r2']:.4f}")
    print(f"decorrelation corr(blend_base, f) = {dec:.4f}")
    print(f"blend base           : skill={report['results']['blend_base']['skill']:+.4f} "
          f"R2={report['results']['blend_base']['r2']:.4f}")
    print(f"formula blend (w={W_HEAD}): skill={skill_head:+.4f} R2={r2_head:.4f}")
    print(f"gain: {g:+.2f}pp (R2 {report['formula_blend_gain']['r2_gain']:+.4f})")
    loo = report["formula_blend_gain"]["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"sweep best w={best[0]} skill={best[1]:+.4f}")
    print(f"wall={time.time()-t0:.0f}s DONE -> {out / 'm2r_formula_blend_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="m2r_doublemut_pred_oof.npz (blend_base, rD_pred, y, keys)")
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_formula_blend(args.npz, args.m2r_csv, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
