#!/usr/bin/env python3
"""compare_m2_spectrum — compare residual-learning vs prior-baseline on M2 spectrum.

Consumes the keyed_predictions_m2_spectrum.jsonl produced by
run_response_spectrum_m2_v1.py and reports:

* per-row weighted MAE (over eligible window positions) for each model variant;
* pooled (all-changers) and design-level (per held design) aggregates;
* residual model (mean of seeds 0..4) vs the wmed_spectrum prior baseline:
    improvement = baseline_wmae - resid_wmae  (>0 means residual helps);
* a per-design horizontal comparison table (baseline vs resid vs n_changers).

Also folds in the per-fold residual training diagnostics (final train MAE of the
model vs the prior, and |delta| growth) to explain WHY the model does / does not
beat the baseline.

Output is a compact JSON + printed tables.
"""
from __future__ import annotations

import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def row_wmae(y, w, pred) -> float | None:
    """Weighted MAE over eligible positions; None if no eligible position."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = w.sum()
    if denom <= 0:
        return None
    return float((w * np.abs(y - pred)).sum() / denom)


def load_predictions(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="keyed_predictions jsonl")
    ap.add_argument("--logdir", default=None, help="fold_logs dir for training diagnostics")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    rows = load_predictions(args.pred)
    # dict[model_variant][pair_id][seed] = wmae
    var_pairs: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    pair_fold: dict[str, str] = {}
    pair_src: dict[str, str] = {}
    n_no_call = 0
    for r in rows:
        if r["coverage_status"] != "CALLED":
            n_no_call += 1
            continue
        wmae = row_wmae(r["y"], r["weight"], r["raw_prediction"])
        if wmae is None:
            continue
        var = r["model_variant"]
        var_pairs[var][r["pair_id"]][int(r["seed"])] = wmae
        pair_fold[r["pair_id"]] = r["fold_id"]
        pair_src[r["pair_id"]] = r["source_accession"]

    # baseline (seed 0) and resid (mean of seeds 0..4)
    base = var_pairs["wmed_spectrum"]
    resid = var_pairs["wmae_resid_spectrum"]
    common = sorted(set(base) & set(resid))
    if not common:
        raise SystemExit("no common pairs between variants")

    def seed_mean(d: dict) -> float:
        return float(np.mean(list(d.values())))

    base_wmae = {pid: seed_mean(base[pid]) for pid in common}
    resid_wmae = {pid: seed_mean(resid[pid]) for pid in common}
    # per-seed resid for robustness
    resid_seeds = {pid: resid[pid] for pid in common}

    # ---- pooled aggregates ----
    b_arr = np.array([base_wmae[p] for p in common])
    r_arr = np.array([resid_wmae[p] for p in common])
    pooled = {
        "n_pairs": len(common),
        "baseline_wmae_mean": float(b_arr.mean()),
        "resid_wmae_mean": float(r_arr.mean()),
        "baseline_wmae_median": float(np.median(b_arr)),
        "resid_wmae_median": float(np.median(r_arr)),
        "improvement_mean": float((b_arr - r_arr).mean()),
        "improvement_median": float(np.median(b_arr - r_arr)),
        "pct_pairs_resid_better": float(((r_arr < b_arr).mean())),
    }

    # ---- design-level aggregates (exchangeable units) ----
    by_design: dict[str, dict] = defaultdict(lambda: {"base": [], "resid": [], "n": 0})
    for pid in common:
        fd = pair_fold[pid]
        by_design[fd]["base"].append(base_wmae[pid])
        by_design[fd]["resid"].append(resid_wmae[pid])
        by_design[fd]["n"] += 1

    des_base = np.array([float(np.mean(v["base"])) for v in by_design.values()])
    des_resid = np.array([float(np.mean(v["resid"])) for v in by_design.values()])
    design_level = {
        "n_designs": len(by_design),
        "baseline_wmae_mean": float(des_base.mean()),
        "resid_wmae_mean": float(des_resid.mean()),
        "improvement_mean": float((des_base - des_resid).mean()),
        "pct_designs_resid_better": float(((des_resid < des_base).mean())),
    }

    # per-seed resid summary
    seed_summary = {}
    for s in sorted({s for d in resid_seeds.values() for s in d}):
        arr = np.array([resid_seeds[p].get(s) for p in common
                        if s in resid_seeds[p]])
        seed_summary[s] = {"mean": float(arr.mean()), "median": float(np.median(arr))}

    # ---- training diagnostics (prior vs model train MAE, delta growth) ----
    diag = {"n_folds_logged": 0, "prior_mae_mean": None, "model_mae_mean": None,
            "delta_abs_mean_mean": None}
    if args.logdir:
        ldir = Path(args.logdir)
        files = sorted(ldir.glob("*.json"))
        prior_m, model_m, delta_m = [], [], []
        for f in files:
            try:
                lg = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            seeds = lg.get("seeds") or {}
            if not seeds:
                continue
            # seed 0 final values
            s0 = seeds.get("0") or seeds.get("0", {})
            fin = s0.get("final") or {}
            diag["n_folds_logged"] += 1
            if "mae_prior_train" in fin:
                prior_m.append(fin["mae_prior_train"])
            if "mae_model_train" in fin:
                model_m.append(fin["mae_model_train"])
            if "delta_abs_mean" in fin:
                delta_m.append(fin["delta_abs_mean"])
        if prior_m:
            diag["prior_mae_mean"] = float(np.mean(prior_m))
        if model_m:
            diag["model_mae_mean"] = float(np.mean(model_m))
        if delta_m:
            diag["delta_abs_mean_mean"] = float(np.mean(delta_m))

    # ---- horizontal per-design table (top 30 by n) ----
    table = []
    for fd, v in sorted(by_design.items(), key=lambda kv: -kv[1]["n"])[:30]:
        b = float(np.mean(v["base"]))
        r = float(np.mean(v["resid"]))
        table.append({"design": fd, "n_changers": v["n"],
                      "baseline_wmae": round(b, 4), "resid_wmae": round(r, 4),
                      "improvement": round(b - r, 4),
                      "better": "resid" if r < b else "base"})

    out = {
        "pooled": pooled,
        "design_level": design_level,
        "resid_by_seed": seed_summary,
        "training_diagnostics": diag,
        "n_no_call_rows": n_no_call,
        "top_design_table": table,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True),
                              encoding="utf-8")

    # ---- printed tables ----
    p = pooled
    print("=== POOLED (all changers, per-row wmae) ===")
    print(f"n_pairs={p['n_pairs']}")
    print(f"baseline (wmed prior)  mean={p['baseline_wmae_mean']:.4f} median={p['baseline_wmae_median']:.4f}")
    print(f"residual (seeds0-4)    mean={p['resid_wmae_mean']:.4f} median={p['resid_wmae_median']:.4f}")
    print(f"improvement mean={p['improvement_mean']:.4f} median={p['improvement_median']:.4f} "
          f"pct_resid_better={p['pct_pairs_resid_better']:.3f}")
    print("\n=== DESIGN-LEVEL ===")
    d = design_level
    print(f"n_designs={d['n_designs']} baseline={d['baseline_wmae_mean']:.4f} "
          f"resid={d['resid_wmae_mean']:.4f} improvement={d['improvement_mean']:.4f} "
          f"pct_designs_resid_better={d['pct_designs_resid_better']:.3f}")
    print("\n=== RESID BY SEED ===")
    for s, v in seed_summary.items():
        print(f"seed {s}: mean={v['mean']:.4f} median={v['median']:.4f}")
    print("\n=== TRAINING DIAG (seed0) ===")
    print(json.dumps(diag, indent=2))
    print("\n=== TOP DESIGNS ===")
    print(f"{'design':<36}{'n':>5}{'base':>9}{'resid':>9}{'impr':>9}  winner")
    for t in table:
        print(f"{t['design']:<36}{t['n_changers']:>5}{t['baseline_wmae']:>9}"
              f"{t['resid_wmae']:>9}{t['improvement']:>9}  {t['better']}")
    print(f"\nWROTE -> {args.out}")


if __name__ == "__main__":
    main()
