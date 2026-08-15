#!/usr/bin/env python3
"""run_m2_attn_finalize.py — once the M2 v4 attention 159-fold run completes, run
the FULL deviation-detection (heavy permutation) and regenerate the integrated
3-way method summary (plain MLP vs position-aware vs attention) with full-data
numbers, for both the pooled-WMAE skill and the deviation-detection capability.

The running run_m2_attn_compare.py already produces the full WMAE
m2_attn_full_comparison.json at 159 folds.  This driver fills the remaining gap:
  * full attn deviation-detection report (m2_deviation_report.py, 300 perms)
  * integrated 3-way method summary (WMAE skill + deviation-detection)

Usage (A100 server, editflow env):
    python run_m2_attn_finalize.py \
        --progress /mnt/.../m2_response_spectrum_attn_gpu3_20260815/fold_progress.json \
        --pred-attn /mnt/.../m2_response_spectrum_attn_gpu3_20260815/keyed_predictions_m2_attn.jsonl \
        --out /mnt/.../m2_response_spectrum_attn_gpu3_20260815/compare_m2_attn \
        --mlp-horizontal /mnt/.../m2_response_spectrum_20260812/compare_m2/m2_horizontal_ensemble_report.json \
        --pa-horizontal /mnt/.../m2_response_spectrum_posaware_20260813/compare_m2_pa/m2_horizontal_ensemble_report_pa_full.json \
        --mlp-deviation /mnt/.../m2_response_spectrum_20260812/deviation_m2/m2_deviation_report.json \
        --pa-deviation /mnt/.../m2_response_spectrum_posaware_20260813/compare_m2_pa/deviation_pa_full/m2_deviation_report.json
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

ATTN_VARIANT = "wmae_resid_attn_spectrum"


def _progress(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return len(d.get("completed_folds", []))
    except Exception:
        return 0


def _hrow(r):
    m = r["mu_ensemble"]
    return {
        "skill": round(m["skill"], 4), "ci_low": round(m["ci_low"], 4),
        "ci_high": round(m["ci_high"], 4), "permutation_p": m["permutation_p"],
        "n_designs": m["n_designs"], "n_positions": m["n_positions"],
        "pct_positive": round(m["per_design"]["pct_positive"], 4),
        "wmae_model": round(m["wmae_model"], 4), "wmae_baseline": round(m["wmae_baseline"], 4),
    }


def _drow(r):
    m = r["mu_ensemble"]
    return {
        "spearman_rho": round(m["spearman_abs"], 4), "auroc": round(m["auroc_abs"], 4),
        "permutation_p": m["permutation_p"], "n_designs": m["n_designs"],
        "n_positions": m["n_positions"],
        "per_design_rho_mean": round(m["per_design"]["mean"], 4),
        "pct_positive_designs": round(m["per_design"]["pct_positive"], 4),
        "loo_rho_min": round(m["robustness"]["pooled_rho_min_over_loo"], 4),
        "pos10_rho": round([p["spearman_abs"] for p in r["per_position"]
                            if p["position"] == 10][0], 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", required=True)
    ap.add_argument("--pred-attn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlp-horizontal", required=True)
    ap.add_argument("--pa-horizontal", required=True)
    ap.add_argument("--mlp-deviation", required=True)
    ap.add_argument("--pa-deviation", required=True)
    ap.add_argument("--total", type=int, default=159)
    ap.add_argument("--poll-secs", type=int, default=600)
    ap.add_argument("--max-wait-secs", type=int, default=6 * 3600)
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_wait_secs
    script = Path(__file__).resolve().parent

    print(f"[attn_finalize] waiting for {args.total}-fold attn run...", flush=True)
    while time.time() < deadline:
        n = _progress(args.progress)
        print(f"[attn_finalize] completed {n}/{args.total}", flush=True)
        if n >= args.total:
            break
        time.sleep(args.poll_secs)
    if _progress(args.progress) < args.total:
        print("[attn_finalize] TIMEOUT waiting for full run", flush=True)
        return 1

    # 1. full deviation-detection (heavy permutation)
    dev_out = out / "deviation_attn_full"
    dev_out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script / "m2_deviation_report.py"),
           "--pred", args.pred_attn, "--out", str(dev_out),
           "--model-variant", ATTN_VARIANT, "--n-perm", str(args.n_perm)]
    print(f"[attn_finalize] running full deviation... {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    attn_dev = json.loads((dev_out / "m2_deviation_report.json").read_text(encoding="utf-8"))

    # 2. full WMAE horizontal (in case the compare driver's output is not yet there)
    attn_h_rep = out / "m2_horizontal_ensemble_report_attn_full.json"
    cmd = [sys.executable, str(script / "m2_horizontal_ensemble_report.py"),
           "--pred", args.pred_attn, "--out", str(out),
           "--model-variant", ATTN_VARIANT]
    print(f"[attn_finalize] running full WMAE horizontal...", flush=True)
    subprocess.run(cmd, check=True)
    (out / "m2_horizontal_ensemble_report.json").replace(attn_h_rep)
    attn_h = json.loads(attn_h_rep.read_text(encoding="utf-8"))

    # 3. integrated 3-way summary
    mlp_h = json.loads(Path(args.mlp_horizontal).read_text(encoding="utf-8"))
    pa_h = json.loads(Path(args.pa_horizontal).read_text(encoding="utf-8"))
    mlp_d = json.loads(Path(args.mlp_deviation).read_text(encoding="utf-8"))
    pa_d = json.loads(Path(args.pa_deviation).read_text(encoding="utf-8"))
    summary = {
        "schema": "reactflow_delta.response_spectrum.m2_attn_method_summary.v1",
        "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
        "baseline": "wmed_spectrum",
        "full_159_fold": True,
        "wmae_skill": {
            "plain_residual_mlp": _hrow(mlp_h),
            "position_aware": _hrow(pa_h),
            "position_aware_attention": _hrow(attn_h),
        },
        "deviation_detection": {
            "plain_residual_mlp": _drow(mlp_d),
            "position_aware": _drow(pa_d),
            "position_aware_attention": _drow(attn_dev),
        },
    }
    (out / "m2_attn_method_summary_full.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== M2 attention FULL (159-fold) 3-way method summary ===")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\nDONE -> {out / 'm2_attn_method_summary_full.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
