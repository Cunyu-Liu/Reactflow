#!/usr/bin/env python3
"""run_m2_seq_compare.py — wait for the M2 residual-MLP+global-seq training to
finish all 159 folds, then produce the horizontal WMAE-skill comparison across M2
variants:
  * wmed_spectrum            (per-position median prior baseline)
  * wmae_resid_spectrum      (plain residual MLP, prior run, mu-ensemble +8.88%)
  * wmae_resid_seq_spectrum  (residual MLP + global seq, this run)
using m2_horizontal_ensemble_report.py on each variant's keyed predictions file, and
merge the per-seed + mu-ensemble rows into a single comparison JSON.

Usage (on the A100 server, editflow conda env):
    python run_m2_seq_compare.py \
        --progress /mnt/.../m2_response_spectrum_seq_20260813/fold_progress.json \
        --pred-mlp /mnt/.../m2_response_spectrum_20260812/keyed_predictions_m2_spectrum.jsonl \
        --pred-seq /mnt/.../m2_response_spectrum_seq_20260813/keyed_predictions_m2_seq.jsonl \
        --mlp-report /mnt/.../m2_response_spectrum_20260812/compare_m2/m2_horizontal_ensemble_report.json \
        --out /mnt/.../m2_response_spectrum_seq_20260813/compare_m2_seq
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _progress(path: str) -> set:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return set(data.get("completed_folds", []))
    except Exception:
        return set()


def _rows_summary(report):
    """Extract the mu-ensemble + per-seed rows into a compact dict."""
    m = report["mu_ensemble"]
    return {
        "mu_ensemble": {
            "skill": m["skill"], "ci_low": m["ci_low"], "ci_high": m["ci_high"],
            "permutation_p": m["permutation_p"], "n_designs": m["n_designs"],
            "n_positions": m["n_positions"],
            "per_design_mean": m["per_design"]["mean"],
            "per_design_median": m["per_design"]["median"],
            "per_design_pct_positive": m["per_design"]["pct_positive"],
        },
        "per_seed": {
            s: {"skill": rep["skill"]} for s, rep in report["n_seed_single_skill"].items()
        },
        "wmae_baseline": report["n_seed_single_skill"]["seed_0"]["wmae_baseline"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", required=True,
                    help="fold_progress.json written by the seq M2 runner")
    ap.add_argument("--total", type=int, default=159)
    ap.add_argument("--pred-mlp", required=True,
                    help="keyed predictions from the plain residual MLP M2 run")
    ap.add_argument("--pred-seq", required=True,
                    help="keyed predictions from the residual-MLP+global-seq M2 run")
    ap.add_argument("--mlp-report", required=True,
                    help="existing m2_horizontal_ensemble_report.json for the MLP run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll-secs", type=int, default=300)
    ap.add_argument("--max-wait-secs", type=int, default=14 * 3600)
    args = ap.parse_args()

    print(f"[m2_seq_compare] total folds={args.total}", flush=True)
    deadline = time.time() + args.max_wait_secs
    while time.time() < deadline:
        done = _progress(args.progress)
        print(f"[m2_seq_compare] completed {len(done)}/{args.total}", flush=True)
        if len(done) >= args.total:
            break
        time.sleep(args.poll_secs)
    else:
        print(f"[m2_seq_compare] TIMEOUT after {args.max_wait_secs}s; running eval on "
              f"whatever is available anyway", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seq_report_path = out / "m2_horizontal_ensemble_report_seq.json"
    cmd = [
        sys.executable, "m2_horizontal_ensemble_report.py",
        "--pred", args.pred_seq, "--out", str(out),
        "--model-variant", "wmae_resid_seq_spectrum",
    ]
    print(f"[m2_seq_compare] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    # the report writes to m2_horizontal_ensemble_report.json; rename to *_seq
    (out / "m2_horizontal_ensemble_report.json").replace(seq_report_path)

    mlp_report = json.loads(Path(args.mlp_report).read_text(encoding="utf-8"))
    seq_report = json.loads(seq_report_path.read_text(encoding="utf-8"))

    report = {
        "schema": "reactflow_delta.response_spectrum.m2_seq_horizontal_compare.v1",
        "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
        "baseline": "wmed_spectrum",
        "variants": {
            "wmae_resid_spectrum": _rows_summary(mlp_report),
            "wmae_resid_seq_spectrum": _rows_summary(seq_report),
        },
    }
    (out / "m2_seq_horizontal_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== M2 horizontal WMAE-skill comparison (mu-ensemble) ===")
    print(f"{'variant':28s} {'skill':>8s} {'ci_low':>8s} {'ci_high':>8s} "
          f"{'perm_p':>8s} {'pct_pos':>8s} {'n_pos':>8s}")
    for variant, v in report["variants"].items():
        m = v["mu_ensemble"]
        print(f"{variant:28s} "
              f"{m['skill'] if m['skill'] is not None else float('nan'):8.4f} "
              f"{m['ci_low'] if m['ci_low'] is not None else float('nan'):8.4f} "
              f"{m['ci_high'] if m['ci_high'] is not None else float('nan'):8.4f} "
              f"{m['permutation_p'] if m['permutation_p'] is not None else float('nan'):8.4f} "
              f"{m['per_design_pct_positive'] if m['per_design_pct_positive'] is not None else float('nan'):8.3f} "
              f"{m['n_positions']:8d}")
    print(f"\nDONE -> {out / 'm2_seq_horizontal_comparison.json'}")


if __name__ == "__main__":
    sys.exit(main())
