#!/usr/bin/env python3
"""run_m2_posaware_compare.py — poll the M2 position-aware training and, once a
minimum number of folds are complete, produce a MATCHED early comparison of
position-aware vs plain residual MLP on the SAME completed folds, plus a final
full comparison once all 159 folds finish.

Purpose: get a skill signal from the position-aware model well before the full run
(~19h) so we can tell whether per-position decoding actually beats the shared-head
residual MLP (+8.9% mu-ensemble), without waiting hours to catch a regression.

Usage (A100 server, editflow env):
    python run_m2_posaware_compare.py \
        --progress /mnt/.../m2_response_spectrum_posaware_20260813/fold_progress.json \
        --pred-mlp /mnt/.../m2_response_spectrum_20260812/keyed_predictions_m2_spectrum.jsonl \
        --pred-pa /mnt/.../m2_response_spectrum_posaware_20260813/keyed_predictions_m2_posaware.jsonl \
        --mlp-report /mnt/.../m2_response_spectrum_20260812/compare_m2/m2_horizontal_ensemble_report.json \
        --out /mnt/.../m2_response_spectrum_posaware_20260813/compare_m2_pa
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

PA_VARIANT = "wmae_resid_posaware_spectrum"


def _progress(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return set(d.get("completed_folds", []))
    except Exception:
        return set()


def _rows_summary(report):
    m = report["mu_ensemble"]
    return {
        "skill": m["skill"], "ci_low": m["ci_low"], "ci_high": m["ci_high"],
        "permutation_p": m["permutation_p"], "n_designs": m["n_designs"],
        "n_positions": m["n_positions"],
        "per_design_mean": m["per_design"]["mean"],
        "per_design_median": m["per_design"]["median"],
        "per_design_pct_positive": m["per_design"]["pct_positive"],
        "per_seed_skill": {s: rep["skill"]
                           for s, rep in report["n_seed_single_skill"].items()},
    }


def _filter_mlp_to_designs(pred_mlp, done, out_filtered):
    """Restrict plain-MLP preds to the given completed designs for a matched compare."""
    keep = 0
    lines = []
    for line in Path(pred_mlp).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["pair_id"].split(":")[0] in done:
            keep += 1
            lines.append(line)
    Path(out_filtered).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", required=True)
    ap.add_argument("--total", type=int, default=159)
    ap.add_argument("--pred-mlp", required=True)
    ap.add_argument("--pred-pa", required=True)
    ap.add_argument("--mlp-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--early-at", type=int, default=40,
                    help="run a matched early comparison once this many folds complete")
    ap.add_argument("--poll-secs", type=int, default=600)
    ap.add_argument("--max-wait-secs", type=int, default=20 * 3600)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    early_done = False
    deadline = time.time() + args.max_wait_secs

    def run_horizontal(pred, variant, tag):
        rep_path = out / f"m2_horizontal_ensemble_report_{tag}.json"
        script = Path(__file__).resolve().parent / "m2_horizontal_ensemble_report.py"
        cmd = [sys.executable, str(script),
               "--pred", pred, "--out", str(out), "--model-variant", variant]
        subprocess.run(cmd, check=True)
        (out / "m2_horizontal_ensemble_report.json").replace(rep_path)
        return json.loads(rep_path.read_text(encoding="utf-8"))

    while time.time() < deadline:
        done = _progress(args.progress)
        n = len(done)
        print(f"[pa_compare] completed {n}/{args.total}", flush=True)
        if n >= args.early_at and not early_done:
            early_done = True
            try:
                print(f"[pa_compare] EARLY matched compare at {n} folds", flush=True)
                mlp_f = out / "mlp_matched_early.jsonl"
                _filter_mlp_to_designs(args.pred_mlp, done, mlp_f)
                pa_report = run_horizontal(args.pred_pa, PA_VARIANT, "pa_early")
                mlp_report = run_horizontal(str(mlp_f), "wmae_resid_spectrum", "mlp_early")
                (out / "m2_pa_early_matched_comparison.json").write_text(json.dumps({
                    "schema": "reactflow_delta.response_spectrum.m2_pa_early_compare.v1",
                    "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
                    "baseline": "wmed_spectrum", "n_designs": n,
                    "variants": {
                        "wmae_resid_spectrum": _rows_summary(mlp_report),
                        "wmae_resid_posaware_spectrum": _rows_summary(pa_report),
                    },
                }, indent=2, sort_keys=True), encoding="utf-8")
                print(f"[pa_compare] EARLY comparison -> "
                      f"{out/'m2_pa_early_matched_comparison.json'}", flush=True)
            except Exception as e:
                print(f"[pa_compare] early compare failed: {e}", flush=True)
        if n >= args.total:
            break
        time.sleep(args.poll_secs)

    # ---- final full comparison ----
    pa_report = run_horizontal(args.pred_pa, PA_VARIANT, "pa_full")
    mlp_report = json.loads(Path(args.mlp_report).read_text(encoding="utf-8"))
    final = {
        "schema": "reactflow_delta.response_spectrum.m2_pa_full_compare.v1",
        "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
        "baseline": "wmed_spectrum",
        "variants": {
            "wmae_resid_spectrum": _rows_summary(mlp_report),
            "wmae_resid_posaware_spectrum": _rows_summary(pa_report),
        },
    }
    (out / "m2_pa_full_comparison.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== M2 position-aware vs plain MLP (mu-ensemble) ===")
    print(f"{'variant':30s} {'skill':>8s} {'ci_low':>8s} {'ci_high':>8s} {'pct_pos':>8s}")
    for variant, v in final["variants"].items():
        m = v
        print(f"{variant:30s} {m['skill'] if m['skill'] is not None else float('nan'):8.4f} "
              f"{m['ci_low'] if m['ci_low'] is not None else float('nan'):8.4f} "
              f"{m['ci_high'] if m['ci_high'] is not None else float('nan'):8.4f} "
              f"{m['per_design_pct_positive'] if m['per_design_pct_positive'] is not None else float('nan'):8.3f}")
    print(f"\nDONE -> {out / 'm2_pa_full_comparison.json'}")


if __name__ == "__main__":
    sys.exit(main())
