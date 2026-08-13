#!/usr/bin/env python3
"""analyze_p2_secondaries_v1: region/distance/coverage secondaries (contract 9.3).

Reads the per-position held prediction rows produced by run_p2_direct_v2 and
computes mandatory secondaries:
  - region breakdown: design_region vs other_assay_region (MAE direct vs zero)
  - continuous sequence-distance breakdown (|edit-readout distance| bands)
  - coverage/calibration diagnostic: empirical residual SD (sharpness) vs the
    fixed nominal scale, and nominal coverage at that scale
GPU-free reporting pass over the saved rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    q = ~np.isnan(pred) & ~np.isnan(target)
    return float(np.mean(np.abs(pred[q] - target[q]))) if q.any() else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.rows_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    tgt = np.array([r["target"] for r in rows])
    pd_ = np.array([r["pred_direct"] if r["pred_direct"] is not None else np.nan for r in rows])
    pz = np.array([r["pred_zero"] if r["pred_zero"] is not None else np.nan for r in rows])
    region = np.array([r["region"] for r in rows])
    dist = np.array([abs(r["dist"]) for r in rows])

    # region breakdown
    region_mae = {}
    for reg in sorted(set(region)):
        m = region == reg
        region_mae[reg] = {"mae_direct": mae(pd_[m], tgt[m]), "mae_zero": mae(pz[m], tgt[m]),
                           "n": int(m.sum())}

    # distance breakdown (edit site 0, near 1-4, mid 5-9, far 10-19, far+ >=20)
    def band(d: int) -> str:
        if d == 0:
            return "0_edit_site"
        if d <= 4:
            return "1_4_near"
        if d <= 9:
            return "5_9_mid"
        if d <= 19:
            return "10_19_far"
        return "20plus_far"
    bands = np.array([band(int(d)) for d in dist])
    dist_mae = {}
    for b in ["0_edit_site", "1_4_near", "5_9_mid", "10_19_far", "20plus_far"]:
        m = bands == b
        if not m.any():
            continue
        dist_mae[b] = {"mae_direct": mae(pd_[m], tgt[m]), "mae_zero": mae(pz[m], tgt[m]),
                       "n": int(m.sum())}

    # calibration/sharpness diagnostic with nominal scale 0.3
    resid = pd_ - tgt
    q = ~np.isnan(resid)
    emp_sd = float(np.std(resid[q])) if q.any() else float("nan")
    nominal_scale = 0.3
    z = np.abs(resid[q]) / nominal_scale
    coverage_95 = float(np.mean(z <= 1.96)) if q.any() else float("nan")

    report = {
        "schema_version": "reactflow_delta.p2_secondaries.v1",
        "n_position_rows": len(rows),
        "region_mae": region_mae,
        "distance_mae": dist_mae,
        "calibration_diagnostic": {
            "nominal_gaussian_scale": nominal_scale,
            "empirical_residual_sd": emp_sd,
            "nominal_95pct_coverage_at_fixed_scale": coverage_95,
            "note": "P2 v1 used a fixed train-residual scale 0.3; empirical SD and "
                    "coverage are diagnostics. Proper model_scale per position is a "
                    "deployment item (contract 11.7).",
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
