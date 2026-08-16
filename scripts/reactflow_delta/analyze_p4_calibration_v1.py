#!/usr/bin/env python3
"""analyze_p4_calibration_v1: P4 coverage/calibration acceptance (contract 12.6).

Contract 12.6 lists coverage/calibration 合格 as a P4 PASS condition. The frozen
protocol operationalized the statistical criteria (K_eff, CI, FWER, LOO) but not
a calibration check; this analyzer closes that mandatory gate using the SAME
frozen inputs (component graph, profiles, B*_external=reg_direct coef).

Frozen-style settings (declared before outcome computation):
  - predictive distribution: Gaussian N(mu_direct, scale=FIXED_SCALE=0.3) per
    shared-region position (identical to P4 evaluator).
  - nominal 68.3% interval: mu +/- 1.0*0.3 ; nominal 95%: mu +/- 1.96*0.3.
  - tolerance for "合格": empirical 95% coverage in [0.85, 0.99] and 68% coverage
    in [0.58, 0.78]. This mirrors the development diagnostic (P2: 95% coverage
    0.896 at fixed scale 0.3, empirical residual SD 0.413).
  - also report the honest empirical residual SD (pooled + per dataset) as a
    deployment/interpretation diagnostic (does NOT change the frozen scale).

Verdict: CALIBRATION_ACCEPTABLE if both coverage tolerances hold, else
CALIBRATION_MISMATCH. Reported as P4 secondary evidence, not a new claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as _st

from scripts.reactflow_delta.run_p4_external_v1 import (
    DIRECT_EXTERNAL, FIXED_SCALE, MIN_SCORED_MUTANTS, MIN_SHARED_NONMISSING,
    _feat, _fit_bstar_external, _load_frozen_graph, _load_profiles, _ref_alt,
)
from reactflow.delta.rdat import parse_rdat

NOMINAL_68 = 0.6827
NOMINAL_95 = 0.95
TOL_95 = (0.85, 0.99)
TOL_68 = (0.58, 0.78)
_Z_68 = 1.0
_Z_95 = 1.96


def _collect_preds(coef: np.ndarray, comps: list[dict],
                   profiles: dict[str, dict]) -> list[dict]:
    """Per-component {pred, target} arrays over shared non-missing positions."""
    rows = []
    for comp in comps:
        wt = profiles.get(comp["wt_name"])
        if wt is None:
            continue
        wt_react = np.asarray(wt["reactivity"], float)
        L = len(wt_react)
        if L == 0:
            continue
        preds, targets = [], []
        n_scored = 0
        for m in comp["mutants"]:
            mu = profiles.get(m["name"])
            if mu is None:
                continue
            mut_react = np.asarray(mu["reactivity"], float)
            if len(mut_react) != L:
                continue
            edit_pos = int(m["edit_pos"])
            ref, alt = _ref_alt(m["name"])
            we = wt_react[edit_pos] if edit_pos < L and not np.isnan(wt_react[edit_pos]) else 0.0
            cnt = 0
            for i in m["shared_region"]:
                if i >= L or np.isnan(mut_react[i]) or np.isnan(wt_react[i]):
                    continue
                f = _feat(we, wt_react[i], i - edit_pos, ref, alt)
                preds.append(float(np.dot(coef, np.concatenate([[1.0], f]))))
                targets.append(float(mut_react[i]))
                cnt += 1
            if cnt >= MIN_SHARED_NONMISSING:
                n_scored += 1
        if n_scored >= MIN_SCORED_MUTANTS:
            rows.append({"wt_name": comp["wt_name"],
                         "dataset": comp.get("dataset", "unknown"),
                         "pred": np.asarray(preds, float),
                         "target": np.asarray(targets, float)})
    return rows


def _coverage(pred: np.ndarray, target: np.ndarray, scale: float, z: float) -> float:
    resid = target - pred
    inside = float(np.mean(np.abs(resid) <= z * scale))
    return inside


def run_calibration(rdat_dir: Path, dev_csv: Path, components_path: Path,
                    out: Path) -> dict:
    comps, total_snv = _load_frozen_graph(components_path)
    for cid in DIRECT_EXTERNAL:
        r = parse_rdat(rdat_dir / f"{cid}.rdat")
        names = {x["profile_name"] for x in r["profiles"]}
        for c in comps:
            if c["wt_name"] in names:
                c["dataset"] = cid
    for c in comps:
        c.setdefault("dataset", "unknown")

    coef = _fit_bstar_external(dev_csv)
    profiles = _load_profiles(rdat_dir)
    rows = _collect_preds(coef, comps, profiles)

    pooled_pred = np.concatenate([r["pred"] for r in rows])
    pooled_target = np.concatenate([r["target"] for r in rows])
    resid = pooled_target - pooled_pred
    emp_sd = float(np.std(resid))

    cov_95 = _coverage(pooled_pred, pooled_target, FIXED_SCALE, _Z_95)
    cov_68 = _coverage(pooled_pred, pooled_target, FIXED_SCALE, _Z_68)
    n_pos = int(len(pooled_pred))

    pass_95 = TOL_95[0] <= cov_95 <= TOL_95[1]
    pass_68 = TOL_68[0] <= cov_68 <= TOL_68[1]
    calib_pass = bool(pass_95 and pass_68)

    ds_diag = {}
    for cid in DIRECT_EXTERNAL:
        ds_rows = [r for r in rows if r.get("dataset") == cid]
        if not ds_rows:
            ds_diag[cid] = None
            continue
        p = np.concatenate([r["pred"] for r in ds_rows])
        t = np.concatenate([r["target"] for r in ds_rows])
        ds_diag[cid] = {
            "n_positions": int(len(p)),
            "empirical_residual_sd": float(np.std(t - p)),
            "nominal_95pct_coverage": _coverage(p, t, FIXED_SCALE, _Z_95),
        }

    report = {
        "schema_version": "reactflow_delta.p4_calibration.v1",
        "frozen_evaluator": "Gaussian CRPS fixed scale 0.3 (identical to P4)",
        "candidate": "reg_direct (B*_external, same coef as P4)",
        "K_eff_realized": len(rows),
        "n_positions": n_pos,
        "pooled": {
            "empirical_residual_sd": emp_sd,
            "nominal_scale": FIXED_SCALE,
            "cov_68": cov_68, "nominal_68": NOMINAL_68, "tol_68": list(TOL_68), "pass_68": pass_68,
            "cov_95": cov_95, "nominal_95": NOMINAL_95, "tol_95": list(TOL_95), "pass_95": pass_95,
        },
        "per_dataset_diagnostic": ds_diag,
        "development_reference": "P2 calibration diagnostic: nominal 95% coverage 0.896, empirical residual SD 0.413 at fixed scale 0.3",
        "calibration_pass": calib_pass,
        "verdict": "CALIBRATION_ACCEPTABLE" if calib_pass else "CALIBRATION_MISMATCH",
        "note": "fixed scale 0.3 is frozen; honest empirical residual SD is a deployment/interpretation diagnostic and does not alter the frozen evaluator",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdat-dir", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--components", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run_calibration(Path(args.rdat_dir), Path(args.dev_csv), Path(args.components), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
