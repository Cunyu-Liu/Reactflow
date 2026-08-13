#!/usr/bin/env python3
"""run_p5_mechanism_v1: LOCKED P5 mechanism contrasts on frozen external components.

Contract 12.7; plan: docs/prospective_v2/p5_frozen_mechanism_plan_20260813.md.

Reuses the SAME frozen inputs as P4 (component graph, rdat profiles, B*_external
= reg_direct refit once on ALL development OK7a_M2). P4 outcome is read only
and carried forward (not recomputed).

Contrasts (frozen):
  A. signed distance curve: component-macro D_vs_zero within |dist| bands
     [0, 1-3, 4-10, 11-25, >=26]; Holm-Bonferroni across the 5 bands.
  A'. distance heterogeneity: D_vs_zero(edit) - D_vs_zero(very-far), CI lower > 0.
  B. negative control: permute each mutant's direct feature vectors across its
     shared positions (seed 20260813); permuted direct must show no skill.
  C. region/biology stratum: D_vs_zero per dataset (direction-level replication).
  D. failure cases: components with D_vs_zero < 0.
  E. claim-evidence map + verdict MECHANISM_EVIDENCE_PASS / NOT_ESTABLISHED.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as _st

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian
from scripts.reactflow_delta.run_p4_external_v1 import (
    DIRECT_EXTERNAL, K_PREACCESS_EXPECTED, MIN_SCORED_MUTANTS,
    MIN_SHARED_NONMISSING, _feat, _fit_bstar_external, _load_frozen_graph,
    _load_profiles, _ref_alt,
)
from reactflow.delta.rdat import parse_rdat

FIXED_SCALE = 0.3
ALPHA = 0.025
NEG_SEED = 20260813
# (low, high) inclusive |dist| bands
DIST_BANDS = [(0, 0), (1, 3), (4, 10), (11, 25), (26, None)]
BAND_LABELS = ["edit_site", "near_1_3", "mid_4_10", "far_11_25", "very_far_26p"]


def _band_of(d: int) -> int:
    a = abs(d)
    for i, (lo, hi) in enumerate(DIST_BANDS):
        if a >= lo and (hi is None or a <= hi):
            return i
    return len(DIST_BANDS) - 1


def _ci_one_sided(x: list[float]) -> dict:
    n = len(x)
    if n < 2:
        return {"n": n, "mean": None, "ci_low": None, "ci_high": None}
    arr = np.asarray(x, float)
    m = float(arr.mean()); s = float(arr.std(ddof=1))
    t = _st.t.ppf(1 - ALPHA, n - 1)
    return {"n": n, "mean": m, "sd": s,
            "ci_low": m - t * s / np.sqrt(n), "ci_high": m + t * s / np.sqrt(n)}


def _pval_one_sided(x: list[float]) -> float:
    n = len(x)
    if n < 2 or float(np.std(x, ddof=1)) <= 0:
        return 1.0
    t = float(np.mean(x)) / (float(np.std(x, ddof=1)) / np.sqrt(n))
    return float(_st.t.sf(t, n - 1))


def _collect_d(coef: np.ndarray, comps: list[dict],
               profiles: dict[str, dict],
               permute: bool = False) -> list[dict]:
    """Per-component rows: {component, dataset, band_D[band] -> D_vs_zero mean}.

    permute=True shuffles each mutant's feature vectors across its shared
    positions (negative control, seed NEG_SEED) so direct predictions are
    feature-mismatched.
    """
    rng = np.random.default_rng(NEG_SEED)
    rows = []
    for comp in comps:
        wt = profiles.get(comp["wt_name"])
        if wt is None:
            continue
        wt_react = np.asarray(wt["reactivity"], float)
        L = len(wt_react)
        if L == 0:
            continue
        n_scored = 0
        band_d = {b: [] for b in range(len(DIST_BANDS))}
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
            # gather (position, feature) for shared non-missing positions
            positions = []
            features = []
            for i in m["shared_region"]:
                if i >= L or np.isnan(mut_react[i]) or np.isnan(wt_react[i]):
                    continue
                positions.append(i)
                features.append(_feat(we, wt_react[i], i - edit_pos, ref, alt))
            if len(features) < MIN_SHARED_NONMISSING:
                continue
            n_scored += 1
            if permute:
                # detach features from their positions so each readout position
                # receives a feature computed for a DIFFERENT position
                # (negative control; reordering tuples would be a no-op).
                rng.shuffle(features)
            for i, f in zip(positions, features):
                pd_ = float(np.dot(coef, np.concatenate([[1.0], f])))
                d = crps_gaussian(wt_react[i], FIXED_SCALE, mut_react[i]) \
                    - crps_gaussian(pd_, FIXED_SCALE, mut_react[i])
                band_d[_band_of(i - edit_pos)].append(d)
        if n_scored < MIN_SCORED_MUTANTS:
            continue
        rows.append({
            "wt_name": comp["wt_name"],
            "dataset": comp["dataset"] if "dataset" in comp else "unknown",
            "n_scored": n_scored,
            "band_D": {BAND_LABELS[b]: float(np.mean(band_d[b])) if band_d[b] else None
                       for b in range(len(DIST_BANDS))},
            "n_band_positions": {BAND_LABELS[b]: len(band_d[b])
                                 for b in range(len(DIST_BANDS))},
        })
    return rows


def _macro(xs: list[float]) -> dict:
    return _ci_one_sided(xs)


def run_p5(rdat_dir: Path, dev_csv: Path, components_path: Path,
           p4_result_path: Path, out: Path) -> dict:
    comps, total_snv = _load_frozen_graph(components_path)
    # annotate each component with its dataset
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

    real_rows = _collect_d(coef, comps, profiles, permute=False)
    perm_rows = _collect_d(coef, comps, profiles, permute=True)

    # --- A. signed distance curve ---
    band_stats = {}
    band_p = {}
    for b, label in enumerate(BAND_LABELS):
        vals = [r["band_D"][label] for r in real_rows
                if r["band_D"][label] is not None]
        band_stats[label] = _macro(vals)
        band_p[label] = _pval_one_sided(vals)
    # Holm-Bonferroni over the 5 band tests (family A)
    p_order = sorted((band_p[l], l) for l in BAND_LABELS)
    holm = {}
    for i, (p, l) in enumerate(p_order):
        holm[l] = {"raw_p": p, "holm_threshold": ALPHA / (len(BAND_LABELS) - i),
                   "pass": bool(p <= ALPHA / (len(BAND_LABELS) - i))}

    # --- A'. distance heterogeneity: edit - very-far ---
    hets = [r["band_D"]["edit_site"] - r["band_D"]["very_far_26p"]
            for r in real_rows
            if r["band_D"]["edit_site"] is not None
            and r["band_D"]["very_far_26p"] is not None]
    het_ci = _macro(hets)
    het_pass = bool(het_ci.get("ci_low") is not None and het_ci["ci_low"] > 0.0)

    # --- B. negative control ---
    perm_vals = [r["band_D"]["edit_site"] for r in perm_rows
                 if r["band_D"]["edit_site"] is not None]
    perm_ci = _macro(perm_vals)
    neg_pass = bool(perm_ci.get("ci_high") is not None and perm_ci["ci_high"] <= 0.0)

    # --- C. region / dataset stratum ---
    ds_stats = {}
    for cid in DIRECT_EXTERNAL:
        rows_ds = [r for r in real_rows if r["dataset"] == cid]
        vals = [r["band_D"]["edit_site"] for r in rows_ds
                if r["band_D"]["edit_site"] is not None]
        ds_stats[cid] = {"n_components": len(rows_ds), "mean_D_edit": float(np.mean(vals)) if vals else None}
    ds_pos = sum(1 for v in ds_stats.values() if (v["mean_D_edit"] or 0.0) > 0.0)
    region_pass = bool(ds_pos >= 2)

    # --- D. failure cases ---
    failures = [{"wt_name": r["wt_name"], "dataset": r["dataset"],
                 "D_vs_zero_edit": r["band_D"]["edit_site"]}
                for r in real_rows if (r["band_D"]["edit_site"] or 0.0) < 0.0]

    # --- verdict ---
    p4 = json.loads(p4_result_path.read_text(encoding="utf-8"))
    p4_pass = p4.get("verdict") == "P4_EXTERNAL_STATISTICAL_PASS"
    edit_holm = holm["edit_site"]["pass"]
    mechanism_pass = bool(p4_pass and edit_holm and het_pass and neg_pass and region_pass)

    claim_map = [
        {"claim": "P4 external statistical replication",
         "evidence": p4.get("verdict"), "pass": p4_pass},
        {"claim": "direct skill concentrated at the signed edit site (distance curve)",
         "evidence": f"edit_site Holm pass={edit_holm}; heterogeneity CI lower={het_ci.get('ci_low')}",
         "pass": bool(edit_holm and het_pass)},
        {"claim": "effect is feature-dependent (negative control: permuted features show no skill)",
         "evidence": f"permuted D CI upper={perm_ci.get('ci_high')}", "pass": neg_pass},
        {"claim": "effect replicates across biology (region stratum >= 2 datasets positive)",
         "evidence": json.dumps(ds_stats), "pass": region_pass},
        {"claim": "failure cases do not overturn primary contrast (P4 leave-dominant-out)",
         "evidence": f"{len(failures)}/24 components negative; P4 LOO CI lower={p4.get('leave_dominant_out_ci', {}).get('ci_low')}",
         "pass": bool((p4.get('leave_dominant_out_ci') or {}).get('ci_low', 0.0) > 0.0)},
    ]

    report = {
        "schema_version": "reactflow_delta.p5_mechanism.v1",
        "frozen_plan": "docs/prospective_v2/p5_frozen_mechanism_plan_20260813.md",
        "candidate": "reg_direct (B*_external, same coef as P4)",
        "component_graph": str(components_path),
        "K_preaccess": K_PREACCESS_EXPECTED,
        "K_eff_realized": len(real_rows),
        "band_stats": band_stats,
        "band_holm": holm,
        "distance_heterogeneity": {"D_edit_minus_vfar": het_ci, "pass": het_pass},
        "negative_control": {"permuted_edit_D": perm_ci, "seed": NEG_SEED, "pass": neg_pass},
        "region_strata": ds_stats,
        "region_replication_pass": region_pass,
        "failure_cases": failures,
        "p4_carried": {"verdict": p4.get("verdict"),
                       "leave_dominant_out_ci": p4.get("leave_dominant_out_ci")},
        "claim_evidence_map": claim_map,
        "verdict": "MECHANISM_EVIDENCE_PASS" if mechanism_pass else "MECHANISM_NOT_ESTABLISHED",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "failure_cases"},
                     indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdat-dir", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--components", required=True)
    ap.add_argument("--p4-result", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run_p5(Path(args.rdat_dir), Path(args.dev_csv), Path(args.components),
           Path(args.p4_result), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
