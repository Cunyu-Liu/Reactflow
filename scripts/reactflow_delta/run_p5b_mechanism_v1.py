#!/usr/bin/env python3
"""run_p5b_mechanism_v1: LOCKED P5b mechanism contrasts on NEW independent set.

Contract 12.7 / 16.1; plan: docs/prospective_v2/p5b_frozen_mechanism_plan_20260814.md.

Purpose: confirm (or refute) on NEW independent components the mechanism claim
that the frozen direct model (reg_direct, B*_external, same coef as P4/P5)
provides full-construct skill extending to REMOTE positions (very-far band,
|dist| >= 26) -- the constructive replacement for the deleted "edit-site
concentration" claim (which failed on the 24-component locked set).

NEW independent set (never outcome-accessed):
  M2RFOK_2A3_0000 (rfam-OK), M2RFPK_2A3_0000/_0001/_0002 (rfam-PK)
  DasLab BigLib2 OneMil2 M2 sub-libraries (2A3-MaP, RNAFramework v2.8.4).
  Frozen graph: p5b_external_components.json (694 components / 106,904 SNV),
  built outcome-blind from sequence identity only, zero dev overlap, disjoint
  from the consumed 24 components.

Frozen PASS criteria (p5b plan section 5):
  P4 carried (P4_EXTERNAL_STATISTICAL_PASS).
  Primary: D_vs_zero(B_vfar) one-sided 95% CI lower > 0 AND Holm family A pass.
  Full-band: D_vs_zero(B_edit) Holm pass too.
  Negative control: permuted direct (seed 20260813) CI upper <= 0.
  Region replication: >= 2 of 4 dataset groups positive at edit band.
  Leave-dominant-out: very-far CI lower remains > 0 after removing dominant.

This is a NEW locked external outcome access (count 1 -> 2). One execution only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as _st

from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian
from scripts.reactflow_delta.run_p4_external_v1 import (
    ALPHA, LAMBDA_RIDGE, FIXED_SCALE, MIN_SHARED_NONMISSING,
    MIN_SCORED_MUTANTS, _feat, _fit_bstar_external,
)
from reactflow.delta.rdat import parse_rdat

NEW_DATASETS = ["M2RFOK_2A3_0000", "M2RFPK_2A3_0000",
                "M2RFPK_2A3_0001", "M2RFPK_2A3_0002"]
NEG_SEED = 20260813
ALPHA_CI = 0.025
# (low, high) inclusive |dist| bands -- identical to P5
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
    t = _st.t.ppf(1 - ALPHA_CI, n - 1)
    return {"n": n, "mean": m, "sd": s,
            "ci_low": m - t * s / np.sqrt(n), "ci_high": m + t * s / np.sqrt(n)}


def _pval_one_sided(x: list[float]) -> float:
    n = len(x)
    if n < 2 or float(np.std(x, ddof=1)) <= 0:
        return 1.0
    t = float(np.mean(x)) / (float(np.std(x, ddof=1)) / np.sqrt(n))
    return float(_st.t.sf(t, n - 1))


def _crps_gauss_vec(loc: np.ndarray, scale: float, y: np.ndarray) -> np.ndarray:
    """Vectorized Gaussian CRPS energy form."""
    z = (y - loc) / scale
    e_abs = scale * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z) \
        + (y - loc) * (2.0 * _st.norm.cdf(z) - 1.0)
    return e_abs - scale / np.sqrt(np.pi)


def _load_profiles(rdat_dir: Path) -> dict[str, dict]:
    by_name = {}
    for cid in NEW_DATASETS:
        r = parse_rdat(rdat_dir / f"{cid}.rdat")
        for x in r["profiles"]:
            by_name[x["profile_name"]] = x
    return by_name


def _ref_alt(name: str) -> tuple[str, str]:
    for tok in reversed(name.split("_")):
        if len(tok) == 3 and tok[0].isdigit() and tok[1] in ALPHA and tok[2] == "-":
            return tok[1], tok[2]
    import re as _re
    m = _re.search(r"(\d+)([ACGU])-([ACGU])", name)
    if m:
        return m.group(2), m.group(3)
    return "A", "U"


def _collect_d(coef: np.ndarray, comps: list[dict], profiles: dict[str, dict],
               permute: bool = False) -> list[dict]:
    """Per-component rows: {wt_name, dataset, n_scored, band_D -> mean D_vs_zero,
    n_band_positions}. permute=True shuffles each mutant's feature array across
    its shared positions (negative control, seed NEG_SEED)."""
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
            pos = np.asarray([i for i in m["shared_region"]
                              if i < L and not np.isnan(mut_react[i])
                              and not np.isnan(wt_react[i])], int)
            if len(pos) < MIN_SHARED_NONMISSING:
                continue
            n_scored += 1
            if permute:
                # detach features from positions (negative control; shuffling
                # the tuple list would be a no-op -- F3 fix).
                feats = np.stack([_feat(we, wt_react[i], i - edit_pos, ref, alt)
                                  for i in pos]).astype(np.float32)
                rng.shuffle(feats)  # shuffle rows (position-feature pairs)
            else:
                feats = np.stack([_feat(we, wt_react[i], i - edit_pos, ref, alt)
                                  for i in pos]).astype(np.float32)
            direct = coef[0] + feats @ coef[1:]
            zero = wt_react[pos]
            target = mut_react[pos]
            d = _crps_gauss_vec(zero, FIXED_SCALE, target) \
                - _crps_gauss_vec(direct, FIXED_SCALE, target)
            bands = np.asarray([_band_of(int(i) - edit_pos) for i in pos], int)
            for b in range(len(DIST_BANDS)):
                sel = d[bands == b]
                if sel.size:
                    band_d[b].extend(sel.tolist())
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


def run_p5b(rdat_dir: Path, dev_csv: Path, components_path: Path,
            p4_result_path: Path, out: Path) -> dict:
    doc = json.loads(components_path.read_text(encoding="utf-8"))
    comps = doc["components"]
    for c in comps:
        c.setdefault("dataset", "unknown")

    coef = _fit_bstar_external(dev_csv)
    profiles = _load_profiles(rdat_dir)

    real_rows = _collect_d(coef, comps, profiles, permute=False)
    perm_rows = _collect_d(coef, comps, profiles, permute=True)

    # --- family A: signed distance curve ---
    band_stats, band_p = {}, {}
    for b, label in enumerate(BAND_LABELS):
        vals = [r["band_D"][label] for r in real_rows
                if r["band_D"][label] is not None]
        band_stats[label] = _ci_one_sided(vals)
        band_p[label] = _pval_one_sided(vals)
    p_order = sorted((band_p[l], l) for l in BAND_LABELS)
    holm = {}
    for i, (p, l) in enumerate(p_order):
        holm[l] = {"raw_p": p, "holm_threshold": ALPHA_CI / (len(BAND_LABELS) - i),
                   "pass": bool(p <= ALPHA_CI / (len(BAND_LABELS) - i))}

    # --- primary: very-far spatial extension ---
    vfar_vals = [r["band_D"]["very_far_26p"] for r in real_rows
                 if r["band_D"]["very_far_26p"] is not None]
    vfar_ci = _ci_one_sided(vfar_vals)
    primary_pass = bool(vfar_ci.get("ci_low") is not None
                        and vfar_ci["ci_low"] > 0.0 and holm["very_far_26p"]["pass"])

    # --- full-band: edit site also Holm-pass ---
    edit_pass = bool(holm["edit_site"]["pass"])

    # --- negative control ---
    perm_vals = [r["band_D"]["edit_site"] for r in perm_rows
                 if r["band_D"]["edit_site"] is not None]
    perm_ci = _ci_one_sided(perm_vals)
    neg_pass = bool(perm_ci.get("ci_high") is not None and perm_ci["ci_high"] <= 0.0)

    # --- region / dataset stratum ---
    ds_stats = {}
    for cid in NEW_DATASETS:
        vals = [r["band_D"]["edit_site"] for r in real_rows
                if r["dataset"] == cid and r["band_D"]["edit_site"] is not None]
        ds_stats[cid] = {"n_components": sum(1 for r in real_rows if r["dataset"] == cid),
                         "mean_D_edit": float(np.mean(vals)) if vals else None}
    region_pass = bool(sum(1 for v in ds_stats.values()
                           if (v["mean_D_edit"] or 0.0) > 0.0) >= 2)

    # --- failure cases + leave-dominant-out (very-far band) ---
    failures = [{"wt_name": r["wt_name"], "dataset": r["dataset"],
                 "D_vs_zero_edit": r["band_D"]["edit_site"]}
                for r in real_rows if (r["band_D"]["edit_site"] or 0.0) < 0.0]
    arr_vfar = np.asarray(vfar_vals, float)
    dom = int(np.argmax(np.abs(arr_vfar))) if len(arr_vfar) else -1
    leave = arr_vfar[np.arange(len(arr_vfar)) != dom].tolist()
    leave_ci = _ci_one_sided(leave)
    leave_pass = bool(leave_ci.get("ci_low") is not None and leave_ci["ci_low"] > 0.0)

    p4 = json.loads(p4_result_path.read_text(encoding="utf-8"))
    p4_pass = p4.get("verdict") == "P4_EXTERNAL_STATISTICAL_PASS"

    mechanism_pass = bool(p4_pass and primary_pass and edit_pass
                          and neg_pass and region_pass and leave_pass)

    claim_map = [
        {"claim": "P4 external statistical replication (carried)",
         "evidence": p4.get("verdict"), "pass": p4_pass},
        {"claim": "full-construct skill extends to REMOTE positions (very-far band)",
         "evidence": f"vfar CI lower={vfar_ci.get('ci_low')}; Holm pass={holm['very_far_26p']['pass']}",
         "pass": primary_pass},
        {"claim": "skill also present at the edit site (construct-wide, not remote-only)",
         "evidence": f"edit CI lower={band_stats['edit_site'].get('ci_low')}; Holm pass={holm['edit_site']['pass']}",
         "pass": edit_pass},
        {"claim": "effect is feature-dependent (negative control: permuted no skill)",
         "evidence": f"permuted edit D CI upper={perm_ci.get('ci_high')}", "pass": neg_pass},
        {"claim": "effect replicates across datasets (>= 2/4 groups positive)",
         "evidence": json.dumps(ds_stats), "pass": region_pass},
        {"claim": "remote-skill contrast robust to dominant component (leave-dominant-out)",
         "evidence": f"vfar leave-dominant CI lower={leave_ci.get('ci_low')}", "pass": leave_pass},
    ]

    report = {
        "schema_version": "reactflow_delta.p5b_mechanism.v1",
        "frozen_plan": "docs/prospective_v2/p5b_frozen_mechanism_plan_20260814.md",
        "candidate": "reg_direct (B*_external, same coef as P4/P5)",
        "component_graph": str(components_path),
        "K_preaccess": doc.get("K_preaccess_components"),
        "K_preaccess_single_snv": doc.get("K_preaccess_single_snv"),
        "K_eff_realized": len(real_rows),
        "band_stats": band_stats,
        "band_holm": holm,
        "primary_very_far": vfar_ci,
        "primary_pass": primary_pass,
        "edit_site_pass": edit_pass,
        "negative_control": {"permuted_edit_D": perm_ci, "seed": NEG_SEED, "pass": neg_pass},
        "region_strata": ds_stats,
        "region_replication_pass": region_pass,
        "failure_cases": failures,
        "n_failure_cases": len(failures),
        "leave_dominant_out_vfar_ci": leave_ci,
        "leave_dominant_out_pass": leave_pass,
        "p4_carried": {"verdict": p4.get("verdict")},
        "claim_evidence_map": claim_map,
        "locked_outcome_access_count": 2,
        "verdict": ("MECHANISM_EVIDENCE_PASS" if mechanism_pass
                    else "MECHANISM_NOT_ESTABLISHED"),
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
    run_p5b(Path(args.rdat_dir), Path(args.dev_csv), Path(args.components),
            Path(args.p4_result), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
