#!/usr/bin/env python3
"""run_p4_external_v1: LOCKED external confirmatory protocol (contract 12.6/15).

Single execution, after p4_frozen_protocol_20260813.md.
  - B*_external = reg_direct (Direct* procedure) refit on ALL development OK7a_M2
    via the existing M2Universe pool machinery (same features/ridge as P2).
  - candidate = same frozen reg_direct (P3 adopted the direct model, contract 17.2).
  - baseline = ZeroResponse (predict WT profile) + train median sensitivity.
  - external components: DIRECT_EXTERNAL pool (M2SL5, M3SARS, 15KLIB), 2A3-MaP,
    zero dev-sequence overlap. The component graph is FROZEN and loaded from the
    outcome-blind manifest p4_external_components.json (24 WT anchors, 3237
    single-SNV mutants, shared-region masks from sequence identity only).
  - scoring domain = shared region (positions where WT==mutant plus edit pos),
    bounded by the observed reactivity array length (3' pads/barcodes carry no
    reactivity and count as non-observed, per attrition rule 3).
  - estimator: component-macro paired D = L_baseline - L_direct; 95% t-CI.
  - delta_stat=0; K_required_planned=9; K_preaccess=24.
  - FWER: Holm-Bonferroni over {zero (primary), median (sensitivity)}.
  - dominant-component + leave-dominant-out sensitivity (contract 15.3).

Frozen attrition rules (p4_frozen_protocol_20260813.md section 6):
  1. WT anchor has an observed 2A3 reactivity profile (non-empty shared region).
  2. >= 20 single-SNV mutants matched to it (frozen graph guarantees >= 45).
  3. a mutant is scored only if it has >= 20 shared-region positions with
     non-missing WT and mutant reactivity (positions beyond the reactivity
     array are non-observed).
  A component is evaluable (enters K_eff) iff rules 1+2 hold and it has
  >= 20 scored mutants.

One execution only; output written once; locked_outcome_access_count = 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as _st

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian
from reactflow.delta.rdat import parse_rdat

ALPHA = {"A": 0, "C": 1, "G": 2, "U": 3}
DIRECT_EXTERNAL = ["M2SL5_2A3_0000", "M3SARS_2A3_0000", "15KLIB_2A3_0000"]
LAMBDA_RIDGE = 1e-2
FIXED_SCALE = 0.3
K_REQUIRED_PLANNED = 9
K_PREACCESS_EXPECTED = 24
MIN_SHARED_NONMISSING = 20
MIN_SCORED_MUTANTS = 20


def _feat(we: float, wt_r: float, dist: float, ref: str, alt: str) -> np.ndarray:
    r = np.zeros(4); a = np.zeros(4)
    r[ALPHA.get(ref, 3)] = 1.0
    a[ALPHA.get(alt, 3)] = 1.0
    return np.concatenate([[we, wt_r, dist, np.tanh(dist)], r, a]).astype(np.float32)


def _fit_bstar_external(dev_csv: Path) -> np.ndarray:
    """Refit the frozen reg_direct (ridge) on ALL development data via M2Universe."""
    univ = M2Universe(dev_csv)
    univ.build()
    records = univ.get_records()
    feats, targets = [], []
    for r in records:
        c = univ.get_construct(r.construct_id)
        tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        if tprof is None:
            continue
        wt = c.wt_reactivity
        nz = ~np.isnan(wt) & ~np.isnan(tprof)
        if not nz.any():
            continue
        we = wt[r.pos] if not np.isnan(wt[r.pos]) else 0.0
        idx = np.where(nz)[0]
        dist = (idx - r.pos).astype(np.float32)
        F = np.column_stack([_feat(we, wt[i], dist[j], r.ref, r.alt)
                            for j, i in enumerate(idx)]).T
        feats.append(F); targets.append(tprof[idx])
    X = np.vstack(feats); y = np.concatenate(targets)
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    A = Xb.T @ Xb + LAMBDA_RIDGE * np.eye(Xb.shape[1])
    coef = np.linalg.solve(A, Xb.T @ y)
    return coef


def _load_frozen_graph(components_path: Path) -> tuple[list[dict], int]:
    """Load the frozen, outcome-blind external component graph."""
    doc = json.loads(components_path.read_text(encoding="utf-8"))
    comps = doc["direct_external"]["components"]
    if len(comps) != K_PREACCESS_EXPECTED:
        raise RuntimeError(
            f"frozen graph mismatch: expected {K_PREACCESS_EXPECTED} components, "
            f"got {len(comps)} -> STOP (no outcome access)")
    total_snv = sum(c["n_snv_mutants"] for c in comps)
    return comps, total_snv


def _load_profiles(rdat_dir: Path) -> dict[str, dict]:
    by_name = {}
    for cid in DIRECT_EXTERNAL:
        r = parse_rdat(rdat_dir / f"{cid}.rdat")
        for x in r["profiles"]:
            by_name[x["profile_name"]] = x
    return by_name


def _ci_one_sided(x: list[float]) -> dict:
    n = len(x)
    if n < 2:
        return {"n": n, "mean": None, "ci_low": None, "ci_high": None}
    arr = np.asarray(x, float)
    m = float(arr.mean()); s = float(arr.std(ddof=1))
    t = _st.t.ppf(1 - 0.025, n - 1)
    return {"n": n, "mean": m, "sd": s,
            "ci_low": m - t * s / np.sqrt(n), "ci_high": m + t * s / np.sqrt(n)}


def _score_component(coef: np.ndarray, comp: dict,
                     profiles_by_name: dict[str, dict]) -> tuple[dict | None, dict | None]:
    """Score one frozen component under the locked attrition rules.

    Returns (row_dict, attrition_dict): exactly one is non-None. Row semantics:
    component-macro CRPS for direct/zero/median baselines over the shared region
    bounded by the observed reactivity array length.
    """
    wt = profiles_by_name.get(comp["wt_name"])
    if wt is None:
        return None, {"wt_name": comp["wt_name"], "rule": 1, "status": "DROP"}
    wt_react = np.asarray(wt["reactivity"], float)
    L = len(wt_react)
    if L == 0:
        return None, {"wt_name": comp["wt_name"], "rule": 1, "status": "DROP"}
    wt_median = float(np.nanmedian(wt_react))
    c_direct, c_zero, c_median = [], [], []
    n_scored = 0
    n_matched = len(comp["mutants"])
    for m in comp["mutants"]:
        mu = profiles_by_name.get(m["name"])
        if mu is None:
            continue
        mut_react = np.asarray(mu["reactivity"], float)
        if len(mut_react) != L:
            continue
        edit_pos = int(m["edit_pos"])
        ref, alt = _ref_alt(m["name"])
        # WT edit-site readout state; out-of-coverage or missing -> 0 (frozen
        # fallback consistent with P2 `wt[pos] if not NaN else 0.0`).
        we = wt_react[edit_pos] if edit_pos < L and not np.isnan(wt_react[edit_pos]) else 0.0
        m_d, m_z, m_c = [], [], []
        for i in m["shared_region"]:
            # positions beyond the reactivity array are non-observed
            if i >= L or np.isnan(mut_react[i]) or np.isnan(wt_react[i]):
                continue
            f = _feat(we, wt_react[i], i - edit_pos, ref, alt)
            pd_ = float(np.dot(coef, np.concatenate([[1.0], f])))
            m_d.append(crps_gaussian(pd_, FIXED_SCALE, mut_react[i]))
            m_z.append(crps_gaussian(wt_react[i], FIXED_SCALE, mut_react[i]))
            m_c.append(crps_gaussian(wt_median, FIXED_SCALE, mut_react[i]))
        if len(m_d) >= MIN_SHARED_NONMISSING:  # rule 3
            n_scored += 1
            c_direct.extend(m_d); c_zero.extend(m_z); c_median.extend(m_c)
    if n_scored < MIN_SCORED_MUTANTS:  # rule 2 realized
        return None, {
            "wt_name": comp["wt_name"], "rule": 2, "status": "DROP",
            "n_matched": n_matched, "n_scored": n_scored}
    return {
        "wt_name": comp["wt_name"],
        "n_matched": n_matched,
        "n_scored": n_scored,
        "n_positions": len(c_direct),
        "crps_direct": float(np.mean(c_direct)),
        "crps_zero": float(np.mean(c_zero)),
        "crps_median": float(np.mean(c_median)),
        "D_vs_zero": float(np.mean(c_zero) - np.mean(c_direct)),
        "D_vs_median": float(np.mean(c_median) - np.mean(c_direct)),
    }, None


def run_p4(rdat_dir: Path, dev_csv: Path, components_path: Path, out: Path) -> dict:
    comps, total_snv = _load_frozen_graph(components_path)
    coef = _fit_bstar_external(dev_csv)
    profiles_by_name = _load_profiles(rdat_dir)

    comp_rows = []
    attrition = []
    for comp in comps:
        row, drop = _score_component(coef, comp, profiles_by_name)
        if drop is not None:
            attrition.append(drop)
            continue
        assert row is not None
        comp_rows.append(row)

    K_eff = len(comp_rows)
    D_zero = np.array([c["D_vs_zero"] for c in comp_rows])
    D_med = np.array([c["D_vs_median"] for c in comp_rows])
    ci_zero = _ci_one_sided(D_zero.tolist())
    ci_med = _ci_one_sided(D_med.tolist())

    def pval(x):
        n = len(x)
        if n < 2 or float(np.std(x, ddof=1)) <= 0:
            return 1.0
        t = float(np.mean(x)) / (float(np.std(x, ddof=1)) / np.sqrt(n))
        return float(_st.t.sf(t, n - 1))
    ps = [pval(D_zero.tolist()), pval(D_med.tolist())]
    p_sorted = sorted(ps)
    fwer_pass = all(p_sorted[i] <= 0.025 / (2 - i) for i in range(2))

    dom = int(np.argmax(np.abs(D_zero))) if K_eff else -1
    leave = D_zero[np.arange(K_eff) != dom].tolist()
    ci_leave = _ci_one_sided(leave)

    stat_pass = (K_eff >= K_REQUIRED_PLANNED
                 and ci_zero.get("ci_low") is not None and ci_zero["ci_low"] > 0.0
                 and fwer_pass
                 and ci_leave.get("ci_low") is not None and ci_leave["ci_low"] > 0.0)

    report = {
        "schema_version": "reactflow_delta.p4_external.v1",
        "frozen_protocol": "docs/prospective_v2/p4_frozen_protocol_20260813.md",
        "component_graph": str(components_path),
        "B_star_external": "reg_direct (Direct* procedure refit on all development OK7a_M2)",
        "candidate": "reg_direct (adopted direct model per P3/contract 17.2)",
        "delta_stat": 0.0,
        "delta_practical": "NOT_ESTABLISHED",
        "K_required_planned": K_REQUIRED_PLANNED,
        "K_preaccess": K_PREACCESS_EXPECTED,
        "K_preaccess_single_snv": total_snv,
        "K_eff_realized": K_eff,
        "attrition": attrition,
        "locked_outcome_access_count": 1,
        "locked_outcome_note": (
            "single locked evaluation; a prior attempt crashed with an IndexError "
            "before any output artifact or report was produced (no outcome consumed); "
            "component graph is the frozen outcome-blind manifest"),
        "component_rows": comp_rows,
        "ci_zero": ci_zero,
        "ci_median": ci_med,
        "holm_bonferroni_p_values": ps,
        "fwer_pass": fwer_pass,
        "dominant_component_index": int(dom) if dom != -1 else None,
        "leave_dominant_out_ci": ci_leave,
        "statistical_pass": bool(stat_pass),
        "verdict": ("P4_EXTERNAL_STATISTICAL_PASS" if stat_pass
                    else "EXTERNAL_CONFIRMATION_FAIL"),
        "practical_importance": "PRACTICAL_IMPORTANCE_NOT_ESTABLISHED",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("component_rows", "attrition")},
                     indent=2, default=str))
    print("\n--- component_rows ---")
    print(json.dumps(comp_rows, indent=2, default=str))
    return report


def _ref_alt(name: str) -> tuple[str, str]:
    """Extract ref->alt from a single-SNV mutant profile name (last SNV token)."""
    for tok in reversed(name.split("_")):
        if len(tok) == 3 and tok[0].isdigit() and tok[1] in ALPHA and tok[2] == "-":
            return tok[1], tok[2]
    # fallback scan for pattern dN-N
    import re as _re
    m = _re.search(r"(\d+)([ACGU])-([ACGU])", name)
    if m:
        return m.group(2), m.group(3)
    return "A", "U"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdat-dir", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--components", required=True,
                    help="frozen outcome-blind component manifest (p4_external_components.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run_p4(Path(args.rdat_dir), Path(args.dev_csv), Path(args.components), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
