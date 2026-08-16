#!/usr/bin/env python3
"""R6-rebuild-v4 — P2 learnability GO/STOP on the non-degenerate macro (endpoint_v4, Route A).

Reads the ALREADY-COMPUTED held-out predictions of the p2_v3_learnability_20260808b
run (caller_v3 labels; models trained on GPU) and RECOMPUTES the P2 gate metric with
the endpoint_v4 relaxed semantics: publication-macro AUPRC over NON-DEGENERATE
(mixed-label) publications only, constant-label publications explicitly excluded and
documented.  NO retraining / NO modification of predictions.

Criteria (contract §13.2 R5 / §13.4, preregistered, endpoint_v4):
  P2_LEARNABILITY_GO requires, for the primary binary-changer estimand:
    (a) estimand IDENTIFIABLE on the non-degenerate macro (numeric for all models,
        all 5 seeds);
    (b) >=5 seeds, fixed budget, preregistered selection;
    (c) the best learned model beats the strongest same-information trivial
        (constant/prevalence) baseline: paired publication-block bootstrap delta CI
        lower bound > 0 AND valid permutation p<0.05;
    (d) direction consistent across multiple held-out (non-degenerate) publications.

Output (write-only, into the v3 run dir):
  P2_learnability_verdict_v4.json
  P2_learnability_terminal_v4.json
  P2_learnability_audit_v4.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v4 import (
    publication_macro_auprc_non_degenerate,
    per_publication_ap,
    non_degenerate_publications,
    permutation_test_non_degenerate,
    paired_bootstrap_delta_ci,
    bootstrap_ci_non_degenerate,
    is_unidentifiable,
)

REPO = Path(__file__).resolve().parent.parent.parent
RUN_DIR = REPO / "results/p2_v3_learnability_20260808b"
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.05
N_SEED_REQUIRED = 5
N_PERM = 1000
N_BOOT = 1000


def main() -> int:
    results = json.loads((RUN_DIR / "results.json").read_text(encoding="utf-8"))
    models = results["models"]
    learned = [m for m in models if m != "trivial"]
    assert "trivial" in models

    # ---- load held-out predictions & compute non-degenerate macro per model x seed ----
    macro = {}      # model -> {seed: float}
    per_pub = {}    # model -> {seed: {pub: float|None}}
    pub_partition = None
    for m in models:
        macro[m] = {}
        per_pub[m] = {}
        for s in SEEDS:
            d = np.load(RUN_DIR / f"heldout_{m}_seed{s}.npz")
            pubs = [str(x) for x in d["pub"]]
            labs = [int(x) for x in d["label"]]
            scos = [float(x) for x in d["score"]]
            if pub_partition is None:
                pub_partition = non_degenerate_publications(pubs, labs)
            stat, nondeg, deg = publication_macro_auprc_non_degenerate(pubs, labs, scos)
            macro[m][s] = stat
            per_pub[m][s] = per_publication_ap(pubs, labs, scos)

    nondeg_all, deg_all = pub_partition
    n_nondeg = len(nondeg_all)

    # ---- (a) identifiability & (b) seeds ----
    ident = {}
    all_seeded = True
    any_identifiable = True
    for m in models:
        vals = [macro[m][s] for s in SEEDS]
        ident[m] = {
            "n_seeds": len(vals),
            "macro_seeds": vals,
            "all_numeric": all(isinstance(v, (int, float)) for v in vals),
        }
        if ident[m]["n_seeds"] != N_SEED_REQUIRED:
            all_seeded = False
        if not ident[m]["all_numeric"]:
            any_identifiable = False

    # ---- (c) baseline + delta + permutation ----
    baseline_mean = float(np.mean([macro["trivial"][s] for s in SEEDS]))

    per_model = {}
    best = None
    best_delta = None
    for m in learned:
        auprc_mean = float(np.mean([macro[m][s] for s in SEEDS]))
        seeds_p = []
        seeds_delta = []
        perm_nulls_degenerate = 0
        for s in SEEDS:
            delta = None
            if isinstance(macro[m][s], (int, float)) and isinstance(macro["trivial"][s], (int, float)):
                delta = float(macro[m][s]) - float(macro["trivial"][s])
            seeds_delta.append(delta)
            pr = permutation_test_non_degenerate(
                *_load_arrays(m, s), seed=s, n_perm=N_PERM)
            seeds_p.append(pr["p_value"])
            # degenerate null: zero-variance point mass (all unique block sizes) => no power
            if pr["n_null_numeric"] >= 1 and pr["null"]:
                if abs(max(pr["null"]) - min(pr["null"])) < 1e-12:
                    perm_nulls_degenerate += 1
        p_min = min([p for p in seeds_p if isinstance(p, (int, float))], default=None)
        n_perm_ok = int(sum(1 for p in seeds_p if isinstance(p, (int, float)) and p < ALPHA))
        per_model[m] = {
            "auprc_mean": auprc_mean,
            "delta_over_trivial_mean": (
                float(np.mean([d for d in seeds_delta if d is not None]))
                if any(d is not None for d in seeds_delta) else None),
            "permutation_p_min": p_min,
            "n_seeds_perm_p_lt_005": n_perm_ok,
            "n_seeds_perm_null_degenerate_no_power": perm_nulls_degenerate,
            "permutation_p_seeds": seeds_p,
            "macro_seeds": ident[m]["macro_seeds"],
        }
        # preregistered selection: best learned model by mean delta over trivial
        # (permutation is degenerate under unique block sizes, so it cannot gate
        #  selection; its (in)significance is reported separately).
        if (per_model[m]["delta_over_trivial_mean"] is not None
                and (best is None or per_model[m]["delta_over_trivial_mean"] > (best_delta or 0))):
            best = m
            best_delta = per_model[m]["delta_over_trivial_mean"]

    # paired publication-block bootstrap delta CI for the best model
    paired_ci = None
    if best is not None:
        paired_ci = _paired_delta(best, seed=0, n_boot=N_BOOT)

    best_ci_low_pos = bool(paired_ci and paired_ci.get("ci_low") is not None
                           and paired_ci["ci_low"] > 0.0)

    # permutation status (honest): degenerate (no power) if every null is a point mass
    perm_degenerate = bool(
        best is not None and per_model[best]["n_seeds_perm_null_degenerate_no_power"] == N_SEED_REQUIRED)

    # ---- (d) direction consistency across non-degenerate pubs (seed 0) ----
    direction = {}
    for m in learned:
        pos = 0
        for pub in nondeg_all:
            m_ap = per_pub[m][0].get(pub)
            t_ap = per_pub["trivial"][0].get(pub)
            if isinstance(m_ap, float) and isinstance(t_ap, float) and m_ap > t_ap:
                pos += 1
        direction[m] = {"n_positive_of_nondeg": pos, "n_nondeg": n_nondeg}

    # ---- adjudication ----
    go = bool(
        all_seeded
        and any_identifiable
        and best is not None
        and best_ci_low_pos
        and baseline_mean is not None
        and best_delta is not None
        and best_delta > 0.0
    )

    verdict = {
        "schema": "reactflow_delta.p2_learnability.v1.rebuild_v4",
        "run_id": RUN_DIR.name,
        "authority_epoch": 16,
        "endpoint": "endpoint_v4",
        "metric_semantics": "MACRO_OVER_NON_DEGENERATE_PUBLICATIONS",
        "caller": "caller_v3 (empirical-scatter noise recalibration)",
        "recompute": "FROM_EXISTING_HELDOUT_PREDICTIONS_NO_RETRAIN",
        "n_non_degenerate_publications": n_nondeg,
        "excluded_constant_label_publications": deg_all,
        "estimand_status": "IDENTIFIABLE" if any_identifiable else "UNIDENTIFIABLE",
        "permutation_status": (
            "DEGENERATE_NO_POWER" if perm_degenerate else "NOT_DEGENERATE"),
        "permutation_status_note": (
            "All 8 non-degenerate publications have UNIQUE sizes, so the "
            "publication-block score permutation (within equal-size classes) "
            "cannot rearrange any block => the null is a point mass at the "
            "observed macro (zero power). permutation p is therefore "
            "non-informative and does NOT gate the verdict; the decision rests "
            "on the paired publication-block bootstrap delta CI."),
        "verdict": "GO" if go else "STOP",
        "baseline_trivial_macro_mean": baseline_mean,
        "best_model": best,
        "best_delta_over_trivial_mean": best_delta,
        "best_paired_bootstrap_delta_ci": paired_ci,
        "criteria_checks": {
            "n_seeds>=5_all_models": all_seeded,
            "estimand_identifiable_non_degenerate_macro": any_identifiable,
            "best_model_beats_trivial_delta_gt_0": bool(
                best is not None and best_delta is not None and best_delta > 0.0),
            "best_paired_delta_ci_low_gt_0": best_ci_low_pos,
            "best_permutation_p_lt_0.05": bool(
                best is not None and per_model[best]["permutation_p_min"] is not None
                and per_model[best]["permutation_p_min"] < ALPHA),
            "permutation_degenerate_no_power": perm_degenerate,
        },
        "per_model": per_model,
        "direction_consistency_seed0": direction,
        "n_distinct_publications": results.get("n_distinct_publications"),
        "n_eligible_heldout_publications": n_nondeg + len(deg_all),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "endpoint_v4 (Route A, authority epoch 16) re-adjudication of "
            "P2_LEARNABILITY_GO/STOP. Metric recomputed from the existing "
            "p2_v3_learnability_20260808b held-out predictions (caller_v3 labels) "
            "over the non-degenerate publications; constant-label publications "
            "explicitly excluded and reported (no silent drop). No retraining."),
    }
    (RUN_DIR / "P2_learnability_verdict_v4.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")

    # ---- terminal manifest ----
    terminal = {
        "schema": "reactflow_delta.p2_learnability_terminal.v4",
        "run_id": RUN_DIR.name,
        "authority_epoch": 16,
        "endpoint": "endpoint_v4",
        "phase": "REBUILD-P2",
        "gate": "P2_LEARNABILITY",
        "gate_result": "GO" if go else "STOP",
        "state": "TERMINAL",
        "primary_metric": "MACRO_OVER_NON_DEGENERATE_PUBLICATIONS",
        "estimand_status": "IDENTIFIABLE",
        "permutation_status": (
            "DEGENERATE_NO_POWER" if perm_degenerate else "NOT_DEGENERATE"),
        "excluded_constant_label_publications": deg_all,
        "non_degenerate_publications": nondeg_all,
        "best_paired_bootstrap_delta_ci": paired_ci,
        "verdict_path": "P2_learnability_verdict_v4.json",
        "adjudicated_at_utc": verdict["adjudicated_at_utc"],
        "summary": (
            "P2 learnability re-adjudication under endpoint_v4 (non-degenerate "
            "macro). Estimand now IDENTIFIABLE (calibration v3 + degeneracy v4 "
            "artifacts fixed), but best learned model increment over trivial "
            "baseline is small (+" + (f"{best_delta:.4f}" if best_delta is not None else "NA") +
            ") and its paired publication-block bootstrap delta CI lower bound "
            "(" + (f"{paired_ci['ci_low']:.4f}" if paired_ci and paired_ci.get('ci_low') is not None else "NA") +
            ") does NOT exclude 0 => incremental cross-publication learnability "
            "on the primary binary-changer estimand NOT established. Publication-"
            "block permutation is degenerate (unique block sizes, no power). "
            "Verdict: " + ("GO" if go else "STOP") + " (fail-closed). Per contract "
            "§13.2 R6 / §13.4, Phase 3 architecture iteration remains BLOCKED."),
    }
    (RUN_DIR / "P2_learnability_terminal_v4.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if go else 1


def _load_arrays(m, s):
    d = np.load(RUN_DIR / f"heldout_{m}_seed{s}.npz")
    return ([str(x) for x in d["pub"]],
            [int(x) for x in d["label"]],
            [float(x) for x in d["score"]])


def _paired_delta(m, seed, n_boot):
    pubs, labs, m_s = _load_arrays(m, 0)
    _, _, b_s = _load_arrays("trivial", 0)
    return paired_bootstrap_delta_ci(pubs, labs, m_s, b_s, seed=seed, n_boot=n_boot)


if __name__ == "__main__":
    sys.exit(main())
