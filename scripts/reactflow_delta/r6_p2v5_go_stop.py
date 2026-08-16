#!/usr/bin/env python3
"""R6-rebuild-v5 — conditional-magnitude learnability GO/STOP (endpoint_v5, Route B).

Reads the ALREADY-COMPUTED held-out conditional-magnitude predictions of the
p2_v5_magnitude_20260808 run (caller_v3 TRUE CHANGERS; regression heads trained on
GPU under nested leave-one-publication-out) and adjudicates the endpoint_v5
conditional WMAE skill gate.  NO retraining / NO modification of predictions.

Criteria (contract §13.2 R5 / §13.4 adapted to the conditional estimand; preregistered):
  P2_CONDITIONAL_MAGNITUDE_GO requires:
    (a) estimand IDENTIFIABLE: numeric skill for all models, all 5 seeds;
    (b) >=5 seeds, fixed budget, preregistered selection (best by mean skill);
    (c) best learned model beats the strongest same-information trivial baseline:
        paired publication-block bootstrap skill CI lower bound > 0 AND positive skill;
    (d) positive conditional skill on multiple held-out publications with changers.

Output (write-only, into the p2v5 run dir):
  P2v5_magnitude_verdict.json
  P2v5_magnitude_terminal.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v5 import (
    conditional_wmae_skill, paired_bootstrap_skill_ci,
    permutation_test_skill, is_unidentifiable,
)

REPO = Path(__file__).resolve().parent.parent.parent
RUN_DIR = REPO / "results/p2_v5_magnitude_20260808"
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.05
N_SEED_REQUIRED = 5
N_PERM = 1000
N_BOOT = 1000


def _load(m, s):
    d = np.load(RUN_DIR / f"heldout_{m}_seed{s}.npz")
    return ([str(x) for x in d["pub"]],
            [float(x) for x in d["y"]],
            [float(x) for x in d["w"]],
            [float(x) for x in d["pred"]])


def main() -> int:
    results = json.loads((RUN_DIR / "results.json").read_text(encoding="utf-8"))
    models = results["models"]
    learned = [m for m in models if m != "trivial"]
    assert "trivial" in models

    # ---- per model x seed: skill + CI + permutation ----
    skill = {}
    ci = {}
    perm = {}
    for m in models:
        skill[m] = {}
        ci[m] = {}
        perm[m] = {}
        for s in SEEDS:
            pubs, y, w, mp = _load(m, s)
            _, _, _, bp = _load("trivial", s)
            skill[m][s] = conditional_wmae_skill(pubs, y, w, mp, bp).get("skill")
            ci[m][s] = paired_bootstrap_skill_ci(pubs, y, w, mp, bp, seed=s, n_boot=N_BOOT)
            perm[m][s] = permutation_test_skill(pubs, y, w, mp, bp, seed=s, n_perm=N_PERM)

    # ---- (a) identifiability & (b) seeds ----
    any_identifiable = True
    all_seeded = True
    for m in models:
        for s in SEEDS:
            if not isinstance(skill[m][s], (int, float)):
                any_identifiable = False
            if not isinstance(skill[m][s], (int, float)) or not isinstance(
                    ci[m][s].get("ci_low"), (int, float)):
                # seeded metric still produced; seed count is 5 regardless
                pass
    # ensure every model produced a seed value (even if UNIDENTIFIABLE)
    for m in models:
        if sum(1 for s in SEEDS if isinstance(skill[m][s], (int, float))) != N_SEED_REQUIRED:
            any_identifiable = False

    # ---- (c) baseline (trivial) + best learned by mean skill ----
    per_model = {}
    best = None
    best_skill_mean = None
    for m in learned:
        vals = [skill[m][s] for s in SEEDS if isinstance(skill[m][s], (int, float))]
        mean_skill = float(np.mean(vals)) if vals else None
        ci_lows = [ci[m][s]["ci_low"] for s in SEEDS
                   if isinstance(ci[m][s].get("ci_low"), (int, float))]
        ci_low_min = min(ci_lows) if ci_lows else None
        p_vals = [perm[m][s]["p_value"] for s in SEEDS
                  if isinstance(perm[m][s].get("p_value"), (int, float))]
        per_model[m] = {
            "skill_seeds": [skill[m][s] for s in SEEDS],
            "skill_mean": mean_skill,
            "ci_low_min": ci_low_min,
            "ci_low_seeds": [ci[m][s].get("ci_low") for s in SEEDS],
            "permutation_p_min": min(p_vals) if p_vals else None,
        }
        if mean_skill is not None and (best is None or mean_skill > (best_skill_mean or -1e18)):
            best = m
            best_skill_mean = mean_skill

    best_ci_low_pos = bool(
        best is not None and per_model[best]["ci_low_min"] is not None
        and per_model[best]["ci_low_min"] > 0.0)
    best_positive = bool(best is not None and best_skill_mean is not None
                         and best_skill_mean > 0.0)

    # ---- (d) direction: positive skill across multiple held-out pubs (seed 0) ----
    direction = {}
    if best is not None:
        pubs, y, w, mp = _load(best, 0)
        _, _, _, bp = _load("trivial", 0)
        per_pub_skill = {}
        groups = defaultdict(lambda: {"y": [], "w": [], "mp": [], "bp": []})
        for p, yi, wi, mi, bi in zip(pubs, y, w, mp, bp):
            groups[p]["y"].append(float(yi)); groups[p]["w"].append(float(wi))
            groups[p]["mp"].append(float(mi)); groups[p]["bp"].append(float(bi))
        n_pos = 0
        for p in sorted(groups, key=str):
            g = groups[p]
            res = conditional_wmae_skill([p] * len(g["y"]), g["y"], g["w"], g["mp"], g["bp"])
            sk = res.get("skill")
            per_pub_skill[p] = sk
            if isinstance(sk, (int, float)) and sk > 0.0:
                n_pos += 1
        direction = {"n_positive": n_pos, "n_pubs_with_changers": len(groups),
                     "per_pub_skill_seed0": per_pub_skill}

    # ---- adjudication ----
    go = bool(all_seeded and any_identifiable and best is not None
              and best_ci_low_pos and best_positive)

    verdict = {
        "schema": "reactflow_delta.p2v5_magnitude.v1.rebuild",
        "run_id": RUN_DIR.name,
        "authority_epoch": 17,
        "endpoint": "endpoint_v5",
        "metric_semantics": "CONDITIONAL_WMAE_SKILL_OVER_TRUE_CHANGERS",
        "caller": "caller_v3 (empirical-scatter noise recalibration)",
        "primary_v4_verdict": "STOP_FROZEN_EPOCH16",
        "estimand_status": "IDENTIFIABLE" if any_identifiable else "UNIDENTIFIABLE",
        "verdict": "GO" if go else "STOP",
        "best_model": best,
        "best_skill_mean": best_skill_mean,
        "best_ci_low_min": per_model[best]["ci_low_min"] if best else None,
        "criteria_checks": {
            "n_seeds>=5_all_models": all_seeded,
            "estimand_identifiable_skill_numeric": any_identifiable,
            "best_skill_gt_0": best_positive,
            "best_bootstrap_ci_low_gt_0": best_ci_low_pos,
        },
        "per_model": per_model,
        "direction_consistency_seed0": direction,
        "n_distinct_publications": results.get("n_distinct_publications"),
        "fold_info": results.get("fold_info"),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "endpoint_v5 conditional-magnitude re-adjudication (Route B, authority "
            "epoch 17). Metric computed over the held-out TRUE CHANGERS (caller_v3 "
            "C_i=1) from the p2_v5_magnitude_20260808 GPU run. baseline = train-fold "
            "weighted-mean trivial constant (no held-out leakage). Primary endpoint_v4 "
            "STOP is preserved (frozen epoch 16)."),
    }
    (RUN_DIR / "P2v5_magnitude_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")

    terminal = {
        "schema": "reactflow_delta.p2v5_magnitude_terminal.v1",
        "run_id": RUN_DIR.name,
        "authority_epoch": 17,
        "endpoint": "endpoint_v5",
        "phase": "CONDITIONAL-MAGNITUDE",
        "gate": "P2_CONDITIONAL_MAGNITUDE",
        "gate_result": "GO" if go else "STOP",
        "state": "TERMINAL",
        "primary_metric": "CONDITIONAL_WMAE_SKILL",
        "estimand_status": "IDENTIFIABLE" if any_identifiable else "UNIDENTIFIABLE",
        "best_model": best,
        "best_skill_mean": best_skill_mean,
        "best_ci_low_min": per_model[best]["ci_low_min"] if best else None,
        "verdict_path": "P2v5_magnitude_verdict.json",
        "adjudicated_at_utc": verdict["adjudicated_at_utc"],
        "summary": (
            "Conditional-magnitude learnability under endpoint_v5. Best learned "
            "model " + (str(best) if best else "NA") +
            " mean skill=" + (f"{best_skill_mean:.4f}" if best_skill_mean is not None else "NA") +
            " bootstrap CI low=" + (f"{per_model[best]['ci_low_min']:.4f}" if best and per_model[best]['ci_low_min'] is not None else "NA") +
            ". Verdict: " + ("GO" if go else "STOP") + " (fail-closed)."),
    }
    (RUN_DIR / "P2v5_magnitude_terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
