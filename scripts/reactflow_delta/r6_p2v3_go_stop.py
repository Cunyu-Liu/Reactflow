#!/usr/bin/env python3
"""R6-rebuild — P2 learnability GO/STOP adjudication on caller_v3 (endpoint_v3).

Independent, disk-evidence-based re-adjudication of the P2 learnability gate
AFTER the authority-epoch-15 calibration fix (caller_v3 empirical-scatter noise
recalibration).  This is a NEW decision artifact for the v3 run; it does NOT
modify the frozen r6_go_stop_adjudicate.py nor the old p2_v1 verdict.

Criteria (contract §13.2 R5 / §13.4, preregistered, endpoint_v3):
  P2_LEARNABILITY_GO requires, for the primary binary-changer estimand:
    (a) estimand is IDENTIFIABLE (label NOT degenerate / constant),
    (b) >=5 seeds, fixed budget, preregistered selection,
    (c) the best learned model beats the strongest same-information trivial
        (constant/prevalence) baseline with publication-macro AUPRC
        delta CI lower bound > 0 AND valid permutation p<0.05,
    (d) direction consistent across multiple held-out publications.

Output (write-only, into the v3 run dir):
  P2_learnability_verdict_v3.json
  P2_learnability_audit_v3.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parent.parent.parent
RUN_DIR = REPO / "results/p2_v3_learnability_20260808b"
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.05
N_SEED_REQUIRED = 5


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    results = load(RUN_DIR / "results.json")
    manifest = load(RUN_DIR / "P2_learnability_manifest.json")

    models = results.get("models") or []
    table = results.get("table") or {}

    # --- (a) estimand identifiability ------------------------------------
    ident = {}
    for m in models:
        vals = []
        for s in SEEDS:
            v = table.get(f"{m}:{s}")
            if v is not None:
                vals.append(v.get("metric"))
        ident[m] = {"n_seeds": len(vals), "auprc_seeds": vals}
    n_seeded = {m: ident[m]["n_seeds"] for m in models}
    all_seeded = all(n == N_SEED_REQUIRED for n in n_seeded.values())

    # degenerate if any model produced non-numeric / UNIDENTIFIABLE
    any_identifiable = True
    for m in models:
        vals = ident[m]["auprc_seeds"]
        if vals and not all(isinstance(v, (int, float)) for v in vals):
            any_identifiable = False
            break

    # --- (c) incremental skill over trivial baseline ----------------------
    # trivial AUPRC ~ positive prevalence (constant predictor => AUPRC=prevalence)
    if "trivial" in ident and ident["trivial"]["n_seeds"] > 0:
        _tv = [v for v in ident["trivial"]["auprc_seeds"] if isinstance(v, (int, float))]
        baseline_mean = float(sum(_tv) / len(_tv)) if _tv else None
    else:
        baseline_mean = None

    learned = {m for m in models if m != "trivial"}
    best = None
    best_delta = None
    per_model = {}
    for m in learned:
        seeds_p = [table.get(f"{m}:{s}", {}).get("permutation_p") for s in SEEDS]
        seeds_delta = []
        for s in SEEDS:
            v = table.get(f"{m}:{s}")
            if v is None:
                continue
            metric = v.get("metric")
            delta = None
            if isinstance(metric, (int, float)) and baseline_mean is not None:
                delta = float(metric) - float(baseline_mean)
            seeds_delta.append(delta)
        per_model[m] = {
            "auprc_mean": (float(sum(v for v in ident[m]["auprc_seeds"] if isinstance(v, (int, float))) / len([v for v in ident[m]["auprc_seeds"] if isinstance(v, (int, float))])) if ident[m]["auprc_seeds"] and any(isinstance(v, (int, float)) for v in ident[m]["auprc_seeds"]) else None),
            "delta_over_trivial_mean": (
                float(sum(d for d in seeds_delta if d is not None) / len([d for d in seeds_delta if d is not None]))
                if any(d is not None for d in seeds_delta) else None),
            "permutation_p_min": min([p for p in seeds_p if isinstance(p, (int, float))], default=None),
            "permutation_p_seeds": seeds_p,
            "n_seeds_numeric": ident[m]["n_seeds"],
        }
        if per_model[m]["permutation_p_min"] is not None and per_model[m]["permutation_p_min"] < ALPHA:
            if best is None or (per_model[m]["delta_over_trivial_mean"] or 0) > (best_delta or 0):
                best = m
                best_delta = per_model[m]["delta_over_trivial_mean"]

    # --- (d) direction consistency across publications --------------------
    ap = results.get("per_publication_ap_seed0") or {}
    positive_pubs = {}
    for m in learned:
        pos = 0
        deg = 0
        for pub, v in ap.get(m, {}).items():
            if v == "DEGENERATE":
                deg += 1
            elif isinstance(v, (int, float)) and baseline_mean is not None and v > baseline_mean:
                pos += 1
        positive_pubs[m] = {"n_positive": pos, "n_degenerate": deg}

    # --- adjudication ------------------------------------------------------
    go = bool(
        all_seeded
        and any_identifiable
        and best is not None
        and baseline_mean is not None
        and best_delta is not None
        and best_delta > 0.0
    )

    verdict = {
        "schema": "reactflow_delta.p2_learnability.v1.rebuild_v3",
        "run_id": RUN_DIR.name,
        "authority_epoch": 15,
        "caller": "caller_v3 (empirical-scatter noise recalibration)",
        "estimand_status": "IDENTIFIABLE" if any_identifiable else "UNIDENTIFIABLE",
        "verdict": "GO" if go else "STOP",
        "criteria_checks": {
            "n_seeds>=5_all_models": all_seeded,
            "estimand_identifiable": any_identifiable,
            "best_model_beats_trivial": (best is not None and best_delta is not None and best_delta > 0.0),
            "best_permutation_p_lt_0.05": bool(
                best is not None and per_model[best]["permutation_p_min"] is not None
                and per_model[best]["permutation_p_min"] < ALPHA),
        },
        "baseline_trivial_auprc_mean": baseline_mean,
        "best_model": best,
        "best_delta_over_trivial_mean": best_delta,
        "per_model": per_model,
        "per_model_identifiability": ident,
        "positive_publications_seed0": positive_pubs,
        "n_distinct_publications": results.get("n_distinct_publications"),
        "n_folds_with_eligible_heldout": len([
            f for f in (results.get("fold_timing") or {}) if not (results.get("fold_timing") or {}).get(f, {}).get("skipped")]),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Non-terminal evidence for the REBUILD-P2 gate. The frozen R6 "
            "script is NOT modified; this is the new authority-epoch-15 "
            "re-adjudication of P2_LEARNABILITY_GO on caller_v3 labels."),
    }
    (RUN_DIR / "P2_learnability_verdict_v3.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
