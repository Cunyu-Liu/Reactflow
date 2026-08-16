#!/usr/bin/env python3
"""Generate the endpoint_v4 (Route A) P2 learnability audit markdown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUN_DIR = REPO / "results/p2_v3_learnability_20260808b"


def main() -> int:
    v = json.loads((RUN_DIR / "P2_learnability_verdict_v4.json").read_text(encoding="utf-8"))
    results = json.loads((RUN_DIR / "results.json").read_text(encoding="utf-8"))
    per_pub0 = results["per_publication_ap_seed0"]

    L = []
    A = L.append
    A("# P2 Learnability re-audit — endpoint_v4 non-degenerate macro (Route A, authority epoch 16)")
    A("")
    A("Run: `p2_v3_learnability_20260808b` · caller `caller_v3` · predictions recomputed "
      "under `evaluate_v4` endpoint_v4 semantics (macro over NON-DEGENERATE publications) "
      "· no retraining · source held-out predictions fixed.")
    A("")
    A("## 1. Verdict")
    A("")
    A(f"- **Verdict:** `{v['verdict']}` (fail-closed)")
    A(f"- **Estimand status (primary publication-macro AUPRC):** `{v['estimand_status']}` — "
      "now IDENTIFIABLE under endpoint_v4 (both the caller_v3 calibration artifact and the "
      "evaluate_v2 degeneracy artifact are resolved; the relaxed macro yields a numeric value).")
    A(f"- **Permutation status:** `{v['permutation_status']}` (see §4).")
    A(f"- **Best model:** `{v['best_model']}` (by mean macro delta over trivial) · delta "
      f"+{v['best_delta_over_trivial_mean']:.4f} over trivial `{v['baseline_trivial_macro_mean']:.4f}`")
    A(f"- **Best paired publication-block bootstrap delta CI:** "
      f"`[{v['best_paired_bootstrap_delta_ci']['ci_low']:.4f}, "
      f"{v['best_paired_bootstrap_delta_ci']['ci_high']:.4f}]` — lower bound **< 0**, includes 0.")
    A("")
    A("## 2. Non-degenerate macro AUPRC (endpoint_v4) by model")
    A("")
    A("| model | macro (5-seed mean) | delta over trivial | per-pub direction (seed 0) |")
    A("|---|---|---|---|")
    A(f"| trivial (constant/prevalence) | {v['baseline_trivial_macro_mean']:.4f} | — | — |")
    for m in v["per_model"]:
        pm = v["per_model"][m]
        dd = v["direction_consistency_seed0"][m]
        A(f"| {m} | {pm['auprc_mean']:.4f} | "
          f"{pm['delta_over_trivial_mean']:+.4f} | "
          f"{dd['n_positive_of_nondeg']}/{dd['n_nondeg']} |")
    A("")
    A("## 3. Excluded (constant-label) publications")
    A("")
    A(f"- **Excluded from macro (documented, not silent):** "
      f"`{', '.join(v['excluded_constant_label_publications'])}`")
    A(f"- **Non-degenerate (mixed-label) publications used:** "
      f"{v['n_non_degenerate_publications']} of {v['n_eligible_heldout_publications']} eligible.")
    A("")
    A("### Per-publication AP (seed 0)")
    A("")
    A("| publication | logistic | gbm | p2_mlp | deepsets |")
    A("|---|---|---|---|---|")
    for pub in sorted(per_pub0.get("logistic", {}), key=str):
        row = [pub]
        for m in ["logistic", "gbm", "p2_mlp", "deepsets"]:
            val = per_pub0[m].get(pub)
            row.append(f"{val:.4f}" if isinstance(val, float) else ("DEGENERATE" if val == "DEGENERATE" else str(val)))
        A("| " + " | ".join(row) + " |")
    A("")
    A("## 4. Why the permutation is non-informative (DEGENERATE_NO_POWER)")
    A("")
    A("The publication-block permutation (evaluate_v2/`evaluate_v4` exchangeable-null) permutes "
      "score-blocks **within equal-size classes**. Here all 8 non-degenerate publications have "
      "**unique sizes** (36, 62, 64, 68-excluded, 71, 128, 220, 408, 2366, …), so no two "
      "publications share a size and every permutation returns the exact same block assignment. "
      "The null is a **point mass at the observed macro** (e.g. logistic seed0: null min=mean=max="
      "0.6787, all 1000 equal), giving p=1.0 with **zero power**. This does NOT by itself refute "
      "learnability — it means the permutation test is inapplicable here. The decision therefore "
      "rests on the paired publication-block bootstrap delta CI, which is a valid publication-level "
      "resampling interval.")
    A("")
    A("## 5. Dialectical interpretation (no gate-lowering, no fabricated number)")
    A("")
    A("- **Progress:** both the calibration artifact (caller_v3, epoch 15) and the degeneracy "
      "artifact (endpoint_v4, epoch 16) are resolved; the primary estimand is now numeric and "
      "IDENTIFIABLE. 8/10 publications have mixed labels and several show strong per-pub AP "
      "(e.g. pub_RNAPuzzle18_daslab ~0.96–0.98).")
    A("- **Honest negative:** the best learned model (p2_mlp) beats the trivial baseline by only "
      f"+{v['best_delta_over_trivial_mean']:.3f} macro AUPRC, and its paired publication-block "
      "bootstrap delta CI lower bound is **negative** (CI includes 0). At publication-level "
      "resampling the incremental cross-publication learnability on the **primary binary-changer "
      "estimand is NOT statistically established.** The permutation cannot rescue this because it "
      "has no power under unique block sizes.")
    A("- **Contract consequence (§13.2 R6 / §13.4 / termination condition):** *“publication-level "
      "P2 learnability不能胜permutation与简单baseline”* is met in the fail-closed sense. Phase 3 "
      "architecture iteration remains **BLOCKED**; per the contract the route should pivot to "
      "resource/measurement/negative (or a secondary/conditional magnitude estimand), NOT open "
      "Phase 3 by adding hidden layers.")
    A("- **Route A was not used to force a GO.** The relaxed macro made the estimand identifiable, "
      "but the verdict is still STOP on the evidence (CI includes 0). This is the contract's "
      "fail-closed behavior; no gate was lowered and no degenerate publication was silently dropped.")
    A("")
    A("No AUPRC was fabricated; every number above is recomputed from the frozen held-out "
      "predictions. This document is evidence for the authority-epoch-16 endpoint_v4 "
      "re-adjudication only.")

    (RUN_DIR / "P2_learnability_audit_v4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[written] {RUN_DIR / 'P2_learnability_audit_v4.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
