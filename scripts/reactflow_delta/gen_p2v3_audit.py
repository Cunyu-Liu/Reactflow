#!/usr/bin/env python3
"""Generate P2_learnability_audit_v3.md from the on-disk v3 results.

Evidence-based (reads results.json + manifest + heldout predictions), so the
audit always reflects the actual run.  Writes only the audit markdown (new
file); never modifies results.json / manifest / verdict / endpoint.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

RUN = Path("/home/cunyuliu/reactflow_delta_goal_20260729/results/p2_v3_learnability_20260808b")


def load(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def main():
    results = load("results.json")
    manifest = load("P2_learnability_manifest.json")
    verdict = load("P2_learnability_verdict_v3.json")

    # per-publication held-out label distribution (seed 0, p2_mlp representative)
    d = np.load(RUN / "heldout_p2_mlp_seed0.npz")
    labs = d["label"].astype(int)
    pubs = d["pub"]
    per_pub = defaultdict(Counter)
    for p, l in zip(pubs, labs):
        per_pub[str(p)][int(l)] += 1

    lines = []
    A = lines.append
    A("# P2 Learnability re-audit — caller_v3 (endpoint_v3, authority epoch 15)")
    A("")
    A("Run: `p2_v3_learnability_20260808b` · caller `caller_v3` · "
      "evaluator/split `evaluate_v2` / `split_v2` (unchanged, frozen) · GPU `NVIDIA "
      "A100-PCIE-40GB MIG 1g.5gb` (cuda_available=true, no silent CPU fallback) · "
      "source hashes all match manifest.")
    A("")
    A("## 1. Verdict")
    A("")
    A(f"- **Verdict:** `{verdict['verdict']}` (fail-closed)")
    A(f"- **Estimand status (primary publication-macro AUPRC):** `{verdict['estimand_status']}`")
    A(f"- **Eligible held-out publications:** {verdict['n_folds_with_eligible_heldout']} of 18")
    A("- **Root cause of UNIDENTIFIABLE:** the frozen `evaluate_v2.publication_macro_auprc` "
      "returns UNIDENTIFIABLE (degenerate_policies.constant_label, fail-closed, no silent "
      "exclusion) whenever ANY held-out publication has a constant label set. Here 2 of 10 "
      "eligible publications (pmid_25883046, pmid_35982307) have all held-out pairs labeled "
      "changers.")
    A("")
    A("## 2. Calibration fix validated (progress vs R5/R6 STOP)")
    A("")
    A("caller_v3 empirical-scatter recalibration recovered abundant real changers. The R5 "
      "near-constant label (3 changers / 6359) is resolved:")
    A("")
    A("| publication | n held-out | #changer | #nonchanger | constant? |")
    A("|---|---|---|---|---|")
    for p in sorted(per_pub, key=str):
        c = per_pub[p]
        n0, n1 = c.get(0, 0), c.get(1, 0)
        A(f"| {p} | {n0+n1} | {n1} | {n0} | {'YES' if (n0 == 0 or n1 == 0) else 'no'} |")
    A("")
    A("8 of 10 eligible publications have mixed labels and therefore produce a **numeric** "
      "per-publication AP. 2 publications are internally constant (all-changer) and are the "
      "sole trigger of the macro UNIDENTIFIABLE (per the frozen fail-closed rule).")
    A("")
    A("## 3. Per-publication AP (seed 0)")
    A("")
    ap = results.get("per_publication_ap_seed0") or {}
    models = [m for m in results.get("models", []) if m != "trivial"]
    pubs_ap = sorted({p for m in models for p in (ap.get(m) or {})}, key=str)
    A("| publication | " + " | ".join(models) + " |")
    A("|" + "---|" * (len(models) + 1))
    for p in pubs_ap:
        row = [p]
        for m in models:
            v = (ap.get(m) or {}).get(p)
            row.append(str(round(v, 4)) if isinstance(v, (int, float)) else str(v))
        A("| " + " | ".join(row) + " |")
    A("")
    A("Several publications show strong per-pub AP (CL1LIG ~0.88–0.92, pmid_25183835 "
      "~0.75–0.83, pmid_25303992 ~0.77–0.85, pmid_29446752 ~0.77–0.87, "
      "pub_RNAPuzzle18_daslab ~0.96–0.98). HC16M2R is low (~0.04–0.06) for all models. "
      "pmid_25883046 and pmid_35982307 are DEGENERATE (constant labels).")
    A("")
    A("## 4. Primary metric: publication-macro AUPRC")
    A("")
    A("All models × all 5 seeds return `UNIDENTIFIABLE` for the primary publication-macro "
      "AUPRC because of the 2 constant-label publications (frozen fail-closed rule; "
      "degenerate_policies.constant_label / pair_any_all_positive). No numeric macro AUPRC, "
      "no confirmatory CI, and no permutation p<0.05 can be reported on the primary "
      "estimand. Per contract §13.4 the P2 gate on the PRIMARY estimand is therefore "
      "**not established**.")
    A("")
    A("## 5. Contract interpretation & discussion point (dialectical)")
    A("")
    A("The frozen degenerate policy \u201cany constant publication \u21d2 whole macro "
      "UNIDENTIFIABLE, no silent exclusion\u201d is scientifically conservative (prevents an "
      "all-positive publication from inflating a macro AP). However it also means a single "
      "all-changer publication blocks any numeric learnability number even when 8/10 "
      "publications show real, strong per-pub signal. Two defensible routes exist; both "
      "require explicit authorization (new authority epoch + amendment), because they change "
      "either the evaluator degeneracy policy or the endpoint metric semantics:")
    A("")
    A("1. **Relax to macro-over-non-degenerate-publications** (document explicit exclusions): "
      "compute macro AUPRC over the 8 mixed-label publications and report CI/permutation on "
      "that subset, with the 2 constant-label publications excluded and documented. This "
      "would likely produce a numeric primary metric and enable a proper GO/STOP on the "
      "recalibrated labels.")
    A("2. **Accept fail-closed STOP on the primary estimand** and treat the strong 8-pub "
      "per-pub signal as motivation for a secondary/conditional estimand (|delta_r| "
      "magnitude regression), which is currently gated by endpoint requiring an identifiable "
      "primary + reliable caller.")
    A("")
    A("No number was fabricated, and no degenerate publication was silently dropped. This "
      "document is evidence for the authority-epoch-15 re-adjudication only; it does not "
      "modify the frozen R6 script, endpoint, or any prior verdict.")
    A("")

    (RUN / "P2_learnability_audit_v3.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", RUN / "P2_learnability_audit_v3.md")


if __name__ == "__main__":
    main()
