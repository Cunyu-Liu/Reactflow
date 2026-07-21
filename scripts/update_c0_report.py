#!/usr/bin/env python3
"""Update c0_correctness_report.md with actual evaluation results.

Reads:
  - c0_artifacts/final_evaluation/metrics.json
  - c0_artifacts/acceptance_decision.json
  - c0_artifacts/final_decoder_manifest.json
  - c0_artifacts/baseline_efold_results.json

Updates sections 6.2, 8.1, 9, and artifact index in docs/c0_correctness_report.md.
"""
import json
import re
import sys
from pathlib import Path

STAGE = Path("/home/cunyuliu/reactflow_c0_stage_20260718")
REPORT = STAGE / "docs" / "c0_correctness_report.md"
METRICS_PATH = STAGE / "c0_artifacts" / "final_evaluation" / "metrics.json"
DECISION_PATH = STAGE / "c0_artifacts" / "acceptance_decision.json"
MANIFEST_PATH = STAGE / "c0_artifacts" / "final_decoder_manifest.json"
BASELINE_PATH = STAGE / "c0_artifacts" / "baseline_efold_results.json"

def load_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def fmt(x, digits=6):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)

def main():
    metrics = load_json(METRICS_PATH)
    decision = load_json(DECISION_PATH)
    manifest = load_json(MANIFEST_PATH)
    baseline = load_json(BASELINE_PATH)

    if metrics is None:
        print("ERROR: metrics.json not found", file=sys.stderr)
        sys.exit(1)
    if decision is None:
        print("ERROR: acceptance_decision.json not found", file=sys.stderr)
        sys.exit(1)

    results = metrics.get("results", {})

    def get_f1(tier, mode):
        key = f"{tier}:{mode}"
        r = results.get(key, {})
        return r.get("mean_exact_f1")

    legacy_test_f1 = get_f1("mmseqs_component_test", "legacy_direct")
    legacy_holdout_f1 = get_f1("mmseqs_component_holdout", "legacy_direct")
    ctmc_test_f1 = get_f1("mmseqs_component_test", "ctmc_sample")
    ctmc_holdout_f1 = get_f1("mmseqs_component_holdout", "ctmc_sample")
    cal_test_f1 = get_f1("mmseqs_component_test", "calibrated_marginal")
    cal_holdout_f1 = get_f1("mmseqs_component_holdout", "calibrated_marginal")
    pdb_cal_f1 = get_f1("PDB", "calibrated_marginal")

    cal_mean_f1 = None
    if cal_test_f1 is not None and cal_holdout_f1 is not None:
        cal_mean_f1 = (cal_test_f1 + cal_holdout_f1) / 2

    baseline_test_f1 = None
    baseline_holdout_f1 = None
    if baseline:
        tiers_data = baseline.get("tiers", {})
        # tiers may be a dict {tier_name: {mean_f1, ...}} or a list of dicts
        if isinstance(tiers_data, dict):
            for tier_name, tier_info in tiers_data.items():
                if tier_name == "mmseqs_component_test":
                    baseline_test_f1 = tier_info.get("mean_f1")
                elif tier_name == "mmseqs_component_holdout":
                    baseline_holdout_f1 = tier_info.get("mean_f1")
        else:
            for tier_data in tiers_data:
                if tier_data.get("tier") == "mmseqs_component_test":
                    baseline_test_f1 = tier_data.get("mean_f1")
                elif tier_data.get("tier") == "mmseqs_component_holdout":
                    baseline_holdout_f1 = tier_data.get("mean_f1")

    selected_inference = manifest.get("selected_inference", {}) if manifest else {}
    selected_decoder = manifest.get("selected_decoder", {}) if manifest else {}

    text = REPORT.read_text(encoding="utf-8")

    # --- Update Section 6.2 ---
    old_62 = """### 6.2 Fixed matrix execution

- The full `calibrate-inference` run (128 coarse + 512 decoder calibration, 9 CTMC configs × 30 decoder configs) is launched in the background (PID 2544995, GPU 4) and writes `c0_artifacts/final_decoder_manifest.json`.
- The subsequent `evaluate-checkpoint` run (1,000 test + 1,000 holdout + 333 PDB) consumes the locked manifest and writes per-tier metrics + probing metrics.
- Both runs use the same `seed=20260718` and the same locked checkpoint SHA-256."""

    new_62 = f"""### 6.2 Fixed matrix execution

- `calibrate-inference` completed at 2026-07-20 11:59:32 (wall time ~23h34m, PID 2544995, GPU 4).
- `final_decoder_manifest.json` (94,244 bytes) records the selected CTMC config (steps={selected_inference.get('num_steps')}, samples={selected_inference.get('num_samples')}) and decoder config (temperature={selected_decoder.get('temperature')}, threshold={selected_decoder.get('threshold')}, policy={selected_decoder.get('matching_policy')}).
- `evaluate-checkpoint` consumed the locked manifest and evaluated 3 inference modes × 3 tiers (test 1,000 + holdout 1,000 + PDB 333).
- Both runs used `seed=20260718` and the same locked checkpoint SHA-256.
- **Runtime gate**: total pipeline exceeded the 24h gate (calibrate ~23.6h + evaluate ~9h = ~32.6h). See `c0_artifacts/runtime_bottleneck_report.md` for the bottleneck analysis. Sample counts were NOT shrunk.
- Validation CTMC coarse-grid mean exact F1: {selected_inference.get('selection_metrics', {}).get('mean_exact_f1', 'N/A'):.6f} (pair_count_ratio={selected_inference.get('selection_metrics', {}).get('pair_count_ratio', 'N/A'):.6f})."""

    text = text.replace(old_62, new_62)

    # --- Update Section 8.1 ---
    old_81 = "1. **SOTA**. The eFold baseline mean F1 is 0.220 (test) / 0.212 (holdout); the selected RF-CF3 checkpoint's validation exact F1 is 0.030. The CTMC/calibrated decoder is designed to recover legal structure under a partner-class DFM, not to beat eFold on full-count Rfam test sets. Any SOTA claim requires the corrected inference to clear the acceptance gate in Section 9."

    new_81 = f"""1. **SOTA**. The eFold baseline mean F1 is {fmt(baseline_test_f1, 3)} (test) / {fmt(baseline_holdout_f1, 3)} (holdout). The calibrated_marginal inference achieved mean exact F1 of {fmt(cal_test_f1)} (test) / {fmt(cal_holdout_f1)} (holdout), which is {'ABOVE' if (cal_holdout_f1 or 0) > (baseline_holdout_f1 or 0) else 'BELOW'} the baseline. The acceptance gate in Section 9 determines the next phase."""

    text = text.replace(old_81, new_81)

    # --- Update Section 9 ---
    old_9 = """The acceptance check runs after `evaluate-checkpoint` finishes and writes `c0_artifacts/final_evaluation_metrics.json`. The decision is recorded in `c0_artifacts/acceptance_decision.json` together with the exact holdout F1 delta and the selected next-goal label. Until that file exists, the acceptance gate is **pending**."""

    meets_gate = decision.get("meets_acceptance_gate", False)
    next_goal = decision.get("next_goal", "unknown")
    delta = decision.get("delta_vs_legacy")
    cal_mean = decision.get("calibrated_marginal_mean_f1")

    new_9 = f"""The acceptance check ran after `evaluate-checkpoint` finished. The decision is recorded in `c0_artifacts/acceptance_decision.json`.

**Result**: {'GATE PASSED' if meets_gate else 'GATE NOT PASSED'}.

| Metric | Value |
|---|---|
| baseline_holdout_f1 (eFold) | {fmt(decision.get('baseline_holdout_f1'), 6)} |
| legacy_direct holdout F1 | {fmt(decision.get('legacy_direct_holdout_f1'), 6)} |
| calibrated_marginal holdout F1 | {fmt(decision.get('calibrated_marginal_holdout_f1'), 6)} |
| calibrated_marginal test F1 | {fmt(decision.get('calibrated_marginal_test_f1'), 6)} |
| calibrated_marginal mean F1 (test+holdout)/2 | {fmt(cal_mean, 6)} |
| delta_vs_legacy (holdout) | {fmt(delta, 6)} |
| meets_improvement_gate (>=0.03) | {decision.get('meets_improvement_gate')} |
| meets_mean_f1_gate (>=0.10) | {decision.get('meets_mean_f1_gate')} |
| meets_acceptance_gate | {meets_gate} |

**Next goal**: {next_goal}

If the split/leakage audit cannot prove clan-level disjointness (which is the case here, see Section 7.1), the OOD wording in any downstream artifact is **permanently downgraded** to MMseqs-component semantics, regardless of the F1 outcome."""

    text = text.replace(old_9, new_9)

    # --- Update artifact index ---
    text = text.replace(
        "| `c0_artifacts/final_decoder_manifest.json` | (pending) full calibrate-inference output |",
        f"| `c0_artifacts/final_decoder_manifest.json` | Full calibrate-inference output (94,244 bytes, selected CTMC + decoder config) |"
    )
    text = text.replace(
        "| `c0_artifacts/final_evaluation_metrics.json` | (pending) full evaluate-checkpoint output |",
        "| `c0_artifacts/final_evaluation/metrics.json` | Full evaluate-checkpoint output (per-tier × per-mode aggregated metrics) |"
    )
    text = text.replace(
        "| `c0_artifacts/acceptance_decision.json` | (pending) Section 9 decision record |",
        "| `c0_artifacts/acceptance_decision.json` | Section 9 decision record (gate result + next goal) |"
    )
    text = text.replace(
        "| `c0_artifacts/runtime_preflight_32plus.json` | 33-sample runtime preflight, 5.59 h projected |",
        "| `c0_artifacts/runtime_preflight_32plus.json` | 33-sample runtime preflight, 5.59 h projected (evaluate-checkpoint only) |\n| `c0_artifacts/runtime_bottleneck_report.md` | Runtime bottleneck report (total pipeline exceeds 24h gate) |"
    )

    REPORT.write_text(text, encoding="utf-8")
    print(f"Report updated: {REPORT}")
    print(f"  Section 6.2: updated with actual calibrate/evaluate runtime")
    print(f"  Section 8.1: updated with actual baseline vs calibrated F1")
    print(f"  Section 9: updated with acceptance decision (gate {'PASSED' if meets_gate else 'NOT PASSED'})")
    print(f"  Artifact index: updated with actual file references")
    print(f"  Next goal: {next_goal}")

if __name__ == "__main__":
    main()
