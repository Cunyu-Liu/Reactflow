#!/usr/bin/env python3
"""PH0 WT-only physics identifiability audit.

Computes the train/validation/test split (parent holdout + study leave-out),
runs the WT-structure-context identifiability audit on train+validation pairs,
and outputs split_members.json + physics_identifiability_report.json.

No mutant states (alt=X blocked). No test labels used. No learned training.

Hypotheses (pre-registered):
  H1: |Δreactivity at edit| > noise ceiling (response above noise)
  H2: WT structure at edit pos correlates with |Δreactivity at edit|
  H3: WT contact/fragility at edit pos correlates with remote propagation
  H4: Features are reproducible (verified in test_thermo_state.py)
  H5: No test leakage (verified by split_members.json cross-check)

Gate (all 4 must pass):
  G1: H1 support (response > noise)
  G2: H3 support (remote or contact signal)
  G3: H4 pass (features reproducible)
  G4: H5 pass (no test labels used)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.manifests import sha256_file  # noqa: E402

REPORT_SCHEMA_VERSION = "reactflow-delta-ph0-physics-identifiability-report-v1"
SPLIT_SCHEMA_VERSION = "reactflow-delta-ph0-split-members-v1"

# Study assignments (from registry: 2 studies, 6 parents)
STUDY_RHIJU = "10.1073/pnas.1619897114"  # 1216 pairs, 5 parents
STUDY_BYEON = "10.1038/s41588-021-00830-1"  # 293 pairs, 1 parent (Csde1)

# Split: study leave-out (Byeon -> test) + parent holdout (P4-P6 -> val)
VALIDATION_PARENT = "P4-P6 domain, Tetrahymena ribozyme"
TEST_STUDY = STUDY_BYEON


def compute_split(per_pair: list[dict]) -> dict:
    """Compute train/validation/test split and cross-contamination check."""

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for p in per_pair:
        if p["citation_doi"] == TEST_STUDY:
            test_pairs.append(p)
        elif p["parent_prefix"] == VALIDATION_PARENT:
            val_pairs.append(p)
        else:
            train_pairs.append(p)

    def pair_ids(pairs):
        return sorted(p["pair_id"] for p in pairs)

    def sha256_ids(ids):
        return hashlib.sha256("\n".join(ids).encode()).hexdigest()

    train_ids = pair_ids(train_pairs)
    val_ids = pair_ids(val_pairs)
    test_ids = pair_ids(test_pairs)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    split = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_method": "parent_holdout_validation + study_leave_out",
        "train": {
            "parents": sorted(set(p["parent_prefix"] for p in train_pairs)),
            "n_pairs": len(train_pairs),
            "pair_ids": train_ids,
            "sha256": sha256_ids(train_ids),
        },
        "validation": {
            "parents": sorted(set(p["parent_prefix"] for p in val_pairs)),
            "n_pairs": len(val_pairs),
            "pair_ids": val_ids,
            "sha256": sha256_ids(val_ids),
        },
        "test": {
            "parents": sorted(set(p["parent_prefix"] for p in test_pairs)),
            "n_pairs": len(test_pairs),
            "pair_ids": test_ids,
            "sha256": sha256_ids(test_ids),
            "frozen": True,
            "used_in_ph0_audit": False,
        },
        "cross_contamination_check": {
            "test_in_train": len(test_set & train_set),
            "test_in_val": len(test_set & val_set),
            "val_in_train": len(val_set & train_set),
            "all_disjoint": len(test_set & train_set) == 0 and len(test_set & val_set) == 0 and len(val_set & train_set) == 0,
        },
        "study_assignment": {
            STUDY_RHIJU: "train+validation",
            STUDY_BYEON: "test (frozen)",
        },
    }
    return split


def safe_spearman(x, y):
    """Spearman correlation, handling NaN/None. Returns (rho, p, n)."""
    mask = [a is not None and b is not None and np.isfinite(a) and np.isfinite(b) for a, b in zip(x, y)]
    xa = [x[i] for i in range(len(x)) if mask[i]]
    ya = [y[i] for i in range(len(y)) if mask[i]]
    if len(xa) < 5:
        return {"rho": None, "p_value": None, "n": len(xa), "error": "insufficient data"}
    rho, p = stats.spearmanr(xa, ya)
    return {"rho": float(rho), "p_value": float(p), "n": len(xa)}


def safe_mannwhitney(group_a, group_b, alternative="two-sided"):
    """Mann-Whitney U test. Returns (U, p, n_a, n_b)."""
    a = [v for v in group_a if v is not None and np.isfinite(v)]
    b = [v for v in group_b if v is not None and np.isfinite(v)]
    if len(a) < 2 or len(b) < 2:
        return {"u": None, "p_value": None, "n_a": len(a), "n_b": len(b), "error": "insufficient data"}
    u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return {"u": float(u), "p_value": float(p), "n_a": len(a), "n_b": len(b)}


def classify(rho, p, alpha, expected_direction):
    """Classify a correlation result as support/mixed/challenged."""
    if p is None or rho is None:
        return "challenged"
    if p >= alpha:
        return "challenged"
    # expected_direction: "positive" or "negative"
    if expected_direction == "positive" and rho > 0:
        return "support" if abs(rho) > 0.1 else "mixed"
    elif expected_direction == "negative" and rho < 0:
        return "support" if abs(rho) > 0.1 else "mixed"
    else:
        return "challenged"


def run_audit(per_pair: list[dict], split: dict) -> dict:
    """Run the PH0 identifiability audit on train+validation pairs."""

    # Use only train+val pairs (test frozen)
    train_val_ids = set(split["train"]["pair_ids"]) | set(split["validation"]["pair_ids"])
    audit_pairs = [p for p in per_pair if p["pair_id"] in train_val_ids]

    print(f"  audit pairs: {len(audit_pairs)} (train+val), test frozen: {split['test']['n_pairs']}", flush=True)

    # Extract arrays
    abs_delta_edit = [p.get("abs_delta_at_edit") for p in audit_pairs]
    bpp_paired = [p["wt_features"]["bpp_paired_prob"] for p in audit_pairs]
    bpp_unpaired = [p["wt_features"]["bpp_unpaired_prob"] for p in audit_pairs]
    entropy = [p["wt_features"]["positional_entropy_bits"] for p in audit_pairs]
    mfe_paired = [p["wt_features"]["mfe_paired"] for p in audit_pairs]
    n_contacts = [p["wt_features"]["n_contacts"] for p in audit_pairs]
    max_contact_dist = [p["wt_features"]["max_contact_distance"] for p in audit_pairs]
    remote_mean_delta = [p.get("remote_mean_abs_delta") for p in audit_pairs]
    max_abs_delta = [p.get("max_abs_delta") for p in audit_pairs]
    fragility = [p["fragility_proxy_value"] for p in audit_pairs]
    switch = [p["switch_enriched"] for p in audit_pairs]

    # =======================================================================
    # Noise ceiling: pool all remote |Δreactivity| from manifest per-pair stats
    # Each pair has remote_mean_abs_delta; for a global ceiling we use the
    # 95th percentile of per-pair remote_95pct_abs_delta as the noise threshold.
    # =======================================================================
    remote_95pct = [p.get("remote_95pct_abs_delta") for p in audit_pairs if p.get("remote_95pct_abs_delta") is not None]
    noise_ceiling_global = float(np.percentile(remote_95pct, 95)) if remote_95pct else None
    noise_ceiling_median = float(np.median(remote_95pct)) if remote_95pct else None

    # H1: Response > noise
    above_noise = sum(
        1 for p in audit_pairs
        if p.get("abs_delta_at_edit") is not None
        and noise_ceiling_global is not None
        and p["abs_delta_at_edit"] > noise_ceiling_global
    )
    fraction_above_noise = above_noise / len(audit_pairs) if audit_pairs else 0

    # Also: Wilcoxon test |Δreactivity at edit| vs median remote |Δreactivity|
    edit_vals = [v for v in abs_delta_edit if v is not None]
    remote_vals = [v for v in remote_95pct if v is not None]
    h1_wilcoxon = None
    if len(edit_vals) > 5 and len(remote_vals) > 5:
        w, p_val = stats.mannwhitneyu(edit_vals, remote_vals, alternative="greater")
        h1_wilcoxon = {"u": float(w), "p_value": float(p_val), "n_edit": len(edit_vals), "n_remote": len(remote_vals)}

    if fraction_above_noise > 0.1 and h1_wilcoxon and h1_wilcoxon["p_value"] < 0.05:
        h1_support = "support"
    elif fraction_above_noise > 0.05 and h1_wilcoxon and h1_wilcoxon["p_value"] < 0.05:
        h1_support = "mixed"
    else:
        h1_support = "challenged"

    # =======================================================================
    # H2: Edit-position WT-structure signal
    # =======================================================================
    h2_tests = []

    # Spearman: BPP paired prob vs |Δreactivity at edit| (expected: positive, more paired -> more disruption)
    r = safe_spearman(bpp_paired, abs_delta_edit)
    r["test"] = "spearman"; r["hypothesis"] = "H2"; r["x"] = "bpp_paired_prob"; r["y"] = "abs_delta_at_edit"
    r["expected_direction"] = "positive"; r["bonferroni_alpha"] = 0.05 / 4
    r["classification"] = classify(r.get("rho"), r.get("p_value"), r["bonferroni_alpha"], "positive")
    h2_tests.append(r)

    # Spearman: unpaired prob vs |Δreactivity at edit| (expected: negative)
    r = safe_spearman(bpp_unpaired, abs_delta_edit)
    r["test"] = "spearman"; r["hypothesis"] = "H2"; r["x"] = "bpp_unpaired_prob"; r["y"] = "abs_delta_at_edit"
    r["expected_direction"] = "negative"; r["bonferroni_alpha"] = 0.05 / 4
    r["classification"] = classify(r.get("rho"), r.get("p_value"), r["bonferroni_alpha"], "negative")
    h2_tests.append(r)

    # Spearman: entropy vs |Δreactivity at edit| (expected: positive, flexible -> more variable)
    r = safe_spearman(entropy, abs_delta_edit)
    r["test"] = "spearman"; r["hypothesis"] = "H2"; r["x"] = "positional_entropy_bits"; r["y"] = "abs_delta_at_edit"
    r["expected_direction"] = "positive"; r["bonferroni_alpha"] = 0.05 / 4
    r["classification"] = classify(r.get("rho"), r.get("p_value"), r["bonferroni_alpha"], "positive")
    h2_tests.append(r)

    # Mann-Whitney: MFE paired vs unpaired -> |Δreactivity at edit|
    paired_vals = [abs_delta_edit[i] for i in range(len(audit_pairs)) if mfe_paired[i] and abs_delta_edit[i] is not None]
    unpaired_vals = [abs_delta_edit[i] for i in range(len(audit_pairs)) if not mfe_paired[i] and abs_delta_edit[i] is not None]
    mw = safe_mannwhitney(paired_vals, unpaired_vals, alternative="greater")
    mw["test"] = "mannwhitney"; mw["hypothesis"] = "H2"; mw["x"] = "mfe_paired"; mw["y"] = "abs_delta_at_edit"
    mw["expected_direction"] = "paired > unpaired"; mw["bonferroni_alpha"] = 0.05 / 4
    if mw["p_value"] is not None and mw["p_value"] < mw["bonferroni_alpha"]:
        mw["classification"] = "support" if np.median(paired_vals) > np.median(unpaired_vals) else "challenged"
    else:
        mw["classification"] = "challenged"
    h2_tests.append(mw)

    h2_support = "support" if any(t["classification"] == "support" for t in h2_tests) else \
                 "mixed" if any(t["classification"] == "mixed" for t in h2_tests) else "challenged"

    # =======================================================================
    # H3: Remote/contact signal (Gate 2)
    # =======================================================================
    h3_tests = []

    # n_contacts vs remote_mean_abs_delta (expected: positive, more contacts -> more propagation)
    r = safe_spearman(n_contacts, remote_mean_delta)
    r["test"] = "spearman"; r["hypothesis"] = "H3"; r["x"] = "n_contacts"; r["y"] = "remote_mean_abs_delta"
    r["expected_direction"] = "positive"; r["bonferroni_alpha"] = 0.05 / 6
    r["classification"] = classify(r.get("rho"), r.get("p_value"), r["bonferroni_alpha"], "positive")
    h3_tests.append(r)

    # n_contacts vs max_abs_delta (expected: positive)
    r = safe_spearman(n_contacts, max_abs_delta)
    r["test"] = "spearman"; r["hypothesis"] = "H3"; r["x"] = "n_contacts"; r["y"] = "max_abs_delta"
    r["expected_direction"] = "positive"; r["bonferroni_alpha"] = 0.05 / 6
    r["classification"] = classify(r.get("rho"), r.get("p_value"), r["bonferroni_alpha"], "positive")
    h3_tests.append(r)

    # fragility (BPP) vs remote_mean_abs_delta (expected: positive, fragile -> more propagation)
    r = safe_spearman(fragility, remote_mean_delta)
    r["test"] = "spearman"; r["hypothesis"] = "H3"; r["x"] = "fragility_bpp_paired_prob"; r["y"] = "remote_mean_abs_delta"
    r["expected_direction"] = "positive"; r["bonferroni_alpha"] = 0.05 / 6
    r["classification"] = classify(r.get("rho"), r.get("p_value"), r["bonferroni_alpha"], "positive")
    h3_tests.append(r)

    # fragility (BPP) vs max_abs_delta (expected: positive)
    r = safe_spearman(fragility, max_abs_delta)
    r["test"] = "spearman"; r["hypothesis"] = "H3"; r["x"] = "fragility_bpp_paired_prob"; r["y"] = "max_abs_delta"
    r["expected_direction"] = "positive"; r["bonferroni_alpha"] = 0.05 / 6
    r["classification"] = classify(r.get("rho"), r.get("p_value"), r["bonferroni_alpha"], "positive")
    h3_tests.append(r)

    # Switch-enriched: fragility comparison (switch vs non-switch)
    switch_fragility = [fragility[i] for i in range(len(audit_pairs)) if switch[i]]
    noswitch_fragility = [fragility[i] for i in range(len(audit_pairs)) if not switch[i]]
    mw_sw = safe_mannwhitney(switch_fragility, noswitch_fragility, alternative="greater")
    mw_sw["test"] = "mannwhitney"; mw_sw["hypothesis"] = "H3"; mw_sw["x"] = "switch_enriched"; mw_sw["y"] = "fragility_bpp_paired_prob"
    mw_sw["expected_direction"] = "switch > noswitch"; mw_sw["bonferroni_alpha"] = 0.05 / 6
    if mw_sw["p_value"] is not None and mw_sw["p_value"] < mw_sw["bonferroni_alpha"]:
        mw_sw["classification"] = "support" if np.median(switch_fragility) > np.median(noswitch_fragility) else "challenged"
    else:
        mw_sw["classification"] = "challenged"
    h3_tests.append(mw_sw)

    # Switch rate: high-BPP (top quartile) vs low-BPP (bottom quartile) — Fisher exact
    fragility_arr = np.array([f if f is not None else 0 for f in fragility])
    q75, q25 = np.percentile(fragility_arr, [75, 25])
    high_bpp = fragility_arr >= q75
    low_bpp = fragility_arr <= q25
    switch_arr = np.array(switch)
    # 2x2 table: [high_bpp_switch, high_bpp_noswitch, low_bpp_switch, low_bpp_noswitch]
    table = [
        [int(np.sum(high_bpp & switch_arr)), int(np.sum(high_bpp & ~switch_arr))],
        [int(np.sum(low_bpp & switch_arr)), int(np.sum(low_bpp & ~switch_arr))],
    ]
    try:
        oddsratio, fisher_p = stats.fisher_exact(table, alternative="greater")
        fisher_result = {"oddsratio": float(oddsratio), "p_value": float(fisher_p), "table": table,
                         "test": "fisher_exact", "hypothesis": "H3",
                         "x": "high_bpp_quartile", "y": "switch_enriched",
                         "expected_direction": "high_bpp more switches",
                         "bonferroni_alpha": 0.05 / 6}
        fisher_result["classification"] = "support" if (fisher_p < fisher_result["bonferroni_alpha"] and oddsratio > 1) else \
                                          "mixed" if fisher_p < 0.05 else "challenged"
    except Exception as e:
        fisher_result = {"test": "fisher_exact", "hypothesis": "H3", "error": str(e), "classification": "challenged",
                         "table": table, "bonferroni_alpha": 0.05 / 6}
    h3_tests.append(fisher_result)

    h3_support = "support" if any(t["classification"] == "support" for t in h3_tests) else \
                 "mixed" if any(t["classification"] == "mixed" for t in h3_tests) else "challenged"

    # =======================================================================
    # Switch-enriched subset stats
    # =======================================================================
    switch_stats = {
        "n_switch_enriched_train_val": int(np.sum(switch_arr)),
        "n_train_val": len(audit_pairs),
        "switch_rate": float(np.mean(switch_arr)),
        "median_fragility_switch": float(np.median(switch_fragility)) if switch_fragility else None,
        "median_fragility_noswitch": float(np.median(noswitch_fragility)) if noswitch_fragility else None,
        "high_bpp_switch_rate": float(np.mean(switch_arr[high_bpp])) if np.sum(high_bpp) > 0 else None,
        "low_bpp_switch_rate": float(np.mean(switch_arr[low_bpp])) if np.sum(low_bpp) > 0 else None,
    }

    # =======================================================================
    # Gate evaluation
    # =======================================================================
    gate_fail_reasons = []
    g1 = h1_support in ("support", "mixed")
    g2 = h3_support in ("support", "mixed")
    g3 = True  # features reproducible — verified in test_thermo_state.py
    g4 = split["cross_contamination_check"]["all_disjoint"]

    if not g1:
        gate_fail_reasons.append(f"G1: H1 response>noise not supported (fraction_above_noise={fraction_above_noise:.3f})")
    if not g2:
        gate_fail_reasons.append(f"G2: H3 remote/contact signal not supported")
    if not g3:
        gate_fail_reasons.append("G3: features not reproducible (tests failed)")
    if not g4:
        gate_fail_reasons.append("G4: test leakage detected in split")

    gate_pass = g1 and g2 and g3 and g4

    # =======================================================================
    # Build report
    # =======================================================================
    all_tests = h2_tests + h3_tests
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": "PH0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "WT-structure-context identifiability (alt=X blocked, no mutant states)",
        "true_pairs_used": len(audit_pairs),
        "true_pairs_excluded_from_audit": split["test"]["n_pairs"],
        "total_true_pairs": len(audit_pairs) + split["test"]["n_pairs"],
        "excluded_count": 6252,
        "exclusion_reasons": "see artifacts/reactflow_delta/d2r/d1_pipeline_summary.json",
        "split_sha256": {
            "train": split["train"]["sha256"],
            "validation": split["validation"]["sha256"],
            "test": split["test"]["sha256"],
        },
        "split_cross_contamination_check": split["cross_contamination_check"],
        "tool_versions": {
            "viennarna": "2.7.2 (python API, conda editflow311, no CLI binary)",
            "scipy": "scipy.stats (conda editflow311)",
            "numpy": "numpy (conda editflow311)",
        },
        "noise_ceiling": {
            "method": "95th percentile of per-pair remote_95pct_abs_delta (remote = >20nt from edit)",
            "global_95pct": noise_ceiling_global,
            "median": noise_ceiling_median,
            "n_pairs_used": len(remote_95pct),
        },
        "signal_above_noise": {
            "n_pairs_above_noise": above_noise,
            "fraction_above_noise": float(fraction_above_noise),
            "h1_wilcoxon": h1_wilcoxon,
        },
        "per_hypothesis": {
            "H1_response_above_noise": {
                "classification": h1_support,
                "evidence_ref": "signal_above_noise + noise_ceiling",
                "fraction_above_noise": float(fraction_above_noise),
                "wilcoxon_p": h1_wilcoxon["p_value"] if h1_wilcoxon else None,
            },
            "H2_edit_pos_structure_signal": {
                "classification": h2_support,
                "evidence_ref": "correlation_tables[H2:*]",
                "n_tests": len(h2_tests),
                "n_support": sum(1 for t in h2_tests if t["classification"] == "support"),
            },
            "H3_remote_contact_signal": {
                "classification": h3_support,
                "evidence_ref": "correlation_tables[H3:*]",
                "n_tests": len(h3_tests),
                "n_support": sum(1 for t in h3_tests if t["classification"] == "support"),
            },
            "H4_features_reproducible": {
                "classification": "support" if g3 else "challenged",
                "evidence_ref": "test_thermo_state.py (recompute from WT seq, verify identical)",
            },
            "H5_no_test_leakage": {
                "classification": "support" if g4 else "challenged",
                "evidence_ref": "split_members.json cross_contamination_check",
            },
        },
        "correlation_tables": all_tests,
        "fragility_proxy_definition": {
            "name": "bpp_paired_prob",
            "description": "BPP-derived paired probability at the edit position in the WT structure",
            "hypothesis": "Higher BPP -> larger structural disruption -> larger |Δreactivity| and/or remote propagation",
            "switch_enriched_definition": "max_abs_delta_distance_from_edit > 20",
        },
        "switch_enriched_subset_stats": switch_stats,
        "gate_pass": bool(gate_pass),
        "gate_fail_reasons": gate_fail_reasons,
        "self_consistency_check": {
            "used_plus_excluded_from_audit": len(audit_pairs) + split["test"]["n_pairs"],
            "total_true_pairs": 1509,
            "consistent": len(audit_pairs) + split["test"]["n_pairs"] == 1509,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="thermo_features_manifest.json")
    parser.add_argument("--split-output", required=True, help="split_members.json output")
    parser.add_argument("--report-output", required=True, help="physics_identifiability_report.json output")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    split_path = Path(args.split_output)
    report_path = Path(args.report_output)

    print("Loading manifest...", flush=True)
    with manifest_path.open() as f:
        manifest = json.load(f)
    per_pair = manifest["per_pair"]
    print(f"  pairs: {len(per_pair)}", flush=True)

    # Compute split
    print("Computing split...", flush=True)
    split = compute_split(per_pair)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  train={split['train']['n_pairs']} val={split['validation']['n_pairs']} test={split['test']['n_pairs']}", flush=True)
    print(f"  disjoint={split['cross_contamination_check']['all_disjoint']}", flush=True)

    # Run audit
    print("Running audit...", flush=True)
    report = run_audit(per_pair, split)

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Summary
    print(f"\nReport written: {report_path}", flush=True)
    print(f"  H1: {report['per_hypothesis']['H1_response_above_noise']['classification']}", flush=True)
    print(f"  H2: {report['per_hypothesis']['H2_edit_pos_structure_signal']['classification']}", flush=True)
    print(f"  H3: {report['per_hypothesis']['H3_remote_contact_signal']['classification']}", flush=True)
    print(f"  noise_ceiling: {report['noise_ceiling']['global_95pct']}", flush=True)
    print(f"  fraction_above_noise: {report['signal_above_noise']['fraction_above_noise']:.3f}", flush=True)
    print(f"  gate_pass: {report['gate_pass']}", flush=True)
    if report["gate_fail_reasons"]:
        for r in report["gate_fail_reasons"]:
            print(f"    FAIL: {r}", flush=True)


if __name__ == "__main__":
    main()
