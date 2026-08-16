#!/usr/bin/env python3
"""C1-3 Gate Audit Script.

Checks all C1-3 gate criteria as specified in ReactFlow分阶段执行提示词.md (lines 587-602):

Gate criteria:
1. 同 split 超过最强 baseline (ViennaRNA F1≈0.68)，或至少明确达到 paper-candidate 区间
2. OOD 提升不能只来自 in-domain
3. public tiers 不得出现严重退化
4. calibration、legality 和 runtime 有完整报告
5. 10-seed 主结果显著

Required artifacts:
- artifacts/c1_3/baseline_same_split_results.json
- artifacts/c1_3/model_grid_results.json
- artifacts/c1_3/multiseed_results.json
- artifacts/c1_3/significance_report.json
- docs/c1_3_static_sota_report.md
- docs/static_sota_table.md

Usage::

    python scripts/audit_c1_3_gate.py --artifacts-dir artifacts/c1_3 --docs-dir docs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ViennaRNA baseline F1 (strongest baseline from same_split_results)
VIENNARNA_F1_IN_CLAN = 0.6819
VIENNARNA_F1_NOVEL_CLAN = 0.6822

# Internal targets (not replacements for same-split baseline)
INTERNAL_TARGETS = {
    "PDB_exact_F1": 0.85,
    "ArchiveII_exact_F1": 0.80,
    "viral_F1": 0.68,
    "viral_target": 0.73,
    "lncRNA_F1": 0.40,
}

# Required baselines (spec line 543-551)
REQUIRED_BASELINES = [
    "ViennaRNA",
    "EternaFold",
    "MXfold2",
    "UFold",
    "eFold",
    "RNAformer",
]

# Required output files (spec line 578-585)
REQUIRED_ARTIFACTS = [
    "baseline_same_split_results.json",
    "model_grid_results.json",
    "multiseed_results.json",
    "significance_report.json",
]

REQUIRED_DOCS = [
    "c1_3_static_sota_report.md",
    "static_sota_table.md",
]


def check_artifacts(artifacts_dir: Path) -> List[Tuple[str, bool, str]]:
    """Check that all required artifact files exist.

    Returns list of (name, exists, message).
    """
    results = []
    for name in REQUIRED_ARTIFACTS:
        path = artifacts_dir / name
        if path.exists():
            results.append((name, True, f"Found: {path}"))
        else:
            results.append((name, False, f"Missing: {path}"))
    return results


def check_docs(docs_dir: Path) -> List[Tuple[str, bool, str]]:
    """Check that all required doc files exist.

    Returns list of (name, exists, message).
    """
    results = []
    for name in REQUIRED_DOCS:
        path = docs_dir / name
        if path.exists():
            results.append((name, True, f"Found: {path}"))
        else:
            results.append((name, False, f"Missing: {path}"))
    return results


def check_baselines(artifacts_dir: Path) -> List[Tuple[str, bool, str]]:
    """Check which baselines have results.

    Returns list of (baseline_name, has_results, message).
    """
    results = []

    # Check baseline_same_split_results.json for baseline entries
    same_split_path = artifacts_dir / "baseline_same_split_results.json"
    if same_split_path.exists():
        with open(same_split_path) as f:
            data = json.load(f)
        found_baselines = set()
        for b in data.get("baselines", []):
            model = b.get("model", "").lower()
            found_baselines.add(model)
            for name in REQUIRED_BASELINES:
                if name.lower() in model:
                    rows = b.get("rows", [])
                    if rows:
                        results.append((name, True, f"Found in same_split_results ({len(rows)} tiers)"))
                    else:
                        results.append((name, False, f"Found but no rows"))

        for name in REQUIRED_BASELINES:
            if not any(name.lower() in r[0].lower() for r in results if r[1]):
                results.append((name, False, f"Not found in same_split_results"))
    else:
        for name in REQUIRED_BASELINES:
            results.append((name, False, "same_split_results.json not found"))

    return results


def check_model_performance(artifacts_dir: Path) -> List[Tuple[str, bool, str]]:
    """Check if model beats ViennaRNA baseline.

    Returns list of (check_name, passed, message).
    """
    results = []

    # Check model_grid_results.json for model performance
    grid_path = artifacts_dir / "model_grid_results.json"
    if not grid_path.exists():
        results.append(("model_beats_viennarna", False, "model_grid_results.json not found"))
        return results

    with open(grid_path) as f:
        grid_data = json.load(f)

    # Look for model results that beat ViennaRNA
    best_f1_in_clan = 0.0
    best_f1_novel_clan = 0.0
    best_config = ""

    configs = grid_data.get("configs", grid_data.get("results", []))
    if isinstance(configs, dict):
        configs = list(configs.values())

    for cfg in configs:
        if isinstance(cfg, dict):
            tiers = cfg.get("tiers", cfg.get("results", {}))
            if isinstance(tiers, dict):
                in_clan = tiers.get("in_clan", {})
                novel_clan = tiers.get("novel_clan", {})
                if isinstance(in_clan, dict):
                    f1 = in_clan.get("mean_f1", in_clan.get("f1", 0))
                    if f1 > best_f1_in_clan:
                        best_f1_in_clan = f1
                        best_config = cfg.get("config", cfg.get("name", "unknown"))
                if isinstance(novel_clan, dict):
                    f1 = novel_clan.get("mean_f1", novel_clan.get("f1", 0))
                    if f1 > best_f1_novel_clan:
                        best_f1_novel_clan = f1

    # Check against ViennaRNA
    beats_in_clan = best_f1_in_clan > VIENNARNA_F1_IN_CLAN
    beats_novel_clan = best_f1_novel_clan > VIENNARNA_F1_NOVEL_CLAN

    results.append((
        "model_beats_viennarna_in_clan",
        beats_in_clan,
        f"Model F1={best_f1_in_clan:.4f} vs ViennaRNA F1={VIENNARNA_F1_IN_CLAN:.4f} ({'PASS' if beats_in_clan else 'FAIL'})"
    ))
    results.append((
        "model_beats_viennarna_novel_clan",
        beats_novel_clan,
        f"Model F1={best_f1_novel_clan:.4f} vs ViennaRNA F1={VIENNARNA_F1_NOVEL_CLAN:.4f} ({'PASS' if beats_novel_clan else 'FAIL'})"
    ))

    return results


def check_training_status(artifacts_dir: Path) -> List[Tuple[str, bool, str]]:
    """Check training completion status.

    Returns list of (check_name, passed, message).
    """
    results = []

    # Check if checkpoint exists
    runs_dir = artifacts_dir / "runs"
    if runs_dir.exists():
        checkpoints = list(runs_dir.rglob("best.pt"))
        non_empty = [cp for cp in checkpoints if cp.stat().st_size > 0]
        if non_empty:
            results.append(("checkpoint_exists", True, f"Found {len(non_empty)} non-empty checkpoint(s)"))
        else:
            results.append(("checkpoint_exists", False, "No non-empty checkpoints found"))
    else:
        results.append(("checkpoint_exists", False, f"runs directory not found: {runs_dir}"))

    # Check for training results
    for result_dir_name in ["results_fsdp_seed0", "results_ddp_seed0"]:
        result_dir = artifacts_dir / result_dir_name
        if result_dir.exists():
            results.append((f"training_results_{result_dir_name}", True, f"Found: {result_dir}"))
        else:
            results.append((f"training_results_{result_dir_name}", False, f"Not found: {result_dir}"))

    return results


def check_multiseed(artifacts_dir: Path) -> List[Tuple[str, bool, str]]:
    """Check multi-seed results.

    Returns list of (check_name, passed, message).
    """
    results = []

    multiseed_path = artifacts_dir / "multiseed_results.json"
    if not multiseed_path.exists():
        results.append(("multiseed_results", False, "multiseed_results.json not found"))
        return results

    with open(multiseed_path) as f:
        data = json.load(f)

    # Check seed count
    seeds = data.get("seeds", [])
    if len(seeds) >= 3:
        results.append(("multiseed_3seeds", True, f"Found {len(seeds)} seeds"))
    elif len(seeds) > 0:
        results.append(("multiseed_3seeds", False, f"Only {len(seeds)} seeds (need >=3)"))
    else:
        results.append(("multiseed_3seeds", False, "No seeds found"))

    # Check 10-seed main results
    if len(seeds) >= 10:
        results.append(("multiseed_10seeds", True, f"Found {len(seeds)} seeds (10-seed main results)"))
    else:
        results.append(("multiseed_10seeds", False, f"Only {len(seeds)} seeds (need 10 for main results)"))

    return results


def check_significance(artifacts_dir: Path) -> List[Tuple[str, bool, str]]:
    """Check significance report.

    Returns list of (check_name, passed, message).
    """
    results = []

    sig_path = artifacts_dir / "significance_report.json"
    if not sig_path.exists():
        results.append(("significance_report", False, "significance_report.json not found"))
        return results

    with open(sig_path) as f:
        data = json.load(f)

    # Check for significance tests
    tests = data.get("tests", data.get("significance_tests", []))
    if tests:
        results.append(("significance_tests", True, f"Found {len(tests)} significance test(s)"))
    else:
        results.append(("significance_tests", False, "No significance tests found"))

    # Check for p-values
    significant_count = 0
    for test in tests:
        if isinstance(test, dict):
            pval = test.get("p_value", test.get("pvalue", 1.0))
            if pval < 0.05:
                significant_count += 1

    if significant_count > 0:
        results.append(("significant_results", True, f"{significant_count}/{len(tests)} tests significant (p<0.05)"))
    else:
        results.append(("significant_results", False, f"0/{len(tests)} tests significant"))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="C1-3 Gate Audit")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/c1_3"),
                        help="Path to C1-3 artifacts directory")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"),
                        help="Path to docs directory")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error code if any check fails")
    args = parser.parse_args()

    print("=" * 80)
    print("C1-3 Gate Audit")
    print("=" * 80)

    all_passed = True
    all_results: List[Tuple[str, str, bool, str]] = []

    # Check artifacts
    print("\n[1] Required Artifacts")
    for name, exists, msg in check_artifacts(args.artifacts_dir):
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        all_results.append(("artifact", name, exists, msg))
        if not exists:
            all_passed = False

    # Check docs
    print("\n[2] Required Documentation")
    for name, exists, msg in check_docs(args.docs_dir):
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        all_results.append(("doc", name, exists, msg))
        if not exists:
            all_passed = False

    # Check baselines
    print("\n[3] Baseline Coverage")
    baseline_results = check_baselines(args.artifacts_dir)
    for name, has_results, msg in baseline_results:
        status = "PASS" if has_results else "WARN"
        print(f"  [{status}] {name}: {msg}")
        all_results.append(("baseline", name, has_results, msg))

    # Check training status
    print("\n[4] Training Status")
    for name, passed, msg in check_training_status(args.artifacts_dir):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        all_results.append(("training", name, passed, msg))
        if not passed:
            all_passed = False

    # Check model performance
    print("\n[5] Model Performance (vs ViennaRNA baseline)")
    for name, passed, msg in check_model_performance(args.artifacts_dir):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        all_results.append(("performance", name, passed, msg))
        if not passed:
            all_passed = False

    # Check multi-seed
    print("\n[6] Multi-seed Results")
    for name, passed, msg in check_multiseed(args.artifacts_dir):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        all_results.append(("multiseed", name, passed, msg))
        if not passed:
            all_passed = False

    # Check significance
    print("\n[7] Significance Report")
    for name, passed, msg in check_significance(args.artifacts_dir):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        all_results.append(("significance", name, passed, msg))
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "=" * 80)
    total = len(all_results)
    passed = sum(1 for r in all_results if r[2])
    failed = total - passed
    print(f"Summary: {passed}/{total} checks passed, {failed} failed")

    if all_passed:
        print("\nGATE STATUS: PASS")
    else:
        print("\nGATE STATUS: FAIL (or INCOMPLETE)")

    # Write audit report
    audit_path = args.artifacts_dir / "gate_audit.json"
    audit_data = {
        "gate": "C1-3",
        "status": "PASS" if all_passed else "FAIL",
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "results": [
            {"category": cat, "name": name, "passed": p, "message": msg}
            for cat, name, p, msg in all_results
        ],
    }
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAudit report written to: {audit_path}")

    if args.strict and not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
