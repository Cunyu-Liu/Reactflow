#!/usr/bin/env python3
"""D0-R: build functional-anchor acceptance (v2) and feasibility report.

Reads the functional-anchor audit artifacts (summary, M2SL5 relations,
COVSL5/SL5CV2 relations), hashes the raw RDAT inputs, runs the D0-R test
suite, and produces an INDEPENDENT acceptance that does NOT overwrite the
original D0 acceptance nor the previous D0-R acceptance (d0r_acceptance.json).

Outputs (in --output-dir):
  - d0r_acceptance_v2.json            (NEW, does not overwrite d0r_acceptance.json)
  - d0r_data_feasibility_audit.md     (rewritten with functional-anchor findings)

Stage permissions are intentionally conservative:
  d1_allowed       = False  (D0-R is audit-only; D1 execution not yet started)
  training_allowed = False  (learned training requires D2 Tier B approval)
  triage_decision  = "non_zero_candidate_pairs_authorize_d1" (recommendation)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "reactflow-delta-d0r-functional-acceptance-v2"

# Expected candidate counts from the D0-R handoff (192/probe * 2 probes).
EXPECTED_PER_PROBE = 192
EXPECTED_TOTAL = 384


def _sha256(path: Path) -> dict[str, Any]:
    """Hash a file in chunks and return path/sha256/size metadata."""
    import hashlib

    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": None,
    }
    if info["exists"]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        info["sha256"] = digest.hexdigest()
    return info


def _run_tests(test_dir: Path, python: str) -> dict[str, Any]:
    """Run the D0-R pytest suite and capture the summary line + counts."""
    cmd = [
        python, "-m", "pytest", str(test_dir),
        "--color=no", "--tb=short", "-p", "no:cacheprovider",
    ]
    env_pythonpath = "src"
    proc = subprocess.run(
        cmd,
        cwd=test_dir.parent.parent,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": env_pythonpath},
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = (stdout + "\n" + stderr).strip()

    passed = failed = errors = skipped = 0
    m = re.search(r"(\d+)\s+passed", combined)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", combined)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+)\s+error", combined)
    if m:
        errors = int(m.group(1))
    m = re.search(r"(\d+)\s+skipped", combined)
    if m:
        skipped = int(m.group(1))

    summary_line = ""
    for line in combined.splitlines():
        if re.search(r"\b(passed|failed|error|skipped|no tests ran)\b", line):
            summary_line = line.strip()
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "summary_line": summary_line,
        "all_passed": proc.returncode == 0 and failed == 0 and errors == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="d0r_functional_anchor_summary.json")
    parser.add_argument("--m2sl5-relations", required=True)
    parser.add_argument("--covsl5-relations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-rdat", nargs="*", default=[], help="Raw RDAT files to hash")
    parser.add_argument("--test-dir", default="tests/reactflow_delta")
    parser.add_argument("--python", default=sys.executable, help="Interpreter for pytest")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.summary).open() as f:
        summary = json.load(f)
    with Path(args.m2sl5_relations).open() as f:
        m2sl5_relations = json.load(f)
    with Path(args.covsl5_relations).open() as f:
        covsl5_relations = json.load(f)

    input_hashes = [_sha256(Path(p)) for p in args.input_rdat]

    if args.skip_tests:
        test_results = {"command": "skipped", "exit_code": None, "all_passed": None,
                        "note": "tests skipped via --skip-tests"}
    else:
        test_results = _run_tests(Path(args.test_dir), args.python)

    m2sl5_block = summary["m2sl5"]
    covsl5_block = summary["covsl5_sl5cv2"]
    anchor_block = summary["anchor_verification"]

    actual_total = m2sl5_block["total_candidates"]
    matches_expected = m2sl5_block["matches_expected"]

    per_probe: dict[str, int] = {}
    exclusion_breakdown: dict[str, dict[str, int]] = {}
    for fname, fstats in m2sl5_block["per_file"].items():
        per_probe[fname] = fstats["classification_counts"].get(
            "candidate_single_functional_anchor", 0
        )
        exclusion_breakdown[fname] = dict(fstats.get("exclusion_reason_counts", {}))

    covsl5_total = covsl5_block["total_candidates"]

    discrepancy_explanation = (
        "Expected 384 (192/probe x 2 probes), likely assuming a 64 nt sub-region "
        "(64 positions x 3 mutations = 192). Actual is 744 (372/probe x 2 probes) = "
        "full saturation of the 124 nt COVSL5 functional anchor "
        "(124 positions x 3 mutations = 372). All 744 candidates are SARS_CoV_2 "
        "(the WT anchor species), every functional position 0-123 is covered with "
        "exactly 3 mutations, and all 744 have functional Hamming == 1 AND matching "
        "pos/ref/alt (0 name_sequence_mismatch exclusions). The handoff explicitly "
        "states expected numbers must NOT be forced; actuals are reported honestly."
    )

    key_findings = [
        "D0-R functional-anchor audit uses a 124 nt window at offset 31 within the "
        "206 nt full anchor (SL5CV2_NOM_0002), stricter than the prior SEQPOS approach.",
        f"Functional anchor verified: 124 nt from COVSL5_NOM_0002 occurs exactly once at "
        f"offset {anchor_block['functional_anchor_verification']['offset']} in the 206 nt "
        f"full anchor (valid={anchor_block['functional_anchor_verification']['valid']}).",
        f"WT anchor (profile 1, SARS_CoV_2) identified by exact 206 nt sequence match in "
        f"both M2SL5 files.",
        f"M2SL5 produced {actual_total} candidate single-mutant relations "
        f"({per_probe.get('M2SL5_2A3_0000', 0)} 2A3 + {per_probe.get('M2SL5_DMS_0000', 0)} DMS), "
        f"vs {EXPECTED_TOTAL} expected. {discrepancy_explanation}",
        f"COVSL5/SL5CV2 files produced {covsl5_total} candidates (their mutation annotations "
        f"use a different format, e.g. G159C, not name-encoded <pos><ref>-<alt>; SL5CV2 files "
        f"lack the WT anchor sequence).",
        "Species partitioning per probe (2499 profiles): SARS_CoV_2=798 (1 WT + 372 single-mut "
        "+ 425 double-mut), MERS=888, BtCoV=813. MERS/BtCoV excluded (functional Hamming >> 1).",
        "Previous SEQPOS-based result (m2sl5_candidate_relations.json, 744 candidates) is "
        "preserved as historical evidence and NOT overwritten.",
        "All candidate relations are candidate_only_pending_parent_lineage_and_functional_"
        "region_validation with true_pair=False. No pair, tier, or model claim is made.",
    ]

    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "original_d0_acceptance_preserved": True,
        "original_d0_acceptance_path": "artifacts/reactflow_delta/d0/d0_acceptance_a79fd073.json",
        "previous_d0r_acceptance_preserved": True,
        "previous_d0r_acceptance_path": "artifacts/reactflow_delta/d0r/d0r_acceptance.json",
        "audit_method": "functional_anchor_124nt_window_offset_31",
        "inputs": {
            "functional_anchor_summary": str(Path(args.summary).resolve()),
            "m2sl5_functional_relations": str(Path(args.m2sl5_relations).resolve()),
            "covsl5_sl5cv2_relations": str(Path(args.covsl5_relations).resolve()),
            "input_rdat_hashes": input_hashes,
        },
        "anchor_verification": anchor_block,
        "candidate_counts": {
            "m2sl5_total": actual_total,
            "m2sl5_per_probe": per_probe,
            "covsl5_sl5cv2_total": covsl5_total,
            "grand_total": actual_total + covsl5_total,
            "expected_total": EXPECTED_TOTAL,
            "expected_per_probe": EXPECTED_PER_PROBE,
            "matches_expected": matches_expected,
            "discrepancy_explanation": discrepancy_explanation,
        },
        "exclusion_breakdown": exclusion_breakdown,
        "test_results": test_results,
        "stage_permissions": {
            "d1_allowed": False,
            "d1_allowed_reason": (
                "D0-R is an audit-only feasibility stage. D1 execution has NOT started. "
                "The triage decision RECOMMENDS authorizing D1 via EPRO v3.1; it does not "
                "itself permit D1 training to run."
            ),
            "training_allowed": False,
            "training_allowed_reason": (
                "Learned training requires D2 Tier B approval. D0-R never authorizes training."
            ),
        },
        "triage_decision": "non_zero_candidate_pairs_authorize_d1",
        "triage_detail": (
            f"{actual_total} functional-anchor candidate single-mutant relations found "
            f"(functional Hamming == 1, name-encoded single mutation with matching "
            f"pos/ref/alt). Recommend publishing EPRO v3.1 to authorize D1. D1 execution "
            f"itself and learned training remain gated (d1_allowed=False, "
            f"training_allowed=False)."
        ),
        "key_findings": key_findings,
        "scientific_boundary": (
            "D0-R is a fail-forward data feasibility audit. Candidate relations are "
            "unverified (candidate_only_pending_parent_lineage_and_functional_region_"
            "validation, true_pair=False). No pair, tier, or model claim is made. The "
            "original D0 acceptance (NO_GO) and the previous D0-R acceptance are both "
            "preserved as historical evidence."
        ),
    }

    acceptance_path = output_dir / "d0r_acceptance_v2.json"
    with acceptance_path.open("w", encoding="utf-8") as f:
        json.dump(acceptance, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote: {acceptance_path}")

    report_path = output_dir / "d0r_data_feasibility_audit.md"
    _write_report(report_path, acceptance, m2sl5_relations, covsl5_relations)
    print(f"Wrote: {report_path}")

    print(json.dumps({
        "triage": acceptance["triage_decision"],
        "m2sl5_total": actual_total,
        "covsl5_total": covsl5_total,
        "matches_expected": matches_expected,
        "tests_all_passed": test_results.get("all_passed"),
    }, indent=2))
    return 0


def _write_report(
    path: Path,
    acceptance: dict[str, Any],
    m2sl5_relations: dict[str, Any],
    covsl5_relations: dict[str, Any],
) -> None:
    cc = acceptance["candidate_counts"]
    av = acceptance["anchor_verification"]
    tr = acceptance["test_results"]
    sp = acceptance["stage_permissions"]
    lines = [
        "# D0-R Functional-Anchor Data Feasibility Audit Report",
        "",
        f"**Generated:** {acceptance['generated_at']}",
        f"**Audit Method:** `{acceptance['audit_method']}`",
        f"**Triage Decision:** `{acceptance['triage_decision']}`",
        f"**Stage Permissions:** d1_allowed=`{sp['d1_allowed']}`, "
        f"training_allowed=`{sp['training_allowed']}`",
        "",
        "## Summary",
        "",
        f"- Audit method: 124 nt functional window at offset 31 within the 206 nt full anchor",
        f"- M2SL5 candidate relations: {cc['m2sl5_total']} "
        f"({cc['m2sl5_per_probe'].get('M2SL5_2A3_0000', 0)} 2A3 + "
        f"{cc['m2sl5_per_probe'].get('M2SL5_DMS_0000', 0)} DMS)",
        f"- COVSL5/SL5CV2 candidate relations: {cc['covsl5_sl5cv2_total']}",
        f"- Grand total candidate relations: {cc['grand_total']}",
        f"- Expected (handoff): {cc['expected_total']} ({cc['expected_per_probe']}/probe)",
        f"- Matches expected: `{cc['matches_expected']}`",
        "",
        "## Anchor Verification",
        "",
        f"- Full anchor (SL5CV2_NOM_0002): {av['sl5cv2_full_anchor_length']} nt, "
        f"matches expected=`{av['sl5cv2_full_anchor_matches_expected']}`",
        f"- Functional anchor (COVSL5_NOM_0002): {av['covsl5_functional_anchor_length']} nt, "
        f"matches expected=`{av['covsl5_functional_anchor_matches_expected']}`",
        f"- Functional anchor in full anchor: "
        f"valid=`{av['functional_anchor_verification']['valid']}`, "
        f"offset={av['functional_anchor_verification']['offset']}, "
        f"occurrences={av['functional_anchor_verification']['occurrences']}",
        "",
        "## Candidate Counts: Actual vs Expected",
        "",
        f"- Actual: **{cc['m2sl5_total']}** M2SL5 candidates "
        f"({cc['m2sl5_per_probe'].get('M2SL5_2A3_0000', 0)}/probe x 2 probes = "
        f"124 positions x 3 mutations = 372/probe)",
        f"- Expected: {cc['expected_total']} ({cc['expected_per_probe']}/probe, likely 64 nt "
        f"sub-region x 3 = 192/probe)",
        "",
        f"**Discrepancy explanation:** {cc['discrepancy_explanation']}",
        "",
        "## Exclusion Breakdown (per M2SL5 file)",
        "",
    ]
    for fname, reasons in acceptance["exclusion_breakdown"].items():
        lines.append(f"### {fname}")
        lines.append("")
        for reason, count in reasons.items():
            lines.append(f"- {reason}: {count}")
        cand = cc["m2sl5_per_probe"].get(fname, 0)
        lines.append(f"- candidate_single_functional_anchor: {cand}")
        lines.append("")
    lines.extend([
        "## COVSL5/SL5CV2 Files",
        "",
        f"- Total candidates: {cc['covsl5_sl5cv2_total']}",
        f"- Source files: {', '.join(covsl5_relations.get('source_files', []))}",
        "- SL5CV2 files lack the WT anchor sequence (wt_anchor_found=false); COVSL5 files "
        "have the WT anchor but their mutation annotations use a different format "
        "(e.g. `G159C`) that does not match the name-encoded `<pos><ref>-<alt>` scheme.",
        "",
        "## Test Results",
        "",
        f"- Command: `{tr.get('command')}`",
        f"- Exit code: {tr.get('exit_code')}",
        f"- Passed: {tr.get('passed')} | Failed: {tr.get('failed')} | "
        f"Errors: {tr.get('errors')} | Skipped: {tr.get('skipped')}",
        f"- All passed: `{tr.get('all_passed')}`",
        f"- Summary: {tr.get('summary_line')}",
        "",
        "## Stage Permissions",
        "",
        f"- **d1_allowed:** `{sp['d1_allowed']}` — {sp['d1_allowed_reason']}",
        f"- **training_allowed:** `{sp['training_allowed']}` — {sp['training_allowed_reason']}",
        f"- **triage_decision:** `{acceptance['triage_decision']}`",
        "",
        acceptance["triage_detail"],
        "",
        "## Key Findings",
        "",
    ])
    for i, finding in enumerate(acceptance["key_findings"], 1):
        lines.append(f"{i}. {finding}")
    lines.extend([
        "",
        "## Scientific Boundary",
        "",
        acceptance["scientific_boundary"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
