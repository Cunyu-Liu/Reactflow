#!/usr/bin/env python3
"""D0-R: build independent acceptance artifact and feasibility audit report.

Reads the D0-R artifacts (construct audit, M2SL5 candidate relations, accession
registry, Ribonanza audit) and produces an independent D0R acceptance that does
NOT modify or replace the original D0 acceptance. Outputs a triage decision:
zero auditable pairs -> freeze EPRO; non-zero candidate pairs -> v3.1 authorize D1.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "reactflow-delta-d0r-acceptance-v1"


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct-audit", required=True)
    parser.add_argument("--m2sl5-relations", required=True)
    parser.add_argument("--accession-summary", required=True)
    parser.add_argument("--ribonanza-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.construct_audit).open() as f:
        audit = json.load(f)
    with Path(args.m2sl5_relations).open() as f:
        relations = json.load(f)
    with Path(args.accession_summary).open() as f:
        accession = json.load(f)
    with Path(args.ribonanza_audit).open() as f:
        ribonanza = json.load(f)

    candidate_count = relations["total_candidate_relations"]
    strict_candidates = sum(
        1 for r in relations["relations"]
        if r["edit_class"] == "candidate_single_from_name"
        and r["functional_edit_count"] == 1
    )

    # Triage decision
    if strict_candidates > 0:
        triage = "non_zero_candidate_pairs_authorize_d1"
        triage_detail = (
            f"{strict_candidates} strict candidate single-mutant relations found "
            f"(functional_edit_count=1, name-encoded single mutation). "
            "Publish v3.1 authorizing D1. Learned training still requires D2 Tier B."
        )
    else:
        triage = "zero_auditable_pairs_freeze_epro"
        triage_detail = (
            "Zero strict candidate pairs found. Freeze EPRO and convert to negative result."
        )

    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "original_d0_acceptance_preserved": True,
        "original_d0_acceptance_path": "artifacts/reactflow_delta/d0/d0_acceptance_a79fd073.json",
        "inputs": {
            "construct_audit": str(Path(args.construct_audit).resolve()),
            "m2sl5_relations": str(Path(args.m2sl5_relations).resolve()),
            "accession_summary": str(Path(args.accession_summary).resolve()),
            "ribonanza_audit": str(Path(args.ribonanza_audit).resolve()),
        },
        "summary": {
            "total_rdat_files_audited": audit["total_files"],
            "files_with_parse_errors": audit["files_with_errors"],
            "files_with_wt_anchor": audit["files_with_wt_anchor"],
            "total_profiles": audit["total_profiles"],
            "profiles_with_per_profile_sequence": audit["total_profiles_with_sequence"],
            "accession_registry_entries": accession["total_entries"],
            "accession_entries_with_pubmed": accession["entries_with_pubmed"],
            "m2sl5_candidate_relations": candidate_count,
            "m2sl5_strict_candidates_functional_edit_1": strict_candidates,
        },
        "key_findings": [
            "D0 used filename-keyword recall; D0-R uses paper-accession mapping from 1024 RMDB _entries YAML front matter.",
            "D0 only downloaded RMDB metadata; D0-R downloaded 13 RDAT payload files (~33MB) with sha256 checksums.",
            "D0 RDAT parser only accepted global SEQUENCE header; D0-R parser supports per-profile SEQUENCE:N lines and sequence: annotation tokens (M2-seq style).",
            "D0 parser rejected 5012 M2SL5 profiles as 'parent sequence masked/noncanonical'; D0-R recovers per-profile sequences from annotation tokens.",
            "D0 pairing used mutation labels only; D0-R computes actual edit sets from per-profile sequence vs WT anchor, partitioned by SEQPOS window.",
            f"ETERNA_R78 (PMID 36192461, paper-explicit) uses RDAT_VERSION 0.33, inaccessible to fail-closed v0.34 parser. Recorded as parse error.",
            f"M2SL5 (Ribonanza pre-competition) produced {strict_candidates} strict candidate single-mutant relations (functional_edit_count=1).",
            "Ribonanza Kaggle data inaccessible (no credentials), but M2SL5 derivative accessible via RMDB. D0 404 reinterpreted as探测 method failure.",
            "All candidate relations are candidate_only_unverified — lineage NOT independently confirmed.",
        ],
        "triage_decision": triage,
        "triage_detail": triage_detail,
        "d1_authorization": {
            "authorized": strict_candidates > 0,
            "conditions": [
                "All D1 training pairs must come from strict candidates (functional_edit_count=1)",
                "Learned training requires D2 Tier B approval",
                "Lineage must be verified before upgrading from candidate to confirmed pair",
                "Raw RDAT files remain read-only (checksum-verified, not modified)",
            ] if strict_candidates > 0 else [],
        },
        "scientific_boundary": (
            "D0-R is a fail-forward data feasibility audit. Candidate relations are "
            "unverified. No pair, tier, or model claim is made. The original D0 "
            "acceptance (NO_GO) is preserved as historical evidence."
        ),
    }

    # Write acceptance JSON
    acceptance_path = output_dir / "d0r_acceptance.json"
    with acceptance_path.open("w", encoding="utf-8") as f:
        json.dump(acceptance, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote: {acceptance_path}")

    # Write markdown report
    report_path = output_dir / "d0r_data_feasibility_audit.md"
    _write_report(report_path, acceptance, relations, ribonanza)
    print(f"Wrote: {report_path}")

    print(json.dumps({"triage": triage, "strict_candidates": strict_candidates}, indent=2))
    return 0


def _write_report(path: Path, acceptance: dict, relations: dict, ribonanza: dict) -> None:
    lines = [
        "# D0-R Data Feasibility Audit Report",
        "",
        f"**Generated:** {acceptance['generated_at']}",
        f"**Triage Decision:** `{acceptance['triage_decision']}`",
        "",
        "## Summary",
        "",
        f"- RDAT files audited: {acceptance['summary']['total_rdat_files_audited']}",
        f"- Files with parse errors: {acceptance['summary']['files_with_parse_errors']}",
        f"- Files with WT anchor: {acceptance['summary']['files_with_wt_anchor']}",
        f"- Total profiles: {acceptance['summary']['total_profiles']}",
        f"- Profiles with per-profile sequence: {acceptance['summary']['profiles_with_per_profile_sequence']}",
        f"- Accession registry entries: {acceptance['summary']['accession_registry_entries']}",
        f"- M2SL5 candidate relations: {acceptance['summary']['m2sl5_candidate_relations']}",
        f"- M2SL5 strict candidates (functional_edit_count=1): {acceptance['summary']['m2sl5_strict_candidates_functional_edit_1']}",
        "",
        "## Triage Decision",
        "",
        acceptance["triage_detail"],
        "",
        "## Key Findings",
        "",
    ]
    for i, finding in enumerate(acceptance["key_findings"], 1):
        lines.append(f"{i}. {finding}")
    lines.extend([
        "",
        "## Ribonanza Audit",
        "",
        f"- Kaggle CLI installed: {ribonanza['kaggle_status']['kaggle_cli_installed']}",
        f"- Kaggle credentials present: {ribonanza['kaggle_status']['kaggle_json_exists']}",
        f"- M2SL5 accessible via RMDB: {ribonanza['m2sl5_via_rmdb']['accessible']}",
        f"- Conclusion: {ribonanza['conclusion']}",
        "",
        "## D1 Authorization",
        "",
        f"**Authorized:** {acceptance['d1_authorization']['authorized']}",
    ])
    if acceptance["d1_authorization"]["conditions"]:
        lines.append("")
        lines.append("Conditions:")
        for cond in acceptance["d1_authorization"]["conditions"]:
            lines.append(f"- {cond}")
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
