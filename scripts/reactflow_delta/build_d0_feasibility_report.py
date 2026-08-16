#!/usr/bin/env python3
"""Build D0 feasibility summary, parser fixture results, and readable audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import write_json_document
from reactflow.delta.feasibility import build_d0_feasibility_summary, render_d0_feasibility_report, write_text_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct-audit", type=Path, required=True)
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--ribonanza-availability", type=Path, required=True)
    parser.add_argument("--filename-candidate-manifest", type=Path, required=True)
    parser.add_argument("--rdat-parse-manifest", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--parser-fixture-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    summary, parser_results = build_d0_feasibility_summary(
        construct_audit_path=args.construct_audit,
        candidate_registry_path=args.candidate_registry,
        matrix_path=args.matrix,
        relation_audit_path=args.relation_audit,
        ribonanza_availability_path=args.ribonanza_availability,
        filename_candidate_manifest_path=args.filename_candidate_manifest,
        rdat_parse_manifest_path=args.rdat_parse_manifest,
    )
    write_json_document(args.summary_output, summary)
    write_json_document(args.parser_fixture_output, parser_results)
    write_text_once(args.report_output, render_d0_feasibility_report(summary))
    print(json.dumps({"d1_allowed": summary["d1_allowed"], "summary_output": str(args.summary_output), "tier": summary["tier_preassessment"]["highest_currently_supported"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
