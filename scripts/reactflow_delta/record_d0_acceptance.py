#!/usr/bin/env python3
"""Write one immutable, fail-closed D0 acceptance certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.acceptance import build_d0_acceptance_certificate
from reactflow.delta.data import write_json_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parser-fixture-results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-generating-commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--push-status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    certificate = build_d0_acceptance_certificate(
        summary_path=args.summary,
        parser_fixture_results_path=args.parser_fixture_results,
        report_path=args.report,
        contract_path=args.contract,
        evidence_generating_commit=args.evidence_generating_commit,
        branch=args.branch,
        push_status=args.push_status,
    )
    write_json_document(args.output, certificate)
    print(json.dumps({"acceptance_status": certificate["acceptance_status"], "d1_allowed": certificate["decision"]["d1_allowed"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
