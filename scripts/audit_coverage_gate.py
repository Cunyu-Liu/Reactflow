#!/usr/bin/env python3
"""Audit ReactFlow test coverage gate from ``coverage json`` output.

This script converts the pytest-cov terminal evidence into a stable artifact.
It reads a coverage.py JSON report, checks the total percentage against a
threshold, and writes JSON/Markdown summaries for the paper bundle.

Complexity: O(C), where C is the coverage JSON size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence


def _read_json(path: Path) -> Mapping[str, object]:
    """Read a JSON mapping from ``path``.

    Complexity: O(file bytes).
    """

    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping):
        raise ValueError("coverage JSON root must be an object")
    return obj


def extract_total_coverage_percent(coverage_obj: Mapping[str, object]) -> float:
    """Extract total line/branch coverage percent from coverage.py JSON.

    ``coverage json`` stores aggregate metrics under ``totals``.  Newer versions
    expose ``percent_covered`` as a numeric value; the display string is only used
    as a fallback.

    Complexity: O(1).
    """

    totals = coverage_obj.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("coverage JSON is missing totals")
    value = totals.get("percent_covered")
    if isinstance(value, (int, float)):
        return float(value)
    display = totals.get("percent_covered_display")
    if isinstance(display, str):
        return float(display.rstrip("%"))
    raise ValueError("coverage totals missing percent_covered")


def run_audit(coverage_json: Path, *, threshold: float) -> dict:
    """Run the coverage gate audit.

    Complexity: O(C), where C is the coverage JSON size.
    """

    obj = _read_json(coverage_json)
    percent = extract_total_coverage_percent(obj)
    totals = obj.get("totals", {})
    passed = percent >= threshold
    return {
        "coverage_json": str(coverage_json),
        "threshold": threshold,
        "percent_covered": percent,
        "passed": passed,
        "totals": totals,
    }


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write the coverage gate audit as Markdown.

    Complexity: O(1).
    """

    lines = [
        "# ReactFlow Coverage Gate Audit",
        "",
        f"- passed: `{result['passed']}`",
        f"- percent_covered: `{result['percent_covered']}`",
        f"- threshold: `{result['threshold']}`",
        f"- coverage_json: `{result['coverage_json']}`",
        "",
        "## Totals",
        "",
        "```json",
        json.dumps(result["totals"], indent=2, sort_keys=True),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(C), where C is the coverage JSON size.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--fail-under", action="store_true")
    args = parser.parse_args(argv)

    result = run_audit(Path(args.coverage_json), threshold=args.threshold)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps({"passed": result["passed"], "percent_covered": result["percent_covered"]}, sort_keys=True))
    return 1 if args.fail_under and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
