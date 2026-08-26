#!/usr/bin/env python3
"""Qualify only V14M2 engineering invariants, never scientific scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.merge_model_rescue_v14 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v14 import merged_integrity_pass


SCHEMA = "reactflow_delta.model_rescue_v14_smoke_qualification.v1"


def qualify(merged: dict[str, Any]) -> dict[str, Any]:
    passed = (
        merged.get("schema_version") == MERGED_SCHEMA
        and merged.get("phase") == "V14M2"
        and merged.get("status") == "V14M2_COMPLETE_UNSCORED_SMOKE_MERGE_PASS"
        and merged_integrity_pass(merged.get("merge_integrity", {}))
        and len(merged.get("folds", [])) == 2
    )
    return {
        "schema_version": SCHEMA,
        "phase": "V14M2",
        "status": (
            "V14M2_ENGINEERING_SMOKE_PASS"
            if passed
            else "V14M2_ENGINEERING_SMOKE_FAIL"
        ),
        "gate_passed": passed,
        "scientific_score_computed": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(json.loads(args.merged_json.read_text(encoding="utf-8")))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
