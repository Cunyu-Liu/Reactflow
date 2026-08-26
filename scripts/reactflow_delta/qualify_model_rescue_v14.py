#!/usr/bin/env python3
"""Apply the frozen V14 top-journal seed-0 Gate mechanically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.qualify_model_rescue_v13 import qualify as qualify_v13
from scripts.reactflow_delta.score_model_rescue_v13 import SCHEMA as V13_SCORE_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v14 import SCHEMA as SCORE_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v14_qualification.v1"


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V14M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("V14 qualifier requires one complete V14 score artifact")
    adapted = dict(scores)
    adapted["schema_version"] = V13_SCORE_SCHEMA
    adapted["status"] = "V13M3_COMPLETE_SCORE_PASS"
    result = qualify_v13(adapted)
    passed = bool(result["gate_passed"])
    return {
        **result,
        "schema_version": SCHEMA,
        "phase": "V14M3",
        "status": (
            "V14M3_TOP_JOURNAL_SCREEN_PASS"
            if passed
            else "V14M3_TOP_JOURNAL_SCREEN_FAIL"
        ),
        "v14m4_authorized": passed,
        "v13m4_authorized": None,
        "attribution_null": "IDENTICAL_ARCHITECTURE_FROM_SCRATCH",
        "evidence_status": "POST_HOC_DEVELOPMENT_SCREEN_ONLY",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError("V14 refuses to overwrite its one qualification")
    result = qualify(json.loads(args.score_json.read_text(encoding="utf-8")))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
