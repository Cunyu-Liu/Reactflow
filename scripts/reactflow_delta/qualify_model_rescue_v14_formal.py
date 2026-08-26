#!/usr/bin/env python3
"""Apply the pre-frozen V14 five-seed formal Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.qualify_model_rescue_v13_formal import (
    qualify as qualify_v13_formal,
)
from scripts.reactflow_delta.qualify_model_rescue_v13 import (
    SCHEMA as V13_SCREEN_QUAL_SCHEMA,
)
from scripts.reactflow_delta.qualify_model_rescue_v14 import (
    SCHEMA as SCREEN_QUAL_SCHEMA,
)
from scripts.reactflow_delta.score_model_rescue_v13_formal import (
    SCHEMA as V13_SCORE_SCHEMA,
)
from scripts.reactflow_delta.score_model_rescue_v14_formal import (
    SCHEMA as SCORE_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v14_formal_qualification.v1"


def qualify(scores: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
    if screen.get("schema_version") != SCREEN_QUAL_SCHEMA or screen.get("status") != (
        "V14M3_TOP_JOURNAL_SCREEN_PASS"
    ) or screen.get("gate_passed") is not True:
        raise ValueError("V14 formal qualifier requires exact screen PASS")
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V14M4_COMPLETE_FORMAL_SCORE_PASS"
    ):
        raise ValueError("V14 formal qualifier requires complete formal scores")
    adapted_screen = dict(screen)
    adapted_screen["schema_version"] = V13_SCREEN_QUAL_SCHEMA
    adapted_screen["status"] = "V13M3_TOP_JOURNAL_SCREEN_PASS"
    adapted_score = dict(scores)
    adapted_score["schema_version"] = V13_SCORE_SCHEMA
    adapted_score["status"] = "V13M4_COMPLETE_FORMAL_SCORE_PASS"
    result = qualify_v13_formal(adapted_score, adapted_screen)
    passed = bool(result["gate_passed"])
    return {
        **result,
        "schema_version": SCHEMA,
        "phase": "V14M4",
        "status": (
            "V14M4_TOP_JOURNAL_FORMAL_PASS"
            if passed
            else "V14M4_TOP_JOURNAL_FORMAL_FAIL"
        ),
        "attribution_null": "IDENTICAL_ARCHITECTURE_FROM_SCRATCH",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--screen-qualification-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(
        json.loads(args.score_json.read_text(encoding="utf-8")),
        json.loads(args.screen_qualification_json.read_text(encoding="utf-8")),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
