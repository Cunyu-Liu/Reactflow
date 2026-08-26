#!/usr/bin/env python3
"""Qualify V13 real-data smoke without scientific scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.merge_model_rescue_v13 import SCHEMA as MERGED_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v13_smoke_qualification.v1"


def qualify(merged: dict[str, Any]) -> dict[str, Any]:
    expected = "V13M2_COMPLETE_UNSCORED_SMOKE_MERGE_PASS"
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != expected:
        raise ValueError("V13 smoke qualifier requires the complete unscored smoke merge")
    rows = merged.get("folds", [])
    folds = sorted(int(row["outer_fold"]) for row in rows)
    seeds = {int(row["seed"]) for row in rows}
    integrity = merged.get("merge_integrity", {})
    passed = (
        folds == [0, 1]
        and seeds == {0}
        and all(row.get("evidence_status") == "ENGINEERING_SMOKE_ONLY" for row in rows)
        and all(row.get("history_lengths", {}).get("candidate_point") == 3 for row in rows)
        and all(row.get("history_lengths", {}).get("null_point") == 3 for row in rows)
        and integrity.get("complete_fold_seed_universe") is True
        and integrity.get("prediction_only_schema") is True
        and integrity.get("exact_point_parameter_and_initial_state_match_all_runs") is True
        and integrity.get("second_pass_only_difference_all_runs") is True
        and integrity.get("null_hidden_delta_at_most_1e_7_all_runs") is True
        and integrity.get("point_frozen_during_calibration_all_runs") is True
        and integrity.get("median_constraint_all_runs") is True
        and integrity.get("partial_scores_inspected") is False
        and integrity.get("external_outcome_accessed") is False
    )
    return {
        "schema_version": SCHEMA,
        "phase": "V13M2",
        "status": (
            "V13M2_ENGINEERING_SMOKE_PASS"
            if passed
            else "V13M2_ENGINEERING_SMOKE_FAIL"
        ),
        "gate_passed": passed,
        "scientific_score_computed": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(json.loads(args.merged_json.read_text(encoding="utf-8")))
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
