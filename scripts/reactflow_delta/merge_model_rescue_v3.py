#!/usr/bin/env python3
"""Merge the complete fixed seed-0 R3M3 fold universe before qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.model_rescue_v3 import CANDIDATE
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE
from scripts.reactflow_delta.run_model_rescue_v3 import SCHEMA


MERGED_SCHEMA = "reactflow_delta.model_rescue_v3_merged_result.v1"
EXPECTED_FOLDS = list(range(20))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def merge_screen_folds(input_dir: Path) -> dict[str, Any]:
    paths = sorted(input_dir.glob("v3_fold_result_fold*_seed0.json"))
    if not paths:
        raise FileNotFoundError(f"no v3 seed-0 fold results below {input_dir}")
    folds = [_read(path) for path in paths]
    fold_ids = [int(row.get("outer_fold", -1)) for row in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate outer fold result")
    if sorted(fold_ids) != EXPECTED_FOLDS:
        raise ValueError("merge requires exactly folds 0 through 19")
    for row in folds:
        fold = int(row["outer_fold"])
        if row.get("schema_version") != SCHEMA or int(row.get("seed", -1)) != 0:
            raise ValueError(f"fold {fold} has wrong schema or seed")
        baseline = row.get("baseline", {})
        candidate = row.get("candidate", {})
        if baseline.get("model_id") != BASELINE:
            raise ValueError(f"fold {fold} is not bound to B1")
        if candidate.get("candidate_id") != CANDIDATE:
            raise ValueError(f"fold {fold} has wrong v3 candidate")
        artifacts = [
            baseline.get("prediction_artifact", ""),
            baseline.get("checkpoint", ""),
            candidate.get("prediction_artifact", ""),
            candidate.get("b1_checkpoint", ""),
            candidate.get("meanaligned_checkpoint", ""),
            candidate.get("calibration_checkpoint", ""),
            candidate.get("inner_crossfit_ledger", ""),
        ]
        for artifact_value in artifacts:
            artifact = Path(artifact_value)
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise FileNotFoundError(f"fold {fold} missing referenced artifact {artifact}")
        invariants = row.get("invariants", {})
        if not (
            invariants.get("held_target_error_mask_invariance") is True
            and invariants.get("inner_crossfit_complete") is True
            and invariants.get("method_used_as_gate_input") is False
            and invariants.get("residual_changed_point_mean") is False
        ):
            raise ValueError(f"fold {fold} did not record all frozen v3 invariants")
    ordered = sorted(folds, key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": MERGED_SCHEMA,
        "source_fold_schema": SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_SCREEN",
        "seed": 0,
        "folds": ordered,
        "merge_integrity": {
            "n_folds": 20,
            "fold_ids": EXPECTED_FOLDS,
            "unique_folds": True,
            "all_referenced_artifacts_present": True,
            "partial_scores_inspected_before_merge": False,
        },
        "qualification": {
            "external": "NOT_ACCESSED",
            "sota": "NOT_ESTABLISHED",
            "r3m4_authorized_before_qualification": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_screen_folds(args.input_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "R3M3_TWENTY_FOLD_MERGE_PASS",
                "n_folds": len(result["folds"]),
                "result": str(args.out_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
