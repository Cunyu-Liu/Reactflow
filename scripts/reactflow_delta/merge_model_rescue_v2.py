#!/usr/bin/env python3
"""Merge the complete fixed seed-0 R2M3 fold universe before qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    MEAN_CANDIDATE,
)
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE, SCHEMA


MERGED_SCHEMA = "reactflow_delta.model_rescue_v2_merged_result.v1"
EXPECTED_FOLDS = list(range(20))


def _read_fold(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def merge_screen_folds(input_dir: Path) -> dict[str, Any]:
    paths = sorted(input_dir.glob("v2_fold_result_fold*_seed0.json"))
    if not paths:
        raise FileNotFoundError(f"no seed-0 fold results below {input_dir}")

    folds = [_read_fold(path) for path in paths]
    fold_ids = [int(row.get("outer_fold", -1)) for row in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate outer fold result")
    if sorted(fold_ids) != EXPECTED_FOLDS:
        raise ValueError("merge requires exactly folds 0 through 19")

    expected_candidates = {MEAN_CANDIDATE, CALIBRATED_CANDIDATE}
    for row in folds:
        fold = int(row["outer_fold"])
        if int(row.get("seed", -1)) != 0:
            raise ValueError(f"fold {fold} does not use frozen seed 0")
        baseline = row.get("baseline")
        if not isinstance(baseline, dict) or baseline.get("model_id") != BASELINE:
            raise ValueError(f"fold {fold} is not bound to frozen B1")
        candidates = row.get("candidates")
        if not isinstance(candidates, dict) or set(candidates) != expected_candidates:
            raise ValueError(f"fold {fold} does not contain the two frozen candidates")
        for candidate_id, candidate in candidates.items():
            if not isinstance(candidate, dict):
                raise ValueError(f"fold {fold} candidate {candidate_id} is invalid")
            for field in (
                "prediction_artifact",
                "mean_checkpoint",
                "calibration_checkpoint",
            ):
                artifact = Path(candidate.get(field, ""))
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"fold {fold} candidate {candidate_id} missing {field}: {artifact}"
                    )

    ordered = sorted(folds, key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": MERGED_SCHEMA,
        "source_fold_schema": SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_SCREEN",
        "seed": 0,
        "mean_epochs": 40,
        "calibration_epochs": 40,
        "folds": ordered,
        "merge_integrity": {
            "n_folds": len(ordered),
            "fold_ids": EXPECTED_FOLDS,
            "unique_folds": True,
            "all_referenced_artifacts_present": True,
            "partial_scores_inspected_before_merge": False,
        },
        "qualification": {
            "external": "NOT_ACCESSED",
            "sota": "NOT_ESTABLISHED",
            "r2m4_authorized_before_qualification": False,
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
                "status": "R2M3_TWENTY_FOLD_MERGE_PASS",
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
