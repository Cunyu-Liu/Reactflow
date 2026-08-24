#!/usr/bin/env python3
"""Merge all 20 V7M2 prediction-only folds before any target join."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.run_model_rescue_v7_probe import (
    FOLD_SCHEMA,
    PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v7_probe_merged.v1"


def _validate_prediction(path: Path, fold: int, expected_rows: int) -> None:
    with np.load(path, allow_pickle=True) as handle:
        fields = set(handle.files)
        required = {
            "schema_version",
            "keys",
            "biological_scoring_key",
            "outer_fold",
            "registered_status",
            *PREDICTION_FIELDS,
        }
        if not required.issubset(fields):
            raise ValueError(f"V7M2 prediction fields are incomplete in {path}")
        forbidden = {
            "target",
            "target_error",
            "target_mask",
            "qualified_target_mask",
            "score",
            "mae",
            "crps",
        }
        if not forbidden.isdisjoint(fields):
            raise ValueError(f"V7M2 prediction contains target-side fields in {path}")
        if str(handle["schema_version"].item()) != PREDICTION_SCHEMA:
            raise ValueError(f"invalid V7M2 prediction schema in {path}")
        keys = handle["keys"]
        if len(keys) != expected_rows:
            raise ValueError(f"V7M2 prediction row count differs in {path}")
        if not np.array_equal(keys, handle["biological_scoring_key"]):
            raise ValueError(f"V7M2 biological keys differ in {path}")
        if len(set(map(str, keys))) != len(keys):
            raise ValueError(f"V7M2 biological keys are duplicated in {path}")
        if set(map(int, handle["outer_fold"])) != {fold}:
            raise ValueError(f"V7M2 outer fold differs in {path}")
        if set(map(str, handle["registered_status"])) != {"covered"}:
            raise ValueError(f"V7M2 registered status differs in {path}")
        for name in PREDICTION_FIELDS:
            if handle[name].shape != (len(keys),) or not np.isfinite(
                handle[name]
            ).all():
                raise ValueError(f"V7M2 prediction {name} is invalid in {path}")


def merge_folds(input_dir: Path) -> dict:
    expected = set(range(20))
    rows = []
    seen: set[int] = set()
    for path in sorted(input_dir.glob("v7_probe_fold_result_fold*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != "V7M2":
            raise ValueError(f"invalid V7M2 fold result {path}")
        fold = int(row["outer_fold"])
        if fold in seen:
            raise ValueError(f"duplicate V7M2 fold {fold}")
        seen.add(fold)
        if row.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
            raise ValueError(f"V7M2 fold {fold} lacks corrected target identity")
        if row.get("corrected_feature41_replay_pass") is not True:
            raise ValueError(f"V7M2 fold {fold} lacks corrected baseline replay")
        required_false = (
            "held_target_used_for_prediction",
            "held_score_computed",
            "partial_score_inspected",
            "model_selection_performed",
            "legacy_target_dependent_prediction_reused",
            "external_outcome_accessed",
        )
        if any(row.get(name) is not False for name in required_false):
            raise ValueError(f"V7M2 fold {fold} violates prediction-only isolation")
        for name in (
            "prediction_artifact",
            "model_artifact",
            "corrected_baseline_reference",
        ):
            if not Path(row[name]).exists():
                raise FileNotFoundError(row[name])
        _validate_prediction(
            Path(row["prediction_artifact"]),
            fold,
            int(row["n_registered_prediction_rows"]),
        )
        rows.append(row)
    if seen != expected or len(rows) != 20:
        raise ValueError(f"V7M2 fold universe incomplete: found {sorted(seen)}")
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "V7M2",
        "status": "V7M2_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": rows,
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_folds": True,
            "referenced_artifacts_exist": True,
            "prediction_schema_valid": True,
            "prediction_only_fields": True,
            "target_identity_exact": True,
            "corrected_feature41_replay_all_folds": True,
            "held_scores_absent": True,
            "partial_score_inspected": False,
            "model_selection_performed": False,
            "legacy_target_dependent_prediction_reused": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_folds(args.input_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
