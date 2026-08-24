#!/usr/bin/env python3
"""Merge all 20 corrected baseline prediction-only folds before scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.run_target_identity_corrected_baselines import (
    FOLD_SCHEMA,
    PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)


SCHEMA = "reactflow_delta.target_identity_corrected_baseline_merged.v1"


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
            raise ValueError(f"corrected prediction fields are incomplete in {path}")
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
            raise ValueError(f"corrected prediction contains target-side fields in {path}")
        if str(handle["schema_version"].item()) != PREDICTION_SCHEMA:
            raise ValueError(f"invalid corrected prediction schema in {path}")
        keys = handle["keys"]
        if len(keys) != expected_rows:
            raise ValueError(f"corrected prediction row count differs in {path}")
        if not np.array_equal(keys, handle["biological_scoring_key"]):
            raise ValueError(f"corrected biological keys differ in {path}")
        if len(set(map(str, keys))) != len(keys):
            raise ValueError(f"corrected biological keys are duplicated in {path}")
        if set(map(int, handle["outer_fold"])) != {fold}:
            raise ValueError(f"corrected outer fold differs in {path}")
        if set(map(str, handle["registered_status"])) != {"covered"}:
            raise ValueError(f"corrected registered status differs in {path}")
        for name in PREDICTION_FIELDS:
            if handle[name].shape != (len(keys),) or not np.isfinite(
                handle[name]
            ).all():
                raise ValueError(f"corrected prediction {name} is invalid in {path}")


def merge_folds(input_dir: Path) -> dict:
    expected = set(range(20))
    paths = sorted(input_dir.glob("tic2a_corrected_fold_result_fold*.json"))
    rows = []
    seen = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != "TIC2A":
            raise ValueError(f"invalid corrected fold result {path}")
        fold = int(row["outer_fold"])
        if fold in seen:
            raise ValueError(f"duplicate corrected fold {fold}")
        seen.add(fold)
        required_true = (
            "v5_v6_feature30_stats_replay_pass",
            "v5_v6_feature30_prediction_replay_pass",
        )
        if any(row.get(name) is not True for name in required_true):
            raise ValueError(f"corrected fold {fold} lacks feature30 replay")
        required_false = (
            "held_target_used_for_prediction",
            "held_score_computed",
            "partial_score_inspected",
            "legacy_prediction_reused",
            "external_outcome_accessed",
        )
        if any(row.get(name) is not False for name in required_false):
            raise ValueError(f"corrected fold {fold} violates prediction-only isolation")
        if row.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
            raise ValueError(f"corrected fold {fold} lacks exact target identity")
        for name in ("prediction_artifact", "model_artifact"):
            if not Path(row[name]).exists():
                raise FileNotFoundError(row[name])
        _validate_prediction(
            Path(row["prediction_artifact"]),
            fold,
            int(row["n_registered_prediction_rows"]),
        )
        rows.append(row)
    if seen != expected or len(rows) != 20:
        raise ValueError(f"corrected fold universe incomplete: found {sorted(seen)}")
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "TIC2A",
        "status": "TIC2A_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": rows,
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_folds": True,
            "referenced_artifacts_exist": True,
            "prediction_schema_valid": True,
            "prediction_only_fields": True,
            "target_identity_exact": True,
            "v5_v6_feature30_replay_all_folds": True,
            "held_scores_absent": True,
            "partial_score_inspected": False,
            "legacy_prediction_reused": False,
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
