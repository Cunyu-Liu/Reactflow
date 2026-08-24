#!/usr/bin/env python3
"""Merge the complete prediction-only V6M2 fold universe without scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.run_model_rescue_v6_probe import (
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v6_probe_merged.v1"


def _validate_prediction(path: Path, fold: int, expected_rows: int) -> None:
    with np.load(path, allow_pickle=True) as handle:
        fields = set(handle.files)
        required = {
            "schema_version",
            "keys",
            "biological_scoring_key",
            "outer_fold",
            "baseline_signed_delta",
            "baseline_absolute_delta",
            "candidate_signed_delta",
            "candidate_absolute_delta",
            "registered_status",
        }
        if not required.issubset(fields):
            raise ValueError(f"V6M2 prediction fields are incomplete in {path}")
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
            raise ValueError(f"V6M2 prediction contains target-side fields in {path}")
        if str(handle["schema_version"].item()) != PREDICTION_SCHEMA:
            raise ValueError(f"invalid V6M2 prediction schema in {path}")
        keys = handle["keys"]
        if len(keys) != expected_rows:
            raise ValueError(f"V6M2 prediction row count differs in {path}")
        if not np.array_equal(keys, handle["biological_scoring_key"]):
            raise ValueError(f"V6M2 biological keys differ in {path}")
        if len(set(map(str, keys))) != len(keys):
            raise ValueError(f"V6M2 biological keys are duplicated in {path}")
        if set(map(int, handle["outer_fold"])) != {fold}:
            raise ValueError(f"V6M2 outer fold differs in {path}")
        if set(map(str, handle["registered_status"])) != {"covered"}:
            raise ValueError(f"V6M2 registered status differs in {path}")
        for name in (
            "baseline_signed_delta",
            "baseline_absolute_delta",
            "candidate_signed_delta",
            "candidate_absolute_delta",
        ):
            if handle[name].shape != (len(keys),) or not np.isfinite(handle[name]).all():
                raise ValueError(f"V6M2 prediction {name} is invalid in {path}")


def merge_folds(input_dir: Path) -> dict:
    expected = set(range(20))
    paths = sorted(input_dir.glob("v6_probe_fold_result_fold*.json"))
    rows = []
    seen = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != "V6M2":
            raise ValueError(f"invalid V6M2 fold result {path}")
        fold = int(row["outer_fold"])
        if fold in seen:
            raise ValueError(f"duplicate V6M2 fold {fold}")
        seen.add(fold)
        if row.get("v5_baseline_replay_pass") is not True:
            raise ValueError(f"fold {fold} lacks the mandatory v5 baseline replay")
        if row.get("held_score_computed") is not False:
            raise ValueError(f"fold {fold} contains a held score")
        if row.get("partial_score_inspected") is not False:
            raise ValueError(f"fold {fold} records partial score access")
        if row.get("external_outcome_accessed") is not False:
            raise ValueError(f"fold {fold} records external outcome access")
        for key in (
            "prediction_artifact",
            "model_artifact",
            "v5_reference_prediction",
        ):
            if not Path(row[key]).exists():
                raise FileNotFoundError(row[key])
        _validate_prediction(
            Path(row["prediction_artifact"]),
            fold,
            int(row["n_registered_prediction_rows"]),
        )
        rows.append(row)
    if seen != expected or len(rows) != 20:
        raise ValueError(f"V6M2 fold universe incomplete: found {sorted(seen)}")
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "V6M2",
        "status": "V6M2_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": rows,
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_folds": True,
            "referenced_artifacts_exist": True,
            "prediction_schema_valid": True,
            "prediction_only_fields": True,
            "v5_baseline_replay_all_folds": True,
            "held_scores_absent": True,
            "partial_score_inspected": False,
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
