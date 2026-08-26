#!/usr/bin/env python3
"""Merge the complete post-V13 prediction-only fold universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.run_post_v13_route_diagnostics import (
    FOLD_SCHEMA,
    PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)


SCHEMA = "reactflow_delta.post_v13_route_diagnostic_merged.v1"
FORBIDDEN_FIELDS = {
    "target",
    "target_error",
    "target_mask",
    "qualified_target_mask",
    "qualified_mask",
    "loss",
    "score",
    "mae",
    "crps",
}


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
        if not required <= fields:
            raise ValueError(f"post-V13 prediction fields are incomplete in {path}")
        if fields & FORBIDDEN_FIELDS:
            raise ValueError(f"post-V13 prediction contains target-side fields in {path}")
        if str(handle["schema_version"].item()) != PREDICTION_SCHEMA:
            raise ValueError(f"invalid post-V13 prediction schema in {path}")
        keys = list(map(str, handle["keys"]))
        if len(keys) != expected_rows or len(keys) != len(set(keys)):
            raise ValueError(f"post-V13 prediction key universe is invalid in {path}")
        if keys != list(map(str, handle["biological_scoring_key"])):
            raise ValueError(f"post-V13 biological keys differ in {path}")
        if set(map(int, handle["outer_fold"])) != {fold}:
            raise ValueError(f"post-V13 outer fold differs in {path}")
        if set(map(str, handle["registered_status"])) != {"covered"}:
            raise ValueError(f"post-V13 registered status differs in {path}")
        for field in PREDICTION_FIELDS:
            if handle[field].shape != (len(keys),) or not np.isfinite(
                handle[field]
            ).all():
                raise ValueError(f"post-V13 prediction {field} is invalid in {path}")


def _recorded_invariants_pass(row: dict[str, Any]) -> bool:
    return (
        row.get("corrected_feature41_replay_pass") is True
        and row.get("held_target_or_error_used_for_prediction") is False
        and row.get("held_score_computed") is False
        and row.get("partial_score_inspected") is False
        and row.get("model_or_threshold_selection_performed") is False
        and row.get("external_outcome_accessed") is False
    )


def merge_folds(input_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted(input_dir.glob("post_v13_diag_fold_result_fold*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != "PV13D2":
            raise ValueError(f"invalid post-V13 fold result {path}")
        fold = int(row.get("outer_fold", -1))
        if fold in seen:
            raise ValueError(f"duplicate post-V13 fold {fold}")
        seen.add(fold)
        if not _recorded_invariants_pass(row):
            raise ValueError(f"post-V13 fold {fold} lacks frozen invariants")
        for field in (
            "prediction_artifact",
            "model_artifact",
            "corrected_feature41_reference",
        ):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        _validate_prediction(
            Path(row["prediction_artifact"]),
            fold,
            int(row["n_registered_prediction_rows"]),
        )
        rows.append(row)
    expected = set(range(20))
    if seen != expected or len(rows) != 20:
        raise ValueError(
            f"post-V13 fold universe incomplete: found={sorted(seen)}"
        )
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "PV13D2",
        "status": "PV13D2_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": rows,
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_folds": True,
            "prediction_only_fields": True,
            "corrected_feature41_replay_all_folds": True,
            "held_scores_absent": True,
            "partial_score_inspected": False,
            "model_or_threshold_selection_performed": False,
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
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
