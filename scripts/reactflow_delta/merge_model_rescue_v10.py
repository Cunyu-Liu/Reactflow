#!/usr/bin/env python3
"""Merge the complete V10M2 prediction-only fold universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.model_rescue_v10 import FOLD_SCHEMA
from scripts.reactflow_delta.qualify_model_rescue_v10_smoke import (
    prediction_checks,
    recorded_invariants_pass,
)


SCHEMA = "reactflow_delta.model_rescue_v10_merged.v1"


def merge_folds(input_dir: Path) -> dict[str, Any]:
    paths = sorted(input_dir.glob("v10_fold_result_fold*_seed0.json"))
    rows = []
    seen = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        fold = int(row.get("outer_fold", -1))
        if fold in seen:
            raise ValueError(f"duplicate V10 fold {fold}")
        seen.add(fold)
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != "V10M2":
            raise ValueError(f"invalid V10M2 fold result in {path}")
        if int(row.get("seed", -1)) != 0 or int(row.get("epochs", -1)) != 40:
            raise ValueError(f"V10 fold {fold} violates seed or epoch freeze")
        if not recorded_invariants_pass(row.get("invariants", {})):
            raise ValueError(f"V10 fold {fold} lacks scientific invariants")
        checks = prediction_checks(
            Path(row["prediction_artifact"]),
            fold,
            int(row["n_registered_prediction_rows"]),
        )
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(f"V10 fold {fold} prediction checks failed: {failed}")
        if row.get("parameter_counts") != {
            "feature41_symmetric": 63491,
            "feature41_asymmetric": 63748,
            "meanaligned_symmetric": 63491,
            "meanaligned_asymmetric": 63748,
        }:
            raise ValueError(f"V10 fold {fold} parameter counts changed")
        if set(row.get("checkpoints", {})) != {
            "feature41_symmetric",
            "feature41_asymmetric",
            "meanaligned_symmetric",
            "meanaligned_asymmetric",
        }:
            raise ValueError(f"V10 fold {fold} checkpoint family changed")
        if not all(Path(value).is_file() for value in row["checkpoints"].values()):
            raise FileNotFoundError(f"V10 fold {fold} lacks a checkpoint")
        rows.append(row)
    if seen != set(range(20)) or len(rows) != 20:
        raise ValueError(f"V10 fold universe is incomplete: found {sorted(seen)}")
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "V10M2",
        "status": "V10M2_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": rows,
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_folds": True,
            "prediction_only_schema": True,
            "target_identity_exact": True,
            "v8_point_replay_all_folds": True,
            "tic2a_feature41_replay_all_folds": True,
            "historical_v9_replay_all_folds": True,
            "matched_head_families_all_folds": True,
            "median_constraint_all_folds": True,
            "outer_train_only_standardization_all_folds": True,
            "partial_scores_inspected": False,
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
