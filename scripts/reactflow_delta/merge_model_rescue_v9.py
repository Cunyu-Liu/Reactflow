#!/usr/bin/env python3
"""Merge the complete V9M2 prediction-only fold universe before scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.reactflow_delta.model_rescue_v9 import FOLD_SCHEMA
from scripts.reactflow_delta.qualify_model_rescue_v9_smoke import _prediction_checks


SCHEMA = "reactflow_delta.model_rescue_v9_merged.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def merge_folds(input_dir: Path) -> dict[str, Any]:
    paths = sorted(input_dir.glob("v9_fold_result_fold*_seed0.json"))
    rows = []
    seen = set()
    for path in paths:
        row = _read(path)
        fold = int(row.get("outer_fold", -1))
        if fold in seen:
            raise ValueError(f"duplicate V9 fold {fold}")
        seen.add(fold)
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != "V9M2":
            raise ValueError(f"invalid V9M2 fold result in {path}")
        if int(row.get("seed", -1)) != 0 or int(
            row.get("calibration_epochs", -1)
        ) != 40:
            raise ValueError(f"V9M2 fold {fold} violates seed or epoch freeze")
        invariants = row.get("invariants", {})
        if not invariants or not all(value is True for value in invariants.values()):
            raise ValueError(f"V9M2 fold {fold} lacks all scientific invariants")
        prediction = Path(row["prediction_artifact"])
        checks = _prediction_checks(
            prediction, fold, int(row["n_registered_prediction_rows"])
        )
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(f"V9M2 fold {fold} prediction checks failed: {failed}")
        checkpoints = [
            Path(row["baseline_calibration_checkpoint"]),
            Path(row["candidate_calibration_checkpoint"]),
        ]
        if not all(path.is_file() and path.stat().st_size > 0 for path in checkpoints):
            raise FileNotFoundError(f"V9M2 fold {fold} lacks calibration checkpoint")
        states = [torch.load(path, map_location="cpu") for path in checkpoints]
        if states[0].keys() != states[1].keys() or any(
            states[0][name].shape != states[1][name].shape for name in states[0]
        ):
            raise ValueError(f"V9M2 fold {fold} residual head schemas differ")
        rows.append(row)
    if seen != set(range(20)) or len(rows) != 20:
        raise ValueError(f"V9M2 fold universe is incomplete: found {sorted(seen)}")
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "V9M2",
        "status": "V9M2_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": rows,
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_folds": True,
            "prediction_only_schema": True,
            "target_identity_exact": True,
            "v8_mean_replay_all_folds": True,
            "tic2a_feature41_replay_all_folds": True,
            "identical_residual_family_all_folds": True,
            "zero_mean_residual_all_folds": True,
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
