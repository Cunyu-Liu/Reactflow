#!/usr/bin/env python3
"""Merge the complete V8M1 prediction-only expert universe before scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.run_model_rescue_v8_expert_rebuild import (
    PREDICTION_SCHEMA,
    SCHEMA as FOLD_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v8_mean_screen_merged.v1"
QUALIFICATION_SCHEMA = (
    "reactflow_delta.model_rescue_v8_corrected_expert_qualification.v1"
)
EXPECTED_FOLDS = list(range(20))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _validate_prediction(path: Path, fold: int) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as handle:
        fields = set(handle.files)
        required = {
            "schema_version",
            "keys",
            "b1_delta_mean",
            "meanaligned_delta_mean",
            "outer_fold",
            "seed",
        }
        if not required.issubset(fields):
            raise ValueError(f"V8 prediction fields are incomplete in {path}")
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
            raise ValueError(f"V8 prediction contains target-side fields in {path}")
        if str(handle["schema_version"].item()) != PREDICTION_SCHEMA:
            raise ValueError(f"invalid V8 prediction schema in {path}")
        keys = handle["keys"]
        if len(set(map(str, keys))) != len(keys):
            raise ValueError(f"V8 prediction keys are duplicated in {path}")
        if set(map(int, handle["outer_fold"])) != {fold}:
            raise ValueError(f"V8 prediction outer fold differs in {path}")
        if set(map(int, handle["seed"])) != {0}:
            raise ValueError(f"V8M2 requires seed 0 in {path}")
        for field in ("b1_delta_mean", "meanaligned_delta_mean"):
            values = handle[field]
            if values.shape != (len(keys),) or not np.isfinite(values).all():
                raise ValueError(f"V8 prediction {field} is invalid in {path}")
        return len(keys)


def merge_folds(input_dir: Path, qualification_json: Path) -> dict[str, Any]:
    qualification = _read_json(qualification_json)
    if qualification.get("schema_version") != QUALIFICATION_SCHEMA:
        raise ValueError("V8M2 merge requires the V8M1 qualification schema")
    if qualification.get("status") != "V8M1_CORRECTED_EXPERT_REBUILD_PASS":
        raise ValueError("V8M2 merge requires exact V8M1 PASS")
    if qualification.get("target_profile_identity_exact") is not True:
        raise ValueError("V8M2 merge requires exact target identity")
    if qualification.get("scores_read") is not False:
        raise ValueError("V8M1 qualification must be score blind")

    paths = sorted(
        input_dir.glob("v8_corrected_expert_fold_result_fold*_seed0.json")
    )
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        row = _read_json(path)
        fold = int(row.get("outer_fold", -1))
        if fold in seen:
            raise ValueError(f"duplicate V8 fold {fold}")
        seen.add(fold)
        if row.get("schema_version") != FOLD_SCHEMA:
            raise ValueError(f"invalid V8 fold schema in {path}")
        if int(row.get("seed", -1)) != 0 or int(row.get("epochs", -1)) != 40:
            raise ValueError(f"V8 fold {fold} violates frozen seed or epoch")
        if row.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
            raise ValueError(f"V8 fold {fold} lacks exact target identity")
        if int(row.get("canonical_mutant_full_profiles", -1)) != 13976:
            raise ValueError(f"V8 fold {fold} lacks the canonical profile universe")
        required_false = (
            "held_score_computed",
            "external_outcome_accessed",
            "legacy_v3_checkpoint_reused",
            "legacy_v3_prediction_reused",
        )
        if any(row.get(name) is not False for name in required_false):
            raise ValueError(f"V8 fold {fold} violates prediction-only isolation")
        for name in ("b1_checkpoint", "meanaligned_checkpoint"):
            artifact = Path(row[name])
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise FileNotFoundError(artifact)
        prediction = Path(row["expert_prediction_artifact"])
        n_rows = _validate_prediction(prediction, fold)
        row["n_registered_prediction_rows"] = n_rows
        rows.append(row)

    if seen != set(EXPECTED_FOLDS) or len(rows) != 20:
        raise ValueError(f"V8 fold universe is incomplete: found {sorted(seen)}")
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "V8M2",
        "status": "V8M2_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": rows,
        "v8m1_qualification_artifact": str(qualification_json),
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_folds": True,
            "prediction_schema_valid": True,
            "prediction_only_fields": True,
            "target_identity_exact": True,
            "fresh_checkpoints_all_folds": True,
            "legacy_v3_reuse": False,
            "held_scores_absent": True,
            "partial_scores_inspected": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--qualification-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_folds(args.input_dir, args.qualification_json)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
