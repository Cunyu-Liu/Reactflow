#!/usr/bin/env python3
"""Qualify the V9 two-fold engineering smoke without reading scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v9 import FOLD_SCHEMA, PREDICTION_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v9_smoke_qualification.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _prediction_checks(path: Path, fold_id: int, expected_rows: int) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    with np.load(path, allow_pickle=True) as handle:
        fields = set(handle.files)
        required = {
            "schema_version",
            "keys",
            "biological_scoring_key",
            "outer_fold",
            "seed",
            "registered_status",
            "feature41_delta_mean",
            "feature41_weights",
            "feature41_locations",
            "feature41_scales",
            "feature41_expected_absolute_delta",
            "meanaligned_delta_mean",
            "meanaligned_weights",
            "meanaligned_locations",
            "meanaligned_scales",
            "meanaligned_expected_absolute_delta",
        }
        forbidden = {
            "target",
            "target_error",
            "target_mask",
            "qualified_target_mask",
            "score",
            "mae",
            "crps",
        }
        checks["prediction_schema"] = (
            str(handle["schema_version"].item()) == PREDICTION_SCHEMA
        )
        checks["required_prediction_fields"] = required.issubset(fields)
        checks["target_and_score_fields_absent"] = forbidden.isdisjoint(fields)
        keys = handle["keys"]
        checks["row_count"] = len(keys) == expected_rows
        checks["biological_keys_equal"] = np.array_equal(
            keys, handle["biological_scoring_key"]
        )
        checks["biological_keys_unique"] = len(set(map(str, keys))) == len(keys)
        checks["fold_and_seed"] = set(map(int, handle["outer_fold"])) == {
            fold_id
        } and set(map(int, handle["seed"])) == {0}
        checks["registered_covered"] = set(map(str, handle["registered_status"])) == {
            "covered"
        }
        for prefix in ("feature41", "meanaligned"):
            mean = handle[f"{prefix}_delta_mean"]
            weights = handle[f"{prefix}_weights"]
            locations = handle[f"{prefix}_locations"]
            scales = handle[f"{prefix}_scales"]
            magnitude = handle[f"{prefix}_expected_absolute_delta"]
            checks[f"{prefix}_shapes"] = (
                mean.shape == (len(keys),)
                and weights.shape == (len(keys), 2)
                and locations.shape == (len(keys), 2)
                and scales.shape == (len(keys), 2)
                and magnitude.shape == (len(keys),)
            )
            checks[f"{prefix}_finite"] = all(
                np.isfinite(value).all()
                for value in (mean, weights, locations, scales, magnitude)
            )
            checks[f"{prefix}_weights"] = np.allclose(
                weights.sum(-1), 1.0, atol=1e-7, rtol=0.0
            )
            checks[f"{prefix}_positive_scales"] = bool((scales > 0).all())
            checks[f"{prefix}_locations_equal_mean"] = np.allclose(
                locations, mean[:, None], atol=1e-7, rtol=0.0
            )
            checks[f"{prefix}_expected_abs_ge_abs_mean"] = bool(
                (magnitude + 1e-7 >= np.abs(mean)).all()
            )
    return checks


def qualify(input_dir: Path) -> dict[str, Any]:
    rows = []
    for fold_id in (0, 1):
        path = input_dir / f"v9_fold_result_fold{fold_id}_seed0.json"
        row = _read_json(path)
        checks = {
            "fold_schema": row.get("schema_version") == FOLD_SCHEMA,
            "phase": row.get("phase") == "V9M1",
            "evidence_status": row.get("evidence_status")
            == "ENGINEERING_SMOKE_ONLY",
            "fold_seed_epochs": int(row.get("outer_fold", -1)) == fold_id
            and int(row.get("seed", -1)) == 0
            and int(row.get("calibration_epochs", -1)) == 3,
        }
        invariants = row.get("invariants", {})
        checks["all_recorded_invariants"] = bool(invariants) and all(
            value is True for value in invariants.values()
        )
        checkpoints = [
            Path(row.get("baseline_calibration_checkpoint", "")),
            Path(row.get("candidate_calibration_checkpoint", "")),
        ]
        checks["calibration_checkpoints_exist"] = all(
            path.is_file() and path.stat().st_size > 0 for path in checkpoints
        )
        if checks["calibration_checkpoints_exist"]:
            states = [torch.load(path, map_location="cpu") for path in checkpoints]
            checks["identical_head_parameter_schema"] = (
                states[0].keys() == states[1].keys()
                and all(states[0][name].shape == states[1][name].shape for name in states[0])
            )
        else:
            checks["identical_head_parameter_schema"] = False
        prediction_path = Path(row.get("prediction_artifact", ""))
        checks["prediction_artifact_exists"] = (
            prediction_path.is_file() and prediction_path.stat().st_size > 0
        )
        if checks["prediction_artifact_exists"]:
            checks.update(
                _prediction_checks(
                    prediction_path,
                    fold_id,
                    int(row.get("n_registered_prediction_rows", -1)),
                )
            )
        rows.append(
            {
                "outer_fold": fold_id,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passed = len(rows) == 2 and all(row["passed"] for row in rows)
    return {
        "schema_version": SCHEMA,
        "status": "V9M1_ENGINEERING_SMOKE_PASS" if passed else "V9M1_ENGINEERING_SMOKE_FAIL",
        "folds": rows,
        "scores_read": False,
        "scientific_gate_evaluated": False,
        "external_outcome_accessed": False,
        "v9m2_authorized": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(args.input_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0 if result["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
