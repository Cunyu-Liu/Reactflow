#!/usr/bin/env python3
"""Qualify V11 real-data smoke without reading scientific outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
import torch

from scripts.reactflow_delta.model_rescue_v11 import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_model_rescue_v11 import (
    EXPECTED_POINT_PARAMETERS,
    FOLD_SCHEMA,
    POINT_NAMES,
)


SCHEMA = "reactflow_delta.model_rescue_v11_smoke_qualification.v1"


def recorded_invariants_pass(invariants: dict[str, Any]) -> bool:
    required_true = (
        "target_profile_identity_exact",
        "exact_point_parameter_match",
        "fixed_skip_only_model_difference",
        "same_point_training_order_and_dropout_stream",
        "point_frozen_during_calibration",
        "v10_residual_family_reused",
        "feature41_replay_at_1e_7",
        "feature41_asymmetric_screen_replay_or_not_applicable",
        "median_constraint_all_held_rows",
    )
    required_false = (
        "held_score_computed",
        "prediction_contains_target_fields",
        "external_outcome_accessed",
    )
    return all(invariants.get(name) is True for name in required_true) and all(
        invariants.get(name) is False for name in required_false
    )


def checkpoint_standardizer_pass(state: dict[str, Any]) -> bool:
    mean = np.asarray(state.get("standardizer_mean"))
    scale = np.asarray(state.get("standardizer_scale"))
    return bool(
        mean.shape == (244,)
        and scale.shape == (244,)
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and (scale > 0.0).all()
    )


def prediction_checks(
    path: Path, *, fold: int, seed: int, expected_rows: int
) -> dict[str, bool]:
    with np.load(path, allow_pickle=True) as handle:
        names = set(handle.files)
        schema = str(handle["schema_version"].item())
        keys = list(map(str, handle["keys"]))
        points = {
            "feature41": np.asarray(handle["feature41_point"]),
            "anchored": np.asarray(handle["anchored_point"]),
            "unanchored": np.asarray(handle["unanchored_point"]),
            "v8": np.asarray(handle["v8_point"]),
        }
        checks: dict[str, bool] = {
            "schema": schema == PREDICTION_SCHEMA,
            "row_count": len(keys) == expected_rows,
            "unique_keys": len(keys) == len(set(keys)),
            "fold": set(map(int, handle["outer_fold"])) == {fold},
            "seed": set(map(int, handle["seed"])) == {seed},
            "registered_covered": bool(
                (np.asarray(handle["registered_status"]) == "covered").all()
            ),
            "no_target_or_score_fields": not any(
                token in name.lower()
                for name in names
                for token in (
                    "target",
                    "qualified_mask",
                    "target_error",
                    "score",
                    "crps",
                    "mae",
                )
            ),
            "finite_points": all(np.isfinite(value).all() for value in points.values()),
        }
        for name in POINT_NAMES:
            weights = np.asarray(handle[f"{name}_weights"])
            locations = np.asarray(handle[f"{name}_locations"])
            scales = np.asarray(handle[f"{name}_scales"])
            expected_absolute = np.asarray(
                handle[f"{name}_expected_absolute_delta"]
            )
            checks[f"{name}_shape"] = (
                weights.shape == (expected_rows, 2)
                and locations.shape == (expected_rows, 2)
                and scales.shape == (expected_rows, 2)
                and expected_absolute.shape == (expected_rows,)
            )
            checks[f"{name}_finite"] = bool(
                np.isfinite(weights).all()
                and np.isfinite(locations).all()
                and np.isfinite(scales).all()
                and np.isfinite(expected_absolute).all()
            )
            checks[f"{name}_valid_distribution"] = bool(
                np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0)
                and (weights > 0.0).all()
                and (scales > 0.0).all()
            )
            cdf = np.sum(
                weights
                * ndtr((points[name][:, None] - locations) / scales),
                axis=1,
            )
            checks[f"{name}_median_constraint"] = bool(
                np.allclose(cdf, 0.5, atol=3e-6, rtol=0.0)
            )
        for prefix in (
            "historical_v10_weights",
            "historical_v10_locations",
            "historical_v10_scales",
            "historical_v10_expected_absolute_delta",
        ):
            checks[f"{prefix}_finite"] = bool(
                np.isfinite(np.asarray(handle[prefix])).all()
            )
    return checks


def qualify(input_dir: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    rows = []
    for fold in (0, 1):
        path = input_dir / f"v11_fold_result_fold{fold}_seed0.json"
        if not path.is_file():
            checks[f"fold{fold}_result_exists"] = False
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
        checks[f"fold{fold}_schema"] = row.get("schema_version") == FOLD_SCHEMA
        checks[f"fold{fold}_phase"] = row.get("phase") == "V11M2"
        checks[f"fold{fold}_seed_epochs"] = (
            int(row.get("seed", -1)) == 0
            and int(row.get("point_epochs", -1)) == 3
            and int(row.get("calibration_epochs", -1)) == 3
        )
        checks[f"fold{fold}_invariants"] = recorded_invariants_pass(
            row.get("invariants", {})
        )
        counts = row.get("point_parameter_counts", {})
        checks[f"fold{fold}_point_parameter_match"] = (
            sorted(counts.values()) == [EXPECTED_POINT_PARAMETERS] * 2
        )
        checks[f"fold{fold}_residual_parameter_counts"] = (
            set(row.get("residual_parameter_counts", {}).values()) == {63748}
        )
        for name, passed in prediction_checks(
            Path(row["prediction_artifact"]),
            fold=fold,
            seed=0,
            expected_rows=int(row["n_registered_prediction_rows"]),
        ).items():
            checks[f"fold{fold}_prediction_{name}"] = passed
        for name, checkpoint in row.get("point_checkpoints", {}).items():
            checkpoint_path = Path(checkpoint)
            checks[f"fold{fold}_{name}_point_checkpoint"] = (
                checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
            )
        for name, checkpoint in row.get("residual_checkpoints", {}).items():
            checkpoint_path = Path(checkpoint)
            exists = checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
            checks[f"fold{fold}_{name}_residual_checkpoint"] = exists
            if exists:
                checks[f"fold{fold}_{name}_standardizer"] = (
                    checkpoint_standardizer_pass(
                        torch.load(checkpoint_path, map_location="cpu")
                    )
                )
    passed = len(rows) == 2 and all(checks.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V11M2",
        "status": (
            "V11M2_ENGINEERING_SMOKE_PASS"
            if passed
            else "V11M2_ENGINEERING_SMOKE_FAIL"
        ),
        "passed": passed,
        "checks": checks,
        "scientific_scores_read": False,
        "external_outcome_accessed": False,
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
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
