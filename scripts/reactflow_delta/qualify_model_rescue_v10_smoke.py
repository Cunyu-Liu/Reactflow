#!/usr/bin/env python3
"""Qualify V10 real-data smoke artifacts without reading scientific scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
import torch

from scripts.reactflow_delta.model_rescue_v10 import (
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v10_smoke_qualification.v1"
HEAD_NAMES = (
    "feature41_symmetric",
    "feature41_asymmetric",
    "meanaligned_symmetric",
    "meanaligned_asymmetric",
)


def recorded_invariants_pass(invariants: dict[str, Any]) -> bool:
    required_true = (
        "target_profile_identity_exact",
        "v8_point_replay_at_1e_7",
        "tic2a_feature41_replay_at_1e_7",
        "outer_train_only_standardization",
        "trained_v8_direct_features_only",
        "fair_feature41_and_meanaligned_head_families",
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


def prediction_checks(path: Path, fold: int, expected_rows: int) -> dict[str, bool]:
    with np.load(path, allow_pickle=True) as handle:
        names = set(handle.files)
        schema = str(handle["schema_version"].item())
        keys = list(map(str, handle["keys"]))
        outer_fold = set(map(int, handle["outer_fold"]))
        seed = set(map(int, handle["seed"]))
        status = np.asarray(handle["registered_status"])
        feature41_point = np.asarray(handle["feature41_point"])
        meanaligned_point = np.asarray(handle["meanaligned_point"])
        checks: dict[str, bool] = {
            "schema": schema == PREDICTION_SCHEMA,
            "row_count": len(keys) == expected_rows,
            "unique_keys": len(keys) == len(set(keys)),
            "fold": outer_fold == {fold},
            "seed": seed == {0},
            "registered_covered": bool((status == "covered").all()),
            "no_target_fields": not any(
                token in name.lower()
                for name in names
                for token in ("target", "qualified_mask", "target_error")
            ),
            "finite_points": bool(
                np.isfinite(feature41_point).all()
                and np.isfinite(meanaligned_point).all()
            ),
        }
        for head in HEAD_NAMES:
            weights = np.asarray(handle[f"{head}_weights"])
            locations = np.asarray(handle[f"{head}_locations"])
            scales = np.asarray(handle[f"{head}_scales"])
            expected_abs = np.asarray(handle[f"{head}_expected_absolute_delta"])
            point = feature41_point if head.startswith("feature41") else meanaligned_point
            checks[f"{head}_shape"] = (
                weights.shape == (expected_rows, 2)
                and locations.shape == (expected_rows, 2)
                and scales.shape == (expected_rows, 2)
                and expected_abs.shape == (expected_rows,)
            )
            checks[f"{head}_finite"] = bool(
                np.isfinite(weights).all()
                and np.isfinite(locations).all()
                and np.isfinite(scales).all()
                and np.isfinite(expected_abs).all()
            )
            checks[f"{head}_distribution"] = bool(
                np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0)
                and (weights > 0.0).all()
                and (scales > 0.0).all()
            )
            if head.endswith("symmetric") and not head.endswith("asymmetric"):
                checks[f"{head}_locations_equal_point"] = bool(
                    np.array_equal(locations[:, 0], point)
                    and np.array_equal(locations[:, 1], point)
                )
            else:
                cdf = np.sum(
                    weights * ndtr((point[:, None] - locations) / scales), axis=1
                )
                checks[f"{head}_median_constraint"] = bool(
                    np.allclose(cdf, 0.5, atol=3e-6, rtol=0.0)
                )
    return checks


def qualify(input_dir: Path) -> dict[str, Any]:
    rows = []
    checks: dict[str, bool] = {}
    for fold in (0, 1):
        path = input_dir / f"v10_fold_result_fold{fold}_seed0.json"
        if not path.is_file():
            checks[f"fold{fold}_result_exists"] = False
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
        checks[f"fold{fold}_schema"] = row.get("schema_version") == FOLD_SCHEMA
        checks[f"fold{fold}_phase"] = row.get("phase") == "V10M1"
        checks[f"fold{fold}_seed_epoch"] = (
            int(row.get("seed", -1)) == 0 and int(row.get("epochs", -1)) == 3
        )
        checks[f"fold{fold}_invariants"] = recorded_invariants_pass(
            row.get("invariants", {})
        )
        checks[f"fold{fold}_parameter_counts"] = row.get("parameter_counts") == {
            "feature41_symmetric": 63491,
            "feature41_asymmetric": 63748,
            "meanaligned_symmetric": 63491,
            "meanaligned_asymmetric": 63748,
        }
        for name, passed in prediction_checks(
            Path(row["prediction_artifact"]),
            fold,
            int(row["n_registered_prediction_rows"]),
        ).items():
            checks[f"fold{fold}_prediction_{name}"] = passed
        for name, checkpoint in row.get("checkpoints", {}).items():
            checkpoint_path = Path(checkpoint)
            checks[f"fold{fold}_{name}_checkpoint"] = (
                checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
            )
            if checks[f"fold{fold}_{name}_checkpoint"]:
                state = torch.load(checkpoint_path, map_location="cpu")
                checks[f"fold{fold}_{name}_standardizer"] = (
                    np.asarray(state["standardizer_mean"]).shape == (244,)
                    and np.asarray(state["standardizer_scale"]).shape == (244,)
                    and np.isfinite(state["standardizer_mean"]).all()
                    and (np.asarray(state["standardizer_scale"]) > 0.0).all()
                )
    passed = len(rows) == 2 and all(checks.values())
    return {
        "schema_version": SCHEMA,
        "phase": "V10M1",
        "status": (
            "V10M1_ENGINEERING_SMOKE_PASS"
            if passed
            else "V10M1_ENGINEERING_SMOKE_FAIL"
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
