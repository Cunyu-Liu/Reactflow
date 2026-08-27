#!/usr/bin/env python3
"""Strict, target-free source binding for the future Puzzle-Set runtime.

This module deliberately accepts only the small TIC2A registry/model projection
needed to construct feature41 and an explicit activation-time source manifest.
It never opens TIC2A prediction artifacts or historical wide result JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.reactflow_delta.model_rescue_v6_probe import (
    CANDIDATE_PROBE_FEATURE_NAMES,
)


EXPECTED_FOLDS = tuple(range(20))
EXPECTED_CONTRACT_ID = "reactflow_delta_puzzle_set_meta_context_v5_20260827"

TIC2A_MERGED_SCHEMA = "reactflow_delta.target_identity_corrected_baseline_merged.v1"
TIC2A_FOLD_SCHEMA = "reactflow_delta.target_identity_corrected_baseline_fold.v1"
TIC2A_MERGED_STATUS = "TIC2A_COMPLETE_UNSCORED_MERGE_PASS"
TIC2A_MERGE_INTEGRITY = {
    "complete_fold_universe": True,
    "external_outcome_accessed": False,
    "held_scores_absent": True,
    "legacy_prediction_reused": False,
    "partial_score_inspected": False,
    "prediction_only_fields": True,
    "prediction_schema_valid": True,
    "referenced_artifacts_exist": True,
    "target_identity_exact": True,
    "unique_folds": True,
    "v5_v6_feature30_replay_all_folds": True,
}
TIC2A_MERGED_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "merge_integrity",
    "folds",
}
TIC2A_FOLD_FIELDS = {
    "schema_version",
    "phase",
    "outer_fold",
    "held_puzzle",
    "target_profile_identity",
    "held_target_used_for_prediction",
    "held_score_computed",
    "partial_score_inspected",
    "legacy_prediction_reused",
    "external_outcome_accessed",
    "model_artifact",
    "prediction_artifact",
    "n_registered_prediction_rows",
    "n_train_cells",
    "n_train_qualified_positions",
    "n_train_valid_mutants",
    "v5_v6_feature30_prediction_replay_pass",
    "v5_v6_feature30_stats_replay_pass",
}
TIC2A_MODEL_FIELDS = {
    "direct18",
    "direct18_feature_names",
    "feature30_feature_names",
    "feature41_feature_names",
    "ridge_alpha",
    "v5_feature30",
    "v6_feature30_replay",
    "v6_feature41",
}
TIC2A_RIDGE_FIELDS = {"mean_x", "scale_x", "mean_y", "coefficient", "alpha"}

SOURCE_MANIFEST_SCHEMA = "reactflow_delta.puzzle_set_meta_context_source_manifest.v1"
SOURCE_MANIFEST_STATUS = "PUZZLE_SET_SOURCE_MANIFEST_BOUND"
SOURCE_BINDING_STATUS = "REALIZED_PATHS_ROLES_AND_COUNTS_BOUND"
SOURCE_MANIFEST_TOP_FIELDS = {
    "schema_version",
    "status",
    "contract_id",
    "binding_status",
    "folds",
}
SOURCE_MANIFEST_FOLD_FIELDS = {"outer_fold", "held_puzzle", "seed", "sources"}
SOURCE_MANIFEST_RECORD_FIELDS = {
    "path",
    "role",
    "used_in_candidate_prediction",
    "outer_fold",
    "seed",
    "realized_parameter_count",
    "trainable_in_p1",
}

FROZEN_PARENT_SEED = 0
FROZEN_INPUT_SOURCE_SPEC = {
    "v13_point_checkpoint": {
        "role": "FROZEN_SAME_FOLD_POINT_ANCHOR",
        "used_in_candidate_prediction": True,
        "seed": FROZEN_PARENT_SEED,
        "realized_parameter_count": 2_064_737,
        "trainable_in_p1": False,
    },
    "v14_encoder_checkpoint": {
        "role": "FROZEN_SAME_FOLD_OUTCOME_BLIND_ENCODER",
        "used_in_candidate_prediction": True,
        "seed": FROZEN_PARENT_SEED,
        "realized_parameter_count": 4_767_280,
        "trainable_in_p1": False,
    },
    "v8_meanaligned_checkpoint": {
        "role": "FROZEN_SAME_FOLD_201D_CALIBRATION_FEATURE_GENERATOR",
        "used_in_candidate_prediction": True,
        "seed": FROZEN_PARENT_SEED,
        "realized_parameter_count": 109_581,
        "trainable_in_p1": False,
    },
    "tic2a_feature41_model_artifact": {
        "role": "FROZEN_OUTER_FOLD_FEATURE41_RIDGE_AND_41D_BASIS",
        "used_in_candidate_prediction": True,
        "seed": None,
        "realized_parameter_count": 84,
        "trainable_in_p1": False,
    },
    "tic2a_merged_registry": {
        "role": "FROZEN_COMPLETE_TWENTY_FOLD_SOURCE_REGISTRY",
        "used_in_candidate_prediction": False,
        "seed": None,
        "realized_parameter_count": 0,
        "trainable_in_p1": False,
    },
    "unconstrained_feature_cache": {
        "role": "FROZEN_OUTCOME_BLIND_ENSEMBLE_FEATURE_CACHE",
        "used_in_candidate_prediction": True,
        "seed": None,
        "realized_parameter_count": 0,
        "trainable_in_p1": False,
    },
    "constrained_feature_cache": {
        "role": "FROZEN_OUTCOME_BLIND_CONSTRAINED_FEATURE_CACHE",
        "used_in_candidate_prediction": True,
        "seed": None,
        "realized_parameter_count": 0,
        "trainable_in_p1": False,
    },
}
FOLD_SCOPED_INPUT_SOURCES = {
    "v13_point_checkpoint",
    "v14_encoder_checkpoint",
    "v8_meanaligned_checkpoint",
    "tic2a_feature41_model_artifact",
}


@dataclass(frozen=True)
class SafeTIC2AFold:
    """Validated target-free source projection for one outer fold."""

    row: dict[str, Any]
    model_path: Path
    feature41_model: dict[str, np.ndarray | float]
    ridge_parameter_counts: dict[str, int]


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def validate_feature41_ridge(
    value: Any,
) -> tuple[dict[str, np.ndarray | float], dict[str, int]]:
    """Validate and normalize the exact 41D/two-output weighted-ridge state."""

    if not isinstance(value, dict) or set(value) != TIC2A_RIDGE_FIELDS:
        raise ValueError("TIC2A feature41 ridge state fields changed")
    arrays = {
        "mean_x": np.asarray(value["mean_x"], dtype=np.float64),
        "scale_x": np.asarray(value["scale_x"], dtype=np.float64),
        "mean_y": np.asarray(value["mean_y"], dtype=np.float64),
        "coefficient": np.asarray(value["coefficient"], dtype=np.float64),
    }
    try:
        alpha = float(value["alpha"])
    except (TypeError, ValueError):
        raise ValueError("TIC2A feature41 ridge alpha changed") from None
    if (
        arrays["mean_x"].shape != (41,)
        or arrays["scale_x"].shape != (41,)
        or arrays["mean_y"].shape != (2,)
        or arrays["coefficient"].shape != (41, 2)
        or alpha != 1.0
        or np.any(arrays["scale_x"] <= 0)
        or not all(np.isfinite(array).all() for array in arrays.values())
    ):
        raise ValueError("TIC2A feature41 ridge shape or value changed")
    normalized: dict[str, np.ndarray | float] = {**arrays, "alpha": alpha}
    counts = {
        "predictive_parameter_count": int(
            arrays["coefficient"].size + arrays["mean_y"].size
        ),
        "stored_fitted_scalar_count": int(sum(array.size for array in arrays.values())),
    }
    return normalized, counts


def validate_tic2a_safe_registry(
    merged: dict[str, Any],
) -> dict[int, SafeTIC2AFold]:
    """Validate the exact target-free TIC2A registry and safe model artifacts."""

    if (
        set(merged) != TIC2A_MERGED_FIELDS
        or merged.get("schema_version") != TIC2A_MERGED_SCHEMA
        or merged.get("phase") != "TIC2A"
        or merged.get("status") != TIC2A_MERGED_STATUS
        or merged.get("merge_integrity") != TIC2A_MERGE_INTEGRITY
    ):
        raise ValueError("TIC2A merged registry schema or integrity changed")
    rows = merged.get("folds")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FOLDS):
        raise ValueError("TIC2A merged registry must contain exactly twenty folds")
    try:
        by_fold = {int(row["outer_fold"]): row for row in rows}
    except (KeyError, TypeError, ValueError):
        raise ValueError("TIC2A merged registry fold identity is malformed") from None
    if tuple(sorted(by_fold)) != EXPECTED_FOLDS or len(by_fold) != len(rows):
        raise ValueError("TIC2A merged registry is not unique folds0-19")

    validated: dict[int, SafeTIC2AFold] = {}
    for fold in EXPECTED_FOLDS:
        row = by_fold[fold]
        held = f"P{fold + 1:02d}"
        if (
            not isinstance(row, dict)
            or set(row) != TIC2A_FOLD_FIELDS
            or row.get("schema_version") != TIC2A_FOLD_SCHEMA
            or row.get("phase") != "TIC2A"
            or row.get("outer_fold") != fold
            or row.get("held_puzzle") != held
            or row.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION"
            or row.get("held_target_used_for_prediction") is not False
            or row.get("held_score_computed") is not False
            or row.get("partial_score_inspected") is not False
            or row.get("legacy_prediction_reused") is not False
            or row.get("external_outcome_accessed") is not False
            or row.get("v5_v6_feature30_prediction_replay_pass") is not True
            or row.get("v5_v6_feature30_stats_replay_pass") is not True
        ):
            raise ValueError(f"fold {fold} TIC2A safe registry row changed")
        for count_field in (
            "n_registered_prediction_rows",
            "n_train_cells",
            "n_train_qualified_positions",
            "n_train_valid_mutants",
        ):
            if not isinstance(row.get(count_field), int) or row[count_field] < 0:
                raise ValueError(f"fold {fold} TIC2A {count_field} changed")
        prediction_path = Path(str(row.get("prediction_artifact", "")))
        model_path = Path(str(row.get("model_artifact", "")))
        if not prediction_path.is_absolute():
            raise ValueError(f"fold {fold} TIC2A prediction path is not absolute")
        if (
            not model_path.is_absolute()
            or model_path.name != f"tic2a_corrected_models_fold{fold}.json"
            or not model_path.is_file()
        ):
            raise FileNotFoundError(f"fold {fold} TIC2A safe model artifact is absent")
        model = read_json_object(model_path)
        if (
            set(model) != TIC2A_MODEL_FIELDS
            or tuple(model.get("feature41_feature_names", ()))
            != tuple(CANDIDATE_PROBE_FEATURE_NAMES)
            or model.get("ridge_alpha") != 1.0
        ):
            raise ValueError(f"fold {fold} TIC2A safe model identity changed")
        feature41, counts = validate_feature41_ridge(model.get("v6_feature41"))
        validated[fold] = SafeTIC2AFold(
            row=dict(row),
            model_path=model_path,
            feature41_model=feature41,
            ridge_parameter_counts=counts,
        )
    return validated


def load_tic2a_safe_registry(path: Path) -> dict[int, SafeTIC2AFold]:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError("TIC2A safe merged registry must be one absolute file")
    return validate_tic2a_safe_registry(read_json_object(path))


def _expected_filename(source_id: str, fold: int) -> str | None:
    return {
        "v13_point_checkpoint": f"v13_candidate_point_fold{fold}_seed0.pt",
        "v14_encoder_checkpoint": f"v14_candidate_point_fold{fold}_seed0.pt",
        "v8_meanaligned_checkpoint": f"v8_corrected_mean_fold{fold}_seed0.pt",
        "tic2a_feature41_model_artifact": (f"tic2a_corrected_models_fold{fold}.json"),
    }.get(source_id)


def validate_source_manifest(path: Path) -> dict[int, dict[str, Any]]:
    """Validate a complete activation-time binding without reading wide results."""

    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError("puzzle-set source manifest must be one absolute file")
    manifest = read_json_object(path)
    if (
        set(manifest) != SOURCE_MANIFEST_TOP_FIELDS
        or manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("status") != SOURCE_MANIFEST_STATUS
        or manifest.get("contract_id") != EXPECTED_CONTRACT_ID
        or manifest.get("binding_status") != SOURCE_BINDING_STATUS
    ):
        raise RuntimeError("puzzle-set source manifest identity or binding changed")
    rows = manifest.get("folds")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FOLDS):
        raise RuntimeError("puzzle-set source manifest must contain twenty folds")
    try:
        by_fold = {int(row["outer_fold"]): row for row in rows}
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(
            "puzzle-set source manifest fold identity is malformed"
        ) from None
    if tuple(sorted(by_fold)) != EXPECTED_FOLDS or len(by_fold) != len(rows):
        raise RuntimeError("puzzle-set source manifest is not unique folds0-19")

    global_source_paths: dict[str, str] = {}
    for fold in EXPECTED_FOLDS:
        row = by_fold[fold]
        if (
            not isinstance(row, dict)
            or set(row) != SOURCE_MANIFEST_FOLD_FIELDS
            or row.get("outer_fold") != fold
            or row.get("held_puzzle") != f"P{fold + 1:02d}"
            or row.get("seed") != FROZEN_PARENT_SEED
            or not isinstance(row.get("sources"), dict)
            or set(row["sources"]) != set(FROZEN_INPUT_SOURCE_SPEC)
        ):
            raise RuntimeError(f"puzzle-set source manifest fold {fold} changed")
        for source_id, expected in FROZEN_INPUT_SOURCE_SPEC.items():
            source = row["sources"][source_id]
            expected_fold = fold if source_id in FOLD_SCOPED_INPUT_SOURCES else None
            if (
                not isinstance(source, dict)
                or set(source) != SOURCE_MANIFEST_RECORD_FIELDS
                or source.get("role") != expected["role"]
                or source.get("used_in_candidate_prediction")
                is not expected["used_in_candidate_prediction"]
                or source.get("outer_fold") != expected_fold
                or source.get("seed") != expected["seed"]
                or source.get("realized_parameter_count")
                != expected["realized_parameter_count"]
                or source.get("trainable_in_p1") is not expected["trainable_in_p1"]
            ):
                raise RuntimeError(
                    f"puzzle-set source manifest fold {fold} source {source_id} changed"
                )
            source_path = Path(str(source.get("path", "")))
            expected_filename = _expected_filename(source_id, fold)
            if (
                not source_path.is_absolute()
                or not source_path.is_file()
                or (
                    expected_filename is not None
                    and source_path.name != expected_filename
                )
            ):
                raise FileNotFoundError(
                    f"puzzle-set source manifest fold {fold} source "
                    f"{source_id} is absent or misbound"
                )
            if source_id not in FOLD_SCOPED_INPUT_SOURCES:
                previous = global_source_paths.setdefault(source_id, str(source_path))
                if previous != str(source_path):
                    raise RuntimeError(
                        f"puzzle-set global source differs across folds: {source_id}"
                    )
    return by_fold


def validate_manifest_fold_runtime_binding(
    *,
    manifest_rows: Mapping[int, dict[str, Any]],
    outer_fold: int,
    runtime_sources: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require each actual CLI-derived source to equal its manifest binding."""

    fold = int(outer_fold)
    if fold not in manifest_rows:
        raise RuntimeError(f"puzzle-set source manifest lacks fold {fold}")
    if set(runtime_sources) != set(FROZEN_INPUT_SOURCE_SPEC):
        raise RuntimeError("puzzle-set runtime source universe changed")
    bound_sources = manifest_rows[fold]["sources"]
    for source_id, runtime in runtime_sources.items():
        bound = bound_sources[source_id]
        for field in (
            "path",
            "role",
            "used_in_candidate_prediction",
            "outer_fold",
            "seed",
        ):
            if runtime.get(field) != bound.get(field):
                raise RuntimeError(
                    f"puzzle-set runtime source differs from manifest: "
                    f"fold {fold} {source_id}.{field}"
                )
