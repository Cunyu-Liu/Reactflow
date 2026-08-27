#!/usr/bin/env python3
"""Merge only the complete, target-free post-V14 branch-5 probe universe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.post_v14_branch5_route_probe import (
    ALIGNED_SHIFT,
    MATCHED_NULL_SHIFT,
    PROBE_FEATURE_WIDTH,
    RIDGE_ALPHA,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    load_tic2a_safe_registry,
)
from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_FOLDS,
    EXPECTED_SEED,
    FROZEN_RUNTIME_PATHS,
    FOLD_SCHEMA,
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_PHASE,
    PREDICTION_SCHEMA,
    RIDGE_SCHEMA,
    STANDARDIZATION_INACTIVE_STD_THRESHOLD,
    V14_HIDDEN_SOURCE,
    _load_source_registry,
    assert_frozen_runtime_paths,
    assert_run_authority,
)


SCHEMA = "reactflow_delta.puzzle_set_branch5_route_probe_merged.v1"
STATUS = "BRANCH5_ROUTE_PROBE_COMPLETE_UNSCORED_MERGE_PASS"
EXPECTED_PREDICTION_FIELDS = {
    "schema_version",
    "keys",
    "biological_scoring_key",
    "outer_fold",
    "seed",
    "registered_status",
    "parent_point",
    "aligned_point",
    "shift17_point",
}
EXPECTED_FOLD_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "outer_fold",
    "held_puzzle",
    "seed",
    "source_provenance",
    "ridge_model_artifact",
    "prediction_artifact",
    "n_registered_prediction_rows",
    "coordinate_frame_count",
    "n_outer_train_puzzles",
    "n_outer_train_supervised_cells",
    "n_outer_train_qualified_rows",
    "invariants",
}
EXPECTED_RIDGE_FIELDS = {
    "schema_version",
    "outer_fold",
    "seed",
    "feature_width",
    "arms_fit_independently",
    "weighted_outer_train_standardization_per_arm",
    "intercept_unpenalized",
    "ridge_alpha",
    "shifts",
    "models",
    "fit_target",
    "v14_hidden_source",
    "reference_preserves_position_and_region",
    "reference_zeros_sequence_reactivity_precision_observed",
    "shift17_applied_only_after_content_contrast",
    "inactive_std_threshold_lt",
    "inactive_std_replacement_scale",
    "constant_feature_standardized_value",
    "sample_weight_hierarchy",
    "n_outer_train_puzzles",
    "n_outer_train_supervised_cells",
    "n_outer_train_qualified_rows",
    "held_puzzle_target_accessed",
    "partial_score_computed",
    "external_outcome_accessed",
}


def assert_merge_authority(
    repo_root: Path, *, input_dir: Path, out_json: Path
) -> dict[str, Any]:
    """Keep the complete prediction-only merge inside the frozen B5RP1 universe."""

    active = assert_run_authority(repo_root)
    assert_frozen_runtime_paths(
        active.get("authority"),
        required_fields=("prediction_dir", "complete_unscored_merge_path"),
        cli_paths={
            "prediction_dir": input_dir,
            "complete_unscored_merge_path": out_json,
        },
    )
    return active


REQUIRED_INVARIANTS_TRUE = (
    "target_profile_identity_exact",
    "samefold_v13_parent_recomputed_from_checkpoint",
    "samefold_v14_encoder_recomputed_from_checkpoint",
    "v14_hidden_is_zero_preserving_content_contrast",
    "v14_coordinate_only_reference_preserves_position_and_region",
    "v14_coordinate_only_reference_zeros_biological_streams",
    "shift17_applied_only_after_v14_content_contrast",
    "tic2a_feature41_loaded_from_strict_target_free_projection",
    "seven_nonfocal_constructs_only",
    "aligned_and_shift17_fit_independently",
    "outer_train_weighted_standardization",
    "ridge_alpha_one_unpenalized_intercept",
    "held_prediction_full_registered_universe",
)
REQUIRED_INVARIANTS_FALSE = (
    "held_score_computed",
    "prediction_contains_target_fields",
    "partial_score_inspected",
    "external_outcome_accessed",
    "model_or_threshold_selection_performed",
)


def _source_checks(row: dict[str, Any], fold: int) -> dict[str, bool]:
    sources = row.get("source_provenance", {})
    expected = {
        "v13_point_checkpoint": (
            "FROZEN_SAME_OUTER_FOLD_V13_POINT_MODEL",
            f"v13_candidate_point_fold{fold}_seed0.pt",
        ),
        "v14_encoder_checkpoint": (
            "FROZEN_SAME_OUTER_FOLD_V14_OUTCOME_BLIND_ENCODER",
            f"v14_candidate_point_fold{fold}_seed0.pt",
        ),
    }
    checks: dict[str, bool] = {
        "exact_source_fields": set(sources)
        == {
            *expected,
            "safe_source_manifest",
            "m2_csv",
            "tic2a_merged_registry",
            "unconstrained_feature_cache",
            "constrained_feature_cache",
            "tic2a_feature41_model_artifact",
        }
    }
    for name, (role, filename) in expected.items():
        source = sources.get(name, {})
        path = Path(str(source.get("path", "")))
        checks[name] = bool(
            set(source) == {"path", "role", "outer_fold", "seed"}
            and source.get("role") == role
            and int(source.get("outer_fold", -1)) == fold
            and int(source.get("seed", -1)) == EXPECTED_SEED
            and path.is_absolute()
            and path.name == filename
            and path.is_file()
        )
    manifest = sources.get("safe_source_manifest", {})
    manifest_path = Path(str(manifest.get("path", "")))
    checks["safe_source_manifest"] = bool(
        set(manifest) == {"path", "role"}
        and manifest.get("role")
        == "POST_V14_BRANCH5_SCORE_PREDICTION_HISTORY_FREE_SOURCE_PROJECTION"
        and manifest_path.is_absolute()
        and manifest_path.is_file()
    )
    global_sources = {
        "m2_csv": "TRAIN_TARGET_AND_OUTCOME_BLIND_CONTEXT_SOURCE",
        "tic2a_merged_registry": "STRICT_TARGET_FREE_FEATURE41_REGISTRY",
        "unconstrained_feature_cache": ("OUTCOME_BLIND_FEATURE41_UNCONSTRAINED_CACHE"),
        "constrained_feature_cache": "OUTCOME_BLIND_FEATURE41_CONSTRAINED_CACHE",
    }
    for name, role in global_sources.items():
        source = sources.get(name, {})
        path = Path(str(source.get("path", "")))
        checks[name] = bool(
            set(source) == {"path", "role", "scope"}
            and source.get("role") == role
            and source.get("scope") == "GLOBAL"
            and path.is_absolute()
            and path.is_file()
        )
    tic2a_model = sources.get("tic2a_feature41_model_artifact", {})
    tic2a_model_path = Path(str(tic2a_model.get("path", "")))
    checks["tic2a_feature41_model_artifact"] = bool(
        set(tic2a_model)
        == {
            "path",
            "role",
            "outer_fold",
            "source_phase",
            "held_target_used_for_prediction",
            "held_score_computed",
            "partial_score_inspected",
            "external_outcome_accessed",
        }
        and tic2a_model.get("role") == "FROZEN_OUTER_FOLD_TIC2A_FEATURE41_MODEL_ONLY"
        and int(tic2a_model.get("outer_fold", -1)) == fold
        and tic2a_model.get("source_phase") == "TIC2A"
        and tic2a_model.get("held_target_used_for_prediction") is False
        and tic2a_model.get("held_score_computed") is False
        and tic2a_model.get("partial_score_inspected") is False
        and tic2a_model.get("external_outcome_accessed") is False
        and tic2a_model_path.is_absolute()
        and tic2a_model_path.name == f"tic2a_corrected_models_fold{fold}.json"
        and tic2a_model_path.is_file()
    )
    return checks


def prediction_checks(path: Path, *, fold: int, expected_rows: int) -> dict[str, bool]:
    if not path.is_file():
        return {"exists": False}
    with np.load(path, allow_pickle=True) as handle:
        names = set(handle.files)
        keys = list(map(str, handle["keys"])) if "keys" in names else []
        numeric = [name for name in names if handle[name].dtype.kind in "fiu"]
        row_fields = (
            "biological_scoring_key",
            "outer_fold",
            "seed",
            "registered_status",
            "parent_point",
            "aligned_point",
            "shift17_point",
        )
        row_shapes = all(
            name in names and handle[name].shape == (len(keys),) for name in row_fields
        )
        return {
            "exists": True,
            "exact_fields": names == EXPECTED_PREDICTION_FIELDS,
            "target_free": not bool(names & FORBIDDEN_PREDICTION_FIELDS),
            "schema": "schema_version" in names
            and str(handle["schema_version"].item()) == PREDICTION_SCHEMA,
            "expected_rows": len(keys) == expected_rows,
            "row_shapes": row_shapes,
            "unique_keys": len(keys) == len(set(keys)),
            "biological_key_match": "biological_scoring_key" in names
            and keys == list(map(str, handle["biological_scoring_key"])),
            "fold": "outer_fold" in names
            and set(map(int, handle["outer_fold"])) == {fold},
            "seed": "seed" in names
            and set(map(int, handle["seed"])) == {EXPECTED_SEED},
            "covered": "registered_status" in names
            and set(map(str, handle["registered_status"])) == {"covered"},
            "finite": all(np.isfinite(handle[name]).all() for name in numeric),
        }


def _ridge_array_valid(model: dict[str, Any]) -> bool:
    if set(model) != {"mean_x", "scale_x", "mean_y", "coefficient", "alpha"}:
        return False
    try:
        mean_x = np.asarray(model["mean_x"], dtype=np.float64)
        scale_x = np.asarray(model["scale_x"], dtype=np.float64)
        coefficient = np.asarray(model["coefficient"], dtype=np.float64)
        mean_y = float(model["mean_y"])
        alpha = float(model["alpha"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        mean_x.shape == (PROBE_FEATURE_WIDTH,)
        and scale_x.shape == (PROBE_FEATURE_WIDTH,)
        and coefficient.shape == (PROBE_FEATURE_WIDTH,)
        and np.isfinite(mean_x).all()
        and np.isfinite(scale_x).all()
        and np.isfinite(coefficient).all()
        and np.isfinite(mean_y)
        and np.all(scale_x > 0)
        and alpha == RIDGE_ALPHA
    )


def ridge_checks(path: Path, *, fold: int) -> dict[str, bool]:
    if not path.is_file():
        return {"exists": False}
    row = json.loads(path.read_text(encoding="utf-8"))
    models = row.get("models", {})
    return {
        "exists": True,
        "exact_fields": set(row) == EXPECTED_RIDGE_FIELDS,
        "schema": row.get("schema_version") == RIDGE_SCHEMA,
        "fold_seed": int(row.get("outer_fold", -1)) == fold
        and int(row.get("seed", -1)) == EXPECTED_SEED,
        "feature_width": int(row.get("feature_width", -1)) == PROBE_FEATURE_WIDTH,
        "independent_arms": row.get("arms_fit_independently") is True,
        "weighted_standardization": row.get(
            "weighted_outer_train_standardization_per_arm"
        )
        is True,
        "unpenalized_intercept": row.get("intercept_unpenalized") is True,
        "alpha": float(row.get("ridge_alpha", float("nan"))) == RIDGE_ALPHA,
        "shifts": row.get("shifts")
        == {"aligned": ALIGNED_SHIFT, "shift17": MATCHED_NULL_SHIFT},
        "model_arms": set(models) == {"aligned", "shift17"},
        "model_values": set(models) == {"aligned", "shift17"}
        and all(_ridge_array_valid(models[name]) for name in models),
        "fit_target": row.get("fit_target") == "SIGNED_DELTA_MINUS_FROZEN_V13_POINT",
        "sample_weight_hierarchy": row.get("sample_weight_hierarchy")
        == "PUZZLE_TO_METHOD_CELL_TO_MUTANT_TO_QUALIFIED_POSITION",
        "v14_content_contrast": row.get("v14_hidden_source") == V14_HIDDEN_SOURCE
        and row.get("reference_preserves_position_and_region") is True
        and row.get("reference_zeros_sequence_reactivity_precision_observed") is True
        and row.get("shift17_applied_only_after_content_contrast") is True,
        "inactive_standardization": float(
            row.get("inactive_std_threshold_lt", float("nan"))
        )
        == STANDARDIZATION_INACTIVE_STD_THRESHOLD
        and float(row.get("inactive_std_replacement_scale", float("nan"))) == 1.0
        and float(row.get("constant_feature_standardized_value", float("nan"))) == 0.0,
        "outer_train_puzzles": int(row.get("n_outer_train_puzzles", -1)) == 19,
        "outer_train_cells": int(row.get("n_outer_train_supervised_cells", -1))
        == (152 if fold == 19 else 151),
        "outer_train_rows": int(row.get("n_outer_train_qualified_rows", -1)) > 0,
        "held_closed": row.get("held_puzzle_target_accessed") is False,
        "partial_closed": row.get("partial_score_computed") is False,
        "external_closed": row.get("external_outcome_accessed") is False,
    }


def merge_folds(
    input_dir: Path,
    *,
    expected_checkpoint_dirs: dict[str, Path] | None = None,
    expected_global_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    expected_names = {
        f"puzzle_set_branch5_probe_fold{fold}_seed0.json" for fold in EXPECTED_FOLDS
    }
    observed_paths = sorted(input_dir.glob("puzzle_set_branch5_probe_fold*_seed*.json"))
    observed_names = {path.name for path in observed_paths}
    if observed_names != expected_names or len(observed_paths) != len(expected_names):
        raise ValueError(
            "branch5 fold universe is missing or unexpected: "
            f"missing={sorted(expected_names - observed_names)} "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    rows = []
    seen: set[int] = set()
    manifest_paths: set[str] = set()
    global_source_paths: dict[str, set[str]] = {
        "m2_csv": set(),
        "tic2a_merged_registry": set(),
        "unconstrained_feature_cache": set(),
        "constrained_feature_cache": set(),
    }
    for path in observed_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        fold = int(row.get("outer_fold", -1))
        if fold in seen:
            raise ValueError(f"duplicate branch5 fold identity {fold}")
        seen.add(fold)
        if (
            set(row) != EXPECTED_FOLD_FIELDS
            or row.get("schema_version") != FOLD_SCHEMA
            or row.get("phase") != PREDICTION_PHASE
            or row.get("status") != "BRANCH5_ROUTE_PROBE_FOLD_PREDICTION_PASS"
            or int(row.get("seed", -1)) != EXPECTED_SEED
        ):
            raise ValueError(f"invalid branch5 fold result in {path}")
        if (
            int(row.get("coordinate_frame_count", -1)) != 20
            or int(row.get("n_outer_train_puzzles", -1)) != 19
            or int(row.get("n_outer_train_supervised_cells", -1))
            != (152 if fold == 19 else 151)
            or int(row.get("n_outer_train_qualified_rows", -1)) <= 0
            or int(row.get("n_registered_prediction_rows", -1)) <= 0
        ):
            raise ValueError(f"branch5 fold {fold} data universe changed")
        invariants = row.get("invariants", {})
        if (
            set(invariants)
            != set(REQUIRED_INVARIANTS_TRUE) | set(REQUIRED_INVARIANTS_FALSE)
            or not all(
                invariants.get(name) is True for name in REQUIRED_INVARIANTS_TRUE
            )
            or not all(
                invariants.get(name) is False for name in REQUIRED_INVARIANTS_FALSE
            )
        ):
            raise ValueError(f"branch5 fold {fold} invariant record is invalid")
        source = _source_checks(row, fold)
        if not all(source.values()):
            raise ValueError(
                f"branch5 fold {fold} provenance failed: "
                f"{[name for name, passed in source.items() if not passed]}"
            )
        manifest_paths.add(
            str(row["source_provenance"]["safe_source_manifest"]["path"])
        )
        for name in global_source_paths:
            global_source_paths[name].add(str(row["source_provenance"][name]["path"]))
        prediction_path = Path(str(row.get("prediction_artifact", "")))
        ridge_path = Path(str(row.get("ridge_model_artifact", "")))
        expected_prediction_path = (
            input_dir / f"puzzle_set_branch5_probe_predictions_fold{fold}_seed0.npz"
        ).resolve()
        expected_ridge_path = (
            input_dir / f"puzzle_set_branch5_probe_ridge_fold{fold}_seed0.json"
        ).resolve()
        if (
            not prediction_path.is_absolute()
            or not ridge_path.is_absolute()
            or prediction_path != expected_prediction_path
            or ridge_path != expected_ridge_path
        ):
            raise ValueError(
                f"branch5 fold {fold} artifact path differs from the frozen input directory"
            )
        prediction = prediction_checks(
            prediction_path,
            fold=fold,
            expected_rows=int(row.get("n_registered_prediction_rows", -1)),
        )
        ridge = ridge_checks(ridge_path, fold=fold)
        if not all(prediction.values()) or not all(ridge.values()):
            raise ValueError(
                f"branch5 fold {fold} artifact validation failed: "
                f"prediction={[k for k,v in prediction.items() if not v]} "
                f"ridge={[k for k,v in ridge.items() if not v]}"
            )
        rows.append(row)
    if (
        seen != set(EXPECTED_FOLDS)
        or len(manifest_paths) != 1
        or any(len(paths) != 1 for paths in global_source_paths.values())
    ):
        raise ValueError("branch5 folds do not share complete global safe sources")
    if expected_global_paths is not None:
        observed_global_paths = {
            "source_manifest_path": Path(next(iter(manifest_paths))),
            **{
                f"{name}_path": Path(next(iter(paths)))
                for name, paths in global_source_paths.items()
            },
        }
        if set(observed_global_paths) != set(expected_global_paths):
            raise ValueError("branch5 frozen global source fields changed")
        for name, expected_path in expected_global_paths.items():
            if observed_global_paths[name] != expected_path:
                raise ValueError(
                    f"branch5 {name} differs from the frozen runtime source"
                )
    registry = _load_source_registry(
        Path(next(iter(manifest_paths))),
        expected_checkpoint_dirs=expected_checkpoint_dirs,
    )
    tic2a_registry = load_tic2a_safe_registry(
        Path(next(iter(global_source_paths["tic2a_merged_registry"])))
    )
    for row in rows:
        fold = int(row["outer_fold"])
        source = row["source_provenance"]
        if (
            source["v13_point_checkpoint"]["path"]
            != registry[fold]["v13_candidate_checkpoint"]
            or source["v14_encoder_checkpoint"]["path"]
            != registry[fold]["v14_candidate_checkpoint"]
            or source["tic2a_feature41_model_artifact"]["path"]
            != str(tic2a_registry[fold].model_path)
            or str(row["held_puzzle"]) != str(registry[fold]["held_puzzle"])
            or str(row["held_puzzle"]) != str(tic2a_registry[fold].row["held_puzzle"])
        ):
            raise ValueError(f"branch5 fold {fold} differs from safe source registry")
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": PREDICTION_PHASE,
        "status": STATUS,
        "expected_folds": EXPECTED_FOLDS,
        "expected_seed": EXPECTED_SEED,
        "folds": rows,
        "merge_integrity": {
            "complete_fold_universe": True,
            "unique_fold_ids": True,
            "prediction_only_schema": True,
            "prediction_key_universe_unique_per_fold": True,
            "samefold_parent_provenance_all_folds": True,
            "samefold_v14_content_contrast_all_folds": True,
            "single_complete_safe_source_registry": True,
            "single_complete_tic2a_safe_registry": True,
            "global_input_provenance_consistent_all_folds": True,
            "tic2a_safe_feature41_projection_all_folds": True,
            "ridge_protocol_exact_all_folds": True,
            "target_profile_identity_exact": True,
            "partial_scores_inspected": False,
            "external_outcome_accessed": False,
            "model_or_threshold_selection_performed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    input_dir = args.input_dir.resolve()
    out_json = args.out_json.resolve()
    assert_merge_authority(
        args.repo_root.resolve(), input_dir=input_dir, out_json=out_json
    )
    if out_json.exists():
        raise FileExistsError("branch5 refuses to overwrite its complete merge")
    result = merge_folds(
        input_dir,
        expected_checkpoint_dirs={
            "v13_checkpoint_dir": FROZEN_RUNTIME_PATHS["v13_checkpoint_dir"],
            "v14_checkpoint_dir": FROZEN_RUNTIME_PATHS["v14_checkpoint_dir"],
        },
        expected_global_paths={
            "source_manifest_path": FROZEN_RUNTIME_PATHS["source_manifest_path"],
            "m2_csv_path": FROZEN_RUNTIME_PATHS["m2_csv_path"],
            "tic2a_merged_registry_path": FROZEN_RUNTIME_PATHS[
                "tic2a_merged_registry_path"
            ],
            "unconstrained_feature_cache_path": FROZEN_RUNTIME_PATHS[
                "unconstrained_feature_cache_path"
            ],
            "constrained_feature_cache_path": FROZEN_RUNTIME_PATHS[
                "constrained_feature_cache_path"
            ],
        },
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_json.with_name(f"{out_json.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, out_json)
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
