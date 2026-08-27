from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.merge_post_v14_branch5_route_probe import (
    SCHEMA,
    STATUS,
    assert_merge_authority,
    merge_folds,
)
from scripts.reactflow_delta.post_v14_branch5_route_probe import PROBE_FEATURE_WIDTH
from scripts.reactflow_delta.model_rescue_v6_probe import (
    CANDIDATE_PROBE_FEATURE_NAMES,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    TIC2A_FOLD_SCHEMA,
    TIC2A_MERGED_SCHEMA,
    TIC2A_MERGED_STATUS,
    TIC2A_MERGE_INTEGRITY,
)
from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    FOLD_SCHEMA,
    FROZEN_RUNTIME_PATHS,
    PREDICTION_PHASE,
    PREDICTION_TOKEN,
    PREDICTION_SCHEMA,
    RIDGE_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    STANDARDIZATION_INACTIVE_STD_THRESHOLD,
    V14_HIDDEN_SOURCE,
)


def test_merge_authority_binds_input_and_output_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    active_path = repo / "configs/reactflow_delta/active_contract.yaml"
    active_path.parent.mkdir(parents=True)
    active_path.write_text(
        yaml.safe_dump(
            {
                "project_task_id": EXPECTED_PROJECT_TASK,
                "parent_state": dict(EXPECTED_PARENT_STATE),
                "authority": {
                    "current_phase": PREDICTION_PHASE,
                    "source_manifest_status": SOURCE_MANIFEST_STATUS,
                    **{name: str(path) for name, path in FROZEN_RUNTIME_PATHS.items()},
                },
                "runnable_phases": [PREDICTION_PHASE],
                "training_allowed": PREDICTION_TOKEN,
                "candidate_model_training_allowed": PREDICTION_TOKEN,
                "held_score_read_allowed": False,
                "partial_fold_score_read_allowed": False,
                "new_external_outcome_access_allowed": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert_merge_authority(
        repo,
        input_dir=FROZEN_RUNTIME_PATHS["prediction_dir"],
        out_json=FROZEN_RUNTIME_PATHS["complete_unscored_merge_path"],
    )
    with pytest.raises(RuntimeError, match="CLI prediction_dir differs"):
        assert_merge_authority(
            repo,
            input_dir=(tmp_path / "alternate").resolve(),
            out_json=FROZEN_RUNTIME_PATHS["complete_unscored_merge_path"],
        )


def _ridge_model() -> dict:
    return {
        "mean_x": [0.0] * PROBE_FEATURE_WIDTH,
        "scale_x": [1.0] * PROBE_FEATURE_WIDTH,
        "mean_y": 0.0,
        "coefficient": [0.0] * PROBE_FEATURE_WIDTH,
        "alpha": 1.0,
    }


def _make_complete_universe(root: Path) -> None:
    safe = root / "safe_source_manifest.json"
    tic2a_registry = root / "tic2a_complete_unscored_merge.json"
    m2_csv = root / "OK7a_M2_data.v4.5.2.csv"
    unconstrained_cache = root / "unconstrained_feature_cache.h5"
    constrained_cache = root / "constrained_feature_cache.h5"
    m2_csv.touch()
    unconstrained_cache.touch()
    constrained_cache.touch()
    safe_rows = []
    tic2a_rows = []
    for fold in range(20):
        v13 = root / f"v13_candidate_point_fold{fold}_seed0.pt"
        v14 = root / f"v14_candidate_point_fold{fold}_seed0.pt"
        v13.touch()
        v14.touch()
        tic2a_model = root / f"tic2a_corrected_models_fold{fold}.json"
        tic2a_model.write_text(
            json.dumps(
                {
                    "direct18": {},
                    "direct18_feature_names": [
                        f"direct_{index}" for index in range(18)
                    ],
                    "feature30_feature_names": [
                        f"feature30_{index}" for index in range(30)
                    ],
                    "feature41_feature_names": list(CANDIDATE_PROBE_FEATURE_NAMES),
                    "ridge_alpha": 1.0,
                    "v5_feature30": {},
                    "v6_feature30_replay": {},
                    "v6_feature41": {
                        "mean_x": [0.0] * 41,
                        "scale_x": [1.0] * 41,
                        "mean_y": [0.0, 0.0],
                        "coefficient": [[0.0, 0.0] for _ in range(41)],
                        "alpha": 1.0,
                    },
                }
            )
        )
        tic2a_rows.append(
            {
                "schema_version": TIC2A_FOLD_SCHEMA,
                "phase": "TIC2A",
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
                "held_target_used_for_prediction": False,
                "held_score_computed": False,
                "partial_score_inspected": False,
                "legacy_prediction_reused": False,
                "external_outcome_accessed": False,
                "model_artifact": str(tic2a_model),
                "prediction_artifact": str(root / f"tic2a_prediction_fold{fold}.npz"),
                "n_registered_prediction_rows": 1,
                "n_train_cells": 1,
                "n_train_qualified_positions": 1,
                "n_train_valid_mutants": 1,
                "v5_v6_feature30_prediction_replay_pass": True,
                "v5_v6_feature30_stats_replay_pass": True,
            }
        )
        safe_rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "seed": 0,
                "v13_source_phase": "V13M3",
                "v13_candidate_checkpoint": str(v13),
                "v14_source_phase": "V14M3",
                "v14_arm": "CANDIDATE",
                "v14_candidate_checkpoint": str(v14),
                "held_score_closed_at_projection": True,
                "external_outcome_accessed": False,
            }
        )
        key = f"openknot_m2|P{fold + 1:02d}|M|C|0|A>C|0"
        prediction = root / f"puzzle_set_branch5_probe_predictions_fold{fold}_seed0.npz"
        np.savez_compressed(
            prediction,
            schema_version=np.asarray(PREDICTION_SCHEMA),
            keys=np.asarray([key], dtype=object),
            biological_scoring_key=np.asarray([key], dtype=object),
            outer_fold=np.asarray([fold]),
            seed=np.asarray([0]),
            registered_status=np.asarray(["covered"], dtype=object),
            parent_point=np.asarray([0.0]),
            aligned_point=np.asarray([0.1]),
            shift17_point=np.asarray([0.2]),
        )
        ridge = root / f"puzzle_set_branch5_probe_ridge_fold{fold}_seed0.json"
        ridge.write_text(
            json.dumps(
                {
                    "schema_version": RIDGE_SCHEMA,
                    "outer_fold": fold,
                    "seed": 0,
                    "feature_width": PROBE_FEATURE_WIDTH,
                    "arms_fit_independently": True,
                    "weighted_outer_train_standardization_per_arm": True,
                    "intercept_unpenalized": True,
                    "ridge_alpha": 1.0,
                    "shifts": {"aligned": 0, "shift17": 17},
                    "models": {
                        "aligned": _ridge_model(),
                        "shift17": _ridge_model(),
                    },
                    "fit_target": "SIGNED_DELTA_MINUS_FROZEN_V13_POINT",
                    "v14_hidden_source": V14_HIDDEN_SOURCE,
                    "reference_preserves_position_and_region": True,
                    "reference_zeros_sequence_reactivity_precision_observed": True,
                    "shift17_applied_only_after_content_contrast": True,
                    "inactive_std_threshold_lt": STANDARDIZATION_INACTIVE_STD_THRESHOLD,
                    "inactive_std_replacement_scale": 1.0,
                    "constant_feature_standardized_value": 0.0,
                    "sample_weight_hierarchy": (
                        "PUZZLE_TO_METHOD_CELL_TO_MUTANT_TO_QUALIFIED_POSITION"
                    ),
                    "n_outer_train_puzzles": 19,
                    "n_outer_train_supervised_cells": 152 if fold == 19 else 151,
                    "n_outer_train_qualified_rows": 100,
                    "held_puzzle_target_accessed": False,
                    "partial_score_computed": False,
                    "external_outcome_accessed": False,
                }
            )
        )
        row = {
            "schema_version": FOLD_SCHEMA,
            "phase": PREDICTION_PHASE,
            "status": "BRANCH5_ROUTE_PROBE_FOLD_PREDICTION_PASS",
            "outer_fold": fold,
            "held_puzzle": f"P{fold + 1:02d}",
            "seed": 0,
            "source_provenance": {
                "v13_point_checkpoint": {
                    "path": str(v13),
                    "role": "FROZEN_SAME_OUTER_FOLD_V13_POINT_MODEL",
                    "outer_fold": fold,
                    "seed": 0,
                },
                "v14_encoder_checkpoint": {
                    "path": str(v14),
                    "role": "FROZEN_SAME_OUTER_FOLD_V14_OUTCOME_BLIND_ENCODER",
                    "outer_fold": fold,
                    "seed": 0,
                },
                "safe_source_manifest": {
                    "path": str(safe),
                    "role": (
                        "POST_V14_BRANCH5_SCORE_PREDICTION_HISTORY_FREE_SOURCE_PROJECTION"
                    ),
                },
                "m2_csv": {
                    "path": str(m2_csv),
                    "role": "TRAIN_TARGET_AND_OUTCOME_BLIND_CONTEXT_SOURCE",
                    "scope": "GLOBAL",
                },
                "tic2a_merged_registry": {
                    "path": str(tic2a_registry),
                    "role": "STRICT_TARGET_FREE_FEATURE41_REGISTRY",
                    "scope": "GLOBAL",
                },
                "unconstrained_feature_cache": {
                    "path": str(unconstrained_cache),
                    "role": "OUTCOME_BLIND_FEATURE41_UNCONSTRAINED_CACHE",
                    "scope": "GLOBAL",
                },
                "constrained_feature_cache": {
                    "path": str(constrained_cache),
                    "role": "OUTCOME_BLIND_FEATURE41_CONSTRAINED_CACHE",
                    "scope": "GLOBAL",
                },
                "tic2a_feature41_model_artifact": {
                    "path": str(tic2a_model),
                    "role": "FROZEN_OUTER_FOLD_TIC2A_FEATURE41_MODEL_ONLY",
                    "outer_fold": fold,
                    "source_phase": "TIC2A",
                    "held_target_used_for_prediction": False,
                    "held_score_computed": False,
                    "partial_score_inspected": False,
                    "external_outcome_accessed": False,
                },
            },
            "ridge_model_artifact": str(ridge),
            "prediction_artifact": str(prediction),
            "n_registered_prediction_rows": 1,
            "coordinate_frame_count": 20,
            "n_outer_train_puzzles": 19,
            "n_outer_train_supervised_cells": 152 if fold == 19 else 151,
            "n_outer_train_qualified_rows": 100,
            "invariants": {
                "target_profile_identity_exact": True,
                "samefold_v13_parent_recomputed_from_checkpoint": True,
                "samefold_v14_encoder_recomputed_from_checkpoint": True,
                "v14_hidden_is_zero_preserving_content_contrast": True,
                "v14_coordinate_only_reference_preserves_position_and_region": True,
                "v14_coordinate_only_reference_zeros_biological_streams": True,
                "shift17_applied_only_after_v14_content_contrast": True,
                "tic2a_feature41_loaded_from_strict_target_free_projection": True,
                "seven_nonfocal_constructs_only": True,
                "aligned_and_shift17_fit_independently": True,
                "outer_train_weighted_standardization": True,
                "ridge_alpha_one_unpenalized_intercept": True,
                "held_prediction_full_registered_universe": True,
                "held_score_computed": False,
                "prediction_contains_target_fields": False,
                "partial_score_inspected": False,
                "external_outcome_accessed": False,
                "model_or_threshold_selection_performed": False,
            },
        }
        (root / f"puzzle_set_branch5_probe_fold{fold}_seed0.json").write_text(
            json.dumps(row)
        )
    safe.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_MANIFEST_SCHEMA,
                "status": SOURCE_MANIFEST_STATUS,
                "parent_state": dict(EXPECTED_PARENT_STATE),
                "folds": safe_rows,
            }
        )
    )
    tic2a_registry.write_text(
        json.dumps(
            {
                "schema_version": TIC2A_MERGED_SCHEMA,
                "phase": "TIC2A",
                "status": TIC2A_MERGED_STATUS,
                "merge_integrity": dict(TIC2A_MERGE_INTEGRITY),
                "folds": tic2a_rows,
            }
        )
    )


def _expected_global_paths(root: Path) -> dict[str, Path]:
    return {
        "source_manifest_path": root / "safe_source_manifest.json",
        "m2_csv_path": root / "OK7a_M2_data.v4.5.2.csv",
        "tic2a_merged_registry_path": root / "tic2a_complete_unscored_merge.json",
        "unconstrained_feature_cache_path": root / "unconstrained_feature_cache.h5",
        "constrained_feature_cache_path": root / "constrained_feature_cache.h5",
    }


def test_merge_accepts_only_complete_target_free_universe(tmp_path: Path) -> None:
    _make_complete_universe(tmp_path)
    result = merge_folds(tmp_path)
    assert result["schema_version"] == SCHEMA
    assert result["status"] == STATUS
    assert [row["outer_fold"] for row in result["folds"]] == list(range(20))
    assert result["merge_integrity"]["samefold_parent_provenance_all_folds"]
    assert result["merge_integrity"]["samefold_v14_content_contrast_all_folds"]
    assert result["merge_integrity"]["tic2a_safe_feature41_projection_all_folds"]
    assert result["merge_integrity"]["global_input_provenance_consistent_all_folds"]


def test_merge_rejects_missing_or_unexpected_fold(tmp_path: Path) -> None:
    _make_complete_universe(tmp_path)
    (tmp_path / "puzzle_set_branch5_probe_fold19_seed0.json").unlink()
    with pytest.raises(ValueError, match="missing or unexpected"):
        merge_folds(tmp_path)

    (tmp_path / "puzzle_set_branch5_probe_fold20_seed0.json").write_text("{}")
    with pytest.raises(ValueError, match="missing or unexpected"):
        merge_folds(tmp_path)


def test_merge_rejects_target_field_or_wrong_fold_provenance(tmp_path: Path) -> None:
    _make_complete_universe(tmp_path)
    prediction = tmp_path / "puzzle_set_branch5_probe_predictions_fold0_seed0.npz"
    with np.load(prediction, allow_pickle=True) as handle:
        values = {name: np.asarray(handle[name]) for name in handle.files}
    values["target"] = np.asarray([0.0])
    np.savez_compressed(prediction, **values)
    with pytest.raises(ValueError, match="artifact validation"):
        merge_folds(tmp_path)

    _make_complete_universe(tmp_path)
    prediction = tmp_path / "puzzle_set_branch5_probe_predictions_fold0_seed0.npz"
    with np.load(prediction, allow_pickle=True) as handle:
        values = {name: np.asarray(handle[name]) for name in handle.files}
    values["seed"] = np.asarray([0, 0])
    np.savez_compressed(prediction, **values)
    with pytest.raises(ValueError, match="artifact validation"):
        merge_folds(tmp_path)

    _make_complete_universe(tmp_path)
    path = tmp_path / "puzzle_set_branch5_probe_fold0_seed0.json"
    row = json.loads(path.read_text())
    row["source_provenance"]["v14_encoder_checkpoint"]["outer_fold"] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="provenance"):
        merge_folds(tmp_path)


def test_merge_rejects_different_global_input_across_folds(tmp_path: Path) -> None:
    _make_complete_universe(tmp_path)
    alternate = tmp_path / "alternate_m2.csv"
    alternate.touch()
    fold = tmp_path / "puzzle_set_branch5_probe_fold0_seed0.json"
    value = json.loads(fold.read_text())
    value["source_provenance"]["m2_csv"]["path"] = str(alternate)
    fold.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="global safe sources"):
        merge_folds(tmp_path)


def test_merge_rejects_fold_artifact_outside_frozen_input_dir(
    tmp_path: Path,
) -> None:
    _make_complete_universe(tmp_path)
    prediction = tmp_path / "puzzle_set_branch5_probe_predictions_fold0_seed0.npz"
    alternate_dir = tmp_path / "alternate"
    alternate_dir.mkdir()
    alternate = alternate_dir / prediction.name
    prediction.replace(alternate)
    fold = tmp_path / "puzzle_set_branch5_probe_fold0_seed0.json"
    value = json.loads(fold.read_text())
    value["prediction_artifact"] = str(alternate)
    fold.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen input directory"):
        merge_folds(tmp_path)


def test_merge_revalidates_frozen_checkpoint_directories(tmp_path: Path) -> None:
    _make_complete_universe(tmp_path)
    with pytest.raises(ValueError, match="checkpoint directory"):
        merge_folds(
            tmp_path,
            expected_checkpoint_dirs={
                "v13_checkpoint_dir": (tmp_path / "expected_v13").resolve(),
                "v14_checkpoint_dir": (tmp_path / "expected_v14").resolve(),
            },
        )


def test_merge_revalidates_all_frozen_global_source_paths(tmp_path: Path) -> None:
    _make_complete_universe(tmp_path)
    merge_folds(tmp_path, expected_global_paths=_expected_global_paths(tmp_path))

    expected = _expected_global_paths(tmp_path)
    expected["m2_csv_path"] = tmp_path / "alternate_m2.csv"
    with pytest.raises(ValueError, match="m2_csv_path differs"):
        merge_folds(tmp_path, expected_global_paths=expected)


def test_merge_rejects_changed_content_contrast_or_tic2a_projection(
    tmp_path: Path,
) -> None:
    _make_complete_universe(tmp_path)
    ridge = tmp_path / "puzzle_set_branch5_probe_ridge_fold0_seed0.json"
    value = json.loads(ridge.read_text())
    value["reference_preserves_position_and_region"] = False
    ridge.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact validation"):
        merge_folds(tmp_path)

    _make_complete_universe(tmp_path)
    fold = tmp_path / "puzzle_set_branch5_probe_fold0_seed0.json"
    value = json.loads(fold.read_text())
    value["source_provenance"]["tic2a_feature41_model_artifact"][
        "held_score_computed"
    ] = True
    fold.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance"):
        merge_folds(tmp_path)
