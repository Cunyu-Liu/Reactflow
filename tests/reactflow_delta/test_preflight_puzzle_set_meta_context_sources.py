from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.preflight_puzzle_set_meta_context_sources import (
    EXPECTED_CONTRACT_SOURCE_SPECS,
    EXPECTED_PARAMETER_COUNTS,
    build_preflight,
    ridge_parameter_counts,
)
from scripts.reactflow_delta.model_rescue_v6_probe import CANDIDATE_PROBE_FEATURE_NAMES


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, v14_folds: set[int]) -> dict[str, Path]:
    paths = {
        "inactive_contract": tmp_path / "puzzle_set_v5.yaml",
        "v8_dir": tmp_path / "v8",
        "v13_dir": tmp_path / "v13",
        "v14_dir": tmp_path / "v14",
        "tic2a_merged_json": tmp_path / "tic2a" / "merged.json",
        "unconstrained_cache": tmp_path / "unconstrained.h5",
        "constrained_cache": tmp_path / "constrained.h5",
    }
    for name in ("v8_dir", "v13_dir", "v14_dir"):
        paths[name].mkdir(parents=True)
    paths["inactive_contract"].write_text(
        yaml.safe_dump(
            {
                "contract_id": "reactflow_delta_puzzle_set_meta_context_v5_20260827",
                "contract_status": "DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE",
                "inactive_authority": {
                    "activation_allowed_now": False,
                    "training_allowed": False,
                },
                "frozen_input_sources": copy.deepcopy(EXPECTED_CONTRACT_SOURCE_SPECS),
            }
        ),
        encoding="utf-8",
    )
    paths["unconstrained_cache"].touch()
    paths["constrained_cache"].touch()
    tic_rows = []
    for fold in range(20):
        held = f"P{fold + 1:02d}"
        v8_checkpoint = paths["v8_dir"] / f"v8_corrected_mean_fold{fold}_seed0.pt"
        v8_checkpoint.touch()
        v13_checkpoint = paths["v13_dir"] / f"v13_candidate_point_fold{fold}_seed0.pt"
        v13_checkpoint.touch()
        if fold in v14_folds:
            (paths["v14_dir"] / f"v14_candidate_point_fold{fold}_seed0.pt").touch()
        model_path = tmp_path / "tic2a" / f"tic2a_corrected_models_fold{fold}.json"
        _write_json(
            model_path,
            {
                "direct18": {},
                "direct18_feature_names": [f"direct_{index}" for index in range(18)],
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
            },
        )
        tic_rows.append(
            {
                "schema_version": (
                    "reactflow_delta.target_identity_corrected_baseline_fold.v1"
                ),
                "phase": "TIC2A",
                "outer_fold": fold,
                "held_puzzle": held,
                "model_artifact": str(model_path),
                "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
                "held_target_used_for_prediction": False,
                "held_score_computed": False,
                "partial_score_inspected": False,
                "legacy_prediction_reused": False,
                "external_outcome_accessed": False,
                "prediction_artifact": str(
                    tmp_path / "tic2a" / f"prediction_fold{fold}.npz"
                ),
                "n_registered_prediction_rows": 1,
                "n_train_cells": 1,
                "n_train_qualified_positions": 1,
                "n_train_valid_mutants": 1,
                "v5_v6_feature30_prediction_replay_pass": True,
                "v5_v6_feature30_stats_replay_pass": True,
            }
        )
    _write_json(
        paths["tic2a_merged_json"],
        {
            "schema_version": (
                "reactflow_delta.target_identity_corrected_baseline_merged.v1"
            ),
            "phase": "TIC2A",
            "status": "TIC2A_COMPLETE_UNSCORED_MERGE_PASS",
            "merge_integrity": {
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
            },
            "folds": tic_rows,
        },
    )
    return paths


def _fake_checkpoint_inspector(**kwargs: Path | None) -> dict[str, int | None]:
    return {
        "v8_meanaligned_checkpoint": EXPECTED_PARAMETER_COUNTS[
            "v8_meanaligned_checkpoint"
        ],
        "v13_point_checkpoint": EXPECTED_PARAMETER_COUNTS["v13_point_checkpoint"],
        "v14_encoder_checkpoint": (
            EXPECTED_PARAMETER_COUNTS["v14_encoder_checkpoint"]
            if kwargs["v14_checkpoint"] is not None
            else None
        ),
    }


def _fake_cache_inspector(**_kwargs: Path) -> dict[str, int | bool]:
    return {
        "biological_key_universe_equal": True,
        "registered_mutants": 13_976,
        "receiver_length": 177,
        "unconstrained_width": 12,
        "constrained_cache_width": 12,
        "constrained_probe_width": 11,
    }


def _preflight(paths: dict[str, Path]) -> dict:
    return build_preflight(
        **paths,
        checkpoint_inspector=_fake_checkpoint_inspector,
        cache_inspector=_fake_cache_inspector,
    )


def test_ridge_parameter_counts_distinguish_predictor_from_fitted_state() -> None:
    counts = ridge_parameter_counts(
        {
            "mean_x": [0.0] * 41,
            "scale_x": [1.0] * 41,
            "mean_y": [0.0, 0.0],
            "coefficient": [[0.0, 0.0] for _ in range(41)],
            "alpha": 1.0,
        }
    )
    assert counts == {
        "predictive_parameter_count": 84,
        "stored_fitted_scalar_count": 166,
    }


def test_preflight_reports_only_active_v14_as_incomplete(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, v14_folds={1, 3, 4})
    report = _preflight(paths)
    assert (
        report["status"]
        == "SOURCE_PREFLIGHT_WAITING_FOR_V14_CHECKPOINT_FILENAME_AND_ARCHITECTURE_UNIVERSE"
    )
    assert report["non_v14_checkpoint_filename_and_architecture_folds"] == list(
        range(20)
    )
    assert report["v14_checkpoint_filename_and_architecture_folds"] == [1, 3, 4]
    assert report["missing_v14_checkpoint_folds"] == [
        fold for fold in range(20) if fold not in {1, 3, 4}
    ]
    assert report["activation_authorized_by_report"] is False
    assert report["activation_ready"] is False
    assert report["binding_candidate"] is None
    assert report["scientific_score_fields_read"] is False
    assert report["m2_target_table_read"] is False
    assert report["active_contract_path_accepted_or_read"] is False
    assert report["terminal_safe_provenance_manifest_accepted_or_read"] is False
    assert report["wide_v8_v13_result_content_read"] is False
    assert (
        report["parameter_accounting"][
            "realized_checkpoint_filename_and_architecture_universe"
        ]
        is None
    )


def test_complete_preflight_measures_full_inference_footprint(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, v14_folds=set(range(20)))
    report = _preflight(paths)
    assert (
        report["status"]
        == "SOURCE_CHECKPOINT_FILENAME_AND_ARCHITECTURE_UNIVERSE_COMPLETE_TERMINAL_SAFE_PROVENANCE_REQUIRED"
    )
    assert report["missing_v14_checkpoint_folds"] == []
    accounting = report["parameter_accounting"]["expected_at_complete"]
    assert accounting["tic2a_feature41_ridge_predictive_parameters"] == 84
    assert accounting["tic2a_feature41_ridge_stored_fitted_scalars"] == 166
    assert accounting["all_consumed_frozen_source_parameters"] == 6_941_682
    assert accounting["external_to_p1_module_upstream_parameters"] == 2_174_402
    assert (
        accounting["candidate_full_prediction_learned_parameter_footprint"] == 8_409_847
    )
    assert accounting["v14_encoder_already_included_in_p1_point_module"] is True
    assert report["authority_modified"] is False
    assert (
        report["checkpoint_same_fold_provenance_binding"]["status"]
        == "TERMINAL_SAFE_PROVENANCE_MANIFEST_REQUIRED"
    )
    assert report["activation_ready"] is False


def test_preflight_rejects_noncanonical_safe_registry_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, v14_folds=set())
    merged = paths["tic2a_merged_json"]
    value = json.loads(merged.read_text(encoding="utf-8"))
    value["folds"][7]["held_puzzle"] = "P09"
    _write_json(merged, value)
    with pytest.raises(ValueError, match="safe registry row changed"):
        _preflight(paths)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("status", "TIC2A_PARTIAL"),
        lambda value: value["merge_integrity"].__setitem__(
            "partial_score_inspected", True
        ),
        lambda value: value["folds"][2].__setitem__("phase", "WRONG"),
        lambda value: value["folds"][2].__setitem__("legacy_prediction_reused", True),
        lambda value: value["folds"][2].__setitem__("held_score_computed", True),
    ],
)
def test_preflight_rejects_tic2a_registry_tampering(tmp_path: Path, mutation) -> None:
    paths = _fixture(tmp_path, v14_folds=set())
    merged = paths["tic2a_merged_json"]
    value = json.loads(merged.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(merged, value)
    with pytest.raises(ValueError, match="TIC2A"):
        _preflight(paths)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("feature41_feature_names", ["wrong"] * 41),
        lambda value: value["v6_feature41"].__setitem__("alpha", 2.0),
        lambda value: value["v6_feature41"].__setitem__("scale_x", [0.0] * 41),
        lambda value: value["v6_feature41"].__setitem__(
            "coefficient", [[float("nan"), 0.0]] * 41
        ),
        lambda value: value["v6_feature41"].__setitem__("extra", 1),
    ],
)
def test_preflight_rejects_tic2a_feature41_model_tampering(
    tmp_path: Path, mutation
) -> None:
    paths = _fixture(tmp_path, v14_folds=set())
    merged = json.loads(paths["tic2a_merged_json"].read_text(encoding="utf-8"))
    model = Path(merged["folds"][4]["model_artifact"])
    value = json.loads(model.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(model, value)
    with pytest.raises(ValueError, match="TIC2A"):
        _preflight(paths)


def test_preflight_rejects_wrong_inactive_contract_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, v14_folds=set())
    paths["inactive_contract"].write_text(
        yaml.safe_dump(
            {
                "contract_id": "wrong",
                "contract_status": "DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE",
                "inactive_authority": {
                    "activation_allowed_now": False,
                    "training_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inactive contract identity"):
        _preflight(paths)


def test_preflight_rejects_changed_contract_source_role(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, v14_folds=set())
    contract = yaml.safe_load(paths["inactive_contract"].read_text(encoding="utf-8"))
    contract["frozen_input_sources"]["v13_seed0_point"]["role"] = "WRONG"
    paths["inactive_contract"].write_text(yaml.safe_dump(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="v13_seed0_point.role changed"):
        _preflight(paths)
