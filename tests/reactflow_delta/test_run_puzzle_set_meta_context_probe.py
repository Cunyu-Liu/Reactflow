from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.model_rescue_v14 import V14PointModel
from scripts.reactflow_delta.model_rescue_v6_probe import (
    CANDIDATE_PROBE_FEATURE_NAMES,
)
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    merge_complete_universe,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    POSITION_DERANGEMENT_SHIFT,
    POSITION_DERANGED_NULL,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    FOLD_SCOPED_INPUT_SOURCES,
    FROZEN_INPUT_SOURCE_SPEC,
    SOURCE_BINDING_STATUS,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    validate_manifest_fold_runtime_binding,
    validate_source_manifest,
)
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import (
    EXPECTED_PROJECT_TASK,
    FOLD_SCHEMA,
    PHASE_TRAINING_TOKENS,
    _assert_parent_checkpoint_identity,
    assert_real_training_authority,
    frozen_input_sources_for_fold,
    main as run_probe_main,
    run_prepared_fold,
    run_real_fold,
    safe_tic2a_source_for_fold,
    validate_tic2a_source_registry,
)


@dataclass
class _Record:
    puzzle: str
    method: str
    construct_id: str
    design_pos: int
    full_pos: int
    ref: str = "A"
    alt: str = "G"


@dataclass
class _Construct:
    sequence: str
    wt_observed: np.ndarray


class _Universe:
    def __init__(self, constructs):
        self.constructs = constructs

    def get_construct(self, construct_id):
        return self.constructs[construct_id]


def _context(length: int):
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.ones(length)
    observed = torch.ones(length)
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed, position, region


def _prepared():
    contexts = [_context(4) for _ in range(8)]
    cells = []
    held_records = []
    held_contexts = {}
    held_feature41 = {}
    constructs = {}
    for focal in range(8):
        edit = torch.tensor([focal % 4])
        distance = torch.arange(4)[None, :] - edit[:, None]
        cells.append(
            {
                "focal_construct_index": focal,
                "construct_id": f"P01_method{focal}",
                "edit_index": edit,
                "signed_distance": distance.float(),
                "refs": ["A"],
                "alts": ["G"],
                "feature41_point": torch.zeros(1, 4),
                "parent_point": torch.full((1, 4), 0.02),
                "prediction_mask": torch.ones(1, 4, dtype=torch.bool),
                "target": torch.full((1, 4), float(focal + 1) / 10.0),
                "qualified_mask": torch.ones(1, 4, dtype=torch.bool),
                "wt": torch.zeros(4),
                "feature41_basis": np.zeros((1, 4, 41), dtype=np.float32),
                "direct_features": np.zeros((1, 4, 201), dtype=np.float32),
            }
        )
        construct_id = f"P20_method{focal}"
        held_records.append(
            _Record("P20", f"method{focal}", construct_id, focal % 4, focal % 4)
        )
        held_contexts[construct_id] = _context(4)
        held_feature41[construct_id] = np.zeros((1, 4), dtype=np.float32)
        constructs[construct_id] = _Construct("ACGU", np.ones(4, dtype=bool))
    return (
        _Universe(constructs),
        {
            "pretraining_batches": [
                {
                    "puzzle": "P01",
                    "contexts": contexts,
                }
            ],
            "training_batches": [
                {
                    "puzzle": "P01",
                    "contexts": contexts,
                    "cells": cells,
                }
            ],
            "held_records": held_records,
            "held_contexts": held_contexts,
            "held_feature41": held_feature41,
            "held_parent_point": {
                construct_id: np.full((1, 4), 0.02, dtype=np.float32)
                for construct_id in held_contexts
            },
            "held_feature41_basis": {
                construct_id: np.zeros((1, 4, 41), dtype=np.float32)
                for construct_id in held_contexts
            },
            "held_direct_features": {
                construct_id: np.zeros((1, 4, 201), dtype=np.float32)
                for construct_id in held_contexts
            },
            "v14_point_state": V14PointModel().state_dict(),
            "frozen_parent_checkpoints": {},
            "coordinate_frames": {"P01": (4, 0, 4)},
        },
    )


def _source_manifest(tmp_path: Path) -> Path:
    global_paths = {
        "tic2a_merged_registry": tmp_path / "tic2a_merged.json",
        "unconstrained_feature_cache": tmp_path / "unconstrained.h5",
        "constrained_feature_cache": tmp_path / "constrained.h5",
    }
    for path in global_paths.values():
        path.touch()
    folds = []
    for fold in range(20):
        paths = {
            "v13_point_checkpoint": (
                tmp_path / f"v13_candidate_point_fold{fold}_seed0.pt"
            ),
            "v14_encoder_checkpoint": (
                tmp_path / f"v14_candidate_point_fold{fold}_seed0.pt"
            ),
            "v8_meanaligned_checkpoint": (
                tmp_path / f"v8_corrected_mean_fold{fold}_seed0.pt"
            ),
            "tic2a_feature41_model_artifact": (
                tmp_path / f"tic2a_corrected_models_fold{fold}.json"
            ),
            **global_paths,
        }
        for path in paths.values():
            path.touch(exist_ok=True)
        sources = {}
        for source_id, expected in FROZEN_INPUT_SOURCE_SPEC.items():
            sources[source_id] = {
                "path": str(paths[source_id]),
                "role": expected["role"],
                "used_in_candidate_prediction": expected[
                    "used_in_candidate_prediction"
                ],
                "outer_fold": (
                    fold if source_id in FOLD_SCOPED_INPUT_SOURCES else None
                ),
                "seed": expected["seed"],
                "realized_parameter_count": expected["realized_parameter_count"],
                "trainable_in_p1": expected["trainable_in_p1"],
            }
        folds.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "seed": 0,
                "sources": sources,
            }
        )
    manifest = tmp_path / "puzzle_set_source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_MANIFEST_SCHEMA,
                "status": SOURCE_MANIFEST_STATUS,
                "contract_id": ("reactflow_delta_puzzle_set_meta_context_v5_20260827"),
                "binding_status": SOURCE_BINDING_STATUS,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_active(
    repo_root: Path,
    *,
    authorized: bool,
    source_manifest: Path,
    phase: str = "P1M3",
) -> None:
    path = repo_root / "configs/reactflow_delta"
    path.mkdir(parents=True)
    m2_csv = (repo_root / "m2.csv").resolve()
    m2_csv.write_text("same-shape fixture\n", encoding="utf-8")
    payload = {
        "project_task_id": EXPECTED_PROJECT_TASK if authorized else "v14",
        "authority": {
            "current_phase": phase,
            "m2_csv_path": str(m2_csv),
            "source_manifest_path": str(source_manifest),
            "source_binding_status": SOURCE_BINDING_STATUS,
            "prediction_dir": str((repo_root / "predictions").resolve()),
        },
        "runnable_phases": [phase],
        "training_allowed": PHASE_TRAINING_TOKENS[phase] if authorized else False,
        "candidate_model_training_allowed": (
            PHASE_TRAINING_TOKENS[phase] if authorized else False
        ),
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }
    (path / "active_contract.yaml").write_text(yaml.safe_dump(payload))


def _frozen_input_sources(
    tmp_path: Path, *, fold: int, v13_parent: Path, v14_parent: Path
) -> dict[str, dict[str, object]]:
    paths = {
        "v8_meanaligned_checkpoint": (
            tmp_path / f"v8_corrected_mean_fold{fold}_seed0.pt"
        ),
        "tic2a_feature41_model_artifact": (
            tmp_path / f"tic2a_corrected_models_fold{fold}.json"
        ),
        "tic2a_merged_registry": tmp_path / "tic2a_merged.json",
        "unconstrained_feature_cache": tmp_path / "unconstrained.h5",
        "constrained_feature_cache": tmp_path / "constrained.h5",
    }
    for path in paths.values():
        path.touch()
    return frozen_input_sources_for_fold(
        outer_fold=fold,
        v13_point_checkpoint=v13_parent,
        v14_encoder_checkpoint=v14_parent,
        **paths,
    )


def _safe_tic2a_merge(tmp_path: Path) -> dict[str, object]:
    rows = []
    for fold in range(20):
        model = tmp_path / f"tic2a_corrected_models_fold{fold}.json"
        model.write_text(
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
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "schema_version": (
                    "reactflow_delta.target_identity_corrected_baseline_fold.v1"
                ),
                "phase": "TIC2A",
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
                "held_target_used_for_prediction": False,
                "held_score_computed": False,
                "partial_score_inspected": False,
                "legacy_prediction_reused": False,
                "external_outcome_accessed": False,
                "model_artifact": str(model),
                "prediction_artifact": str(tmp_path / f"prediction_fold{fold}.npz"),
                "n_registered_prediction_rows": 1,
                "n_train_cells": 1,
                "n_train_qualified_positions": 1,
                "n_train_valid_mutants": 1,
                "v5_v6_feature30_prediction_replay_pass": True,
                "v5_v6_feature30_stats_replay_pass": True,
            }
        )
    return {
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
        "folds": rows,
    }


def test_current_or_other_authority_cannot_run_real_puzzle_set_training(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "absent_source_manifest.json"
    _write_active(tmp_path, authorized=False, source_manifest=source_manifest)
    try:
        assert_real_training_authority(
            tmp_path,
            "P1M3",
            source_manifest,
            m2_csv=(tmp_path / "m2.csv").resolve(),
        )
    except RuntimeError as error:
        assert "not the active task" in str(error)
    else:
        raise AssertionError("non-puzzle-set authority opened real training")


def test_exact_future_authority_shape_is_accepted(tmp_path: Path) -> None:
    for phase in sorted(PHASE_TRAINING_TOKENS):
        phase_root = tmp_path / phase
        phase_root.mkdir()
        source_manifest = _source_manifest(phase_root)
        _write_active(
            phase_root,
            authorized=True,
            source_manifest=source_manifest,
            phase=phase,
        )
        assert_real_training_authority(
            phase_root,
            phase,
            source_manifest,
            m2_csv=(phase_root / "m2.csv").resolve(),
        )


def test_training_token_is_phase_specific(tmp_path: Path) -> None:
    source_manifest = _source_manifest(tmp_path)
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=source_manifest,
        phase="P1M3",
    )
    active_path = tmp_path / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    active["training_allowed"] = PHASE_TRAINING_TOKENS["P1M2"]
    active["candidate_model_training_allowed"] = PHASE_TRAINING_TOKENS["P1M2"]
    active_path.write_text(yaml.safe_dump(active), encoding="utf-8")
    try:
        assert_real_training_authority(
            tmp_path,
            "P1M3",
            source_manifest,
            m2_csv=(tmp_path / "m2.csv").resolve(),
        )
    except RuntimeError as error:
        assert "token is absent" in str(error)
    else:
        raise AssertionError("P1M3 accepted the P1M2 training token")


def test_training_token_without_bound_manifest_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing_manifest.json"
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=missing,
        phase="P1M3",
    )
    with pytest.raises(FileNotFoundError, match="source manifest"):
        assert_real_training_authority(
            tmp_path,
            "P1M3",
            missing,
            m2_csv=(tmp_path / "m2.csv").resolve(),
        )


def test_training_authority_rejects_pending_or_different_manifest(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=manifest,
        phase="P1M3",
    )
    active_path = tmp_path / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    active["authority"][
        "source_binding_status"
    ] = "REALIZED_PATHS_ROLES_AND_COUNTS_PENDING"
    active_path.write_text(yaml.safe_dump(active), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-manifest binding is absent"):
        assert_real_training_authority(
            tmp_path,
            "P1M3",
            manifest,
            m2_csv=(tmp_path / "m2.csv").resolve(),
        )

    active["authority"]["source_binding_status"] = SOURCE_BINDING_STATUS
    active_path.write_text(yaml.safe_dump(active), encoding="utf-8")
    different = tmp_path / "different_manifest.json"
    different.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-manifest binding is absent"):
        assert_real_training_authority(
            tmp_path,
            "P1M3",
            different,
            m2_csv=(tmp_path / "m2.csv").resolve(),
        )


def test_training_authority_binds_prediction_directory(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path)
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=manifest,
        phase="P1M3",
    )
    prediction_dir = (tmp_path / "predictions").resolve()
    assert_real_training_authority(
        tmp_path,
        "P1M3",
        manifest,
        m2_csv=(tmp_path / "m2.csv").resolve(),
        prediction_dir=prediction_dir,
    )
    with pytest.raises(RuntimeError, match="differs from active prediction_dir"):
        assert_real_training_authority(
            tmp_path,
            "P1M3",
            manifest,
            m2_csv=(tmp_path / "m2.csv").resolve(),
            prediction_dir=(tmp_path / "alternate").resolve(),
        )


def test_training_authority_binds_exact_m2_csv_path(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path)
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=manifest,
        phase="P1M3",
    )
    canonical = (tmp_path / "m2.csv").resolve()
    assert_real_training_authority(
        tmp_path,
        "P1M3",
        manifest,
        m2_csv=canonical,
    )

    alternate = (tmp_path / "alternate_m2.csv").resolve()
    alternate.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="M2 CSV differs from active m2_csv_path"):
        assert_real_training_authority(
            tmp_path,
            "P1M3",
            manifest,
            m2_csv=alternate,
        )


def test_main_rejects_same_shape_alternate_m2_csv_before_loading_data(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=manifest,
        phase="P1M2",
    )
    canonical = (tmp_path / "m2.csv").resolve()
    alternate = (tmp_path / "alternate_m2.csv").resolve()
    alternate.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="M2 CSV differs from active m2_csv_path"):
        run_probe_main(
            [
                "--repo-root",
                str(tmp_path),
                "--phase",
                "P1M2",
                "--m2-csv",
                str(alternate),
                "--source-manifest",
                str(manifest),
                "--v8-dir",
                str(tmp_path / "v8"),
                "--v13-dir",
                str(tmp_path / "v13"),
                "--v14-dir",
                str(tmp_path / "v14"),
                "--tic2a-merged-json",
                str(tmp_path / "tic2a.json"),
                "--unconstrained-cache",
                str(tmp_path / "unconstrained.h5"),
                "--constrained-cache",
                str(tmp_path / "constrained.h5"),
                "--out-dir",
                str(tmp_path / "predictions"),
                "--folds",
                "0",
                "--pretraining-epochs",
                "3",
                "--point-epochs",
                "3",
                "--calibration-epochs",
                "3",
                "--seed",
                "0",
            ]
        )


def test_run_real_fold_rejects_same_shape_alternate_m2_csv(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=manifest,
        phase="P1M3",
    )
    canonical = (tmp_path / "m2.csv").resolve()
    alternate = (tmp_path / "alternate_m2.csv").resolve()
    alternate.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="M2 CSV differs from active m2_csv_path"):
        run_real_fold(
            repo_root=tmp_path,
            m2_csv=alternate,
            univ=None,
            records=[],
            fold=None,
            feature41_model={},
            unconstrained=None,
            constrained=None,
            v8_model=None,
            v13_parent_checkpoint=tmp_path / "unused_v13.pt",
            v14_parent_checkpoint=tmp_path / "unused_v14.pt",
            frozen_input_sources={},
            source_manifest=manifest,
            phase="P1M3",
            seed=0,
            pretraining_epochs=200,
            point_epochs=40,
            calibration_epochs=40,
            device="cpu",
            out_dir=(tmp_path / "predictions").resolve(),
        )


def test_run_real_fold_rejects_universe_loaded_from_alternate_m2_csv(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    _write_active(
        tmp_path,
        authorized=True,
        source_manifest=manifest,
        phase="P1M3",
    )
    canonical = (tmp_path / "m2.csv").resolve()
    alternate = (tmp_path / "alternate_m2.csv").resolve()
    alternate.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    universe = type("AlternateUniverse", (), {"csv_path": alternate})()

    with pytest.raises(RuntimeError, match="loaded from a different M2 CSV"):
        run_real_fold(
            repo_root=tmp_path,
            m2_csv=canonical,
            univ=universe,
            records=[],
            fold=None,
            feature41_model={},
            unconstrained=None,
            constrained=None,
            v8_model=None,
            v13_parent_checkpoint=tmp_path / "unused_v13.pt",
            v14_parent_checkpoint=tmp_path / "unused_v14.pt",
            frozen_input_sources={},
            source_manifest=manifest,
            phase="P1M3",
            seed=0,
            pretraining_epochs=200,
            point_epochs=40,
            calibration_epochs=40,
            device="cpu",
            out_dir=(tmp_path / "predictions").resolve(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["folds"][0]["sources"][
                "v13_point_checkpoint"
            ].__setitem__("realized_parameter_count", 1),
            "source v13_point_checkpoint changed",
        ),
        (
            lambda value: value["folds"][3].__setitem__("held_puzzle", "P05"),
            "fold 3 changed",
        ),
        (
            lambda value: value["folds"][0]["sources"][
                "unconstrained_feature_cache"
            ].__setitem__("trainable_in_p1", True),
            "source unconstrained_feature_cache changed",
        ),
        (
            lambda value: value["folds"][0].__setitem__("score", 0.1),
            "fold 0 changed",
        ),
    ],
)
def test_source_manifest_rejects_field_and_count_changes(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest = _source_manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(value)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        validate_source_manifest(manifest)


def test_manifest_and_runtime_cli_paths_must_match(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path)
    rows = validate_source_manifest(manifest)
    bound = rows[4]["sources"]
    runtime = {
        source_id: {
            field: value
            for field, value in record.items()
            if field
            in {"path", "role", "used_in_candidate_prediction", "outer_fold", "seed"}
        }
        for source_id, record in bound.items()
    }
    wrong = tmp_path / "v13_candidate_point_fold4_seed0_wrong.pt"
    wrong.touch()
    runtime["v13_point_checkpoint"]["path"] = str(wrong)
    with pytest.raises(RuntimeError, match="differs from manifest"):
        validate_manifest_fold_runtime_binding(
            manifest_rows=rows,
            outer_fold=4,
            runtime_sources=runtime,
        )


def test_source_manifest_rejects_missing_realized_path(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["folds"][7]["sources"]["v14_encoder_checkpoint"]["path"] = str(
        tmp_path / "v14_candidate_point_fold7_seed0_missing.pt"
    )
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="absent or misbound"):
        validate_source_manifest(manifest)


def test_source_manifest_requires_one_global_cache_binding(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path)
    other = tmp_path / "other_constrained.h5"
    other.touch()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["folds"][8]["sources"]["constrained_feature_cache"]["path"] = str(other)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="global source differs across folds"):
        validate_source_manifest(manifest)


def test_training_source_interface_has_no_v10_input(tmp_path: Path) -> None:
    sentinel = tmp_path / "v10_fold_result_fold4_seed0.json"
    sentinel.write_text("THIS_IS_NOT_JSON_AND_MUST_NOT_BE_READ", encoding="utf-8")
    assert (
        "v10_fold_comparator"
        not in inspect.signature(frozen_input_sources_for_fold).parameters
    )
    assert "v10_dir" not in inspect.signature(run_prepared_fold).parameters
    assert (
        sentinel.read_text(encoding="utf-8") == "THIS_IS_NOT_JSON_AND_MUST_NOT_BE_READ"
    )


@pytest.mark.parametrize(
    "controller",
    [
        "run_puzzle_set_meta_context_smoke_controller.sh",
        "run_puzzle_set_meta_context_screen_controller.sh",
        "run_puzzle_set_meta_context_formal_controller.sh",
    ],
)
def test_future_controller_does_not_pass_v10_to_training_runner(
    controller: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/reactflow_delta" / controller).read_text(encoding="utf-8")
    assert "v10_dir" not in source
    assert "--v10-dir" not in source
    assert "source_manifest=" in source
    assert '--source-manifest "${source_manifest}"' in source


def test_parent_checkpoint_identity_is_fixed_to_same_fold_and_seed_zero(
    tmp_path: Path,
) -> None:
    v13 = tmp_path / "v13_candidate_point_fold4_seed0.pt"
    v14 = tmp_path / "v14_candidate_point_fold4_seed0.pt"
    v13.touch()
    v14.touch()
    _assert_parent_checkpoint_identity(
        v13_checkpoint=v13, v14_checkpoint=v14, outer_fold=4
    )
    wrong = tmp_path / "v14_candidate_point_fold5_seed0.pt"
    wrong.touch()
    try:
        _assert_parent_checkpoint_identity(
            v13_checkpoint=v13, v14_checkpoint=wrong, outer_fold=4
        )
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("puzzle-set runner accepted a wrong-fold parent")


def test_safe_tic2a_source_must_match_outer_fold_and_held_puzzle(
    tmp_path: Path,
) -> None:
    merged = _safe_tic2a_merge(tmp_path)
    observed, ridge = safe_tic2a_source_for_fold(
        outer_fold=4, held_puzzle="P05", tic2a_merged=merged
    )
    assert observed["outer_fold"] == 4
    assert np.asarray(ridge["coefficient"]).shape == (41, 2)

    merged["folds"][4]["held_puzzle"] = "P04"
    try:
        safe_tic2a_source_for_fold(outer_fold=4, held_puzzle="P05", tic2a_merged=merged)
    except RuntimeError as error:
        assert "safe TIC2A source rejected" in str(error)
    else:
        raise AssertionError("puzzle-set accepted wrong held-puzzle identity")

    try:
        safe_tic2a_source_for_fold(outer_fold=4, held_puzzle="P04", tic2a_merged=merged)
    except RuntimeError as error:
        assert "not canonical" in str(error)
    else:
        raise AssertionError("puzzle-set accepted noncanonical split-v4 identity")


def test_frozen_input_sources_reject_wrong_fold_v8_checkpoint(tmp_path: Path) -> None:
    v13 = tmp_path / "v13_candidate_point_fold4_seed0.pt"
    v14 = tmp_path / "v14_candidate_point_fold4_seed0.pt"
    wrong_v8 = tmp_path / "v8_corrected_mean_fold3_seed0.pt"
    tic2a_model = tmp_path / "tic2a_corrected_models_fold4.json"
    registry = tmp_path / "tic2a_merged.json"
    unconstrained = tmp_path / "unconstrained.h5"
    constrained = tmp_path / "constrained.h5"
    for path in (
        v13,
        v14,
        wrong_v8,
        tic2a_model,
        registry,
        unconstrained,
        constrained,
    ):
        path.touch()
    try:
        frozen_input_sources_for_fold(
            outer_fold=4,
            v13_point_checkpoint=v13,
            v14_encoder_checkpoint=v14,
            v8_meanaligned_checkpoint=wrong_v8,
            tic2a_feature41_model_artifact=tic2a_model,
            tic2a_merged_registry=registry,
            unconstrained_feature_cache=unconstrained,
            constrained_feature_cache=constrained,
        )
    except RuntimeError as error:
        assert "filename changed: v8_meanaligned_checkpoint" in str(error)
    else:
        raise AssertionError("puzzle-set accepted a wrong-fold V8 checkpoint")


def test_tic2a_source_registry_requires_unique_folds_zero_through_nineteen() -> None:
    # The shared validator checks the whole safe registry rather than accepting
    # a fold-number-only projection.
    rows = [{"outer_fold": fold} for fold in range(20)]
    try:
        validate_tic2a_source_registry({"folds": rows + [{"outer_fold": 0}]})
    except RuntimeError as error:
        assert "safe TIC2A registry rejected" in str(error)
    else:
        raise AssertionError("puzzle-set accepted a duplicate TIC2A source fold")


def test_prepared_fold_rejects_held_puzzle_pretraining(tmp_path: Path) -> None:
    univ, prepared = _prepared()
    prepared["training_batches"][0]["puzzle"] = "P20"
    prepared["pretraining_batches"][0]["puzzle"] = "P20"
    try:
        run_prepared_fold(
            univ=univ,
            prepared=prepared,
            outer_fold=19,
            held_puzzle="P20",
            phase="P1M3",
            seed=0,
            pretraining_epochs=1,
            point_epochs=1,
            calibration_epochs=1,
            device="cpu",
            out_dir=tmp_path,
        )
    except RuntimeError as error:
        assert "exclude the held puzzle" in str(error)
    else:
        raise AssertionError("puzzle-set pretraining accepted the held puzzle")


def test_prepared_fold_emits_target_free_artifacts_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    univ, prepared = _prepared()
    v13_parent = tmp_path / "v13_candidate_point_fold19_seed0.pt"
    v14_parent = tmp_path / "v14_candidate_point_fold19_seed0.pt"
    v13_parent.touch()
    v14_parent.touch()
    prepared["frozen_parent_checkpoints"] = {
        "v13_point": str(v13_parent),
        "v14_encoder": str(v14_parent),
    }
    prepared["frozen_input_sources"] = _frozen_input_sources(
        tmp_path,
        fold=19,
        v13_parent=v13_parent,
        v14_parent=v14_parent,
    )
    result = run_prepared_fold(
        univ=univ,
        prepared=prepared,
        outer_fold=19,
        held_puzzle="P20",
        phase="P1M2",
        seed=0,
        pretraining_epochs=1,
        point_epochs=1,
        calibration_epochs=1,
        device="cpu",
        out_dir=tmp_path,
    )
    assert result["schema_version"] == FOLD_SCHEMA
    assert result["candidate_parameter_count"] == result["null_parameter_count"]
    assert result["null_connectivity"] == POSITION_DERANGED_NULL
    assert result["position_derangement_shift"] == POSITION_DERANGEMENT_SHIFT
    assert result["invariants"]["candidate_null_equal_attention_support"] is True
    assert result["invariants"]["attention_weight_dropout_disabled"] is True
    assert set(result["residual_parameter_counts"].values()) == {63748}
    assert set(result["candidate_specific_trainable_parameter_counts"].values()) == {
        result["candidate_trainable_parameter_count"] + 63748
    }
    assert result["pretraining_puzzle_ids"] == ["P01"]
    assert (
        result["frozen_input_sources"]["v8_meanaligned_checkpoint"][
            "used_in_candidate_prediction"
        ]
        is True
    )
    assert "v10_fold_comparator" not in result["frozen_input_sources"]
    assert result["outer_train_puzzle_ids"] == ["P01"]
    assert result["held_puzzle"] not in result["pretraining_puzzle_ids"]
    assert result["expected_pretraining_eligible_construct_counts"] == [8]
    assert result["point_training_summaries"]["candidate"]["warmup_context_unchanged"]
    assert result["point_training_summaries"]["candidate"]["context_update_steps"] == 0
    assert set(result["context_retention_diagnostics"]) == {"candidate", "null"}
    for arm, diagnostic in result["context_retention_diagnostics"].items():
        assert diagnostic["arm"] == arm
        assert diagnostic["training_mask_epochs"] == [0, 0]
        assert diagnostic["mutant_outcome_used"] is False
        assert diagnostic["held_puzzle_accessed"] is False
    assert result["n_registered_prediction_rows"] == 32
    with np.load(result["prediction_artifact"], allow_pickle=True) as handle:
        assert not (set(handle.files) & FORBIDDEN_PREDICTION_FIELDS)
        assert len(handle["keys"]) == 32
        for name in ("candidate", "null"):
            point = torch.tensor(handle[f"{name}_point"])
            weights = torch.tensor(handle[f"{name}_weights"])
            locations = torch.tensor(handle[f"{name}_locations"])
            scales = torch.tensor(handle[f"{name}_scales"])
            cdf = torch.sum(
                weights * torch.special.ndtr((point[:, None] - locations) / scales),
                dim=-1,
            )
            assert torch.allclose(cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0)
    merged = merge_complete_universe(
        tmp_path,
        expected_phase="P1M2",
        expected_folds=[19],
        expected_seeds=[0],
        expected_pretraining_epochs=1,
        expected_point_epochs=1,
        expected_calibration_epochs=1,
        expected_parameter_count=result["candidate_parameter_count"],
        expected_trainable_parameter_count=result[
            "candidate_trainable_parameter_count"
        ],
    )
    assert merged["status"] == "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged["context_retention_gate_required"] is False
    try:
        run_prepared_fold(
            univ=univ,
            prepared=prepared,
            outer_fold=19,
            held_puzzle="P20",
            phase="P1M2",
            seed=0,
            pretraining_epochs=1,
            point_epochs=1,
            calibration_epochs=1,
            device="cpu",
            out_dir=tmp_path,
        )
    except FileExistsError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("prepared fold runner overwrote frozen artifacts")
