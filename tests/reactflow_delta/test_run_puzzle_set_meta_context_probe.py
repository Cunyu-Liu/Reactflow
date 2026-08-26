from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.model_rescue_v14 import V14PointModel
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
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import (
    EXPECTED_PROJECT_TASK,
    FOLD_SCHEMA,
    PHASE_TRAINING_TOKENS,
    _assert_parent_checkpoint_identity,
    assert_real_training_authority,
    frozen_input_sources_for_fold,
    run_prepared_fold,
    validate_fold_source_rows,
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


def _write_active(repo_root: Path, *, authorized: bool, phase: str = "P1M3") -> None:
    path = repo_root / "configs/reactflow_delta"
    path.mkdir(parents=True)
    payload = {
        "project_task_id": EXPECTED_PROJECT_TASK if authorized else "v14",
        "authority": {"current_phase": phase},
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
        "v10_fold_comparator": tmp_path / f"v10_fold_result_fold{fold}_seed0.json",
    }
    for path in paths.values():
        path.touch()
    return frozen_input_sources_for_fold(
        outer_fold=fold,
        v13_point_checkpoint=v13_parent,
        v14_encoder_checkpoint=v14_parent,
        **paths,
    )


def test_current_or_other_authority_cannot_run_real_puzzle_set_training(
    tmp_path: Path,
) -> None:
    _write_active(tmp_path, authorized=False)
    try:
        assert_real_training_authority(tmp_path, "P1M3")
    except RuntimeError as error:
        assert "not the active task" in str(error)
    else:
        raise AssertionError("non-puzzle-set authority opened real training")


def test_exact_future_authority_shape_is_accepted(tmp_path: Path) -> None:
    for phase in sorted(PHASE_TRAINING_TOKENS):
        phase_root = tmp_path / phase
        _write_active(phase_root, authorized=True, phase=phase)
        assert_real_training_authority(phase_root, phase)


def test_training_token_is_phase_specific(tmp_path: Path) -> None:
    _write_active(tmp_path, authorized=True, phase="P1M3")
    active_path = tmp_path / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    active["training_allowed"] = PHASE_TRAINING_TOKENS["P1M2"]
    active["candidate_model_training_allowed"] = PHASE_TRAINING_TOKENS["P1M2"]
    active_path.write_text(yaml.safe_dump(active), encoding="utf-8")
    try:
        assert_real_training_authority(tmp_path, "P1M3")
    except RuntimeError as error:
        assert "token is absent" in str(error)
    else:
        raise AssertionError("P1M3 accepted the P1M2 training token")


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


def test_fold_source_rows_must_match_outer_fold_and_held_puzzle() -> None:
    v8 = {
        "outer_fold": 4,
        "seed": 0,
        "held_puzzle": "P05",
        "held_score_computed": False,
        "external_outcome_accessed": False,
    }
    tic2a = {"outer_fold": 4, "held_puzzle": "P05"}
    v10 = {"outer_fold": 4, "seed": 0, "held_puzzle": "P05"}
    validate_fold_source_rows(
        outer_fold=4,
        held_puzzle="P05",
        v8_row=v8,
        tic2a_row=tic2a,
        v10_row=v10,
    )
    for row_name, field, value in (
        ("v8", "outer_fold", 3),
        ("v8", "seed", 1),
        ("v8", "held_puzzle", "P04"),
        ("v8", "held_score_computed", True),
        ("tic2a", "held_puzzle", "P04"),
        ("v10", "outer_fold", 3),
    ):
        rows = {"v8": dict(v8), "tic2a": dict(tic2a), "v10": dict(v10)}
        rows[row_name][field] = value
        try:
            validate_fold_source_rows(
                outer_fold=4,
                held_puzzle="P05",
                v8_row=rows["v8"],
                tic2a_row=rows["tic2a"],
                v10_row=rows["v10"],
            )
        except RuntimeError as error:
            assert "does not match the outer fold identity" in str(error)
        else:
            raise AssertionError(f"puzzle-set accepted wrong-fold {row_name}.{field}")


def test_frozen_input_sources_reject_wrong_fold_v8_checkpoint(tmp_path: Path) -> None:
    v13 = tmp_path / "v13_candidate_point_fold4_seed0.pt"
    v14 = tmp_path / "v14_candidate_point_fold4_seed0.pt"
    wrong_v8 = tmp_path / "v8_corrected_mean_fold3_seed0.pt"
    tic2a_model = tmp_path / "tic2a_corrected_models_fold4.json"
    registry = tmp_path / "tic2a_merged.json"
    unconstrained = tmp_path / "unconstrained.h5"
    constrained = tmp_path / "constrained.h5"
    v10 = tmp_path / "v10_fold_result_fold4_seed0.json"
    for path in (
        v13,
        v14,
        wrong_v8,
        tic2a_model,
        registry,
        unconstrained,
        constrained,
        v10,
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
            v10_fold_comparator=v10,
        )
    except RuntimeError as error:
        assert "filename changed: v8_meanaligned_checkpoint" in str(error)
    else:
        raise AssertionError("puzzle-set accepted a wrong-fold V8 checkpoint")


def test_tic2a_source_registry_requires_unique_folds_zero_through_nineteen() -> None:
    rows = [{"outer_fold": fold} for fold in range(20)]
    validate_tic2a_source_registry({"folds": rows})
    duplicated = rows + [{"outer_fold": 0}]
    try:
        validate_tic2a_source_registry({"folds": duplicated})
    except RuntimeError as error:
        assert "exactly twenty" in str(error)
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
    assert (
        result["frozen_input_sources"]["v10_fold_comparator"][
            "used_in_candidate_prediction"
        ]
        is False
    )
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
