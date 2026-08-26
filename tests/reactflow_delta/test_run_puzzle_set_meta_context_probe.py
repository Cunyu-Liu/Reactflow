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
    EXPECTED_TRAINING_TOKEN,
    FOLD_SCHEMA,
    _assert_parent_checkpoint_identity,
    assert_real_training_authority,
    run_prepared_fold,
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


def _write_active(repo_root: Path, *, authorized: bool) -> None:
    path = repo_root / "configs/reactflow_delta"
    path.mkdir(parents=True)
    payload = {
        "project_task_id": EXPECTED_PROJECT_TASK if authorized else "v14",
        "authority": {"current_phase": "P1M3"},
        "runnable_phases": ["P1M3"],
        "training_allowed": EXPECTED_TRAINING_TOKEN if authorized else False,
        "candidate_model_training_allowed": (
            EXPECTED_TRAINING_TOKEN if authorized else False
        ),
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }
    (path / "active_contract.yaml").write_text(yaml.safe_dump(payload))


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
    _write_active(tmp_path, authorized=True)
    assert_real_training_authority(tmp_path, "P1M3")


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
    result = run_prepared_fold(
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
    assert result["schema_version"] == FOLD_SCHEMA
    assert result["candidate_parameter_count"] == result["null_parameter_count"]
    assert result["null_connectivity"] == POSITION_DERANGED_NULL
    assert result["position_derangement_shift"] == POSITION_DERANGEMENT_SHIFT
    assert result["invariants"]["candidate_null_equal_attention_support"] is True
    assert result["invariants"]["attention_weight_dropout_disabled"] is True
    assert set(result["residual_parameter_counts"].values()) == {63748}
    assert result["pretraining_puzzle_ids"] == ["P01"]
    assert result["outer_train_puzzle_ids"] == ["P01"]
    assert result["held_puzzle"] not in result["pretraining_puzzle_ids"]
    assert result["expected_pretraining_eligible_construct_counts"] == [8]
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
        expected_phase="P1M3",
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
    except FileExistsError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("prepared fold runner overwrote frozen artifacts")
