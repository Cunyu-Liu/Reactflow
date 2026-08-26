from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    merge_complete_universe,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
)
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import (
    EXPECTED_PROJECT_TASK,
    EXPECTED_TRAINING_TOKEN,
    FOLD_SCHEMA,
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
            "held_feature41_basis": {
                construct_id: np.zeros((1, 4, 41), dtype=np.float32)
                for construct_id in held_contexts
            },
            "held_direct_features": {
                construct_id: np.zeros((1, 4, 201), dtype=np.float32)
                for construct_id in held_contexts
            },
        },
    )


def _write_active(repo_root: Path, *, authorized: bool) -> None:
    path = repo_root / "configs/reactflow_delta"
    path.mkdir(parents=True)
    payload = {
        "project_task_id": EXPECTED_PROJECT_TASK if authorized else "v14",
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
        assert_real_training_authority(tmp_path)
    except RuntimeError as error:
        assert "not the active task" in str(error)
    else:
        raise AssertionError("non-puzzle-set authority opened real training")


def test_exact_future_authority_shape_is_accepted(tmp_path: Path) -> None:
    _write_active(tmp_path, authorized=True)
    assert_real_training_authority(tmp_path)


def test_prepared_fold_emits_target_free_artifacts_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    univ, prepared = _prepared()
    result = run_prepared_fold(
        univ=univ,
        prepared=prepared,
        outer_fold=19,
        held_puzzle="P20",
        seed=0,
        point_epochs=1,
        calibration_epochs=1,
        device="cpu",
        out_dir=tmp_path,
    )
    assert result["schema_version"] == FOLD_SCHEMA
    assert result["candidate_parameter_count"] == result["null_parameter_count"]
    assert set(result["residual_parameter_counts"].values()) == {63748}
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
                weights
                * torch.special.ndtr((point[:, None] - locations) / scales),
                dim=-1,
            )
            assert torch.allclose(
                cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0
            )
    merged = merge_complete_universe(
        tmp_path,
        expected_folds=[19],
        expected_seeds=[0],
        expected_point_epochs=1,
        expected_calibration_epochs=1,
        expected_parameter_count=result["candidate_parameter_count"],
    )
    assert merged["status"] == "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
    try:
        run_prepared_fold(
            univ=univ,
            prepared=prepared,
            outer_fold=19,
            held_puzzle="P20",
            seed=0,
            point_epochs=1,
            calibration_epochs=1,
            device="cpu",
            out_dir=tmp_path,
        )
    except FileExistsError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("prepared fold runner overwrote frozen artifacts")
