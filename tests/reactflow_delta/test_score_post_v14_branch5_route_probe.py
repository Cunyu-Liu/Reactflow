from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    FROZEN_RUNTIME_PATHS,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_post_v14_branch5_route_probe import (
    SCORE_PHASE,
    SCORE_TOKEN,
    assert_score_authority,
    merged_integrity_pass,
    score_fold,
)


def _score_active() -> dict:
    return {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "authority": {
            "current_phase": SCORE_PHASE,
            "complete_unscored_merge_path": str(
                FROZEN_RUNTIME_PATHS["complete_unscored_merge_path"]
            ),
            "m2_csv_path": str(FROZEN_RUNTIME_PATHS["m2_csv_path"]),
            "complete_score_path": str(FROZEN_RUNTIME_PATHS["complete_score_path"]),
        },
        "runnable_phases": [SCORE_PHASE],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": SCORE_TOKEN,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "parent_state": dict(EXPECTED_PARENT_STATE),
    }


def test_score_authority_requires_score_once_and_closed_training(
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    (config / "active_contract.yaml").write_text(yaml.safe_dump(_score_active()))
    assert_score_authority(tmp_path)
    assert_score_authority(
        tmp_path,
        merged_json=FROZEN_RUNTIME_PATHS["complete_unscored_merge_path"],
        m2_csv=FROZEN_RUNTIME_PATHS["m2_csv_path"],
        out_json=FROZEN_RUNTIME_PATHS["complete_score_path"],
    )

    with pytest.raises(RuntimeError, match="CLI complete_score_path differs"):
        assert_score_authority(
            tmp_path,
            merged_json=FROZEN_RUNTIME_PATHS["complete_unscored_merge_path"],
            m2_csv=FROZEN_RUNTIME_PATHS["m2_csv_path"],
            out_json=(tmp_path / "wrong.json").resolve(),
        )

    changed = _score_active()
    changed["held_score_read_allowed"] = False
    (config / "active_contract.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(RuntimeError, match="score-once"):
        assert_score_authority(tmp_path)

    changed = _score_active()
    changed["training_allowed"] = "still_open"
    (config / "active_contract.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(RuntimeError, match="training"):
        assert_score_authority(tmp_path)

    changed = _score_active()
    changed["authority"][
        "complete_unscored_merge_path"
    ] = "/mnt/cunyuliu/wrong-merge.json"
    (config / "active_contract.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(RuntimeError, match="active authority complete_unscored"):
        assert_score_authority(tmp_path)


def test_scorer_requires_content_contrast_and_safe_tic2a_merge_provenance() -> None:
    integrity = {
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
    }
    assert merged_integrity_pass(integrity)
    integrity["samefold_v14_content_contrast_all_folds"] = False
    assert not merged_integrity_pass(integrity)


@dataclass
class Record:
    puzzle: str
    method: str
    construct_id: str
    wt_id: str
    design_pos: int
    ref: str
    alt: str


@dataclass
class Construct:
    sequence: str
    wt_observed: np.ndarray
    wt_reactivity: np.ndarray


class FakeUniverse:
    def __init__(self, records: list[Record]) -> None:
        self.records = records
        self.constructs = {
            row.construct_id: Construct(
                sequence="AA",
                wt_observed=np.asarray([True, True]),
                wt_reactivity=np.asarray([0.0, 0.0]),
            )
            for row in records
        }

    def get_construct(self, construct_id: str) -> Construct:
        return self.constructs[construct_id]

    def mutant_full_profile(self, wt_id: str, *_args):
        return np.asarray([1.0, 1.0]), np.asarray([0.1, 0.1])


def test_score_fold_uses_method_balanced_position_mutant_method_order() -> None:
    records = [
        Record("P01", "M1", "P01_M1", "wt1", 0, "A", "C"),
        Record("P01", "M2", "P01_M2", "wt2", 0, "A", "C"),
        Record("P01", "M2", "P01_M2", "wt2", 1, "A", "G"),
    ]
    univ = FakeUniverse(records)
    keys = [_bio_key(univ, row, position) for row in records for position in range(2)]
    aligned = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    prediction = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.zeros(len(keys), dtype=np.int64),
        "seed": np.zeros(len(keys), dtype=np.int64),
        "registered_status": np.full(len(keys), "covered", dtype=object),
        "parent_point": np.zeros(len(keys)),
        "aligned_point": aligned,
        "shift17_point": np.zeros(len(keys)),
    }
    result = score_fold(univ, records, prediction)
    # M1 mutant loss=0, M2's two mutant losses=1, so equal-method macro=0.5.
    # A pooled-mutant mean would be 2/3 and is intentionally not the estimand.
    assert result["aligned_signed_delta_mae"] == pytest.approx(0.5)
    assert result["aligned_point_absolute_delta_mae"] == pytest.approx(0.5)
    assert result["parent_signed_delta_mae"] == pytest.approx(1.0)
    assert result["registered_prediction_coverage"] == 1.0
    assert result["score_integrity_pass"] is True


def test_score_fold_refuses_to_score_an_incomplete_registered_universe() -> None:
    record = Record("P01", "M1", "P01_M1", "wt1", 0, "A", "C")
    univ = FakeUniverse([record])
    key = _bio_key(univ, record, 0)
    prediction = {
        "keys": np.asarray([key], dtype=object),
        "parent_point": np.asarray([0.0]),
        "aligned_point": np.asarray([0.0]),
        "shift17_point": np.asarray([0.0]),
    }
    result = score_fold(univ, [record], prediction)
    assert result["score_integrity_pass"] is False
    assert result["registered_prediction_coverage"] == pytest.approx(0.5)
    assert "aligned_signed_delta_mae" not in result
