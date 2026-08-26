from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.puzzle_set_meta_context_data import PREDICTION_SCHEMA
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import MERGED_SCHEMA
from scripts.reactflow_delta.score_puzzle_set_meta_context import (
    EXPECTED_PHASE,
    EXPECTED_PROJECT_TASK,
    EXPECTED_SCORE_TOKEN,
    _assert_parent_and_baseline_replay,
    assert_score_authority,
    merged_integrity_pass,
    score_complete,
    score_fold,
)


@dataclass
class _Record:
    puzzle: str = "P01"
    method: str = "method0"
    construct_id: str = "P01_method0"
    design_pos: int = 1
    ref: str = "A"
    alt: str = "G"
    wt_id: str = "wt0"


@dataclass
class _Construct:
    sequence: str
    wt_observed: np.ndarray
    wt_reactivity: np.ndarray


class _Universe:
    def __init__(self) -> None:
        self.construct = _Construct(
            sequence="AC",
            wt_observed=np.ones(2, dtype=bool),
            wt_reactivity=np.zeros(2, dtype=np.float64),
        )

    def get_construct(self, _construct_id: str) -> _Construct:
        return self.construct

    def mutant_full_profile(self, *_args):
        return np.asarray([0.2, -0.1]), np.asarray([0.01, 0.01])


def _prediction() -> dict[str, np.ndarray]:
    keys = np.asarray(
        [
            "openknot_m2|P01|method0|P01_method0|1|A>G|0",
            "openknot_m2|P01|method0|P01_method0|1|A>G|1",
        ],
        dtype=object,
    )
    result = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.zeros(2, dtype=np.int64),
        "seed": np.zeros(2, dtype=np.int64),
        "registered_status": np.full(2, "covered", dtype=object),
        "feature41_point": np.asarray([0.0, 0.0]),
        "parent_point": np.asarray([0.1, -0.05]),
        "candidate_point": np.asarray([0.18, -0.09]),
        "null_point": np.asarray([0.12, -0.06]),
    }
    for name, point, scale in (
        ("candidate", result["candidate_point"], 0.05),
        ("null", result["null_point"], 0.08),
    ):
        result[f"{name}_weights"] = np.tile([0.5, 0.5], (2, 1))
        result[f"{name}_locations"] = np.stack([point, point], axis=1)
        result[f"{name}_scales"] = np.full((2, 2), scale)
        result[f"{name}_expected_absolute_delta"] = np.abs(point)
    return result


def _integrity() -> dict[str, bool]:
    return {
        "complete_fold_seed_universe": True,
        "unique_fold_seed_pairs": True,
        "prediction_only_schema": True,
        "outcome_blind_puzzle_set_inputs_all_runs": True,
        "exact_parameter_and_initialization_match_all_runs": True,
        "candidate_nonfocal_only_cross_attention_all_runs": True,
        "null_position_deranged_nonfocal_cross_attention_all_runs": True,
        "candidate_null_equal_attention_support_all_runs": True,
        "attention_weight_dropout_disabled_all_runs": True,
        "puzzle_balanced_training_all_runs": True,
        "position_aligned_nonfocal_cross_values_all_runs": True,
        "nonfocal_summary_alignment_statistics_all_runs": True,
        "matched_null_position_deranged_summary_statistics_all_runs": True,
        "nonfocal_only_cross_values_all_runs": True,
        "focal_excluded_from_cross_kv_all_runs": True,
        "eight_token_cross_support_all_runs": True,
        "paired_cross_block_reference_cancellation_all_runs": True,
        "zero_nonfocal_exact_cross_replay_all_runs": True,
        "paired_point_head_reference_cancellation_all_runs": True,
        "zero_cross_exact_parent_replay_all_runs": True,
        "fixed_position_derangement_shift_17_all_runs": True,
        "outer_train_wt_only_puzzle_set_pretraining_all_runs": True,
        "held_puzzle_excluded_from_pretraining_all_runs": True,
        "mutant_outcome_excluded_from_pretraining_all_runs": True,
        "candidate_null_equal_pretraining_budget_all_runs": True,
        "pretraining_decoder_frozen_downstream_all_runs": True,
        "encoder_and_point_unchanged_during_pretraining_all_runs": True,
        "masked_wt_pretraining_protocol_all_runs": True,
        "puzzle_coordinate_frames_validated_all_runs": True,
        "frozen_v13_point_parent_all_runs": True,
        "frozen_v14_context_encoder_all_runs": True,
        "complete_frozen_input_provenance_all_runs": True,
        "parent_replay_before_and_after_pretraining_all_runs": True,
        "point_head_only_warmup_all_runs": True,
        "point_discriminative_learning_rates_all_runs": True,
        "pretraining_capability_retention_diagnostic_complete_all_runs": True,
        "point_frozen_during_calibration_all_runs": True,
        "v10_residual_family_all_runs": True,
        "puzzle_balanced_residual_calibration_all_runs": True,
        "median_constraint_all_runs": True,
        "partial_scores_inspected": False,
        "external_outcome_accessed": False,
    }


def _context_retention_summary() -> dict[str, object]:
    return {
        "candidate_pretraining_established_all_runs": True,
        "candidate_retention_positive_all_runs": True,
        "null_pretraining_established_all_runs": True,
        "null_retention_positive_all_runs": True,
        "fold_seed_diagnostics": [],
        "selection_performed": False,
        "mutant_outcome_used": False,
        "held_puzzle_accessed": False,
    }


def test_score_fold_uses_parent_candidate_null_and_complete_key_universe() -> None:
    prediction = _prediction()
    keys = list(map(str, prediction["keys"]))
    score = score_fold(
        _Universe(),
        [_Record()],
        prediction,
        {keys[0]: 0.2, keys[1]: 0.1},
    )
    assert score["candidate_signed_delta_mae"] < score["parent_signed_delta_mae"]
    assert score["candidate_signed_delta_mae"] < score["null_signed_delta_mae"]
    assert score["registered_prediction_coverage"] == 1.0
    assert score["failure_rate"] == 0.0
    assert score["n_unexpected_prediction_keys"] == 0
    assert np.isfinite(score["candidate_crps"])


def test_score_fold_rejects_a_nonexact_registered_universe() -> None:
    prediction = _prediction()
    prediction["keys"] = prediction["keys"][:1]
    with pytest.raises(ValueError, match="registered key universes"):
        score_fold(_Universe(), [_Record()], prediction, {})


def test_parent_and_feature41_replay_is_mechanical() -> None:
    observed = {
        "feature41_signed_delta_mae": 0.3,
        "feature41_absolute_delta_mae": 0.4,
        "parent_signed_delta_mae": 0.2,
        "parent_point_absolute_delta_mae": 0.25,
    }
    reference = {
        "feature41_signed_delta_mae": 0.3,
        "feature41_absolute_delta_mae": 0.4,
        "candidate_signed_delta_mae": 0.2,
        "candidate_point_absolute_delta_mae": 0.25,
    }
    _assert_parent_and_baseline_replay(observed, reference)
    observed["parent_signed_delta_mae"] += 1e-4
    with pytest.raises(RuntimeError, match="parent/baseline replay"):
        _assert_parent_and_baseline_replay(observed, reference)


def test_score_authority_requires_complete_score_once_token(tmp_path: Path) -> None:
    active = {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "authority": {"current_phase": EXPECTED_PHASE},
        "runnable_phases": [EXPECTED_PHASE],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": EXPECTED_SCORE_TOKEN,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }
    path = tmp_path / "configs/reactflow_delta"
    path.mkdir(parents=True)
    (path / "active_contract.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    assert_score_authority(tmp_path)
    active["training_allowed"] = "still-open"
    (path / "active_contract.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    with pytest.raises(RuntimeError, match="training must be closed"):
        assert_score_authority(tmp_path)


def test_complete_merge_integrity_is_required_as_one_unit() -> None:
    integrity = _integrity()
    assert merged_integrity_pass(integrity)
    integrity["position_aligned_nonfocal_cross_values_all_runs"] = False
    assert not merged_integrity_pass(integrity)
    integrity = _integrity()
    integrity["mutant_outcome_excluded_from_pretraining_all_runs"] = False
    assert not merged_integrity_pass(integrity)
    integrity = _integrity()
    integrity["complete_frozen_input_provenance_all_runs"] = False
    assert not merged_integrity_pass(integrity)


def test_scorer_cannot_score_smoke_or_nonseedzero_merge(tmp_path: Path) -> None:
    merged = {
        "schema_version": MERGED_SCHEMA,
        "status": "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS",
        "phase": "P1M2",
        "expected_folds": [0, 1],
        "expected_seeds": [0],
        "expected_pretraining_epochs": 3,
        "expected_point_epochs": 3,
        "expected_calibration_epochs": 3,
        "context_retention_summary": _context_retention_summary(),
    }
    with pytest.raises(ValueError, match="complete unscored merge"):
        score_complete(merged, {}, {}, tmp_path / "unused.csv")
