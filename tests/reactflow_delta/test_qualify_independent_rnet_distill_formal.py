from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import scripts.reactflow_delta.qualify_independent_rnet_distill_formal as qualifier


def _row(fold: int, *, candidate: float = 0.8) -> dict:
    row = {
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "feature41_signed_delta_mae": 1.0,
        "candidate_signed_delta_mae": candidate,
        "null_signed_delta_mae": 0.9,
        "historical_v14_signed_delta_mae": 0.9,
        "feature41_point_absolute_delta_mae": 1.0,
        "candidate_point_absolute_delta_mae": candidate,
        "null_point_absolute_delta_mae": 0.9,
        "historical_v14_point_absolute_delta_mae": 0.9,
        "feature41_crps": 1.0,
        "candidate_crps": candidate,
        "null_crps": 0.9,
        "historical_v14_crps": 0.9,
        "feature41_distribution_absolute_delta_mae": 1.0,
        "candidate_distribution_absolute_delta_mae": candidate,
        "null_distribution_absolute_delta_mae": 0.9,
        "historical_v10_distribution_absolute_delta_mae": 0.9,
        "feature41_coverage95": 0.95,
        "candidate_coverage95": 0.95,
        "null_coverage95": 0.95,
        "n_qualified_positions": 8,
        "n_registered_expected": 10,
        "n_registered_observed": 10,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "failed_rows": 0,
        "n_duplicate_prediction_keys": 0,
        "n_unexpected_prediction_keys": 0,
        "score_integrity_pass": True,
    }
    assert set(row) == qualifier.SCORE_ROW_FIELDS
    return row


def _formal_score(*, weak_seeds: set[int] = frozenset()) -> dict:
    mixture = [_row(fold) for fold in range(20)]
    individual = {
        str(seed): [
            _row(fold, candidate=0.95 if seed in weak_seeds else 0.8)
            for fold in range(20)
        ]
        for seed in qualifier.EXPECTED_SEEDS
    }
    score = {
        "schema_version": qualifier.FORMAL_SCORE_SCHEMA,
        "phase": qualifier.FORMAL_SCORE_PHASE,
        "status": qualifier.FORMAL_SCORE_STATUS,
        "mixture_scores": mixture,
        "individual_seed_scores": individual,
        "integrity_errors": [],
        "complete_valid_score": True,
        "complete_source_fold_seed_universe": True,
        "complete_assembly_fold_universe": True,
        "expected_fold_seed_count": qualifier.EXPECTED_FOLD_SEED_COUNT,
        "actual_fold_seed_count": qualifier.EXPECTED_FOLD_SEED_COUNT,
        "expected_fold_count": 20,
        "actual_fold_count": 20,
        "expected_seed_count": 5,
        "actual_seed_count": 5,
        "failed_rows": 0,
        "duplicate_or_unexpected_artifacts": 0,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge_and_assembly": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES_NOT_100_FOLD_SEEDS",
        "attribution_null": "RNET2_SHIFT17_SINGLE_FEATURE_DISTILLATION",
        "feature41_comparator": "AUTHORITATIVE_FEATURE41_SEED0_REPLAY_FIXED_ACROSS_SEEDS",
        "historical_parent_source": "FROZEN_V14_CANONICAL_COMPLETE_SCORE",
        "historical_distribution_comparator": "FROZEN_V10_COMPARATOR_FIXED_ACROSS_SEEDS",
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "partial_seed_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_exposure_status": qualifier.EVIDENCE_STATUS,
    }
    assert set(score) == qualifier.FORMAL_SCORE_TOP_FIELDS
    return score


def _screen_pass() -> dict:
    return {
        "schema_version": qualifier.SCREEN_QUALIFICATION_SCHEMA,
        "phase": "RND5",
        "status": qualifier.SCREEN_PASS_STATUS,
        "gate_passed": True,
        "integrity_passed": True,
        "integrity_errors": [],
        "rnd6_authorized": True,
        "model_or_threshold_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "new_external_outcome_accessed": False,
        "evidence_status": qualifier.EVIDENCE_STATUS,
        "clean_ood": "NOT_ESTABLISHED",
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def test_formal_pass_requires_mixture_and_at_least_four_positive_seeds() -> None:
    result = qualifier.qualify_formal(
        _formal_score(weak_seeds={4}),
        _screen_pass(),
        qualifier.FROZEN_SCREEN_GATES,
        qualifier.FROZEN_FORMAL_GATES,
    )
    assert result["status"] == qualifier.FORMAL_PASS_STATUS
    assert result["gate_passed"] is True
    assert set(result["positive_seed_counts"].values()) == {4}
    assert result["evidence_status"] == "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY"
    assert result["publication_ready"] is False


def test_three_positive_seeds_is_complete_scientific_fail() -> None:
    result = qualifier.qualify_formal(
        _formal_score(weak_seeds={3, 4}),
        _screen_pass(),
        qualifier.FROZEN_SCREEN_GATES,
        qualifier.FROZEN_FORMAL_GATES,
    )
    assert result["status"] == qualifier.FORMAL_FAIL_STATUS
    assert result["integrity_passed"] is True
    assert result["gate_passed"] is False
    assert set(result["positive_seed_counts"].values()) == {3}


def test_screen_prerequisite_failure_is_engineering_indeterminate() -> None:
    screen = _screen_pass()
    screen["rnd6_authorized"] = False
    result = qualifier.qualify_formal(
        _formal_score(),
        screen,
        qualifier.FROZEN_SCREEN_GATES,
        qualifier.FROZEN_FORMAL_GATES,
    )
    assert result["status"] == qualifier.FORMAL_INDETERMINATE_STATUS
    assert result["integrity_passed"] is False
    assert "screen_rnd6_authorized" in result["integrity_errors"]


def test_changed_formal_gate_is_rejected() -> None:
    changed = copy.deepcopy(qualifier.FROZEN_FORMAL_GATES)
    changed["individual_seed_positive_vs_matched_null_minimum"]["task_crps"] = 3
    with pytest.raises(RuntimeError, match="changed or lowered"):
        qualifier.qualify_formal(
            _formal_score(),
            _screen_pass(),
            qualifier.FROZEN_SCREEN_GATES,
            changed,
        )


def test_formal_gate_loader_requires_exact_contract(tmp_path: Path) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    contract = {
        "schema_version": "reactflow_delta.independent_rnet_distill_contract.v1",
        "project_task_id": qualifier.PROJECT_TASK_ID,
        "screen_gates": copy.deepcopy(qualifier.FROZEN_SCREEN_GATES),
        "formal_gates": copy.deepcopy(qualifier.FROZEN_FORMAL_GATES),
    }
    (config / "independent_rnet_distill_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )
    assert qualifier.load_frozen_formal_gates(tmp_path) == (
        qualifier.FROZEN_SCREEN_GATES,
        qualifier.FROZEN_FORMAL_GATES,
    )
