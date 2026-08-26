from __future__ import annotations

import copy

import pytest

from scripts.reactflow_delta.qualify_puzzle_set_meta_context import (
    SCHEMA as SCREEN_QUALIFICATION_SCHEMA,
)
from scripts.reactflow_delta.qualify_puzzle_set_meta_context_formal import qualify
from scripts.reactflow_delta.score_puzzle_set_meta_context_formal import (
    SCHEMA as FORMAL_SCORE_SCHEMA,
)


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


def _row(fold: int, *, candidate: float = 0.80) -> dict:
    return {
        "outer_fold": fold,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "n_unexpected_prediction_keys": 0,
        "feature41_signed_delta_mae": 1.0,
        "terminal_v12_signed_delta_mae": 0.91,
        "parent_signed_delta_mae": 0.90,
        "null_signed_delta_mae": 0.86,
        "candidate_signed_delta_mae": candidate,
        "feature41_absolute_delta_mae": 1.0,
        "terminal_v11_point_absolute_delta_mae": 0.91,
        "parent_point_absolute_delta_mae": 0.90,
        "null_point_absolute_delta_mae": 0.86,
        "candidate_point_absolute_delta_mae": 0.80,
        "feature41_crps": 1.0,
        "terminal_v12_crps": 0.90,
        "null_crps": 0.86,
        "candidate_crps": candidate,
        "terminal_v10_distribution_absolute_delta_mae": 0.90,
        "null_distribution_absolute_delta_mae": 0.86,
        "candidate_distribution_absolute_delta_mae": 0.80,
        "feature41_coverage68": 0.68,
        "candidate_coverage68": 0.68,
        "feature41_coverage95": 0.95,
        "candidate_coverage95": 0.95,
    }


def _screen() -> dict:
    return {
        "schema_version": SCREEN_QUALIFICATION_SCHEMA,
        "status": "PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_PASS",
        "gate_passed": True,
        "puzzle_set_m4_authorized": True,
    }


def _formal_score() -> dict:
    rows = [_row(fold) for fold in range(20)]
    return {
        "schema_version": FORMAL_SCORE_SCHEMA,
        "status": "PUZZLE_SET_M4_COMPLETE_FORMAL_SCORE_PASS",
        "mixture_scores": copy.deepcopy(rows),
        "individual_seed_scores": {str(seed): copy.deepcopy(rows) for seed in range(5)},
        "equal_seed_mixture": True,
        "best_seed_selection_performed": False,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "v13_parent_and_feature41_replay_at_5e_7": True,
        "feature41_reference_fixed_across_seeds": True,
        "formal_assembly_reconstructed_exactly_from_merged_sources": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "context_retention_summary": _context_retention_summary(),
    }


def test_formal_gate_repeats_all_screen_gates_and_requires_seed_stability() -> None:
    result = qualify(_formal_score(), _screen())
    assert result["status"] == "PUZZLE_SET_M4_TOP_JOURNAL_FORMAL_PASS"
    assert result["gate_passed"] is True
    assert result["gates"]["screen_prerequisite_exact_pass"] is True
    assert result["gates"]["signed_positive_individual_seeds_ge_4"] is True
    assert result["gates"]["task_crps_positive_individual_seeds_ge_4"] is True
    assert result["gates"]["signed_gain_vs_terminal_v12_ge_2pct"] is True
    assert result["gates"]["point_absolute_gain_vs_v13_parent_ge_2pct"] is True


def test_formal_gate_rejects_a_screen_without_explicit_m4_authority() -> None:
    screen = _screen()
    screen["puzzle_set_m4_authorized"] = False
    with pytest.raises(ValueError, match="exact P1M3 PASS"):
        qualify(_formal_score(), screen)


def test_three_of_five_positive_seeds_cannot_pass() -> None:
    score = _formal_score()
    for seed in (3, 4):
        for row in score["individual_seed_scores"][str(seed)]:
            row["candidate_signed_delta_mae"] = 1.01
    result = qualify(score, _screen())
    assert result["gates"]["signed_positive_individual_seeds_ge_4"] is False
    assert result["gate_passed"] is False


def test_missing_individual_seed_fold_is_rejected() -> None:
    score = _formal_score()
    score["individual_seed_scores"]["2"].pop()
    with pytest.raises(ValueError, match="seed2 lacks unique folds0-19"):
        qualify(score, _screen())


def test_individual_seed_key_integrity_is_a_formal_gate() -> None:
    score = _formal_score()
    score["individual_seed_scores"]["4"][0]["failure_rate"] = 0.01
    result = qualify(score, _screen())
    assert result["gates"]["individual_seed_prediction_integrity"] is False
    assert result["gate_passed"] is False


def test_formal_mixture_does_not_drop_the_terminal_comparator_gate() -> None:
    score = _formal_score()
    for row in score["mixture_scores"]:
        row["terminal_v12_signed_delta_mae"] = 0.79
    result = qualify(score, _screen())
    assert result["gates"]["signed_gain_vs_terminal_v12_ge_2pct"] is False
    assert result["gate_passed"] is False


def test_formal_gate_rejects_negative_candidate_retention() -> None:
    score = _formal_score()
    score["context_retention_summary"][
        "candidate_pretraining_established_all_runs"
    ] = False
    result = qualify(score, _screen())
    assert result["gates"]["candidate_pretraining_established_all_runs"] is False
    assert result["gate_passed"] is False


def test_formal_gate_rejects_score_without_exact_source_assembly_link() -> None:
    score = _formal_score()
    score["formal_assembly_reconstructed_exactly_from_merged_sources"] = False
    with pytest.raises(ValueError, match="violates the frozen protocol"):
        qualify(score, _screen())
