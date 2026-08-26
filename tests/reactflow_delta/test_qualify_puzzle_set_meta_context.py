from __future__ import annotations

import copy

import pytest

from scripts.reactflow_delta.qualify_puzzle_set_meta_context import qualify
from scripts.reactflow_delta.score_puzzle_set_meta_context import SCHEMA as SCORE_SCHEMA


def _score_artifact() -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
                "feature41_signed_delta_mae": 1.0,
                "terminal_v12_signed_delta_mae": 0.91,
                "parent_signed_delta_mae": 0.90,
                "null_signed_delta_mae": 0.86,
                "candidate_signed_delta_mae": 0.80,
                "feature41_absolute_delta_mae": 1.0,
                "terminal_v11_point_absolute_delta_mae": 0.91,
                "parent_point_absolute_delta_mae": 0.90,
                "null_point_absolute_delta_mae": 0.86,
                "candidate_point_absolute_delta_mae": 0.80,
                "feature41_crps": 1.0,
                "terminal_v12_crps": 0.90,
                "null_crps": 0.86,
                "candidate_crps": 0.80,
                "terminal_v10_distribution_absolute_delta_mae": 0.90,
                "null_distribution_absolute_delta_mae": 0.86,
                "candidate_distribution_absolute_delta_mae": 0.80,
                "feature41_coverage68": 0.68,
                "candidate_coverage68": 0.68,
                "feature41_coverage95": 0.95,
                "candidate_coverage95": 0.95,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "PUZZLE_SET_M3_COMPLETE_SCORE_PASS",
        "scores": rows,
    }


def test_top_journal_gate_passes_only_when_all_comparators_pass() -> None:
    result = qualify(_score_artifact())
    assert result["status"] == "PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_PASS"
    assert result["gate_passed"] is True
    assert result["puzzle_set_m4_authorized"] is True
    assert all(result["gates"].values())


def test_parent_near_miss_remains_a_failure() -> None:
    score = _score_artifact()
    for row in score["scores"]:
        row["candidate_signed_delta_mae"] = 0.89
    result = qualify(score)
    assert result["status"] == "PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_FAIL"
    assert result["gate_passed"] is False
    assert result["gates"]["signed_gain_vs_v13_parent_ge_2pct"] is False


def test_qualifier_rejects_incomplete_fold_universe() -> None:
    score = _score_artifact()
    score["scores"] = score["scores"][:-1]
    with pytest.raises(ValueError, match="folds0-19"):
        qualify(score)


def test_coverage_or_unexpected_key_failure_cannot_be_overridden() -> None:
    score = copy.deepcopy(_score_artifact())
    score["scores"][0]["n_unexpected_prediction_keys"] = 1
    result = qualify(score)
    assert result["gates"]["prediction_integrity"] is False
    assert result["gate_passed"] is False
