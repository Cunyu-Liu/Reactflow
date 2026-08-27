from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.qualify_post_v14_branch5_route_probe import (
    FAIL_STATUS,
    INDETERMINATE_STATUS,
    PASS_STATUS,
    QUALIFICATION_PHASE,
    assert_qualifier_authority,
    qualify,
)
from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    FROZEN_RUNTIME_PATHS,
)
from scripts.reactflow_delta.score_post_v14_branch5_route_probe import (
    COMPLETE_STATUS,
    SCHEMA as SCORE_SCHEMA,
)


def _complete_score(*, aligned: float = 0.98) -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "parent_signed_delta_mae": 1.0,
                "aligned_signed_delta_mae": aligned,
                "shift17_signed_delta_mae": 1.0,
                "parent_point_absolute_delta_mae": 1.0,
                "aligned_point_absolute_delta_mae": aligned,
                "shift17_point_absolute_delta_mae": 1.0,
                "n_registered_expected": 10,
                "n_registered_observed": 10,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
                "n_qualified_positions": 8,
                "score_integrity_pass": True,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "phase": "B5RP2",
        "status": COMPLETE_STATUS,
        "scores": rows,
        "integrity_errors": [],
        "complete_valid_score": True,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES",
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_provenance_complete": True,
    }


def test_qualifier_exact_pass_requires_all_four_comparisons() -> None:
    result = qualify(_complete_score(aligned=0.98))
    assert result["status"] == PASS_STATUS
    assert result["gate_passed"] is True
    assert result["puzzle_set_v5_eligible"] is True
    assert len(result["comparisons"]) == 4
    assert all(result["gates"].values())


def test_qualifier_complete_scientific_failure_is_not_indeterminate() -> None:
    result = qualify(_complete_score(aligned=0.995))
    assert result["status"] == FAIL_STATUS
    assert result["integrity_passed"] is True
    assert result["puzzle_set_v5_eligible"] is False
    assert result["route_after_complete_fail"] == "P3_STOP_MODEL_RESCUE"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda score: score.__setitem__("status", "NOT_COMPLETE"),
        lambda score: score["scores"].pop(),
        lambda score: score["scores"][0].__setitem__(
            "registered_prediction_coverage", 0.9
        ),
        lambda score: score["scores"][0].__setitem__(
            "registered_prediction_coverage", "invalid"
        ),
        lambda score: score["scores"][0].__setitem__("held_puzzle", "P02"),
        lambda score: score["scores"][0].__setitem__(
            "aligned_signed_delta_mae", float("nan")
        ),
        lambda score: score.__setitem__("source_provenance_complete", False),
    ],
)
def test_qualifier_integrity_failure_is_indeterminate(mutation) -> None:
    score = _complete_score()
    mutation(score)
    result = qualify(score)
    assert result["status"] == INDETERMINATE_STATUS
    assert result["integrity_passed"] is False
    assert result["puzzle_set_v5_eligible"] is False
    assert result["route_after_indeterminate"] == "P3_STOP_MODEL_RESCUE"


def test_qualifier_authority_requires_b5rp3_with_score_access_closed(
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    active = {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "authority": {
            "current_phase": QUALIFICATION_PHASE,
            "complete_score_path": str(FROZEN_RUNTIME_PATHS["complete_score_path"]),
            "qualification_path": str(FROZEN_RUNTIME_PATHS["qualification_path"]),
        },
        "runnable_phases": [QUALIFICATION_PHASE],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "parent_state": dict(EXPECTED_PARENT_STATE),
    }
    (config / "active_contract.yaml").write_text(yaml.safe_dump(active))
    assert_qualifier_authority(tmp_path)
    assert_qualifier_authority(
        tmp_path,
        score_json=FROZEN_RUNTIME_PATHS["complete_score_path"],
        out_json=FROZEN_RUNTIME_PATHS["qualification_path"],
    )

    with pytest.raises(RuntimeError, match="CLI qualification_path differs"):
        assert_qualifier_authority(
            tmp_path,
            score_json=FROZEN_RUNTIME_PATHS["complete_score_path"],
            out_json=(tmp_path / "wrong.json").resolve(),
        )

    active["held_score_read_allowed"] = True
    (config / "active_contract.yaml").write_text(yaml.safe_dump(active))
    with pytest.raises(RuntimeError, match="outcome access closed"):
        assert_qualifier_authority(tmp_path)

    active["held_score_read_allowed"] = False
    active["authority"]["complete_score_path"] = "/mnt/cunyuliu/wrong-score.json"
    (config / "active_contract.yaml").write_text(yaml.safe_dump(active))
    with pytest.raises(RuntimeError, match="active authority complete_score_path"):
        assert_qualifier_authority(tmp_path)
