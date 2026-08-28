from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import scripts.reactflow_delta.qualify_independent_rnet_distill as qualifier
from scripts.reactflow_delta.score_independent_rnet_distill import (
    MERGED_PATH,
    QUALIFICATION_PATH,
    SCORE_PATH,
    SCORE_ROW_FIELDS,
    SCORE_SCHEMA,
    SCORE_STATUS,
    SCREEN_DIR,
)


def _row(fold: int, *, candidate: float = 0.8, coverage95: float = 0.95) -> dict:
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
        "candidate_coverage95": coverage95,
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
    assert set(row) == SCORE_ROW_FIELDS
    return row


def _complete_score(*, candidate: float = 0.8, coverage95: float = 0.95) -> dict:
    return {
        "schema_version": SCORE_SCHEMA,
        "phase": "RND4",
        "status": SCORE_STATUS,
        "scores": [
            _row(fold, candidate=candidate, coverage95=coverage95)
            for fold in range(20)
        ],
        "integrity_errors": [],
        "complete_valid_score": True,
        "complete_fold_artifact_universe": True,
        "expected_fold_count": 20,
        "actual_fold_count": 20,
        "failed_rows": 0,
        "duplicate_or_unexpected_artifacts": 0,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES",
        "attribution_null": "RNET2_SHIFT17_SINGLE_FEATURE_DISTILLATION",
        "feature41_comparator": "AUTHORITATIVE_FEATURE41_SEED0_REPLAY",
        "historical_parent_source": "FROZEN_V14_CANONICAL_COMPLETE_SCORE",
        "historical_distribution_comparator": (
            "FROZEN_V10_COMPARATOR_CARRIED_IN_CURRENT_PREDICTION"
        ),
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_exposure_status": qualifier.EVIDENCE_STATUS,
    }


def test_exact_pass_applies_all_frozen_comparator_gates_and_evidence_ceiling() -> None:
    result = qualifier.qualify(
        _complete_score(), qualifier.FROZEN_SCREEN_GATES
    )
    assert result["status"] == qualifier.PASS_STATUS
    assert result["gate_passed"] is True
    assert result["rnd6_authorized"] is True
    assert len(result["comparisons"]) == 12
    assert all(result["gates"].values())
    assert result["evidence_status"] == "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY"
    assert result["clean_ood"] == "NOT_ESTABLISHED"
    assert result["sota"] == "NOT_ESTABLISHED"
    assert result["publication_ready"] is False


def test_complete_scientific_gate_failure_is_not_engineering_indeterminate() -> None:
    result = qualifier.qualify(
        _complete_score(candidate=0.895), qualifier.FROZEN_SCREEN_GATES
    )
    assert result["status"] == qualifier.FAIL_STATUS
    assert result["gate_passed"] is False
    assert result["integrity_passed"] is True
    assert result["rnd6_authorized"] is False
    assert not all(result["gates"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda score: score.__setitem__("status", "NOT_COMPLETE"),
        lambda score: score["scores"].pop(),
        lambda score: score["scores"][0].__setitem__(
            "registered_prediction_coverage", 0.9
        ),
        lambda score: score["scores"][0].__setitem__("failed_rows", 1),
        lambda score: score["scores"][0].__setitem__(
            "candidate_signed_delta_mae", float("nan")
        ),
        lambda score: score.__setitem__(
            "source_exposure_status", "CLEAN_OOD"
        ),
    ],
)
def test_integrity_failure_is_engineering_indeterminate(mutation) -> None:
    score = _complete_score()
    mutation(score)
    result = qualifier.qualify(score, qualifier.FROZEN_SCREEN_GATES)
    assert result["status"] == qualifier.INDETERMINATE_STATUS
    assert result["integrity_passed"] is False
    assert result["rnd6_authorized"] is False
    assert result["evidence_status"] == "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY"


def test_changed_or_lowered_contract_gate_is_rejected() -> None:
    changed = copy.deepcopy(qualifier.FROZEN_SCREEN_GATES)
    changed["matched_null_relative_gain_minimum"]["signed_delta"] = 0.0
    with pytest.raises(RuntimeError, match="changed or lowered"):
        qualifier.qualify(_complete_score(), changed)


def test_frozen_gate_loader_reads_exact_machine_contract(
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    contract = {
        "schema_version": "reactflow_delta.independent_rnet_distill_contract.v1",
        "project_task_id": qualifier.PROJECT_TASK_ID,
        "screen_gates": copy.deepcopy(qualifier.FROZEN_SCREEN_GATES),
    }
    path = config / "independent_rnet_distill_contract.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    assert qualifier.load_frozen_screen_gates(tmp_path) == qualifier.FROZEN_SCREEN_GATES

    contract["screen_gates"]["single_puzzle_influence_maximum"] = 1.0
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differ"):
        qualifier.load_frozen_screen_gates(tmp_path)


def test_qualifier_authority_uses_independent_validator_and_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    active = {
        "project_task_id": qualifier.PROJECT_TASK_ID,
        "authority": {
            "screen_prediction_dir": str(SCREEN_DIR),
            "complete_unscored_merge_path": str(MERGED_PATH),
            "complete_score_path": str(SCORE_PATH),
            "qualification_path": str(QUALIFICATION_PATH),
        },
    }
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(active), encoding="utf-8"
    )
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        qualifier,
        "assert_run_authority",
        lambda root, phase: calls.append((root, phase)),
    )
    qualifier.assert_qualifier_authority(
        tmp_path, score_json=SCORE_PATH, out_json=QUALIFICATION_PATH
    )
    assert calls == [(tmp_path, "RND5")]

    with pytest.raises(RuntimeError, match="qualification_path differs"):
        qualifier.assert_qualifier_authority(
            tmp_path, score_json=SCORE_PATH, out_json=tmp_path / "wrong.json"
        )


def test_main_writes_complete_scientific_fail_then_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score_path = tmp_path / "score.json"
    out_path = tmp_path / "qualification.json"
    score_path.write_text(json.dumps(_complete_score(candidate=0.895)), encoding="utf-8")
    monkeypatch.setattr(qualifier, "assert_qualifier_authority", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        qualifier,
        "load_frozen_screen_gates",
        lambda _root: qualifier.FROZEN_SCREEN_GATES,
    )
    exit_code = qualifier.main(
        [
            "--repo-root",
            str(tmp_path),
            "--score-json",
            str(score_path),
            "--out-json",
            str(out_path),
        ]
    )
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert result["status"] == qualifier.FAIL_STATUS
    assert result["integrity_passed"] is True


def test_main_writes_indeterminate_then_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _complete_score()
    score["scores"].pop()
    score_path = tmp_path / "score.json"
    out_path = tmp_path / "qualification.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    monkeypatch.setattr(qualifier, "assert_qualifier_authority", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        qualifier,
        "load_frozen_screen_gates",
        lambda _root: qualifier.FROZEN_SCREEN_GATES,
    )
    exit_code = qualifier.main(
        [
            "--repo-root",
            str(tmp_path),
            "--score-json",
            str(score_path),
            "--out-json",
            str(out_path),
        ]
    )
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert result["status"] == qualifier.INDETERMINATE_STATUS
    assert result["integrity_passed"] is False
