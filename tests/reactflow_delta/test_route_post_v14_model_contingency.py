from __future__ import annotations

import json
from pathlib import Path

import yaml

import scripts.reactflow_delta.route_post_v14_model_contingency as router_module
from scripts.reactflow_delta.qualify_model_rescue_v14 import qualify
from scripts.reactflow_delta.route_post_v14_model_contingency import (
    AUDIT_FAILURE_STATUS,
    AUTHORITY_ACTION,
    AUTHORITY_TOKEN,
    main,
    route,
)
from scripts.reactflow_delta.score_model_rescue_v14 import SCHEMA as SCORE_SCHEMA


def _complete_score(
    *,
    signed_candidate: float = 0.5,
    signed_null: float = 0.8,
    point_candidate: float = 0.5,
    point_null: float = 0.8,
    crps_candidate: float = 0.5,
    crps_null: float = 0.8,
    distribution_candidate: float = 0.5,
    distribution_null: float = 0.8,
    candidate_coverage68: float = 0.68,
) -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
                "n_qualified_positions": 100,
                "n_registered_expected": 120,
                "n_registered_observed": 120,
                "feature41_signed_delta_mae": 1.0,
                "terminal_v12_signed_delta_mae": 0.9,
                "null_signed_delta_mae": signed_null,
                "candidate_signed_delta_mae": signed_candidate,
                "feature41_absolute_delta_mae": 1.0,
                "terminal_v11_point_absolute_delta_mae": 0.9,
                "null_point_absolute_delta_mae": point_null,
                "candidate_point_absolute_delta_mae": point_candidate,
                "feature41_crps": 1.0,
                "terminal_v12_crps": 0.9,
                "null_crps": crps_null,
                "candidate_crps": crps_candidate,
                "terminal_v10_distribution_absolute_delta_mae": 0.9,
                "null_distribution_absolute_delta_mae": distribution_null,
                "candidate_distribution_absolute_delta_mae": distribution_candidate,
                "feature41_coverage68": 0.68,
                "candidate_coverage68": candidate_coverage68,
                "feature41_coverage95": 0.95,
                "candidate_coverage95": 0.95,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "phase": "V14M3",
        "status": "V14M3_COMPLETE_SCORE_PASS",
        "scores": rows,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "terminal_parent_metrics_from_frozen_complete_v12_score": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def _route(score: dict) -> dict:
    return route(
        score,
        qualify(score),
        score_path="/complete-score.json",
        qualification_path="/qualification.json",
    )


def _write_router_authority(
    path: Path, *, score: Path, qualification: Path, output: Path
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "reactflow_delta.active_contract.v14",
                "project_task_id": "reactflow_delta_model_rescue_v14",
                "authority": {"current_phase": "V14M3"},
                "runnable_phases": ["V14M3"],
                "training_allowed": False,
                "candidate_model_training_allowed": False,
                "held_score_read_allowed": AUTHORITY_TOKEN,
                "partial_fold_score_read_allowed": False,
                "new_external_outcome_access_allowed": False,
                "next_allowed_action": AUTHORITY_ACTION,
                "post_v14_router_authority": {
                    "runtime_authority_token": AUTHORITY_TOKEN,
                    "complete_score_path": str(score.resolve()),
                    "qualification_path": str(qualification.resolve()),
                    "router_output_path": str(output.resolve()),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_router_branch1_opens_only_v14m4_on_exact_pass() -> None:
    result = _route(_complete_score())
    assert result["selected_router_branch_id"] == "1"
    assert result["route_classification"] == "V14M3_TOP_JOURNAL_SCREEN_PASS"
    assert result["next_action"] == "OPEN_ONLY_V14M4_FIXED_FIVE_SEED_FORMAL"


def test_router_branch2_precedes_scientific_routing_on_audit_failure() -> None:
    score = _complete_score()
    score["scores"][0]["failure_rate"] = 1.0
    result = _route(score)
    assert result["selected_router_branch_id"] == "2"
    assert result["audit_valid"] is False
    assert result["branch_predicates"]["2"] is True
    assert result["status"] == AUDIT_FAILURE_STATUS
    assert result["artifact_class"] == "ENGINEERING_EVIDENCE_AUDIT"
    assert result["scientific_route_selected"] is False


def test_router_branch3_capacity_without_pretraining_increment() -> None:
    score = _complete_score(
        signed_null=0.5,
        point_null=0.5,
        crps_null=0.5,
        distribution_null=0.5,
    )
    result = _route(score)
    assert result["selected_router_branch_id"] == "3"
    assert result["primitives"]["candidate_historical_all_full"] is True
    assert result["primitives"]["candidate_null_all_full"] is False


def test_router_branch3_wins_when_branch6_raw_predicate_also_matches() -> None:
    score = _complete_score(crps_candidate=0.8, crps_null=0.8)
    result = _route(score)
    assert result["branch_predicates"]["3"] is True
    assert result["branch_predicates"]["6"] is True
    assert result["selected_router_branch_id"] == "3"


def test_router_branch4_wins_when_branch5_raw_predicate_also_matches() -> None:
    score = _complete_score(
        signed_candidate=0.895,
        signed_null=1.1,
        point_candidate=0.895,
        point_null=1.1,
    )
    result = _route(score)
    assert result["branch_predicates"]["4"] is True
    assert result["branch_predicates"]["5"] is True
    assert result["selected_router_branch_id"] == "4"


def test_router_branch5_requires_the_frozen_linear_route_probe() -> None:
    score = _complete_score(
        signed_candidate=0.95,
        signed_null=0.96,
        point_candidate=0.95,
        point_null=0.96,
    )
    result = _route(score)
    assert result["selected_router_branch_id"] == "5"
    assert result["primitives"]["candidate_historical_point_margins_all"] is False
    assert result["primitives"]["null_historical_point_margins_all"] is False
    assert result["route_probe"] == {"requirement": "REQUIRED", "status": "NOT_RUN"}


def test_router_branch6_binds_one_pre_score_tail_diagnostic() -> None:
    score = _complete_score(crps_candidate=0.895, crps_null=0.8)
    result = _route(score)
    assert result["selected_router_branch_id"] == "6"
    assert result["primitives"]["candidate_point_all_full"] is True
    assert result["primitives"]["candidate_crps_all_full"] is False
    diagnostic = result["branch6_residual_diagnostic"]
    assert diagnostic["predeclared_before_v14_score"] is True
    assert diagnostic["primary_statistic"] == "LOWER_MINUS_UPPER_TAIL_MISS90"
    assert diagnostic["n_independent_units_required"] == 20


def test_router_falls_back_to_p3_for_calibration_only_failure() -> None:
    result = _route(_complete_score(candidate_coverage68=0.0))
    assert result["selected_router_branch_id"] == "P3"
    assert result["route_classification"] == "STOP_MODEL_RESCUE"


def test_router_recomputes_and_exactly_matches_the_full_qualification() -> None:
    score = _complete_score()
    qualification = qualify(score)
    qualification["publication_ready"] = True
    result = route(
        score,
        qualification,
        score_path="/complete-score.json",
        qualification_path="/qualification.json",
    )
    assert result["selected_router_branch_id"] == "2"
    assert any("complete structure recomputed" in item for item in result["audit_errors"])


def test_router_requires_exact_json_types_for_gates_and_score_rows() -> None:
    score = _complete_score()
    qualification = qualify(score)
    qualification["gates"]["prediction_integrity"] = 1
    invalid_gate = route(
        score,
        qualification,
        score_path="/complete-score.json",
        qualification_path="/qualification.json",
    )
    assert invalid_gate["selected_router_branch_id"] == "2"
    assert "qualification contains a non-boolean Gate" in invalid_gate["audit_errors"]

    qualification = qualify(score)
    score["scores"][0]["registered_prediction_coverage"] = "1.0"
    invalid_score = route(
        score,
        qualification,
        score_path="/complete-score.json",
        qualification_path="/qualification.json",
    )
    assert invalid_score["selected_router_branch_id"] == "2"
    assert any("invalid field" in item for item in invalid_score["audit_errors"])


def test_router_audits_registered_counts_and_exact_row_fields() -> None:
    score = _complete_score()
    score["scores"][0]["n_registered_observed"] = 119
    count_mismatch = _route(score)
    assert count_mismatch["selected_router_branch_id"] == "2"
    assert any("observed count differs" in item for item in count_mismatch["audit_errors"])

    score = _complete_score()
    score["scores"][0]["n_qualified_positions"] = 100.0
    noninteger_count = _route(score)
    assert noninteger_count["selected_router_branch_id"] == "2"
    assert any("invalid count" in item for item in noninteger_count["audit_errors"])

    score = _complete_score()
    score["scores"][0]["unregistered_extra_field"] = 0
    changed_fields = _route(score)
    assert changed_fields["selected_router_branch_id"] == "2"
    assert any("field universe changed" in item for item in changed_fields["audit_errors"])


def test_router_main_binds_resolved_cli_paths_to_active_authority(
    tmp_path: Path, monkeypatch,
) -> None:
    score_json = tmp_path / "v14-score.json"
    qualification_json = tmp_path / "v14-qualification.json"
    out_json = tmp_path / "post-v14-route.json"
    active_contract = tmp_path / "active-contract.yaml"
    score = _complete_score()
    score_json.write_text(json.dumps(score), encoding="utf-8")
    qualification_json.write_text(json.dumps(qualify(score)), encoding="utf-8")
    _write_router_authority(
        active_contract,
        score=score_json,
        qualification=qualification_json,
        output=out_json,
    )
    monkeypatch.setattr(
        router_module, "CANONICAL_ACTIVE_CONTRACT", active_contract.resolve()
    )

    assert (
        main(
            [
                "--active-contract",
                str(active_contract),
                "--score-json",
                str(score_json),
                "--qualification-json",
                str(qualification_json),
                "--out-json",
                str(out_json),
            ]
        )
        == 0
    )
    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["source_artifacts"]["complete_score"]["path"] == str(
        score_json.resolve()
    )
    assert result["source_artifacts"]["qualification"]["path"] == str(
        qualification_json.resolve()
    )


def test_router_rejects_cli_path_drift_before_reading_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    score_json = tmp_path / "canonical-score.json"
    qualification_json = tmp_path / "canonical-qualification.json"
    out_json = tmp_path / "post-v14-route.json"
    active_contract = tmp_path / "active-contract.yaml"
    _write_router_authority(
        active_contract,
        score=score_json,
        qualification=qualification_json,
        output=out_json,
    )
    monkeypatch.setattr(
        router_module, "CANONICAL_ACTIVE_CONTRACT", active_contract.resolve()
    )
    try:
        main(
            [
                "--active-contract",
                str(active_contract),
                "--score-json",
                str(tmp_path / "alternate-score.json"),
                "--qualification-json",
                str(qualification_json),
                "--out-json",
                str(out_json),
            ]
        )
    except RuntimeError as error:
        assert "complete_score_path differs from the CLI path" in str(error)
    else:
        raise AssertionError("post-V14 router accepted an unbound score path")
    assert not out_json.exists()


def test_router_refuses_to_overwrite_before_reading_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    out_json = tmp_path / "post_v14_route.json"
    original = b"do not replace the canonical router artifact\n"
    out_json.write_bytes(original)
    score_json = tmp_path / "missing-score.json"
    qualification_json = tmp_path / "missing-qualification.json"
    active_contract = tmp_path / "active-contract.yaml"
    _write_router_authority(
        active_contract,
        score=score_json,
        qualification=qualification_json,
        output=out_json,
    )
    monkeypatch.setattr(
        router_module, "CANONICAL_ACTIVE_CONTRACT", active_contract.resolve()
    )
    try:
        main(
            [
                "--active-contract",
                str(active_contract),
                "--score-json",
                str(score_json),
                "--qualification-json",
                str(qualification_json),
                "--out-json",
                str(out_json),
            ]
        )
    except FileExistsError as error:
        assert "refuses to overwrite" in str(error)
    else:
        raise AssertionError("post-V14 router overwrote its canonical artifact")
    assert out_json.read_bytes() == original


def test_router_rejects_alternate_active_contract_before_input_reads(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "repo/configs/reactflow_delta/active_contract.yaml"
    alternate = tmp_path / "alternate-active-contract.yaml"
    score_json = tmp_path / "missing-score.json"
    qualification_json = tmp_path / "missing-qualification.json"
    out_json = tmp_path / "route.json"
    _write_router_authority(
        alternate,
        score=score_json,
        qualification=qualification_json,
        output=out_json,
    )
    monkeypatch.setattr(
        router_module, "CANONICAL_ACTIVE_CONTRACT", canonical.resolve()
    )
    try:
        main(
            [
                "--active-contract",
                str(alternate),
                "--score-json",
                str(score_json),
                "--qualification-json",
                str(qualification_json),
                "--out-json",
                str(out_json),
            ]
        )
    except RuntimeError as error:
        assert "canonical active contract" in str(error)
    else:
        raise AssertionError("post-V14 router accepted an alternate authority file")
    assert not out_json.exists()
