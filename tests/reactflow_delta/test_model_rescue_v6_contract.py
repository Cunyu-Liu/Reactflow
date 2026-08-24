from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v6m2_terminal_fail_is_preserved_by_later_active_amendment() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    v6 = _yaml("configs/reactflow_delta/model_rescue_v6_amendment.yaml")
    v5 = _yaml("configs/reactflow_delta/model_rescue_v5_amendment.yaml")
    v4 = _yaml("configs/reactflow_delta/model_rescue_v4_amendment.yaml")
    v2 = _yaml("configs/reactflow_delta/model_rescue_v2_amendment.yaml")

    assert active["authority"]["current_phase"] == "V7M0"
    assert active["runnable_phases"] == ["V7M0"]
    assert active["parent_state"]["model_rescue_v6_status"] == (
        "MODEL_RESCUE_V6_FAIL_IMMUTABLE"
    )
    assert active["training_allowed"] is False
    assert active["candidate_model_training_allowed"] is False
    assert active["outcome_blind_cache_allowed"] is False
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert v6["contract_status"] == (
        "TERMINAL_V6M2_MODEL_RESCUE_V6_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    phase_status = {row["id"]: row["status"] for row in v6["phase_graph"]}
    assert phase_status["V6M1"] == "PASS"
    assert phase_status["V6M2"] == "FAIL_SIGNED_DELTA_RELATIVE_GATE"
    assert phase_status["V6M3"] == "NOT_AUTHORIZED"
    assert phase_status["V6M6"] == "FAIL_HANDOFF_COMPLETE"
    assert v6["parent"]["v5_terminal_status"] == "MODEL_RESCUE_V5_FAIL"
    assert v5["contract_status"] == (
        "TERMINAL_V5M2_MODEL_RESCUE_V5_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v4["contract_status"] == (
        "TERMINAL_V4M3_MODEL_RESCUE_V4_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v2["contract_status"] == (
        "TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY"
    )


def test_v6_has_one_fixed_constraint_protocol_and_incremental_gate() -> None:
    v6 = _yaml("configs/reactflow_delta/model_rescue_v6_amendment.yaml")
    engine = v6["constraint_engine"]
    probe = v6["eligibility_probe"]

    assert engine["version"] == "2.7.2"
    assert engine["slope_m"] == 1.8
    assert engine["intercept_b"] == -0.6
    assert engine["wt_and_mutant_share_identical_constraint_vector"] is True
    assert engine["remove_mutation_site_constraint"] is False
    assert engine["finite_negative_reactivity"] == "CLAMP_TO_ZERO"
    assert engine["missing_reactivity"] == "MINUS_999_UNCONSTRAINED"
    assert engine["searches_allowed"] == []
    assert len(engine["constrained_features"]) == 12
    assert probe["baseline_features"] == "DIRECT_18_PLUS_V5_UNCONSTRAINED_12"
    assert probe["candidate_features"] == (
        "BASELINE_PLUS_V6_CONSTRAINED_INDEPENDENT_11"
    )
    assert probe["gate"]["signed_delta_relative_mae_gain_min"] == 0.01
    invariants = probe["implementation_invariants"]
    assert invariants["baseline_replay"] == (
        "V6_BASELINE_MUST_MATCH_V5_CANDIDATE_PREDICTIONS"
    )
    assert invariants["merge_before_target_join"] is True
    assert invariants["model_or_feature_selection_allowed"] is False
    assert invariants["alpha_search_allowed"] is False
    assert invariants["cache_feature_width"] == 12
    assert invariants["constrained_probe_feature_width"] == 11


def test_v6_neural_controls_are_not_a_search_and_gate_remains_top_journal() -> None:
    v6 = _yaml("configs/reactflow_delta/model_rescue_v6_amendment.yaml")
    candidate = v6["candidate_model"]
    gate = v6["development_gate"]

    assert candidate["selection_allowed"] is False
    assert candidate["equal_structure_input_width"] == 22
    assert candidate["independent_structure_basis"]["unconstrained_width"] == 11
    assert candidate["independent_structure_basis"]["constrained_width"] == 11
    assert candidate["controls"] == [
        "b1_zero_structure_residual",
        "b1_unconstrained_ensemble_residual",
    ]
    assert gate["versus_corrected_b1"]["crps_relative_gain_min"] == 0.05
    assert gate["versus_corrected_b1"]["signed_delta_mae_relative_gain_min"] == 0.05
    assert gate["attribution"]["primary_vs_each_control_crps_ci_lower_gt"] == 0.0
    assert v6["claim_policy"]["publication_ready"] is False
