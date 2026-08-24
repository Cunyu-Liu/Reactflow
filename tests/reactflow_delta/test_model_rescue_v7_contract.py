from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v7m1_authorizes_only_outcome_blind_foundation_cache() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")

    assert active["authority"]["current_phase"] == "V7M1"
    assert active["runnable_phases"] == ["V7M1"]
    assert active["authorization"]["implementation_allowed"] is True
    assert (
        active["authorization"]["outcome_blind_foundation_preparation_allowed"]
        is True
    )
    assert active["authorization"]["outcome_blind_cache_preparation_allowed"] is True
    assert active["authorization"]["internal_development_probe_allowed"] is False
    assert active["training_allowed"] is False
    assert active["candidate_model_training_allowed"] is False
    assert active["outcome_blind_cache_allowed"] is True
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert v7["contract_status"] == (
        "V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_AUTHORIZED"
    )

    phase_status = {row["id"]: row["status"] for row in v7["phase_graph"]}
    assert phase_status["V7M0"] == "PASS"
    assert phase_status["V7M1"] == "AUTHORIZED"
    assert phase_status["V7M2"] == "NOT_AUTHORIZED"
    assert phase_status["V7M3"] == "NOT_AUTHORIZED"


def test_v7_dependency_definition_is_one_fixed_published_intervention() -> None:
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")
    method = v7["literature_and_implementation"]["dependency_method"]
    model = v7["literature_and_implementation"]["foundation_model"]
    dependency = v7["dependency_definition"]

    assert method["doi"] == "10.1038/s41588-025-02347-3"
    assert model["doi"] == "10.1038/s41467-025-60872-5"
    assert model["model_name"] == "giga-v1"
    assert model["parameters"] == 650000000
    assert model["weights_trainable"] is False
    assert dependency["sequence_input"] == (
        "FULL_UNMASKED_WT_AND_EXACT_REGISTERED_MUTANT_SEQUENCE"
    )
    assert dependency["self_dependency"] == (
        "ZERO_AT_RECEIVER_EQUAL_TO_MUTATION_SOURCE"
    )
    assert len(dependency["fixed_feature_basis"]) == 6
    assert dependency["mutant_outcome_columns_allowed"] is False
    assert dependency["external_outcome_allowed"] is False
    assert dependency["feature_or_layer_search_allowed"] is False
    assert dependency["model_size_search_allowed"] is False


def test_v7_requires_incremental_probe_and_top_journal_model_gate() -> None:
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")
    probe = v7["eligibility_probe"]
    candidate = v7["candidate_model"]
    gate = v7["development_gate"]

    assert probe["baseline_features"] == (
        "DIRECT_18_PLUS_V5_UNCONSTRAINED_12_PLUS_V6_CONSTRAINED_11"
    )
    assert probe["candidate_features"] == "BASELINE_PLUS_V7_RINALMO_DEPENDENCY_6"
    assert probe["implementation_invariants"]["baseline_replay"] == (
        "V7_BASELINE_MUST_MATCH_V6_CANDIDATE_PREDICTIONS"
    )
    assert probe["implementation_invariants"]["baseline_replay_atol"] == 1e-12
    assert probe["gate"]["signed_delta_relative_mae_gain_min"] == 0.01
    assert candidate["prerequisite"] == (
        "EXACT_V7M2_RINALMO_DEPENDENCY_SIGNAL_ELIGIBLE"
    )
    assert candidate["selection_allowed"] is False
    assert candidate["controls"] == [
        "EQUAL_CAPACITY_ZERO_DEPENDENCY_OPERATOR",
        "EQUAL_CAPACITY_HALF_LENGTH_CYCLIC_RECEIVER_SHIFTED_DEPENDENCY_OPERATOR",
    ]
    assert gate["versus_corrected_b1"]["crps_relative_gain_min"] == 0.05
    assert gate["versus_corrected_b1"]["signed_delta_mae_relative_gain_min"] == 0.05
    assert gate["attribution"]["primary_vs_each_control_crps_ci_lower_gt"] == 0.0
    assert v7["formal_confirmation"]["seeds"] == [0, 1, 2, 3, 4]
    assert v7["claim_policy"]["publication_ready"] is False


def test_v7_preserves_all_prior_terminal_states_and_running_v3() -> None:
    v7 = _yaml("configs/reactflow_delta/model_rescue_v7_amendment.yaml")
    v6 = _yaml("configs/reactflow_delta/model_rescue_v6_amendment.yaml")
    v5 = _yaml("configs/reactflow_delta/model_rescue_v5_amendment.yaml")
    v4 = _yaml("configs/reactflow_delta/model_rescue_v4_amendment.yaml")
    v2 = _yaml("configs/reactflow_delta/model_rescue_v2_amendment.yaml")

    assert v7["parent"]["v3_status_at_fork"] == (
        "R3C3_CORRECTED_EXPERT_REBUILD_IN_PROGRESS"
    )
    assert v7["parent"]["v6_terminal_status"] == "MODEL_RESCUE_V6_FAIL"
    assert v6["contract_status"] == (
        "TERMINAL_V6M2_MODEL_RESCUE_V6_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v5["contract_status"] == (
        "TERMINAL_V5M2_MODEL_RESCUE_V5_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v4["contract_status"] == (
        "TERMINAL_V4M3_MODEL_RESCUE_V4_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v2["contract_status"] == (
        "TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY"
    )
