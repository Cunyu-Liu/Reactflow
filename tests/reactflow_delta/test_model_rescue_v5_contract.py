from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v5_complete_score_authority_preserves_every_prior_authority() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    v5 = _yaml("configs/reactflow_delta/model_rescue_v5_amendment.yaml")
    v4 = _yaml("configs/reactflow_delta/model_rescue_v4_amendment.yaml")
    v3 = _yaml("configs/reactflow_delta/model_rescue_v3_coordinate_correction_amendment.yaml")
    v2 = _yaml("configs/reactflow_delta/model_rescue_v2_amendment.yaml")
    v1 = _yaml("configs/reactflow_delta/model_rescue_contract_v1.yaml")

    assert active["authority"]["current_phase"] == "V5M2"
    assert active["runnable_phases"] == ["V5M2"]
    assert active["training_allowed"] is False
    assert active["outcome_blind_cache_allowed"] is False
    assert active["held_score_read_allowed"] is True
    assert active["new_external_outcome_access_allowed"] is False
    assert v5["parent"]["disposition"] == (
        "PRESERVE_ALL_PRIOR_RESULTS_AUTHORITIES_AND_RUNNING_V3_UNCHANGED"
    )
    assert v4["contract_status"] == (
        "TERMINAL_V4M3_MODEL_RESCUE_V4_FAIL_BENCHMARK_ROUTE_LOCKED"
    )
    assert v3["contract_status"] == "R3C3_CORRECTED_EXPERT_REBUILD_IN_PROGRESS_WAITING_FOR_GPU"
    assert v2["contract_status"] == "TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY"
    assert v1["contract_status"] == "TERMINAL_M2_NO_RESCUE_CANDIDATE_BENCHMARK_ROUTE_LOCKED"


def test_v5_has_one_exact_ensemble_hypothesis_and_no_search_surface() -> None:
    v5 = _yaml("configs/reactflow_delta/model_rescue_v5_amendment.yaml")
    engine = v5["structure_engine"]
    candidate = v5["candidate_model"]

    assert v5["scientific_scope"]["primary_candidate"] == (
        "b1_exact_ensemble_delta_residual"
    )
    assert engine["algorithm"] == "GLOBAL_MCCASKILL_PARTITION_FUNCTION"
    assert engine["wt_and_exact_mutant_required"] is True
    assert engine["outcome_columns_allowed"] is False
    assert len(engine["cache_features"]) == 12
    assert candidate["base"] == "corrected_b1"
    assert candidate["base_training"].endswith("THEN_FROZEN")
    assert candidate["residual_head"] == (
        "LINEAR_GELU_LINEAR_HIDDEN_64_ZERO_INITIALIZED_OUTPUT"
    )
    assert len(candidate["disallowed"]) == 8


def test_v5_requires_signal_eligibility_and_keeps_top_journal_gate() -> None:
    v5 = _yaml("configs/reactflow_delta/model_rescue_v5_amendment.yaml")
    probe = v5["eligibility_probe"]
    gate = v5["development_gate"]

    assert probe["complete_before_score_access"] is True
    assert probe["gate"]["signed_delta_relative_mae_gain_min"] == 0.01
    assert probe["gate"]["signed_delta_positive_puzzles_min"] == 14
    assert gate["versus_corrected_b1"]["crps_relative_gain_min"] == 0.05
    assert gate["versus_corrected_b1"]["signed_delta_mae_relative_gain_min"] == 0.05
    assert gate["versus_corrected_b1"]["crps_positive_puzzles_min"] == 16
    assert gate["versus_corrected_b1"]["signed_delta_mae_positive_puzzles_min"] == 16
    assert v5["claim_policy"]["eligibility_probe_is_model_pass"] is False
    assert v5["claim_policy"]["publication_ready"] is False
