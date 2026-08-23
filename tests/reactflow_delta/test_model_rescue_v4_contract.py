from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v4_authority_is_isolated_fail_closed_and_preserves_prior_results() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    v4 = _yaml("configs/reactflow_delta/model_rescue_v4_amendment.yaml")
    v3 = _yaml("configs/reactflow_delta/model_rescue_v3_coordinate_correction_amendment.yaml")
    v2 = _yaml("configs/reactflow_delta/model_rescue_v2_amendment.yaml")
    v1 = _yaml("configs/reactflow_delta/model_rescue_contract_v1.yaml")

    assert active["authority"]["current_phase"] == "V4M2"
    assert active["runnable_phases"] == ["V4M2"]
    assert active["training_allowed"] == "ENGINEERING_SMOKE_ONLY"
    assert active["candidate_model_training_allowed"] == "ENGINEERING_SMOKE_ONLY"
    assert active["new_external_outcome_access_allowed"] is False
    assert active["resource_partition"]["v4_allowed_physical_gpus"] == list(range(8))
    assert active["resource_partition"]["v3_preferred_physical_gpus"] == [0, 1, 2, 3, 4, 5]
    assert active["resource_partition"]["co_location_when_memory_sufficient"] is True
    assert active["parent_state"]["model_rescue_v3_disposition"] == "PRESERVE_RUNNING_DIAGNOSTIC_BASELINE_UNCHANGED"
    assert v4["parent"]["disposition"] == "PRESERVE_ALL_PRIOR_RESULTS_AND_RUNNING_V3_UNCHANGED"
    assert v3["parent"]["disposition"] == "PRESERVE_METHOD_AND_GATES_INVALIDATE_COORDINATE_FRAME"
    assert v2["contract_status"] == "TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY"
    assert v1["contract_status"] == "TERMINAL_M2_NO_RESCUE_CANDIDATE_BENCHMARK_ROUTE_LOCKED"


def test_v4_has_one_primary_model_fixed_foundation_and_no_search_surface() -> None:
    v4 = _yaml("configs/reactflow_delta/model_rescue_v4_amendment.yaml")
    primary = v4["models"]["primary"]
    search = v4["models"]["search"]
    foundation = v4["foundation"]["primary"]

    assert v4["scientific_scope"]["primary_candidate"] == "v4_dual_tower_rnafm"
    assert primary["sequence_width"] == 512
    assert primary["wt_sequence_blocks"] == 5
    assert primary["response_sequence_blocks"] == 5
    assert primary["pair_axial_blocks"] == 5
    assert primary["trainable_parameter_min"] == 35_000_000
    assert primary["trainable_parameter_max"] == 45_000_000
    assert foundation["repository_commit"] == "348951516e0963d22bbb33b3c9fc18c89081d38e"
    assert foundation["gradient_allowed"] is False
    assert foundation["exact_openknot_sequence_overlap"] == "UNKNOWN_NOT_ASSERTED"
    assert all(value is False for value in search.values())
    assert set(v4["models"]["required_controls"]) == {
        "corrected_b1",
        "v4_dual_tower_scratch",
        "v4_rnafm_only",
        "v4_capacity_matched_sequence_null",
    }


def test_v4_top_journal_gate_cannot_be_promoted_from_internal_only() -> None:
    v4 = _yaml("configs/reactflow_delta/model_rescue_v4_amendment.yaml")
    gate = v4["development_gate"]
    external = v4["external_gate"]
    claims = v4["claim_policy"]

    assert gate["versus_corrected_b1"]["crps_relative_gain_min"] == 0.05
    assert gate["versus_corrected_b1"]["signed_delta_mae_relative_gain_min"] == 0.05
    assert gate["versus_corrected_b1"]["crps_positive_puzzles_min"] == 16
    assert gate["versus_corrected_b1"]["signed_delta_mae_positive_puzzles_min"] == 16
    assert gate["attribution"]["beat_capacity_matched_null_on_both_metrics"] is True
    assert gate["complete_before_score_access"] is True
    assert external["new_independent_outcome_required"] is True
    assert external["crps_relative_gain_min"] == 0.03
    assert external["signed_delta_mae_relative_gain_min"] == 0.03
    assert claims["internal_pass"] == "HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS"
    assert claims["publication_ready"] is False
