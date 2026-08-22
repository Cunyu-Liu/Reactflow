from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v3_authority_is_fail_closed_and_preserves_terminal_parents() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    v3 = _yaml("configs/reactflow_delta/model_rescue_v3_amendment.yaml")
    v2 = _yaml("configs/reactflow_delta/model_rescue_v2_amendment.yaml")
    v1 = _yaml("configs/reactflow_delta/model_rescue_contract_v1.yaml")

    assert active["authority"]["machine_contract_path"] == (
        "configs/reactflow_delta/model_rescue_v3_amendment.yaml"
    )
    assert active["authority"]["current_phase"] == "R3M3"
    assert active["authority"]["binding_status"] == (
        "R3M2_REAL_DATA_ENGINEERING_SMOKE_PASS"
    )
    assert active["runnable_phases"] == ["R3M3"]
    assert active["training_allowed"] is True
    assert active["candidate_model_training_allowed"] is True
    assert active["new_external_outcome_access_allowed"] is False
    assert v3["authorization"]["current_phase"] == "R3M3"
    assert v3["authorization"]["training_allowed"] is True
    assert v3["phase_graph"][2]["status"] == "PASS"
    assert v3["phase_graph"][3]["status"] == "IN_PROGRESS"
    assert v3["r3m2_engineering_smoke"]["status"] == (
        "R3M2_REAL_DATA_ENGINEERING_SMOKE_PASS"
    )
    assert v3["parent"]["disposition"] == "IMMUTABLE_PRESERVE_UNCHANGED"
    assert v2["contract_status"] == (
        "TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY"
    )
    assert v2["r2m3_result"]["overall_status"] == "MODEL_RESCUE_V2_FAIL"
    assert v1["contract_status"] == (
        "TERMINAL_M2_NO_RESCUE_CANDIDATE_BENCHMARK_ROUTE_LOCKED"
    )


def test_v3_gate_uses_only_legal_outputs_and_has_no_search_surface() -> None:
    v3 = _yaml("configs/reactflow_delta/model_rescue_v3_amendment.yaml")
    endpoint = _yaml(
        "configs/reactflow_delta/endpoint_v7_all_mutant_full_spectrum.yaml"
    )
    gate = v3["models"]["gate"]

    assert endpoint["schema_version"].endswith(".v1")
    assert gate["feature"] == "ABS_B1_DELTA_MINUS_MEANALIGNED_DELTA"
    assert gate["threshold_quantile"] == 0.95
    assert gate["threshold_search_allowed"] is False
    assert gate["bins"] == 2
    assert gate["alpha_bounds"] == [0.0, 1.0]
    assert gate["inner_crossfit_folds"] == 4
    assert gate["inner_split_unit"] == "PUZZLE"
    assert gate["in_sample_gate_fit_allowed"] is False
    forbidden = set(v3["input_permission"]["gate_forbidden_inputs"])
    assert {"design_method", "held_target", "held_target_mask", "external_outcome"} <= forbidden


def test_v3_formal_gate_matches_frozen_v2_r2m4_thresholds() -> None:
    v3 = _yaml("configs/reactflow_delta/model_rescue_v3_amendment.yaml")[
        "formal_confirmation"
    ]
    v2 = _yaml("configs/reactflow_delta/model_rescue_v2_amendment.yaml")[
        "formal_confirmation"
    ]
    keys = {
        "seeds",
        "outer_folds",
        "ci_method",
        "crps_ci_low_gt",
        "crps_gain_min_absolute",
        "crps_gain_min_relative",
        "signed_delta_mae_ci_low_gt",
        "signed_delta_mae_gain_min_relative",
        "crps_positive_puzzles_min",
        "signed_delta_mae_positive_puzzles_min",
        "leave_one_puzzle_effect_positive",
        "max_single_puzzle_effect_fraction",
        "prediction_coverage",
        "failure_rate",
        "unexpected_keys",
        "max_coverage_error_worsening_pp",
        "coverage_guardrail_applies_separately_to",
    }
    assert {key: v3[key] for key in keys} == {key: v2[key] for key in keys}
    assert v3["required_qualification_status"] == (
        "R2M4_POST_HOC_DEVELOPMENT_PASS"
    )
