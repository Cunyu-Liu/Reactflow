from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v8_terminal_preserves_gate_failure_and_closes_training() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    contract = _yaml("configs/reactflow_delta/model_rescue_v8_amendment.yaml")
    assert active["authority"]["current_phase"] == "V8M6"
    assert active["runnable_phases"] == ["M6"]
    assert active["training_allowed"] is False
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert active["legacy_target_dependent_prediction_reuse_allowed"] is False
    assert active["legacy_v3_expert_reuse_allowed"] is False
    assert contract["contract_status"] == (
        "TERMINAL_V8M2_ABSOLUTE_GUARDRAIL_FAIL_SIGNED_MEAN_BREAKTHROUGH"
    )
    assert contract["parent"]["v3_checkpoint_or_prediction_reuse_allowed"] is False
    status = {row["id"]: row["status"] for row in contract["phase_graph"]}
    assert status["V8M1"] == "PASS"
    assert status["V8M2"] == "FAIL"
    assert status["V8M3"] == "PERMANENTLY_NOT_AUTHORIZED"
    result = contract["v8m2_mean_screen"]["terminal_result"]
    assert result["signed_delta_relative_gain_vs_feature41"] > 0.08
    assert result["absolute_delta_relative_gain_vs_feature41"] < -0.005
    assert result["v8m3_authorized"] is False


def test_v8_freezes_mean_and_top_journal_gates() -> None:
    contract = _yaml("configs/reactflow_delta/model_rescue_v8_amendment.yaml")
    mean_gate = contract["v8m2_mean_screen"]["gate"]
    assert mean_gate["signed_delta_relative_gain_vs_feature41_min"] == 0.01
    assert mean_gate["signed_delta_relative_gain_vs_b1_min"] == 0.01
    assert mean_gate["positive_puzzles_vs_feature41_min"] == 14
    large = contract["v8m3_large_residual_model"]
    assert large["fixed_architecture"] == {
        "feature41_outer_train_base": True,
        "final_mean": "FEATURE41_BASE_PLUS_NEURAL_RESIDUAL",
        "d": 192,
        "heads": 8,
        "attention_blocks": 4,
        "hidden": 128,
        "objective": "EXACT_METHOD_BALANCED_SIGNED_DELTA_L1",
        "epochs": 40,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_clip": 5.0,
    }
    gate = large["top_journal_screen_gate"]
    assert gate["signed_delta_relative_gain_vs_feature41_min"] == 0.05
    assert gate["crps_relative_gain_vs_feature41_min"] == 0.05
    assert gate["positive_puzzles_both_min"] == 16
    assert contract["formal_confirmation"]["seeds"] == [0, 1, 2, 3, 4]
