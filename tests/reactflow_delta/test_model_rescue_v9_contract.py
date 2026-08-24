from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v9m1_authorizes_only_two_fold_score_blind_smoke() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    contract = _yaml("configs/reactflow_delta/model_rescue_v9_amendment.yaml")
    assert active["authority"]["current_phase"] == "V9M1"
    assert active["runnable_phases"] == ["V9M1"]
    assert active["training_allowed"] == "V9_ZERO_MEAN_RESIDUAL_SMOKE_ONLY"
    assert active["held_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert contract["parent"]["v8_gate_changed"] is False
    assert contract["parent"]["v8m3_opened"] is False
    assert contract["contract_status"] == (
        "V9M1_TWO_FOLD_ENGINEERING_SMOKE_AUTHORIZED"
    )


def test_v9_freezes_equicalibration_and_top_journal_gate() -> None:
    contract = _yaml("configs/reactflow_delta/model_rescue_v9_amendment.yaml")
    baseline = contract["models"]["baseline"]
    candidate = contract["models"]["candidate"]
    assert baseline["residual_head_input"] == candidate["residual_head_input"]
    family = contract["models"]["identical_residual_family"]
    assert family["components"] == 2
    assert family["locations"] == "BOTH_EXACTLY_EQUAL_TO_FROZEN_SIGNED_MEAN"
    assert family["architecture_or_loss_search_allowed"] is False
    gate = contract["v9m2_screen"]["top_journal_gate"]
    assert gate["signed_delta_relative_gain_vs_feature41_min"] == 0.05
    assert gate["absolute_delta_relative_gain_vs_feature41_min"] == 0.01
    assert gate["crps_relative_gain_vs_equicalibrated_feature41_min"] == 0.05
    assert gate["crps_positive_puzzles_min"] == 16
