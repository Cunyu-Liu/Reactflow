from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v10_parent_preserves_v9_terminal_and_formal_closure() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    contract = _yaml("configs/reactflow_delta/model_rescue_v9_amendment.yaml")
    assert active["authority"]["current_phase"] == "V10M0"
    assert active["runnable_phases"] == ["V10M0"]
    assert active["training_allowed"] is False
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert contract["parent"]["v8_gate_changed"] is False
    assert contract["parent"]["v8m3_opened"] is False
    assert contract["contract_status"] == (
        "TERMINAL_V9M3_CRPS_TOP_JOURNAL_MARGIN_FAIL_NO_FORMAL_CONFIRMATION"
    )
    assert contract["formal_confirmation"]["authorized"] is False
    assert contract["v9m3_terminal_result"]["v9m4_authorized"] is False
    assert contract["v9m3_terminal_result"]["all_other_prefrozen_gates_passed"] is True


def test_v9_freezes_equicalibration_and_top_journal_gate() -> None:
    contract = _yaml("configs/reactflow_delta/model_rescue_v9_amendment.yaml")
    baseline = contract["models"]["baseline"]
    candidate = contract["models"]["candidate"]
    assert baseline["residual_head_input"] == candidate["residual_head_input"]
    family = contract["models"]["identical_residual_family"]
    assert family["components"] == 2
    assert family["locations"] == "BOTH_EXACTLY_EQUAL_TO_FROZEN_SIGNED_MEAN"
    assert family["architecture_or_loss_search_allowed"] is False
    assert "FEATURE41_MEAN_REPLAYS_TIC2A_AT_1E_7" in contract["v9m1_smoke"][
        "required_invariants"
    ]
    gate = contract["v9m2_screen"]["top_journal_gate"]
    assert gate["signed_delta_relative_gain_vs_feature41_min"] == 0.05
    assert gate["absolute_delta_relative_gain_vs_feature41_min"] == 0.01
    assert gate["crps_relative_gain_vs_equicalibrated_feature41_min"] == 0.05
    assert gate["crps_positive_puzzles_min"] == 16
