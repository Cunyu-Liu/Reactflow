from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v10m1_pass_authorizes_only_fixed_complete_universe_screen() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    contract = _yaml("configs/reactflow_delta/model_rescue_v10_amendment.yaml")
    assert active["authority"]["current_phase"] == "V10M2"
    assert active["runnable_phases"] == ["V10M2"]
    assert active["training_allowed"] == "V10_FIXED_SEED0_TWENTY_FOLD_SCREEN_ONLY"
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert contract["parent"]["v9_gate_changed"] is False
    assert contract["parent"]["v9m4_opened"] is False
    assert contract["formal_confirmation"]["authorized"] is False
    assert contract["contract_status"] == (
        "V10M1_ENGINEERING_SMOKE_PASS_V10M2_SCREEN_AUTHORIZED"
    )
    assert contract["v10m1_smoke"]["scientific_scores_read"] is False
    assert contract["v10m2_screen"]["complete_before_score"] is True


def test_v10_freezes_identification_ladder_and_top_journal_gate() -> None:
    contract = _yaml("configs/reactflow_delta/model_rescue_v10_amendment.yaml")
    models = contract["models"]
    assert models["meanaligned_capacity_symmetric"]["parameters"] == 63491
    assert models["meanaligned_median_asymmetric"]["parameters"] == 63748
    assert (
        models["meanaligned_median_asymmetric"]["parameters"]
        - models["meanaligned_capacity_symmetric"]["parameters"]
    ) == 257
    assert contract["input"]["width"] == 244
    assert contract["residual_family"]["cdf_interior_epsilon"] == 1.0e-4
    gate = contract["v10m2_screen"]["top_journal_gate"]
    assert gate["signed_delta_relative_gain_vs_feature41_min"] == 0.05
    assert (
        gate["absolute_delta_relative_gain_vs_independent_feature41_head_min"]
        == 0.05
    )
    assert gate["crps_relative_gain_vs_feature41_asymmetric_min"] == 0.05
    assert gate["crps_relative_gain_vs_historical_v9_min"] == 0.01
    assert gate["crps_relative_gain_asymmetric_vs_symmetric_min"] == 0.01
