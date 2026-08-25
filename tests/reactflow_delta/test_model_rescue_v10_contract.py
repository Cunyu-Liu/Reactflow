from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.run_model_rescue_v10 import assert_run_authority


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_v10_terminal_fail_closes_training_scoring_and_formal() -> None:
    active = _yaml("configs/reactflow_delta/active_contract.yaml")
    contract = _yaml("configs/reactflow_delta/model_rescue_v10_amendment.yaml")
    assert active["schema_version"] == "reactflow_delta.active_contract.v11"
    assert active["parent_state"]["model_rescue_v10_status"] == (
        "TERMINAL_TOP_JOURNAL_TASK_CRPS_MARGIN_FAIL"
    )
    assert active["parent_state"]["model_rescue_v10_formal_opened"] is False
    assert active["v10_terminal_verdict_change_allowed"] is False
    assert active["training_allowed"] is False
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert contract["parent"]["v9_gate_changed"] is False
    assert contract["parent"]["v9m4_opened"] is False
    assert contract["formal_confirmation"]["authorized"] is False
    assert contract["contract_status"] == (
        "TERMINAL_V10M3_TOP_JOURNAL_SCREEN_FAIL_V10M4_PERMANENTLY_CLOSED"
    )
    assert contract["v10m1_smoke"]["scientific_scores_read"] is False
    assert contract["v10m2_screen"]["complete_before_score"] is True
    assert contract["input"]["meanaligned_held_point_authority"] == (
        "AUTHORITATIVE_V8_PREDICTION_ARTIFACT_BY_BIOLOGICAL_KEY"
    )
    assert contract["input"]["current_and_future_held_point_materialization"] == (
        "DIRECT_KEYED_READ_FROM_AUTHORITY"
    )
    assert contract["input"]["preserved_pre_fix_fold_materialization"] == (
        "SAME_CHECKPOINT_RECOMPUTATION_WITHIN_FROZEN_1E_7_ONLY"
    )
    formal = contract["formal_confirmation"]
    assert formal["authorized"] is False
    assert formal["prediction_assembly"]["distribution"] == "EQUAL_SEED_MIXTURE"
    assert formal["prediction_assembly"]["assembled_components_per_head"] == 10
    assert formal["formal_gate"]["repeat_all_v10m3_top_journal_gates_on_five_seed_mixture"] is True
    assert formal["formal_gate"]["task_crps_positive_individual_seeds_min"] == 4
    assert formal["formal_gate"]["asymmetric_increment_positive_individual_seeds_min"] == 4
    assert formal["failed_seed_removal_allowed"] is False
    with pytest.raises(RuntimeError, match="outside active V10M2"):
        assert_run_authority(ROOT, "V10M2")
    assert contract["v10m3_result"]["failed_gate"] == (
        "TASK_CRPS_RELATIVE_GAIN_GE_0_05"
    )
    assert contract["v10m3_result"]["v10m4_authorized"] is False


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
