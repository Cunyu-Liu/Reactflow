from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_corrected_baseline_contract_is_fixed_and_score_closed() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/target_identity_corrected_baseline_rebuild.yaml").read_text()
    )
    assert contract["scope"]["candidates"] == [
        "direct18",
        "v5_feature30",
        "v6_feature41",
    ]
    assert contract["algorithm"]["ridge_alpha"] == 1.0
    assert contract["algorithm"]["v5_v6_shared_feature30_replay_atol"] == 1e-12
    assert contract["execution"]["complete_before_score"] is True
    assert contract["authorization"]["held_score_read_allowed"] is False
    assert contract["authorization"]["partial_score_read_allowed"] is False
    assert contract["authorization"]["candidate_model_training_allowed"] is False
    assert contract["authorization"]["v7_dependency_allowed"] is False
    assert contract["authorization"]["new_external_outcome_access_allowed"] is False


def test_corrected_baseline_active_authority_is_isolated_from_v7() -> None:
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    assert active["authority"]["current_phase"] == "TIC2A"
    assert active["training_allowed"] == "FIXED_CORRECTED_WEIGHTED_RIDGE_BASELINES_ONLY"
    assert active["outcome_blind_cache_allowed"] is False
    assert active["held_score_read_allowed"] is False
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
