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
    cache_identity = contract["input_artifacts"]["corrected_identity_qualification"]
    assert cache_identity["v5_unique_keys"] == 13976
    assert cache_identity["v5_missing_keys"] == 0
    assert cache_identity["v5_full_pos_mismatches"] == 0
    assert cache_identity["v6_unique_keys"] == 13976
    assert cache_identity["v6_missing_keys"] == 0
    assert cache_identity["v6_full_pos_mismatches"] == 0
    assert contract["execution"]["complete_before_score"] is True
    assert contract["authorization"]["held_score_read_allowed"] is False
    assert contract["authorization"]["partial_score_read_allowed"] is False
    assert contract["authorization"]["candidate_model_training_allowed"] is False
    assert contract["authorization"]["v7_dependency_allowed"] is False
    assert contract["authorization"]["new_external_outcome_access_allowed"] is False


def test_corrected_baseline_complete_merge_opens_only_held_scoring() -> None:
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    assert active["authority"]["current_phase"] == "TIC2A"
    assert active["authority"]["current_authority_state"] == (
        "TIC2A_COMPLETE_UNSCORED_MERGE_SCORE_AUTHORIZED"
    )
    assert active["training_allowed"] is False
    assert active["outcome_blind_cache_allowed"] is False
    assert active["held_score_read_allowed"] is True
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False

    ledger = yaml.safe_load(
        (ROOT / "docs/prospective_v2/target_identity_corrected_baseline_ledger.yaml").read_text()
    )
    assert ledger["execution"]["folds_complete"] == 20
    assert ledger["execution"]["target_join_allowed"] is True
    assert ledger["execution"]["partial_score_read_allowed"] is False
