from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_target_profile_identity_correction_contract_is_fail_closed() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/target_profile_identity_correction_amendment.yaml").read_text()
    )
    assert contract["confirmed_impact"]["affected_mutants"] == 3494
    assert contract["confirmed_impact"]["affected_puzzle_method_cells"] == 40
    assert contract["confirmed_impact"]["corrected_real_data_wrong_target"] == 0
    assert contract["confirmed_impact"]["corrected_real_data_wrong_error"] == 0
    assert contract["history_policy"]["preserve_all_artifacts"] is True
    assert contract["history_policy"]["v3_old_checkpoint_reuse_allowed"] is False
    assert contract["history_policy"]["v3_resume_missing_folds_allowed"] is False
    assert contract["corrected_path"]["tic2_v7m2"]["old_v6_prediction_replay_allowed"] is False
    assert contract["authorization"]["corrected_training_allowed"] is False
    assert contract["authorization"]["held_score_read_allowed"] is False
    assert contract["authorization"]["partial_score_read_allowed"] is False
    assert contract["authorization"]["new_external_outcome_access_allowed"] is False


def test_active_v8_authority_preserves_identity_invalidation() -> None:
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    assert active["authority"]["current_phase"] == "V8M2"
    assert active["training_allowed"] is False
    assert active["held_score_read_allowed"] is True
    assert active["partial_fold_score_read_allowed"] is False
    assert active["legacy_target_dependent_prediction_reuse_allowed"] is False
    assert active["legacy_target_dependent_score_reuse_allowed"] is False
    assert active["legacy_v3_expert_reuse_allowed"] is False
    assert active["overwrite_existing_v8_fold_allowed"] is False
    assert active["gate_state"]["TARGET_PROFILE_IDENTITY"].endswith("PASS")
    assert active["gate_state"]["LEGACY_V3_CORRECTED_EXPERTS"] == (
        "INVALIDATED_TARGET_IDENTITY"
    )
