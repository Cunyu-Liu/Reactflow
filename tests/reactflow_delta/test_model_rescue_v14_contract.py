from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.validate_model_rescue_v14_contract import (
    EXPECTED_B5RP0_ACTION,
    EXPECTED_B5RP0_AUTHORITY,
    EXPECTED_B5RP0_PARENT_STATE,
    EXPECTED_B5RP0_PROJECT_TASK,
    EXPECTED_B5RP1_ACTION,
    EXPECTED_B5RP1_AUTHORITY,
    EXPECTED_B5RP1_TOKEN,
    EXPECTED_B5RP2_ACTION,
    EXPECTED_B5RP2_AUTHORITY,
    EXPECTED_B5RP2_TOKEN,
    EXPECTED_B5RP3_ACTION,
    EXPECTED_B5RP3_AUTHORITY,
    EXPECTED_B5RP3_TERMINAL_ACTION,
    EXPECTED_B5RP3_TERMINAL_AUTHORITY,
    assert_branch5_b5rp0_authority_is_narrow,
    assert_branch5_b5rp1_authority_is_narrow,
    assert_branch5_b5rp2_authority_is_narrow,
    assert_branch5_b5rp3_authority_is_narrow,
    assert_branch5_b5rp3_terminal_authority_is_narrow,
    assert_outcome_authority_is_narrow,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]
POST_V14_CASES = (
    {
        "id": "router",
        "token": "POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY",
        "action": "RUN_SINGLE_POST_V14_FIRST_MATCHING_ROUTER",
        "mapping_name": "post_v14_router_authority",
        "mapping": {
            "runtime_authority_token": "POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY",
            "complete_score_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/v14m3_complete_score.json"
            ),
            "qualification_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/v14m3_qualification.json"
            ),
            "router_output_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/post_v14_first_matching_route.json"
            ),
        },
    },
    {
        "id": "branch6",
        "token": "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_ONCE_ONLY",
        "action": "RUN_SINGLE_POST_V14_BRANCH6_TAIL_DIAGNOSTIC",
        "mapping_name": "post_v14_branch6_diagnostic_authority",
        "mapping": {
            "runtime_authority_token": (
                "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_ONCE_ONLY"
            ),
            "complete_unscored_merge_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/v14m3_complete_unscored_merge.json"
            ),
            "complete_score_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/v14m3_complete_score.json"
            ),
            "qualification_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/v14m3_qualification.json"
            ),
            "router_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/post_v14_first_matching_route.json"
            ),
            "m2_csv_path": (
                "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/"
                "reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
            ),
            "diagnostic_output_path": (
                "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
                "v14m3_screen_seed0/post_v14_branch6_tail_diagnostic.json"
            ),
        },
    },
)


def _post_v14_active(case: dict[str, object]) -> dict[str, object]:
    return {
        "authority": {"current_phase": "V14M3"},
        "runnable_phases": ["V14M3"],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": case["token"],
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "parent_terminal_verdict_change_allowed": False,
        "next_allowed_action": case["action"],
        str(case["mapping_name"]): copy.deepcopy(case["mapping"]),
    }


def test_frozen_v14_contract_passes() -> None:
    result = validate_contract(ROOT)
    assert result["status"] == "V14_CONTRACT_VALIDATION_PASS"
    assert result["phase"] in {
        "V14M1",
        "V14M2",
        "V14M3",
        "V14M4",
        "V14M5",
        "M6",
        "B5RP0",
        "B5RP1",
        "B5RP2",
        "B5RP3",
    }
    assert result["held_score_read_allowed"] in {
        False,
        "V14_COMPLETE_MERGE_SCORE_ONCE_ONLY",
        "V14_FORMAL_COMPLETE_SCORE_ONCE_ONLY",
        "POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY",
        "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_ONCE_ONLY",
        "POST_V14_BRANCH5_COMPLETE_MERGE_SCORE_ONCE_ONLY",
    }
    assert result["external_outcome_access_allowed"] is False


def _b5rp0_active() -> dict[str, object]:
    return {
        "project_task_id": EXPECTED_B5RP0_PROJECT_TASK,
        "parent_state": copy.deepcopy(EXPECTED_B5RP0_PARENT_STATE),
        "authority": copy.deepcopy(EXPECTED_B5RP0_AUTHORITY),
        "authorization": {
            "neural_training_allowed": False,
            "screen_allowed": False,
        },
        "runnable_phases": ["B5RP0"],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "next_allowed_action": EXPECTED_B5RP0_ACTION,
    }


def test_validator_accepts_exact_branch5_b5rp0_authority() -> None:
    assert_branch5_b5rp0_authority_is_narrow(_b5rp0_active())


@pytest.mark.parametrize(
    "failure",
    (
        "project",
        "parent",
        "phase",
        "runnable",
        "training",
        "held",
        "path",
        "status",
        "screen",
        "action",
    ),
)
def test_validator_rejects_broadened_or_changed_branch5_b5rp0_authority(
    failure: str,
) -> None:
    active = _b5rp0_active()
    authority = active["authority"]
    assert isinstance(authority, dict)
    if failure == "project":
        active["project_task_id"] = "wrong"
    elif failure == "parent":
        parent = active["parent_state"]
        assert isinstance(parent, dict)
        parent["unexpected"] = True
    elif failure == "phase":
        authority["current_phase"] = "B5RP1"
    elif failure == "runnable":
        active["runnable_phases"] = ["B5RP0", "B5RP1"]
    elif failure == "training":
        active["training_allowed"] = True
    elif failure == "held":
        active["held_score_read_allowed"] = True
    elif failure == "path":
        authority["source_manifest_path"] = "/mnt/cunyuliu/wrong.json"
    elif failure == "status":
        authority["source_manifest_status"] = "PASS"
    elif failure == "screen":
        authorization = active["authorization"]
        assert isinstance(authorization, dict)
        authorization["screen_allowed"] = True
    else:
        active["next_allowed_action"] = "RUN_SOMETHING_ELSE"
    with pytest.raises(RuntimeError, match="B5RP0"):
        assert_branch5_b5rp0_authority_is_narrow(active)


def _b5rp1_active() -> dict[str, object]:
    return {
        "project_task_id": EXPECTED_B5RP0_PROJECT_TASK,
        "parent_state": copy.deepcopy(EXPECTED_B5RP0_PARENT_STATE),
        "authority": copy.deepcopy(EXPECTED_B5RP1_AUTHORITY),
        "authorization": {
            "neural_training_allowed": False,
            "screen_allowed": True,
        },
        "runnable_phases": ["B5RP1"],
        "training_allowed": EXPECTED_B5RP1_TOKEN,
        "candidate_model_training_allowed": EXPECTED_B5RP1_TOKEN,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "next_allowed_action": EXPECTED_B5RP1_ACTION,
    }


def test_validator_accepts_exact_branch5_b5rp1_authority() -> None:
    assert_branch5_b5rp1_authority_is_narrow(_b5rp1_active())


@pytest.mark.parametrize(
    "failure",
    (
        "parent",
        "phase",
        "runnable",
        "training",
        "candidate_training",
        "neural_training",
        "held",
        "path",
        "manifest_status",
        "screen",
        "action",
    ),
)
def test_validator_rejects_broadened_or_changed_branch5_b5rp1_authority(
    failure: str,
) -> None:
    active = _b5rp1_active()
    authority = active["authority"]
    assert isinstance(authority, dict)
    if failure == "parent":
        parent = active["parent_state"]
        assert isinstance(parent, dict)
        parent["post_v14_first_matching_branch_id"] = "4"
    elif failure == "phase":
        authority["current_phase"] = "B5RP0"
    elif failure == "runnable":
        active["runnable_phases"] = ["B5RP0", "B5RP1"]
    elif failure == "training":
        active["training_allowed"] = False
    elif failure == "candidate_training":
        active["candidate_model_training_allowed"] = False
    elif failure == "neural_training":
        authorization = active["authorization"]
        assert isinstance(authorization, dict)
        authorization["neural_training_allowed"] = EXPECTED_B5RP1_TOKEN
    elif failure == "held":
        active["held_score_read_allowed"] = True
    elif failure == "path":
        authority["prediction_dir"] = "/mnt/cunyuliu/wrong"
    elif failure == "manifest_status":
        authority["source_manifest_status"] = "PENDING"
    elif failure == "screen":
        authorization = active["authorization"]
        assert isinstance(authorization, dict)
        authorization["screen_allowed"] = False
    else:
        active["next_allowed_action"] = "RUN_SOMETHING_ELSE"
    with pytest.raises(RuntimeError, match="B5RP1"):
        assert_branch5_b5rp1_authority_is_narrow(active)


def _b5rp2_active() -> dict[str, object]:
    return {
        "project_task_id": EXPECTED_B5RP0_PROJECT_TASK,
        "parent_state": copy.deepcopy(EXPECTED_B5RP0_PARENT_STATE),
        "authority": copy.deepcopy(EXPECTED_B5RP2_AUTHORITY),
        "authorization": {
            "neural_training_allowed": False,
            "screen_allowed": False,
        },
        "runnable_phases": ["B5RP2"],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": EXPECTED_B5RP2_TOKEN,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "next_allowed_action": EXPECTED_B5RP2_ACTION,
    }


def test_validator_accepts_exact_branch5_b5rp2_authority() -> None:
    assert_branch5_b5rp2_authority_is_narrow(_b5rp2_active())


@pytest.mark.parametrize(
    "failure",
    (
        "parent",
        "phase",
        "runnable",
        "training",
        "held",
        "path",
        "status",
        "neural_training",
        "screen",
        "action",
    ),
)
def test_validator_rejects_broadened_or_changed_branch5_b5rp2_authority(
    failure: str,
) -> None:
    active = _b5rp2_active()
    authority = active["authority"]
    assert isinstance(authority, dict)
    if failure == "parent":
        parent = active["parent_state"]
        assert isinstance(parent, dict)
        parent["post_v14_first_matching_branch_id"] = "6"
    elif failure == "phase":
        authority["current_phase"] = "B5RP1"
    elif failure == "runnable":
        active["runnable_phases"] = ["B5RP1", "B5RP2"]
    elif failure == "training":
        active["training_allowed"] = EXPECTED_B5RP1_TOKEN
    elif failure == "held":
        active["held_score_read_allowed"] = True
    elif failure == "path":
        authority["complete_score_path"] = "/mnt/cunyuliu/wrong.json"
    elif failure == "status":
        authority["complete_unscored_merge_status"] = "PASS"
    elif failure == "neural_training":
        authorization = active["authorization"]
        assert isinstance(authorization, dict)
        authorization["neural_training_allowed"] = True
    elif failure == "screen":
        authorization = active["authorization"]
        assert isinstance(authorization, dict)
        authorization["screen_allowed"] = True
    else:
        active["next_allowed_action"] = "RUN_SOMETHING_ELSE"
    with pytest.raises(RuntimeError, match="B5RP2"):
        assert_branch5_b5rp2_authority_is_narrow(active)


def _b5rp3_active() -> dict[str, object]:
    return {
        "project_task_id": EXPECTED_B5RP0_PROJECT_TASK,
        "parent_state": copy.deepcopy(EXPECTED_B5RP0_PARENT_STATE),
        "authority": copy.deepcopy(EXPECTED_B5RP3_AUTHORITY),
        "authorization": {
            "neural_training_allowed": False,
            "screen_allowed": False,
        },
        "runnable_phases": ["B5RP3"],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "next_allowed_action": EXPECTED_B5RP3_ACTION,
    }


def test_validator_accepts_exact_branch5_b5rp3_authority() -> None:
    assert_branch5_b5rp3_authority_is_narrow(_b5rp3_active())


@pytest.mark.parametrize(
    "failure",
    (
        "parent",
        "phase",
        "runnable",
        "training",
        "held",
        "path",
        "status",
        "neural_training",
        "screen",
        "action",
    ),
)
def test_validator_rejects_broadened_or_changed_branch5_b5rp3_authority(
    failure: str,
) -> None:
    active = _b5rp3_active()
    authority = active["authority"]
    assert isinstance(authority, dict)
    if failure == "parent":
        parent = active["parent_state"]
        assert isinstance(parent, dict)
        parent["post_v14_first_matching_branch_id"] = "4"
    elif failure == "phase":
        authority["current_phase"] = "B5RP2"
    elif failure == "runnable":
        active["runnable_phases"] = ["B5RP2", "B5RP3"]
    elif failure == "training":
        active["candidate_model_training_allowed"] = True
    elif failure == "held":
        active["held_score_read_allowed"] = EXPECTED_B5RP2_TOKEN
    elif failure == "path":
        authority["qualification_path"] = "/mnt/cunyuliu/wrong.json"
    elif failure == "status":
        authority["complete_score_status"] = "PASS"
    elif failure == "neural_training":
        authorization = active["authorization"]
        assert isinstance(authorization, dict)
        authorization["neural_training_allowed"] = True
    elif failure == "screen":
        authorization = active["authorization"]
        assert isinstance(authorization, dict)
        authorization["screen_allowed"] = True
    else:
        active["next_allowed_action"] = "RUN_SOMETHING_ELSE"
    with pytest.raises(RuntimeError, match="B5RP3"):
        assert_branch5_b5rp3_authority_is_narrow(active)


def _b5rp3_terminal_active() -> dict[str, object]:
    return {
        "project_task_id": EXPECTED_B5RP0_PROJECT_TASK,
        "parent_state": copy.deepcopy(EXPECTED_B5RP0_PARENT_STATE),
        "authority": copy.deepcopy(EXPECTED_B5RP3_TERMINAL_AUTHORITY),
        "authorization": {
            "implementation_allowed": False,
            "neural_training_allowed": False,
            "screen_allowed": False,
            "formal_confirmation_allowed": False,
        },
        "runnable_phases": [],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "next_allowed_action": EXPECTED_B5RP3_TERMINAL_ACTION,
    }


def test_validator_accepts_exact_branch5_b5rp3_terminal_authority() -> None:
    assert_branch5_b5rp3_terminal_authority_is_narrow(_b5rp3_terminal_active())


@pytest.mark.parametrize(
    "failure",
    (
        "parent",
        "state",
        "runnable",
        "training",
        "held",
        "path",
        "score_status",
        "qualification_status",
        "route",
        "implementation",
        "neural_training",
        "screen",
        "formal",
        "action",
    ),
)
def test_validator_rejects_broadened_or_changed_branch5_b5rp3_terminal_authority(
    failure: str,
) -> None:
    active = _b5rp3_terminal_active()
    authority = active["authority"]
    authorization = active["authorization"]
    assert isinstance(authority, dict)
    assert isinstance(authorization, dict)
    if failure == "parent":
        parent = active["parent_state"]
        assert isinstance(parent, dict)
        parent["post_v14_first_matching_branch_id"] = "4"
    elif failure == "state":
        authority["current_authority_state"] = "NOT_TERMINAL"
    elif failure == "runnable":
        active["runnable_phases"] = ["B5RP3"]
    elif failure == "training":
        active["candidate_model_training_allowed"] = True
    elif failure == "held":
        active["held_score_read_allowed"] = EXPECTED_B5RP2_TOKEN
    elif failure == "path":
        authority["qualification_path"] = "/mnt/cunyuliu/wrong.json"
    elif failure == "score_status":
        authority["complete_score_status"] = "PASS"
    elif failure == "qualification_status":
        authority["qualification_status"] = "PASS"
    elif failure == "route":
        authority["route_after_qualification"] = "P1"
    elif failure == "implementation":
        authorization["implementation_allowed"] = True
    elif failure == "neural_training":
        authorization["neural_training_allowed"] = True
    elif failure == "screen":
        authorization["screen_allowed"] = True
    elif failure == "formal":
        authorization["formal_confirmation_allowed"] = True
    else:
        active["next_allowed_action"] = "RUN_SOMETHING_ELSE"
    with pytest.raises(RuntimeError, match="B5RP3 terminal"):
        assert_branch5_b5rp3_terminal_authority_is_narrow(active)


def test_v14_freezes_matched_null_and_top_journal_gates() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/model_rescue_v14_amendment.yaml").read_text()
    )
    assert contract["models"]["exact_parameter_counts"]["total_each"] == 5_117_874
    assert contract["models"]["matched_null"]["id"] == (
        "v14_from_scratch_feature41_anchor"
    )
    assert contract["pretraining"]["data"] == "OUTER_TRAIN_WT_CONSTRUCTS_ONLY"
    assert contract["pretraining"]["eligibility"][
        "zero_observed_constructs"
    ] == "EXCLUDED_BECAUSE_NO_RECONSTRUCTION_TARGET_EXISTS"
    assert contract["v14m3_screen"]["gates"]["task_crps"][
        "relative_gain_vs_from_scratch_null_min"
    ] == 0.015
    assert contract["v14m3_screen"]["gates"]["signed_delta"][
        "relative_gain_vs_feature41_min"
    ] == 0.12


def test_validator_rejects_broader_v14_score_authority(tmp_path: Path) -> None:
    copied = tmp_path / "repo"
    relative_files = (
        "configs/reactflow_delta/active_contract.yaml",
        "configs/reactflow_delta/model_rescue_v14_amendment.yaml",
        "docs/prospective_v2/model_rescue_v14_amendment_20260827.md",
        "docs/prospective_v2/model_rescue_v14_decision_ledger.yaml",
        "docs/plans/2026-08-27-model-rescue-v14.md",
        "docs/plans/2026-08-27-post-v14-model-contingency.md",
        "autoresearch/orchestrator-260827-v14-wt-profile/research.md",
    )
    for relative in relative_files:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text())
    active_path = copied / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text())
    bad = copy.deepcopy(active)
    bad["held_score_read_allowed"] = True
    active_path.write_text(yaml.safe_dump(bad, sort_keys=False))
    with pytest.raises(RuntimeError, match="held-score authority|B5RP0|B5RP1|B5RP2|B5RP3"):
        validate_contract(copied)


@pytest.mark.parametrize(
    "case", POST_V14_CASES, ids=[case["id"] for case in POST_V14_CASES]
)
def test_validator_accepts_exact_post_v14_once_only_authority(
    case: dict[str, object],
) -> None:
    assert_outcome_authority_is_narrow(_post_v14_active(case))


@pytest.mark.parametrize("replacement", [None, "POST_V14_UNKNOWN_ONCE_ONLY"])
def test_validator_rejects_missing_or_wrong_post_v14_token(
    replacement: str | None,
) -> None:
    active = _post_v14_active(POST_V14_CASES[0])
    if replacement is None:
        active.pop("held_score_read_allowed")
    else:
        active["held_score_read_allowed"] = replacement
    with pytest.raises(RuntimeError, match="held-score authority"):
        assert_outcome_authority_is_narrow(active)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("training_allowed", True, "training closed"),
        ("candidate_model_training_allowed", True, "training closed"),
        ("partial_fold_score_read_allowed", True, "partial score authority"),
        ("new_external_outcome_access_allowed", True, "external outcome authority"),
    ),
)
def test_validator_rejects_reopened_post_v14_authority(
    field: str, value: object, message: str
) -> None:
    active = _post_v14_active(POST_V14_CASES[0])
    active[field] = value
    with pytest.raises(RuntimeError, match=message):
        assert_outcome_authority_is_narrow(active)


@pytest.mark.parametrize("failure", ["phase", "runnable"])
def test_validator_rejects_non_v14m3_post_v14_authority(failure: str) -> None:
    active = _post_v14_active(POST_V14_CASES[0])
    if failure == "phase":
        active["authority"] = {"current_phase": "V14M4"}
    else:
        active["runnable_phases"] = ["V14M3", "V14M4"]
    with pytest.raises(RuntimeError, match="V14M3"):
        assert_outcome_authority_is_narrow(active)


@pytest.mark.parametrize(
    "case", POST_V14_CASES, ids=[case["id"] for case in POST_V14_CASES]
)
def test_validator_rejects_wrong_post_v14_action(case: dict[str, object]) -> None:
    active = _post_v14_active(case)
    active["next_allowed_action"] = "RUN_SOMETHING_ELSE"
    with pytest.raises(RuntimeError, match="once-only action"):
        assert_outcome_authority_is_narrow(active)


@pytest.mark.parametrize(
    "case", POST_V14_CASES, ids=[case["id"] for case in POST_V14_CASES]
)
@pytest.mark.parametrize(
    "failure", ["missing", "wrong_token", "wrong_path", "noncanonical", "extra"]
)
def test_validator_rejects_wrong_post_v14_mapping(
    case: dict[str, object], failure: str
) -> None:
    active = _post_v14_active(case)
    mapping_name = str(case["mapping_name"])
    mapping = active[mapping_name]
    assert isinstance(mapping, dict)
    if failure == "missing":
        active.pop(mapping_name)
    elif failure == "wrong_token":
        mapping["runtime_authority_token"] = "POST_V14_UNKNOWN_ONCE_ONLY"
    elif failure == "extra":
        mapping["unexpected_path"] = "/mnt/cunyuliu/unexpected.json"
    else:
        path_field = next(key for key in mapping if key != "runtime_authority_token")
        mapping[path_field] = (
            "/mnt/cunyuliu/wrong.json"
            if failure == "wrong_path"
            else f"{mapping[path_field]}/../noncanonical.json"
        )
    with pytest.raises(RuntimeError, match="authority mapping or canonical paths"):
        assert_outcome_authority_is_narrow(active)
