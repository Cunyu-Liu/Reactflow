#!/usr/bin/env python3
"""Mechanical consistency checks for the frozen V14 amendment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


EXPECTED_BRANCH = "codex/reactflow-delta-model-rescue-v14-20260827"
EXPECTED_PARENT_HEAD = "ec38e701f0528f8141e5724f334923dd49c266e4"
EXPECTED_V13_STATUS = "TERMINAL_V13M3_TOP_JOURNAL_SCREEN_FAIL"
EXPECTED_POST_V13_STATUS = (
    "TERMINAL_PV13D3_A_AND_C_CLOSED_WT_PROFILE_PRETRAINING_ONLY"
)
POST_V14_ONCE_ONLY_AUTHORITIES: dict[str, dict[str, Any]] = {
    "POST_V14_FIRST_MATCHING_ROUTER_ONCE_ONLY": {
        "next_allowed_action": "RUN_SINGLE_POST_V14_FIRST_MATCHING_ROUTER",
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
    "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_ONCE_ONLY": {
        "next_allowed_action": "RUN_SINGLE_POST_V14_BRANCH6_TAIL_DIAGNOSTIC",
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
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V14 YAML root must be a mapping: {path}")
    return value


def _assert_post_v14_once_only_authority(
    active: dict[str, Any], authority: dict[str, Any]
) -> None:
    if active.get("authority", {}).get("current_phase") != "V14M3":
        raise RuntimeError("V14 post-terminal once-only authority requires V14M3")
    if active.get("runnable_phases") != ["V14M3"]:
        raise RuntimeError(
            "V14 post-terminal once-only authority must expose only V14M3"
        )
    if active.get("training_allowed") is not False or active.get(
        "candidate_model_training_allowed"
    ) is not False:
        raise RuntimeError(
            "V14 post-terminal once-only authority requires training closed"
        )
    if active.get("next_allowed_action") != authority["next_allowed_action"]:
        raise RuntimeError("V14 post-terminal once-only action changed")
    mapping_name = authority["mapping_name"]
    if active.get(mapping_name) != authority["mapping"]:
        raise RuntimeError(
            "V14 post-terminal once-only authority mapping or canonical paths changed"
        )


def assert_outcome_authority_is_narrow(active: dict[str, Any]) -> None:
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V14 partial score authority is too broad")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V14 external outcome authority is too broad")
    if active.get("parent_terminal_verdict_change_allowed") is not False:
        raise RuntimeError("V14 parent-verdict authority is too broad")

    held = active.get("held_score_read_allowed")
    if held is False:
        return
    post_v14_authority = (
        POST_V14_ONCE_ONLY_AUTHORITIES.get(held)
        if isinstance(held, str)
        else None
    )
    if post_v14_authority is not None:
        _assert_post_v14_once_only_authority(active, post_v14_authority)
        return
    phase = active.get("authority", {}).get("current_phase")
    expected = {
        "V14M3": "V14_COMPLETE_MERGE_SCORE_ONCE_ONLY",
        "V14M4": "V14_FORMAL_COMPLETE_SCORE_ONCE_ONLY",
    }.get(phase)
    if (
        expected is None
        or held != expected
        or active.get("training_allowed") is not False
        or active.get("candidate_model_training_allowed") is not False
        or active.get("runnable_phases") != [phase]
    ):
        raise RuntimeError("V14 held-score authority is not training-closed score-once")


def _assert_frozen_model(contract: dict[str, Any]) -> None:
    models = contract["models"]
    if models["candidate"]["id"] != (
        "v14_masked_wt_profile_pretrained_feature41_anchor"
    ):
        raise RuntimeError("V14 candidate changed")
    if models["matched_null"]["id"] != "v14_from_scratch_feature41_anchor":
        raise RuntimeError("V14 matched null changed")
    architecture = models["architecture"]
    expected_architecture = {
        "input_channels": 11,
        "context_width": 256,
        "attention_heads": 8,
        "context_blocks": 6,
        "ffn_width": 1024,
        "relative_distance_window": 256,
        "dropout": 0.1,
        "point_head_width": 384,
        "point_head_hidden_layers": 2,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise RuntimeError(f"V14 architecture changed: {key}")
    counts = models["exact_parameter_counts"]
    expected_counts = {
        "total_each": 5_117_874,
        "encoder_without_heads": 4_767_280,
        "pretraining_decoder_each": 769,
        "downstream_residual_head_each": 349_825,
        "downstream_trainable_each_after_decoder_freeze": 5_117_105,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise RuntimeError(f"V14 exact parameter count changed: {key}")
    if counts.get("candidate_and_null_total_equal") is not True:
        raise RuntimeError("V14 parameter-matched null invariant is absent")


def _assert_frozen_pretraining(contract: dict[str, Any]) -> None:
    pretraining = contract["pretraining"]
    if pretraining.get("data") != "OUTER_TRAIN_WT_CONSTRUCTS_ONLY":
        raise RuntimeError("V14 pretraining data universe changed")
    if pretraining.get("registered_outer_train_constructs_per_fold") != 152:
        raise RuntimeError("V14 registered outer-train WT count changed")
    eligibility = pretraining.get("eligibility", {})
    if eligibility.get("minimum_wt_observed_positions") != 2 or eligibility.get(
        "zero_observed_constructs"
    ) != "EXCLUDED_BECAUSE_NO_RECONSTRUCTION_TARGET_EXISTS":
        raise RuntimeError("V14 zero-observed pretraining rule changed")
    if pretraining.get("held_puzzle_wt_profile_access_allowed") is not False:
        raise RuntimeError("V14 held puzzle WT entered pretraining authority")
    if pretraining.get("mutant_outcome_access_allowed") is not False:
        raise RuntimeError("V14 mutant outcome entered pretraining authority")
    if pretraining["masking"].get("fraction") != 0.40:
        raise RuntimeError("V14 masking fraction changed")
    if pretraining.get("objective") != (
        "MEAN_L1_OVER_MASKED_POSITIONS_THEN_EQUAL_CONSTRUCT_EXPOSURE"
    ):
        raise RuntimeError("V14 pretraining objective changed")
    if pretraining.get("epochs") != {"smoke": 3, "screen_and_formal": 200}:
        raise RuntimeError("V14 pretraining epochs changed")
    if pretraining.get("residual_head_gradient_allowed") is not False:
        raise RuntimeError("V14 pretraining can update the residual head")


def _assert_frozen_gates(contract: dict[str, Any]) -> None:
    gates = contract["v14m3_screen"]["gates"]
    expected = {
        "signed_delta": (0.12, 0.02, 0.015),
        "point_absolute": (0.07, 0.02, 0.01),
        "task_crps": (0.05, 0.02, 0.015),
        "distribution_absolute": (0.15, 0.02, 0.01),
    }
    for metric, values in expected.items():
        gate = gates[metric]
        relative_values = [
            value for key, value in gate.items() if key.startswith("relative_gain_")
        ]
        if sorted(relative_values) != sorted(values):
            raise RuntimeError(f"V14 headline gate changed: {metric}")
        if gate.get("paired_ci_lower_each_gt") != 0.0:
            raise RuntimeError(f"V14 paired CI gate changed: {metric}")
    stability = gates["stability"]
    if stability != {
        "leave_one_puzzle_positive_all_headline_comparisons": True,
        "max_single_puzzle_effect_fraction": 0.20,
        "max_coverage_error_worsening_pp": 1.0,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "unexpected_keys": 0,
    }:
        raise RuntimeError("V14 stability Gate changed")


def validate_contract(repo_root: Path) -> dict[str, Any]:
    active = _load(repo_root / "configs/reactflow_delta/active_contract.yaml")
    contract = _load(
        repo_root / "configs/reactflow_delta/model_rescue_v14_amendment.yaml"
    )
    ledger = _load(
        repo_root / "docs/prospective_v2/model_rescue_v14_decision_ledger.yaml"
    )
    if active.get("schema_version") != "reactflow_delta.active_contract.v14":
        raise RuntimeError("V14 active authority schema is not active")
    authority = active["authority"]
    if authority.get("branch") != EXPECTED_BRANCH:
        raise RuntimeError("V14 active branch changed")
    for key in (
        "machine_contract_path",
        "human_contract_path",
        "decision_ledger_path",
        "implementation_plan_path",
        "research_record_path",
    ):
        if not (repo_root / authority[key]).is_file():
            raise RuntimeError(f"V14 authority target is missing: {key}")

    if active["parent_state"].get("v13_status") != EXPECTED_V13_STATUS:
        raise RuntimeError("V14 active authority changed the V13 terminal status")
    if active["parent_state"].get("post_v13_status") != EXPECTED_POST_V13_STATUS:
        raise RuntimeError("V14 active authority changed post-V13 terminal status")
    if active["parent_state"].get("parent_head") != EXPECTED_PARENT_HEAD:
        raise RuntimeError("V14 parent head changed")
    if contract["parent"].get("v13_status") != EXPECTED_V13_STATUS:
        raise RuntimeError("V14 contract changed V13 terminal status")
    if contract["parent"].get("post_v13_diagnostic_status") != (
        EXPECTED_POST_V13_STATUS
    ):
        raise RuntimeError("V14 contract changed post-V13 terminal status")
    if contract["parent"].get("parent_head") != EXPECTED_PARENT_HEAD:
        raise RuntimeError("V14 machine parent head changed")
    immutable = ledger["immutable_parent_verdicts"]
    if immutable.get("v13") != EXPECTED_V13_STATUS:
        raise RuntimeError("V14 ledger changed V13 terminal status")
    if immutable.get("post_v13") != EXPECTED_POST_V13_STATUS:
        raise RuntimeError("V14 ledger changed post-V13 terminal status")
    if immutable.get("parent_head") != EXPECTED_PARENT_HEAD:
        raise RuntimeError("V14 ledger parent head changed")

    assert_outcome_authority_is_narrow(active)
    _assert_frozen_model(contract)
    _assert_frozen_pretraining(contract)
    _assert_frozen_gates(contract)
    return {
        "status": "V14_CONTRACT_VALIDATION_PASS",
        "phase": authority["current_phase"],
        "training_allowed": active["training_allowed"],
        "held_score_read_allowed": active["held_score_read_allowed"],
        "external_outcome_access_allowed": active[
            "new_external_outcome_access_allowed"
        ],
    }


def assert_run_authority(repo_root: Path, phase: str) -> None:
    active = _load(repo_root / "configs/reactflow_delta/active_contract.yaml")
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(f"V14 {phase} is not the sole active authority")
    if active.get("runnable_phases") != [phase]:
        raise RuntimeError(f"V14 {phase} is not the sole runnable phase")
    required = {
        "V14M1": "V14_IMPLEMENTATION_AND_FOCUSED_TESTS_ONLY",
        "V14M2": "V14_REAL_DATA_ENGINEERING_SMOKE_ONLY",
        "V14M3": "V14_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY",
        "V14M4": "V14_FIXED_FIVE_SEED_FORMAL_ONLY",
    }.get(phase)
    if required is None:
        raise RuntimeError(f"V14 phase is not runnable: {phase}")
    if phase == "V14M1":
        if active["authorization"].get("implementation_allowed") is not True:
            raise RuntimeError("V14M1 implementation authority is absent")
        if active.get("training_allowed") is not False:
            raise RuntimeError("V14M1 cannot authorize training")
    elif (
        active.get("training_allowed") != required
        or active.get("candidate_model_training_allowed") != required
    ):
        raise RuntimeError(f"V14 {phase} training authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError(f"V14 {phase} requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError(f"V14 {phase} requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError(f"V14 {phase} requires external outcomes locked")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate_contract(args.repo_root.resolve())
    print(yaml.safe_dump(result, sort_keys=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
