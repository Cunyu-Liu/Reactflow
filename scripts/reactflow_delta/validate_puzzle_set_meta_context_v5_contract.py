#!/usr/bin/env python3
"""Validate the frozen, inactive Puzzle-Set V5 declaration without results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from scripts.reactflow_delta.assemble_puzzle_set_meta_context_formal import (
    EXPECTED_CALIBRATION_EPOCHS as FORMAL_CALIBRATION_EPOCHS,
    EXPECTED_FOLDS as FORMAL_FOLDS,
    EXPECTED_POINT_EPOCHS as FORMAL_POINT_EPOCHS,
    EXPECTED_PRETRAINING_EPOCHS as FORMAL_PRETRAINING_EPOCHS,
    EXPECTED_SEEDS as FORMAL_SEEDS,
)
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v13 import (
    EXPECTED_POINT_PARAMETERS as V13_POINT_PARAMETERS,
)
from scripts.reactflow_delta.model_rescue_v14 import (
    EXPECTED_ENCODER_PARAMETERS as V14_ENCODER_PARAMETERS,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    FULL_CROSS_CONSTRUCT,
    POINT_CONTEXT_LR,
    POINT_GRADIENT_CLIP,
    POINT_HEAD_LR,
    POINT_HEAD_WARMUP_EPOCHS,
    POSITION_ALIGNED_OPERATOR,
    POSITION_DERANGEMENT_SHIFT,
    POSITION_DERANGED_NULL,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    EXPECTED_RESIDUAL_PARAMETERS,
)
from scripts.reactflow_delta.puzzle_set_meta_context_pretraining import (
    EXPECTED_CONTEXT_PRETRAINING_PARAMETERS,
    EXPECTED_DECODER_PARAMETERS,
    EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS,
    PRETRAINING_MASK_FRACTION,
)
from scripts.reactflow_delta.puzzle_set_meta_context_retention import (
    RETENTION_DIAGNOSTIC_EPOCH,
)
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import (
    EXPECTED_PROJECT_TASK as RUNTIME_PROJECT_TASK,
    FOLD_SCHEMA,
    FROZEN_INPUT_SOURCE_SPEC,
    PHASE_TRAINING_TOKENS,
    RUNNABLE_PHASES as RUNTIME_PHASES,
)
from scripts.reactflow_delta.score_puzzle_set_meta_context import (
    EXPECTED_PHASE as SCREEN_SCORE_PHASE,
    EXPECTED_PROJECT_TASK as SCREEN_SCORE_PROJECT,
    EXPECTED_SCORE_TOKEN as SCREEN_SCORE_TOKEN,
)
from scripts.reactflow_delta.score_puzzle_set_meta_context_formal import (
    EXPECTED_PHASE as FORMAL_SCORE_PHASE,
    EXPECTED_PROJECT_TASK as FORMAL_SCORE_PROJECT,
    EXPECTED_SCORE_TOKEN as FORMAL_SCORE_TOKEN,
)


EXPECTED_SCHEMA = "reactflow_delta.puzzle_set_meta_context_v5_amendment.v1"
EXPECTED_CONTRACT_ID = "reactflow_delta_puzzle_set_meta_context_v5_20260827"
EXPECTED_STATUS = "DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE"
EXPECTED_LEDGER_SCHEMA = (
    "reactflow_delta.puzzle_set_meta_context_v5_decision_ledger.draft.v1"
)

EXPECTED_ACTIVE_SCHEMA = "reactflow_delta.active_contract.v14"
EXPECTED_ACTIVE_PROJECT = "reactflow_delta_model_rescue_v14"
EXPECTED_ACTIVE_ROLE = "V14_SINGLE_ACTIVE_AUTHORITY"
EXPECTED_ACTIVE_BRANCH = "codex/reactflow-delta-model-rescue-v14-20260827"
EXPECTED_ACTIVE_WORKTREE = (
    "/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v14_20260827"
)
EXPECTED_ACTIVE_ARTIFACT_ROOT = "/mnt/cunyuliu/reactflow_delta_model_rescue_v14"
EXPECTED_ACTIVE_PHASE = "V14M3"
EXPECTED_ACTIVE_STATE = "TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY"
EXPECTED_ACTIVE_BINDING = (
    "V14M3_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_AUTHORIZED_SCORE_CLOSED"
)
EXPECTED_ACTIVE_TRAINING_TOKEN = "V14_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY"
EXPECTED_V14_MACHINE_CONTRACT = (
    "configs/reactflow_delta/model_rescue_v14_amendment.yaml"
)
EXPECTED_V14_MACHINE_SCHEMA = "reactflow_delta.model_rescue_v14_amendment.v1"
EXPECTED_V14_MACHINE_ID = (
    "reactflow_delta_model_rescue_v14_masked_wt_profile_pretraining_20260827"
)

EXPECTED_DOCUMENTS = {
    "human_contract_path": (
        "docs/prospective_v2/puzzle_set_meta_context_v5_amendment_20260827.md"
    ),
    "decision_ledger_path": (
        "docs/prospective_v2/puzzle_set_meta_context_v5_decision_ledger.yaml"
    ),
    "implementation_plan_path": (
        "docs/plans/2026-08-27-puzzle-set-meta-context-v5-amendment.md"
    ),
    "implementation_design_path": (
        "docs/plans/2026-08-27-puzzle-set-cross-only-retention-design.md"
    ),
    "architecture_decision_path": "docs/adr/0001-propose-puzzle-set-meta-context.md",
}
EXPECTED_ROUTER_PATH = "docs/plans/2026-08-27-post-v14-model-contingency.md"
EXPECTED_EXECUTABLE_PATHS = {
    "model": "scripts/reactflow_delta/puzzle_set_meta_context.py",
    "data": "scripts/reactflow_delta/puzzle_set_meta_context_data.py",
    "pretraining": "scripts/reactflow_delta/puzzle_set_meta_context_pretraining.py",
    "retention": "scripts/reactflow_delta/puzzle_set_meta_context_retention.py",
    "calibration": "scripts/reactflow_delta/puzzle_set_meta_context_calibration.py",
    "fold_runner": "scripts/reactflow_delta/run_puzzle_set_meta_context_probe.py",
    "smoke_controller": (
        "scripts/reactflow_delta/run_puzzle_set_meta_context_smoke_controller.sh"
    ),
    "smoke_qualifier": (
        "scripts/reactflow_delta/qualify_puzzle_set_meta_context_smoke.py"
    ),
    "screen_controller": (
        "scripts/reactflow_delta/run_puzzle_set_meta_context_screen_controller.sh"
    ),
    "merge": "scripts/reactflow_delta/merge_puzzle_set_meta_context_probe.py",
    "screen_scorer": "scripts/reactflow_delta/score_puzzle_set_meta_context.py",
    "screen_qualifier": "scripts/reactflow_delta/qualify_puzzle_set_meta_context.py",
    "formal_controller": (
        "scripts/reactflow_delta/run_puzzle_set_meta_context_formal_controller.sh"
    ),
    "formal_assembler": (
        "scripts/reactflow_delta/assemble_puzzle_set_meta_context_formal.py"
    ),
    "formal_scorer": "scripts/reactflow_delta/score_puzzle_set_meta_context_formal.py",
    "formal_qualifier": (
        "scripts/reactflow_delta/qualify_puzzle_set_meta_context_formal.py"
    ),
}

EXPECTED_HISTORY_BOUNDARY = {
    "v1_through_v13_terminal_verdicts_immutable": True,
    "v14_terminal_verdict_exists_at_draft": False,
    "v14_frozen_protocol_change_allowed": False,
    "v14_current_gates_change_allowed": False,
    "v14_future_terminal_handoff_reinterpretation_allowed": False,
}
EXPECTED_INACTIVE_AUTHORITY = {
    "active_pointer_path": "configs/reactflow_delta/active_contract.yaml",
    "sole_active_project_task_id": EXPECTED_ACTIVE_PROJECT,
    "sole_active_pointer_role": EXPECTED_ACTIVE_ROLE,
    "sole_active_branch": EXPECTED_ACTIVE_BRANCH,
    "this_amendment_may_replace_active_pointer": False,
    "activation_allowed_now": False,
    "activation_prerequisites": [
        "COMPLETE_V14_TERMINAL_HANDOFF",
        "POST_V14_FIRST_MATCHING_ROUTER_EXPLICITLY_SELECTS_P1",
        (
            "ROUTER_BRANCH_SPECIFIC_PROBE_STATE_BOUND_AS_NOT_APPLICABLE_OR_"
            "REQUIRED_EXACT_PASS"
        ),
        "ALL_REALIZED_FROZEN_INPUT_PATHS_ROLES_AND_COUNTS_BOUND",
        "CONTRACT_VALIDATOR_AND_V5_FOCUSED_TESTS_PASS_WITH_TRAINING_CLOSED",
        "FOCUSED_PUZZLE_SET_V5_ACTIVATION_COMMIT",
    ],
    "v14_exact_pass_routes_only_to_v14m4": True,
    "v14m4_formal_failure_can_activate_p1": False,
    "training_allowed": False,
    "candidate_model_training_allowed": False,
    "smoke_allowed": False,
    "screen_allowed": False,
    "formal_confirmation_allowed": False,
    "held_score_read_allowed": False,
    "partial_fold_score_read_allowed": False,
    "new_external_outcome_access_allowed": False,
}
EXPECTED_PHASE_TOKENS = {
    "P1M2": "PUZZLE_SET_P1M2_REAL_DATA_ENGINEERING_SMOKE_ONLY",
    "P1M3": "PUZZLE_SET_P1M3_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY",
    "P1M4": "PUZZLE_SET_P1M4_FIXED_FIVE_SEED_FORMAL_ONLY",
}
EXPECTED_FUTURE_PHASE_TOKENS = {
    "tokens_issued_now": False,
    **EXPECTED_PHASE_TOKENS,
    "generic_training_token_allowed": False,
}

EXPECTED_SCOPE_IDENTITY = {
    "task": "ENDPOINT_V7_ALL_MUTANT_FULL_CONSTRUCT_SIGNED_DELTA_DISTRIBUTION",
    "data": "REAL_OPENKNOT_M2_V4_5_2_DEVELOPMENT_CONSUMED",
    "split": "SPLIT_V4_TWENTY_FOLD_LOPO_PUZZLE",
    "evaluator": "METHOD_BALANCED_POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
    "target_identity": "EXACT_PUZZLE_METHOD_MUTATION",
}
EXPECTED_FROZEN_PARENTS = {
    "point_anchor": {
        "model": "V13_CANDIDATE_POINT",
        "seed": 0,
        "outer_fold_matched": True,
        "trainable": False,
    },
    "representation_anchor": {
        "model": "V14_CANDIDATE_OUTCOME_BLIND_WT_ENCODER",
        "seed": 0,
        "outer_fold_matched": True,
        "used_regardless_of_v14_score": True,
        "trainable": False,
    },
}
EXPECTED_ROUTE_PROBE_CRITERIA = {
    "signed_delta_relative_gain_min": 0.01,
    "point_absolute_relative_gain_min": 0.01,
    "signed_delta_paired_ci_lower_gt": 0.0,
    "point_absolute_paired_ci_lower_gt": 0.0,
    "signed_delta_positive_puzzles_min": 14,
    "point_absolute_positive_puzzles_min": 14,
    "independent_units": "20_PUZZLES",
    "data_access": "OUTER_TRAIN_ONLY",
}
EXPECTED_POST_V14_ROUTER = {
    "source_path": EXPECTED_ROUTER_PATH,
    "rule": "FIRST_MATCHING_BRANCH_CONTROLS",
    "selected_router_branch_id": "PENDING_COMPLETE_V14_TERMINAL_HANDOFF",
    "current_route_probe": {
        "requirement": "NOT_EVALUATED",
        "status": "NOT_EVALUATED",
    },
    "p1_eligible_branches": {
        "3": {
            "classification": "CAPACITY_WITHOUT_PRETRAINING_INCREMENT",
            "route_probe_at_activation": {
                "requirement": "NOT_APPLICABLE",
                "status": "NOT_APPLICABLE",
            },
        },
        "4": {
            "classification": "PRETRAINING_SIGNAL_INSUFFICIENT_FOR_TRANSFER",
            "route_probe_at_activation": {
                "requirement": "NOT_APPLICABLE",
                "status": "NOT_APPLICABLE",
            },
        },
        "5": {
            "classification": "INDEPENDENT_CONSTRUCT_TRANSFER_LIMITED",
            "route_probe_at_activation": {
                "requirement": "REQUIRED",
                "status": "EXACT_PASS",
                "exact_pass_criteria": EXPECTED_ROUTE_PROBE_CRITERIA,
            },
        },
    },
    "excluded_branches": {
        "1": "V14M4_ONLY",
        "2": "REPAIR_SAME_FROZEN_V14_UNIVERSE_ONLY",
        "6": "P2_OR_P3_ONLY",
        "7": "P3_ONLY_NO_P1_WITHOUT_NEW_INDEPENDENT_DATA",
    },
    "activation_binding": {
        "selected_router_branch_id_must_be_one_of": ["3", "4", "5"],
        "branch_3_or_4_probe_requirement_and_status": "NOT_APPLICABLE",
        "branch_5_probe_requirement": "REQUIRED",
        "branch_5_probe_status": "EXACT_PASS",
    },
}

EXPECTED_FROZEN_INPUT_SOURCES = {
    "activation_binding_status": "REALIZED_PATHS_ROLES_AND_COUNTS_PENDING",
    "bind_realized_sources_before_p1m1": True,
    "bind_each_outer_fold_0_through_19": True,
    "v13_seed0_point": {
        "role": "IMMUTABLE_POINT_ANCHOR",
        "source": "SAME_OUTER_FOLD_V13_CANDIDATE_SEED0_POINT_CHECKPOINT",
        "expected_filename_pattern": "v13_candidate_point_fold{outer_fold}_seed0.pt",
        "parameter_count": V13_POINT_PARAMETERS,
        "trainable_in_p1": False,
    },
    "v14_seed0_encoder": {
        "role": "IMMUTABLE_OUTCOME_BLIND_WT_REPRESENTATION",
        "source": (
            "SAME_OUTER_FOLD_V14_CANDIDATE_SEED0_POINT_CHECKPOINT_ENCODER_SUBSET"
        ),
        "expected_filename_pattern": "v14_candidate_point_fold{outer_fold}_seed0.pt",
        "imported_parameter_count": V14_ENCODER_PARAMETERS,
        "trainable_in_p1": False,
    },
    "v8_seed0_meanaligned": {
        "role": "FROZEN_TRAINED_DIRECT_FEATURE_SOURCE_FOR_CALIBRATION",
        "source": "SAME_OUTER_FOLD_V8_SEED0_MEANALIGNED_CHECKPOINT",
        "expected_checkpoint_filename_pattern": (
            "v8_corrected_mean_fold{outer_fold}_seed0.pt"
        ),
        "path_source_field": (
            "v8_corrected_expert_fold_result_fold{outer_fold}_seed0.json:"
            "meanaligned_checkpoint"
        ),
        "calibration_direct_feature_width": 201,
        "parameter_count": 109_581,
        "trainable_in_p1": False,
    },
    "tic2a_feature41_ridge": {
        "role": "OUTER_FOLD_FROZEN_FEATURE41_WEIGHTED_RIDGE_AND_41D_FEATURE_BASIS",
        "source": "TIC2A_COMPLETE_CORRECTED_OUTER_FOLD_MODEL_ARTIFACT_V6_FEATURE41",
        "expected_model_artifact_filename_pattern": (
            "tic2a_corrected_models_fold{outer_fold}.json"
        ),
        "path_source_field": (
            "tic2a_complete_merged.fold[{outer_fold}].model_artifact:v6_feature41"
        ),
        "feature_width": 41,
        "realized_parameter_count": "PENDING_ACTIVATION_BINDING",
        "trainable_in_p1": False,
    },
    "tic2a_merged_registry": {
        "role": "COMPLETE_TWENTY_FOLD_SOURCE_REGISTRY_PROVENANCE_ONLY",
        "source": "TIC2A_COMPLETE_CORRECTED_MERGED_UNSCORED_JSON",
        "realized_path": "PENDING_ACTIVATION_BINDING",
        "feature_source": False,
        "trainable_in_p1": False,
    },
    "unconstrained_feature_cache": {
        "role": "FROZEN_FEATURE41_CONSTRUCTION_INPUT",
        "source": "UNCONSTRAINED_ENSEMBLE_FEATURE_CACHE",
        "realized_path": "PENDING_ACTIVATION_BINDING",
        "trainable_in_p1": False,
    },
    "constrained_feature_cache": {
        "role": "FROZEN_FEATURE41_CONSTRUCTION_INPUT",
        "source": "CONSTRAINED_ENSEMBLE_FEATURE_CACHE",
        "realized_path": "PENDING_ACTIVATION_BINDING",
        "trainable_in_p1": False,
    },
    "v10": {
        "role": "TERMINAL_HISTORICAL_COMPARATOR_AND_RESIDUAL_FAMILY_PROVENANCE_ONLY",
        "expected_result_filename_pattern": (
            "v10_fold_result_fold{outer_fold}_seed0.json"
        ),
        "same_fold_result_path": "PENDING_ACTIVATION_BINDING",
        "learned_checkpoint_imported_into_p1": False,
        "feature_source": False,
        "point_anchor": False,
    },
    "activation_binding_required_fields": [
        "OUTER_FOLD",
        "SOURCE_ID",
        "REALIZED_PATH",
        "ROLE",
        "SEED",
        "REALIZED_PARAMETER_COUNT",
        "TRAINABLE_IN_P1",
    ],
    "feature41_ridge_realized_parameter_count_must_be_bound": True,
    "full_upstream_parameter_footprint_must_be_bound": True,
    "full_upstream_parameter_footprint": "PENDING_ACTIVATION_BINDING",
}

_SOURCE_SCOPES = {
    "v13_point_checkpoint": "SAME_OUTER_FOLD",
    "v14_encoder_checkpoint": "SAME_OUTER_FOLD",
    "v8_meanaligned_checkpoint": "SAME_OUTER_FOLD",
    "tic2a_feature41_model_artifact": "SAME_OUTER_FOLD",
    "tic2a_merged_registry": "GLOBAL",
    "unconstrained_feature_cache": "GLOBAL",
    "constrained_feature_cache": "GLOBAL",
    "v10_fold_comparator": "SAME_OUTER_FOLD",
}
EXPECTED_SOURCE_RECORDS = {
    name: {
        **spec,
        "outer_fold_scope": _SOURCE_SCOPES[name],
    }
    for name, spec in FROZEN_INPUT_SOURCE_SPEC.items()
}
EXPECTED_ARTIFACT_PROVENANCE = {
    "fold_schema": FOLD_SCHEMA,
    "merged_schema": MERGED_SCHEMA,
    "fold_frozen_input_sources_field": "frozen_input_sources",
    "source_record_fields_exact": [
        "path",
        "role",
        "used_in_candidate_prediction",
        "outer_fold",
        "seed",
    ],
    "required_source_records": EXPECTED_SOURCE_RECORDS,
    "fold_parameter_provenance": {
        "point_module_total_field": "candidate_parameter_count",
        "point_module_trainable_field": "candidate_trainable_parameter_count",
        "residual_trainable_field": "residual_parameter_counts.candidate",
        "candidate_specific_trainable_field": (
            "candidate_specific_trainable_parameter_counts.candidate"
        ),
        "candidate_specific_trainable_expected": 1_468_165,
    },
    "merged_integrity_required": {
        "complete_frozen_input_provenance_all_runs": True,
        "expected_candidate_specific_trainable_parameter_count_each": 1_468_165,
    },
}

EXPECTED_OPERATOR = {
    "id": POSITION_ALIGNED_OPERATOR,
    "constructs_per_puzzle": EXPECTED_CONSTRUCTS_PER_PUZZLE,
    "focal_query_tokens": 1,
    "focal_present_in_key_value": False,
    "nonfocal_individual_key_value_tokens": 7,
    "nonfocal_summary_key_value_tokens": 1,
    "nonfocal_summary_dedicated_learned_token": False,
    "total_key_value_tokens": 8,
    "attention_weight_dropout": 0.0,
    "residual_and_ffn_dropout": 0.1,
    "focal_query_value_residual": False,
    "raw_zero_reference_before_learned_projection": True,
    "raw_zero_streams": ["HIDDEN", "WT_REACTIVITY", "WT_OBSERVED"],
    "actual_and_reference_focal_query_elementwise_equal": True,
    "actual_and_reference_full_shared_path": "PROJECTION_ATTENTION_FFN_OUTPUT_NORM",
    "actual_and_reference_same_dropout_draw": True,
    "actual_and_reference_rng_advances_as_one_call": True,
    "returned_cross_state": "F_Q_V_MINUS_F_Q_V0",
    "raw_zero_cross_state_exact_zero": True,
}
EXPECTED_POINT_INCREMENT = {
    "family": "SHARED_PAIRED_HEAD_CONTRAST",
    "expression": "H_BASE_CROSS_MINUS_H_BASE_ZERO",
    "actual_and_reference_same_parameters": True,
    "actual_and_reference_same_dropout_draw": True,
    "actual_and_reference_rng_advances_as_one_call": True,
    "zero_cross_increment_exact_zero": True,
    "prediction": "FROZEN_V13_POINT_PLUS_PAIRED_INCREMENT",
}
EXPECTED_PARAMETER_COUNTS = {
    "p1_point_module_total": EXPECTED_TOTAL_PARAMETERS,
    "p1_point_module_frozen_v14_encoder": V14_ENCODER_PARAMETERS,
    "p1_point_module_trainable": EXPECTED_TRAINABLE_PARAMETERS,
    "set_operator_trainable": EXPECTED_CONTEXT_PRETRAINING_PARAMETERS,
    "paired_point_head_trainable": (
        EXPECTED_TRAINABLE_PARAMETERS - EXPECTED_CONTEXT_PRETRAINING_PARAMETERS
    ),
    "residual_calibration_head_trainable": EXPECTED_RESIDUAL_PARAMETERS,
    "p1_point_plus_residual_modules_total": (
        EXPECTED_TOTAL_PARAMETERS + EXPECTED_RESIDUAL_PARAMETERS
    ),
    "candidate_specific_trainable_point_plus_distribution": (
        EXPECTED_TRAINABLE_PARAMETERS + EXPECTED_RESIDUAL_PARAMETERS
    ),
    "temporary_pretraining_decoder": EXPECTED_DECODER_PARAMETERS,
    "pretraining_trainable_with_decoder": EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS,
    "frozen_v13_point_upstream": V13_POINT_PARAMETERS,
    "frozen_v8_meanaligned_upstream": 109_581,
    "feature41_ridge_upstream": "PENDING_ACTIVATION_BINDING",
    "full_upstream_footprint": "PENDING_ACTIVATION_BINDING",
    "p1_point_module_total_is_full_pipeline_total": False,
    "candidate_and_null_equal": True,
}
EXPECTED_RETENTION_GATE = {
    "required_phases": ["P1M3", "P1M4"],
    "data": "OUTER_TRAIN_PUZZLE_CONTEXTS_ONLY",
    "held_puzzle_access_allowed": False,
    "mutant_outcome_access_allowed": False,
    "diagnostic_mask_epoch": RETENTION_DIAGNOSTIC_EPOCH,
    "training_mask_epochs": [0, 199],
    "same_final_frozen_decoder_for_all_snapshots": True,
    "snapshots": ["INITIAL", "POST_PRETRAINING", "POST_POINT"],
    "candidate_pretraining_established_all_runs_required": True,
    "candidate_retention_positive_all_runs_required": True,
    "null_retention_report_only": True,
    "selection_allowed": False,
    "failure_status": "PUZZLE_SET_TRAIN_ONLY_RETENTION_GATE_FAIL",
    "held_score_remains_closed_on_failure": True,
}

EXPECTED_SCREEN_GATES = {
    "signed_delta": {
        "relative_gain_vs_feature41_min": 0.12,
        "relative_gain_vs_terminal_v12_min": 0.02,
        "relative_gain_vs_frozen_v13_parent_min": 0.02,
        "relative_gain_vs_matched_null_min": 0.015,
        "positive_puzzles": {
            "feature41": 16,
            "terminal_v12": 14,
            "frozen_v13_parent": 14,
            "matched_null": 14,
        },
        "paired_ci_lower_each_gt": 0.0,
    },
    "point_absolute": {
        "relative_gain_vs_feature41_min": 0.07,
        "relative_gain_vs_terminal_v11_min": 0.02,
        "relative_gain_vs_frozen_v13_parent_min": 0.02,
        "relative_gain_vs_matched_null_min": 0.01,
        "positive_puzzles": {
            "feature41": 16,
            "terminal_v11": 14,
            "frozen_v13_parent": 14,
            "matched_null": 14,
        },
        "paired_ci_lower_each_gt": 0.0,
    },
    "task_crps": {
        "relative_gain_vs_feature41_min": 0.05,
        "relative_gain_vs_terminal_v12_min": 0.02,
        "relative_gain_vs_matched_null_min": 0.015,
        "positive_puzzles": {
            "feature41": 16,
            "terminal_v12": 14,
            "matched_null": 14,
        },
        "paired_ci_lower_each_gt": 0.0,
    },
    "distribution_absolute": {
        "relative_gain_vs_feature41_min": 0.15,
        "relative_gain_vs_terminal_v10_min": 0.02,
        "relative_gain_vs_matched_null_min": 0.01,
        "positive_puzzles": {
            "feature41": 16,
            "terminal_v10": 14,
            "matched_null": 14,
        },
        "paired_ci_lower_each_gt": 0.0,
    },
    "stability": {
        "leave_one_puzzle_positive_all_headline_comparisons": True,
        "max_single_puzzle_effect_fraction": 0.20,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "unexpected_keys": 0,
    },
    "calibration": {
        "levels": [0.68, 0.95],
        "max_absolute_error_worsening_vs_feature41": 0.01,
    },
}
EXPECTED_SCORE_ONCE = {
    "training_must_be_closed": True,
    "candidate_training_must_be_closed": True,
    "partial_fold_score_read_allowed": False,
    "external_outcome_access_allowed": False,
    "screen_token": SCREEN_SCORE_TOKEN,
    "formal_token": FORMAL_SCORE_TOKEN,
    "score_invocations_per_complete_universe": 1,
    "overwrite_allowed": False,
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Puzzle-Set V5 YAML root must be a mapping: {path}")
    return value


def _assert_declared_paths_exist(repo_root: Path, contract: dict[str, Any]) -> None:
    if contract.get("documents") != EXPECTED_DOCUMENTS:
        raise RuntimeError("Puzzle-Set V5 document registry changed")
    for label, relative_path in EXPECTED_DOCUMENTS.items():
        if not (repo_root / relative_path).is_file():
            raise RuntimeError(f"Puzzle-Set V5 document is missing: {label}")
    if (
        contract.get("post_v14_router", {}).get("source_path") != (EXPECTED_ROUTER_PATH)
        or not (repo_root / EXPECTED_ROUTER_PATH).is_file()
    ):
        raise RuntimeError("Puzzle-Set V5 post-V14 router is missing")
    for label, relative_path in EXPECTED_EXECUTABLE_PATHS.items():
        if not (repo_root / relative_path).is_file():
            raise RuntimeError(f"Puzzle-Set V5 declared runtime is missing: {label}")


def _assert_v14_remains_sole_active(repo_root: Path, active: dict[str, Any]) -> None:
    if active.get("schema_version") != EXPECTED_ACTIVE_SCHEMA:
        raise RuntimeError("V14 active authority schema is not the sole pointer")
    if active.get("project_task_id") != EXPECTED_ACTIVE_PROJECT:
        raise RuntimeError("V14 is not the sole active project")
    authority = active.get("authority", {})
    expected_authority = {
        "pointer_role": EXPECTED_ACTIVE_ROLE,
        "current_phase": EXPECTED_ACTIVE_PHASE,
        "current_authority_state": EXPECTED_ACTIVE_STATE,
        "current_runnable_phase": EXPECTED_ACTIVE_PHASE,
        "branch": EXPECTED_ACTIVE_BRANCH,
        "worktree": EXPECTED_ACTIVE_WORKTREE,
        "artifact_root": EXPECTED_ACTIVE_ARTIFACT_ROOT,
        "machine_contract_path": EXPECTED_V14_MACHINE_CONTRACT,
        "binding_status": EXPECTED_ACTIVE_BINDING,
    }
    for key, expected in expected_authority.items():
        if authority.get(key) != expected:
            raise RuntimeError(f"V14 active execution state changed: {key}")
    for key, expected in {
        "runnable_phases": [EXPECTED_ACTIVE_PHASE],
        "training_allowed": EXPECTED_ACTIVE_TRAINING_TOKEN,
        "candidate_model_training_allowed": EXPECTED_ACTIVE_TRAINING_TOKEN,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }.items():
        if active.get(key) != expected:
            raise RuntimeError(f"V14 active execution state changed: {key}")
    active_authorization = active.get("authorization", {})
    if {
        "neural_training_allowed": active_authorization.get("neural_training_allowed"),
        "smoke_allowed": active_authorization.get("smoke_allowed"),
        "screen_allowed": active_authorization.get("screen_allowed"),
        "formal_confirmation_allowed": active_authorization.get(
            "formal_confirmation_allowed"
        ),
        "new_external_outcome_access_allowed": active_authorization.get(
            "new_external_outcome_access_allowed"
        ),
    } != {
        "neural_training_allowed": EXPECTED_ACTIVE_TRAINING_TOKEN,
        "smoke_allowed": False,
        "screen_allowed": True,
        "formal_confirmation_allowed": False,
        "new_external_outcome_access_allowed": False,
    }:
        raise RuntimeError("V14 active authorization changed")
    gate_state = active.get("gate_state", {})
    if (
        gate_state.get("V14M3") != "TWENTY_FOLD_PREDICTION_ONLY_SCREEN_AUTHORIZED"
        or gate_state.get("V14M4") != "NOT_AUTHORIZED"
    ):
        raise RuntimeError("V14 active Gate state changed")
    v14_machine_path = repo_root / EXPECTED_V14_MACHINE_CONTRACT
    if not v14_machine_path.is_file():
        raise RuntimeError("V14 referenced machine contract is missing")
    v14_machine = _load(v14_machine_path)
    if (
        v14_machine.get("schema_version") != EXPECTED_V14_MACHINE_SCHEMA
        or v14_machine.get("contract_id") != EXPECTED_V14_MACHINE_ID
    ):
        raise RuntimeError("V14 referenced machine contract identity changed")


def _assert_inactive_declaration(contract: dict[str, Any]) -> None:
    if contract.get("history_boundary") != EXPECTED_HISTORY_BOUNDARY:
        raise RuntimeError("Puzzle-Set V5 history boundary changed")
    if contract.get("inactive_authority") != EXPECTED_INACTIVE_AUTHORITY:
        raise RuntimeError("Puzzle-Set V5 inactive authority changed")
    if contract.get("post_v14_router") != EXPECTED_POST_V14_ROUTER:
        raise RuntimeError("Puzzle-Set V5 post-V14 route conditions changed")
    if contract.get("future_phase_training_tokens") != EXPECTED_FUTURE_PHASE_TOKENS:
        raise RuntimeError("Puzzle-Set V5 future phase token declaration changed")


def _assert_scope_parents_inputs_and_artifacts(contract: dict[str, Any]) -> None:
    scope = contract.get("scope", {})
    if {key: scope.get(key) for key in EXPECTED_SCOPE_IDENTITY} != (
        EXPECTED_SCOPE_IDENTITY
    ):
        raise RuntimeError("Puzzle-Set V5 task, data, split, or estimand changed")
    if contract.get("frozen_parents") != EXPECTED_FROZEN_PARENTS:
        raise RuntimeError("Puzzle-Set V5 frozen parent identity changed")
    if contract.get("frozen_input_sources") != EXPECTED_FROZEN_INPUT_SOURCES:
        raise RuntimeError("Puzzle-Set V5 frozen input-source declaration changed")
    if contract.get("artifact_schemas_and_provenance") != (
        EXPECTED_ARTIFACT_PROVENANCE
    ):
        raise RuntimeError("Puzzle-Set V5 artifact provenance declaration changed")
    if scope.get("model_or_threshold_selection_allowed") is not False:
        raise RuntimeError("Puzzle-Set V5 model selection was authorized")
    if scope.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("Puzzle-Set V5 external outcomes were authorized")
    if (
        scope.get("evidence_ceiling") != "POST_HOC_DEVELOPMENT_PASS"
        or scope.get("publication_ready") is not False
        or scope.get("sota") != "NOT_ESTABLISHED"
        or scope.get("external_replication") != "NOT_ESTABLISHED"
    ):
        raise RuntimeError("Puzzle-Set V5 evidence ceiling was broadened")
    if scope.get("prohibited_prediction_inputs") != [
        "METHOD_ID",
        "PUZZLE_ID",
        "DATASET_ID",
        "HELD_MUTANT_REACTIVITY",
        "HELD_MUTANT_ERROR",
        "HELD_QUALIFIED_TARGET_MASK",
        "EXTERNAL_OUTCOME",
    ]:
        raise RuntimeError("Puzzle-Set V5 prohibited input boundary changed")


def _assert_runtime_constant_alignment() -> None:
    if {
        RUNTIME_PROJECT_TASK,
        SCREEN_SCORE_PROJECT,
        FORMAL_SCORE_PROJECT,
    } != {"reactflow_delta_puzzle_set_meta_context"}:
        raise RuntimeError("Puzzle-Set runtime project-task constant changed")
    if PHASE_TRAINING_TOKENS != EXPECTED_PHASE_TOKENS or RUNTIME_PHASES != set(
        EXPECTED_PHASE_TOKENS
    ):
        raise RuntimeError("Puzzle-Set runtime phase token constants changed")
    if SCREEN_SCORE_PHASE != "P1M3" or FORMAL_SCORE_PHASE != "P1M4":
        raise RuntimeError("Puzzle-Set runtime score phases changed")
    if tuple(FORMAL_FOLDS) != tuple(range(20)) or tuple(FORMAL_SEEDS) != tuple(
        range(5)
    ):
        raise RuntimeError("Puzzle-Set formal runtime universe changed")
    if (
        FORMAL_PRETRAINING_EPOCHS,
        FORMAL_POINT_EPOCHS,
        FORMAL_CALIBRATION_EPOCHS,
    ) != (200, 40, 40):
        raise RuntimeError("Puzzle-Set formal runtime schedule changed")


def _assert_frozen_models(contract: dict[str, Any]) -> None:
    models = contract.get("models", {})
    if models.get("only_allowed_candidate") != {
        "id": "puzzle_set_meta_context_v5_aligned",
        "connectivity": FULL_CROSS_CONSTRUCT,
        "nonfocal_coordinate_alignment": "REGISTERED",
    }:
        raise RuntimeError("Puzzle-Set V5 candidate changed")
    if models.get("only_allowed_matched_null") != {
        "id": "puzzle_set_meta_context_v5_shift17_null",
        "connectivity": POSITION_DERANGED_NULL,
        "nonfocal_coordinate_alignment": (
            f"CIRCULAR_SHIFT_{POSITION_DERANGEMENT_SHIFT}"
        ),
    }:
        raise RuntimeError("Puzzle-Set V5 matched null changed")
    if models.get("only_allowed_difference") != (
        "REGISTERED_VERSUS_FIXED_SHIFT17_NONFOCAL_ALIGNMENT"
    ):
        raise RuntimeError("Puzzle-Set V5 attribution contrast changed")
    if models.get("operator") != EXPECTED_OPERATOR:
        raise RuntimeError("Puzzle-Set V5 operator changed")
    if models.get("point_increment") != EXPECTED_POINT_INCREMENT:
        raise RuntimeError("Puzzle-Set V5 paired point cancellation changed")
    counts = models.get("exact_parameter_counts_each")
    if counts != EXPECTED_PARAMETER_COUNTS:
        raise RuntimeError("Puzzle-Set V5 exact parameter accounting changed")
    if (
        counts["p1_point_module_frozen_v14_encoder"]
        + counts["p1_point_module_trainable"]
        != counts["p1_point_module_total"]
        or counts["set_operator_trainable"] + counts["paired_point_head_trainable"]
        != counts["p1_point_module_trainable"]
        or counts["p1_point_module_total"]
        + counts["residual_calibration_head_trainable"]
        != counts["p1_point_plus_residual_modules_total"]
        or counts["p1_point_module_trainable"]
        + counts["residual_calibration_head_trainable"]
        != counts["candidate_specific_trainable_point_plus_distribution"]
        or counts["set_operator_trainable"] + counts["temporary_pretraining_decoder"]
        != counts["pretraining_trainable_with_decoder"]
    ):
        raise RuntimeError("Puzzle-Set V5 parameter accounting is inconsistent")


def _assert_frozen_training(contract: dict[str, Any]) -> None:
    if contract.get("pretraining") != {
        "data": "OUTER_TRAIN_WT_PUZZLE_CONTEXTS_ONLY",
        "mutant_outcome_access_allowed": False,
        "held_puzzle_access_allowed": False,
        "mask_fraction": PRETRAINING_MASK_FRACTION,
        "objective": "PUZZLE_BALANCED_MASKED_WT_REACTIVITY_L1",
        "epochs": {"smoke": 3, "screen_and_formal": 200},
        "optimizer": "ADAMW",
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "gradient_clip": 5.0,
        "encoder_trainable": False,
        "point_head_trainable": False,
        "only_set_operator_and_temporary_decoder_trainable": True,
        "candidate_and_null_same_masks_order_budget_and_random_stream": True,
        "early_stopping_allowed": False,
    }:
        raise RuntimeError("Puzzle-Set V5 pretraining protocol changed")
    if contract.get("point_training") != {
        "objective": "EXACT_METHOD_BALANCED_SIGNED_DELTA_L1",
        "epochs": {"smoke": 3, "screen_and_formal": 40},
        "head_only_warmup_epochs": POINT_HEAD_WARMUP_EPOCHS,
        "joint_context_and_head_epochs": {"smoke": 2, "screen_and_formal": 39},
        "optimizer": "ADAM",
        "head_learning_rate": POINT_HEAD_LR,
        "context_learning_rate": POINT_CONTEXT_LR,
        "weight_decay": 0.0,
        "gradient_clip": POINT_GRADIENT_CLIP,
        "early_stopping_allowed": False,
        "epoch_or_checkpoint_selection_allowed": False,
        "candidate_and_null_same_order_budget_and_random_stream": True,
    }:
        raise RuntimeError("Puzzle-Set V5 point-training protocol changed")
    if contract.get("residual_calibration") != {
        "family": ("V10_MEDIAN_PRESERVING_ASYMMETRIC_TWO_COMPONENT_GAUSSIAN_MIXTURE"),
        "point_frozen": True,
        "objective": "EXACT_METHOD_BALANCED_GAUSSIAN_MIXTURE_CRPS",
        "epochs": {"smoke": 3, "screen_and_formal": 40},
        "optimizer": "ADAM",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_clip": 5.0,
        "point_gradient_allowed": False,
        "family_or_hyperparameter_search_allowed": False,
    }:
        raise RuntimeError("Puzzle-Set V5 residual-calibration protocol changed")
    if contract.get("retention_pre_score_gate") != EXPECTED_RETENTION_GATE:
        raise RuntimeError("Puzzle-Set V5 train-only retention Gate changed")


def _assert_frozen_universes_and_gates(contract: dict[str, Any]) -> None:
    if contract.get("p1m2_smoke") != {
        "activation_prerequisite": "FOCUSED_PUZZLE_SET_V5_ACTIVATION_COMMIT",
        "training_token": EXPECTED_PHASE_TOKENS["P1M2"],
        "folds": [0, 1],
        "seeds": [0],
        "pretraining_epochs": 3,
        "point_epochs": 3,
        "calibration_epochs": 3,
        "scientific_score_allowed": False,
        "artifact_status": "ENGINEERING_SMOKE_ONLY",
        "exact_pass_status": "P1M2_ENGINEERING_SMOKE_PASS",
    }:
        raise RuntimeError("Puzzle-Set V5 smoke universe changed")
    screen = contract.get("p1m3_screen", {})
    expected_screen = {
        "prerequisite": "P1M2_ENGINEERING_SMOKE_PASS",
        "training_token": EXPECTED_PHASE_TOKENS["P1M3"],
        "folds": 20,
        "fold_ids": list(range(20)),
        "seeds": [0],
        "pretraining_epochs": 200,
        "point_epochs": 40,
        "calibration_epochs": 40,
        "complete_before_score_access": True,
        "score_once_after_complete_merge": True,
    }
    if {key: screen.get(key) for key in expected_screen} != expected_screen:
        raise RuntimeError("Puzzle-Set V5 screen universe changed")
    if screen.get("gates") != EXPECTED_SCREEN_GATES:
        raise RuntimeError("Puzzle-Set V5 screen Gate changed")
    if contract.get("p1m4_formal") != {
        "prerequisite": "PUZZLE_SET_M3_TOP_JOURNAL_SCREEN_PASS",
        "training_token": EXPECTED_PHASE_TOKENS["P1M4"],
        "folds_per_seed": len(FORMAL_FOLDS),
        "fold_ids": list(FORMAL_FOLDS),
        "seeds": list(FORMAL_SEEDS),
        "screen_seed_zero_reused": False,
        "pretraining_epochs": FORMAL_PRETRAINING_EPOCHS,
        "point_epochs": FORMAL_POINT_EPOCHS,
        "calibration_epochs": FORMAL_CALIBRATION_EPOCHS,
        "equal_seed_mixture": True,
        "gaussian_components_per_arm": 10,
        "repeat_all_screen_gates": True,
        "minimum_positive_individual_seeds_signed_vs_feature41": 4,
        "minimum_positive_individual_seeds_crps_vs_feature41": 4,
        "formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources": (
            True
        ),
        "model_or_seed_selection_allowed": False,
        "failed_seed_removal_allowed": False,
    }:
        raise RuntimeError("Puzzle-Set V5 formal universe or source Gate changed")
    if contract.get("score_once") != EXPECTED_SCORE_ONCE:
        raise RuntimeError("Puzzle-Set V5 score-once protocol changed")


def _assert_decision_ledger(contract: dict[str, Any], ledger: dict[str, Any]) -> None:
    if ledger.get("schema_version") != EXPECTED_LEDGER_SCHEMA:
        raise RuntimeError("Puzzle-Set V5 ledger schema changed")
    if ledger.get("contract_id") != EXPECTED_CONTRACT_ID:
        raise RuntimeError("Puzzle-Set V5 ledger contract identity changed")
    if ledger.get("status") != EXPECTED_STATUS:
        raise RuntimeError("Puzzle-Set V5 ledger status changed")
    expected_authority = {
        "active_experiment": "MODEL_RESCUE_V14",
        "proposed_task_id": RUNTIME_PROJECT_TASK,
        "p1_training_allowed": False,
        "p1_candidate_training_allowed": False,
        "p1_held_score_read_allowed": False,
        "partial_fold_scores_allowed": False,
        "external_outcome_access_allowed": False,
        "modifies_v14": False,
        "implementation_design_head": ("b5450ff51699cd64f43d7c59481be6362baee247"),
        "activation_parent_head": (
            "TO_BE_RECORDED_AFTER_COMPLETE_V14_TERMINAL_HANDOFF"
        ),
        "activation_requires_new_focused_commit": True,
    }
    if ledger.get("authority") != expected_authority:
        raise RuntimeError("Puzzle-Set V5 ledger authority changed")
    if ledger.get("activation_conditions") != [
        "COMPLETE_V14_TERMINAL_HANDOFF",
        "FIRST_MATCHING_POST_V14_CONTINGENCY_BRANCH_SELECTS_P1",
        (
            "BRANCH_SPECIFIC_ROUTE_PROBE_STATE_BOUND_AS_NOT_APPLICABLE_OR_"
            "REQUIRED_EXACT_PASS"
        ),
        "ALL_REALIZED_FROZEN_INPUT_PATHS_ROLES_AND_COUNTS_BOUND",
        (
            "INACTIVE_MACHINE_CONTRACT_PROMOTED_AND_ACTIVE_POINTER_FROZEN_IN_"
            "FOCUSED_COMMIT"
        ),
        "CONTRACT_VALIDATOR_AND_V5_FOCUSED_TESTS_PASS_WITH_TRAINING_CLOSED",
    ]:
        raise RuntimeError("Puzzle-Set V5 ledger activation conditions changed")
    if ledger.get("activation_exclusions") != {
        "exact_v14m3_pass_routes_only_to_v14m4": True,
        "v14m4_formal_failure_routes_to_p3_without_new_independent_data": True,
        "implementation_or_test_pass_can_activate_p1": False,
        "p1_and_p2_may_be_active_together": False,
    }:
        raise RuntimeError("Puzzle-Set V5 ledger activation exclusions changed")
    router = ledger.get("post_v14_router", {})
    if (
        router.get("source_path") != EXPECTED_ROUTER_PATH
        or router.get("selection_rule") != "FIRST_MATCHING_BRANCH_CONTROLS"
        or router.get("selected_router_branch_id")
        != "PENDING_COMPLETE_V14_TERMINAL_HANDOFF"
        or router.get("route_probe")
        != {"requirement": "NOT_EVALUATED", "status": "NOT_EVALUATED"}
        or set(router.get("activation_cases", {})) != {"3", "4", "5"}
        or set(router.get("non_p1_cases", {})) != {"1", "2", "6", "7"}
    ):
        raise RuntimeError("Puzzle-Set V5 ledger router state changed")
    branch_5_criteria = router["activation_cases"]["5"]["route_probe"].get(
        "fixed_complete_outer_train_only_criteria"
    )
    if branch_5_criteria != {
        key: value
        for key, value in EXPECTED_ROUTE_PROBE_CRITERIA.items()
        if key != "data_access"
    }:
        raise RuntimeError("Puzzle-Set V5 ledger branch-5 probe changed")
    if ledger.get("phase_state") != {
        "P1M0": "DRAFT_FROZEN_INACTIVE",
        "P1M1": "NOT_AUTHORIZED",
        "P1M2": "NOT_AUTHORIZED",
        "P1M3": "NOT_AUTHORIZED",
        "P1M4": "NOT_AUTHORIZED",
        "P1M5": "NOT_RUN",
    }:
        raise RuntimeError("Puzzle-Set V5 ledger phase state changed")
    formal_gate = ledger.get("formal_additional_gates", {})
    if (
        formal_gate.get(
            "formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources"
        )
        is not True
        or contract.get("p1m4_formal", {}).get(
            "formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources"
        )
        is not True
    ):
        raise RuntimeError("Puzzle-Set V5 formal source Gate changed")
    ledger_inputs = ledger.get("frozen_input_sources", {})
    expected_ledger_sources = {
        key: EXPECTED_FROZEN_INPUT_SOURCES[key]
        for key in (
            "v13_seed0_point",
            "v14_seed0_encoder",
            "v8_seed0_meanaligned",
            "tic2a_feature41_ridge",
            "tic2a_merged_registry",
            "unconstrained_feature_cache",
            "constrained_feature_cache",
            "v10",
        )
    }
    if (
        ledger_inputs.get("activation_binding_status")
        != "REALIZED_PATHS_ROLES_AND_COUNTS_PENDING"
        or ledger_inputs.get("each_outer_fold_0_through_19_must_be_bound") is not True
        or ledger_inputs.get("sources") != expected_ledger_sources
        or ledger_inputs.get("realized_binding_fields")
        != EXPECTED_FROZEN_INPUT_SOURCES["activation_binding_required_fields"]
        or ledger_inputs.get("feature41_ridge_realized_parameter_count")
        != "PENDING_ACTIVATION_BINDING"
        or ledger_inputs.get("full_upstream_parameter_footprint")
        != "PENDING_ACTIVATION_BINDING"
    ):
        raise RuntimeError("Puzzle-Set V5 ledger input binding state changed")
    ledger_artifacts = ledger.get("artifact_schemas_and_provenance", {})
    expected_ledger_records = {
        name: [
            spec["role"],
            spec["used_in_candidate_prediction"],
            spec["outer_fold_scope"],
            spec["seed"],
        ]
        for name, spec in EXPECTED_SOURCE_RECORDS.items()
    }
    if (
        ledger_artifacts.get("fold_schema") != FOLD_SCHEMA
        or ledger_artifacts.get("merged_schema") != MERGED_SCHEMA
        or ledger_artifacts.get("required_source_records") != expected_ledger_records
        or ledger_artifacts.get("fold_candidate_specific_trainable_parameter_expected")
        != 1_468_165
        or ledger_artifacts.get(
            "merged_complete_frozen_input_provenance_all_runs_required"
        )
        is not True
        or ledger_artifacts.get(
            "merged_candidate_specific_trainable_parameter_expected"
        )
        != 1_468_165
    ):
        raise RuntimeError("Puzzle-Set V5 ledger artifact provenance changed")


def validate_contract(repo_root: Path) -> dict[str, Any]:
    contract = _load(
        repo_root / "configs/reactflow_delta/puzzle_set_meta_context_v5_amendment.yaml"
    )
    active = _load(repo_root / "configs/reactflow_delta/active_contract.yaml")
    ledger = _load(repo_root / EXPECTED_DOCUMENTS["decision_ledger_path"])
    if contract.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError("Puzzle-Set V5 machine schema changed")
    if contract.get("contract_id") != EXPECTED_CONTRACT_ID:
        raise RuntimeError("Puzzle-Set V5 contract identity changed")
    if contract.get("contract_status") != EXPECTED_STATUS:
        raise RuntimeError("Puzzle-Set V5 inactive status changed")

    _assert_declared_paths_exist(repo_root, contract)
    _assert_v14_remains_sole_active(repo_root, active)
    _assert_inactive_declaration(contract)
    _assert_scope_parents_inputs_and_artifacts(contract)
    _assert_runtime_constant_alignment()
    _assert_frozen_models(contract)
    _assert_frozen_training(contract)
    _assert_frozen_universes_and_gates(contract)
    _assert_decision_ledger(contract, ledger)

    if contract.get("terminal_rules") != {
        "any_screen_or_formal_gate_failure": (
            "TERMINATE_PUZZLE_SET_META_CONTEXT_FAMILY"
        ),
        "gate_lowering_allowed": False,
        "same_family_successor_allowed": False,
        "external_confirmation_requires_separate_amendment": True,
    }:
        raise RuntimeError("Puzzle-Set V5 terminal rules changed")
    return {
        "status": "PUZZLE_SET_V5_INACTIVE_DECLARATION_VALIDATION_PASS",
        "contract_status": contract["contract_status"],
        "active_project_task_id": active["project_task_id"],
        "active_phase": active["authority"]["current_phase"],
        "activation_allowed_now": contract["inactive_authority"][
            "activation_allowed_now"
        ],
        "training_allowed": contract["inactive_authority"]["training_allowed"],
        "held_score_read_allowed": contract["inactive_authority"][
            "held_score_read_allowed"
        ],
        "external_outcome_access_allowed": contract["inactive_authority"][
            "new_external_outcome_access_allowed"
        ],
        "runtime_execution_authorized": False,
        "validation_scope": "INACTIVE_DECLARATION_AND_RUNTIME_CONSTANT_ALIGNMENT_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate_contract(args.repo_root.resolve())
    print(yaml.safe_dump(result, sort_keys=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
