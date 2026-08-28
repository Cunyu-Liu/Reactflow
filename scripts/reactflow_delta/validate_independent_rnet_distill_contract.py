#!/usr/bin/env python3
"""Fail-closed authority validator for the independent RNet2 distillation project."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


PROJECT_TASK_ID = "reactflow_delta_independent_rnet_distill"
BRANCH = "codex/reactflow-delta-independent-rnet-distill-20260828"
WORKTREE = Path(
    "/home/cunyuliu/reactflow_delta_worktrees/independent_rnet_distill_20260828"
)
ARTIFACT_ROOT = Path("/mnt/cunyuliu/reactflow_delta_independent_rnet_distill")
ACTIVE_PATH = Path("configs/reactflow_delta/active_contract.yaml")
CONTRACT_PATH = Path(
    "configs/reactflow_delta/independent_rnet_distill_contract.yaml"
)
LEDGER_PATH = Path(
    "docs/prospective_v2/independent_rnet_distill_decision_ledger.yaml"
)
RESEARCH_PATH = Path(
    "autoresearch/orchestrator-260828-independent-rnet-distill/research.md"
)
PHASES = (
    "RND0",
    "RND1",
    "RND2",
    "RND3",
    "RND4",
    "RND5",
    "RND5T",
    "RND6P",
    "RND6S",
    "RND6Q",
    "RND6T",
)
FORMAL_PHASES = ("RND6P", "RND6S", "RND6Q", "RND6T")
TOKENS = {
    "RND1": "RNET_DISTILL_PAIRED_GPU_PRETRAIN_ONCE_ONLY",
    "RND2": "RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE_ONLY",
    "RND3": "RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY",
    "RND4": "RNET_DISTILL_COMPLETE_MERGE_SCORE_ONCE_ONLY",
    "RND5": "RNET_DISTILL_COMPLETE_SCORE_QUALIFIER_ONCE_ONLY",
    "RND5T": "RNET_DISTILL_SCREEN_TERMINAL_CLOSED",
    "RND6P": "RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_GPU_PREDICTION_ONLY",
    "RND6S": "RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_SCORE_ONCE_ONLY",
    "RND6Q": "RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_QUALIFIER_ONCE_ONLY",
    "RND6T": "RNET_DISTILL_FORMAL_TERMINAL_CLOSED",
}
RND0_TOKEN = "RND0_IMPLEMENTATION_SOURCE_BINDING_ONLY"
RND1_PASS = "RND1_PAIRED_PRETRAIN_EXACT_PASS"
RND2_MERGE_PASS = "RND2_COMPLETE_UNSCORED_ENGINEERING_SMOKE_MERGE_PASS"
RND3_MERGE_PASS = "RND3_COMPLETE_UNSCORED_PREDICTION_MERGE_PASS"
RND4_SCORE_PASS = "RND4_COMPLETE_SCORE_PASS"
RND5_SCREEN_PASS = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_PASS"
RND5_SCREEN_FAIL = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_FAIL"
RND5_SCREEN_INDETERMINATE = (
    "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_INDETERMINATE"
)
RND5_NONPASS_STATUSES = (RND5_SCREEN_FAIL, RND5_SCREEN_INDETERMINATE)
RND6_MERGE_PASS = "RND6P_COMPLETE_UNSCORED_FORMAL_MERGE_PASS"
RND6_ASSEMBLY_PASS = "RND6P_EQUAL_SEED_PREDICTION_ONLY_ASSEMBLY_PASS"
RND6_SCORE_PASS = "RND6S_COMPLETE_FORMAL_SCORE_PASS"
RND6_QUALIFICATION_PASS = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_FORMAL_PASS"
RND6_QUALIFICATION_FAIL = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_FORMAL_FAIL"
RND6_QUALIFICATION_INDETERMINATE = (
    "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_FORMAL_INDETERMINATE"
)
RND6_QUALIFICATION_STATUSES = (
    RND6_QUALIFICATION_PASS,
    RND6_QUALIFICATION_FAIL,
    RND6_QUALIFICATION_INDETERMINATE,
)
FORMAL_PHASE_ORDER = list(FORMAL_PHASES)
FORMAL_FOLDS = list(range(20))
FORMAL_SEEDS = list(range(5))
FORMAL_PAIR_COUNT = 100
FORMAL_POINT_EPOCHS = 40
FORMAL_CALIBRATION_EPOCHS = 40
FORMAL_EQUAL_SEED_WEIGHT = 0.2
FORMAL_INACTIVE_STATUS = "FROZEN_INACTIVE_PENDING_RND5_RESULT"
FORMAL_NOT_RUN_RND5_NONPASS_STATUS = "TERMINAL_RND5_NONPASS_RND6_NOT_RUN"
FORMAL_LIFECYCLE_BY_PHASE = {
    "RND6P": "ACTIVE_RND6P",
    "RND6S": "ACTIVE_RND6S",
    "RND6Q": "ACTIVE_RND6Q",
    "RND6T": "TERMINAL_RND6_CLOSED",
}
TERMINAL_PHASES = ("RND5T", "RND6T")
RND1_ACTION = "RUN_SINGLE_RND1_PAIRED_GPU_PRETRAIN"
RND1_LEDGER_ACTION = "RUN_SINGLE_RND1_PAIRED_GPU_PRETRAIN_AND_VERIFY_REAL_CUDA_PLACEMENT_ONCE"
RND1_AUTHORIZATION = {
    "scope": "RND1_SINGLE_PAIRED_GPU_PRETRAIN_ONLY",
    "implementation_allowed": False,
    "neural_training_allowed": True,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": False,
    "qualification_allowed": False,
    "formal_confirmation_allowed": False,
    "new_external_outcome_access_allowed": False,
}
RND1_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": "AUTHORIZED_SINGLE_PAIRED_GPU_PRETRAIN",
    "RND2": "NOT_AUTHORIZED",
    "RND3": "NOT_AUTHORIZED",
    "RND4": "NOT_AUTHORIZED",
    "RND5": "NOT_AUTHORIZED",
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": "NOT_AUTHORIZED",
    "RND6S": "NOT_AUTHORIZED",
    "RND6Q": "NOT_AUTHORIZED",
    "RND6T": "NOT_AUTHORIZED",
}
RND2_SCOPE = "RND2_TWO_FOLD_GPU_ENGINEERING_SMOKE_ONLY"
RND2_ACTION = "RUN_SINGLE_RND2_TWO_FOLD_GPU_ENGINEERING_SMOKE"
RND2_DECISION = (
    "CLOSE_RND1_AND_AUTHORIZE_SINGLE_RND2_TWO_FOLD_GPU_ENGINEERING_SMOKE"
)
RND2_AUTHORIZATION = {
    "scope": RND2_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": True,
    "smoke_allowed": True,
    "screen_allowed": False,
    "score_allowed": False,
    "qualification_allowed": False,
    "formal_confirmation_allowed": False,
    "new_external_outcome_access_allowed": False,
}
RND2_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": RND1_PASS,
    "RND2": "AUTHORIZED_TWO_FOLD_GPU_ENGINEERING_SMOKE",
    "RND3": "NOT_AUTHORIZED",
    "RND4": "NOT_AUTHORIZED",
    "RND5": "NOT_AUTHORIZED",
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": "NOT_AUTHORIZED",
    "RND6S": "NOT_AUTHORIZED",
    "RND6Q": "NOT_AUTHORIZED",
    "RND6T": "NOT_AUTHORIZED",
}
RND3_SCOPE = "RND3_COMPLETE_SEED0_PREDICTION_ONLY"
RND3_ACTION = "RUN_SINGLE_RND3_COMPLETE_SEED0_PREDICTION_ONLY_CONTROLLER"
RND3_DECISION = (
    "CLOSE_RND2_AND_AUTHORIZE_SINGLE_RND3_COMPLETE_SEED0_PREDICTION_ONLY_CONTROLLER"
)
RND3_AUTHORIZATION = {
    "scope": RND3_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": True,
    "smoke_allowed": False,
    "screen_allowed": True,
    "score_allowed": False,
    "qualification_allowed": False,
    "formal_confirmation_allowed": False,
    "new_external_outcome_access_allowed": False,
}
RND3_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": RND1_PASS,
    "RND2": RND2_MERGE_PASS,
    "RND3": "AUTHORIZED_COMPLETE_SEED0_PREDICTION_ONLY",
    "RND4": "NOT_AUTHORIZED",
    "RND5": "NOT_AUTHORIZED",
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": "NOT_AUTHORIZED",
    "RND6S": "NOT_AUTHORIZED",
    "RND6Q": "NOT_AUTHORIZED",
    "RND6T": "NOT_AUTHORIZED",
}
RND4_SCOPE = "RND4_COMPLETE_MERGE_SCORE_ONCE_ONLY"
RND4_ACTION = "RUN_SINGLE_RND4_COMPLETE_SCORE_ONCE"
RND4_DECISION = "CLOSE_RND3_AND_AUTHORIZE_SINGLE_RND4_COMPLETE_SCORE_ONCE"
RND4_AUTHORIZATION = {
    "scope": RND4_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": False,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": True,
    "qualification_allowed": False,
    "formal_confirmation_allowed": False,
    "new_external_outcome_access_allowed": False,
}
RND4_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": RND1_PASS,
    "RND2": RND2_MERGE_PASS,
    "RND3": RND3_MERGE_PASS,
    "RND4": "AUTHORIZED_COMPLETE_SCORE_ONCE",
    "RND5": "NOT_AUTHORIZED",
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": "NOT_AUTHORIZED",
    "RND6S": "NOT_AUTHORIZED",
    "RND6Q": "NOT_AUTHORIZED",
    "RND6T": "NOT_AUTHORIZED",
}
RND5_SCOPE = "RND5_COMPLETE_SCORE_QUALIFIER_ONCE_ONLY"
RND5_ACTION = "RUN_SINGLE_RND5_COMPLETE_QUALIFIER_ONCE"
RND5_DECISION = "CLOSE_RND4_AND_AUTHORIZE_SINGLE_RND5_COMPLETE_QUALIFIER_ONCE"
RND5_AUTHORIZATION = {
    "scope": RND5_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": False,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": False,
    "qualification_allowed": True,
    "formal_confirmation_allowed": False,
    "new_external_outcome_access_allowed": False,
}
RND5_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": RND1_PASS,
    "RND2": RND2_MERGE_PASS,
    "RND3": RND3_MERGE_PASS,
    "RND4": RND4_SCORE_PASS,
    "RND5": "AUTHORIZED_COMPLETE_QUALIFIER_ONCE",
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": "NOT_AUTHORIZED",
    "RND6S": "NOT_AUTHORIZED",
    "RND6Q": "NOT_AUTHORIZED",
    "RND6T": "NOT_AUTHORIZED",
}

RND5T_SCOPE = "RND5T_SCREEN_TERMINAL_CLOSED"
RND5T_ACTION = "STOP_RND5_SCREEN_NONPASS_ALL_RUNTIME_RIGHTS_CLOSED"
RND5T_DECISION = "CLOSE_RND5_AND_RECORD_SCREEN_NONPASS_WITHOUT_RND6"
RND5T_AUTHORIZATION = {
    "scope": RND5T_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": False,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": False,
    "qualification_allowed": False,
    "formal_confirmation_allowed": False,
    "new_external_outcome_access_allowed": False,
}

RND6P_SCOPE = "RND6P_FIXED_SEEDS_0_TO_4_FORMAL_GPU_PREDICTION_ONLY"
RND6P_ACTION = "RUN_SINGLE_RND6P_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY_CONTROLLER"
RND6P_DECISION = (
    "CLOSE_RND5_AND_AUTHORIZE_SINGLE_RND6P_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY_CONTROLLER"
)
RND6P_AUTHORIZATION = {
    "scope": RND6P_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": True,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": False,
    "qualification_allowed": False,
    "formal_confirmation_allowed": True,
    "new_external_outcome_access_allowed": False,
}
RND6P_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": RND1_PASS,
    "RND2": RND2_MERGE_PASS,
    "RND3": RND3_MERGE_PASS,
    "RND4": RND4_SCORE_PASS,
    "RND5": RND5_SCREEN_PASS,
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": "AUTHORIZED_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY",
    "RND6S": "NOT_AUTHORIZED",
    "RND6Q": "NOT_AUTHORIZED",
    "RND6T": "NOT_AUTHORIZED",
}
RND6S_SCOPE = "RND6S_COMPLETE_FORMAL_SCORE_ONCE_ONLY"
RND6S_ACTION = "RUN_SINGLE_RND6S_COMPLETE_FORMAL_SCORE_ONCE"
RND6S_DECISION = "CLOSE_RND6P_AND_AUTHORIZE_SINGLE_RND6S_COMPLETE_FORMAL_SCORE_ONCE"
RND6S_AUTHORIZATION = {
    "scope": RND6S_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": False,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": True,
    "qualification_allowed": False,
    "formal_confirmation_allowed": True,
    "new_external_outcome_access_allowed": False,
}
RND6S_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": RND1_PASS,
    "RND2": RND2_MERGE_PASS,
    "RND3": RND3_MERGE_PASS,
    "RND4": RND4_SCORE_PASS,
    "RND5": RND5_SCREEN_PASS,
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": RND6_ASSEMBLY_PASS,
    "RND6S": "AUTHORIZED_COMPLETE_FORMAL_SCORE_ONCE",
    "RND6Q": "NOT_AUTHORIZED",
    "RND6T": "NOT_AUTHORIZED",
}
RND6Q_SCOPE = "RND6Q_COMPLETE_FORMAL_QUALIFIER_ONCE_ONLY"
RND6Q_ACTION = "RUN_SINGLE_RND6Q_COMPLETE_FORMAL_QUALIFIER_ONCE"
RND6Q_DECISION = (
    "CLOSE_RND6S_AND_AUTHORIZE_SINGLE_RND6Q_COMPLETE_FORMAL_QUALIFIER_ONCE"
)
RND6Q_AUTHORIZATION = {
    "scope": RND6Q_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": False,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": False,
    "qualification_allowed": True,
    "formal_confirmation_allowed": True,
    "new_external_outcome_access_allowed": False,
}
RND6Q_GATE_STATE = {
    "RND0": "RND0_SOURCE_AND_IMPLEMENTATION_EXACT_PASS",
    "RND1": RND1_PASS,
    "RND2": RND2_MERGE_PASS,
    "RND3": RND3_MERGE_PASS,
    "RND4": RND4_SCORE_PASS,
    "RND5": RND5_SCREEN_PASS,
    "RND5T": "NOT_AUTHORIZED",
    "RND6P": RND6_ASSEMBLY_PASS,
    "RND6S": RND6_SCORE_PASS,
    "RND6Q": "AUTHORIZED_COMPLETE_FORMAL_QUALIFIER_ONCE",
    "RND6T": "NOT_AUTHORIZED",
}
RND6T_SCOPE = "RND6T_FORMAL_TERMINAL_CLOSED"
RND6T_ACTION = "STOP_RND6_FORMAL_ALL_RUNTIME_RIGHTS_CLOSED"
RND6T_DECISION = "CLOSE_RND6Q_AND_RECORD_RND6_FORMAL_TERMINAL_VERDICT"
RND6T_AUTHORIZATION = {
    "scope": RND6T_SCOPE,
    "implementation_allowed": False,
    "neural_training_allowed": False,
    "smoke_allowed": False,
    "screen_allowed": False,
    "score_allowed": False,
    "qualification_allowed": False,
    "formal_confirmation_allowed": False,
    "new_external_outcome_access_allowed": False,
}

RND1_PRETRAIN_DIR = ARTIFACT_ROOT / "rnd1_pretrain"
RND2_PREDICTION_DIR = ARTIFACT_ROOT / "rnd2_smoke_seed0"
RND3_PREDICTION_DIR = ARTIFACT_ROOT / "rnd3_screen_seed0"
RND2_MERGED_PATH = RND2_PREDICTION_DIR / "rnet_distill_complete_unscored_merge.json"
RND3_MERGED_PATH = RND3_PREDICTION_DIR / "rnet_distill_complete_unscored_merge.json"
RND4_SCORE_PATH = RND3_PREDICTION_DIR / "rnet_distill_complete_score.json"
RND5_QUALIFICATION_PATH = RND3_PREDICTION_DIR / "rnet_distill_qualification.json"
RND6_FORMAL_DIR = ARTIFACT_ROOT / "rnd6_formal_seeds0_4"
RND6_MERGED_PATH = RND6_FORMAL_DIR / "rnet_distill_complete_unscored_merge.json"
RND6_ASSEMBLY_DIR = RND6_FORMAL_DIR / "assembled"
RND6_ASSEMBLY_MANIFEST_PATH = (
    RND6_ASSEMBLY_DIR / "rnet_distill_five_seed_prediction_only_assembly.json"
)
RND6_SCORE_PATH = RND6_FORMAL_DIR / "rnet_distill_complete_formal_score.json"
RND6_QUALIFICATION_PATH = RND6_FORMAL_DIR / "rnet_distill_formal_qualification.json"
SCREEN_REPORT_PATH = Path(
    "docs/prospective_v2/independent_rnet_distill_screen_result.md"
)
FORMAL_REPORT_PATH = Path(
    "docs/prospective_v2/independent_rnet_distill_formal_result.md"
)
FINALIZER_PATH = Path(
    "scripts/reactflow_delta/finalize_independent_rnet_distill_result.py"
)
RESULT_REGISTRY_LOCATION = (
    "docs/prospective_v2/independent_rnet_distill_decision_ledger.yaml#result_registry"
)
FORMAL_PENDING_STATUS = "PENDING_RND6_FORMAL"
FORMAL_NOT_RUN_STATUS = "NOT_RUN_RND5_NONPASS"
RESULT_EXPERIMENT_IDS = {
    "screen": "RND3_RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY",
    "formal": "RND6P_RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY",
}
M2_PATH = Path(
    "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/"
    "reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
)
TIC2A_PATH = Path(
    "/mnt/cunyuliu/reactflow_delta_target_identity_correction/"
    "tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json"
)
UNCONSTRAINED_CACHE_PATH = Path(
    "/mnt/cunyuliu/reactflow_delta_model_rescue_v5/"
    "v5m1_full/ensemble_delta_cache.h5"
)
CONSTRAINED_CACHE_PATH = Path(
    "/mnt/cunyuliu/reactflow_delta_model_rescue_v6/"
    "v6m1_full/constrained_cache.h5"
)
HISTORICAL_V8_DIR = Path(
    "/mnt/cunyuliu/reactflow_delta_model_rescue_v8/"
    "v8m1_corrected_experts_seed0"
)
HISTORICAL_V10_DIR = Path(
    "/mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0"
)
HISTORICAL_V14_SCORE_PATH = Path(
    "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
    "v14m3_screen_seed0/v14m3_complete_score.json"
)
RND3_AUTHORITY_PATHS = {
    "pretraining_dir": RND1_PRETRAIN_DIR,
    "screen_prediction_dir": RND3_PREDICTION_DIR,
    "m2_csv_path": M2_PATH,
    "tic2a_merged_registry_path": TIC2A_PATH,
    "unconstrained_feature_cache_path": UNCONSTRAINED_CACHE_PATH,
    "constrained_feature_cache_path": CONSTRAINED_CACHE_PATH,
    "historical_v8_dir": HISTORICAL_V8_DIR,
    "historical_v10_dir": HISTORICAL_V10_DIR,
}
RND4_AUTHORITY_PATHS = {
    "screen_prediction_dir": RND3_PREDICTION_DIR,
    "complete_unscored_merge_path": RND3_MERGED_PATH,
    "m2_csv_path": M2_PATH,
    "historical_v14_score_path": HISTORICAL_V14_SCORE_PATH,
    "complete_score_path": RND4_SCORE_PATH,
    "qualification_path": RND5_QUALIFICATION_PATH,
}
RND5_AUTHORITY_PATHS = {
    "screen_prediction_dir": RND3_PREDICTION_DIR,
    "complete_unscored_merge_path": RND3_MERGED_PATH,
    "complete_score_path": RND4_SCORE_PATH,
    "qualification_path": RND5_QUALIFICATION_PATH,
}
RND5T_AUTHORITY_PATHS = RND5_AUTHORITY_PATHS
RND6_CANONICAL_PATHS = {
    "formal_prediction_dir": RND6_FORMAL_DIR,
    "formal_complete_unscored_merge_path": RND6_MERGED_PATH,
    "formal_assembly_dir": RND6_ASSEMBLY_DIR,
    "formal_assembly_path": RND6_ASSEMBLY_MANIFEST_PATH,
    "formal_complete_score_path": RND6_SCORE_PATH,
    "formal_qualification_path": RND6_QUALIFICATION_PATH,
    "screen_qualification_path": RND5_QUALIFICATION_PATH,
}
RND6_CANONICAL_STATUSES = {
    "screen_pass": RND5_SCREEN_PASS,
    "formal_merge_pass": RND6_MERGE_PASS,
    "formal_assembly_pass": RND6_ASSEMBLY_PASS,
    "formal_score_pass": RND6_SCORE_PASS,
    "formal_qualification_pass": RND6_QUALIFICATION_PASS,
    "formal_qualification_fail": RND6_QUALIFICATION_FAIL,
    "formal_qualification_indeterminate": RND6_QUALIFICATION_INDETERMINATE,
}
FORMAL_GATES = {
    "screen_prerequisite_status": RND5_SCREEN_PASS,
    "equal_seed_mixture_required": True,
    "mixture_must_pass_frozen_screen_gates": True,
    "individual_seed_positive_vs_matched_null_minimum": {
        "signed_delta": 4,
        "point_absolute": 4,
        "task_crps": 4,
        "distribution_absolute": 4,
    },
    "strict_positive_mean_gain_required": True,
    "best_seed_selection_allowed": False,
    "extra_seed_selection_allowed": False,
    "evidence_ceiling": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
}
ACTIVE_RESULT_FINALIZATION = {
    "production_entry": str(FINALIZER_PATH),
    "screen_report_path": str(SCREEN_REPORT_PATH),
    "formal_report_path": str(FORMAL_REPORT_PATH),
    "result_registry_location": RESULT_REGISTRY_LOCATION,
    "overwrite_allowed": False,
    "evidence_ceiling": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
    "clean_ood": "NOT_ESTABLISHED",
    "external_replication": "NOT_ESTABLISHED",
    "sota": "NOT_ESTABLISHED",
    "publication_ready": False,
}
CONTRACT_RESULT_FINALIZATION = {
    "production_entry": str(FINALIZER_PATH),
    "screen_report_path": str(SCREEN_REPORT_PATH),
    "formal_report_path": str(FORMAL_REPORT_PATH),
    "result_registry_location": RESULT_REGISTRY_LOCATION,
    "rnd5_pass_next_phase": "RND6P",
    "rnd5_nonpass_terminal_phase": "RND5T",
    "rnd6_qualification_terminal_phase": "RND6T",
    "rnd5_nonpass_formal_registry_status": FORMAL_NOT_RUN_STATUS,
    "overwrite_allowed": False,
    "evidence_ceiling": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
    "clean_ood": "NOT_ESTABLISHED",
    "external_replication": "NOT_ESTABLISHED",
    "sota": "NOT_ESTABLISHED",
    "publication_ready": False,
}
FORMAL_OUTPUT_STATE_BY_PHASE = {
    "RND6P": {
        "complete_unscored_merge_exists": False,
        "equal_seed_assembly_exists": False,
        "complete_formal_score_exists": False,
        "formal_qualification_exists": False,
    },
    "RND6S": {
        "complete_unscored_merge_exists": True,
        "equal_seed_assembly_exists": True,
        "complete_formal_score_exists": False,
        "formal_qualification_exists": False,
    },
    "RND6Q": {
        "complete_unscored_merge_exists": True,
        "equal_seed_assembly_exists": True,
        "complete_formal_score_exists": True,
        "formal_qualification_exists": False,
    },
    "RND6T": {
        "complete_unscored_merge_exists": True,
        "equal_seed_assembly_exists": True,
        "complete_formal_score_exists": True,
        "formal_qualification_exists": True,
    },
}
INACTIVE_FORMAL_OUTPUT_STATE = FORMAL_OUTPUT_STATE_BY_PHASE["RND6P"]


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a mapping at {path}")
    return value


def _load_research_frontmatter(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise RuntimeError("research record frontmatter is missing")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise RuntimeError("research record frontmatter is unterminated") from exc
    value = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(value, dict):
        raise RuntimeError("research record frontmatter must be a mapping")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _check_frozen_scientific_contract(contract: dict[str, Any]) -> None:
    source = contract["source_binding"]
    _require(source["expected_record_count"] == 208905, "teacher record count changed")
    _require(source["expected_shard_count"] == 409, "teacher shard count changed")
    _require(source["expected_single_feature_dim"] == 384, "teacher width changed")
    _require(source["license"] == "MIT", "teacher license declaration changed")
    _require(
        source["overlap_audit_interpretation"]
        == "EXACT_SEQUENCE_ONLY_NEAR_NEIGHBOR_EXPOSURE_NOT_EXCLUDED",
        "teacher exposure ceiling changed",
    )
    attribution = contract["attribution"]
    _require(
        attribution["candidate_teacher_target"]
        == "ALIGNED_PER_NUCLEOTIDE_SINGLE_FEATURE",
        "candidate target changed",
    )
    _require(
        attribution["null_teacher_target"]
        == "CYCLIC_SHIFT_MIN_17_L_MINUS_1_WITHIN_EACH_SEQUENCE",
        "matched null changed",
    )
    _require(
        attribution["null_target_index_mapping"]
        == "TEACHER_AT_I_MINUS_MIN_17_L_MINUS_1_MOD_L",
        "matched-null direction changed",
    )
    for key in (
        "same_initial_state",
        "same_architecture",
        "same_record_universe",
        "same_shard_and_batch_order",
        "same_dropout_random_stream",
        "same_optimizer_and_schedule",
        "same_downstream_training_and_calibration",
    ):
        _require(attribution[key] == "exact", f"attribution equality changed: {key}")
    _require(attribution["teacher_pair_feature_use_allowed"] is False, "pair teacher reopened")
    _require(attribution["live_ribonanzanet_inference_allowed"] is False, "live teacher reopened")
    schedule = contract["frozen_schedule"]
    _require(schedule["distillation_seed"] == 20260828, "distillation seed changed")
    _require(schedule["data_order_seed"] == 20260828, "data order seed changed")
    _require(schedule["distillation_epochs"] == 1, "distillation epoch count changed")
    _require(schedule["batch_size"] == 16, "distillation batch size changed")
    _require(schedule["rnd2_engineering_smoke"] == {
        "seed": 0,
        "folds": [0, 1],
        "point_epochs": 3,
        "calibration_epochs": 3,
        "scientific_score_allowed": False,
    }, "RND2 schedule changed")
    _require(
        schedule["rnd3_complete_screen"]["folds"] == list(range(20)),
        "RND3 fold universe changed",
    )
    _require(schedule["rnd3_complete_screen"]["seed"] == 0, "RND3 seed changed")
    expected_formal_schedule = {
        "lifecycle_status": FORMAL_INACTIVE_STATUS,
        "folds": FORMAL_FOLDS,
        "seeds": FORMAL_SEEDS,
        "expected_fold_seed_pairs": FORMAL_PAIR_COUNT,
        "point_epochs": FORMAL_POINT_EPOCHS,
        "calibration_epochs": FORMAL_CALIBRATION_EPOCHS,
        "equal_seed_mixture": True,
        "equal_seed_weight": FORMAL_EQUAL_SEED_WEIGHT,
        "best_seed_selection_allowed": False,
        "extra_seed_selection_allowed": False,
        "complete_prediction_universe_before_score": True,
        "authorized_only_after_exact_rnd5_pass": True,
    }
    _require(
        schedule["rnd6_formal"] == expected_formal_schedule,
        "formal schedule changed",
    )
    gates = contract["screen_gates"]
    _require(gates["gate_lowering_after_score_access_allowed"] is False, "Gate lowering opened")
    _require(gates["extra_seed_selection_allowed"] is False, "seed selection opened")
    _require(contract["formal_gates"] == FORMAL_GATES, "formal Gates changed")
    outcome = contract["outcome_policy"]
    for key in (
        "pretraining_may_read_openknot_mutant_outcome",
        "prediction_runner_may_read_held_target",
        "partial_fold_score_allowed",
        "new_external_outcome_access_allowed",
    ):
        _require(outcome[key] is False, f"outcome boundary widened: {key}")
    expected_formal_outcome_policy = {
        "formal_prediction_runner_may_read_held_target": False,
        "formal_partial_fold_score_allowed": False,
        "formal_complete_score_allowed_only_in_RND6S": True,
        "formal_qualification_allowed_only_in_RND6Q": True,
        "formal_external_outcome_access_allowed": False,
    }
    _require(
        {key: outcome.get(key) for key in expected_formal_outcome_policy}
        == expected_formal_outcome_policy,
        "formal outcome boundary changed",
    )
    artifact_policy = contract["artifact_policy"]
    _require(artifact_policy["overwrite_allowed"] is False, "artifact overwrite opened")
    _require(
        artifact_policy["formal_existing_output_rerun_allowed"] is False,
        "formal output rerun opened",
    )
    _require(
        artifact_policy["one_canonical_output_per_phase"] is True,
        "canonical formal output uniqueness changed",
    )
    _require(
        contract.get("result_finalization") == CONTRACT_RESULT_FINALIZATION,
        "result-finalization contract changed",
    )
    gpu = contract["gpu_policy"]
    _require(gpu["training_and_gpu_validation_device_class"] == "CUDA_ONLY", "CUDA-only changed")
    _require(gpu["cpu_model_or_loss_fallback_allowed"] is False, "CPU fallback opened")
    _require(gpu["minimum_free_vram_gate_allowed"] is False, "VRAM gate opened")
    expected_predecessors = {
        "RND3": RND2_MERGE_PASS,
        "RND4": RND3_MERGE_PASS,
        "RND5": RND4_SCORE_PASS,
        "RND6P": RND5_SCREEN_PASS,
        "RND6Q": RND6_SCORE_PASS,
    }
    for phase, expected in expected_predecessors.items():
        _require(
            contract["phase_contract"][phase]["required_predecessor"] == expected,
            f"{phase} predecessor diverged from canonical status {expected}",
        )
    _require(
        contract["phase_contract"]["RND6S"]["required_predecessors"]
        == [RND6_MERGE_PASS, RND6_ASSEMBLY_PASS],
        "RND6S predecessors diverged from canonical statuses",
    )
    _require(
        contract["phase_contract"]["RND5T"]["required_predecessor_one_of"]
        == list(RND5_NONPASS_STATUSES),
        "RND5T predecessor statuses changed",
    )
    _require(
        contract["phase_contract"]["RND6T"]["required_predecessor_one_of"]
        == list(RND6_QUALIFICATION_STATUSES),
        "RND6T predecessor statuses changed",
    )
    expected_formal_contracts = {
        "RND5T": {
            "authority_token": TOKENS["RND5T"],
            "required_predecessor_one_of": list(RND5_NONPASS_STATUSES),
            "next_action": RND5T_ACTION,
            "runnable": False,
        },
        "RND6P": {
            "authority_token": TOKENS["RND6P"],
            "required_predecessor": RND5_SCREEN_PASS,
            "next_action": RND6P_ACTION,
            "outputs": [
                "exact_100_fold_seed_prediction_sets",
                "complete_unscored_merge",
                "equal_seed_prediction_only_assembly",
            ],
            "scientific_conclusion_allowed": False,
        },
        "RND6S": {
            "authority_token": TOKENS["RND6S"],
            "required_predecessors": [RND6_MERGE_PASS, RND6_ASSEMBLY_PASS],
            "next_action": RND6S_ACTION,
            "outputs": ["complete_formal_score"],
            "scientific_conclusion_allowed": False,
        },
        "RND6Q": {
            "authority_token": TOKENS["RND6Q"],
            "required_predecessor": RND6_SCORE_PASS,
            "next_action": RND6Q_ACTION,
            "outputs": ["formal_qualification"],
            "scientific_conclusion_allowed": True,
        },
        "RND6T": {
            "authority_token": TOKENS["RND6T"],
            "required_predecessor_one_of": list(RND6_QUALIFICATION_STATUSES),
            "next_action": RND6T_ACTION,
            "runnable": False,
        },
    }
    for phase, expected in expected_formal_contracts.items():
        _require(
            contract["phase_contract"][phase] == expected,
            f"{phase} contract changed",
        )


def _formal_lifecycle(phase: str) -> tuple[str, bool]:
    if phase == "RND5T":
        return FORMAL_NOT_RUN_RND5_NONPASS_STATUS, False
    if phase not in FORMAL_PHASES:
        return FORMAL_INACTIVE_STATUS, False
    return FORMAL_LIFECYCLE_BY_PHASE[phase], phase != "RND6T"


def _expected_contract_formal_chain(phase: str) -> dict[str, Any]:
    lifecycle_status, activation_allowed = _formal_lifecycle(phase)
    return {
        "lifecycle_status": lifecycle_status,
        "activation_allowed": activation_allowed,
        "activation_predecessor": RND5_SCREEN_PASS,
        "phase_order": FORMAL_PHASE_ORDER,
        "prediction_score_qualification_permissions_mutually_exclusive": True,
        "evidence_ceiling": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
        "canonical_paths": {
            key: str(path) for key, path in RND6_CANONICAL_PATHS.items()
        },
        "canonical_statuses": RND6_CANONICAL_STATUSES,
    }


def _expected_active_formal_chain(phase: str) -> dict[str, Any]:
    lifecycle_status, activation_allowed = _formal_lifecycle(phase)
    return {
        "lifecycle_status": lifecycle_status,
        "activation_allowed": activation_allowed,
        "activation_predecessor": RND5_SCREEN_PASS,
        "phase_order": FORMAL_PHASE_ORDER,
        "prediction_score_qualification_permissions_mutually_exclusive": True,
        "folds": FORMAL_FOLDS,
        "seeds": FORMAL_SEEDS,
        "expected_fold_seed_pairs": FORMAL_PAIR_COUNT,
        "point_epochs": FORMAL_POINT_EPOCHS,
        "calibration_epochs": FORMAL_CALIBRATION_EPOCHS,
        "equal_seed_weight": FORMAL_EQUAL_SEED_WEIGHT,
        "canonical_paths": {
            key: str(path) for key, path in RND6_CANONICAL_PATHS.items()
        },
        "canonical_statuses": RND6_CANONICAL_STATUSES,
        "formal_gates": FORMAL_GATES,
    }


def _check_formal_design_state(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
    research: dict[str, Any],
    *,
    phase: str,
) -> None:
    lifecycle_status, activation_allowed = _formal_lifecycle(phase)
    _require(
        contract["formal_chain"] == _expected_contract_formal_chain(phase),
        "machine formal-chain state changed",
    )
    _require(
        active["inactive_formal_chain"] == _expected_active_formal_chain(phase),
        "active formal-chain state changed",
    )
    _require(
        ledger.get("formal_chain_status") == lifecycle_status,
        "ledger formal-chain lifecycle changed",
    )
    _require(
        research.get("formal_chain_status") == lifecycle_status,
        "research formal-chain lifecycle changed",
    )
    _require(
        research.get("formal_phase_order") == FORMAL_PHASE_ORDER,
        "research formal phase order changed",
    )
    _require(
        research.get("formal_activation_allowed") is activation_allowed,
        "research formal activation state changed",
    )
    expected_access = {
        "formal_score_accessed": phase in {"RND6Q", "RND6T"},
        "formal_qualification_accessed": phase == "RND6T",
    }
    for key, expected in expected_access.items():
        _require(ledger.get(key) is expected, f"ledger {key} changed")
        _require(research.get(key) is expected, f"research {key} changed")
    expected_outputs = (
        FORMAL_OUTPUT_STATE_BY_PHASE[phase]
        if phase in FORMAL_PHASES
        else INACTIVE_FORMAL_OUTPUT_STATE
    )
    _require(
        active.get("formal_output_state") == expected_outputs,
        "formal output-existence state changed",
    )
    decisions = ledger.get("decisions")
    _require(isinstance(decisions, list), "formal design decision ledger is missing")
    design_events = [
        event
        for event in decisions
        if isinstance(event, dict)
        and event.get("event") == "RND6_FORMAL_CHAIN_FROZEN_INACTIVE"
    ]
    _require(len(design_events) == 1, "exactly one frozen formal design event required")
    design = design_events[0]
    expected_design = {
        "decision": "FREEZE_FOUR_MUTUALLY_EXCLUSIVE_RND6_PHASES_PENDING_UNKNOWN_RND5_RESULT",
        "lifecycle_status": FORMAL_INACTIVE_STATUS,
        "activation_allowed": False,
        "activation_predecessor": RND5_SCREEN_PASS,
        "phase_order": FORMAL_PHASE_ORDER,
        "prediction_score_qualification_permissions_mutually_exclusive": True,
        "folds": FORMAL_FOLDS,
        "seeds": FORMAL_SEEDS,
        "expected_fold_seed_pairs": FORMAL_PAIR_COUNT,
        "point_epochs": FORMAL_POINT_EPOCHS,
        "calibration_epochs": FORMAL_CALIBRATION_EPOCHS,
        "equal_seed_weight": FORMAL_EQUAL_SEED_WEIGHT,
        "best_seed_selection_allowed": False,
        "extra_seed_selection_allowed": False,
        "formal_score_accessed": False,
        "formal_qualification_accessed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "canonical_paths": {
            key: str(path) for key, path in RND6_CANONICAL_PATHS.items()
        },
        "canonical_statuses": RND6_CANONICAL_STATUSES,
        "formal_gates": FORMAL_GATES,
        "current_rnd1_phase_unchanged": True,
        "current_rnd1_permissions_unchanged": True,
    }
    observed_design = {key: design.get(key) for key in expected_design}
    _require(observed_design == expected_design, "frozen formal design event changed")
    if phase not in FORMAL_PHASES:
        for formal_phase in FORMAL_PHASES:
            _require(
                active["gate_state"].get(formal_phase) == "NOT_AUTHORIZED",
                f"{formal_phase} activated before exact RND5 PASS",
            )


_SCREEN_REGISTRY_FIELDS = {
    "phase",
    "status",
    "experiment_id",
    "authority_branch",
    "recorded_at",
    "report_path",
    "report_exists",
    "canonical_merge_path",
    "canonical_score_path",
    "canonical_qualification_path",
    "folds",
    "seeds",
    "point_epochs",
    "calibration_epochs",
    "training_devices",
    "gpu_names",
    "started_at_utc",
    "finished_at_utc",
    "source_commits",
    "gate_passed",
    "integrity_passed",
    "failed_gates",
    "integrity_errors",
    "evidence_status",
    "clean_ood",
    "external_replication",
    "sota",
    "publication_ready",
    "finalizer_source_commit",
}
_FORMAL_REGISTRY_FIELDS = {
    *_SCREEN_REGISTRY_FIELDS,
    "canonical_assembly_path",
    "expected_fold_seed_pairs",
    "equal_seed_weight",
}
_PENDING_FORMAL_REGISTRY = {
    "status": FORMAL_PENDING_STATUS,
    "reason": "RND5_EXACT_PASS_AUTHORIZED_RND6P",
    "report_path": str(FORMAL_REPORT_PATH),
    "report_exists": False,
    "publication_ready": False,
}


def _check_registry_provenance(
    entry: dict[str, Any],
    *,
    label: str,
    expected_fields: set[str],
    expected_folds: list[int],
    expected_seeds: list[int],
    expected_experiment_id: str,
) -> None:
    _require(set(entry) == expected_fields, f"{label} result-registry fields changed")
    _require(entry["folds"] == expected_folds, f"{label} fold universe changed")
    _require(entry["seeds"] == expected_seeds, f"{label} seed universe changed")
    _require(
        entry["experiment_id"] == expected_experiment_id,
        f"{label} experiment id changed",
    )
    _require(entry["authority_branch"] == BRANCH, f"{label} authority branch changed")
    _require(
        entry["evidence_status"] == "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
        f"{label} evidence ceiling changed",
    )
    for key in ("clean_ood", "external_replication", "sota"):
        _require(entry[key] == "NOT_ESTABLISHED", f"{label} {key} claim widened")
    _require(entry["publication_ready"] is False, f"{label} publication claim widened")
    _require(
        isinstance(entry["recorded_at"], str) and bool(entry["recorded_at"]),
        f"{label} recorded time is missing",
    )
    _require(
        isinstance(entry["started_at_utc"], str) and bool(entry["started_at_utc"]),
        f"{label} start time is missing",
    )
    _require(
        isinstance(entry["finished_at_utc"], str) and bool(entry["finished_at_utc"]),
        f"{label} finish time is missing",
    )
    commits = entry["source_commits"]
    _require(
        isinstance(commits, list)
        and bool(commits)
        and len(commits) == 1
        and commits == sorted(set(commits))
        and all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{40}", value) is not None
            for value in commits
        ),
        f"{label} source commits are invalid",
    )
    finalizer_commit = entry["finalizer_source_commit"]
    _require(
        isinstance(finalizer_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", finalizer_commit) is not None,
        f"{label} finalizer source commit is invalid",
    )
    devices = entry["training_devices"]
    _require(
        devices == ["cuda:0"],
        f"{label} CUDA provenance changed",
    )
    gpu_names = entry["gpu_names"]
    _require(
        isinstance(gpu_names, list)
        and bool(gpu_names)
        and gpu_names == sorted(set(gpu_names))
        and all(isinstance(value, str) and bool(value) for value in gpu_names),
        f"{label} GPU names are missing",
    )
    for name in ("failed_gates", "integrity_errors"):
        value = entry[name]
        _require(
            isinstance(value, list)
            and value == sorted(set(value))
            and all(isinstance(item, str) and bool(item) for item in value),
            f"{label} {name} changed",
        )


def _check_result_report(
    repo_root: Path, entry: dict[str, Any], *, label: str
) -> None:
    report_path = repo_root / Path(entry["report_path"])
    _require(report_path.is_file(), f"{label} canonical report file is missing")
    text = report_path.read_text(encoding="utf-8")
    controlled_lines = {
        "- Qualification status:": f"- Qualification status: `{entry['status']}`",
        "- Evidence ceiling:": (
            "- Evidence ceiling: `EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY`"
        ),
        "- Publication ready:": "- Publication ready: `false`",
        "- Experiment ID:": f"- Experiment ID: `{entry['experiment_id']}`",
        "- Authority branch:": f"- Authority branch: `{BRANCH}`",
        "- Clean out-of-distribution evidence": (
            "- Clean out-of-distribution evidence is not established."
        ),
        "- Independent external replication": (
            "- Independent external replication is not established."
        ),
        "- State of the art": "- State of the art is not established.",
        "- Publication readiness": "- Publication readiness is false.",
        "## Canonical calibration": "## Canonical calibration",
        "- Exact per-fold runner commands:": (
            "- Exact per-fold runner commands: recorded in the canonical merge "
            f"`{entry['canonical_merge_path']}` under `folds[*].command`; "
            "not duplicated into the decision ledger."
        ),
    }
    lines = text.splitlines()
    for prefix, expected in controlled_lines.items():
        matches = [line for line in lines if line.startswith(prefix)]
        _require(
            matches == [expected],
            f"{label} canonical report binding changed: {prefix}",
        )


def _check_result_registry(
    repo_root: Path,
    active: dict[str, Any],
    ledger: dict[str, Any],
    research: dict[str, Any],
    *,
    phase: str,
) -> None:
    _require(
        active.get("result_finalization") == ACTIVE_RESULT_FINALIZATION,
        "active result-finalization binding changed",
    )
    _require(research.get("publication_ready") is False, "research publication claim widened")
    registry = ledger.get("result_registry")
    _require(isinstance(registry, dict), "canonical result registry is missing")
    if phase in {"RND0", "RND1", "RND2", "RND3", "RND4", "RND5"}:
        _require(registry == {}, "result registry populated before RND5 finalization")
        _require(
            research.get("screen_result_status") == "NOT_FINALIZED"
            and research.get("formal_result_status") == "NOT_FINALIZED",
            "research result state advanced before finalization",
        )
        _require(
            not (repo_root / SCREEN_REPORT_PATH).exists()
            and not (repo_root / FORMAL_REPORT_PATH).exists(),
            "canonical result report exists before RND5 finalization",
        )
        return

    _require(set(registry) == {"screen", "formal"}, "result registry sections changed")
    screen = registry["screen"]
    _require(isinstance(screen, dict), "screen result registry is malformed")
    _check_registry_provenance(
        screen,
        label="screen",
        expected_fields=_SCREEN_REGISTRY_FIELDS,
        expected_folds=FORMAL_FOLDS,
        expected_seeds=[0],
        expected_experiment_id=RESULT_EXPERIMENT_IDS["screen"],
    )
    _require(screen["phase"] == "RND5", "screen registry phase changed")
    _require(screen["report_path"] == str(SCREEN_REPORT_PATH), "screen report path changed")
    _require(screen["report_exists"] is True, "screen canonical report is not recorded")
    _require(
        screen["canonical_merge_path"] == str(RND3_MERGED_PATH)
        and screen["canonical_score_path"] == str(RND4_SCORE_PATH)
        and screen["canonical_qualification_path"] == str(RND5_QUALIFICATION_PATH),
        "screen registry canonical paths changed",
    )
    _require(
        screen["point_epochs"] == FORMAL_POINT_EPOCHS
        and screen["calibration_epochs"] == FORMAL_CALIBRATION_EPOCHS,
        "screen registry schedule changed",
    )
    status = screen["status"]
    _require(
        status in (RND5_SCREEN_PASS, *RND5_NONPASS_STATUSES),
        "screen registry status is not canonical",
    )
    expected_screen_semantics = {
        RND5_SCREEN_PASS: (True, True),
        RND5_SCREEN_FAIL: (False, True),
        RND5_SCREEN_INDETERMINATE: (False, False),
    }[status]
    _require(
        screen["gate_passed"] is expected_screen_semantics[0]
        and screen["integrity_passed"] is expected_screen_semantics[1],
        "screen registry verdict semantics changed",
    )
    expected_screen_reasons = {
        RND5_SCREEN_PASS: (False, False),
        RND5_SCREEN_FAIL: (True, False),
        RND5_SCREEN_INDETERMINATE: (False, True),
    }[status]
    _require(
        (bool(screen["failed_gates"]), bool(screen["integrity_errors"]))
        == expected_screen_reasons,
        "screen registry reason semantics changed",
    )
    _require(
        research.get("screen_result_status") == status,
        "research screen result diverged from registry",
    )
    _check_result_report(repo_root, screen, label="screen")

    formal = registry["formal"]
    _require(isinstance(formal, dict), "formal result registry is malformed")
    if phase == "RND5T":
        _require(status in RND5_NONPASS_STATUSES, "RND5T requires screen nonpass")
        decisions = ledger.get("decisions")
        _require(
            isinstance(decisions, list)
            and bool(decisions)
            and isinstance(decisions[-1], dict)
            and decisions[-1].get("canonical_qualification_status") == status
            and active.get("gate_state", {}).get("RND5") == status,
            "RND5T registry status diverged from terminal authority",
        )
        expected_formal = {
            "status": FORMAL_NOT_RUN_STATUS,
            "reason": status,
            "report_path": str(FORMAL_REPORT_PATH),
            "report_exists": False,
            "publication_ready": False,
        }
        _require(formal == expected_formal, "RND5T formal-not-run registry changed")
        _require(
            research.get("formal_result_status") == FORMAL_NOT_RUN_STATUS,
            "research formal-not-run status changed",
        )
        _require(
            not (repo_root / FORMAL_REPORT_PATH).exists(),
            "formal report exists even though RND6 was not run",
        )
        return

    _require(status == RND5_SCREEN_PASS, f"{phase} requires exact screen PASS registry")
    if phase in {"RND6P", "RND6S", "RND6Q"}:
        _require(formal == _PENDING_FORMAL_REGISTRY, "pending formal registry changed")
        _require(
            research.get("formal_result_status") == FORMAL_PENDING_STATUS,
            "research pending formal status changed",
        )
        _require(
            not (repo_root / FORMAL_REPORT_PATH).exists(),
            "formal report exists while formal result is pending",
        )
        return

    _require(phase == "RND6T", "unexpected result-registry phase")
    _check_registry_provenance(
        formal,
        label="formal",
        expected_fields=_FORMAL_REGISTRY_FIELDS,
        expected_folds=FORMAL_FOLDS,
        expected_seeds=FORMAL_SEEDS,
        expected_experiment_id=RESULT_EXPERIMENT_IDS["formal"],
    )
    _require(formal["phase"] == "RND6Q", "formal registry phase changed")
    _require(formal["report_path"] == str(FORMAL_REPORT_PATH), "formal report path changed")
    _require(formal["report_exists"] is True, "formal canonical report is not recorded")
    _require(
        formal["canonical_merge_path"] == str(RND6_MERGED_PATH)
        and formal["canonical_assembly_path"] == str(RND6_ASSEMBLY_MANIFEST_PATH)
        and formal["canonical_score_path"] == str(RND6_SCORE_PATH)
        and formal["canonical_qualification_path"] == str(RND6_QUALIFICATION_PATH),
        "formal registry canonical paths changed",
    )
    _require(
        formal["point_epochs"] == FORMAL_POINT_EPOCHS
        and formal["calibration_epochs"] == FORMAL_CALIBRATION_EPOCHS
        and formal["expected_fold_seed_pairs"] == FORMAL_PAIR_COUNT
        and formal["equal_seed_weight"] == FORMAL_EQUAL_SEED_WEIGHT,
        "formal registry schedule changed",
    )
    formal_status = formal["status"]
    _require(
        formal_status in RND6_QUALIFICATION_STATUSES,
        "formal registry status is not canonical",
    )
    decisions = ledger.get("decisions")
    _require(
        isinstance(decisions, list)
        and bool(decisions)
        and isinstance(decisions[-1], dict)
        and decisions[-1].get("canonical_formal_qualification_status")
        == formal_status
        and active.get("gate_state", {}).get("RND6Q") == formal_status,
        "RND6T registry status diverged from terminal authority",
    )
    expected_formal_semantics = {
        RND6_QUALIFICATION_PASS: (True, True),
        RND6_QUALIFICATION_FAIL: (False, True),
        RND6_QUALIFICATION_INDETERMINATE: (False, False),
    }[formal_status]
    _require(
        formal["gate_passed"] is expected_formal_semantics[0]
        and formal["integrity_passed"] is expected_formal_semantics[1],
        "formal registry verdict semantics changed",
    )
    expected_formal_reasons = {
        RND6_QUALIFICATION_PASS: (False, False),
        RND6_QUALIFICATION_FAIL: (True, False),
        RND6_QUALIFICATION_INDETERMINATE: (False, True),
    }[formal_status]
    _require(
        (bool(formal["failed_gates"]), bool(formal["integrity_errors"]))
        == expected_formal_reasons,
        "formal registry reason semantics changed",
    )
    _require(
        research.get("formal_result_status") == formal_status,
        "research formal result diverged from registry",
    )
    _check_result_report(repo_root, formal, label="formal")


def _check_rnd1_authority(active: dict[str, Any], ledger: dict[str, Any]) -> None:
    authorization = active["authorization"]
    observed_authorization = {
        key: authorization.get(key) for key in RND1_AUTHORIZATION
    }
    _require(
        observed_authorization == RND1_AUTHORIZATION,
        "RND1 authorization scope or permissions changed",
    )
    _require(active["runnable_phases"] == ["RND1"], "RND1 is not solely runnable")
    _require(active["training_allowed"] is True, "RND1 training is not open")
    _require(
        active["candidate_model_training_allowed"] is True,
        "RND1 candidate training is not open",
    )
    _require(active["held_score_read_allowed"] is False, "RND1 held score opened")
    _require(active["gate_state"] == RND1_GATE_STATE, "RND1 gate state changed")
    _require(active["next_allowed_action"] == RND1_ACTION, "RND1 action changed")
    _require(ledger["next_action"] == RND1_LEDGER_ACTION, "RND1 ledger action changed")


def _check_rnd2_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    authorization = active["authorization"]
    observed_authorization = {
        key: authorization.get(key) for key in RND2_AUTHORIZATION
    }
    _require(
        observed_authorization == RND2_AUTHORIZATION,
        "RND2 authorization scope or permissions changed",
    )
    _require(active["runnable_phases"] == ["RND2"], "RND2 is not solely runnable")
    _require(active["training_allowed"] is True, "RND2 training is not open")
    _require(
        active["candidate_model_training_allowed"] is True,
        "RND2 candidate training is not open",
    )
    _require(active["held_score_read_allowed"] is False, "RND2 held score opened")
    _require(
        active["partial_fold_score_read_allowed"] is False,
        "RND2 partial score opened",
    )
    _require(
        active["new_external_outcome_access_allowed"] is False,
        "RND2 external outcome opened",
    )
    _require(active["gate_state"] == RND2_GATE_STATE, "RND2 gate state changed")
    _require(active["next_allowed_action"] == RND2_ACTION, "RND2 action changed")
    _require(ledger["next_action"] == RND2_ACTION, "RND2 ledger action changed")
    _require(
        contract["phase_contract"]["RND2"]["required_predecessor"] == RND1_PASS,
        "RND2 predecessor changed",
    )
    decisions = ledger.get("decisions")
    _require(
        isinstance(decisions, list) and decisions,
        "RND2 predecessor event is missing",
    )
    event = decisions[-1]
    _require(isinstance(event, dict), "RND2 predecessor event must be a mapping")
    _require(event.get("event") == RND1_PASS, "RND2 predecessor event changed")
    _require(
        event.get("decision") == RND2_DECISION,
        "RND2 predecessor decision changed",
    )
    _require(
        event.get("authority_token") == TOKENS["RND2"],
        "RND2 predecessor token changed",
    )
    _require(event.get("exit_code") == 0, "RND1 terminal exit is not zero")
    _require(event.get("artifact_count") == 3, "RND1 terminal artifact count changed")
    device_actual = event.get("device_actual")
    _require(
        isinstance(device_actual, str) and device_actual.startswith("cuda:"),
        "RND1 terminal device is not CUDA",
    )
    expected_terminal_flags = {
        "cpu_fallback": False,
        "outcome_accessed": False,
        "training_loss_accessed": False,
        "scientific_metric_accessed": False,
        "residual_heads_identical": True,
        "pretrained_encoders_different": True,
    }
    observed_terminal_flags = {
        key: event.get(key) for key in expected_terminal_flags
    }
    _require(
        observed_terminal_flags == expected_terminal_flags,
        "RND1 terminal evidence changed",
    )


def _require_authority_paths(
    authority: dict[str, Any], expected: dict[str, Path], *, phase: str
) -> None:
    for name, expected_path in expected.items():
        observed = authority.get(name)
        _require(
            isinstance(observed, str)
            and Path(observed).expanduser().resolve() == expected_path.resolve(),
            f"{phase} canonical authority path changed: {name}",
        )


def _check_rnd3_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    authorization = active["authorization"]
    observed_authorization = {
        key: authorization.get(key) for key in RND3_AUTHORIZATION
    }
    _require(
        observed_authorization == RND3_AUTHORIZATION,
        "RND3 authorization scope or permissions changed",
    )
    _require(active["runnable_phases"] == ["RND3"], "RND3 is not solely runnable")
    _require(active["training_allowed"] is True, "RND3 training is not open")
    _require(
        active["candidate_model_training_allowed"] is True,
        "RND3 candidate training is not open",
    )
    _require(active["held_score_read_allowed"] is False, "RND3 held score opened")
    _require(
        active["partial_fold_score_read_allowed"] is False,
        "RND3 partial score opened",
    )
    _require(
        active["new_external_outcome_access_allowed"] is False,
        "RND3 external outcome opened",
    )
    _require(active["gate_state"] == RND3_GATE_STATE, "RND3 gate state changed")
    _require(active["next_allowed_action"] == RND3_ACTION, "RND3 action changed")
    _require(ledger["next_action"] == RND3_ACTION, "RND3 ledger action changed")
    _require(ledger["score_accessed"] is False, "RND3 score was accessed")
    _require(
        contract["phase_contract"]["RND3"]["required_predecessor"]
        == RND2_MERGE_PASS,
        "RND3 predecessor changed",
    )
    _require_authority_paths(
        active["authority"], RND3_AUTHORITY_PATHS, phase="RND3"
    )
    decisions = ledger.get("decisions")
    _require(
        isinstance(decisions, list) and decisions,
        "RND3 predecessor event is missing",
    )
    event = decisions[-1]
    _require(isinstance(event, dict), "RND3 predecessor event must be a mapping")
    expected_event = {
        "event": RND2_MERGE_PASS,
        "decision": RND3_DECISION,
        "experiment_id": "RND2_RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE",
        "folds": [0, 1],
        "seed": 0,
        "point_epochs": 3,
        "calibration_epochs": 3,
        "controller_exit_code": 0,
        "runner_exit_codes": [0, 0],
        "cuda_only": True,
        "cpu_fallback": False,
        "held_target_accessed": False,
        "score_accessed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "canonical_merge_path": str(RND2_MERGED_PATH),
        "canonical_merge_status": RND2_MERGE_PASS,
        "authority_token": TOKENS["RND3"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND2 terminal merge evidence changed")


def _check_rnd4_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    authorization = active["authorization"]
    observed_authorization = {
        key: authorization.get(key) for key in RND4_AUTHORIZATION
    }
    _require(
        observed_authorization == RND4_AUTHORIZATION,
        "RND4 authorization scope or permissions changed",
    )
    _require(active["runnable_phases"] == ["RND4"], "RND4 is not solely runnable")
    _require(active["training_allowed"] is False, "RND4 training remained open")
    _require(
        active["candidate_model_training_allowed"] is False,
        "RND4 candidate training remained open",
    )
    _require(active["held_score_read_allowed"] is True, "RND4 score is not open")
    _require(
        active["partial_fold_score_read_allowed"] is False,
        "RND4 partial score opened",
    )
    _require(
        active["new_external_outcome_access_allowed"] is False,
        "RND4 external outcome opened",
    )
    _require(active["gate_state"] == RND4_GATE_STATE, "RND4 gate state changed")
    _require(active["next_allowed_action"] == RND4_ACTION, "RND4 action changed")
    _require(ledger["next_action"] == RND4_ACTION, "RND4 ledger action changed")
    _require(ledger["score_accessed"] is False, "RND4 score was already accessed")
    _require(
        contract["phase_contract"]["RND4"]["required_predecessor"]
        == RND3_MERGE_PASS,
        "RND4 predecessor changed",
    )
    _require_authority_paths(
        active["authority"], RND4_AUTHORITY_PATHS, phase="RND4"
    )
    decisions = ledger.get("decisions")
    _require(
        isinstance(decisions, list) and decisions,
        "RND4 predecessor event is missing",
    )
    event = decisions[-1]
    _require(isinstance(event, dict), "RND4 predecessor event must be a mapping")
    expected_event = {
        "event": RND3_MERGE_PASS,
        "decision": RND4_DECISION,
        "experiment_id": "RND3_RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY",
        "folds": list(range(20)),
        "seed": 0,
        "artifact_count": 20,
        "held_target_accessed": False,
        "score_accessed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "canonical_merge_path": str(RND3_MERGED_PATH),
        "canonical_merge_status": RND3_MERGE_PASS,
        "authority_token": TOKENS["RND4"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND3 terminal merge evidence changed")


def _check_rnd5_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    authorization = active["authorization"]
    observed_authorization = {
        key: authorization.get(key) for key in RND5_AUTHORIZATION
    }
    _require(
        observed_authorization == RND5_AUTHORIZATION,
        "RND5 authorization scope or permissions changed",
    )
    _require(active["runnable_phases"] == ["RND5"], "RND5 is not solely runnable")
    _require(active["training_allowed"] is False, "RND5 training reopened")
    _require(
        active["candidate_model_training_allowed"] is False,
        "RND5 candidate training reopened",
    )
    _require(active["held_score_read_allowed"] is False, "RND5 scorer rerun opened")
    _require(
        active["partial_fold_score_read_allowed"] is False,
        "RND5 partial score opened",
    )
    _require(
        active["new_external_outcome_access_allowed"] is False,
        "RND5 external outcome opened",
    )
    _require(active["gate_state"] == RND5_GATE_STATE, "RND5 gate state changed")
    _require(active["next_allowed_action"] == RND5_ACTION, "RND5 action changed")
    _require(ledger["next_action"] == RND5_ACTION, "RND5 ledger action changed")
    _require(ledger["score_accessed"] is True, "RND5 score access is not recorded")
    _require(
        contract["phase_contract"]["RND5"]["required_predecessor"]
        == RND4_SCORE_PASS,
        "RND5 predecessor changed",
    )
    _require_authority_paths(
        active["authority"], RND5_AUTHORITY_PATHS, phase="RND5"
    )
    decisions = ledger.get("decisions")
    _require(
        isinstance(decisions, list) and decisions,
        "RND5 predecessor event is missing",
    )
    event = decisions[-1]
    _require(isinstance(event, dict), "RND5 predecessor event must be a mapping")
    expected_event = {
        "event": RND4_SCORE_PASS,
        "decision": RND5_DECISION,
        "canonical_score_path": str(RND4_SCORE_PATH),
        "canonical_score_status": RND4_SCORE_PASS,
        "exit_code": 0,
        "complete_valid_score": True,
        "actual_fold_count": 20,
        "score_accessed": True,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "authority_token": TOKENS["RND5"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND4 terminal score evidence changed")


def _check_rnd5t_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    decisions = ledger.get("decisions")
    _require(isinstance(decisions, list) and decisions, "RND5T terminal event is missing")
    event = decisions[-1]
    _require(isinstance(event, dict), "RND5T terminal event must be a mapping")
    qualification_status = event.get("canonical_qualification_status")
    _require(
        qualification_status in RND5_NONPASS_STATUSES,
        "RND5T qualification status is not canonical",
    )
    terminal_gate_state = {
        **RND5_GATE_STATE,
        "RND5": qualification_status,
        "RND5T": f"TERMINAL_{qualification_status}",
    }
    authorization = active["authorization"]
    observed_authorization = {
        key: authorization.get(key) for key in RND5T_AUTHORIZATION
    }
    _require(
        observed_authorization == RND5T_AUTHORIZATION,
        "RND5T authorization scope or permissions changed",
    )
    _require(active["runnable_phases"] == [], "RND5T must have no runnable phase")
    _require(active["training_allowed"] is False, "RND5T training reopened")
    _require(
        active["candidate_model_training_allowed"] is False,
        "RND5T candidate training reopened",
    )
    _require(active["held_score_read_allowed"] is False, "RND5T score rerun opened")
    _require(active["gate_state"] == terminal_gate_state, "RND5T gate state changed")
    _require(active["next_allowed_action"] == RND5T_ACTION, "RND5T action changed")
    _require(ledger["next_action"] == RND5T_ACTION, "RND5T ledger action changed")
    _require(ledger["score_accessed"] is True, "RND5T score access is not recorded")
    _require(
        ledger["formal_score_accessed"] is False
        and ledger["formal_qualification_accessed"] is False,
        "RND5T formal artifacts were accessed",
    )
    _require_authority_paths(
        active["authority"], RND5T_AUTHORITY_PATHS, phase="RND5T"
    )
    _require(
        contract["phase_contract"]["RND5T"]["required_predecessor_one_of"]
        == list(RND5_NONPASS_STATUSES),
        "RND5T predecessor changed",
    )
    expected_semantics = {
        RND5_SCREEN_FAIL: (1, False, True, True),
        RND5_SCREEN_INDETERMINATE: (2, False, False, False),
    }
    exit_code, gate_passed, integrity_passed, complete_valid = expected_semantics[
        qualification_status
    ]
    expected_event = {
        "event": qualification_status,
        "decision": RND5T_DECISION,
        "canonical_qualification_path": str(RND5_QUALIFICATION_PATH),
        "canonical_qualification_status": qualification_status,
        "exit_code": exit_code,
        "gate_passed": gate_passed,
        "integrity_passed": integrity_passed,
        "complete_valid_qualification": complete_valid,
        "rnd6_authorized": False,
        "formal_result_status": FORMAL_NOT_RUN_STATUS,
        "score_accessed": True,
        "formal_score_accessed": False,
        "formal_qualification_accessed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
        "authority_token": TOKENS["RND5T"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND5 terminal qualification evidence changed")


def _check_formal_common_authority(
    active: dict[str, Any],
    ledger: dict[str, Any],
    *,
    phase: str,
    authorization_expected: dict[str, Any],
    action_expected: str,
    gate_state_expected: dict[str, str],
) -> dict[str, Any]:
    authorization = active["authorization"]
    observed_authorization = {
        key: authorization.get(key) for key in authorization_expected
    }
    _require(
        observed_authorization == authorization_expected,
        f"{phase} authorization scope or permissions changed",
    )
    exclusive_rights = (
        authorization["neural_training_allowed"],
        authorization["score_allowed"],
        authorization["qualification_allowed"],
    )
    _require(
        sum(value is True for value in exclusive_rights) <= 1,
        f"{phase} prediction, score and qualification rights overlap",
    )
    _require(active["partial_fold_score_read_allowed"] is False, f"{phase} partial score opened")
    _require(
        active["new_external_outcome_access_allowed"] is False,
        f"{phase} external outcome opened",
    )
    _require(active["gate_state"] == gate_state_expected, f"{phase} gate state changed")
    _require(active["next_allowed_action"] == action_expected, f"{phase} action changed")
    _require(ledger["next_action"] == action_expected, f"{phase} ledger action changed")
    _require_authority_paths(active["authority"], RND6_CANONICAL_PATHS, phase=phase)
    decisions = ledger.get("decisions")
    _require(isinstance(decisions, list) and decisions, f"{phase} predecessor event is missing")
    event = decisions[-1]
    _require(isinstance(event, dict), f"{phase} predecessor event must be a mapping")
    return event


def _check_rnd6p_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    event = _check_formal_common_authority(
        active,
        ledger,
        phase="RND6P",
        authorization_expected=RND6P_AUTHORIZATION,
        action_expected=RND6P_ACTION,
        gate_state_expected=RND6P_GATE_STATE,
    )
    _require(active["runnable_phases"] == ["RND6P"], "RND6P is not solely runnable")
    _require(active["training_allowed"] is True, "RND6P training is not open")
    _require(active["candidate_model_training_allowed"] is True, "RND6P candidate training is not open")
    _require(active["held_score_read_allowed"] is False, "RND6P held score opened")
    _require(ledger["score_accessed"] is True, "RND5 screen access is not recorded")
    _require(ledger["formal_score_accessed"] is False, "RND6P formal score was accessed")
    expected_event = {
        "event": RND5_SCREEN_PASS,
        "decision": RND6P_DECISION,
        "canonical_qualification_path": str(RND5_QUALIFICATION_PATH),
        "canonical_qualification_status": RND5_SCREEN_PASS,
        "exit_code": 0,
        "gate_passed": True,
        "integrity_passed": True,
        "rnd6_authorized": True,
        "evidence_status": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
        "score_accessed": True,
        "formal_score_accessed": False,
        "formal_qualification_accessed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "authority_token": TOKENS["RND6P"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND5 formal-activation evidence changed")


def _check_rnd6s_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    event = _check_formal_common_authority(
        active,
        ledger,
        phase="RND6S",
        authorization_expected=RND6S_AUTHORIZATION,
        action_expected=RND6S_ACTION,
        gate_state_expected=RND6S_GATE_STATE,
    )
    _require(active["runnable_phases"] == ["RND6S"], "RND6S is not solely runnable")
    _require(active["training_allowed"] is False, "RND6S training remained open")
    _require(active["candidate_model_training_allowed"] is False, "RND6S candidate training remained open")
    _require(active["held_score_read_allowed"] is True, "RND6S formal score is not open")
    _require(ledger["score_accessed"] is True, "prior screen score access is not recorded")
    _require(ledger["formal_score_accessed"] is False, "RND6S formal score was already accessed")
    expected_event = {
        "event": RND6_ASSEMBLY_PASS,
        "decision": RND6S_DECISION,
        "experiment_id": "RND6P_RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY",
        "folds": FORMAL_FOLDS,
        "seeds": FORMAL_SEEDS,
        "expected_fold_seed_pairs": FORMAL_PAIR_COUNT,
        "actual_fold_seed_pairs": FORMAL_PAIR_COUNT,
        "point_epochs": FORMAL_POINT_EPOCHS,
        "calibration_epochs": FORMAL_CALIBRATION_EPOCHS,
        "controller_exit_code": 0,
        "runner_exit_codes_all_zero": True,
        "cuda_only": True,
        "cpu_fallback": False,
        "held_target_accessed": False,
        "score_accessed": True,
        "formal_score_accessed": False,
        "formal_qualification_accessed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "best_seed_selection_performed": False,
        "equal_seed_mixture": True,
        "equal_seed_weight": FORMAL_EQUAL_SEED_WEIGHT,
        "canonical_merge_path": str(RND6_MERGED_PATH),
        "canonical_merge_status": RND6_MERGE_PASS,
        "canonical_assembly_path": str(RND6_ASSEMBLY_MANIFEST_PATH),
        "canonical_assembly_status": RND6_ASSEMBLY_PASS,
        "authority_token": TOKENS["RND6S"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND6P terminal assembly evidence changed")


def _check_rnd6q_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    event = _check_formal_common_authority(
        active,
        ledger,
        phase="RND6Q",
        authorization_expected=RND6Q_AUTHORIZATION,
        action_expected=RND6Q_ACTION,
        gate_state_expected=RND6Q_GATE_STATE,
    )
    _require(active["runnable_phases"] == ["RND6Q"], "RND6Q is not solely runnable")
    _require(active["training_allowed"] is False, "RND6Q training reopened")
    _require(active["candidate_model_training_allowed"] is False, "RND6Q candidate training reopened")
    _require(active["held_score_read_allowed"] is False, "RND6Q scorer rerun opened")
    _require(ledger["formal_score_accessed"] is True, "RND6Q formal score access is not recorded")
    _require(ledger["formal_qualification_accessed"] is False, "RND6Q qualifier was already accessed")
    expected_event = {
        "event": RND6_SCORE_PASS,
        "decision": RND6Q_DECISION,
        "canonical_formal_score_path": str(RND6_SCORE_PATH),
        "canonical_formal_score_status": RND6_SCORE_PASS,
        "exit_code": 0,
        "complete_valid_score": True,
        "actual_fold_count": 20,
        "actual_seed_count": 5,
        "actual_fold_seed_pairs": FORMAL_PAIR_COUNT,
        "equal_seed_mixture": True,
        "equal_seed_weight": FORMAL_EQUAL_SEED_WEIGHT,
        "best_seed_selection_performed": False,
        "formal_score_accessed": True,
        "formal_qualification_accessed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "authority_token": TOKENS["RND6Q"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND6S terminal score evidence changed")


def _check_rnd6t_authority(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    decisions = ledger.get("decisions")
    _require(isinstance(decisions, list) and decisions, "RND6T terminal event is missing")
    event = decisions[-1]
    _require(isinstance(event, dict), "RND6T terminal event must be a mapping")
    qualification_status = event.get("canonical_formal_qualification_status")
    _require(
        qualification_status in RND6_QUALIFICATION_STATUSES,
        "RND6T qualification status is not canonical",
    )
    terminal_gate_state = {
        **RND6Q_GATE_STATE,
        "RND6Q": qualification_status,
        "RND6T": f"TERMINAL_{qualification_status}",
    }
    _check_formal_common_authority(
        active,
        ledger,
        phase="RND6T",
        authorization_expected=RND6T_AUTHORIZATION,
        action_expected=RND6T_ACTION,
        gate_state_expected=terminal_gate_state,
    )
    _require(active["runnable_phases"] == [], "RND6T must have no runnable phase")
    _require(active["training_allowed"] is False, "RND6T training reopened")
    _require(active["candidate_model_training_allowed"] is False, "RND6T candidate training reopened")
    _require(active["held_score_read_allowed"] is False, "RND6T score rerun opened")
    expected_semantics = {
        RND6_QUALIFICATION_PASS: (0, True, True, True),
        RND6_QUALIFICATION_FAIL: (1, False, True, True),
        RND6_QUALIFICATION_INDETERMINATE: (2, False, False, False),
    }
    exit_code, gate_passed, integrity_passed, complete_valid = expected_semantics[
        qualification_status
    ]
    expected_event = {
        "event": qualification_status,
        "decision": RND6T_DECISION,
        "canonical_formal_qualification_path": str(RND6_QUALIFICATION_PATH),
        "canonical_formal_qualification_status": qualification_status,
        "exit_code": exit_code,
        "gate_passed": gate_passed,
        "integrity_passed": integrity_passed,
        "complete_valid_qualification": complete_valid,
        "formal_score_accessed": True,
        "formal_qualification_accessed": True,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
        "authority_token": TOKENS["RND6T"],
    }
    observed_event = {key: event.get(key) for key in expected_event}
    _require(observed_event == expected_event, "RND6Q terminal qualification evidence changed")


def validate_contract(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    active = _load_yaml(repo_root / ACTIVE_PATH)
    contract = _load_yaml(repo_root / CONTRACT_PATH)
    ledger = _load_yaml(repo_root / LEDGER_PATH)
    research_path = repo_root / RESEARCH_PATH
    _require(research_path.is_file(), "research record is missing")
    research = _load_research_frontmatter(research_path)
    _require(active["project_task_id"] == PROJECT_TASK_ID, "wrong active project")
    _require(contract["project_task_id"] == PROJECT_TASK_ID, "wrong machine contract")
    _require(ledger["project_task_id"] == PROJECT_TASK_ID, "wrong decision ledger")
    authority = active["authority"]
    phase = str(authority["current_phase"])
    _require(phase in PHASES, "unknown active phase")
    expected_runnable_phase = "NONE" if phase in TERMINAL_PHASES else phase
    expected_runnable_phases = [] if phase in TERMINAL_PHASES else [phase]
    _require(
        authority["current_runnable_phase"] == expected_runnable_phase,
        "runnable phase diverged",
    )
    _require(
        active["runnable_phases"] == expected_runnable_phases,
        "runnable phase set changed",
    )
    token = RND0_TOKEN if phase == "RND0" else TOKENS[phase]
    _require(contract["contract_status"] == token, "machine contract status diverged")
    _require(
        authority["current_authority_state"] == token,
        "active authority state diverged",
    )
    _require(authority["binding_status"] == token, "active binding status diverged")
    _require(ledger["current_status"] == token, "ledger status diverged")
    _require(research["status"] == token, "research status diverged")
    _require(active["allowed_phases"] == list(PHASES), "allowed phase universe changed")
    _require(authority["branch"] == BRANCH, "active branch binding changed")
    _require(Path(authority["worktree"]) == WORKTREE, "active worktree binding changed")
    _require(Path(authority["artifact_root"]) == ARTIFACT_ROOT, "artifact root changed")
    _require(
        authority["machine_contract_path"] == str(CONTRACT_PATH),
        "machine contract pointer changed",
    )
    _require(authority["decision_ledger_path"] == str(LEDGER_PATH), "ledger pointer changed")
    _require(authority["research_record_path"] == str(RESEARCH_PATH), "research pointer changed")
    _require(ledger["current_phase"] == phase, "ledger phase diverged")
    _require(
        ledger["score_accessed"] is False
        or phase in {"RND5", "RND5T", "RND6P", "RND6S", "RND6Q", "RND6T"},
        "score flag widened early",
    )
    _require(ledger["partial_score_accessed"] is False, "partial score was accessed")
    _require(ledger["new_external_outcome_accessed"] is False, "external outcome was accessed")
    _require(active["partial_fold_score_read_allowed"] is False, "partial score authority opened")
    _require(active["new_external_outcome_access_allowed"] is False, "external outcome authority opened")
    _require(active["new_split_allowed"] is False, "new split authority opened")
    _require(active["terminal_parent_artifact_overwrite_allowed"] is False, "parent overwrite opened")
    _require(active["parent_terminal_verdict_change_allowed"] is False, "parent verdict reopened")
    parent = active["parent_state"]
    _require(parent["old_route_revival_allowed"] is False, "old route revival opened")
    _require(parent["old_artifact_rerun_allowed"] is False, "old rerun opened")
    _check_frozen_scientific_contract(contract)
    _check_formal_design_state(
        active, contract, ledger, research, phase=phase
    )
    _check_result_registry(repo_root, active, ledger, research, phase=phase)

    if phase == "RND0":
        _require(active["authorization"]["implementation_allowed"] is True, "RND0 implementation closed")
        _require(active["training_allowed"] is False, "RND0 training opened")
        _require(active["held_score_read_allowed"] is False, "RND0 score opened")
    else:
        _require(contract["phase_contract"][phase]["authority_token"] == token, f"{phase} contract token changed")
        if phase == "RND1":
            _check_rnd1_authority(active, ledger)
        elif phase == "RND2":
            _check_rnd2_authority(active, contract, ledger)
        elif phase == "RND3":
            _check_rnd3_authority(active, contract, ledger)
        elif phase == "RND4":
            _check_rnd4_authority(active, contract, ledger)
        elif phase == "RND5":
            _check_rnd5_authority(active, contract, ledger)
        elif phase == "RND5T":
            _check_rnd5t_authority(active, contract, ledger)
        elif phase == "RND6P":
            _check_rnd6p_authority(active, contract, ledger)
        elif phase == "RND6S":
            _check_rnd6s_authority(active, contract, ledger)
        elif phase == "RND6Q":
            _check_rnd6q_authority(active, contract, ledger)
        elif phase == "RND6T":
            _check_rnd6t_authority(active, contract, ledger)
        if phase in {"RND1", "RND2", "RND3", "RND6P"}:
            _require(active["training_allowed"] is True, f"{phase} training not open")
            _require(active["held_score_read_allowed"] is False, f"{phase} held score opened")
        elif phase in {"RND4", "RND6S"}:
            _require(active["training_allowed"] is False, "RND4 training remained open")
            _require(active["held_score_read_allowed"] is True, f"{phase} score not open")
        elif phase in {"RND5", "RND5T", "RND6Q", "RND6T"}:
            _require(active["training_allowed"] is False, f"{phase} training remained open")
            _require(active["held_score_read_allowed"] is False, f"{phase} scorer rerun opened")

    actual_branch = _git(repo_root, "branch", "--show-current")
    _require(actual_branch == BRANCH, f"checked out branch is {actual_branch!r}")
    _require(not _git(repo_root, "status", "--porcelain"), "worktree must be clean for authority validation")
    return {
        "schema_version": "reactflow_delta.independent_rnet_distill_authority_validation.v1",
        "status": "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS",
        "project_task_id": PROJECT_TASK_ID,
        "phase": phase,
        "branch": actual_branch,
        "training_allowed": bool(active["training_allowed"]),
        "held_score_read_allowed": bool(active["held_score_read_allowed"]),
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
    }


def assert_run_authority(repo_root: Path, phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase}")
    result = validate_contract(repo_root)
    if result["phase"] != phase:
        raise RuntimeError(f"active phase {result['phase']} does not authorize {phase}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(validate_contract(args.repo_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
