#!/usr/bin/env python3
"""Fail-closed authority validator for the independent RNet2 distillation project."""

from __future__ import annotations

import argparse
import json
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
PHASES = ("RND0", "RND1", "RND2", "RND3", "RND4", "RND5", "RND6")
TOKENS = {
    "RND1": "RNET_DISTILL_PAIRED_GPU_PRETRAIN_ONCE_ONLY",
    "RND2": "RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE_ONLY",
    "RND3": "RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY",
    "RND4": "RNET_DISTILL_COMPLETE_MERGE_SCORE_ONCE_ONLY",
    "RND5": "RNET_DISTILL_COMPLETE_SCORE_QUALIFIER_ONCE_ONLY",
    "RND6": "RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_ONLY",
}
RND0_TOKEN = "RND0_IMPLEMENTATION_SOURCE_BINDING_ONLY"
RND1_PASS = "RND1_PAIRED_PRETRAIN_EXACT_PASS"
RND2_MERGE_PASS = "RND2_COMPLETE_UNSCORED_ENGINEERING_SMOKE_MERGE_PASS"
RND3_MERGE_PASS = "RND3_COMPLETE_UNSCORED_PREDICTION_MERGE_PASS"
RND4_SCORE_PASS = "RND4_COMPLETE_SCORE_PASS"
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
    "RND6": "NOT_AUTHORIZED",
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
    "RND6": "NOT_AUTHORIZED",
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
    "RND6": "NOT_AUTHORIZED",
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
    "RND6": "NOT_AUTHORIZED",
}

RND1_PRETRAIN_DIR = ARTIFACT_ROOT / "rnd1_pretrain"
RND2_PREDICTION_DIR = ARTIFACT_ROOT / "rnd2_smoke_seed0"
RND3_PREDICTION_DIR = ARTIFACT_ROOT / "rnd3_screen_seed0"
RND2_MERGED_PATH = RND2_PREDICTION_DIR / "rnet_distill_complete_unscored_merge.json"
RND3_MERGED_PATH = RND3_PREDICTION_DIR / "rnet_distill_complete_unscored_merge.json"
RND4_SCORE_PATH = RND3_PREDICTION_DIR / "rnet_distill_complete_score.json"
RND5_QUALIFICATION_PATH = RND3_PREDICTION_DIR / "rnet_distill_qualification.json"
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
    _require(
        schedule["rnd6_formal"]["seeds"] == list(range(5)),
        "formal seed universe changed",
    )
    gates = contract["screen_gates"]
    _require(gates["gate_lowering_after_score_access_allowed"] is False, "Gate lowering opened")
    _require(gates["extra_seed_selection_allowed"] is False, "seed selection opened")
    outcome = contract["outcome_policy"]
    for key in (
        "pretraining_may_read_openknot_mutant_outcome",
        "prediction_runner_may_read_held_target",
        "partial_fold_score_allowed",
        "new_external_outcome_access_allowed",
    ):
        _require(outcome[key] is False, f"outcome boundary widened: {key}")
    gpu = contract["gpu_policy"]
    _require(gpu["training_and_gpu_validation_device_class"] == "CUDA_ONLY", "CUDA-only changed")
    _require(gpu["cpu_model_or_loss_fallback_allowed"] is False, "CPU fallback opened")
    _require(gpu["minimum_free_vram_gate_allowed"] is False, "VRAM gate opened")
    expected_predecessors = {
        "RND3": RND2_MERGE_PASS,
        "RND4": RND3_MERGE_PASS,
        "RND5": RND4_SCORE_PASS,
    }
    for phase, expected in expected_predecessors.items():
        _require(
            contract["phase_contract"][phase]["required_predecessor"] == expected,
            f"{phase} predecessor diverged from canonical status {expected}",
        )


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
    _require(authority["current_runnable_phase"] == phase, "runnable phase diverged")
    _require(active["runnable_phases"] == [phase], "exact single runnable phase required")
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
    _require(ledger["score_accessed"] is False or phase in {"RND5", "RND6"}, "score flag widened early")
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

    if phase == "RND0":
        _require(active["authorization"]["implementation_allowed"] is True, "RND0 implementation closed")
        _require(active["training_allowed"] is False, "RND0 training opened")
        _require(active["held_score_read_allowed"] is False, "RND0 score opened")
    else:
        _require(contract["phase_contract"][phase]["authority_token"] == token, f"{phase} contract token changed")
        if phase == "RND2":
            _check_rnd2_authority(active, contract, ledger)
        elif phase == "RND3":
            _check_rnd3_authority(active, contract, ledger)
        elif phase == "RND4":
            _check_rnd4_authority(active, contract, ledger)
        elif phase == "RND5":
            _check_rnd5_authority(active, contract, ledger)
        if phase in {"RND1", "RND2", "RND3", "RND6"}:
            _require(active["training_allowed"] is True, f"{phase} training not open")
            _require(active["held_score_read_allowed"] is False, f"{phase} held score opened")
        elif phase == "RND4":
            _require(active["training_allowed"] is False, "RND4 training remained open")
            _require(active["held_score_read_allowed"] is True, "RND4 score not open")
        elif phase == "RND5":
            _require(active["training_allowed"] is False, "RND5 training remained open")
            _require(active["held_score_read_allowed"] is False, "RND5 scorer rerun opened")

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
