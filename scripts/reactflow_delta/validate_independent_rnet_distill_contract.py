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


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a mapping at {path}")
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


def validate_contract(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    active = _load_yaml(repo_root / ACTIVE_PATH)
    contract = _load_yaml(repo_root / CONTRACT_PATH)
    ledger = _load_yaml(repo_root / LEDGER_PATH)
    _require((repo_root / RESEARCH_PATH).is_file(), "research record is missing")
    _require(active["project_task_id"] == PROJECT_TASK_ID, "wrong active project")
    _require(contract["project_task_id"] == PROJECT_TASK_ID, "wrong machine contract")
    _require(ledger["project_task_id"] == PROJECT_TASK_ID, "wrong decision ledger")
    authority = active["authority"]
    phase = str(authority["current_phase"])
    _require(phase in PHASES, "unknown active phase")
    _require(authority["current_runnable_phase"] == phase, "runnable phase diverged")
    _require(active["runnable_phases"] == [phase], "exact single runnable phase required")
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
        _require(contract["contract_status"] == "RND0_IMPLEMENTATION_SOURCE_BINDING_ONLY", "RND0 contract status changed")
        _require(authority["current_authority_state"] == "RND0_IMPLEMENTATION_SOURCE_BINDING_ONLY", "RND0 state changed")
        _require(active["authorization"]["implementation_allowed"] is True, "RND0 implementation closed")
        _require(active["training_allowed"] is False, "RND0 training opened")
        _require(active["held_score_read_allowed"] is False, "RND0 score opened")
    else:
        token = TOKENS[phase]
        _require(authority["current_authority_state"] == token, f"{phase} token changed")
        _require(contract["phase_contract"][phase]["authority_token"] == token, f"{phase} contract token changed")
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
