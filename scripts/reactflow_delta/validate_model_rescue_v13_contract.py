#!/usr/bin/env python3
"""Mechanical consistency checks for the active V13 amendment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V13 YAML root must be a mapping: {path}")
    return value


def validate_contract(repo_root: Path) -> dict[str, Any]:
    active = _load(repo_root / "configs/reactflow_delta/active_contract.yaml")
    contract = _load(
        repo_root / "configs/reactflow_delta/model_rescue_v13_amendment.yaml"
    )
    ledger = _load(
        repo_root / "docs/prospective_v2/model_rescue_v13_decision_ledger.yaml"
    )
    if active.get("schema_version") != "reactflow_delta.active_contract.v13":
        raise RuntimeError("V13 active authority schema is not active")
    authority = active["authority"]
    for key in (
        "human_contract_path",
        "machine_contract_path",
        "decision_ledger_path",
        "implementation_plan_path",
        "architecture_decision_path",
    ):
        if not (repo_root / authority[key]).is_file():
            raise RuntimeError(f"V13 authority target is missing: {key}")
    parent = "TERMINAL_V12M3_TOP_JOURNAL_SCREEN_FAIL_DIAGNOSTICS_COMPLETE"
    if active["parent_state"]["v12_status"] != parent:
        raise RuntimeError("V13 active authority changed the V12 terminal status")
    if contract["parent"]["v12_status"] != parent:
        raise RuntimeError("V13 machine contract changed the V12 terminal status")
    if ledger["immutable_parent_verdicts"]["v12"] != parent:
        raise RuntimeError("V13 ledger changed the V12 terminal status")
    if any(
        active[name] is not False
        for name in (
            "held_score_read_allowed",
            "partial_fold_score_read_allowed",
            "new_external_outcome_access_allowed",
            "v12_terminal_verdict_change_allowed",
        )
    ):
        raise RuntimeError("V13 outcome or parent-verdict authority is too broad")
    if contract["scope"]["candidate"] != (
        "v13_feature41_anchored_exact_mutant_contrast"
    ) or contract["scope"]["nested_null"] != (
        "v13_feature41_anchored_wt_replay_null"
    ):
        raise RuntimeError("V13 candidate or nested null differs from the freeze")
    if contract["model"]["search"] != {
        "architecture": False,
        "width": False,
        "depth": False,
        "loss": False,
        "epoch": False,
        "seed_subset": False,
        "calibration_family": False,
    }:
        raise RuntimeError("V13 search space was reopened")
    if contract["model"]["exact_nested_contrast"].get(
        "exact_trainable_parameters_each"
    ) != 2_064_737:
        raise RuntimeError("V13 exact point parameter count changed")
    if contract["model"]["exact_nested_contrast"].get(
        "paired_encoder_dropout_mask_shared"
    ) is not True:
        raise RuntimeError("V13 paired encoder dropout invariant is absent")
    return {
        "status": "V13_CONTRACT_VALIDATION_PASS",
        "phase": authority["current_phase"],
        "training_allowed": active["training_allowed"],
        "held_score_read_allowed": active["held_score_read_allowed"],
    }


def assert_run_authority(repo_root: Path, phase: str) -> None:
    active = _load(repo_root / "configs/reactflow_delta/active_contract.yaml")
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(f"V13 {phase} is not the sole active authority")
    if active.get("runnable_phases") != [phase]:
        raise RuntimeError(f"V13 {phase} is not the sole runnable phase")
    required = {
        "V13M2": "V13_REAL_DATA_ENGINEERING_SMOKE_ONLY",
        "V13M3": "V13_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY",
        "V13M4": "V13_FIXED_FIVE_SEED_FORMAL_ONLY",
    }.get(phase)
    if required is None or active.get("training_allowed") != required:
        raise RuntimeError(f"V13 {phase} training authority is absent")
    if active.get("candidate_model_training_allowed") != required:
        raise RuntimeError(f"V13 {phase} candidate training authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError(f"V13 {phase} requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError(f"V13 {phase} requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError(f"V13 {phase} requires external outcomes locked")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate_contract(args.repo_root.resolve())
    print(yaml.safe_dump(result, sort_keys=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
