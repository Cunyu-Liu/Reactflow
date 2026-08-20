#!/usr/bin/env python3
"""Authority validation for the narrowly scoped Model Rescue v2 amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


MACHINE = Path("configs/reactflow_delta/model_rescue_v2_amendment.yaml")
ACTIVE = Path("configs/reactflow_delta/active_contract.yaml")
LEDGER = Path("docs/prospective_v2/model_rescue_v2_decision_ledger.yaml")
HUMAN = Path("docs/prospective_v2/model_rescue_v2_amendment_20260820.md")
PARENT_MACHINE = Path("configs/reactflow_delta/model_rescue_contract_v1.yaml")
PARENT_RESULT = Path("docs/prospective_v2/audit/m2_qualification_v1.json")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def validate_contract(repo_root: Path) -> dict[str, Any]:
    machine = _load_yaml(repo_root / MACHINE)
    active = _load_yaml(repo_root / ACTIVE)
    ledger = _load_yaml(repo_root / LEDGER)
    parent = _load_yaml(repo_root / PARENT_MACHINE)
    parent_result = json.loads((repo_root / PARENT_RESULT).read_text(encoding="utf-8"))
    phase_ids = [row["id"] for row in machine["phase_graph"]]
    prohibited = set(machine["authorization"]["prohibited"])
    checks = {
        "machine_schema": machine.get("schema_version")
        == "reactflow_delta.model_rescue_v2_amendment.v1",
        "active_schema": active.get("schema_version")
        == "reactflow_delta.active_contract.v2",
        "human_exists": (repo_root / HUMAN).is_file(),
        "active_points_machine": active["authority"]["machine_contract_path"]
        == MACHINE.as_posix(),
        "active_points_human": active["authority"]["human_contract_path"]
        == HUMAN.as_posix(),
        "active_points_ledger": active["authority"]["decision_ledger_path"]
        == LEDGER.as_posix(),
        "branch": active["authority"]["branch"]
        == "codex/reactflow-delta-model-rescue-v2-20260820",
        "parent_head": machine["parent"]["parent_head"]
        == "00c0cf3a804effb89ff99a8e9ea009963dc650d0",
        "parent_terminal_contract": parent["contract_status"]
        == "TERMINAL_M2_NO_RESCUE_CANDIDATE_BENCHMARK_ROUTE_LOCKED",
        "parent_terminal_result": parent_result["overall_status"]
        == "M2_NO_RESCUE_CANDIDATE",
        "scope_narrow": machine["authority"]["amendment_scope"]
        == "MEAN_FIRST_ZERO_MEAN_RESIDUAL_ONLY",
        "phase_order": phase_ids
        == ["R2M0", "R2M1", "R2M2", "R2M3", "R2M4", "R2M5"],
        "r2m1_active": active["authority"]["current_phase"] == "R2M1"
        and active["gate_state"]["R2M0"] == "PASS"
        and active["gate_state"]["R2M1"] == "IN_PROGRESS",
        "training_closed_during_implementation": active["training_allowed"] is False
        and active["candidate_model_training_allowed"] is False,
        "external_closed": active["new_external_outcome_access_allowed"] is False
        and machine["claim_policy"]["external_replication"] == "NOT_ESTABLISHED",
        "only_two_candidates": set(machine["models"])
        == {"mean_aligned", "global_residual", "calibrated_residual"},
        "fixed_mean_loss": machine["models"]["mean_aligned"]["loss"]
        == "METHOD_BALANCED_EXACT_SIGNED_DELTA_L1",
        "zero_mean_calibration": machine["models"]["calibrated_residual"]["center"]
        == "FROZEN_MEAN_ALIGNED"
        and machine["models"]["calibrated_residual"]["mean_gradient_allowed"] is False,
        "no_model_search": machine["compute"]["model_search_allowed"] is False,
        "forbidden_families_closed": {
            "ADD_LOW_RANK_OR_RANK_SEARCH",
            "ADD_SPARSE_CHANGE_GATE",
            "ADD_STRUCTURE_MODEL",
            "ADD_TEACHER_OR_FOUNDATION_ENSEMBLE",
        }.issubset(prohibited),
        "ledger_parent": ledger["parent_head"]
        == "00c0cf3a804effb89ff99a8e9ea009963dc650d0",
        "claims_closed": ledger["claims"]["external_replication"]
        == "NOT_ESTABLISHED"
        and ledger["claims"]["sota"] == "NOT_ESTABLISHED"
        and ledger["claims"]["publication"] == "PUBLICATION_NOT_READY",
    }
    return {
        "schema_version": "reactflow_delta.model_rescue_v2_contract_validation.v1",
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = validate_contract(args.repo_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
