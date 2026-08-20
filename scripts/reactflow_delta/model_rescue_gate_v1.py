#!/usr/bin/env python3
"""Contract validation and scientific gates for model-rescue v1.

The module is deliberately small: it checks the active contract binding and
implements the two decisions that change the project route. It does not inspect
or authorize external outcome access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


MACHINE_CONTRACT = Path("configs/reactflow_delta/model_rescue_contract_v1.yaml")
ACTIVE_CONTRACT = Path("configs/reactflow_delta/active_contract.yaml")
HUMAN_CONTRACT = Path("docs/prospective_v2/model_rescue_contract_v1_20260820.md")
DECISION_LEDGER = Path("docs/prospective_v2/model_rescue_decision_ledger_v1.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def validate_contract(repo_root: Path) -> dict[str, Any]:
    machine = _load_yaml(repo_root / MACHINE_CONTRACT)
    active = _load_yaml(repo_root / ACTIVE_CONTRACT)
    ledger = _load_yaml(repo_root / DECISION_LEDGER)

    phase_ids = [row["id"] for row in machine["phase_graph"]]
    claims = {row["id"]: row["current_status"] for row in ledger["claims"]}
    allocations = machine["compute_budget"]["allocations"]

    checks = {
        "machine_schema": machine.get("schema_version")
        == "reactflow_delta.model_rescue_contract.v1",
        "active_schema": active.get("schema_version")
        == "reactflow_delta.active_contract.v2",
        "human_contract_exists": (repo_root / HUMAN_CONTRACT).is_file(),
        "active_points_to_machine": active["authority"]["machine_contract_path"]
        == MACHINE_CONTRACT.as_posix(),
        "active_points_to_human": active["authority"]["human_contract_path"]
        == HUMAN_CONTRACT.as_posix(),
        "active_points_to_ledger": active["authority"]["decision_ledger_path"]
        == DECISION_LEDGER.as_posix(),
        "phase_order": phase_ids == ["M0", "M1", "M2", "M3", "M4", "M5", "M6"],
        "m2_terminal_benchmark_route": active["authority"]["current_phase"] == "M6"
        and active["gate_state"]["M0"] == "PASS"
        and active["gate_state"]["M1"] == "PASS"
        and active["gate_state"]["M2"] == "FAIL_NO_RESCUE_CANDIDATE"
        and active["gate_state"]["M3"] == "NOT_RUN_PREREQUISITE_FAILED"
        and machine["contract_status"]
        == "TERMINAL_M2_NO_RESCUE_CANDIDATE_BENCHMARK_ROUTE_LOCKED"
        and ledger["phase_state"]["M2"] == "FAIL_NO_RESCUE_CANDIDATE"
        and ledger["terminal_route"] == "BENCHMARK_ROUTE_LOCKED"
        and active["authority"]["binding_status"]
        == "M2_TERMINAL_RESULT_FOCUSED_COMMIT_COMPLETE",
        "candidate_training_closed": active["training_allowed"] is False
        and active["candidate_model_training_allowed"] is False,
        "new_external_outcomes_closed": active["new_external_outcome_access_allowed"]
        is False,
        "development_consumed": machine["scope"]["development_status"]
        == "DEVELOPMENT_CONSUMED",
        "dual_primary": machine["primary_metrics"]["rule"]
        == "INTERSECTION_UNION_BOTH_MUST_PASS",
        "compute_budget_exact": abs(sum(float(v) for v in allocations.values()) - 5.0)
        < 1e-9,
        "lrso_kill_allowed": active["authorization"]["scope"]
        == "PERFORMANCE_FIRST_MODEL_RESCUE_WITH_LRSO_KILL_ALLOWED",
        "low_rank_claim_downgraded": claims.get("C_LOW_RANK_METHOD")
        == "SMALL_SIGNIFICANT_BELOW_PRACTICAL_GATE",
        "current_network_signed_delta_qualified": claims.get("C_SIGNED_MUTATION_EFFECT")
        == "CURRENT_NETWORK_DEVELOPMENT_PASS_VS_WT",
        "structure_candidate_excluded": machine["m1_structure_probe"]["status"]
        == "COMPLETED_STRUCTDELTA_EXCLUDED",
        "external_not_established": claims.get("C_EXTERNAL_GENERALIZATION")
        == "NOT_ESTABLISHED",
        "sota_not_established": claims.get("C_SOTA") == "SOTA_NOT_ESTABLISHED",
        "model_rescue_failed": claims.get("C_MODEL_RESCUE") == "METHOD_RESCUE_FAIL",
    }
    return {
        "schema_version": "reactflow_delta.model_rescue_contract_validation.v1",
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def resolve_internal_gate(
    *,
    crps_gain: float,
    crps_ci_low: float,
    baseline_crps: float,
    delta_mae_gain: float,
    delta_mae_ci_low: float,
    baseline_delta_mae: float,
    beats_wt_anchor: bool,
    crps_positive_puzzles: int,
    delta_mae_positive_puzzles: int,
    leave_one_puzzle_positive: bool,
    max_single_puzzle_fraction: float,
    prediction_coverage: float,
    failure_rate: float,
    coverage_error_worsening_pp: float,
) -> dict[str, Any]:
    crps_required = max(0.003, 0.02 * baseline_crps)
    mae_required = 0.02 * baseline_delta_mae
    checks = {
        "crps_ci": crps_ci_low > 0.0,
        "crps_practical": crps_gain >= crps_required,
        "delta_mae_ci": delta_mae_ci_low > 0.0,
        "delta_mae_practical": delta_mae_gain >= mae_required,
        "beats_wt_anchor": bool(beats_wt_anchor),
        "crps_puzzle_consistency": crps_positive_puzzles >= 14,
        "delta_mae_puzzle_consistency": delta_mae_positive_puzzles >= 12,
        "leave_one_puzzle_positive": bool(leave_one_puzzle_positive),
        "influence_bounded": max_single_puzzle_fraction <= 0.25,
        "coverage": prediction_coverage >= 0.995,
        "failure_rate": failure_rate <= 0.0,
        "calibration_guardrail": coverage_error_worsening_pp <= 2.0,
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "required_crps_gain": crps_required,
        "required_delta_mae_gain": mae_required,
        "status": "POST_HOC_DEVELOPMENT_PASS" if passed else "METHOD_RESCUE_FAIL",
        "next_route": "M4_FINAL_MODEL_FREEZE" if passed else "BENCHMARK_ROUTE_LOCKED",
    }


def resolve_external_gate(
    *,
    joint_units: int,
    lineages: int,
    max_lineage_fraction: float,
    crps_ci_low: float,
    crps_relative_gain: float,
    delta_mae_ci_low: float,
    delta_mae_relative_gain: float,
    leave_one_lineage_nonnegative: bool,
    model_frozen_before_access: bool,
) -> dict[str, Any]:
    eligibility = {
        "joint_units": joint_units >= 9,
        "lineages": lineages >= 3,
        "lineage_balance": max_lineage_fraction <= 0.50,
        "model_frozen": bool(model_frozen_before_access),
    }
    if not all(eligibility.values()):
        return {
            "eligibility": eligibility,
            "status": "EXPLORATORY_EXTERNAL_ONLY",
            "next_route": "BENCHMARK_ROUTE_LOCKED",
        }
    scientific = {
        "crps": crps_ci_low > 0.0 and crps_relative_gain >= 0.02,
        "delta_mae": delta_mae_ci_low > 0.0 and delta_mae_relative_gain >= 0.02,
        "leave_one_lineage": bool(leave_one_lineage_nonnegative),
    }
    passed = all(scientific.values())
    return {
        "eligibility": eligibility,
        "scientific": scientific,
        "status": "PROSPECTIVE_EXTERNAL_PASS" if passed else "PROSPECTIVE_EXTERNAL_FAIL",
        "next_route": (
            "METHOD_AUGMENTED_BENCHMARK_ELIGIBLE" if passed else "BENCHMARK_ROUTE_LOCKED"
        ),
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
