from __future__ import annotations

from pathlib import Path

from scripts.reactflow_delta.model_rescue_gate_v1 import (
    resolve_external_gate,
    resolve_internal_gate,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_contract_files_and_active_pointer_are_consistent():
    result = validate_contract(ROOT)
    assert result["status"] == "PASS", result
    assert all(result["checks"].values())


def test_internal_gate_requires_both_primary_metrics():
    common = dict(
        crps_gain=0.005,
        crps_ci_low=0.001,
        baseline_crps=0.18,
        delta_mae_gain=0.006,
        delta_mae_ci_low=0.001,
        baseline_delta_mae=0.25,
        beats_wt_anchor=True,
        crps_positive_puzzles=15,
        delta_mae_positive_puzzles=13,
        leave_one_puzzle_positive=True,
        max_single_puzzle_fraction=0.20,
        prediction_coverage=1.0,
        failure_rate=0.0,
        coverage_error_worsening_pp=1.0,
    )
    passed = resolve_internal_gate(**common)
    assert passed["status"] == "POST_HOC_DEVELOPMENT_PASS"

    crps_only = dict(common, delta_mae_gain=-0.001, delta_mae_ci_low=-0.002)
    failed = resolve_internal_gate(**crps_only)
    assert failed["status"] == "METHOD_RESCUE_FAIL"
    assert failed["next_route"] == "BENCHMARK_ROUTE_LOCKED"


def test_internal_gate_enforces_practical_not_only_statistical_gain():
    result = resolve_internal_gate(
        crps_gain=0.001,
        crps_ci_low=0.0001,
        baseline_crps=0.18,
        delta_mae_gain=0.001,
        delta_mae_ci_low=0.0001,
        baseline_delta_mae=0.25,
        beats_wt_anchor=True,
        crps_positive_puzzles=20,
        delta_mae_positive_puzzles=20,
        leave_one_puzzle_positive=True,
        max_single_puzzle_fraction=0.10,
        prediction_coverage=1.0,
        failure_rate=0.0,
        coverage_error_worsening_pp=0.0,
    )
    assert result["status"] == "METHOD_RESCUE_FAIL"
    assert not result["checks"]["crps_practical"]
    assert not result["checks"]["delta_mae_practical"]


def test_external_gate_does_not_inflate_anchor_count_into_independent_n():
    underpowered = resolve_external_gate(
        joint_units=2,
        lineages=2,
        max_lineage_fraction=0.5,
        crps_ci_low=0.01,
        crps_relative_gain=0.10,
        delta_mae_ci_low=0.01,
        delta_mae_relative_gain=0.10,
        leave_one_lineage_nonnegative=True,
        model_frozen_before_access=True,
    )
    assert underpowered["status"] == "EXPLORATORY_EXTERNAL_ONLY"
    assert underpowered["next_route"] == "BENCHMARK_ROUTE_LOCKED"


def test_external_gate_requires_dual_metric_replication():
    result = resolve_external_gate(
        joint_units=9,
        lineages=3,
        max_lineage_fraction=0.45,
        crps_ci_low=0.01,
        crps_relative_gain=0.03,
        delta_mae_ci_low=-0.001,
        delta_mae_relative_gain=0.03,
        leave_one_lineage_nonnegative=True,
        model_frozen_before_access=True,
    )
    assert result["status"] == "PROSPECTIVE_EXTERNAL_FAIL"
    assert result["next_route"] == "BENCHMARK_ROUTE_LOCKED"
