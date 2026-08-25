#!/usr/bin/env python3
"""Fail-closed validator for the Model Rescue v12 authority and contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


EXPECTED_TRAINING = {
    "V12M2": "V12_REAL_DATA_ENGINEERING_SMOKE_ONLY",
    "V12M3": "V12_TWENTY_FOLD_SCORE_BLIND_SCREEN_ONLY",
    "V12M4": "V12_FIXED_FIVE_SEED_FORMAL_ONLY",
}


def _read(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected YAML mapping at {path}")
    return value


def validate_contract(repo_root: Path) -> dict[str, object]:
    active = _read(repo_root / "configs/reactflow_delta/active_contract.yaml")
    contract = _read(
        repo_root / "configs/reactflow_delta/model_rescue_v12_amendment.yaml"
    )
    ledger = _read(
        repo_root / "docs/prospective_v2/model_rescue_v12_decision_ledger.yaml"
    )
    if active.get("schema_version") != "reactflow_delta.active_contract.v12":
        raise RuntimeError("V12 active schema mismatch")
    if contract.get("schema_version") != "reactflow_delta.model_rescue_v12_amendment.v1":
        raise RuntimeError("V12 machine contract schema mismatch")
    if ledger.get("schema_version") != "reactflow_delta.model_rescue_v12_decision_ledger.v1":
        raise RuntimeError("V12 decision ledger schema mismatch")
    if active["parent_state"]["model_rescue_v11_status"] != (
        "TERMINAL_V11M3_TOP_JOURNAL_SCREEN_FAIL_DIAGNOSTICS_COMPLETE"
    ) or ledger["immutable_parent_verdicts"]["v11"] != (
        "TERMINAL_V11M3_TOP_JOURNAL_SCREEN_FAIL_DIAGNOSTICS_COMPLETE"
    ):
        raise RuntimeError("V11 terminal verdict is not preserved")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V12 partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V12 external outcomes must remain locked")
    if contract["candidate"]["backbone_change_allowed"] is not False:
        raise RuntimeError("V12 cannot change the parent backbone")
    if contract["gate"]["trainable_parameters"] != 4:
        raise RuntimeError("V12 gate must contain exactly four parameters")
    if contract["gate"]["gate_family_or_input_search_allowed"] is not False:
        raise RuntimeError("V12 gate family search is prohibited")
    if contract["inner_crossfit"]["in_sample_outer_point_predictions_for_gate_fit_allowed"] is not False:
        raise RuntimeError("V12 gate requires inner-OOF point predictions")
    if contract["residual_calibration"]["family_change_allowed"] is not False:
        raise RuntimeError("V12 residual family must remain V10")
    gates = contract["v12m3_screen"]["gates"]
    if gates["signed_delta_relative_gain_vs_feature41_min"] != 0.10:
        raise RuntimeError("V12 signed top-journal Gate changed")
    if gates["task_crps_relative_gain_vs_feature41_min"] != 0.05:
        raise RuntimeError("V12 CRPS top-journal Gate changed")
    if gates["distribution_absolute_delta_relative_gain_vs_feature41_min"] != 0.15:
        raise RuntimeError("V12 magnitude top-journal Gate changed")
    return {
        "status": "V12_CONTRACT_VALIDATION_PASS",
        "phase": active["authority"]["current_phase"],
        "training_allowed": active["training_allowed"],
        "held_score_read_allowed": active["held_score_read_allowed"],
    }


def assert_run_authority(repo_root: Path, phase: str) -> None:
    active = _read(repo_root / "configs/reactflow_delta/active_contract.yaml")
    expected = EXPECTED_TRAINING.get(phase)
    if expected is None:
        raise RuntimeError(f"V12 training is undefined for {phase}")
    if active["authority"]["current_phase"] != phase or active.get(
        "runnable_phases"
    ) != [phase]:
        raise RuntimeError("V12 requested phase is not the sole active authority")
    if active.get("training_allowed") != expected or active.get(
        "candidate_model_training_allowed"
    ) != expected:
        raise RuntimeError("V12 training token mismatch")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("V12 training requires held score access closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V12 partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V12 external outcomes must remain locked")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(validate_contract(args.repo_root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
