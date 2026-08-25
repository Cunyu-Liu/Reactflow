#!/usr/bin/env python3
"""Merge only complete score-blind Model Rescue v12 fold universes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.model_rescue_v12 import (
    CANDIDATE,
    GATE_PARAMETERS,
    PREDICTION_SCHEMA,
    TASK_MATCHED_NULL,
)
from scripts.reactflow_delta.run_model_rescue_v12 import (
    FOLD_SCHEMA,
    INNER_LEDGER_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v12_merged.v1"


def _expected_universe(phase: str) -> tuple[list[int], list[int], tuple[int, int, int], str]:
    if phase == "V12M2":
        return [0, 1], [0], (3, 20, 3), "V12M2_COMPLETE_UNSCORED_SMOKE_MERGE_PASS"
    if phase == "V12M3":
        return list(range(20)), [0], (40, 500, 40), "V12M3_COMPLETE_UNSCORED_MERGE_PASS"
    if phase == "V12M4":
        return list(range(20)), list(range(5)), (40, 500, 40), "V12M4_COMPLETE_UNSCORED_MERGE_PASS"
    raise ValueError("unsupported V12 merge phase")


def _prediction_checks(path: Path, fold: int, seed: int, expected_rows: int) -> None:
    prohibited = {"target", "target_error", "target_mask", "score", "crps", "signed_delta_mae"}
    required = {
        "keys",
        "biological_scoring_key",
        "feature41_point",
        "v11_parent_point",
        "gate_distance_factor",
        "gate_magnitude_factor",
        "gate_value",
        "candidate_point",
        "candidate_weights",
        "candidate_locations",
        "candidate_scales",
        "parent_weights",
        "parent_locations",
        "parent_scales",
        "registered_status",
    }
    with np.load(path, allow_pickle=True) as handle:
        if str(handle["schema_version"].item()) != PREDICTION_SCHEMA:
            raise ValueError(f"invalid V12 prediction schema in {path}")
        if prohibited & set(handle.files) or not required <= set(handle.files):
            raise ValueError(f"invalid V12 prediction fields in {path}")
        keys = list(map(str, handle["keys"]))
        if len(keys) != expected_rows or len(keys) != len(set(keys)):
            raise ValueError(f"invalid V12 key universe in {path}")
        if not np.array_equal(handle["keys"], handle["biological_scoring_key"]):
            raise ValueError(f"V12 biological key mismatch in {path}")
        if set(map(int, handle["outer_fold"])) != {fold} or set(
            map(int, handle["seed"])
        ) != {seed}:
            raise ValueError(f"V12 fold or seed mismatch in {path}")
        if not np.all(np.asarray(handle["registered_status"]) == "covered"):
            raise ValueError(f"V12 registered coverage failed in {path}")
        gate = np.asarray(handle["gate_value"], dtype=np.float64)
        if not np.isfinite(gate).all() or not ((gate > 0.0) & (gate < 1.0)).all():
            raise ValueError(f"V12 gate range failed in {path}")
        for name in ("candidate_point", "candidate_weights", "candidate_locations", "candidate_scales"):
            if not np.isfinite(np.asarray(handle[name], dtype=np.float64)).all():
                raise ValueError(f"V12 non-finite {name} in {path}")
        if not (np.asarray(handle["candidate_scales"]) > 0.0).all():
            raise ValueError(f"V12 non-positive scale in {path}")
        if not np.allclose(
            np.asarray(handle["candidate_weights"]).sum(axis=1),
            1.0,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(f"V12 mixture weights do not sum to one in {path}")
        candidate = np.asarray(handle["feature41_point"]) + gate * (
            np.asarray(handle["v11_parent_point"])
            - np.asarray(handle["feature41_point"])
        )
        if not np.allclose(
            np.asarray(handle["candidate_point"]), candidate, atol=1e-12, rtol=0.0
        ):
            raise ValueError(f"V12 candidate composition changed in {path}")


def _ledger_checks(path: Path, outer_fold: int, seed: int) -> None:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != INNER_LEDGER_SCHEMA:
        raise ValueError(f"invalid V12 inner ledger schema in {path}")
    if int(ledger.get("outer_fold", -1)) != outer_fold or int(
        ledger.get("seed", -1)
    ) != seed:
        raise ValueError(f"V12 inner ledger fold or seed mismatch in {path}")
    if len(ledger.get("inner_folds", [])) != 4:
        raise ValueError(f"V12 inner ledger is not four-fold in {path}")
    if ledger.get("outer_train_puzzles_covered_once") is not True:
        raise ValueError(f"V12 inner ledger is incomplete in {path}")
    if ledger.get("target_values_stored") is not False:
        raise ValueError(f"V12 inner ledger stores targets in {path}")
    if ledger.get("method_used_as_gate_input") is not False or ledger.get(
        "puzzle_id_used_as_gate_input"
    ) is not False:
        raise ValueError(f"V12 gate uses a forbidden identifier in {path}")
    if set(ledger.get("gate_parameters", {})) != {
        "b_distance",
        "raw_w_distance",
        "b_magnitude",
        "raw_w_magnitude",
    } or len(ledger["gate_parameters"]) != GATE_PARAMETERS:
        raise ValueError(f"V12 gate parameterization changed in {path}")


def merge_folds(input_dir: Path, phase: str) -> dict[str, Any]:
    expected_folds, expected_seeds, schedule, status = _expected_universe(phase)
    expected = {(fold, seed) for seed in expected_seeds for fold in expected_folds}
    rows = []
    seen: set[tuple[int, int]] = set()
    for path in sorted(input_dir.glob("v12_fold_result_fold*_seed*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        pair = (int(row.get("outer_fold", -1)), int(row.get("seed", -1)))
        if pair in seen:
            raise ValueError(f"duplicate V12 fold-seed {pair}")
        seen.add(pair)
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != phase:
            raise ValueError(f"invalid {phase} fold artifact in {path}")
        if row.get("candidate_id") != CANDIDATE or row.get("task_matched_null") != TASK_MATCHED_NULL:
            raise ValueError(f"V12 candidate or null changed in {path}")
        actual_schedule = (
            int(row.get("inner_point_epochs", -1)),
            int(row.get("gate_steps", -1)),
            int(row.get("calibration_epochs", -1)),
        )
        if actual_schedule != schedule:
            raise ValueError(f"V12 schedule changed for fold-seed {pair}")
        invariants = row.get("invariants", {})
        required_true = (
            "inner_crossfit_complete",
            "parent_v11_exact_replay",
            "gate_range_pass",
            "candidate_distribution_median_fixed",
            "prediction_only_artifact",
        )
        required_false = (
            "outer_held_target_used_for_gate_fit",
            "method_used_as_gate_input",
            "partial_score_inspected",
            "external_outcome_accessed",
        )
        if not all(invariants.get(name) is True for name in required_true) or not all(
            invariants.get(name) is False for name in required_false
        ):
            raise ValueError(f"V12 recorded invariant failed for fold-seed {pair}")
        if invariants.get("registered_prediction_coverage") != 1.0 or invariants.get(
            "failure_rate"
        ) != 0.0 or invariants.get("unexpected_keys") != 0:
            raise ValueError(f"V12 prediction integrity failed for fold-seed {pair}")
        prediction_path = Path(row["prediction_artifact"])
        ledger_path = Path(row["inner_crossfit_ledger"])
        checkpoint_path = Path(row["candidate_residual_checkpoint"])
        if not prediction_path.is_file() or not ledger_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"V12 fold-seed {pair} lacks an artifact")
        _prediction_checks(
            prediction_path,
            fold=pair[0],
            seed=pair[1],
            expected_rows=int(row["n_registered_prediction_rows"]),
        )
        _ledger_checks(ledger_path, outer_fold=pair[0], seed=pair[1])
        rows.append(row)
    if seen != expected or len(rows) != len(expected):
        raise ValueError(
            f"V12 fold-seed universe incomplete: found={sorted(seen)} expected={sorted(expected)}"
        )
    rows.sort(key=lambda row: (int(row["seed"]), int(row["outer_fold"])))
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "status": status,
        "folds": rows,
        "merge_integrity": {
            "complete_fold_seed_universe": True,
            "unique_fold_seed_pairs": True,
            "prediction_only_schema": True,
            "target_identity_exact": True,
            "inner_crossfit_complete_all_runs": True,
            "outer_held_target_excluded_from_gate_fit_all_runs": True,
            "four_parameter_monotone_gate_all_runs": True,
            "exact_v11_parent_replay_all_runs": True,
            "v10_median_residual_family_all_runs": True,
            "median_constraint_all_runs": True,
            "partial_scores_inspected": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("V12M2", "V12M3", "V12M4"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_folds(args.input_dir, args.phase)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
