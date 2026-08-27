#!/usr/bin/env python3
"""Mechanically apply the frozen four-comparison branch-5 route Gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
import yaml

from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_FOLDS,
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    assert_frozen_runtime_paths,
)
from scripts.reactflow_delta.score_post_v14_branch5_route_probe import (
    COMPLETE_STATUS as SCORE_COMPLETE_STATUS,
    SCHEMA as SCORE_SCHEMA,
)


QUALIFICATION_PHASE = "B5RP3"
SCHEMA = "reactflow_delta.puzzle_set_branch5_route_probe_qualification.v1"
PASS_STATUS = "BRANCH5_ROUTE_PROBE_EXACT_PASS"
FAIL_STATUS = "BRANCH5_ROUTE_PROBE_COMPLETE_FAIL_P3"
INDETERMINATE_STATUS = "BRANCH5_ROUTE_PROBE_INDETERMINATE"
RELATIVE_GAIN_MIN = 0.01
POSITIVE_PUZZLES_MIN = 14
T_CRITICAL_DF19 = 2.093024054408263
EXPECTED_SCORE_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "scores",
    "integrity_errors",
    "complete_valid_score",
    "target_profile_identity",
    "target_join_after_complete_merge",
    "aggregation",
    "independent_units",
    "partial_fold_scores_inspected",
    "external_outcome_accessed",
    "model_or_threshold_selection_performed",
    "source_provenance_complete",
}
EXPECTED_SCORE_ROW_FIELDS = {
    "outer_fold",
    "held_puzzle",
    "parent_signed_delta_mae",
    "aligned_signed_delta_mae",
    "shift17_signed_delta_mae",
    "parent_point_absolute_delta_mae",
    "aligned_point_absolute_delta_mae",
    "shift17_point_absolute_delta_mae",
    "n_registered_expected",
    "n_registered_observed",
    "registered_prediction_coverage",
    "failure_rate",
    "n_unexpected_prediction_keys",
    "n_qualified_positions",
    "score_integrity_pass",
}


def assert_qualifier_authority(
    repo_root: Path,
    *,
    score_json: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("branch5 qualifier is not the active project")
    if active.get("authority", {}).get("current_phase") != QUALIFICATION_PHASE:
        raise RuntimeError("branch5 qualifier is closed outside B5RP3")
    if active.get("runnable_phases") != [QUALIFICATION_PHASE]:
        raise RuntimeError("B5RP3 must be the only runnable phase")
    if (
        active.get("training_allowed") is not False
        or active.get("candidate_model_training_allowed") is not False
        or active.get("held_score_read_allowed") is not False
        or active.get("partial_fold_score_read_allowed") is not False
        or active.get("new_external_outcome_access_allowed") is not False
    ):
        raise RuntimeError(
            "branch5 qualifier requires all training and outcome access closed"
        )
    parent = active.get("parent_state", {})
    if any(parent.get(name) != value for name, value in EXPECTED_PARENT_STATE.items()):
        raise RuntimeError("branch5 qualifier parent route is not exact")
    provided = {
        "complete_score_path": score_json,
        "qualification_path": out_json,
    }
    assert_frozen_runtime_paths(
        active.get("authority"),
        required_fields=tuple(provided),
        cli_paths=(
            {name: value for name, value in provided.items() if value is not None}
            if any(value is not None for value in provided.values())
            else None
        ),
    )
    return active


def paired_summary(
    rows: list[dict[str, Any]], comparator_field: str, aligned_field: str
) -> dict[str, Any]:
    comparator = np.asarray([float(row[comparator_field]) for row in rows])
    aligned = np.asarray([float(row[aligned_field]) for row in rows])
    if (
        comparator.shape != (20,)
        or aligned.shape != (20,)
        or not np.isfinite(comparator).all()
        or not np.isfinite(aligned).all()
        or float(comparator.mean()) <= 0.0
    ):
        raise ValueError("branch5 paired comparison is incomplete or nonfinite")
    effects = comparator - aligned
    mean_gain = float(effects.mean())
    half = float(T_CRITICAL_DF19 * effects.std(ddof=1) / math.sqrt(20))
    comparator_mean = float(comparator.mean())
    return {
        "comparator_field": comparator_field,
        "aligned_field": aligned_field,
        "comparator_mean": comparator_mean,
        "aligned_mean": float(aligned.mean()),
        "mean_gain": mean_gain,
        "relative_gain": mean_gain / comparator_mean,
        "ci95": [mean_gain - half, mean_gain + half],
        "positive_puzzles": int((effects > 0.0).sum()),
        "per_puzzle": effects.tolist(),
    }


def _integrity_errors(scores: dict[str, Any]) -> list[str]:
    errors = []
    if set(scores) != EXPECTED_SCORE_FIELDS:
        errors.append("score_fields")
    if scores.get("schema_version") != SCORE_SCHEMA:
        errors.append("score_schema")
    if scores.get("status") != SCORE_COMPLETE_STATUS:
        errors.append("complete_score_status")
    if scores.get("phase") != "B5RP2":
        errors.append("score_phase")
    if scores.get("complete_valid_score") is not True:
        errors.append("complete_valid_score")
    if scores.get("integrity_errors") != []:
        errors.append("score_integrity_errors")
    if scores.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        errors.append("target_identity")
    expected_top = {
        "target_join_after_complete_merge": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES",
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_provenance_complete": True,
    }
    errors.extend(
        name for name, expected in expected_top.items() if scores.get(name) != expected
    )
    rows = scores.get("scores", [])
    try:
        ordered = sorted(rows, key=lambda row: int(row["outer_fold"]))
        folds = [int(row["outer_fold"]) for row in ordered]
    except (KeyError, TypeError, ValueError):
        errors.append("fold_rows_malformed")
        return sorted(set(errors))
    if len(ordered) != 20 or folds != EXPECTED_FOLDS:
        errors.append("fold_universe")
        return sorted(set(errors))
    metrics = (
        "parent_signed_delta_mae",
        "aligned_signed_delta_mae",
        "shift17_signed_delta_mae",
        "parent_point_absolute_delta_mae",
        "aligned_point_absolute_delta_mae",
        "shift17_point_absolute_delta_mae",
    )
    for row in ordered:
        fold = int(row["outer_fold"])
        if set(row) != EXPECTED_SCORE_ROW_FIELDS:
            errors.append(f"fold{fold}_score_fields")
        try:
            coverage = float(row.get("registered_prediction_coverage", float("nan")))
            failure = float(row.get("failure_rate", float("nan")))
            unexpected = int(row.get("n_unexpected_prediction_keys", -1))
            expected_rows = int(row.get("n_registered_expected", -1))
            observed_rows = int(row.get("n_registered_observed", -1))
            qualified_rows = int(row.get("n_qualified_positions", -1))
        except (TypeError, ValueError):
            errors.append(f"fold{fold}_coverage_or_integrity")
            continue
        if (
            str(row.get("held_puzzle")) != f"P{fold + 1:02d}"
            or coverage != 1.0
            or failure != 0.0
            or unexpected != 0
            or expected_rows <= 0
            or observed_rows != expected_rows
            or qualified_rows <= 0
            or row.get("score_integrity_pass") is not True
        ):
            errors.append(f"fold{row.get('outer_fold')}_coverage_or_integrity")
        try:
            values = np.asarray([float(row[name]) for name in metrics])
        except (KeyError, TypeError, ValueError):
            errors.append(f"fold{row.get('outer_fold')}_metric_missing")
            continue
        if not np.isfinite(values).all() or np.any(values < 0):
            errors.append(f"fold{row.get('outer_fold')}_metric_nonfinite")
    return sorted(set(errors))


def qualify(scores: dict[str, Any]) -> dict[str, Any]:
    integrity_errors = _integrity_errors(scores)
    if integrity_errors:
        return {
            "schema_version": SCHEMA,
            "phase": QUALIFICATION_PHASE,
            "status": INDETERMINATE_STATUS,
            "gate_passed": False,
            "integrity_passed": False,
            "integrity_errors": integrity_errors,
            "gates": {},
            "comparisons": {},
            "puzzle_set_v5_eligible": False,
            "route_after_indeterminate": "P3_STOP_MODEL_RESCUE",
            "external_replication": "NOT_ESTABLISHED",
            "sota": "NOT_ESTABLISHED",
            "publication_ready": False,
        }
    rows = sorted(scores["scores"], key=lambda row: int(row["outer_fold"]))
    try:
        comparisons = {
            "signed_aligned_vs_v13_parent": paired_summary(
                rows, "parent_signed_delta_mae", "aligned_signed_delta_mae"
            ),
            "point_absolute_aligned_vs_v13_parent": paired_summary(
                rows,
                "parent_point_absolute_delta_mae",
                "aligned_point_absolute_delta_mae",
            ),
            "signed_aligned_vs_shift17": paired_summary(
                rows, "shift17_signed_delta_mae", "aligned_signed_delta_mae"
            ),
            "point_absolute_aligned_vs_shift17": paired_summary(
                rows,
                "shift17_point_absolute_delta_mae",
                "aligned_point_absolute_delta_mae",
            ),
        }
    except ValueError as error:
        return {
            "schema_version": SCHEMA,
            "phase": QUALIFICATION_PHASE,
            "status": INDETERMINATE_STATUS,
            "gate_passed": False,
            "integrity_passed": False,
            "integrity_errors": [f"paired_comparison:{error}"],
            "gates": {},
            "comparisons": {},
            "puzzle_set_v5_eligible": False,
            "route_after_indeterminate": "P3_STOP_MODEL_RESCUE",
            "external_replication": "NOT_ESTABLISHED",
            "sota": "NOT_ESTABLISHED",
            "publication_ready": False,
        }
    gates: dict[str, bool] = {"prediction_and_score_integrity": True}
    for name, result in comparisons.items():
        gates[f"{name}_relative_gain_ge_1pct"] = (
            float(result["relative_gain"]) >= RELATIVE_GAIN_MIN
        )
        gates[f"{name}_ci_lower_gt_zero"] = float(result["ci95"][0]) > 0.0
        gates[f"{name}_positive_puzzles_ge_14"] = (
            int(result["positive_puzzles"]) >= POSITIVE_PUZZLES_MIN
        )
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": QUALIFICATION_PHASE,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "gate_passed": passed,
        "integrity_passed": True,
        "integrity_errors": [],
        "gates": gates,
        "comparisons": comparisons,
        "puzzle_set_v5_eligible": passed,
        "route_after_complete_fail": None if passed else "P3_STOP_MODEL_RESCUE",
        "model_or_threshold_selection_performed": False,
        "evidence_status": "POST_HOC_DEVELOPMENT_ROUTE_PROBE_ONLY",
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    score_json = args.score_json.resolve()
    out_json = args.out_json.resolve()
    assert_qualifier_authority(
        args.repo_root.resolve(), score_json=score_json, out_json=out_json
    )
    if out_json.exists():
        raise FileExistsError("branch5 refuses to overwrite its qualification")
    result = qualify(json.loads(score_json.read_text(encoding="utf-8")))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_json.with_name(f"{out_json.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, out_json)
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    if result["status"] == PASS_STATUS:
        return 0
    return 1 if result["status"] == FAIL_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
