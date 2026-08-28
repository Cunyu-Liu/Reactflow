#!/usr/bin/env python3
"""Mechanically apply the frozen independent RNet-distillation RND5 Gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
import yaml

from scripts.reactflow_delta.score_independent_rnet_distill import (
    EXPECTED_FOLDS,
    MERGED_PATH,
    QUALIFICATION_PATH,
    SCORE_PATH,
    SCORE_ROW_FIELDS,
    SCORE_SCHEMA,
    SCORE_STATUS,
    SCREEN_DIR,
)
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    PROJECT_TASK_ID,
    assert_run_authority,
)


QUALIFICATION_PHASE = "RND5"
QUALIFICATION_SCHEMA = "reactflow_delta.independent_rnet_distill_qualification.v1"
PASS_STATUS = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_PASS"
FAIL_STATUS = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_FAIL"
INDETERMINATE_STATUS = (
    "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_INDETERMINATE"
)
EVIDENCE_STATUS = "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY"

FROZEN_SCREEN_GATES = {
    "completeness": {
        "exact_fold_count": 20,
        "registered_prediction_coverage": 1.0,
        "failed_rows": 0,
        "duplicate_or_unexpected_artifacts": 0,
    },
    "matched_null_relative_gain_minimum": {
        "signed_delta": 0.03,
        "point_absolute": 0.03,
        "task_crps": 0.015,
        "distribution_absolute": 0.015,
    },
    "matched_null_paired_ci_lower_must_exceed_zero": True,
    "matched_null_positive_puzzles_minimum": 14,
    "feature41_relative_gain_minimum": {
        "signed_delta": 0.10,
        "point_absolute": 0.05,
        "task_crps": 0.05,
        "distribution_absolute": 0.05,
    },
    "historical_parent_relative_gain_minimum": {
        "signed_delta": 0.02,
        "point_absolute": 0.02,
        "task_crps": 0.01,
        "distribution_absolute": 0.01,
    },
    "historical_parent_by_metric": {
        "signed_delta": "V14_CANDIDATE",
        "point_absolute": "V14_CANDIDATE",
        "task_crps": "V14_CANDIDATE",
        "distribution_absolute": "V10_HISTORICAL_DISTRIBUTION",
    },
    "historical_parent_paired_ci_lower_must_exceed_zero": True,
    "historical_parent_positive_puzzles_minimum": 14,
    "coverage_95_interval": [0.94, 0.96],
    "single_puzzle_influence_maximum": 0.20,
    "gate_lowering_after_score_access_allowed": False,
    "extra_seed_selection_allowed": False,
}

SCORE_TOP_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "scores",
    "integrity_errors",
    "complete_valid_score",
    "complete_fold_artifact_universe",
    "expected_fold_count",
    "actual_fold_count",
    "failed_rows",
    "duplicate_or_unexpected_artifacts",
    "target_profile_identity",
    "target_join_after_complete_merge",
    "aggregation",
    "independent_units",
    "attribution_null",
    "feature41_comparator",
    "historical_parent_source",
    "historical_distribution_comparator",
    "partial_fold_scores_inspected",
    "external_outcome_accessed",
    "model_or_threshold_selection_performed",
    "source_exposure_status",
}

METRIC_FIELDS = {
    "signed_delta": {
        "candidate": "candidate_signed_delta_mae",
        "matched_null": "null_signed_delta_mae",
        "feature41": "feature41_signed_delta_mae",
        "historical_parent": "historical_v14_signed_delta_mae",
    },
    "point_absolute": {
        "candidate": "candidate_point_absolute_delta_mae",
        "matched_null": "null_point_absolute_delta_mae",
        "feature41": "feature41_point_absolute_delta_mae",
        "historical_parent": "historical_v14_point_absolute_delta_mae",
    },
    "task_crps": {
        "candidate": "candidate_crps",
        "matched_null": "null_crps",
        "feature41": "feature41_crps",
        "historical_parent": "historical_v14_crps",
    },
    "distribution_absolute": {
        "candidate": "candidate_distribution_absolute_delta_mae",
        "matched_null": "null_distribution_absolute_delta_mae",
        "feature41": "feature41_distribution_absolute_delta_mae",
        "historical_parent": "historical_v10_distribution_absolute_delta_mae",
    },
}


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a mapping at {path}")
    return value


def assert_qualifier_authority(
    repo_root: Path, *, score_json: Path, out_json: Path
) -> dict[str, Any]:
    """Require the independent validator's exact RND5 authority and paths."""

    assert_run_authority(repo_root, QUALIFICATION_PHASE)
    active = _load_yaml(repo_root / "configs/reactflow_delta/active_contract.yaml")
    if active.get("project_task_id") != PROJECT_TASK_ID:
        raise RuntimeError("independent RNet qualifier is not the active project")
    authority = active.get("authority", {})
    expected = {
        "screen_prediction_dir": SCREEN_DIR,
        "complete_unscored_merge_path": MERGED_PATH,
        "complete_score_path": SCORE_PATH,
        "qualification_path": QUALIFICATION_PATH,
    }
    for name, path in expected.items():
        if name not in authority or _resolved(authority[name]) != path.resolve():
            raise RuntimeError(f"RND5 active authority {name} is not exact")
    if _resolved(score_json) != expected["complete_score_path"].resolve():
        raise RuntimeError("RND5 CLI complete_score_path differs from active authority")
    if _resolved(out_json) != expected["qualification_path"].resolve():
        raise RuntimeError("RND5 CLI qualification_path differs from active authority")
    return active


def load_frozen_screen_gates(repo_root: Path) -> dict[str, Any]:
    contract = _load_yaml(
        repo_root / "configs/reactflow_delta/independent_rnet_distill_contract.yaml"
    )
    if (
        contract.get("schema_version")
        != "reactflow_delta.independent_rnet_distill_contract.v1"
        or contract.get("project_task_id") != PROJECT_TASK_ID
        or contract.get("screen_gates") != FROZEN_SCREEN_GATES
    ):
        raise RuntimeError("RND5 frozen screen Gates differ from the signed contract")
    return contract["screen_gates"]


def paired_summary(
    rows: list[dict[str, Any]], comparator_field: str, candidate_field: str
) -> dict[str, Any]:
    comparator = np.asarray([float(row[comparator_field]) for row in rows])
    candidate = np.asarray([float(row[candidate_field]) for row in rows])
    if (
        comparator.shape != (20,)
        or candidate.shape != (20,)
        or not np.isfinite(comparator).all()
        or not np.isfinite(candidate).all()
        or np.any(comparator < 0.0)
        or np.any(candidate < 0.0)
        or float(comparator.mean()) <= 0.0
    ):
        raise ValueError("RND5 paired comparison is incomplete or nonfinite")
    effects = comparator - candidate
    mean_gain = float(effects.mean())
    half = float(student_t.ppf(0.975, 19) * effects.std(ddof=1) / math.sqrt(20))
    comparator_mean = float(comparator.mean())
    leave_one_out = [float(np.delete(effects, index).mean()) for index in range(20)]
    effect_sum = float(effects.sum())
    max_fraction = (
        float(np.max(np.abs(effects)) / abs(effect_sum))
        if effect_sum != 0.0
        else float("inf")
    )
    return {
        "comparator_field": comparator_field,
        "candidate_field": candidate_field,
        "comparator_mean": comparator_mean,
        "candidate_mean": float(candidate.mean()),
        "mean_gain": mean_gain,
        "relative_gain": mean_gain / comparator_mean,
        "ci95": [mean_gain - half, mean_gain + half],
        "positive_puzzles": int((effects > 0.0).sum()),
        "per_puzzle": effects.tolist(),
        "leave_one_puzzle_out": leave_one_out,
        "leave_one_puzzle_out_all_positive": all(value > 0.0 for value in leave_one_out),
        "max_single_puzzle_effect_fraction": max_fraction,
    }


def _score_integrity_errors(score: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(score) != SCORE_TOP_FIELDS:
        errors.append("score_fields")
    expected_top = {
        "schema_version": SCORE_SCHEMA,
        "phase": "RND4",
        "status": SCORE_STATUS,
        "integrity_errors": [],
        "complete_valid_score": True,
        "complete_fold_artifact_universe": True,
        "expected_fold_count": 20,
        "actual_fold_count": 20,
        "failed_rows": 0,
        "duplicate_or_unexpected_artifacts": 0,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES",
        "attribution_null": "RNET2_SHIFT17_SINGLE_FEATURE_DISTILLATION",
        "feature41_comparator": "AUTHORITATIVE_FEATURE41_SEED0_REPLAY",
        "historical_parent_source": "FROZEN_V14_CANONICAL_COMPLETE_SCORE",
        "historical_distribution_comparator": (
            "FROZEN_V10_COMPARATOR_CARRIED_IN_CURRENT_PREDICTION"
        ),
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_exposure_status": EVIDENCE_STATUS,
    }
    errors.extend(
        name for name, expected in expected_top.items() if score.get(name) != expected
    )
    rows = score.get("scores")
    if not isinstance(rows, list):
        return sorted(set([*errors, "score_rows_malformed"]))
    try:
        ordered = sorted(rows, key=lambda row: int(row["outer_fold"]))
        folds = [int(row["outer_fold"]) for row in ordered]
    except (KeyError, TypeError, ValueError):
        return sorted(set([*errors, "score_rows_malformed"]))
    if len(ordered) != 20 or folds != EXPECTED_FOLDS:
        return sorted(set([*errors, "fold_universe"]))
    metric_fields = sorted(
        {
            field
            for fields in METRIC_FIELDS.values()
            for field in fields.values()
        }
    )
    for row in ordered:
        fold = int(row["outer_fold"])
        if set(row) != SCORE_ROW_FIELDS:
            errors.append(f"fold{fold}_score_fields")
        try:
            metrics = np.asarray([float(row[name]) for name in metric_fields])
            coverages = np.asarray(
                [
                    float(row["feature41_coverage95"]),
                    float(row["candidate_coverage95"]),
                    float(row["null_coverage95"]),
                ]
            )
            expected_rows = int(row["n_registered_expected"])
            observed_rows = int(row["n_registered_observed"])
            qualified_rows = int(row["n_qualified_positions"])
            failed_rows = int(row["failed_rows"])
            duplicates = int(row["n_duplicate_prediction_keys"])
            unexpected = int(row["n_unexpected_prediction_keys"])
            registered_coverage = float(row["registered_prediction_coverage"])
            failure_rate = float(row["failure_rate"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"fold{fold}_metric_or_integrity_malformed")
            continue
        if (
            str(row.get("held_puzzle")) != f"P{fold + 1:02d}"
            or not np.isfinite(metrics).all()
            or np.any(metrics < 0.0)
            or not np.isfinite(coverages).all()
            or np.any(coverages < 0.0)
            or np.any(coverages > 1.0)
            or expected_rows <= 0
            or observed_rows != expected_rows
            or qualified_rows <= 0
            or registered_coverage != 1.0
            or failure_rate != 0.0
            or failed_rows != 0
            or duplicates != 0
            or unexpected != 0
            or row.get("score_integrity_pass") is not True
        ):
            errors.append(f"fold{fold}_metric_or_integrity")
    return sorted(set(errors))


def _indeterminate(errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "phase": QUALIFICATION_PHASE,
        "status": INDETERMINATE_STATUS,
        "gate_passed": False,
        "integrity_passed": False,
        "integrity_errors": errors,
        "gates": {},
        "comparisons": {},
        "calibration": {},
        "rnd6_authorized": False,
        "evidence_status": EVIDENCE_STATUS,
        "clean_ood": "NOT_ESTABLISHED",
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def qualify(score: dict[str, Any], screen_gates: dict[str, Any]) -> dict[str, Any]:
    if screen_gates != FROZEN_SCREEN_GATES:
        raise RuntimeError("RND5 cannot apply changed or lowered screen Gates")
    integrity_errors = _score_integrity_errors(score)
    if integrity_errors:
        return _indeterminate(integrity_errors)
    rows = sorted(score["scores"], key=lambda row: int(row["outer_fold"]))
    try:
        comparisons = {
            f"{metric}_vs_{comparator}": paired_summary(
                rows, fields[comparator], fields["candidate"]
            )
            for metric, fields in METRIC_FIELDS.items()
            for comparator in ("matched_null", "feature41", "historical_parent")
        }
    except (KeyError, TypeError, ValueError) as error:
        return _indeterminate([f"paired_comparison:{error}"])

    gates: dict[str, bool] = {
        "prediction_and_score_integrity": True,
        "exact_fold_count_20": len(rows)
        == int(screen_gates["completeness"]["exact_fold_count"]),
        "registered_prediction_coverage_eq_1": all(
            float(row["registered_prediction_coverage"])
            == float(screen_gates["completeness"]["registered_prediction_coverage"])
            for row in rows
        ),
        "failed_rows_eq_0": sum(int(row["failed_rows"]) for row in rows)
        == int(screen_gates["completeness"]["failed_rows"]),
        "duplicate_or_unexpected_artifacts_eq_0": int(
            score["duplicate_or_unexpected_artifacts"]
        )
        == int(screen_gates["completeness"]["duplicate_or_unexpected_artifacts"]),
    }
    for metric in METRIC_FIELDS:
        null_result = comparisons[f"{metric}_vs_matched_null"]
        feature_result = comparisons[f"{metric}_vs_feature41"]
        parent_result = comparisons[f"{metric}_vs_historical_parent"]
        gates[f"{metric}_gain_vs_matched_null_ge_frozen_minimum"] = (
            float(null_result["relative_gain"])
            >= float(screen_gates["matched_null_relative_gain_minimum"][metric])
        )
        gates[f"{metric}_gain_vs_feature41_ge_frozen_minimum"] = (
            float(feature_result["relative_gain"])
            >= float(screen_gates["feature41_relative_gain_minimum"][metric])
        )
        gates[f"{metric}_gain_vs_historical_parent_ge_frozen_minimum"] = (
            float(parent_result["relative_gain"])
            >= float(screen_gates["historical_parent_relative_gain_minimum"][metric])
        )
    null_results = [
        comparisons[f"{metric}_vs_matched_null"] for metric in METRIC_FIELDS
    ]
    parent_results = [
        comparisons[f"{metric}_vs_historical_parent"] for metric in METRIC_FIELDS
    ]
    gates["matched_null_ci_lower_each_gt_zero"] = all(
        float(result["ci95"][0]) > 0.0 for result in null_results
    )
    gates["matched_null_positive_puzzles_each_ge_14"] = all(
        int(result["positive_puzzles"])
        >= int(screen_gates["matched_null_positive_puzzles_minimum"])
        for result in null_results
    )
    gates["historical_parent_ci_lower_each_gt_zero"] = all(
        float(result["ci95"][0]) > 0.0 for result in parent_results
    )
    gates["historical_parent_positive_puzzles_each_ge_14"] = all(
        int(result["positive_puzzles"])
        >= int(screen_gates["historical_parent_positive_puzzles_minimum"])
        for result in parent_results
    )
    coverage95 = float(np.mean([row["candidate_coverage95"] for row in rows]))
    lower, upper = map(float, screen_gates["coverage_95_interval"])
    gates["candidate_coverage95_in_frozen_interval"] = lower <= coverage95 <= upper
    max_influence = float(screen_gates["single_puzzle_influence_maximum"])
    gates["max_single_puzzle_effect_fraction_all_comparisons_le_0_20"] = all(
        float(result["max_single_puzzle_effect_fraction"]) <= max_influence
        for result in comparisons.values()
    )
    passed = all(gates.values())
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "phase": QUALIFICATION_PHASE,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "gate_passed": passed,
        "integrity_passed": True,
        "integrity_errors": [],
        "gates": gates,
        "comparisons": comparisons,
        "calibration": {
            "coverage95": {
                "candidate": coverage95,
                "frozen_interval": [lower, upper],
                "within_interval": gates[
                    "candidate_coverage95_in_frozen_interval"
                ],
            }
        },
        "frozen_gate_values": screen_gates,
        "rnd6_authorized": passed,
        "model_or_threshold_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "new_external_outcome_accessed": False,
        "evidence_status": EVIDENCE_STATUS,
        "clean_ood": "NOT_ESTABLISHED",
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def _write_json_once(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("RND5 refuses to overwrite its canonical qualification")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    score_json = args.score_json.resolve()
    out_json = args.out_json.resolve()
    try:
        assert_qualifier_authority(repo_root, score_json=score_json, out_json=out_json)
        if out_json.exists():
            raise FileExistsError("RND5 qualification already exists; refusing rerun")
        screen_gates = load_frozen_screen_gates(repo_root)
        score = json.loads(score_json.read_text(encoding="utf-8"))
        result = qualify(score, screen_gates)
        _write_json_once(out_json, result)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": INDETERMINATE_STATUS, "error": str(error)}
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    if result["status"] == PASS_STATUS:
        return 0
    return 1 if result["status"] == FAIL_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
