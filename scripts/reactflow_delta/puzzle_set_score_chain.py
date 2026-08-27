#!/usr/bin/env python3
"""Exact runtime and historical-reference checks for Puzzle-Set scoring.

The pure merge, assembly, score, and qualification functions remain usable with
small fixtures.  Their command-line entry points use the helpers here to bind
every real artifact to the active P1 authority before opening scientific data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from scripts.reactflow_delta.puzzle_set_safe_sources import (
    SOURCE_BINDING_STATUS,
    validate_source_manifest,
)
from scripts.reactflow_delta.score_model_rescue_v13 import (
    SCHEMA as V13_SCORE_SCHEMA,
)


EXPECTED_PROJECT_TASK = "reactflow_delta_puzzle_set_meta_context"
EXPECTED_FOLDS = tuple(range(20))
SOURCE_BOUND_PHASES = frozenset({"P1M2", "P1M3", "P1M4"})

V13_HISTORICAL_TOP_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "scores",
    "target_profile_identity",
    "target_join_after_complete_merge",
    "terminal_parent_metrics_from_frozen_complete_v12_score",
    "partial_fold_scores_inspected",
    "external_outcome_accessed",
    "model_or_threshold_selection_performed",
}
V13_HISTORICAL_METRIC_FIELDS = {
    "feature41_signed_delta_mae",
    "candidate_signed_delta_mae",
    "null_signed_delta_mae",
    "feature41_absolute_delta_mae",
    "candidate_point_absolute_delta_mae",
    "null_point_absolute_delta_mae",
    "candidate_distribution_absolute_delta_mae",
    "null_distribution_absolute_delta_mae",
    "feature41_crps",
    "candidate_crps",
    "null_crps",
    "feature41_coverage68",
    "candidate_coverage68",
    "feature41_coverage95",
    "candidate_coverage95",
    "registered_prediction_coverage",
    "failure_rate",
    "terminal_v12_signed_delta_mae",
    "terminal_v11_point_absolute_delta_mae",
    "terminal_v12_crps",
    "terminal_v10_distribution_absolute_delta_mae",
}
V13_HISTORICAL_COUNT_FIELDS = {
    "n_qualified_positions",
    "n_registered_expected",
    "n_registered_observed",
    "n_unexpected_prediction_keys",
}
V13_HISTORICAL_ROW_FIELDS = {
    "outer_fold",
    "held_puzzle",
    *V13_HISTORICAL_METRIC_FIELDS,
    *V13_HISTORICAL_COUNT_FIELDS,
}

P1_SCORE_METRIC_FIELDS = {
    "feature41_signed_delta_mae",
    "parent_signed_delta_mae",
    "candidate_signed_delta_mae",
    "null_signed_delta_mae",
    "feature41_absolute_delta_mae",
    "parent_point_absolute_delta_mae",
    "candidate_point_absolute_delta_mae",
    "null_point_absolute_delta_mae",
    "candidate_distribution_absolute_delta_mae",
    "null_distribution_absolute_delta_mae",
    "candidate_crps",
    "null_crps",
    "candidate_coverage68",
    "null_coverage68",
    "candidate_coverage95",
    "null_coverage95",
    "feature41_crps",
    "feature41_coverage68",
    "feature41_coverage95",
    "historical_v13_signed_delta_mae",
    "historical_v13_point_absolute_delta_mae",
    "terminal_v12_signed_delta_mae",
    "terminal_v11_point_absolute_delta_mae",
    "terminal_v12_crps",
    "terminal_v10_distribution_absolute_delta_mae",
    "registered_prediction_coverage",
    "failure_rate",
}
P1_SCORE_COUNT_FIELDS = {
    "n_qualified_positions",
    "n_registered_expected",
    "n_registered_observed",
    "n_unexpected_prediction_keys",
}
P1_SCORE_ROW_FIELDS = {
    "outer_fold",
    "held_puzzle",
    *P1_SCORE_METRIC_FIELDS,
    *P1_SCORE_COUNT_FIELDS,
}


def _finite_nonnegative_metric(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} is malformed")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} is invalid")
    return normalized


def validate_p1_score_rows(rows: Any, *, source: str) -> list[dict[str, Any]]:
    """Require exact, finite, nonnegative method-balanced puzzle score rows."""

    by_fold = _canonical_fold_rows(rows, source=source)
    for fold, row in by_fold.items():
        if set(row) != P1_SCORE_ROW_FIELDS:
            raise ValueError(f"{source} fold {fold} score fields changed")
        for field in P1_SCORE_METRIC_FIELDS:
            _finite_nonnegative_metric(
                row[field], label=f"{source} fold {fold} {field}"
            )
        for field in P1_SCORE_COUNT_FIELDS:
            value = row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{source} fold {fold} {field} is invalid")
        for field in (
            "candidate_coverage68",
            "null_coverage68",
            "candidate_coverage95",
            "null_coverage95",
            "feature41_coverage68",
            "feature41_coverage95",
            "registered_prediction_coverage",
            "failure_rate",
        ):
            if float(row[field]) > 1.0:
                raise ValueError(f"{source} fold {fold} {field} exceeds one")
    return [by_fold[fold] for fold in EXPECTED_FOLDS]


def validate_retention_score_protocol(
    value: Any, *, expected_run_count: int
) -> dict[str, Any]:
    fields = {
        "candidate_pretraining_established_all_runs",
        "candidate_retention_positive_all_runs",
        "null_pretraining_established_all_runs",
        "null_retention_positive_all_runs",
        "fold_seed_diagnostics",
        "selection_performed",
        "mutant_outcome_used",
        "held_puzzle_accessed",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Puzzle-Set score retention protocol fields changed")
    diagnostics = value.get("fold_seed_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != expected_run_count:
        raise ValueError("Puzzle-Set score retention universe is incomplete")
    if (
        value.get("selection_performed") is not False
        or value.get("mutant_outcome_used") is not False
        or value.get("held_puzzle_accessed") is not False
        or not all(
            isinstance(value.get(name), bool)
            for name in (
                "candidate_pretraining_established_all_runs",
                "candidate_retention_positive_all_runs",
                "null_pretraining_established_all_runs",
                "null_retention_positive_all_runs",
            )
        )
    ):
        raise ValueError("Puzzle-Set score retention protocol changed")
    return value


def load_active_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs/reactflow_delta/active_contract.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Puzzle-Set active contract must be one mapping")
    return value


def assert_active_phase(
    repo_root: Path,
    *,
    phase: str,
    score_token: str | None = None,
    training_must_be_closed: bool = False,
    held_score_must_be_closed: bool = False,
) -> dict[str, Any]:
    """Require the exact active P1 phase and its access boundary."""

    active = load_active_contract(repo_root)
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("Puzzle-Set is not the active project task")
    authority = active.get("authority")
    if not isinstance(authority, dict) or authority.get("current_phase") != phase:
        raise RuntimeError(f"Puzzle-Set runtime is closed outside {phase}")
    if active.get("runnable_phases") != [phase]:
        raise RuntimeError(f"{phase} must be the only runnable phase")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("Puzzle-Set partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("Puzzle-Set external outcomes must remain locked")
    if phase in SOURCE_BOUND_PHASES:
        source_manifest = Path(str(authority.get("source_manifest_path", "")))
        if authority.get("source_binding_status") != SOURCE_BINDING_STATUS:
            raise RuntimeError(
                f"{phase} requires the exact bound Puzzle-Set source manifest"
            )
        if not source_manifest.is_absolute():
            raise RuntimeError(
                f"{phase} source_manifest_path must be one exact absolute path"
            )
        validate_source_manifest(source_manifest)
    if training_must_be_closed and (
        active.get("training_allowed") is not False
        or active.get("candidate_model_training_allowed") is not False
    ):
        raise RuntimeError("Puzzle-Set training must be closed before scoring")
    if score_token is not None and active.get("held_score_read_allowed") != score_token:
        raise RuntimeError("Puzzle-Set complete score-once authority is closed")
    if held_score_must_be_closed and active.get("held_score_read_allowed") is not False:
        raise RuntimeError("Puzzle-Set held score access must remain closed")
    return active


def assert_authority_paths(
    active: Mapping[str, Any], bindings: Mapping[str, Path]
) -> None:
    """Bind a complete CLI path set exactly to absolute active-authority paths."""

    authority = active.get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("Puzzle-Set active authority must be one mapping")
    if not bindings:
        raise RuntimeError("Puzzle-Set authority path binding cannot be empty")
    for field, cli_path in bindings.items():
        authority_path = Path(str(authority.get(field, "")))
        observed = Path(cli_path)
        if not authority_path.is_absolute():
            raise RuntimeError(f"Puzzle-Set authority {field} is not an absolute path")
        if not observed.is_absolute() or observed != authority_path:
            raise RuntimeError(f"Puzzle-Set CLI {field} differs from active authority")


def _canonical_fold_rows(rows: Any, *, source: str) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FOLDS):
        raise ValueError(f"{source} must contain exactly twenty fold rows")
    validated: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{source} contains a malformed fold row")
        fold = row.get("outer_fold")
        if not isinstance(fold, int) or isinstance(fold, bool):
            raise ValueError(f"{source} fold identity is malformed")
        if fold in validated:
            raise ValueError(f"{source} contains duplicate outer fold {fold}")
        validated[fold] = row
    if tuple(sorted(validated)) != EXPECTED_FOLDS:
        raise ValueError(f"{source} must contain canonical folds0-19")
    for fold, row in validated.items():
        if row.get("held_puzzle") != f"P{fold + 1:02d}":
            raise ValueError(f"{source} fold {fold} held puzzle is noncanonical")
    return validated


def validate_v13_historical_bundle(
    score: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Validate the sole transitive V10/V11/V12/V13 score-stage lineage."""

    if (
        set(score) != V13_HISTORICAL_TOP_FIELDS
        or score.get("schema_version") != V13_SCORE_SCHEMA
        or score.get("phase") != "V13M3"
        or score.get("status") != "V13M3_COMPLETE_SCORE_PASS"
        or score.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION"
        or score.get("target_join_after_complete_merge") is not True
        or score.get("terminal_parent_metrics_from_frozen_complete_v12_score")
        is not True
        or score.get("partial_fold_scores_inspected") is not False
        or score.get("external_outcome_accessed") is not False
        or score.get("model_or_threshold_selection_performed") is not False
    ):
        raise ValueError("Puzzle-Set V13 historical bundle protocol changed")
    rows = _canonical_fold_rows(score.get("scores"), source="V13 historical bundle")
    for fold, row in rows.items():
        if set(row) != V13_HISTORICAL_ROW_FIELDS:
            raise ValueError(f"V13 historical bundle fold {fold} fields changed")
        for field in V13_HISTORICAL_METRIC_FIELDS:
            _finite_nonnegative_metric(
                row[field],
                label=f"V13 historical bundle fold {fold} {field}",
            )
        for field in V13_HISTORICAL_COUNT_FIELDS:
            value = row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"V13 historical bundle fold {fold} {field} is invalid"
                )
        for field in (
            "feature41_coverage68",
            "candidate_coverage68",
            "feature41_coverage95",
            "candidate_coverage95",
            "registered_prediction_coverage",
            "failure_rate",
        ):
            if float(row[field]) > 1.0:
                raise ValueError(
                    f"V13 historical bundle fold {fold} {field} exceeds one"
                )
        if (
            float(row["registered_prediction_coverage"]) != 1.0
            or float(row["failure_rate"]) != 0.0
            or int(row["n_unexpected_prediction_keys"]) != 0
            or int(row["n_registered_expected"]) <= 0
            or int(row["n_registered_observed"]) != int(row["n_registered_expected"])
            or int(row["n_qualified_positions"]) <= 0
        ):
            raise ValueError(
                f"V13 historical bundle fold {fold} is not a qualified reference"
            )
    return rows
