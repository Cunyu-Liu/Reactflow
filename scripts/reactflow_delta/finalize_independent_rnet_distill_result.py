#!/usr/bin/env python3
"""Finalize an existing independent-RNet canonical result without rerunning science.

This is the single production transition from an already-written RND5
qualification to either RND6P or RND5T, and from an already-written RND6Q
qualification to RND6T.  It reads canonical JSON records only, writes authority
and reporting records, and never trains, scores, qualifies, or opens a new
outcome source.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.reactflow_delta.assemble_independent_rnet_distill_formal import (
    ASSEMBLY_STATUS,
    EXPECTED_ASSEMBLY_FIELDS,
    EXPECTED_FOLD_MANIFEST_FIELDS,
    SCHEMA as ASSEMBLY_SCHEMA,
)
from scripts.reactflow_delta.merge_independent_rnet_distill import (
    EXPECTED_FOLD_FIELDS,
    EXPECTED_MERGED_FIELDS,
    MERGE_INTEGRITY,
    SCHEMA as MERGE_SCHEMA,
    STATUS as MERGE_STATUS,
)
from scripts.reactflow_delta.qualify_independent_rnet_distill import (
    EVIDENCE_STATUS,
    FAIL_STATUS as SCREEN_FAIL_STATUS,
    FROZEN_SCREEN_GATES,
    INDETERMINATE_STATUS as SCREEN_INDETERMINATE_STATUS,
    METRIC_FIELDS,
    PASS_STATUS as SCREEN_PASS_STATUS,
    QUALIFICATION_SCHEMA as SCREEN_QUALIFICATION_SCHEMA,
    SCORE_TOP_FIELDS,
)
from scripts.reactflow_delta.qualify_independent_rnet_distill_formal import (
    FORMAL_FAIL_STATUS,
    FROZEN_FORMAL_GATES,
    FORMAL_INDETERMINATE_STATUS,
    FORMAL_PASS_STATUS,
    FORMAL_QUALIFICATION_PHASE,
    FORMAL_QUALIFICATION_SCHEMA,
    FORMAL_SCORE_TOP_FIELDS,
)
from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EVIDENCE_STATUS as PREDICTION_EVIDENCE_STATUS,
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_SCHEDULE,
)
from scripts.reactflow_delta.score_independent_rnet_distill import (
    SCORE_SCHEMA,
    SCORE_STATUS,
)
from scripts.reactflow_delta.score_independent_rnet_distill_formal import (
    FORMAL_SCORE_PHASE,
    FORMAL_SCORE_SCHEMA,
    FORMAL_SCORE_STATUS,
)
import scripts.reactflow_delta.validate_independent_rnet_distill_contract as authority


SCREEN_MERGE_PATH = authority.RND3_MERGED_PATH
SCREEN_SCORE_PATH = authority.RND4_SCORE_PATH
SCREEN_QUALIFICATION_PATH = authority.RND5_QUALIFICATION_PATH
FORMAL_MERGE_PATH = authority.RND6_MERGED_PATH
FORMAL_ASSEMBLY_PATH = authority.RND6_ASSEMBLY_MANIFEST_PATH
FORMAL_SCORE_PATH = authority.RND6_SCORE_PATH
FORMAL_QUALIFICATION_PATH = authority.RND6_QUALIFICATION_PATH
SCREEN_REPORT_PATH = authority.SCREEN_REPORT_PATH
FORMAL_REPORT_PATH = authority.FORMAL_REPORT_PATH

SCREEN_STATUSES = (
    SCREEN_PASS_STATUS,
    SCREEN_FAIL_STATUS,
    SCREEN_INDETERMINATE_STATUS,
)
FORMAL_STATUSES = (
    FORMAL_PASS_STATUS,
    FORMAL_FAIL_STATUS,
    FORMAL_INDETERMINATE_STATUS,
)
SCREEN_SEMANTICS = {
    SCREEN_PASS_STATUS: (0, True, True, True, True),
    SCREEN_FAIL_STATUS: (1, False, True, True, False),
    SCREEN_INDETERMINATE_STATUS: (2, False, False, False, False),
}
FORMAL_SEMANTICS = {
    FORMAL_PASS_STATUS: (0, True, True, True),
    FORMAL_FAIL_STATUS: (1, False, True, True),
    FORMAL_INDETERMINATE_STATUS: (2, False, False, False),
}
CLAIM_BOUNDARY = {
    "evidence_status": EVIDENCE_STATUS,
    "clean_ood": "NOT_ESTABLISHED",
    "external_replication": "NOT_ESTABLISHED",
    "sota": "NOT_ESTABLISHED",
    "publication_ready": False,
}

METRICS = tuple(METRIC_FIELDS)
COMPARATORS = ("matched_null", "feature41", "historical_parent")
SCREEN_ENGINEERING_GATE_NAMES = (
    "prediction_and_score_integrity",
    "exact_fold_count_20",
    "registered_prediction_coverage_eq_1",
    "failed_rows_eq_0",
    "duplicate_or_unexpected_artifacts_eq_0",
)
SCREEN_GATE_NAMES = {
    *SCREEN_ENGINEERING_GATE_NAMES,
    *(
        f"{metric}_gain_vs_{comparator}_ge_frozen_minimum"
        for metric in METRICS
        for comparator in COMPARATORS
    ),
    "matched_null_ci_lower_each_gt_zero",
    "matched_null_positive_puzzles_each_ge_14",
    "historical_parent_ci_lower_each_gt_zero",
    "historical_parent_positive_puzzles_each_ge_14",
    "candidate_coverage95_in_frozen_interval",
    "max_single_puzzle_effect_fraction_all_comparisons_le_0_20",
}
SCREEN_COMPARISON_NAMES = {
    f"{metric}_vs_{comparator}"
    for metric in METRICS
    for comparator in COMPARATORS
}
PAIRED_SUMMARY_FIELDS = {
    "comparator_field",
    "candidate_field",
    "comparator_mean",
    "candidate_mean",
    "mean_gain",
    "relative_gain",
    "ci95",
    "positive_puzzles",
    "per_puzzle",
    "leave_one_puzzle_out",
    "leave_one_puzzle_out_all_positive",
    "max_single_puzzle_effect_fraction",
}
SCREEN_COMPLETE_QUALIFICATION_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "gate_passed",
    "integrity_passed",
    "integrity_errors",
    "gates",
    "comparisons",
    "calibration",
    "frozen_gate_values",
    "rnd6_authorized",
    "model_or_threshold_selection_performed",
    "partial_fold_scores_inspected",
    "new_external_outcome_accessed",
    "evidence_status",
    "clean_ood",
    "external_replication",
    "sota",
    "publication_ready",
}
SCREEN_INDETERMINATE_QUALIFICATION_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "gate_passed",
    "integrity_passed",
    "integrity_errors",
    "gates",
    "comparisons",
    "calibration",
    "rnd6_authorized",
    "evidence_status",
    "clean_ood",
    "external_replication",
    "sota",
    "publication_ready",
}
FORMAL_GATE_NAMES = {
    "screen_prerequisite_exact_pass",
    "equal_seed_mixture_exact",
    "no_best_seed_selection",
    "mixture_repeats_every_frozen_screen_gate",
    *(
        f"{metric}_positive_individual_seeds_ge_"
        f"{FROZEN_FORMAL_GATES['individual_seed_positive_vs_matched_null_minimum'][metric]}"
        for metric in METRICS
    ),
}
FORMAL_COMPLETE_QUALIFICATION_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "gate_passed",
    "integrity_passed",
    "integrity_errors",
    "gates",
    "mixture_gates",
    "mixture_comparisons",
    "mixture_calibration",
    "individual_seed_directions",
    "positive_seed_counts",
    "frozen_screen_gate_values",
    "frozen_formal_gate_values",
    "screen_prerequisite_status",
    "equal_seed_mixture",
    "equal_seed_weight",
    "best_seed_selection_performed",
    "model_or_threshold_selection_performed",
    "partial_fold_scores_inspected",
    "partial_seed_scores_inspected",
    "new_external_outcome_accessed",
    "evidence_status",
    "clean_ood",
    "external_replication",
    "sota",
    "publication_ready",
    "terminal_closure_required",
}
FORMAL_INDETERMINATE_QUALIFICATION_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "gate_passed",
    "integrity_passed",
    "integrity_errors",
    "gates",
    "mixture_gates",
    "mixture_comparisons",
    "mixture_calibration",
    "individual_seed_directions",
    "positive_seed_counts",
    "evidence_status",
    "clean_ood",
    "external_replication",
    "sota",
    "publication_ready",
    "terminal_closure_required",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected canonical JSON object: {path}")
    return value


def _load_research(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise RuntimeError("research record frontmatter is missing")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError("research record frontmatter is unterminated")
    frontmatter = yaml.safe_load(text[4:end])
    if not isinstance(frontmatter, dict):
        raise RuntimeError("research record frontmatter must be a mapping")
    return frontmatter, text[end + 5 :]


def _dump_yaml(value: dict[str, Any]) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _dump_research(frontmatter: dict[str, Any], body: str) -> str:
    return "---\n" + _dump_yaml(frontmatter) + "---\n" + body


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.finalizing.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validated_recorded_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("--recorded-at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise RuntimeError("--recorded-at must include an explicit UTC offset")
    return value


def _resolved(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} canonical path is missing")
    return Path(value).expanduser().resolve()


def _require_active_path(
    active: dict[str, Any], name: str, expected: Path, *, label: str
) -> None:
    observed = _resolved(active.get("authority", {}).get(name), label=label)
    _require(observed == expected.resolve(), f"{label} differs from canonical path")


def _require_claim_boundary(qualification: dict[str, Any], *, label: str) -> None:
    for name, expected in CLAIM_BOUNDARY.items():
        _require(
            _json_exact(qualification.get(name), expected),
            f"{label} claim boundary changed: {name}",
        )


def _validate_merge(
    merged: dict[str, Any], *, phase: str, folds: list[int], seeds: list[int]
) -> dict[str, Any]:
    _require(set(merged) == set(EXPECTED_MERGED_FIELDS), f"{phase} merge fields changed")
    _require(merged.get("schema_version") == MERGE_SCHEMA, f"{phase} merge schema changed")
    _require(merged.get("phase") == phase, f"{phase} merge phase changed")
    _require(merged.get("status") == MERGE_STATUS[phase], f"{phase} merge status changed")
    _require(merged.get("merge_integrity") == MERGE_INTEGRITY, f"{phase} merge integrity changed")
    rows = merged.get("folds")
    _require(isinstance(rows, list), f"{phase} merge rows are malformed")
    expected_pairs = [(fold, seed) for seed in seeds for fold in folds]
    observed_pairs: list[tuple[int, int]] = []
    commits: set[str] = set()
    devices: set[str] = set()
    gpu_names: set[str] = set()
    starts: list[str] = []
    finishes: list[str] = []
    for row in rows:
        _require(isinstance(row, dict), f"{phase} merge row is malformed")
        _require(set(row) == set(EXPECTED_FOLD_FIELDS), f"{phase} merge row fields changed")
        fold = int(row.get("outer_fold"))
        seed = int(row.get("seed"))
        observed_pairs.append((fold, seed))
        _require(row.get("phase") == phase, f"{phase} fold phase changed")
        _require(
            row.get("experiment_id") == EXPECTED_EXPERIMENT_ID[phase],
            f"{phase} experiment id changed",
        )
        _require(
            row.get("evidence_status") == PREDICTION_EVIDENCE_STATUS[phase]
            and row.get("metric_eligibility") == PREDICTION_EVIDENCE_STATUS[phase],
            f"{phase} prediction evidence class changed",
        )
        point_epochs, calibration_epochs = EXPECTED_SCHEDULE[phase]
        _require(
            int(row.get("point_epochs")) == point_epochs
            and int(row.get("calibration_epochs")) == calibration_epochs,
            f"{phase} schedule changed",
        )
        device = row.get("training_device")
        _require(device == "cuda:0", f"{phase} training device is not exact cuda:0")
        gpu_name = row.get("gpu_name")
        _require(isinstance(gpu_name, str) and bool(gpu_name), f"{phase} GPU name is missing")
        command = row.get("command")
        _require(
            isinstance(command, list)
            and bool(command)
            and all(isinstance(item, str) and bool(item) for item in command),
            f"{phase} exact per-fold runner command is missing",
        )
        invariants = row.get("invariants")
        _require(
            isinstance(invariants, dict)
            and invariants.get("cuda_only_training") is True
            and invariants.get("held_target_read") is False
            and invariants.get("held_score_computed") is False
            and invariants.get("partial_score_inspected") is False
            and invariants.get("external_outcome_accessed") is False,
            f"{phase} outcome or CUDA invariants changed",
        )
        _require(int(row.get("exit_code")) == 0, f"{phase} fold did not exit zero")
        commit = str(row.get("git_commit", "")).lower()
        _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, f"{phase} git commit is invalid")
        started = row.get("started_at_utc")
        finished = row.get("finished_at_utc")
        _require(isinstance(started, str) and bool(started), f"{phase} start time is missing")
        _require(isinstance(finished, str) and bool(finished), f"{phase} finish time is missing")
        commits.add(commit)
        devices.add(device)
        gpu_names.add(gpu_name)
        starts.append(started)
        finishes.append(finished)
    _require(sorted(observed_pairs) == sorted(expected_pairs), f"{phase} fold-seed universe changed")
    _require(len(commits) == 1, f"{phase} source commit is not exact and single")
    return {
        "experiment_id": EXPECTED_EXPERIMENT_ID[phase],
        "authority_branch": authority.BRANCH,
        "folds": folds,
        "seeds": seeds,
        "point_epochs": EXPECTED_SCHEDULE[phase][0],
        "calibration_epochs": EXPECTED_SCHEDULE[phase][1],
        "training_devices": sorted(devices),
        "gpu_names": sorted(gpu_names),
        "started_at_utc": min(starts),
        "finished_at_utc": max(finishes),
        "source_commits": sorted(commits),
    }


def _validate_screen_score(score: dict[str, Any]) -> None:
    _require(set(score) == set(SCORE_TOP_FIELDS), "RND4 score fields changed")
    expected = {
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
        "independent_units": "20_PUZZLES",
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_exposure_status": EVIDENCE_STATUS,
    }
    for name, value in expected.items():
        _require(score.get(name) == value, f"RND4 score invariant changed: {name}")
    rows = score.get("scores")
    _require(
        isinstance(rows, list)
        and sorted(int(row.get("outer_fold")) for row in rows) == list(range(20)),
        "RND4 score fold universe changed",
    )


def _validate_formal_score(score: dict[str, Any]) -> None:
    _require(set(score) == set(FORMAL_SCORE_TOP_FIELDS), "RND6S score fields changed")
    expected = {
        "schema_version": FORMAL_SCORE_SCHEMA,
        "phase": FORMAL_SCORE_PHASE,
        "status": FORMAL_SCORE_STATUS,
        "integrity_errors": [],
        "complete_valid_score": True,
        "complete_source_fold_seed_universe": True,
        "complete_assembly_fold_universe": True,
        "expected_fold_seed_count": 100,
        "actual_fold_seed_count": 100,
        "expected_fold_count": 20,
        "actual_fold_count": 20,
        "expected_seed_count": 5,
        "actual_seed_count": 5,
        "failed_rows": 0,
        "duplicate_or_unexpected_artifacts": 0,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge_and_assembly": True,
        "independent_units": "20_PUZZLES_NOT_100_FOLD_SEEDS",
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "partial_seed_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_exposure_status": EVIDENCE_STATUS,
    }
    for name, value in expected.items():
        _require(score.get(name) == value, f"RND6S score invariant changed: {name}")
    mixture = score.get("mixture_scores")
    individual = score.get("individual_seed_scores")
    _require(
        isinstance(mixture, list)
        and sorted(int(row.get("outer_fold")) for row in mixture) == list(range(20)),
        "RND6S mixture fold universe changed",
    )
    _require(
        isinstance(individual, dict) and set(individual) == {str(seed) for seed in range(5)},
        "RND6S individual seed universe changed",
    )
    for seed in range(5):
        rows = individual[str(seed)]
        _require(
            isinstance(rows, list)
            and sorted(int(row.get("outer_fold")) for row in rows) == list(range(20)),
            f"RND6S seed {seed} fold universe changed",
        )


def _is_float(value: object) -> bool:
    return type(value) is float


def _json_exact(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _json_exact(observed[name], expected[name]) for name in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_exact(left, right) for left, right in zip(observed, expected)
        )
    return observed == expected


def _validate_bool_map(
    value: object, *, expected_names: set[str], label: str
) -> dict[str, bool]:
    _require(isinstance(value, dict), f"{label} is malformed")
    _require(set(value) == expected_names, f"{label} name universe changed")
    _require(
        all(isinstance(item, bool) for item in value.values()),
        f"{label} values are not booleans",
    )
    return value


def _validate_comparisons(value: object, *, label: str) -> None:
    _require(isinstance(value, dict), f"{label} comparisons are malformed")
    _require(
        set(value) == SCREEN_COMPARISON_NAMES,
        f"{label} comparison universe changed",
    )
    for metric, fields in METRIC_FIELDS.items():
        for comparator in COMPARATORS:
            name = f"{metric}_vs_{comparator}"
            row = value[name]
            _require(isinstance(row, dict), f"{label} comparison {name} is malformed")
            _require(
                set(row) == PAIRED_SUMMARY_FIELDS,
                f"{label} comparison {name} fields changed",
            )
            _require(
                row["comparator_field"] == fields[comparator]
                and row["candidate_field"] == fields["candidate"],
                f"{label} comparison {name} metric fields changed",
            )
            for field in (
                "comparator_mean",
                "candidate_mean",
                "mean_gain",
                "relative_gain",
            ):
                _require(
                    _is_float(row[field]),
                    f"{label} comparison {name} {field} is malformed",
                )
            _require(
                _is_float(row["max_single_puzzle_effect_fraction"])
                and row["max_single_puzzle_effect_fraction"] >= 0.0,
                f"{label} comparison {name} single-puzzle fraction is malformed",
            )
            positive = row["positive_puzzles"]
            _require(
                isinstance(positive, int)
                and not isinstance(positive, bool)
                and 0 <= positive <= 20,
                f"{label} comparison {name} positive-puzzle count is malformed",
            )
            for field, length in (
                ("ci95", 2),
                ("per_puzzle", 20),
                ("leave_one_puzzle_out", 20),
            ):
                values = row[field]
                _require(
                    isinstance(values, list)
                    and len(values) == length
                    and all(_is_float(item) for item in values),
                    f"{label} comparison {name} {field} is malformed",
                )
            _require(
                isinstance(row["leave_one_puzzle_out_all_positive"], bool),
                f"{label} comparison {name} leave-one-out flag is malformed",
            )


def _validate_calibration(
    value: object, *, gates: dict[str, bool], label: str
) -> None:
    _require(
        isinstance(value, dict) and set(value) == {"coverage95"},
        f"{label} calibration fields changed",
    )
    coverage = value["coverage95"]
    _require(
        isinstance(coverage, dict)
        and set(coverage) == {"candidate", "frozen_interval", "within_interval"},
        f"{label} coverage calibration fields changed",
    )
    _require(
        _is_float(coverage["candidate"])
        and 0.0 <= coverage["candidate"] <= 1.0,
        f"{label} coverage calibration candidate is malformed",
    )
    _require(
        _json_exact(
            coverage["frozen_interval"], FROZEN_SCREEN_GATES["coverage_95_interval"]
        ),
        f"{label} frozen calibration interval changed",
    )
    _require(
        isinstance(coverage["within_interval"], bool)
        and coverage["within_interval"]
        is gates["candidate_coverage95_in_frozen_interval"],
        f"{label} coverage calibration Gate binding changed",
    )


def _validate_screen_complete_qualification(
    qualification: dict[str, Any], *, label: str
) -> dict[str, bool]:
    _require(
        set(qualification) == SCREEN_COMPLETE_QUALIFICATION_FIELDS,
        f"{label} complete qualification fields changed",
    )
    gates = _validate_bool_map(
        qualification["gates"], expected_names=SCREEN_GATE_NAMES, label=f"{label} Gates"
    )
    _require(
        all(gates[name] for name in SCREEN_ENGINEERING_GATE_NAMES),
        f"{label} complete-result engineering Gates changed",
    )
    _validate_comparisons(qualification["comparisons"], label=label)
    _validate_calibration(qualification["calibration"], gates=gates, label=label)
    _require(
        _json_exact(qualification["frozen_gate_values"], FROZEN_SCREEN_GATES),
        f"{label} frozen Gate values changed",
    )
    for name in (
        "model_or_threshold_selection_performed",
        "partial_fold_scores_inspected",
        "new_external_outcome_accessed",
    ):
        _require(qualification[name] is False, f"{label} selection/access flag changed: {name}")
    return gates


def _validate_formal_complete_qualification(
    qualification: dict[str, Any], *, label: str
) -> dict[str, bool]:
    _require(
        set(qualification) == FORMAL_COMPLETE_QUALIFICATION_FIELDS,
        f"{label} complete qualification fields changed",
    )
    gates = _validate_bool_map(
        qualification["gates"], expected_names=FORMAL_GATE_NAMES, label=f"{label} Gates"
    )
    mixture_gates = _validate_bool_map(
        qualification["mixture_gates"],
        expected_names=SCREEN_GATE_NAMES,
        label=f"{label} mixture Gates",
    )
    _require(
        all(mixture_gates[name] for name in SCREEN_ENGINEERING_GATE_NAMES),
        f"{label} mixture complete-result engineering Gates changed",
    )
    _validate_comparisons(qualification["mixture_comparisons"], label=f"{label} mixture")
    _validate_calibration(
        qualification["mixture_calibration"],
        gates=mixture_gates,
        label=f"{label} mixture",
    )
    _require(
        _json_exact(
            qualification["frozen_screen_gate_values"], FROZEN_SCREEN_GATES
        )
        and _json_exact(
            qualification["frozen_formal_gate_values"], FROZEN_FORMAL_GATES
        ),
        f"{label} frozen Gate values changed",
    )
    _require(
        qualification["screen_prerequisite_status"] == SCREEN_PASS_STATUS
        and qualification["equal_seed_mixture"] is True
        and qualification["equal_seed_weight"] == 0.2
        and qualification["best_seed_selection_performed"] is False
        and qualification["terminal_closure_required"] is True,
        f"{label} formal-chain invariants changed",
    )
    for name in (
        "model_or_threshold_selection_performed",
        "partial_fold_scores_inspected",
        "partial_seed_scores_inspected",
        "new_external_outcome_accessed",
    ):
        _require(qualification[name] is False, f"{label} selection/access flag changed: {name}")

    directions = qualification["individual_seed_directions"]
    _require(
        isinstance(directions, dict)
        and set(directions) == {str(seed) for seed in range(5)},
        f"{label} individual-seed universe changed",
    )
    observed_positive = {metric: 0 for metric in METRICS}
    for seed in range(5):
        seed_directions = directions[str(seed)]
        _require(
            isinstance(seed_directions, dict) and set(seed_directions) == set(METRICS),
            f"{label} seed {seed} metric universe changed",
        )
        for metric in METRICS:
            result = seed_directions[metric]
            _require(
                isinstance(result, dict)
                and set(result)
                == {
                    "mean_gain_vs_matched_null",
                    "relative_gain_vs_matched_null",
                    "positive",
                },
                f"{label} seed {seed} {metric} direction fields changed",
            )
            _require(
                _is_float(result["mean_gain_vs_matched_null"])
                and _is_float(result["relative_gain_vs_matched_null"])
                and isinstance(result["positive"], bool),
                f"{label} seed {seed} {metric} direction is malformed",
            )
            _require(
                result["positive"] is (result["mean_gain_vs_matched_null"] > 0.0),
                f"{label} seed {seed} {metric} positive direction changed",
            )
            observed_positive[metric] += int(result["positive"])

    counts = qualification["positive_seed_counts"]
    _require(
        isinstance(counts, dict) and set(counts) == set(METRICS),
        f"{label} positive-seed metric universe changed",
    )
    _require(
        all(
            isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= 5
            for count in counts.values()
        )
        and counts == observed_positive,
        f"{label} positive-seed counts changed",
    )
    _require(
        gates["screen_prerequisite_exact_pass"] is True
        and gates["equal_seed_mixture_exact"] is True
        and gates["no_best_seed_selection"] is True
        and gates["mixture_repeats_every_frozen_screen_gate"]
        is all(mixture_gates.values()),
        f"{label} formal Gate bindings changed",
    )
    minima = FROZEN_FORMAL_GATES["individual_seed_positive_vs_matched_null_minimum"]
    for metric in METRICS:
        name = f"{metric}_positive_individual_seeds_ge_{minima[metric]}"
        _require(
            gates[name] is (counts[metric] >= minima[metric]),
            f"{label} {metric} positive-seed Gate binding changed",
        )
    return gates


def _validate_qualification(
    qualification: dict[str, Any], *, formal: bool
) -> tuple[str, list[str], list[str]]:
    statuses = FORMAL_STATUSES if formal else SCREEN_STATUSES
    semantics = FORMAL_SEMANTICS if formal else SCREEN_SEMANTICS
    expected_schema = FORMAL_QUALIFICATION_SCHEMA if formal else SCREEN_QUALIFICATION_SCHEMA
    expected_phase = FORMAL_QUALIFICATION_PHASE if formal else "RND5"
    label = "RND6Q" if formal else "RND5"
    _require(qualification.get("schema_version") == expected_schema, f"{label} schema changed")
    _require(qualification.get("phase") == expected_phase, f"{label} phase changed")
    status = qualification.get("status")
    _require(status in statuses, f"{label} status is not canonical")
    _require_claim_boundary(qualification, label=label)
    expected = semantics[status]
    gate_passed, integrity_passed = expected[1], expected[2]
    _require(qualification.get("gate_passed") is gate_passed, f"{label} Gate semantics changed")
    _require(
        qualification.get("integrity_passed") is integrity_passed,
        f"{label} integrity semantics changed",
    )
    if not formal:
        _require(
            qualification.get("rnd6_authorized") is expected[4],
            "RND5 RND6 authorization semantics changed",
        )
    if formal:
        _require(
            qualification.get("terminal_closure_required") is True,
            f"{label} terminal closure semantics changed",
        )

    indeterminate = status in {
        SCREEN_INDETERMINATE_STATUS,
        FORMAL_INDETERMINATE_STATUS,
    }
    if indeterminate:
        expected_fields = (
            FORMAL_INDETERMINATE_QUALIFICATION_FIELDS
            if formal
            else SCREEN_INDETERMINATE_QUALIFICATION_FIELDS
        )
        _require(set(qualification) == expected_fields, f"{label} indeterminate fields changed")
        empty_names = (
            (
                "gates",
                "mixture_gates",
                "mixture_comparisons",
                "mixture_calibration",
                "individual_seed_directions",
                "positive_seed_counts",
            )
            if formal
            else ("gates", "comparisons", "calibration")
        )
        _require(
            all(qualification[name] == {} for name in empty_names),
            f"{label} indeterminate payload is not empty",
        )
        gates: dict[str, bool] = {}
    elif formal:
        gates = _validate_formal_complete_qualification(qualification, label=label)
    else:
        gates = _validate_screen_complete_qualification(qualification, label=label)

    errors = qualification.get("integrity_errors")
    _require(
        isinstance(errors, list) and all(isinstance(value, str) and value for value in errors),
        f"{label} integrity errors are malformed",
    )
    integrity_errors = sorted(set(errors))
    _require(
        errors == integrity_errors,
        f"{label} integrity errors are not canonical sorted unique",
    )
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    if integrity_passed:
        _require(not integrity_errors, f"{label} valid qualification has integrity errors")
        _require(
            qualification["gate_passed"] is all(gates.values()),
            f"{label} overall Gate binding changed",
        )
    else:
        _require(bool(integrity_errors), f"{label} indeterminate lacks integrity reason")
    if status.endswith("_PASS"):
        _require(bool(gates) and not failed_gates, f"{label} PASS Gate universe is not exact PASS")
    elif status.endswith("_FAIL"):
        _require(bool(gates) and bool(failed_gates), f"{label} FAIL lacks failed Gates")
    return str(status), failed_gates, integrity_errors


def _validate_assembly(assembly: dict[str, Any]) -> None:
    _require(set(assembly) == set(EXPECTED_ASSEMBLY_FIELDS), "RND6P assembly fields changed")
    expected = {
        "schema_version": ASSEMBLY_SCHEMA,
        "phase": "RND6P",
        "status": ASSEMBLY_STATUS,
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "score_computed": False,
        "target_accessed": False,
        "external_outcome_accessed": False,
    }
    for name, value in expected.items():
        _require(assembly.get(name) == value, f"RND6P assembly invariant changed: {name}")
    rows = assembly.get("folds")
    _require(isinstance(rows, list), "RND6P assembly folds are malformed")
    observed: list[int] = []
    for row in rows:
        _require(
            isinstance(row, dict) and set(row) == set(EXPECTED_FOLD_MANIFEST_FIELDS),
            "RND6P assembly fold fields changed",
        )
        observed.append(int(row.get("outer_fold")))
        _require(row.get("seeds") == list(range(5)), "RND6P assembly seed universe changed")
        _require(int(row.get("n_registered_prediction_rows")) > 0, "RND6P assembly fold is empty")
    _require(sorted(observed) == list(range(20)), "RND6P assembly fold universe changed")


def _screen_bundle(active: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name, path in (
        ("complete_unscored_merge_path", SCREEN_MERGE_PATH),
        ("complete_score_path", SCREEN_SCORE_PATH),
        ("qualification_path", SCREEN_QUALIFICATION_PATH),
    ):
        _require_active_path(active, name, path, label=f"screen {name}")
        _require(path.is_file(), f"canonical screen artifact is missing: {path}")
    merged = _load_json(SCREEN_MERGE_PATH)
    score = _load_json(SCREEN_SCORE_PATH)
    qualification = _load_json(SCREEN_QUALIFICATION_PATH)
    provenance = _validate_merge(merged, phase="RND3", folds=list(range(20)), seeds=[0])
    _validate_screen_score(score)
    _validate_qualification(qualification, formal=False)
    return merged, score, qualification, provenance


def _formal_bundle(active: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name, path in (
        ("formal_complete_unscored_merge_path", FORMAL_MERGE_PATH),
        ("formal_assembly_path", FORMAL_ASSEMBLY_PATH),
        ("formal_complete_score_path", FORMAL_SCORE_PATH),
        ("formal_qualification_path", FORMAL_QUALIFICATION_PATH),
        ("screen_qualification_path", SCREEN_QUALIFICATION_PATH),
    ):
        _require_active_path(active, name, path, label=f"formal {name}")
        _require(path.is_file(), f"canonical formal artifact is missing: {path}")
    merged = _load_json(FORMAL_MERGE_PATH)
    assembly = _load_json(FORMAL_ASSEMBLY_PATH)
    score = _load_json(FORMAL_SCORE_PATH)
    qualification = _load_json(FORMAL_QUALIFICATION_PATH)
    screen_qualification = _load_json(SCREEN_QUALIFICATION_PATH)
    provenance = _validate_merge(
        merged, phase="RND6P", folds=list(range(20)), seeds=list(range(5))
    )
    _validate_assembly(assembly)
    _validate_formal_score(score)
    _validate_qualification(screen_qualification, formal=False)
    _require(
        screen_qualification.get("status") == SCREEN_PASS_STATUS,
        "formal result lacks exact RND5 PASS prerequisite",
    )
    _validate_qualification(qualification, formal=True)
    return merged, assembly, score, qualification, screen_qualification, provenance


def _registry_entry(
    *,
    phase: str,
    status: str,
    report_path: Path,
    merge_path: Path,
    score_path: Path,
    qualification_path: Path,
    provenance: dict[str, Any],
    qualification: dict[str, Any],
    recorded_at: str,
    finalizer_source_commit: str,
    assembly_path: Path | None = None,
) -> dict[str, Any]:
    _, failed_gates, integrity_errors = _validate_qualification(
        qualification, formal=phase == "RND6Q"
    )
    entry: dict[str, Any] = {
        "phase": phase,
        "status": status,
        "recorded_at": recorded_at,
        "report_path": str(report_path),
        "report_exists": True,
        "canonical_merge_path": str(merge_path),
        "canonical_score_path": str(score_path),
        "canonical_qualification_path": str(qualification_path),
        **provenance,
        "gate_passed": bool(qualification["gate_passed"]),
        "integrity_passed": bool(qualification["integrity_passed"]),
        "failed_gates": failed_gates,
        "integrity_errors": integrity_errors,
        **CLAIM_BOUNDARY,
        "finalizer_source_commit": finalizer_source_commit,
    }
    if assembly_path is not None:
        entry["canonical_assembly_path"] = str(assembly_path)
        entry["expected_fold_seed_pairs"] = 100
        entry["equal_seed_weight"] = 0.2
    return entry


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    return str(value).replace("|", "\\|")


def _comparison_table(comparisons: object, *, title: str) -> list[str]:
    if not isinstance(comparisons, dict) or not comparisons:
        return []
    lines = [
        f"## {title}",
        "",
        "| Comparison | Comparator mean | Candidate mean | Mean gain | Relative gain | 95% CI | Positive puzzles | Max single-puzzle fraction |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    fields = (
        "comparator_mean",
        "candidate_mean",
        "mean_gain",
        "relative_gain",
        "ci95",
        "positive_puzzles",
        "max_single_puzzle_effect_fraction",
    )
    for name in sorted(comparisons):
        row = comparisons[name]
        if not isinstance(row, dict) or not all(field in row for field in fields):
            continue
        values = " | ".join(_format_scalar(row[field]) for field in fields)
        lines.append(f"| {_format_scalar(name)} | {values} |")
    lines.append("")
    return lines


def _calibration_section(calibration: object) -> list[str]:
    payload = calibration if isinstance(calibration, dict) else {}
    return [
        "## Canonical calibration",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
    ]


def _render_report(
    *,
    kind: str,
    qualification: dict[str, Any],
    registry: dict[str, Any],
    canonical_paths: list[tuple[str, Path]],
) -> str:
    status = str(qualification["status"])
    gates: dict[str, bool] = dict(qualification.get("gates", {}))
    if kind == "formal":
        mixture_gates = qualification.get("mixture_gates", {})
        if isinstance(mixture_gates, dict):
            gates.update({f"mixture.{name}": bool(value) for name, value in mixture_gates.items()})
    failed = sorted(name for name, passed in gates.items() if not passed)
    integrity_errors = sorted(set(qualification.get("integrity_errors", [])))
    title = (
        "Independent RNet distillation screen result"
        if kind == "screen"
        else "Independent RNet distillation five-seed formal result"
    )
    lines = [
        f"# {title}",
        "",
        f"- Qualification status: `{status}`",
        f"- Gate passed: `{str(bool(qualification['gate_passed'])).lower()}`",
        f"- Integrity passed: `{str(bool(qualification['integrity_passed'])).lower()}`",
        f"- Evidence ceiling: `{EVIDENCE_STATUS}`",
        "- Publication ready: `false`",
        f"- Recorded at: `{registry['recorded_at']}`",
        "",
        "## Provenance and fixed universe",
        "",
        f"- Experiment ID: `{registry['experiment_id']}`",
        f"- Authority branch: `{registry['authority_branch']}`",
        f"- Finalizer source commit: `{registry['finalizer_source_commit']}`",
        f"- Source run commits: `{', '.join(registry['source_commits'])}`",
        f"- Fold universe: `{registry['folds']}`",
        f"- Seed universe: `{registry['seeds']}`",
        f"- Downstream schedule: `{registry['point_epochs']}+{registry['calibration_epochs']}` point/calibration epochs",
        f"- Training devices recorded by canonical fold results: `{registry['training_devices']}`",
        f"- GPU names recorded by canonical fold results: `{registry['gpu_names']}`",
        f"- Earliest fold start: `{registry['started_at_utc']}`",
        f"- Latest fold finish: `{registry['finished_at_utc']}`",
    ]
    if kind == "formal":
        lines.extend(
            [
                f"- Exact fold-seed pairs: `{registry['expected_fold_seed_pairs']}`",
                f"- Equal seed weight: `{registry['equal_seed_weight']}`",
            ]
        )
    lines.extend(["", "## Canonical records", ""])
    lines.extend(f"- {label}: `{path}`" for label, path in canonical_paths)
    lines.append(
        "- Exact per-fold runner commands: recorded in the canonical merge "
        f"`{registry['canonical_merge_path']}` under `folds[*].command`; "
        "not duplicated into the decision ledger."
    )
    lines.extend(["", "## Frozen Gate record", ""])
    if gates:
        lines.extend(["| Gate | Result |", "| --- | --- |"])
        lines.extend(
            f"| {_format_scalar(name)} | {'PASS' if passed else 'FAIL'} |"
            for name, passed in sorted(gates.items())
        )
    else:
        lines.append("No Gate booleans were emitted because the canonical qualification is engineering-indeterminate.")
    lines.extend(["", "## Failure or indeterminacy reasons", ""])
    if failed:
        lines.extend(f"- Failed Gate: `{name}`" for name in failed)
    if integrity_errors:
        lines.extend(f"- Integrity error: `{name}`" for name in integrity_errors)
    if not failed and not integrity_errors:
        lines.append("- None recorded in the canonical qualification.")
    comparisons = (
        qualification.get("comparisons")
        if kind == "screen"
        else qualification.get("mixture_comparisons")
    )
    lines.extend([""])
    lines.extend(_comparison_table(comparisons, title="Canonical paired summaries"))
    calibration = (
        qualification.get("calibration")
        if kind == "screen"
        else qualification.get("mixture_calibration")
    )
    lines.extend(_calibration_section(calibration))
    if kind == "formal" and isinstance(qualification.get("positive_seed_counts"), dict):
        lines.extend(["## Individual-seed direction stability", ""])
        for metric, count in sorted(qualification["positive_seed_counts"].items()):
            lines.append(f"- `{metric}`: `{count}/5` seeds positive versus matched null")
        lines.append("")
    lines.extend(["## Claim boundary", "", "Allowed claims:", ""])
    if kind == "screen":
        if status == SCREEN_PASS_STATUS:
            lines.append("- The fixed RND5 development screen passed and authorizes only the frozen RND6 formal chain.")
        elif status == SCREEN_FAIL_STATUS:
            lines.append("- The complete fixed RND5 development screen did not pass; RND6 formal confirmation was not run.")
        else:
            lines.append("- RND5 is engineering-indeterminate; no scientific pass or fail is established and RND6 was not run.")
    else:
        if status == FORMAL_PASS_STATUS:
            lines.append("- The fixed equal-weight five-seed development formal Gate passed.")
        elif status == FORMAL_FAIL_STATUS:
            lines.append("- The complete fixed equal-weight five-seed development formal Gate did not pass.")
        else:
            lines.append("- The formal qualification is engineering-indeterminate; no scientific pass or fail is established.")
    lines.extend(
        [
            f"- The result is `{EVIDENCE_STATUS}` on the disclosed, repeatedly consumed development benchmark.",
            "",
            "Prohibited claims:",
            "",
            "- Clean out-of-distribution evidence is not established.",
            "- Independent external replication is not established.",
            "- State of the art is not established.",
            "- Publication readiness is false.",
            "- Training loss, smoke output, prediction coverage, and engineering checks are not scientific conclusions.",
            "",
        ]
    )
    return "\n".join(lines)


def _set_phase(
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
    research: dict[str, Any],
    *,
    phase: str,
    recorded_at: str,
) -> None:
    token = authority.TOKENS[phase]
    active["updated_at"] = recorded_at
    pointer = active["authority"]
    pointer["current_phase"] = phase
    pointer["current_runnable_phase"] = "NONE" if phase in authority.TERMINAL_PHASES else phase
    pointer["current_authority_state"] = token
    pointer["binding_status"] = token
    contract["contract_status"] = token
    ledger["current_phase"] = phase
    ledger["current_status"] = token
    research["status"] = token
    active["runnable_phases"] = [] if phase in authority.TERMINAL_PHASES else [phase]

    lifecycle, activation_allowed = authority._formal_lifecycle(phase)
    contract["formal_chain"] = authority._expected_contract_formal_chain(phase)
    active["inactive_formal_chain"] = authority._expected_active_formal_chain(phase)
    ledger["formal_chain_status"] = lifecycle
    research["formal_chain_status"] = lifecycle
    research["formal_activation_allowed"] = activation_allowed
    active["formal_output_state"] = copy.deepcopy(
        authority.FORMAL_OUTPUT_STATE_BY_PHASE[phase]
        if phase in authority.FORMAL_PHASES
        else authority.INACTIVE_FORMAL_OUTPUT_STATE
    )
    formal_score_accessed = phase in {"RND6Q", "RND6T"}
    formal_qualification_accessed = phase == "RND6T"
    ledger["formal_score_accessed"] = formal_score_accessed
    ledger["formal_qualification_accessed"] = formal_qualification_accessed
    research["formal_score_accessed"] = formal_score_accessed
    research["formal_qualification_accessed"] = formal_qualification_accessed


def _append_research_record(body: str, record: str) -> str:
    marker = (
        "The production finalizer recorded canonical RND6Q status"
        if "canonical RND6Q status" in record
        else "The production finalizer recorded canonical RND5 status"
    )
    matching = [line for line in body.splitlines() if marker in line]
    if matching:
        _require(matching == [record], "research finalization record already differs")
        return body
    if not body.endswith("\n"):
        body += "\n"
    return body + record + "\n"


def _screen_transition(
    *,
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
    research: dict[str, Any],
    research_body: str,
    qualification: dict[str, Any],
    provenance: dict[str, Any],
    recorded_at: str,
    finalizer_source_commit: str,
) -> tuple[str, str]:
    status, _, _ = _validate_qualification(qualification, formal=False)
    screen_registry = _registry_entry(
        phase="RND5",
        status=status,
        report_path=SCREEN_REPORT_PATH,
        merge_path=SCREEN_MERGE_PATH,
        score_path=SCREEN_SCORE_PATH,
        qualification_path=SCREEN_QUALIFICATION_PATH,
        provenance=provenance,
        qualification=qualification,
        recorded_at=recorded_at,
        finalizer_source_commit=finalizer_source_commit,
    )
    if status == SCREEN_PASS_STATUS:
        next_phase = "RND6P"
        formal_registry = copy.deepcopy(authority._PENDING_FORMAL_REGISTRY)
    else:
        next_phase = "RND5T"
        formal_registry = {
            "status": authority.FORMAL_NOT_RUN_STATUS,
            "reason": status,
            "report_path": str(FORMAL_REPORT_PATH),
            "report_exists": False,
            "publication_ready": False,
        }
    registry = {"screen": screen_registry, "formal": formal_registry}
    existing_registry = ledger.get("result_registry")
    _require(existing_registry in ({}, registry), "canonical result registry already differs")
    ledger["result_registry"] = registry
    _set_phase(
        active,
        contract,
        ledger,
        research,
        phase=next_phase,
        recorded_at=recorded_at,
    )
    ledger["score_accessed"] = True
    research["screen_result_status"] = status
    research["formal_result_status"] = (
        authority.FORMAL_PENDING_STATUS
        if next_phase == "RND6P"
        else authority.FORMAL_NOT_RUN_STATUS
    )
    research["publication_ready"] = False

    pointer = active["authority"]
    if next_phase == "RND6P":
        active["authorization"].update(authority.RND6P_AUTHORIZATION)
        active["training_allowed"] = True
        active["candidate_model_training_allowed"] = True
        active["held_score_read_allowed"] = False
        active["gate_state"] = copy.deepcopy(authority.RND6P_GATE_STATE)
        active["next_allowed_action"] = authority.RND6P_ACTION
        ledger["next_action"] = authority.RND6P_ACTION
        for name, path in authority.RND6_CANONICAL_PATHS.items():
            pointer[name] = str(path)
        event = {
            "time": recorded_at,
            "event": status,
            "decision": authority.RND6P_DECISION,
            "canonical_qualification_path": str(SCREEN_QUALIFICATION_PATH),
            "canonical_qualification_status": status,
            "exit_code": 0,
            "gate_passed": True,
            "integrity_passed": True,
            "rnd6_authorized": True,
            "evidence_status": EVIDENCE_STATUS,
            "score_accessed": True,
            "formal_score_accessed": False,
            "formal_qualification_accessed": False,
            "partial_score_accessed": False,
            "new_external_outcome_accessed": False,
            "model_or_threshold_selection_performed": False,
            "canonical_screen_report_path": str(SCREEN_REPORT_PATH),
            "authority_token": authority.TOKENS["RND6P"],
        }
    else:
        active["authorization"].update(authority.RND5T_AUTHORIZATION)
        active["training_allowed"] = False
        active["candidate_model_training_allowed"] = False
        active["held_score_read_allowed"] = False
        active["gate_state"] = {
            **authority.RND5_GATE_STATE,
            "RND5": status,
            "RND5T": f"TERMINAL_{status}",
        }
        active["next_allowed_action"] = authority.RND5T_ACTION
        ledger["next_action"] = authority.RND5T_ACTION
        exit_code, gate_passed, integrity_passed, complete_valid, _ = SCREEN_SEMANTICS[status]
        event = {
            "time": recorded_at,
            "event": status,
            "decision": authority.RND5T_DECISION,
            "canonical_qualification_path": str(SCREEN_QUALIFICATION_PATH),
            "canonical_qualification_status": status,
            "exit_code": exit_code,
            "gate_passed": gate_passed,
            "integrity_passed": integrity_passed,
            "complete_valid_qualification": complete_valid,
            "rnd6_authorized": False,
            "formal_result_status": authority.FORMAL_NOT_RUN_STATUS,
            "score_accessed": True,
            "formal_score_accessed": False,
            "formal_qualification_accessed": False,
            "partial_score_accessed": False,
            "new_external_outcome_accessed": False,
            "model_or_threshold_selection_performed": False,
            "evidence_status": EVIDENCE_STATUS,
            "canonical_screen_report_path": str(SCREEN_REPORT_PATH),
            "authority_token": authority.TOKENS["RND5T"],
        }
    ledger["decisions"].append(event)
    record = (
        f"- {recorded_at}: The production finalizer recorded canonical RND5 status "
        f"`{status}` at `{SCREEN_REPORT_PATH}` and transitioned to `{next_phase}`. "
        f"Evidence remains `{EVIDENCE_STATUS}`; publication readiness is false."
    )
    research_body = _append_research_record(research_body, record)
    report = _render_report(
        kind="screen",
        qualification=qualification,
        registry=screen_registry,
        canonical_paths=[
            ("Complete target-free merge", SCREEN_MERGE_PATH),
            ("Complete score", SCREEN_SCORE_PATH),
            ("Qualification", SCREEN_QUALIFICATION_PATH),
        ],
    )
    return report, research_body


def _formal_transition(
    *,
    active: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
    research: dict[str, Any],
    research_body: str,
    qualification: dict[str, Any],
    provenance: dict[str, Any],
    recorded_at: str,
    finalizer_source_commit: str,
) -> tuple[str, str]:
    status, _, _ = _validate_qualification(qualification, formal=True)
    registry = ledger.get("result_registry")
    _require(isinstance(registry, dict), "canonical result registry is missing")
    screen_registry = registry.get("screen")
    _require(
        isinstance(screen_registry, dict)
        and screen_registry.get("status") == SCREEN_PASS_STATUS,
        "formal finalization requires registered RND5 PASS",
    )
    _require(
        registry.get("formal") == authority._PENDING_FORMAL_REGISTRY,
        "formal result registry is not pending exact RND6 completion",
    )
    formal_registry = _registry_entry(
        phase="RND6Q",
        status=status,
        report_path=FORMAL_REPORT_PATH,
        merge_path=FORMAL_MERGE_PATH,
        assembly_path=FORMAL_ASSEMBLY_PATH,
        score_path=FORMAL_SCORE_PATH,
        qualification_path=FORMAL_QUALIFICATION_PATH,
        provenance=provenance,
        qualification=qualification,
        recorded_at=recorded_at,
        finalizer_source_commit=finalizer_source_commit,
    )
    ledger["result_registry"] = {
        "screen": screen_registry,
        "formal": formal_registry,
    }
    _set_phase(
        active,
        contract,
        ledger,
        research,
        phase="RND6T",
        recorded_at=recorded_at,
    )
    ledger["score_accessed"] = True
    research["screen_result_status"] = SCREEN_PASS_STATUS
    research["formal_result_status"] = status
    research["publication_ready"] = False
    active["authorization"].update(authority.RND6T_AUTHORIZATION)
    active["training_allowed"] = False
    active["candidate_model_training_allowed"] = False
    active["held_score_read_allowed"] = False
    active["gate_state"] = {
        **authority.RND6Q_GATE_STATE,
        "RND6Q": status,
        "RND6T": f"TERMINAL_{status}",
    }
    active["next_allowed_action"] = authority.RND6T_ACTION
    ledger["next_action"] = authority.RND6T_ACTION
    exit_code, gate_passed, integrity_passed, complete_valid = FORMAL_SEMANTICS[status]
    ledger["decisions"].append(
        {
            "time": recorded_at,
            "event": status,
            "decision": authority.RND6T_DECISION,
            "canonical_formal_qualification_path": str(FORMAL_QUALIFICATION_PATH),
            "canonical_formal_qualification_status": status,
            "exit_code": exit_code,
            "gate_passed": gate_passed,
            "integrity_passed": integrity_passed,
            "complete_valid_qualification": complete_valid,
            "formal_score_accessed": True,
            "formal_qualification_accessed": True,
            "partial_score_accessed": False,
            "new_external_outcome_accessed": False,
            "model_or_threshold_selection_performed": False,
            "evidence_status": EVIDENCE_STATUS,
            "canonical_formal_report_path": str(FORMAL_REPORT_PATH),
            "authority_token": authority.TOKENS["RND6T"],
        }
    )
    record = (
        f"- {recorded_at}: The production finalizer recorded canonical RND6Q status "
        f"`{status}` at `{FORMAL_REPORT_PATH}` and closed all runtime rights in `RND6T`. "
        f"Evidence remains `{EVIDENCE_STATUS}`; publication readiness is false."
    )
    research_body = _append_research_record(research_body, record)
    report = _render_report(
        kind="formal",
        qualification=qualification,
        registry=formal_registry,
        canonical_paths=[
            ("RND5 prerequisite qualification", SCREEN_QUALIFICATION_PATH),
            ("Exact-100 target-free merge", FORMAL_MERGE_PATH),
            ("Equal-seed prediction-only assembly", FORMAL_ASSEMBLY_PATH),
            ("Complete formal score", FORMAL_SCORE_PATH),
            ("Formal qualification", FORMAL_QUALIFICATION_PATH),
        ],
    )
    return report, research_body


def _assert_report_compatible(path: Path, expected: str) -> None:
    if path.exists():
        _require(
            path.read_text(encoding="utf-8") == expected,
            f"canonical report already exists with different content: {path}",
        )


def _already_finalized(
    repo_root: Path,
    *,
    phase: str,
    active: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    registry = ledger.get("result_registry")
    _require(isinstance(registry, dict), "canonical result registry is missing")
    if phase in {"RND6P", "RND5T"}:
        _, _, qualification, provenance = _screen_bundle(active)
        screen = registry.get("screen")
        _require(isinstance(screen, dict), "screen result registry is missing")
        expected_screen = _registry_entry(
            phase="RND5",
            status=str(qualification["status"]),
            report_path=SCREEN_REPORT_PATH,
            merge_path=SCREEN_MERGE_PATH,
            score_path=SCREEN_SCORE_PATH,
            qualification_path=SCREEN_QUALIFICATION_PATH,
            provenance=provenance,
            qualification=qualification,
            recorded_at=str(screen.get("recorded_at")),
            finalizer_source_commit=str(screen.get("finalizer_source_commit")),
        )
        _require(screen == expected_screen, "existing screen result registry differs")
        expected_formal = (
            authority._PENDING_FORMAL_REGISTRY
            if phase == "RND6P"
            else {
                "status": authority.FORMAL_NOT_RUN_STATUS,
                "reason": qualification["status"],
                "report_path": str(FORMAL_REPORT_PATH),
                "report_exists": False,
                "publication_ready": False,
            }
        )
        _require(registry.get("formal") == expected_formal, "existing formal registry differs")
        report = _render_report(
            kind="screen",
            qualification=qualification,
            registry=screen,
            canonical_paths=[
                ("Complete target-free merge", SCREEN_MERGE_PATH),
                ("Complete score", SCREEN_SCORE_PATH),
                ("Qualification", SCREEN_QUALIFICATION_PATH),
            ],
        )
        report_path = repo_root / SCREEN_REPORT_PATH
        _require(report_path.is_file(), "canonical screen report is missing")
        _assert_report_compatible(report_path, report)
        status = str(qualification["status"])
        source_phase = "RND5"
    else:
        _require(phase == "RND6T", f"phase {phase} is not an idempotent finalization phase")
        _, _, _, qualification, _, provenance = _formal_bundle(active)
        formal = registry.get("formal")
        _require(isinstance(formal, dict), "formal result registry is missing")
        expected_formal = _registry_entry(
            phase="RND6Q",
            status=str(qualification["status"]),
            report_path=FORMAL_REPORT_PATH,
            merge_path=FORMAL_MERGE_PATH,
            assembly_path=FORMAL_ASSEMBLY_PATH,
            score_path=FORMAL_SCORE_PATH,
            qualification_path=FORMAL_QUALIFICATION_PATH,
            provenance=provenance,
            qualification=qualification,
            recorded_at=str(formal.get("recorded_at")),
            finalizer_source_commit=str(formal.get("finalizer_source_commit")),
        )
        _require(formal == expected_formal, "existing formal result registry differs")
        report = _render_report(
            kind="formal",
            qualification=qualification,
            registry=formal,
            canonical_paths=[
                ("RND5 prerequisite qualification", SCREEN_QUALIFICATION_PATH),
                ("Exact-100 target-free merge", FORMAL_MERGE_PATH),
                ("Equal-seed prediction-only assembly", FORMAL_ASSEMBLY_PATH),
                ("Complete formal score", FORMAL_SCORE_PATH),
                ("Formal qualification", FORMAL_QUALIFICATION_PATH),
            ],
        )
        report_path = repo_root / FORMAL_REPORT_PATH
        _require(report_path.is_file(), "canonical formal report is missing")
        _assert_report_compatible(report_path, report)
        status = str(qualification["status"])
        source_phase = "RND6Q"
    return {
        "schema_version": "reactflow_delta.independent_rnet_distill_finalization.v1",
        "status": "INDEPENDENT_RNET_DISTILL_FINALIZATION_ALREADY_EXACT_NO_CHANGES",
        "source_phase": source_phase,
        "next_phase": phase,
        "qualification_status": status,
        "report_path": str(report_path.relative_to(repo_root)),
        "publication_ready": False,
        "commit_or_push_performed": False,
        "next_action": "NONE_ALREADY_FINALIZED_AND_AUTHORITY_VALIDATED",
    }


def finalize(repo_root: Path, *, recorded_at: str) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    recorded_at = _validated_recorded_at(recorded_at)
    validation = authority.validate_contract(repo_root)
    phase = str(validation["phase"])
    active_path = repo_root / authority.ACTIVE_PATH
    contract_path = repo_root / authority.CONTRACT_PATH
    ledger_path = repo_root / authority.LEDGER_PATH
    research_path = repo_root / authority.RESEARCH_PATH
    active = _load_yaml(active_path)
    contract = _load_yaml(contract_path)
    ledger = _load_yaml(ledger_path)
    research, research_body = _load_research(research_path)
    _require(active.get("result_finalization") == authority.ACTIVE_RESULT_FINALIZATION, "active finalizer binding changed")
    _require(contract.get("result_finalization") == authority.CONTRACT_RESULT_FINALIZATION, "machine finalizer binding changed")
    if phase in {"RND6P", "RND5T", "RND6T"}:
        return _already_finalized(
            repo_root,
            phase=phase,
            active=active,
            ledger=ledger,
        )
    _require(phase in {"RND5", "RND6Q"}, f"phase {phase} is not a finalization source phase")
    finalizer_source_commit = _git(repo_root, "rev-parse", "HEAD").lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", finalizer_source_commit) is not None,
        "finalizer source HEAD is not a canonical commit",
    )

    if phase == "RND5":
        _, _, qualification, provenance = _screen_bundle(active)
        report, research_body = _screen_transition(
            active=active,
            contract=contract,
            ledger=ledger,
            research=research,
            research_body=research_body,
            qualification=qualification,
            provenance=provenance,
            recorded_at=recorded_at,
            finalizer_source_commit=finalizer_source_commit,
        )
        report_path = repo_root / SCREEN_REPORT_PATH
    else:
        _, _, _, qualification, _, provenance = _formal_bundle(active)
        report, research_body = _formal_transition(
            active=active,
            contract=contract,
            ledger=ledger,
            research=research,
            research_body=research_body,
            qualification=qualification,
            provenance=provenance,
            recorded_at=recorded_at,
            finalizer_source_commit=finalizer_source_commit,
        )
        report_path = repo_root / FORMAL_REPORT_PATH

    _assert_report_compatible(report_path, report)
    _write_text_atomic(report_path, report)
    _write_text_atomic(active_path, _dump_yaml(active))
    _write_text_atomic(contract_path, _dump_yaml(contract))
    _write_text_atomic(ledger_path, _dump_yaml(ledger))
    _write_text_atomic(research_path, _dump_research(research, research_body))
    return {
        "schema_version": "reactflow_delta.independent_rnet_distill_finalization.v1",
        "status": "INDEPENDENT_RNET_DISTILL_FINALIZATION_WRITTEN_PENDING_FOCUSED_COMMIT",
        "source_phase": phase,
        "next_phase": active["authority"]["current_phase"],
        "qualification_status": qualification["status"],
        "report_path": str(report_path.relative_to(repo_root)),
        "publication_ready": False,
        "commit_or_push_performed": False,
        "next_action": "OPERATOR_FOCUSED_COMMIT_THEN_CLEAN_WORKTREE_AUTHORITY_VALIDATION",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--recorded-at", required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(args.repo_root, recorded_at=args.recorded_at)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "INDEPENDENT_RNET_DISTILL_FINALIZATION_REFUSED",
                    "error": str(error),
                    "commit_or_push_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
