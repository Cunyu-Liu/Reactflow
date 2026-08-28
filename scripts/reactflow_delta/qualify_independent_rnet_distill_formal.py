#!/usr/bin/env python3
"""Apply the pre-result fixed five-seed independent-RNet formal Gate once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.qualify_independent_rnet_distill import (
    EVIDENCE_STATUS,
    FROZEN_SCREEN_GATES,
    METRIC_FIELDS,
    PASS_STATUS as SCREEN_PASS_STATUS,
    QUALIFICATION_SCHEMA as SCREEN_QUALIFICATION_SCHEMA,
    _score_integrity_errors,
    paired_summary,
    qualify as qualify_screen,
)
from scripts.reactflow_delta.score_independent_rnet_distill import (
    SCORE_ROW_FIELDS,
    SCORE_SCHEMA,
    SCORE_STATUS,
)
from scripts.reactflow_delta.score_independent_rnet_distill_formal import (
    EXPECTED_FOLD_SEED_COUNT,
    EXPECTED_SEEDS,
    FORMAL_DIR,
    FORMAL_QUALIFICATION_PATH,
    FORMAL_SCORE_PATH,
    FORMAL_SCORE_PHASE,
    FORMAL_SCORE_SCHEMA,
    FORMAL_SCORE_STATUS,
    SCREEN_QUALIFICATION_PATH,
)
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    PROJECT_TASK_ID,
    assert_run_authority,
)


FORMAL_QUALIFICATION_PHASE = "RND6Q"
FORMAL_QUALIFICATION_SCHEMA = (
    "reactflow_delta.independent_rnet_distill_formal_qualification.v1"
)
FORMAL_PASS_STATUS = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_FORMAL_PASS"
FORMAL_FAIL_STATUS = "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_FORMAL_FAIL"
FORMAL_INDETERMINATE_STATUS = (
    "RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_FORMAL_INDETERMINATE"
)

FROZEN_FORMAL_GATES = {
    "screen_prerequisite_status": SCREEN_PASS_STATUS,
    "equal_seed_mixture_required": True,
    "mixture_must_pass_frozen_screen_gates": True,
    "individual_seed_positive_vs_matched_null_minimum": {
        "signed_delta": 4,
        "point_absolute": 4,
        "task_crps": 4,
        "distribution_absolute": 4,
    },
    "strict_positive_mean_gain_required": True,
    "best_seed_selection_allowed": False,
    "extra_seed_selection_allowed": False,
    "evidence_ceiling": EVIDENCE_STATUS,
}

FORMAL_SCORE_TOP_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "mixture_scores",
    "individual_seed_scores",
    "integrity_errors",
    "complete_valid_score",
    "complete_source_fold_seed_universe",
    "complete_assembly_fold_universe",
    "expected_fold_seed_count",
    "actual_fold_seed_count",
    "expected_fold_count",
    "actual_fold_count",
    "expected_seed_count",
    "actual_seed_count",
    "failed_rows",
    "duplicate_or_unexpected_artifacts",
    "target_profile_identity",
    "target_join_after_complete_merge_and_assembly",
    "aggregation",
    "independent_units",
    "attribution_null",
    "feature41_comparator",
    "historical_parent_source",
    "historical_distribution_comparator",
    "equal_seed_mixture",
    "equal_seed_weight",
    "best_seed_selection_performed",
    "partial_fold_scores_inspected",
    "partial_seed_scores_inspected",
    "external_outcome_accessed",
    "model_or_threshold_selection_performed",
    "source_exposure_status",
}


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a mapping at {path}")
    return value


def assert_formal_qualifier_authority(
    repo_root: Path,
    *,
    screen_qualification_json: Path,
    score_json: Path,
    out_json: Path,
) -> dict[str, Any]:
    """Require exact RND6Q authority and canonical CLI paths."""

    assert_run_authority(repo_root, FORMAL_QUALIFICATION_PHASE)
    active = _load_yaml(repo_root / "configs/reactflow_delta/active_contract.yaml")
    if active.get("project_task_id") != PROJECT_TASK_ID:
        raise RuntimeError("formal qualifier is not under the independent RNet project")
    authority = active.get("authority", {})
    expected = {
        "formal_prediction_dir": FORMAL_DIR,
        "formal_complete_score_path": FORMAL_SCORE_PATH,
        "formal_qualification_path": FORMAL_QUALIFICATION_PATH,
        "screen_qualification_path": SCREEN_QUALIFICATION_PATH,
    }
    for name, path in expected.items():
        if name not in authority or _resolved(authority[name]) != path.resolve():
            raise RuntimeError(f"RND6Q active authority {name} is not exact")
    provided = {
        "screen_qualification_path": screen_qualification_json,
        "formal_complete_score_path": score_json,
        "formal_qualification_path": out_json,
    }
    for name, path in provided.items():
        if _resolved(path) != expected[name].resolve():
            raise RuntimeError(f"RND6Q CLI {name} differs from active authority")
    return active


def load_frozen_formal_gates(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _load_yaml(
        repo_root / "configs/reactflow_delta/independent_rnet_distill_contract.yaml"
    )
    if (
        contract.get("schema_version")
        != "reactflow_delta.independent_rnet_distill_contract.v1"
        or contract.get("project_task_id") != PROJECT_TASK_ID
        or contract.get("screen_gates") != FROZEN_SCREEN_GATES
        or contract.get("formal_gates") != FROZEN_FORMAL_GATES
    ):
        raise RuntimeError("RND6Q frozen screen or formal Gates differ from contract")
    return contract["screen_gates"], contract["formal_gates"]


def _adapt_screen_score(rows: list[dict[str, Any]], formal_score: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCORE_SCHEMA,
        "phase": "RND4",
        "status": SCORE_STATUS,
        "scores": rows,
        "integrity_errors": [],
        "complete_valid_score": True,
        "complete_fold_artifact_universe": True,
        "expected_fold_count": 20,
        "actual_fold_count": 20,
        "failed_rows": int(formal_score["failed_rows"]),
        "duplicate_or_unexpected_artifacts": int(
            formal_score["duplicate_or_unexpected_artifacts"]
        ),
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


def _formal_score_integrity_errors(score: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(score) != FORMAL_SCORE_TOP_FIELDS:
        errors.append("formal_score_fields")
    expected_top = {
        "schema_version": FORMAL_SCORE_SCHEMA,
        "phase": FORMAL_SCORE_PHASE,
        "status": FORMAL_SCORE_STATUS,
        "integrity_errors": [],
        "complete_valid_score": True,
        "complete_source_fold_seed_universe": True,
        "complete_assembly_fold_universe": True,
        "expected_fold_seed_count": EXPECTED_FOLD_SEED_COUNT,
        "actual_fold_seed_count": EXPECTED_FOLD_SEED_COUNT,
        "expected_fold_count": 20,
        "actual_fold_count": 20,
        "expected_seed_count": 5,
        "actual_seed_count": 5,
        "failed_rows": 0,
        "duplicate_or_unexpected_artifacts": 0,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge_and_assembly": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES_NOT_100_FOLD_SEEDS",
        "attribution_null": "RNET2_SHIFT17_SINGLE_FEATURE_DISTILLATION",
        "feature41_comparator": (
            "AUTHORITATIVE_FEATURE41_SEED0_REPLAY_FIXED_ACROSS_SEEDS"
        ),
        "historical_parent_source": "FROZEN_V14_CANONICAL_COMPLETE_SCORE",
        "historical_distribution_comparator": (
            "FROZEN_V10_COMPARATOR_FIXED_ACROSS_SEEDS"
        ),
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "partial_seed_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_exposure_status": EVIDENCE_STATUS,
    }
    errors.extend(name for name, expected in expected_top.items() if score.get(name) != expected)
    mixture = score.get("mixture_scores")
    individual = score.get("individual_seed_scores")
    if not isinstance(mixture, list):
        errors.append("mixture_scores_malformed")
    else:
        errors.extend(_score_integrity_errors(_adapt_screen_score(mixture, score)))
    if not isinstance(individual, dict) or set(individual) != {
        str(seed) for seed in EXPECTED_SEEDS
    }:
        errors.append("individual_seed_universe")
    else:
        for seed in EXPECTED_SEEDS:
            rows = individual[str(seed)]
            if not isinstance(rows, list):
                errors.append(f"seed{seed}_scores_malformed")
                continue
            errors.extend(
                f"seed{seed}_{error}"
                for error in _score_integrity_errors(_adapt_screen_score(rows, score))
            )
    return sorted(set(errors))


def _screen_prerequisite_errors(screen: dict[str, Any]) -> list[str]:
    expected = {
        "schema_version": SCREEN_QUALIFICATION_SCHEMA,
        "phase": "RND5",
        "status": SCREEN_PASS_STATUS,
        "gate_passed": True,
        "integrity_passed": True,
        "integrity_errors": [],
        "rnd6_authorized": True,
        "model_or_threshold_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "new_external_outcome_accessed": False,
        "evidence_status": EVIDENCE_STATUS,
        "clean_ood": "NOT_ESTABLISHED",
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
    }
    return sorted(name for name, value in expected.items() if screen.get(name) != value)


def _indeterminate(errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_QUALIFICATION_SCHEMA,
        "phase": FORMAL_QUALIFICATION_PHASE,
        "status": FORMAL_INDETERMINATE_STATUS,
        "gate_passed": False,
        "integrity_passed": False,
        "integrity_errors": sorted(set(errors)),
        "gates": {},
        "mixture_gates": {},
        "mixture_comparisons": {},
        "mixture_calibration": {},
        "individual_seed_directions": {},
        "positive_seed_counts": {},
        "evidence_status": EVIDENCE_STATUS,
        "clean_ood": "NOT_ESTABLISHED",
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
        "terminal_closure_required": True,
    }


def qualify_formal(
    score: dict[str, Any],
    screen_qualification: dict[str, Any],
    screen_gates: dict[str, Any],
    formal_gates: dict[str, Any],
) -> dict[str, Any]:
    if screen_gates != FROZEN_SCREEN_GATES or formal_gates != FROZEN_FORMAL_GATES:
        raise RuntimeError("RND6Q cannot apply changed or lowered formal Gates")
    errors = [
        *(f"screen_{name}" for name in _screen_prerequisite_errors(screen_qualification)),
        *_formal_score_integrity_errors(score),
    ]
    if errors:
        return _indeterminate(errors)

    mixture_score = _adapt_screen_score(score["mixture_scores"], score)
    mixture = qualify_screen(mixture_score, screen_gates)
    if mixture.get("integrity_passed") is not True:
        return _indeterminate(
            [f"mixture_{error}" for error in mixture.get("integrity_errors", [])]
        )

    individual_seed_directions: dict[str, dict[str, Any]] = {}
    positive_seed_counts = {metric: 0 for metric in METRIC_FIELDS}
    try:
        for seed in EXPECTED_SEEDS:
            rows = sorted(
                score["individual_seed_scores"][str(seed)],
                key=lambda row: int(row["outer_fold"]),
            )
            seed_summary: dict[str, Any] = {}
            for metric, fields in METRIC_FIELDS.items():
                result = paired_summary(
                    rows, fields["matched_null"], fields["candidate"]
                )
                positive = float(result["mean_gain"]) > 0.0
                positive_seed_counts[metric] += int(positive)
                seed_summary[metric] = {
                    "mean_gain_vs_matched_null": result["mean_gain"],
                    "relative_gain_vs_matched_null": result["relative_gain"],
                    "positive": positive,
                }
            individual_seed_directions[str(seed)] = seed_summary
    except (KeyError, TypeError, ValueError) as error:
        return _indeterminate([f"individual_seed_comparison:{error}"])

    gates = {
        "screen_prerequisite_exact_pass": True,
        "equal_seed_mixture_exact": score["equal_seed_mixture"] is True
        and float(score["equal_seed_weight"]) == 0.2,
        "no_best_seed_selection": score["best_seed_selection_performed"] is False,
        "mixture_repeats_every_frozen_screen_gate": bool(mixture["gate_passed"]),
    }
    minimum = formal_gates["individual_seed_positive_vs_matched_null_minimum"]
    for metric in METRIC_FIELDS:
        gates[f"{metric}_positive_individual_seeds_ge_{int(minimum[metric])}"] = (
            int(positive_seed_counts[metric]) >= int(minimum[metric])
        )
    passed = all(gates.values())
    return {
        "schema_version": FORMAL_QUALIFICATION_SCHEMA,
        "phase": FORMAL_QUALIFICATION_PHASE,
        "status": FORMAL_PASS_STATUS if passed else FORMAL_FAIL_STATUS,
        "gate_passed": passed,
        "integrity_passed": True,
        "integrity_errors": [],
        "gates": gates,
        "mixture_gates": mixture["gates"],
        "mixture_comparisons": mixture["comparisons"],
        "mixture_calibration": mixture["calibration"],
        "individual_seed_directions": individual_seed_directions,
        "positive_seed_counts": positive_seed_counts,
        "frozen_screen_gate_values": screen_gates,
        "frozen_formal_gate_values": formal_gates,
        "screen_prerequisite_status": screen_qualification["status"],
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "model_or_threshold_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "partial_seed_scores_inspected": False,
        "new_external_outcome_accessed": False,
        "evidence_status": EVIDENCE_STATUS,
        "clean_ood": "NOT_ESTABLISHED",
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "publication_ready": False,
        "terminal_closure_required": True,
    }


def _write_json_once(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("RND6Q refuses to overwrite formal qualification")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--screen-qualification-json", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    screen_qualification_json = args.screen_qualification_json.resolve()
    score_json = args.score_json.resolve()
    out_json = args.out_json.resolve()
    try:
        assert_formal_qualifier_authority(
            repo_root,
            screen_qualification_json=screen_qualification_json,
            score_json=score_json,
            out_json=out_json,
        )
        if out_json.exists():
            raise FileExistsError("RND6Q formal qualification exists; refusing rerun")
        screen_gates, formal_gates = load_frozen_formal_gates(repo_root)
        screen_qualification = json.loads(
            screen_qualification_json.read_text(encoding="utf-8")
        )
        score = json.loads(score_json.read_text(encoding="utf-8"))
        result = qualify_formal(
            score, screen_qualification, screen_gates, formal_gates
        )
        _write_json_once(out_json, result)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": FORMAL_INDETERMINATE_STATUS, "error": str(error)}
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    if result["status"] == FORMAL_PASS_STATUS:
        return 0
    return 1 if result["status"] == FORMAL_FAIL_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
