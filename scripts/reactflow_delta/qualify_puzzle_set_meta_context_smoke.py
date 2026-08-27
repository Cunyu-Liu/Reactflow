#!/usr/bin/env python3
"""Qualify the target-free P1M2 engineering smoke and nothing scientific."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    MERGED_SCHEMA,
    prediction_checks,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    EXPECTED_RESIDUAL_PARAMETERS,
)
from scripts.reactflow_delta.puzzle_set_meta_context_pretraining import (
    EXPECTED_DECODER_PARAMETERS,
    EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS,
)
from scripts.reactflow_delta.puzzle_set_score_chain import (
    assert_active_phase,
    assert_authority_paths,
)


SCHEMA = "reactflow_delta.puzzle_set_meta_context_smoke_qualification.proposed.v1"
EXPECTED_PARAMETER_COUNT = EXPECTED_TOTAL_PARAMETERS
EXPECTED_TRAINABLE_PARAMETER_COUNT = EXPECTED_TRAINABLE_PARAMETERS

REQUIRED_INTEGRITY_TRUE = (
    "complete_fold_seed_universe",
    "unique_fold_seed_pairs",
    "prediction_only_schema",
    "outcome_blind_puzzle_set_inputs_all_runs",
    "exact_parameter_and_initialization_match_all_runs",
    "candidate_nonfocal_only_cross_attention_all_runs",
    "null_position_deranged_nonfocal_cross_attention_all_runs",
    "candidate_null_equal_attention_support_all_runs",
    "attention_weight_dropout_disabled_all_runs",
    "puzzle_balanced_training_all_runs",
    "position_aligned_nonfocal_cross_values_all_runs",
    "nonfocal_summary_alignment_statistics_all_runs",
    "matched_null_position_deranged_summary_statistics_all_runs",
    "nonfocal_only_cross_values_all_runs",
    "focal_excluded_from_cross_kv_all_runs",
    "eight_token_cross_support_all_runs",
    "paired_cross_block_reference_cancellation_all_runs",
    "zero_nonfocal_exact_cross_replay_all_runs",
    "paired_point_head_reference_cancellation_all_runs",
    "zero_cross_exact_parent_replay_all_runs",
    "fixed_position_derangement_shift_17_all_runs",
    "outer_train_wt_only_puzzle_set_pretraining_all_runs",
    "held_puzzle_excluded_from_pretraining_all_runs",
    "mutant_outcome_excluded_from_pretraining_all_runs",
    "candidate_null_equal_pretraining_budget_all_runs",
    "pretraining_decoder_frozen_downstream_all_runs",
    "encoder_and_point_unchanged_during_pretraining_all_runs",
    "masked_wt_pretraining_protocol_all_runs",
    "puzzle_coordinate_frames_validated_all_runs",
    "frozen_v13_point_parent_all_runs",
    "frozen_v14_context_encoder_all_runs",
    "complete_frozen_input_provenance_all_runs",
    "parent_replay_before_and_after_pretraining_all_runs",
    "point_head_only_warmup_all_runs",
    "point_discriminative_learning_rates_all_runs",
    "pretraining_capability_retention_diagnostic_complete_all_runs",
    "point_frozen_during_calibration_all_runs",
    "v10_residual_family_all_runs",
    "puzzle_balanced_residual_calibration_all_runs",
    "median_constraint_all_runs",
)
REQUIRED_INTEGRITY_FALSE = (
    "partial_scores_inspected",
    "external_outcome_accessed",
)

# These are the fields emitted only after joining held outcomes in the P1M3
# scorer. Their presence means the input is not the frozen unscored smoke merge.
SCIENTIFIC_SCORE_FIELDS = frozenset(
    {
        "scores",
        "target_profile_identity",
        "target_join_after_complete_merge",
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
        "feature41_crps",
        "candidate_crps",
        "null_crps",
        "candidate_coverage68",
        "null_coverage68",
        "candidate_coverage95",
        "null_coverage95",
        "n_qualified_positions",
        "registered_prediction_coverage",
        "failure_rate",
        "n_unexpected_prediction_keys",
    }
)


def _nested_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def _merge_integrity_pass(integrity: Any) -> bool:
    if not isinstance(integrity, dict):
        return False
    return all(integrity.get(name) is True for name in REQUIRED_INTEGRITY_TRUE) and all(
        integrity.get(name) is False for name in REQUIRED_INTEGRITY_FALSE
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _paired_count_is(mapping: Any, expected: int) -> bool:
    if not isinstance(mapping, dict) or set(mapping) != {"candidate", "null"}:
        return False
    return {_as_int(value) for value in mapping.values()} == {expected}


def _fold_seed_pairs(rows: Any) -> list[tuple[int, int]] | None:
    if not isinstance(rows, list):
        return None
    try:
        return sorted(
            (int(row["outer_fold"]), int(row["seed"]))
            for row in rows
            if isinstance(row, dict)
        )
    except (KeyError, TypeError, ValueError):
        return None


def _row_protocol_pass(rows: Any) -> bool:
    if not isinstance(rows, list) or len(rows) != 2:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        if (
            row.get("phase") != "P1M2"
            or row.get("evidence_status") != "ENGINEERING_SMOKE_ONLY"
            or _as_int(row.get("pretraining_epochs")) != 3
            or _as_int(row.get("point_epochs")) != 3
            or _as_int(row.get("calibration_epochs")) != 3
            or _as_int(row.get("candidate_parameter_count")) != EXPECTED_PARAMETER_COUNT
            or _as_int(row.get("null_parameter_count")) != EXPECTED_PARAMETER_COUNT
            or _as_int(row.get("candidate_trainable_parameter_count"))
            != EXPECTED_TRAINABLE_PARAMETER_COUNT
            or _as_int(row.get("null_trainable_parameter_count"))
            != EXPECTED_TRAINABLE_PARAMETER_COUNT
            or not _paired_count_is(
                row.get("residual_parameter_counts"), EXPECTED_RESIDUAL_PARAMETERS
            )
            or not _paired_count_is(
                row.get("candidate_specific_trainable_parameter_counts"),
                EXPECTED_TRAINABLE_PARAMETER_COUNT + EXPECTED_RESIDUAL_PARAMETERS,
            )
            or not _paired_count_is(
                row.get("pretraining_decoder_parameter_counts"),
                EXPECTED_DECODER_PARAMETERS,
            )
            or (_as_int(row.get("n_registered_prediction_rows")) or 0) <= 0
        ):
            return False
        invariants = row.get("invariants", {})
        if (
            invariants.get("prediction_target_free") is not True
            or invariants.get("held_score_computed") is not False
            or invariants.get("external_outcome_accessed") is not False
        ):
            return False
    return True


def _prediction_artifacts_pass(rows: Any) -> bool:
    if not isinstance(rows, list) or len(rows) != 2:
        return False
    seen_keys: set[str] = set()
    try:
        for row in rows:
            checks, keys = prediction_checks(
                Path(row["prediction_artifact"]),
                fold=int(row["outer_fold"]),
                seed=int(row["seed"]),
                expected_rows=int(row["n_registered_prediction_rows"]),
            )
            if not all(checks.values()) or seen_keys.intersection(keys):
                return False
            seen_keys.update(keys)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return bool(seen_keys)


def qualify(merged: dict[str, Any]) -> dict[str, Any]:
    rows = merged.get("folds", [])
    integrity = merged.get("merge_integrity", {})
    retention = merged.get("context_retention_summary", {})
    score_fields_found = sorted(set(_nested_keys(merged)) & SCIENTIFIC_SCORE_FIELDS)
    gates = {
        "complete_unscored_p1m2_merge": (
            merged.get("schema_version") == MERGED_SCHEMA
            and merged.get("status") == "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
            and merged.get("phase") == "P1M2"
        ),
        "exact_fold_seed_universe": (
            merged.get("expected_folds") == [0, 1]
            and merged.get("expected_seeds") == [0]
            and _fold_seed_pairs(rows) == [(0, 0), (1, 0)]
        ),
        "exact_three_stage_smoke_schedule": (
            _as_int(merged.get("expected_pretraining_epochs")) == 3
            and _as_int(merged.get("expected_point_epochs")) == 3
            and _as_int(merged.get("expected_calibration_epochs")) == 3
        ),
        "exact_parameter_counts": (
            _as_int(merged.get("expected_parameter_count_each"))
            == EXPECTED_PARAMETER_COUNT
            and _as_int(merged.get("expected_trainable_parameter_count_each"))
            == EXPECTED_TRAINABLE_PARAMETER_COUNT
            and _as_int(merged.get("expected_residual_parameter_count_each"))
            == EXPECTED_RESIDUAL_PARAMETERS
            and _as_int(
                merged.get("expected_candidate_specific_trainable_parameter_count_each")
            )
            == EXPECTED_TRAINABLE_PARAMETER_COUNT + EXPECTED_RESIDUAL_PARAMETERS
            and _as_int(merged.get("expected_pretraining_decoder_parameter_count_each"))
            == EXPECTED_DECODER_PARAMETERS
            and _as_int(
                merged.get("expected_pretraining_trainable_parameter_count_each", -1)
            )
            == EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS
        ),
        "engineering_only_fold_protocol": _row_protocol_pass(rows),
        "complete_prediction_only_integrity": (
            _merge_integrity_pass(integrity) and _prediction_artifacts_pass(rows)
        ),
        "retention_diagnostic_is_report_only_and_outcome_blind": (
            merged.get("context_retention_gate_required") is False
            and retention.get("selection_performed") is False
            and retention.get("mutant_outcome_used") is False
            and retention.get("held_puzzle_accessed") is False
        ),
        "scientific_score_fields_absent": not score_fields_found,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA,
        "phase": "P1M2",
        "status": (
            "P1M2_ENGINEERING_SMOKE_PASS" if passed else "P1M2_ENGINEERING_SMOKE_FAIL"
        ),
        "gate_passed": passed,
        "gates": gates,
        "scientific_score_fields_found": score_fields_found,
        "scientific_score_computed": False,
        "held_target_read": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
        "evidence_status": "ENGINEERING_SMOKE_ONLY",
        "p1m3_activation_eligible": passed,
        "p1m3_authorized": False,
    }


def assert_smoke_qualifier_authority(
    repo_root: Path, *, merged_json: Path, out_json: Path
) -> dict[str, Any]:
    active = assert_active_phase(
        repo_root,
        phase="P1M2",
        held_score_must_be_closed=True,
    )
    assert_authority_paths(
        active,
        {
            "complete_unscored_merge_path": merged_json,
            "qualification_path": out_json,
        },
    )
    return active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    merged_json = args.merged_json.resolve()
    out_json = args.out_json.resolve()
    assert_smoke_qualifier_authority(
        repo_root, merged_json=merged_json, out_json=out_json
    )
    if out_json.exists():
        raise FileExistsError("puzzle-set refuses to overwrite its smoke qualification")
    result = qualify(json.loads(merged_json.read_text(encoding="utf-8")))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_json.with_name(f"{out_json.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, out_json)
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
