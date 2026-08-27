#!/usr/bin/env python3
"""Score one complete puzzle-set universe after the score-blind merge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    MERGED_SCHEMA,
)
from scripts.reactflow_delta.model_rescue_v1 import (
    weighted_gaussian_mixture_crps,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    SafeTIC2AFold,
    validate_tic2a_safe_registry,
)
from scripts.reactflow_delta.puzzle_set_score_chain import (
    EXPECTED_PROJECT_TASK,
    assert_active_phase,
    assert_authority_paths,
    validate_v13_historical_bundle,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v10 import _load_tic2a_absolute
from scripts.reactflow_delta.score_model_rescue_v13 import (
    _central_covered,
)
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.puzzle_set_meta_context_score.proposed.v2"
EXPECTED_PHASE = "P1M3"
EXPECTED_SCORE_TOKEN = "PUZZLE_SET_COMPLETE_MERGE_SCORE_ONCE_ONLY"


def assert_score_authority(
    repo_root: Path,
    *,
    merged_json: Path | None = None,
    tic2a_merged_json: Path | None = None,
    v13_historical_bundle: Path | None = None,
    m2_csv: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    active = assert_active_phase(
        repo_root,
        phase=EXPECTED_PHASE,
        score_token=EXPECTED_SCORE_TOKEN,
        training_must_be_closed=True,
    )
    provided = {
        "complete_unscored_merge_path": merged_json,
        "tic2a_merged_registry_path": tic2a_merged_json,
        "v13_historical_bundle_path": v13_historical_bundle,
        "m2_csv_path": m2_csv,
        "complete_score_path": out_json,
    }
    present = {name: value for name, value in provided.items() if value is not None}
    if present:
        if len(present) != len(provided):
            raise RuntimeError("Puzzle-Set scorer CLI path binding is incomplete")
        assert_authority_paths(active, present)
    return active


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    required_true = (
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
    required_false = ("partial_scores_inspected", "external_outcome_accessed")
    return all(integrity.get(name) is True for name in required_true) and all(
        integrity.get(name) is False for name in required_false
    )


def _load_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: np.asarray(handle[name]) for name in handle.files}
    if str(prediction.get("schema_version", np.asarray("")).item()) != (
        PREDICTION_SCHEMA
    ):
        raise ValueError(f"invalid puzzle-set prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {int(fold)} or set(
        map(int, prediction["seed"])
    ) != {0}:
        raise ValueError(f"puzzle-set fold or seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate puzzle-set keys in {path}")
    return prediction


def score_fold(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
    tic2a_absolute: dict[str, float],
) -> dict[str, Any]:
    """Apply the frozen method-balanced held-puzzle estimator."""

    index = {str(key): row for row, key in enumerate(prediction["keys"])}
    expected = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    if set(index) != expected or set(tic2a_absolute) != expected:
        raise ValueError("puzzle-set/TIC2A registered key universes are not exact")
    metric_names = (
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
    )
    values: dict[str, dict[str, float]] = {name: {} for name in metric_names}
    n_qualified = 0
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        positions = np.flatnonzero(construct.wt_observed & np.isfinite(target))
        keys = [_bio_key(univ, record, int(position)) for position in positions]
        rows = np.asarray([index[key] for key in keys], dtype=np.int64)
        signed = target[positions] - construct.wt_reactivity[positions]
        absolute = np.abs(signed)
        distributions = {
            name: (
                prediction[f"{name}_weights"][rows],
                prediction[f"{name}_locations"][rows],
                prediction[f"{name}_scales"][rows],
            )
            for name in ("candidate", "null")
        }
        arrays = {
            "feature41_signed_delta_mae": np.abs(
                signed - prediction["feature41_point"][rows]
            ),
            "parent_signed_delta_mae": np.abs(
                signed - prediction["parent_point"][rows]
            ),
            "candidate_signed_delta_mae": np.abs(
                signed - prediction["candidate_point"][rows]
            ),
            "null_signed_delta_mae": np.abs(signed - prediction["null_point"][rows]),
            "feature41_absolute_delta_mae": np.abs(
                absolute - np.asarray([tic2a_absolute[key] for key in keys])
            ),
            "parent_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["parent_point"][rows])
            ),
            "candidate_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["candidate_point"][rows])
            ),
            "null_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["null_point"][rows])
            ),
            "candidate_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["candidate_expected_absolute_delta"][rows]
            ),
            "null_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["null_expected_absolute_delta"][rows]
            ),
            "candidate_crps": weighted_gaussian_mixture_crps(
                distributions["candidate"][1],
                distributions["candidate"][2],
                distributions["candidate"][0],
                signed,
            ),
            "null_crps": weighted_gaussian_mixture_crps(
                distributions["null"][1],
                distributions["null"][2],
                distributions["null"][0],
                signed,
            ),
            "candidate_coverage68": _central_covered(
                signed, *distributions["candidate"], 0.68
            ),
            "null_coverage68": _central_covered(signed, *distributions["null"], 0.68),
            "candidate_coverage95": _central_covered(
                signed, *distributions["candidate"], 0.95
            ),
            "null_coverage95": _central_covered(signed, *distributions["null"], 0.95),
        }
        for name, array in arrays.items():
            values[name].update({key: float(value) for key, value in zip(keys, array)})
        n_qualified += len(keys)
    result = {name: _puzzle_macro(data) for name, data in values.items()}
    result.update(
        {
            "n_qualified_positions": n_qualified,
            "n_registered_expected": len(expected),
            "n_registered_observed": len(index),
            "registered_prediction_coverage": len(expected & set(index))
            / max(len(expected), 1),
            "failure_rate": len(expected - set(index)) / max(len(expected), 1),
            "n_unexpected_prediction_keys": len(set(index) - expected),
        }
    )
    return result


def _v13_reference_rows(score: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return validate_v13_historical_bundle(score)


def _validate_tic2a_registry_and_provenance(
    merged: dict[str, Any],
    tic2a_merged: dict[str, Any],
    *,
    registry_path: Path | None,
) -> dict[int, SafeTIC2AFold]:
    """Validate the safe registry before any TIC2A prediction is opened."""

    safe = validate_tic2a_safe_registry(tic2a_merged)
    if registry_path is None:
        return safe
    expected_registry = Path(registry_path)
    if not expected_registry.is_absolute():
        raise ValueError("Puzzle-Set TIC2A registry path must be absolute")
    rows = merged.get("folds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Puzzle-Set merged provenance has no fold rows")
    for row in rows:
        fold = int(row.get("outer_fold", -1))
        sources = row.get("frozen_input_sources")
        if not isinstance(sources, dict):
            raise ValueError(f"Puzzle-Set fold {fold} source provenance is absent")
        registry = sources.get("tic2a_merged_registry")
        model = sources.get("tic2a_feature41_model_artifact")
        if (
            not isinstance(registry, dict)
            or Path(str(registry.get("path", ""))) != expected_registry
            or not isinstance(model, dict)
            or fold not in safe
            or Path(str(model.get("path", ""))) != safe[fold].model_path
        ):
            raise ValueError(
                f"Puzzle-Set fold {fold} TIC2A provenance differs from registry"
            )
    return safe


def _assert_parent_and_baseline_replay(
    observed: dict[str, Any], reference: dict[str, Any]
) -> None:
    pairs = (
        ("feature41_signed_delta_mae", "feature41_signed_delta_mae"),
        ("feature41_absolute_delta_mae", "feature41_absolute_delta_mae"),
        ("parent_signed_delta_mae", "candidate_signed_delta_mae"),
        ("parent_point_absolute_delta_mae", "candidate_point_absolute_delta_mae"),
    )
    for observed_name, reference_name in pairs:
        if not np.isclose(
            float(observed[observed_name]),
            float(reference[reference_name]),
            atol=5e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"puzzle-set parent/baseline replay differs at {observed_name}"
            )


def score_complete(
    merged: dict[str, Any],
    tic2a_merged: dict[str, Any],
    v13_score: dict[str, Any],
    m2_csv: Path,
    *,
    tic2a_registry_path: Path | None = None,
) -> dict[str, Any]:
    if (
        merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("status") != "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
        or merged.get("phase") != EXPECTED_PHASE
        or merged.get("expected_folds") != list(range(20))
        or merged.get("expected_seeds") != [0]
        or int(merged.get("expected_pretraining_epochs", -1)) != 200
        or int(merged.get("expected_point_epochs", -1)) != 40
        or int(merged.get("expected_calibration_epochs", -1)) != 40
    ):
        raise ValueError("puzzle-set scorer requires one complete unscored merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("puzzle-set merged integrity is not qualified")
    fold_rows = {int(row["outer_fold"]): row for row in merged.get("folds", [])}
    tic_rows = _validate_tic2a_registry_and_provenance(
        merged,
        tic2a_merged,
        registry_path=tic2a_registry_path,
    )
    reference_rows = _v13_reference_rows(v13_score)
    if sorted(fold_rows) != list(range(20)) or sorted(tic_rows) != list(range(20)):
        raise ValueError("puzzle-set scorer requires folds0-19 in both universes")

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("puzzle-set scorer requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    rows = []
    for fold_id in range(20):
        fold = folds[fold_id]
        reference = reference_rows[fold_id]
        if str(reference.get("held_puzzle")) != str(fold.held_puzzle):
            raise ValueError("puzzle-set/V13 held puzzle alignment differs")
        held_records = [
            record for record in records if record.puzzle == fold.held_puzzle
        ]
        score = score_fold(
            univ,
            held_records,
            _load_prediction(Path(fold_rows[fold_id]["prediction_artifact"]), fold_id),
            _load_tic2a_absolute(
                Path(tic_rows[fold_id].row["prediction_artifact"]), fold_id
            ),
        )
        _assert_parent_and_baseline_replay(score, reference)
        score.update(
            {
                "outer_fold": fold_id,
                "held_puzzle": str(fold.held_puzzle),
                "feature41_crps": float(reference["feature41_crps"]),
                "feature41_coverage68": float(reference["feature41_coverage68"]),
                "feature41_coverage95": float(reference["feature41_coverage95"]),
                "historical_v13_signed_delta_mae": float(
                    reference["candidate_signed_delta_mae"]
                ),
                "historical_v13_point_absolute_delta_mae": float(
                    reference["candidate_point_absolute_delta_mae"]
                ),
                "terminal_v12_signed_delta_mae": float(
                    reference["terminal_v12_signed_delta_mae"]
                ),
                "terminal_v11_point_absolute_delta_mae": float(
                    reference["terminal_v11_point_absolute_delta_mae"]
                ),
                "terminal_v12_crps": float(reference["terminal_v12_crps"]),
                "terminal_v10_distribution_absolute_delta_mae": float(
                    reference["terminal_v10_distribution_absolute_delta_mae"]
                ),
            }
        )
        rows.append(score)
    return {
        "schema_version": SCHEMA,
        "phase": EXPECTED_PHASE,
        "status": "PUZZLE_SET_M3_COMPLETE_SCORE_PASS",
        "scores": rows,
        "context_retention_summary": merged["context_retention_summary"],
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "v13_parent_and_feature41_replay_at_5e_7": True,
        "v13_historical_bundle_protocol_validated": True,
        "tic2a_registry_cross_linked_to_merged_provenance": (
            tic2a_registry_path is not None
        ),
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--v13-score-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    merged_json = args.merged_json.resolve()
    tic2a_merged_json = args.tic2a_merged_json.resolve()
    v13_historical_bundle = args.v13_score_json.resolve()
    m2_csv = args.m2_csv.resolve()
    out_json = args.out_json.resolve()
    assert_score_authority(
        args.repo_root.resolve(),
        merged_json=merged_json,
        tic2a_merged_json=tic2a_merged_json,
        v13_historical_bundle=v13_historical_bundle,
        m2_csv=m2_csv,
        out_json=out_json,
    )
    if out_json.exists():
        raise FileExistsError("puzzle-set refuses to overwrite its complete score")
    result = score_complete(
        json.loads(merged_json.read_text(encoding="utf-8")),
        json.loads(tic2a_merged_json.read_text(encoding="utf-8")),
        json.loads(v13_historical_bundle.read_text(encoding="utf-8")),
        m2_csv,
        tic2a_registry_path=tic2a_merged_json,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_json.with_name(f"{out_json.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, out_json)
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
