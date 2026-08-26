#!/usr/bin/env python3
"""Score one complete puzzle-set universe after the score-blind merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v10 import _load_tic2a_absolute
from scripts.reactflow_delta.score_model_rescue_v13 import (
    SCHEMA as V13_SCORE_SCHEMA,
    _central_covered,
)
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.score_model_rescue_v9 import TIC2A_MERGED_SCHEMA
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.puzzle_set_meta_context_score.proposed.v1"
EXPECTED_PROJECT_TASK = "reactflow_delta_puzzle_set_meta_context"
EXPECTED_PHASE = "P1M3"
EXPECTED_SCORE_TOKEN = "PUZZLE_SET_COMPLETE_MERGE_SCORE_ONCE_ONLY"


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("puzzle-set scorer is not the active project task")
    authority = active.get("authority", {})
    if authority.get("current_phase") != EXPECTED_PHASE or active.get(
        "runnable_phases"
    ) != [EXPECTED_PHASE]:
        raise RuntimeError("puzzle-set scorer is closed outside complete P1M3")
    if active.get("training_allowed") is not False or active.get(
        "candidate_model_training_allowed"
    ) is not False:
        raise RuntimeError("puzzle-set training must be closed before scoring")
    if active.get("held_score_read_allowed") != EXPECTED_SCORE_TOKEN:
        raise RuntimeError("puzzle-set complete score-once authority is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("puzzle-set partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("puzzle-set scoring requires external outcomes locked")


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    required_true = (
        "complete_fold_seed_universe",
        "unique_fold_seed_pairs",
        "prediction_only_schema",
        "outcome_blind_puzzle_set_inputs_all_runs",
        "exact_parameter_and_initialization_match_all_runs",
        "candidate_full_cross_construct_attention_all_runs",
        "null_block_diagonal_attention_all_runs",
        "puzzle_balanced_training_all_runs",
        "position_aligned_cross_construct_attention_all_runs",
        "leave_one_construct_alignment_statistics_all_runs",
        "matched_null_self_only_alignment_statistics_all_runs",
        "puzzle_coordinate_frames_validated_all_runs",
        "frozen_v13_point_parent_all_runs",
        "frozen_v14_context_encoder_all_runs",
        "zero_initialized_parent_replay_all_runs",
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
            "null_signed_delta_mae": np.abs(
                signed - prediction["null_point"][rows]
            ),
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
            "null_coverage68": _central_covered(
                signed, *distributions["null"], 0.68
            ),
            "candidate_coverage95": _central_covered(
                signed, *distributions["candidate"], 0.95
            ),
            "null_coverage95": _central_covered(
                signed, *distributions["null"], 0.95
            ),
        }
        for name, array in arrays.items():
            values[name].update(
                {key: float(value) for key, value in zip(keys, array)}
            )
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
    if score.get("schema_version") != V13_SCORE_SCHEMA or score.get("status") != (
        "V13M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("puzzle-set scorer requires the frozen complete V13 score")
    rows = {int(row["outer_fold"]): row for row in score.get("scores", [])}
    if sorted(rows) != list(range(20)):
        raise ValueError("puzzle-set scorer requires complete V13 folds0-19")
    return rows


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
) -> dict[str, Any]:
    if (
        merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("status") != "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
        or merged.get("phase") != EXPECTED_PHASE
        or merged.get("expected_folds") != list(range(20))
        or merged.get("expected_seeds") != [0]
        or int(merged.get("expected_point_epochs", -1)) != 40
        or int(merged.get("expected_calibration_epochs", -1)) != 40
    ):
        raise ValueError("puzzle-set scorer requires one complete unscored merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("puzzle-set merged integrity is not qualified")
    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA or tic2a_merged.get(
        "status"
    ) != "TIC2A_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("puzzle-set scorer requires the corrected TIC2A merge")
    fold_rows = {int(row["outer_fold"]): row for row in merged.get("folds", [])}
    tic_rows = {int(row["outer_fold"]): row for row in tic2a_merged.get("folds", [])}
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
            _load_prediction(
                Path(fold_rows[fold_id]["prediction_artifact"]), fold_id
            ),
            _load_tic2a_absolute(
                Path(tic_rows[fold_id]["prediction_artifact"]), fold_id
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
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "v13_parent_and_feature41_replay_at_5e_7": True,
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
    assert_score_authority(args.repo_root.resolve())
    if args.out_json.exists():
        raise FileExistsError("puzzle-set refuses to overwrite its complete score")
    result = score_complete(
        json.loads(args.merged_json.read_text(encoding="utf-8")),
        json.loads(args.tic2a_merged_json.read_text(encoding="utf-8")),
        json.loads(args.v13_score_json.read_text(encoding="utf-8")),
        args.m2_csv,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
