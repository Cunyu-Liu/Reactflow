#!/usr/bin/env python3
"""Score one complete V12M3 universe after the score-blind merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v12 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v12 import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v10 import _load_tic2a_absolute
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.score_model_rescue_v9 import TIC2A_MERGED_SCHEMA
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v12_score.v1"


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    if active["authority"]["current_phase"] != "V12M3" or active.get(
        "runnable_phases"
    ) != ["V12M3"]:
        raise RuntimeError("V12 scorer is closed outside the complete V12M3 authority")
    if active.get("training_allowed") is not False or active.get(
        "candidate_model_training_allowed"
    ) is not False:
        raise RuntimeError("V12 training must be closed before scientific scoring")
    if active.get("held_score_read_allowed") != "V12_COMPLETE_MERGE_SCORE_ONCE_ONLY":
        raise RuntimeError("V12 complete score authority is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V12 partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V12 external outcomes must remain locked")


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    required_true = (
        "complete_fold_seed_universe",
        "unique_fold_seed_pairs",
        "prediction_only_schema",
        "target_identity_exact",
        "inner_crossfit_complete_all_runs",
        "outer_held_target_excluded_from_gate_fit_all_runs",
        "four_parameter_monotone_gate_all_runs",
        "exact_v11_parent_replay_all_runs",
        "v10_median_residual_family_all_runs",
        "median_constraint_all_runs",
    )
    required_false = ("partial_scores_inspected", "external_outcome_accessed")
    return all(integrity.get(name) is True for name in required_true) and all(
        integrity.get(name) is False for name in required_false
    )


def _load_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: np.asarray(handle[name]) for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V12 prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold} or set(
        map(int, prediction["seed"])
    ) != {0}:
        raise ValueError(f"V12 fold or seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate V12 keys in {path}")
    return prediction


def _central_covered(
    target: np.ndarray,
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    level: float,
) -> np.ndarray:
    cdf = np.sum(weights * ndtr((target[:, None] - locations) / scales), axis=1)
    lower = (1.0 - level) / 2.0
    return ((cdf >= lower) & (cdf <= 1.0 - lower)).astype(np.float64)


def score_fold(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
    tic2a_absolute: dict[str, float],
) -> dict[str, Any]:
    index = {str(key): row for row, key in enumerate(prediction["keys"])}
    expected = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    if set(index) != expected or set(tic2a_absolute) != expected:
        raise ValueError("V12/TIC2A registered key universes are not exact")
    metric_names = (
        "feature41_signed_delta_mae",
        "parent_v11_signed_delta_mae",
        "candidate_signed_delta_mae",
        "feature41_absolute_delta_mae",
        "parent_v11_point_absolute_delta_mae",
        "candidate_point_absolute_delta_mae",
        "candidate_distribution_absolute_delta_mae",
        "historical_v10_distribution_absolute_delta_mae",
        "feature41_crps",
        "parent_v11_crps",
        "candidate_crps",
        "historical_v10_crps",
        "feature41_coverage68",
        "candidate_coverage68",
        "feature41_coverage95",
        "candidate_coverage95",
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
            for name in ("feature41", "parent", "candidate", "historical_v10")
        }
        feature_weights, feature_locations, feature_scales = distributions["feature41"]
        candidate_weights, candidate_locations, candidate_scales = distributions[
            "candidate"
        ]
        arrays = {
            "feature41_signed_delta_mae": np.abs(
                signed - prediction["feature41_point"][rows]
            ),
            "parent_v11_signed_delta_mae": np.abs(
                signed - prediction["v11_parent_point"][rows]
            ),
            "candidate_signed_delta_mae": np.abs(
                signed - prediction["candidate_point"][rows]
            ),
            "feature41_absolute_delta_mae": np.abs(
                absolute - np.asarray([tic2a_absolute[key] for key in keys])
            ),
            "parent_v11_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["v11_parent_point"][rows])
            ),
            "candidate_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["candidate_point"][rows])
            ),
            "candidate_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["candidate_expected_absolute_delta"][rows]
            ),
            "historical_v10_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["historical_v10_expected_absolute_delta"][rows]
            ),
            "feature41_crps": weighted_gaussian_mixture_crps(
                feature_locations, feature_scales, feature_weights, signed
            ),
            "parent_v11_crps": weighted_gaussian_mixture_crps(
                distributions["parent"][1],
                distributions["parent"][2],
                distributions["parent"][0],
                signed,
            ),
            "candidate_crps": weighted_gaussian_mixture_crps(
                candidate_locations, candidate_scales, candidate_weights, signed
            ),
            "historical_v10_crps": weighted_gaussian_mixture_crps(
                distributions["historical_v10"][1],
                distributions["historical_v10"][2],
                distributions["historical_v10"][0],
                signed,
            ),
            "feature41_coverage68": _central_covered(
                signed, feature_weights, feature_locations, feature_scales, 0.68
            ),
            "candidate_coverage68": _central_covered(
                signed, candidate_weights, candidate_locations, candidate_scales, 0.68
            ),
            "feature41_coverage95": _central_covered(
                signed, feature_weights, feature_locations, feature_scales, 0.95
            ),
            "candidate_coverage95": _central_covered(
                signed, candidate_weights, candidate_locations, candidate_scales, 0.95
            ),
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
            "registered_prediction_coverage": 1.0,
            "failure_rate": 0.0,
            "n_unexpected_prediction_keys": 0,
        }
    )
    return result


def score_complete(
    merged: dict[str, Any], tic2a_merged: dict[str, Any], m2_csv: Path
) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V12M3_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V12 scorer requires one complete V12M3 merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("V12 merged integrity is not qualified")
    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA or tic2a_merged.get(
        "status"
    ) != "TIC2A_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V12 scorer requires the corrected TIC2A merge")
    v12_rows = {int(row["outer_fold"]): row for row in merged["folds"]}
    tic_rows = {int(row["outer_fold"]): row for row in tic2a_merged["folds"]}
    if sorted(v12_rows) != list(range(20)) or sorted(tic_rows) != list(range(20)):
        raise ValueError("V12 scorer requires folds0-19 in both universes")
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V12 scorer requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    rows = []
    for fold_id in range(20):
        fold = folds[fold_id]
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        score = score_fold(
            univ,
            held_records,
            _load_prediction(Path(v12_rows[fold_id]["prediction_artifact"]), fold_id),
            _load_tic2a_absolute(
                Path(tic_rows[fold_id]["prediction_artifact"]), fold_id
            ),
        )
        score["outer_fold"] = fold_id
        score["held_puzzle"] = str(fold.held_puzzle)
        rows.append(score)
    return {
        "schema_version": SCHEMA,
        "phase": "V12M3",
        "status": "V12M3_COMPLETE_SCORE_PASS",
        "scores": rows,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_score_authority(args.repo_root.resolve())
    if args.out_json.exists():
        raise FileExistsError(f"V12 refuses to overwrite its one complete score: {args.out_json}")
    result = score_complete(
        json.loads(args.merged_json.read_text()),
        json.loads(args.tic2a_merged_json.read_text()),
        args.m2_csv,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
