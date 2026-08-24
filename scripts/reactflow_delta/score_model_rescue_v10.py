#!/usr/bin/env python3
"""Score one complete V10 universe under the frozen hierarchical estimand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v10 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v10 import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.score_model_rescue_v9 import (
    TIC2A_MERGED_SCHEMA,
    TIC2A_PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v10_score.v1"


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    required_true = (
        "complete_fold_universe",
        "unique_folds",
        "prediction_only_schema",
        "target_identity_exact",
        "v8_point_replay_all_folds",
        "tic2a_feature41_replay_all_folds",
        "historical_v9_replay_all_folds",
        "matched_head_families_all_folds",
        "median_constraint_all_folds",
        "outer_train_only_standardization_all_folds",
    )
    required_false = ("partial_scores_inspected", "external_outcome_accessed")
    return all(integrity.get(name) is True for name in required_true) and all(
        integrity.get(name) is False for name in required_false
    )


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V10M3":
        raise RuntimeError("V10 scorer is closed outside V10M3")
    if active.get("runnable_phases") != ["V10M3"]:
        raise RuntimeError("V10M3 must be the only runnable phase")
    if active.get("training_allowed") is not False:
        raise RuntimeError("training must be closed during V10M3")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("complete V10 score access is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial V10 score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V10M3 requires external outcomes locked")


def _load_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V10 prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold}:
        raise ValueError(f"V10 prediction fold mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate V10 prediction keys in {path}")
    return prediction


def _load_tic2a_absolute(path: Path, fold: int) -> dict[str, float]:
    with np.load(path, allow_pickle=True) as handle:
        if str(handle["schema_version"].item()) != TIC2A_PREDICTION_SCHEMA:
            raise ValueError(f"invalid TIC2A prediction schema in {path}")
        if set(map(int, handle["outer_fold"])) != {fold}:
            raise ValueError(f"TIC2A prediction fold mismatch in {path}")
        keys = list(map(str, handle["keys"]))
        values = np.asarray(handle["v6_feature41_absolute_delta"])
    if len(keys) != len(set(keys)) or values.shape != (len(keys),):
        raise ValueError(f"invalid TIC2A absolute predictions in {path}")
    return {key: float(value) for key, value in zip(keys, values)}


def _central_covered(
    target: np.ndarray,
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    level: float,
) -> np.ndarray:
    cdf = np.sum(weights * ndtr((target[:, None] - locations) / scales), axis=1)
    lower = (1.0 - level) / 2.0
    upper = 1.0 - lower
    return ((cdf >= lower) & (cdf <= upper)).astype(np.float64)


def score_fold(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
    tic2a_absolute: dict[str, float],
) -> dict[str, Any]:
    index = {str(key): i for i, key in enumerate(prediction["keys"])}
    expected = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    if set(index) != expected or set(tic2a_absolute) != expected:
        raise ValueError("V10/TIC2A registered key universes are not exact")
    metric_names = (
        "feature41_signed_delta_mae",
        "meanaligned_signed_delta_mae",
        "feature41_absolute_delta_mae",
        "meanaligned_asymmetric_absolute_delta_mae",
        "feature41_asymmetric_crps",
        "meanaligned_asymmetric_crps",
        "meanaligned_symmetric_crps",
        "historical_v9_crps",
        "feature41_asymmetric_coverage68",
        "meanaligned_asymmetric_coverage68",
        "feature41_asymmetric_coverage95",
        "meanaligned_asymmetric_coverage95",
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
        distributions = {}
        for name in (
            "feature41_asymmetric",
            "meanaligned_asymmetric",
            "meanaligned_symmetric",
            "historical_v9",
        ):
            distributions[name] = (
                prediction[f"{name}_weights"][rows],
                prediction[f"{name}_locations"][rows],
                prediction[f"{name}_scales"][rows],
            )
        f_weights, f_locations, f_scales = distributions["feature41_asymmetric"]
        m_weights, m_locations, m_scales = distributions["meanaligned_asymmetric"]
        arrays = {
            "feature41_signed_delta_mae": np.abs(
                signed - prediction["feature41_point"][rows]
            ),
            "meanaligned_signed_delta_mae": np.abs(
                signed - prediction["meanaligned_point"][rows]
            ),
            "feature41_absolute_delta_mae": np.abs(
                absolute - np.asarray([tic2a_absolute[key] for key in keys])
            ),
            "meanaligned_asymmetric_absolute_delta_mae": np.abs(
                absolute
                - prediction["meanaligned_asymmetric_expected_absolute_delta"][rows]
            ),
            "feature41_asymmetric_crps": weighted_gaussian_mixture_crps(
                f_locations, f_scales, f_weights, signed
            ),
            "meanaligned_asymmetric_crps": weighted_gaussian_mixture_crps(
                m_locations, m_scales, m_weights, signed
            ),
            "meanaligned_symmetric_crps": weighted_gaussian_mixture_crps(
                distributions["meanaligned_symmetric"][1],
                distributions["meanaligned_symmetric"][2],
                distributions["meanaligned_symmetric"][0],
                signed,
            ),
            "historical_v9_crps": weighted_gaussian_mixture_crps(
                distributions["historical_v9"][1],
                distributions["historical_v9"][2],
                distributions["historical_v9"][0],
                signed,
            ),
            "feature41_asymmetric_coverage68": _central_covered(
                signed, f_weights, f_locations, f_scales, 0.68
            ),
            "meanaligned_asymmetric_coverage68": _central_covered(
                signed, m_weights, m_locations, m_scales, 0.68
            ),
            "feature41_asymmetric_coverage95": _central_covered(
                signed, f_weights, f_locations, f_scales, 0.95
            ),
            "meanaligned_asymmetric_coverage95": _central_covered(
                signed, m_weights, m_locations, m_scales, 0.95
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
        "V10M2_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V10 scorer requires one complete V10M2 merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("V10 merged integrity is not qualified")
    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA or tic2a_merged.get(
        "status"
    ) != "TIC2A_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V10 scorer requires corrected TIC2A merge")
    v10_rows = {int(row["outer_fold"]): row for row in merged["folds"]}
    tic_rows = {int(row["outer_fold"]): row for row in tic2a_merged["folds"]}
    if sorted(v10_rows) != list(range(20)) or sorted(tic_rows) != list(range(20)):
        raise ValueError("V10 scorer requires folds0-19 in both universes")
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("V10 scorer requires exact target identity")
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
            _load_prediction(Path(v10_rows[fold_id]["prediction_artifact"]), fold_id),
            _load_tic2a_absolute(
                Path(tic_rows[fold_id]["prediction_artifact"]), fold_id
            ),
        )
        score["outer_fold"] = fold_id
        score["held_puzzle"] = str(fold.held_puzzle)
        rows.append(score)
    return {
        "schema_version": SCHEMA,
        "phase": "V10M3",
        "status": "V10M3_COMPLETE_SCORE_PASS",
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
    result = score_complete(
        json.loads(args.merged_json.read_text(encoding="utf-8")),
        json.loads(args.tic2a_merged_json.read_text(encoding="utf-8")),
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
