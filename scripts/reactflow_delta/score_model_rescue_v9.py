#!/usr/bin/env python3
"""Score the complete frozen V9M2 universe once under the common estimand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v9 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v9 import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v9_score.v1"
TIC2A_MERGED_SCHEMA = "reactflow_delta.target_identity_corrected_baseline_merged.v1"
TIC2A_PREDICTION_SCHEMA = (
    "reactflow_delta.target_identity_corrected_baseline_prediction.v1"
)


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    """Validate complete-merge invariants using their recorded semantics."""
    required_true = (
        "complete_fold_universe",
        "unique_folds",
        "prediction_only_schema",
        "target_identity_exact",
        "v8_mean_replay_all_folds",
        "tic2a_feature41_replay_all_folds",
        "identical_residual_family_all_folds",
        "zero_mean_residual_all_folds",
    )
    required_false = (
        "partial_scores_inspected",
        "external_outcome_accessed",
    )
    return all(integrity.get(name) is True for name in required_true) and all(
        integrity.get(name) is False for name in required_false
    )


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V9M3":
        raise RuntimeError("V9 scorer is closed outside V9M3")
    if active.get("runnable_phases") != ["V9M3"]:
        raise RuntimeError("V9M3 must be the only runnable phase")
    if active.get("training_allowed") is not False:
        raise RuntimeError("training must be closed during V9M3 scoring")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("complete V9M3 score access is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial V9 scores must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V9M3 requires external outcomes locked")


def _load_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V9 prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold}:
        raise ValueError(f"V9 prediction fold mismatch in {path}")
    keys = prediction["keys"]
    if not np.array_equal(keys, prediction["biological_scoring_key"]):
        raise ValueError(f"V9 prediction biological keys disagree in {path}")
    if len(set(map(str, keys))) != len(keys):
        raise ValueError(f"V9 prediction keys are duplicated in {path}")
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
        raise ValueError(f"invalid TIC2A absolute prediction in {path}")
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
        raise ValueError("V9/TIC2A registered key universes are not exact")
    metric_names = (
        "feature41_signed_delta_mae",
        "meanaligned_signed_delta_mae",
        "feature41_absolute_delta_mae",
        "meanaligned_absolute_delta_mae",
        "feature41_distribution_absolute_delta_mae",
        "feature41_crps",
        "meanaligned_crps",
        "feature41_coverage68",
        "meanaligned_coverage68",
        "feature41_coverage95",
        "meanaligned_coverage95",
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
        qualified = construct.wt_observed.astype(bool) & np.isfinite(target)
        positions = np.flatnonzero(qualified)
        keys = [_bio_key(univ, record, int(position)) for position in positions]
        rows = np.asarray([index[key] for key in keys], dtype=np.int64)
        signed = target[positions] - construct.wt_reactivity[positions]
        absolute = np.abs(signed)
        f_weights = prediction["feature41_weights"][rows]
        f_locations = prediction["feature41_locations"][rows]
        f_scales = prediction["feature41_scales"][rows]
        m_weights = prediction["meanaligned_weights"][rows]
        m_locations = prediction["meanaligned_locations"][rows]
        m_scales = prediction["meanaligned_scales"][rows]
        f_crps = weighted_gaussian_mixture_crps(
            f_locations, f_scales, f_weights, signed
        )
        m_crps = weighted_gaussian_mixture_crps(
            m_locations, m_scales, m_weights, signed
        )
        f_cov68 = _central_covered(
            signed, f_weights, f_locations, f_scales, 0.68
        )
        m_cov68 = _central_covered(
            signed, m_weights, m_locations, m_scales, 0.68
        )
        f_cov95 = _central_covered(
            signed, f_weights, f_locations, f_scales, 0.95
        )
        m_cov95 = _central_covered(
            signed, m_weights, m_locations, m_scales, 0.95
        )
        arrays = {
            "feature41_signed_delta_mae": np.abs(
                signed - prediction["feature41_delta_mean"][rows]
            ),
            "meanaligned_signed_delta_mae": np.abs(
                signed - prediction["meanaligned_delta_mean"][rows]
            ),
            "feature41_absolute_delta_mae": np.abs(
                absolute - np.asarray([tic2a_absolute[key] for key in keys])
            ),
            "meanaligned_absolute_delta_mae": np.abs(
                absolute - prediction["meanaligned_expected_absolute_delta"][rows]
            ),
            "feature41_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["feature41_expected_absolute_delta"][rows]
            ),
            "feature41_crps": f_crps,
            "meanaligned_crps": m_crps,
            "feature41_coverage68": f_cov68,
            "meanaligned_coverage68": m_cov68,
            "feature41_coverage95": f_cov95,
            "meanaligned_coverage95": m_cov95,
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
        "V9M2_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V9 scorer requires one complete V9M2 merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("V9 merged integrity is not fully qualified")
    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA or tic2a_merged.get(
        "status"
    ) != "TIC2A_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V9 scorer requires the complete corrected TIC2A merge")
    v9_rows = {int(row["outer_fold"]): row for row in merged["folds"]}
    tic_rows = {int(row["outer_fold"]): row for row in tic2a_merged["folds"]}
    if sorted(v9_rows) != list(range(20)) or sorted(tic_rows) != list(range(20)):
        raise ValueError("V9 scorer requires both fold universes 0 through 19")
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("V9 scorer requires exact target identity")
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
            _load_prediction(Path(v9_rows[fold_id]["prediction_artifact"]), fold_id),
            _load_tic2a_absolute(
                Path(tic_rows[fold_id]["prediction_artifact"]), fold_id
            ),
        )
        score["outer_fold"] = fold_id
        score["held_puzzle"] = str(fold.held_puzzle)
        rows.append(score)
    return {
        "schema_version": SCHEMA,
        "phase": "V9M3",
        "status": "V9M3_COMPLETE_SCORE_PASS",
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
