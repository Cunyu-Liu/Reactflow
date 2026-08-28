#!/usr/bin/env python3
"""Score the complete RNet2-distillation seed-0 screen exactly once.

The scorer first validates all twenty target-free prediction artifacts.  Only
after that complete universe is established does it read the frozen complete
V14 parent score and join OpenKnot M2 outcomes.  It never emits a partial score
or performs qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v14 import SCHEMA as V14_SCORE_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    ARTIFACT_ROOT,
    PROJECT_TASK_ID,
    assert_run_authority,
)


EXPECTED_FOLDS = list(range(20))
EXPECTED_SEED = 0
SCORE_PHASE = "RND4"
SCORE_SCHEMA = "reactflow_delta.independent_rnet_distill_score.v1"
SCORE_STATUS = "RND4_COMPLETE_SCORE_PASS"
MERGED_SCHEMA = "reactflow_delta.independent_rnet_distill_complete_unscored_merge.v1"
MERGED_STATUS = "RND3_COMPLETE_UNSCORED_PREDICTION_MERGE_PASS"
PREDICTION_SCHEMA = "reactflow_delta.independent_rnet_distill_prediction.v1"
SCREEN_DIR = ARTIFACT_ROOT / "rnd3_screen_seed0"
MERGED_PATH = SCREEN_DIR / "rnet_distill_complete_unscored_merge.json"
SCORE_PATH = SCREEN_DIR / "rnet_distill_complete_score.json"
QUALIFICATION_PATH = SCREEN_DIR / "rnet_distill_qualification.json"
M2_PATH = Path(
    "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/"
    "reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
)
V14_SCORE_PATH = Path(
    "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/"
    "v14m3_screen_seed0/v14m3_complete_score.json"
)

MIXTURE_NAMES = ("feature41", "candidate", "null", "historical_v10")
POINT_NAMES = ("feature41", "v8", "candidate", "null")
EXPECTED_PREDICTION_FIELDS = {
    "schema_version",
    "keys",
    "biological_scoring_key",
    "outer_fold",
    "seed",
    "registered_status",
    *(f"{name}_point" for name in POINT_NAMES),
    *(
        f"{name}_{suffix}"
        for name in MIXTURE_NAMES
        for suffix in ("weights", "locations", "scales", "expected_absolute_delta")
    ),
}
FORBIDDEN_PREDICTION_FIELDS = {
    "target",
    "targets",
    "held_target",
    "score",
    "scores",
    "loss",
    "losses",
    "qualified",
    "qualified_mask",
}
EXPECTED_MERGED_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "folds",
    "merge_integrity",
}
MERGE_TRUE_INVARIANTS = {
    "complete_fold_seed_universe",
    "unique_fold_seed_pairs",
    "prediction_only_schema",
    "target_free_all_runs",
    "target_identity_exact",
    "pretrained_source_pair_bound_all_runs",
    "residual_heads_equal_before_downstream_all_runs",
    "pretrained_encoders_different_before_downstream_all_runs",
    "same_downstream_training_order_and_dropout_stream_all_runs",
    "point_frozen_during_calibration_all_runs",
    "v10_residual_family_all_runs",
    "feature41_replay_all_runs",
    "authoritative_feature41_seed0_comparator_provenance_all_runs",
    "median_constraint_all_runs",
    "cuda_only_all_runs",
}
MERGE_FALSE_INVARIANTS = {
    "partial_scores_inspected",
    "external_outcome_accessed",
}

SCORE_ROW_FIELDS = {
    "outer_fold",
    "held_puzzle",
    "feature41_signed_delta_mae",
    "candidate_signed_delta_mae",
    "null_signed_delta_mae",
    "historical_v14_signed_delta_mae",
    "feature41_point_absolute_delta_mae",
    "candidate_point_absolute_delta_mae",
    "null_point_absolute_delta_mae",
    "historical_v14_point_absolute_delta_mae",
    "feature41_crps",
    "candidate_crps",
    "null_crps",
    "historical_v14_crps",
    "feature41_distribution_absolute_delta_mae",
    "candidate_distribution_absolute_delta_mae",
    "null_distribution_absolute_delta_mae",
    "historical_v10_distribution_absolute_delta_mae",
    "feature41_coverage95",
    "candidate_coverage95",
    "null_coverage95",
    "n_qualified_positions",
    "n_registered_expected",
    "n_registered_observed",
    "registered_prediction_coverage",
    "failure_rate",
    "failed_rows",
    "n_duplicate_prediction_keys",
    "n_unexpected_prediction_keys",
    "score_integrity_pass",
}


class ScoreIntegrityError(RuntimeError):
    """The complete scientific score cannot be established."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _active_contract(repo_root: Path) -> dict[str, Any]:
    value = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise RuntimeError("independent RNet scorer requires a mapping active contract")
    return value


def assert_score_authority(
    repo_root: Path,
    *,
    merged_json: Path,
    m2_csv: Path,
    historical_v14_score_json: Path,
    out_json: Path,
) -> dict[str, Any]:
    """Require exact RND4 authority and its separately frozen artifact paths."""

    assert_run_authority(repo_root, SCORE_PHASE)
    active = _active_contract(repo_root)
    if active.get("project_task_id") != PROJECT_TASK_ID:
        raise RuntimeError("independent RNet scorer is not the active project")
    authority = active.get("authority", {})
    expected = {
        "screen_prediction_dir": SCREEN_DIR,
        "complete_unscored_merge_path": MERGED_PATH,
        "m2_csv_path": M2_PATH,
        "historical_v14_score_path": V14_SCORE_PATH,
        "complete_score_path": SCORE_PATH,
        "qualification_path": QUALIFICATION_PATH,
    }
    for name, path in expected.items():
        if name not in authority or _resolved(authority[name]) != path.resolve():
            raise RuntimeError(f"RND4 active authority {name} is not exact")
    provided = {
        "complete_unscored_merge_path": merged_json,
        "m2_csv_path": m2_csv,
        "historical_v14_score_path": historical_v14_score_json,
        "complete_score_path": out_json,
    }
    for name, path in provided.items():
        if _resolved(path) != expected[name].resolve():
            raise RuntimeError(f"RND4 CLI {name} differs from active authority")
    return active


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    return (
        set(integrity) == MERGE_TRUE_INVARIANTS | MERGE_FALSE_INVARIANTS
        and all(integrity.get(name) is True for name in MERGE_TRUE_INVARIANTS)
        and all(integrity.get(name) is False for name in MERGE_FALSE_INVARIANTS)
    )


def _validated_merge_rows(merged: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if (
        set(merged) != EXPECTED_MERGED_FIELDS
        or merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("phase") != "RND3"
        or merged.get("status") != MERGED_STATUS
        or not merged_integrity_pass(merged.get("merge_integrity", {}))
    ):
        raise ScoreIntegrityError(
            "RND4 requires the exact complete target-free RND3 merge"
        )
    rows = merged.get("folds")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FOLDS):
        raise ScoreIntegrityError("RND4 requires exactly twenty fold artifacts")
    try:
        by_fold = {int(row["outer_fold"]): row for row in rows}
    except (KeyError, TypeError, ValueError) as error:
        raise ScoreIntegrityError("RND3 merge fold rows are malformed") from error
    if len(by_fold) != len(rows) or sorted(by_fold) != EXPECTED_FOLDS:
        raise ScoreIntegrityError("RND3 merge fold universe is not unique folds0-19")
    for fold, row in by_fold.items():
        if (
            int(row.get("seed", -1)) != EXPECTED_SEED
            or str(row.get("phase")) != "RND3"
            or str(row.get("held_puzzle")) != f"P{fold + 1:02d}"
        ):
            raise ScoreIntegrityError(f"RND3 fold{fold} identity is not exact")
    return by_fold


def _load_prediction(path: Path, *, fold: int, prediction_root: Path) -> dict[str, np.ndarray]:
    expected_name = f"rnet_distill_predictions_fold{fold}_seed0.npz"
    if path.name != expected_name or path.resolve().parent != prediction_root.resolve():
        raise ScoreIntegrityError(f"RND3 fold{fold} prediction path is not canonical")
    try:
        with np.load(path, allow_pickle=True) as handle:
            fields = set(handle.files)
            if fields != EXPECTED_PREDICTION_FIELDS or fields & FORBIDDEN_PREDICTION_FIELDS:
                raise ScoreIntegrityError(
                    f"RND3 fold{fold} prediction field universe changed"
                )
            result = {name: np.asarray(handle[name]) for name in fields}
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ScoreIntegrityError):
            raise
        raise ScoreIntegrityError(f"RND3 fold{fold} prediction cannot be read") from error

    if str(result["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ScoreIntegrityError(f"RND3 fold{fold} prediction schema changed")
    keys = list(map(str, result["keys"]))
    n_rows = len(keys)
    if (
        n_rows < 1
        or len(keys) != len(set(keys))
        or result["biological_scoring_key"].shape != (n_rows,)
        or keys != list(map(str, result["biological_scoring_key"]))
        or result["outer_fold"].shape != (n_rows,)
        or set(map(int, result["outer_fold"])) != {fold}
        or result["seed"].shape != (n_rows,)
        or set(map(int, result["seed"])) != {EXPECTED_SEED}
        or result["registered_status"].shape != (n_rows,)
        or set(map(str, result["registered_status"])) != {"covered"}
    ):
        raise ScoreIntegrityError(f"RND3 fold{fold} registered identity changed")
    for name in POINT_NAMES:
        values = result[f"{name}_point"]
        if values.shape != (n_rows,) or not np.isfinite(values).all():
            raise ScoreIntegrityError(f"RND3 fold{fold} {name} point is invalid")
    for name in MIXTURE_NAMES:
        weights = result[f"{name}_weights"]
        locations = result[f"{name}_locations"]
        scales = result[f"{name}_scales"]
        expected_absolute = result[f"{name}_expected_absolute_delta"]
        if (
            weights.ndim != 2
            or weights.shape[0] != n_rows
            or weights.shape[1] < 1
            or locations.shape != weights.shape
            or scales.shape != weights.shape
            or expected_absolute.shape != (n_rows,)
            or not np.isfinite(weights).all()
            or not np.isfinite(locations).all()
            or not np.isfinite(scales).all()
            or not np.isfinite(expected_absolute).all()
            or np.any(weights < 0.0)
            or np.any(scales <= 0.0)
            or np.any(expected_absolute < 0.0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
        ):
            raise ScoreIntegrityError(
                f"RND3 fold{fold} {name} distribution is invalid"
            )
    return result


def load_complete_predictions(
    merged: dict[str, Any], *, prediction_root: Path
) -> dict[int, dict[str, np.ndarray]]:
    """Validate every target-free fold before any held outcome is read."""

    rows = _validated_merge_rows(merged)
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for fold in EXPECTED_FOLDS:
        source = rows[fold]
        prediction_path = Path(str(source.get("prediction_artifact", "")))
        predictions[fold] = _load_prediction(
            prediction_path, fold=fold, prediction_root=prediction_root
        )
    return predictions


def _v14_parent_rows(score: dict[str, Any]) -> dict[int, dict[str, Any]]:
    expected_top = {
        "schema_version": V14_SCORE_SCHEMA,
        "phase": "V14M3",
        "status": "V14M3_COMPLETE_SCORE_PASS",
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }
    if any(score.get(name) != value for name, value in expected_top.items()):
        raise ScoreIntegrityError("RND4 requires the frozen complete V14 score")
    rows = score.get("scores")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FOLDS):
        raise ScoreIntegrityError("frozen V14 score is not complete")
    try:
        by_fold = {int(row["outer_fold"]): row for row in rows}
    except (KeyError, TypeError, ValueError) as error:
        raise ScoreIntegrityError("frozen V14 score rows are malformed") from error
    if len(by_fold) != len(rows) or sorted(by_fold) != EXPECTED_FOLDS:
        raise ScoreIntegrityError("frozen V14 score fold universe changed")
    fields = (
        "candidate_signed_delta_mae",
        "candidate_point_absolute_delta_mae",
        "candidate_crps",
    )
    for fold, row in by_fold.items():
        try:
            values = np.asarray([float(row[name]) for name in fields])
        except (KeyError, TypeError, ValueError) as error:
            raise ScoreIntegrityError(f"frozen V14 fold{fold} metric is missing") from error
        if (
            str(row.get("held_puzzle")) != f"P{fold + 1:02d}"
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise ScoreIntegrityError(f"frozen V14 fold{fold} identity or metric changed")
    return by_fold


def _central_coverage95(
    target: np.ndarray,
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    cdf = np.sum(weights * ndtr((target[:, None] - locations) / scales), axis=1)
    return ((cdf >= 0.025) & (cdf <= 0.975)).astype(np.float64)


def score_fold(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, Any]:
    keys = list(map(str, prediction["keys"]))
    index = {key: row for row, key in enumerate(keys)}
    expected = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    observed = set(index)
    if observed != expected or len(index) != len(keys):
        raise ScoreIntegrityError("registered prediction universe is not exact")

    metric_names = (
        "feature41_signed_delta_mae",
        "candidate_signed_delta_mae",
        "null_signed_delta_mae",
        "feature41_point_absolute_delta_mae",
        "candidate_point_absolute_delta_mae",
        "null_point_absolute_delta_mae",
        "feature41_crps",
        "candidate_crps",
        "null_crps",
        "feature41_distribution_absolute_delta_mae",
        "candidate_distribution_absolute_delta_mae",
        "null_distribution_absolute_delta_mae",
        "historical_v10_distribution_absolute_delta_mae",
        "feature41_coverage95",
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
        row_keys = [_bio_key(univ, record, int(position)) for position in positions]
        rows = np.asarray([index[key] for key in row_keys], dtype=np.int64)
        signed = target[positions] - construct.wt_reactivity[positions]
        absolute = np.abs(signed)
        distributions = {
            name: (
                prediction[f"{name}_weights"][rows],
                prediction[f"{name}_locations"][rows],
                prediction[f"{name}_scales"][rows],
            )
            for name in ("feature41", "candidate", "null")
        }
        arrays = {
            "feature41_signed_delta_mae": np.abs(
                signed - prediction["feature41_point"][rows]
            ),
            "candidate_signed_delta_mae": np.abs(
                signed - prediction["candidate_point"][rows]
            ),
            "null_signed_delta_mae": np.abs(signed - prediction["null_point"][rows]),
            "feature41_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["feature41_point"][rows])
            ),
            "candidate_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["candidate_point"][rows])
            ),
            "null_point_absolute_delta_mae": np.abs(
                absolute - np.abs(prediction["null_point"][rows])
            ),
            "feature41_crps": weighted_gaussian_mixture_crps(
                distributions["feature41"][1],
                distributions["feature41"][2],
                distributions["feature41"][0],
                signed,
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
            "feature41_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["feature41_expected_absolute_delta"][rows]
            ),
            "candidate_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["candidate_expected_absolute_delta"][rows]
            ),
            "null_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["null_expected_absolute_delta"][rows]
            ),
            "historical_v10_distribution_absolute_delta_mae": np.abs(
                absolute - prediction["historical_v10_expected_absolute_delta"][rows]
            ),
            "feature41_coverage95": _central_coverage95(
                signed, *distributions["feature41"]
            ),
            "candidate_coverage95": _central_coverage95(
                signed, *distributions["candidate"]
            ),
            "null_coverage95": _central_coverage95(signed, *distributions["null"]),
        }
        for name, array in arrays.items():
            values[name].update(
                {key: float(value) for key, value in zip(row_keys, array)}
            )
        n_qualified += len(row_keys)
    metrics = {name: _puzzle_macro(data) for name, data in values.items()}
    finite = n_qualified > 0 and np.isfinite(
        np.asarray(list(metrics.values()), dtype=np.float64)
    ).all()
    if not finite:
        raise ScoreIntegrityError("fold has no complete finite qualified score")
    return {
        **metrics,
        "n_qualified_positions": n_qualified,
        "n_registered_expected": len(expected),
        "n_registered_observed": len(observed),
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "failed_rows": 0,
        "n_duplicate_prediction_keys": 0,
        "n_unexpected_prediction_keys": 0,
        "score_integrity_pass": True,
    }


def score_complete(
    *,
    merged: dict[str, Any],
    historical_v14_score: dict[str, Any],
    m2_csv: Path,
    prediction_root: Path,
    validated_predictions: dict[int, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    # This order is the partial-score boundary: all prediction artifacts are
    # validated before either prior scores or held targets are inspected.
    predictions = (
        load_complete_predictions(merged, prediction_root=prediction_root)
        if validated_predictions is None
        else validated_predictions
    )
    if sorted(predictions) != EXPECTED_FOLDS:
        raise ScoreIntegrityError("RND4 validated prediction universe changed")
    parent_rows = _v14_parent_rows(historical_v14_score)

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if (
        identity.get("n_canonical_mutant_full_profiles") != 13976
        or identity.get("canonical_mutant_full_profile_identity")
        != "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise ScoreIntegrityError("RND4 requires the exact canonical M2 target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(row.outer_fold): row for row in split["folds"]}
    if sorted(folds) != EXPECTED_FOLDS:
        raise ScoreIntegrityError("RND4 split_v4 fold universe changed")

    score_rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        held_puzzle = str(folds[fold].held_puzzle)
        parent = parent_rows[fold]
        if held_puzzle != str(parent.get("held_puzzle")):
            raise ScoreIntegrityError(f"RND4/V14 held puzzle differs at fold{fold}")
        held_records = [record for record in records if record.puzzle == held_puzzle]
        row = score_fold(univ, held_records, predictions[fold])
        row.update(
            {
                "outer_fold": fold,
                "held_puzzle": held_puzzle,
                "historical_v14_signed_delta_mae": float(
                    parent["candidate_signed_delta_mae"]
                ),
                "historical_v14_point_absolute_delta_mae": float(
                    parent["candidate_point_absolute_delta_mae"]
                ),
                "historical_v14_crps": float(parent["candidate_crps"]),
            }
        )
        if set(row) != SCORE_ROW_FIELDS:
            raise ScoreIntegrityError(f"RND4 fold{fold} score schema is incomplete")
        score_rows.append(row)

    return {
        "schema_version": SCORE_SCHEMA,
        "phase": SCORE_PHASE,
        "status": SCORE_STATUS,
        "scores": score_rows,
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
        "source_exposure_status": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
    }


def _write_json_once(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("RND4 refuses to overwrite its canonical complete score")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--historical-v14-score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    merged_json = args.merged_json.resolve()
    m2_csv = args.m2_csv.resolve()
    historical_v14_score_json = args.historical_v14_score_json.resolve()
    out_json = args.out_json.resolve()
    try:
        assert_score_authority(
            repo_root,
            merged_json=merged_json,
            m2_csv=m2_csv,
            historical_v14_score_json=historical_v14_score_json,
            out_json=out_json,
        )
        if out_json.exists() or QUALIFICATION_PATH.exists():
            raise FileExistsError(
                "RND4 score or downstream qualification already exists; refusing rerun"
            )
        merged = json.loads(merged_json.read_text(encoding="utf-8"))
        # Complete target-free validation happens before this complete parent is read.
        predictions = load_complete_predictions(
            merged, prediction_root=merged_json.parent
        )
        historical_v14_score = json.loads(
            historical_v14_score_json.read_text(encoding="utf-8")
        )
        result = score_complete(
            merged=merged,
            historical_v14_score=historical_v14_score,
            m2_csv=m2_csv,
            prediction_root=merged_json.parent,
            validated_predictions=predictions,
        )
        _write_json_once(out_json, result)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "RND4_COMPLETE_SCORE_ENGINEERING_INDETERMINATE",
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
