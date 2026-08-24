#!/usr/bin/env python3
"""Run prediction-only folds for the fixed v6 constrained-feature probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache as UnconstrainedFeatureCache,
    fit_weighted_standardized_ridge,
    predict_weighted_ridge,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    BASELINE_PROBE_FEATURE_NAMES,
    CANDIDATE_PROBE_FEATURE_NAMES,
    ConstrainedFeatureCache,
    accumulate_train_stats,
    prediction_features,
    validate_cache_alignment,
)
from scripts.reactflow_delta.run_model_rescue_v5_probe import (
    PREDICTION_SCHEMA as V5_PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


FOLD_SCHEMA = "reactflow_delta.model_rescue_v6_probe_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v6_probe_prediction.v1"
RIDGE_ALPHA = 1.0
REPLAY_ATOL = 1e-12


def assert_probe_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V6M2":
        raise RuntimeError("v6 eligibility probe is closed outside active V6M2")
    if active.get("training_allowed") != "FIXED_WEIGHTED_RIDGE_ELIGIBILITY_ONLY":
        raise RuntimeError("v6 fixed-ridge training authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("v6 prediction requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("v6 prediction requires partial scores prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v6 prediction requires external outcomes locked")


def _model_to_json(model: dict[str, np.ndarray | float]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def predict_registered_held(
    univ: Any,
    records: list[Any],
    unconstrained_cache: UnconstrainedFeatureCache,
    constrained_cache: ConstrainedFeatureCache,
    baseline_model: dict[str, np.ndarray | float],
    candidate_model: dict[str, np.ndarray | float],
    *,
    outer_fold: int,
) -> dict[str, np.ndarray]:
    keys: list[str] = []
    baseline_signed: list[float] = []
    baseline_absolute: list[float] = []
    candidate_signed: list[float] = []
    candidate_absolute: list[float] = []
    for record in records:
        construct = univ.get_construct(record.construct_id)
        receiver = np.arange(len(construct.sequence), dtype=np.int64)
        baseline, candidate = prediction_features(
            construct,
            record,
            receiver,
            unconstrained_cache,
            constrained_cache,
        )
        baseline_prediction = predict_weighted_ridge(baseline_model, baseline)
        candidate_prediction = predict_weighted_ridge(candidate_model, candidate)
        for position in receiver:
            keys.append(_bio_key(univ, record, int(position)))
        baseline_signed.extend(baseline_prediction[:, 0].tolist())
        baseline_absolute.extend(baseline_prediction[:, 1].tolist())
        candidate_signed.extend(candidate_prediction[:, 0].tolist())
        candidate_absolute.extend(candidate_prediction[:, 1].tolist())
    n = len(keys)
    result = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(n, outer_fold, dtype=np.int64),
        "baseline_signed_delta": np.asarray(baseline_signed, dtype=np.float64),
        "baseline_absolute_delta": np.asarray(baseline_absolute, dtype=np.float64),
        "candidate_signed_delta": np.asarray(candidate_signed, dtype=np.float64),
        "candidate_absolute_delta": np.asarray(candidate_absolute, dtype=np.float64),
        "registered_status": np.full(n, "covered", dtype=object),
    }
    if len(set(keys)) != n:
        raise RuntimeError("v6 probe prediction contains duplicate biological keys")
    for name in (
        "baseline_signed_delta",
        "baseline_absolute_delta",
        "candidate_signed_delta",
        "candidate_absolute_delta",
    ):
        if not np.isfinite(result[name]).all():
            raise RuntimeError(f"v6 probe produced non-finite {name}")
    return result


def assert_v5_baseline_replay(
    prediction: dict[str, np.ndarray],
    reference_path: Path,
) -> None:
    with np.load(reference_path, allow_pickle=True) as handle:
        reference = {name: handle[name] for name in handle.files}
    if str(reference.get("schema_version", np.asarray("")).item()) != V5_PREDICTION_SCHEMA:
        raise ValueError("v6 baseline replay requires the frozen v5 prediction schema")
    if not np.array_equal(prediction["keys"], reference["keys"]):
        raise ValueError("v6 baseline and v5 candidate biological keys differ")
    comparisons = (
        ("baseline_signed_delta", "candidate_signed_delta"),
        ("baseline_absolute_delta", "candidate_absolute_delta"),
    )
    for current_name, reference_name in comparisons:
        if not np.allclose(
            prediction[current_name],
            reference[reference_name],
            atol=REPLAY_ATOL,
            rtol=0.0,
        ):
            maximum = float(
                np.max(np.abs(prediction[current_name] - reference[reference_name]))
            )
            raise ValueError(
                f"v6 baseline failed v5 candidate replay for {current_name}: max={maximum}"
            )


def run_fold(
    univ: M2Universe,
    records: list[Any],
    unconstrained_cache: UnconstrainedFeatureCache,
    constrained_cache: ConstrainedFeatureCache,
    fold: Any,
    out_dir: Path,
    v5_reference_path: Path,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    baseline_stats, candidate_stats, counts = accumulate_train_stats(
        univ,
        train_records,
        unconstrained_cache,
        constrained_cache,
    )
    baseline_model = fit_weighted_standardized_ridge(baseline_stats, RIDGE_ALPHA)
    candidate_model = fit_weighted_standardized_ridge(candidate_stats, RIDGE_ALPHA)
    prediction = predict_registered_held(
        univ,
        held_records,
        unconstrained_cache,
        constrained_cache,
        baseline_model,
        candidate_model,
        outer_fold=int(fold.outer_fold),
    )
    assert_v5_baseline_replay(prediction, v5_reference_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / f"v6_probe_predictions_fold{fold.outer_fold}.npz"
    model_path = out_dir / f"v6_probe_models_fold{fold.outer_fold}.json"
    np.savez_compressed(prediction_path, **prediction)
    model_path.write_text(
        json.dumps(
            {
                "ridge_alpha": RIDGE_ALPHA,
                "baseline_feature_names": list(BASELINE_PROBE_FEATURE_NAMES),
                "candidate_feature_names": list(CANDIDATE_PROBE_FEATURE_NAMES),
                "baseline": _model_to_json(baseline_model),
                "candidate": _model_to_json(candidate_model),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": "V6M2",
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "prediction_artifact": str(prediction_path),
        "model_artifact": str(model_path),
        "v5_reference_prediction": str(v5_reference_path),
        "n_registered_prediction_rows": len(prediction["keys"]),
        **counts,
        "v5_baseline_replay_pass": True,
        "held_target_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }
    result_path = out_dir / f"v6_probe_fold_result_fold{fold.outer_fold}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parse_folds(value: str) -> list[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 0 or fold >= 20 for fold in folds):
        raise ValueError("folds must be a non-empty subset of 0 through 19")
    return folds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--v5-reference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    assert_probe_authority(args.repo_root.resolve())
    univ = M2Universe(args.m2_csv)
    univ.build()
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    unconstrained = UnconstrainedFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    try:
        validate_cache_alignment(unconstrained, constrained)
        for fold_id in _parse_folds(args.folds):
            result_path = args.out_dir / f"v6_probe_fold_result_fold{fold_id}.json"
            if result_path.exists():
                raise FileExistsError(f"refusing to overwrite completed fold {fold_id}")
            reference = args.v5_reference_dir / f"v5_probe_predictions_fold{fold_id}.npz"
            if not reference.exists():
                raise FileNotFoundError(reference)
            run_fold(
                univ,
                records,
                unconstrained,
                constrained,
                fold_map[fold_id],
                args.out_dir,
                reference,
            )
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
