#!/usr/bin/env python3
"""Generate prediction-only folds for the frozen post-V13 route diagnostics."""

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
    CANDIDATE_PROBE_FEATURE_NAMES as FEATURE41_NAMES,
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.post_v13_route_diagnostics import (
    accumulate_train_stats,
    coherent_signed_magnitude,
    feature41_matrix,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


FOLD_SCHEMA = "reactflow_delta.post_v13_route_diagnostic_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.post_v13_route_diagnostic_prediction.v1"
CORRECTED_REFERENCE_SCHEMA = (
    "reactflow_delta.target_identity_corrected_baseline_prediction.v1"
)
RIDGE_ALPHA = 1.0
REPLAY_ATOL = 1.0e-12
PREDICTION_FIELDS = (
    "baseline_signed_delta",
    "baseline_absolute_delta",
    "noise_aware_signed_delta",
    "noise_aware_absolute_delta",
    "coherent_signed_delta",
)


def assert_prediction_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active.get("authority", {}).get("current_phase") != "PV13D2":
        raise RuntimeError("post-V13 prediction is closed outside PV13D2")
    if active.get("training_allowed") != "FIXED_WEIGHTED_RIDGE_DIAGNOSTIC_ONLY":
        raise RuntimeError("post-V13 fixed-ridge diagnostic authority is absent")
    if active.get("candidate_model_training_allowed") is not False:
        raise RuntimeError("post-V13 neural candidate training must remain closed")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("post-V13 prediction requires held score access closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("post-V13 prediction requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("post-V13 prediction requires external outcomes locked")


def _parse_folds(value: str) -> list[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 0 or fold >= 20 for fold in folds):
        raise ValueError("folds must be a non-empty subset of 0 through 19")
    return folds


def _model_json(model: dict[str, np.ndarray | float]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def predict_registered_held(
    univ: M2Universe,
    records: list[Any],
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
    baseline_model: dict[str, np.ndarray | float],
    noise_aware_model: dict[str, np.ndarray | float],
    *,
    outer_fold: int,
) -> dict[str, np.ndarray]:
    keys: list[str] = []
    values: dict[str, list[float]] = {field: [] for field in PREDICTION_FIELDS}
    for record in records:
        construct = univ.get_construct(record.construct_id)
        receiver = np.arange(len(construct.sequence), dtype=np.int64)
        features = feature41_matrix(
            construct, record, receiver, unconstrained, constrained
        )
        baseline = predict_weighted_ridge(baseline_model, features)
        noise_aware = predict_weighted_ridge(noise_aware_model, features)
        coherent = coherent_signed_magnitude(baseline[:, 0], baseline[:, 1])
        keys.extend(_bio_key(univ, record, int(position)) for position in receiver)
        values["baseline_signed_delta"].extend(baseline[:, 0].tolist())
        values["baseline_absolute_delta"].extend(baseline[:, 1].tolist())
        values["noise_aware_signed_delta"].extend(noise_aware[:, 0].tolist())
        values["noise_aware_absolute_delta"].extend(noise_aware[:, 1].tolist())
        values["coherent_signed_delta"].extend(coherent.tolist())

    n_rows = len(keys)
    if len(set(keys)) != n_rows:
        raise RuntimeError("post-V13 prediction contains duplicate biological keys")
    prediction: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(n_rows, outer_fold, dtype=np.int64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
    }
    for field in PREDICTION_FIELDS:
        prediction[field] = np.asarray(values[field], dtype=np.float64)
        if prediction[field].shape != (n_rows,) or not np.isfinite(
            prediction[field]
        ).all():
            raise RuntimeError(f"post-V13 prediction produced invalid {field}")
    return prediction


def assert_corrected_feature41_replay(
    prediction: dict[str, np.ndarray], reference_path: Path
) -> None:
    with np.load(reference_path, allow_pickle=True) as handle:
        reference = {name: handle[name] for name in handle.files}
    if str(reference.get("schema_version", np.asarray("")).item()) != (
        CORRECTED_REFERENCE_SCHEMA
    ):
        raise ValueError("post-V13 diagnostic requires corrected TIC2A predictions")
    if not np.array_equal(prediction["keys"], reference["keys"]):
        raise ValueError("post-V13 and corrected TIC2A key orders differ")
    for current, frozen in (
        ("baseline_signed_delta", "v6_feature41_signed_delta"),
        ("baseline_absolute_delta", "v6_feature41_absolute_delta"),
    ):
        if not np.allclose(
            prediction[current], reference[frozen], atol=REPLAY_ATOL, rtol=0.0
        ):
            maximum = float(np.max(np.abs(prediction[current] - reference[frozen])))
            raise ValueError(
                f"post-V13 feature41 replay failed for {current}: max={maximum}"
            )


def run_fold(
    univ: M2Universe,
    records: list[Any],
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
    fold: Any,
    out_dir: Path,
    corrected_reference_path: Path,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    ordinary_stats, noise_stats, counts = accumulate_train_stats(
        univ, train_records, unconstrained, constrained
    )
    baseline_model = fit_weighted_standardized_ridge(ordinary_stats, RIDGE_ALPHA)
    noise_model = fit_weighted_standardized_ridge(noise_stats, RIDGE_ALPHA)
    prediction = predict_registered_held(
        univ,
        held_records,
        unconstrained,
        constrained,
        baseline_model,
        noise_model,
        outer_fold=int(fold.outer_fold),
    )
    assert_corrected_feature41_replay(prediction, corrected_reference_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_id = int(fold.outer_fold)
    prediction_path = out_dir / f"post_v13_diag_predictions_fold{fold_id}.npz"
    model_path = out_dir / f"post_v13_diag_models_fold{fold_id}.json"
    np.savez_compressed(prediction_path, **prediction)
    model_path.write_text(
        json.dumps(
            {
                "ridge_alpha": RIDGE_ALPHA,
                "feature_names": list(FEATURE41_NAMES),
                "baseline": _model_json(baseline_model),
                "noise_aware": _model_json(noise_model),
                "coherent_fitted_parameters": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": "PV13D2",
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "prediction_artifact": str(prediction_path),
        "model_artifact": str(model_path),
        "corrected_feature41_reference": str(corrected_reference_path),
        "n_registered_prediction_rows": len(prediction["keys"]),
        **counts,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "corrected_feature41_replay_pass": True,
        "held_target_or_error_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "model_or_threshold_selection_performed": False,
        "external_outcome_accessed": False,
    }
    result_path = out_dir / f"post_v13_diag_fold_result_fold{fold_id}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--corrected-baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    assert_prediction_authority(args.repo_root.resolve())
    univ = M2Universe(args.m2_csv)
    ledger = univ.build()
    if ledger.get("n_canonical_mutant_full_profiles") != 13976:
        raise RuntimeError("post-V13 diagnostic requires corrected target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    unconstrained = UnconstrainedFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    try:
        validate_cache_alignment(unconstrained, constrained)
        for fold_id in _parse_folds(args.folds):
            result_path = args.out_dir / f"post_v13_diag_fold_result_fold{fold_id}.json"
            if result_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite completed post-V13 fold {fold_id}"
                )
            reference = (
                args.corrected_baseline_dir
                / f"tic2a_corrected_predictions_fold{fold_id}.npz"
            )
            if not reference.is_file():
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
