#!/usr/bin/env python3
"""Run prediction-only folds for the corrected fixed V7M2 eligibility probe."""

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
    accumulate_train_stats as accumulate_v6_train_stats,
)
from scripts.reactflow_delta.model_rescue_v7_probe import (
    CANDIDATE_PROBE_FEATURE_NAMES as FEATURE47_NAMES,
    DependencyFeatureCache,
    accumulate_candidate_train_stats,
    prediction_features,
    validate_cache_alignment,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


FOLD_SCHEMA = "reactflow_delta.model_rescue_v7_probe_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v7_probe_prediction.v1"
CORRECTED_REFERENCE_SCHEMA = (
    "reactflow_delta.target_identity_corrected_baseline_prediction.v1"
)
RIDGE_ALPHA = 1.0
REPLAY_ATOL = 1.0e-12
PREDICTION_FIELDS = (
    "baseline_signed_delta",
    "baseline_absolute_delta",
    "candidate_signed_delta",
    "candidate_absolute_delta",
)


def assert_probe_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V7M2":
        raise RuntimeError("V7M2 eligibility probe is closed outside active V7M2")
    if (
        active.get("training_allowed")
        != "FIXED_CORRECTED_WEIGHTED_RIDGE_ELIGIBILITY_ONLY"
    ):
        raise RuntimeError("V7M2 fixed corrected ridge authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("V7M2 prediction requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V7M2 prediction requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V7M2 requires external outcomes locked")


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
    dependency: DependencyFeatureCache,
    baseline_model: dict[str, np.ndarray | float],
    candidate_model: dict[str, np.ndarray | float],
    *,
    outer_fold: int,
) -> dict[str, np.ndarray]:
    keys: list[str] = []
    values: dict[str, list[float]] = {field: [] for field in PREDICTION_FIELDS}
    for record in records:
        construct = univ.get_construct(record.construct_id)
        receiver = np.arange(len(construct.sequence), dtype=np.int64)
        feature41, feature47 = prediction_features(
            construct,
            record,
            receiver,
            unconstrained,
            constrained,
            dependency,
        )
        baseline = predict_weighted_ridge(baseline_model, feature41)
        candidate = predict_weighted_ridge(candidate_model, feature47)
        keys.extend(_bio_key(univ, record, int(position)) for position in receiver)
        values["baseline_signed_delta"].extend(baseline[:, 0].tolist())
        values["baseline_absolute_delta"].extend(baseline[:, 1].tolist())
        values["candidate_signed_delta"].extend(candidate[:, 0].tolist())
        values["candidate_absolute_delta"].extend(candidate[:, 1].tolist())
    n_rows = len(keys)
    if len(set(keys)) != n_rows:
        raise RuntimeError("V7M2 prediction contains duplicate biological keys")
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
            raise RuntimeError(f"V7M2 produced invalid {field}")
    return prediction


def assert_corrected_baseline_replay(
    prediction: dict[str, np.ndarray], reference_path: Path
) -> None:
    with np.load(reference_path, allow_pickle=True) as handle:
        reference = {name: handle[name] for name in handle.files}
    if str(reference.get("schema_version", np.asarray("")).item()) != (
        CORRECTED_REFERENCE_SCHEMA
    ):
        raise ValueError("V7M2 requires the corrected TIC2A prediction schema")
    if not np.array_equal(prediction["keys"], reference["keys"]):
        raise ValueError("V7M2 baseline and corrected TIC2A biological keys differ")
    comparisons = (
        ("baseline_signed_delta", "v6_feature41_signed_delta"),
        ("baseline_absolute_delta", "v6_feature41_absolute_delta"),
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
                f"V7M2 corrected baseline replay failed for {current_name}: "
                f"max={maximum}"
            )


def run_fold(
    univ: M2Universe,
    records: list[Any],
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
    dependency: DependencyFeatureCache,
    fold: Any,
    out_dir: Path,
    corrected_reference_path: Path,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    _feature30_stats, feature41_stats, counts41 = accumulate_v6_train_stats(
        univ,
        train_records,
        unconstrained,
        constrained,
    )
    feature47_stats, counts47 = accumulate_candidate_train_stats(
        univ,
        train_records,
        unconstrained,
        constrained,
        dependency,
    )
    if counts41 != counts47:
        raise RuntimeError("V7M2 baseline and candidate training universes differ")
    baseline_model = fit_weighted_standardized_ridge(feature41_stats, RIDGE_ALPHA)
    candidate_model = fit_weighted_standardized_ridge(feature47_stats, RIDGE_ALPHA)
    prediction = predict_registered_held(
        univ,
        held_records,
        unconstrained,
        constrained,
        dependency,
        baseline_model,
        candidate_model,
        outer_fold=int(fold.outer_fold),
    )
    assert_corrected_baseline_replay(prediction, corrected_reference_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_id = int(fold.outer_fold)
    prediction_path = out_dir / f"v7_probe_predictions_fold{fold_id}.npz"
    model_path = out_dir / f"v7_probe_models_fold{fold_id}.json"
    np.savez_compressed(prediction_path, **prediction)
    model_path.write_text(
        json.dumps(
            {
                "ridge_alpha": RIDGE_ALPHA,
                "baseline_feature_names": list(FEATURE41_NAMES),
                "candidate_feature_names": list(FEATURE47_NAMES),
                "baseline": _model_json(baseline_model),
                "candidate": _model_json(candidate_model),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": "V7M2",
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "prediction_artifact": str(prediction_path),
        "model_artifact": str(model_path),
        "corrected_baseline_reference": str(corrected_reference_path),
        "n_registered_prediction_rows": len(prediction["keys"]),
        **counts47,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "corrected_feature41_replay_pass": True,
        "held_target_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "model_selection_performed": False,
        "legacy_target_dependent_prediction_reused": False,
        "external_outcome_accessed": False,
    }
    result_path = out_dir / f"v7_probe_fold_result_fold{fold_id}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--dependency-cache", type=Path, required=True)
    parser.add_argument("--corrected-baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    assert_probe_authority(args.repo_root.resolve())
    univ = M2Universe(args.m2_csv)
    ledger = univ.build()
    if ledger.get("n_canonical_mutant_full_profiles") != 13976:
        raise RuntimeError("V7M2 requires the qualified corrected target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    unconstrained = UnconstrainedFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    dependency = DependencyFeatureCache(args.dependency_cache)
    try:
        validate_cache_alignment(unconstrained, constrained, dependency)
        for fold_id in _parse_folds(args.folds):
            result_path = args.out_dir / f"v7_probe_fold_result_fold{fold_id}.json"
            if result_path.exists():
                raise FileExistsError(f"refusing to overwrite V7M2 fold {fold_id}")
            reference = (
                args.corrected_baseline_dir
                / f"tic2a_corrected_predictions_fold{fold_id}.npz"
            )
            if not reference.exists():
                raise FileNotFoundError(reference)
            run_fold(
                univ,
                records,
                unconstrained,
                constrained,
                dependency,
                fold_map[fold_id],
                args.out_dir,
                reference,
            )
    finally:
        unconstrained.close()
        constrained.close()
        dependency.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
