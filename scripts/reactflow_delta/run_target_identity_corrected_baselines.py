#!/usr/bin/env python3
"""Fit prediction-only corrected direct18/feature30/feature41 LOPO folds."""

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
    BASELINE_FEATURE_NAMES as DIRECT_FEATURE_NAMES,
    EnsembleFeatureCache as UnconstrainedFeatureCache,
    WeightedRidgeStats,
    fit_weighted_standardized_ridge,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    BASELINE_PROBE_FEATURE_NAMES as FEATURE30_NAMES,
    CANDIDATE_PROBE_FEATURE_NAMES as FEATURE41_NAMES,
    ConstrainedFeatureCache,
    accumulate_train_stats as accumulate_v6_stats,
    validate_cache_alignment,
)
from scripts.reactflow_delta.run_model_rescue_v5_probe import (
    accumulate_train_stats as accumulate_v5_stats,
    predict_registered_held as predict_v5,
)
from scripts.reactflow_delta.run_model_rescue_v6_probe import (
    predict_registered_held as predict_v6,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


FOLD_SCHEMA = "reactflow_delta.target_identity_corrected_baseline_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.target_identity_corrected_baseline_prediction.v1"
RIDGE_ALPHA = 1.0
REPLAY_ATOL = 1.0e-12
PREDICTION_FIELDS = (
    "direct18_signed_delta",
    "direct18_absolute_delta",
    "v5_feature30_signed_delta",
    "v5_feature30_absolute_delta",
    "v6_feature41_signed_delta",
    "v6_feature41_absolute_delta",
)


def assert_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "TIC2A":
        raise RuntimeError("corrected baseline rebuild is closed outside TIC2A")
    if (
        active.get("training_allowed")
        != "FIXED_CORRECTED_WEIGHTED_RIDGE_BASELINES_ONLY"
    ):
        raise RuntimeError("fixed corrected ridge authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("TIC2A prediction requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("TIC2A prediction requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("TIC2A requires external outcomes locked")


def _stats_equal(
    left: WeightedRidgeStats,
    right: WeightedRidgeStats,
    *,
    atol: float = REPLAY_ATOL,
) -> bool:
    if not np.isclose(left.sum_weight, right.sum_weight, atol=atol, rtol=0.0):
        return False
    return all(
        np.allclose(getattr(left, name), getattr(right, name), atol=atol, rtol=0.0)
        for name in ("sum_x", "sum_x2", "xtx", "sum_y", "xty")
    )


def _model_json(model: dict[str, np.ndarray | float]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def _parse_folds(value: str) -> list[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 0 or fold >= 20 for fold in folds):
        raise ValueError("folds must be a non-empty subset of 0 through 19")
    return folds


def run_fold(
    univ: M2Universe,
    records: list[Any],
    unconstrained: UnconstrainedFeatureCache,
    constrained: ConstrainedFeatureCache,
    fold: Any,
    out_dir: Path,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]

    direct_stats, feature30_stats_v5, counts_v5 = accumulate_v5_stats(
        univ, train_records, unconstrained
    )
    feature30_stats_v6, feature41_stats, counts_v6 = accumulate_v6_stats(
        univ, train_records, unconstrained, constrained
    )
    if counts_v5 != counts_v6:
        raise RuntimeError("corrected v5/v6 training universes differ")
    if not _stats_equal(feature30_stats_v5, feature30_stats_v6):
        raise RuntimeError("corrected feature30 sufficient statistics do not replay")

    direct_model = fit_weighted_standardized_ridge(direct_stats, RIDGE_ALPHA)
    feature30_model_v5 = fit_weighted_standardized_ridge(
        feature30_stats_v5, RIDGE_ALPHA
    )
    feature30_model_v6 = fit_weighted_standardized_ridge(
        feature30_stats_v6, RIDGE_ALPHA
    )
    feature41_model = fit_weighted_standardized_ridge(feature41_stats, RIDGE_ALPHA)

    v5_prediction = predict_v5(
        univ,
        held_records,
        unconstrained,
        direct_model,
        feature30_model_v5,
        outer_fold=int(fold.outer_fold),
    )
    v6_prediction = predict_v6(
        univ,
        held_records,
        unconstrained,
        constrained,
        feature30_model_v6,
        feature41_model,
        outer_fold=int(fold.outer_fold),
    )
    if not np.array_equal(v5_prediction["keys"], v6_prediction["keys"]):
        raise RuntimeError("corrected v5/v6 held biological key order differs")
    for name in ("signed_delta", "absolute_delta"):
        if not np.allclose(
            v5_prediction[f"candidate_{name}"],
            v6_prediction[f"baseline_{name}"],
            atol=REPLAY_ATOL,
            rtol=0.0,
        ):
            raise RuntimeError(f"corrected feature30 {name} predictions do not replay")

    keys = v5_prediction["keys"]
    prediction = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": v5_prediction["outer_fold"],
        "direct18_signed_delta": v5_prediction["baseline_signed_delta"],
        "direct18_absolute_delta": v5_prediction["baseline_absolute_delta"],
        "v5_feature30_signed_delta": v5_prediction["candidate_signed_delta"],
        "v5_feature30_absolute_delta": v5_prediction["candidate_absolute_delta"],
        "v6_feature41_signed_delta": v6_prediction["candidate_signed_delta"],
        "v6_feature41_absolute_delta": v6_prediction["candidate_absolute_delta"],
        "registered_status": np.full(len(keys), "covered", dtype=object),
    }
    if len(set(map(str, keys))) != len(keys):
        raise RuntimeError("corrected baseline prediction keys are duplicated")
    for name in PREDICTION_FIELDS:
        if prediction[name].shape != (len(keys),) or not np.isfinite(
            prediction[name]
        ).all():
            raise RuntimeError(f"corrected baseline produced invalid {name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_id = int(fold.outer_fold)
    prediction_path = out_dir / f"tic2a_corrected_predictions_fold{fold_id}.npz"
    model_path = out_dir / f"tic2a_corrected_models_fold{fold_id}.json"
    np.savez_compressed(prediction_path, **prediction)
    model_path.write_text(
        json.dumps(
            {
                "ridge_alpha": RIDGE_ALPHA,
                "direct18_feature_names": list(DIRECT_FEATURE_NAMES),
                "feature30_feature_names": list(FEATURE30_NAMES),
                "feature41_feature_names": list(FEATURE41_NAMES),
                "direct18": _model_json(direct_model),
                "v5_feature30": _model_json(feature30_model_v5),
                "v6_feature30_replay": _model_json(feature30_model_v6),
                "v6_feature41": _model_json(feature41_model),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": "TIC2A",
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "prediction_artifact": str(prediction_path),
        "model_artifact": str(model_path),
        "n_registered_prediction_rows": len(keys),
        **counts_v5,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "v5_v6_feature30_stats_replay_pass": True,
        "v5_v6_feature30_prediction_replay_pass": True,
        "held_target_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "legacy_prediction_reused": False,
        "external_outcome_accessed": False,
    }
    result_path = out_dir / f"tic2a_corrected_fold_result_fold{fold_id}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    assert_authority(args.repo_root.resolve())
    univ = M2Universe(args.m2_csv)
    ledger = univ.build()
    if ledger.get("n_canonical_mutant_full_profiles") != 13976:
        raise RuntimeError("TIC2A requires the qualified real-data target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    unconstrained = UnconstrainedFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    try:
        validate_cache_alignment(unconstrained, constrained)
        for fold_id in _parse_folds(args.folds):
            result = args.out_dir / f"tic2a_corrected_fold_result_fold{fold_id}.json"
            if result.exists():
                raise FileExistsError(f"refusing to overwrite corrected fold {fold_id}")
            run_fold(
                univ,
                records,
                unconstrained,
                constrained,
                fold_map[fold_id],
                args.out_dir,
            )
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
