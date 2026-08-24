#!/usr/bin/env python3
"""Run prediction-only folds for the fixed v5 ensemble-delta ridge probe."""

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
    BASELINE_FEATURE_NAMES,
    EnsembleFeatureCache,
    WeightedRidgeStats,
    baseline_features,
    cell_position_weights,
    fit_weighted_standardized_ridge,
    predict_weighted_ridge,
)
from scripts.reactflow_delta.model_rescue_v5_schema import FEATURE_NAMES
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v5_probe_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v5_probe_prediction.v1"
RIDGE_ALPHA = 1.0


def assert_probe_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V5M2":
        raise RuntimeError("v5 eligibility probe is closed outside active V5M2")
    if active.get("training_allowed") != "FIXED_WEIGHTED_RIDGE_ELIGIBILITY_ONLY":
        raise RuntimeError("v5 fixed-ridge training authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError("v5 probe prediction requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("v5 probe requires partial scores prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v5 probe requires external outcomes locked")


def _target_profile(univ: M2Universe, record: Any) -> np.ndarray | None:
    target, _error = univ.mutant_full_profile(
        record.wt_id, record.design_pos, record.ref, record.alt
    )
    return target


def accumulate_train_stats(
    univ: M2Universe,
    records: list[Any],
    cache: EnsembleFeatureCache,
) -> tuple[WeightedRidgeStats, WeightedRidgeStats, dict[str, int]]:
    baseline_stats = WeightedRidgeStats.zeros(len(BASELINE_FEATURE_NAMES), 2)
    candidate_stats = WeightedRidgeStats.zeros(
        len(BASELINE_FEATURE_NAMES) + len(FEATURE_NAMES), 2
    )
    by_cell: dict[str, list[Any]] = {}
    for record in records:
        by_cell.setdefault(record.construct_id, []).append(record)
    n_positions = 0
    n_valid_mutants = 0
    for construct_id, cell_records in sorted(by_cell.items()):
        construct = univ.get_construct(construct_id)
        prepared: list[tuple[Any, np.ndarray, np.ndarray]] = []
        counts = []
        for record in cell_records:
            target = _target_profile(univ, record)
            if target is None:
                qualified = np.zeros(len(construct.sequence), dtype=bool)
            else:
                qualified = construct.wt_observed.astype(bool) & np.isfinite(target)
            receiver = np.flatnonzero(qualified)
            signed = (
                target[receiver].astype(np.float64)
                - construct.wt_reactivity[receiver].astype(np.float64)
                if len(receiver)
                else np.zeros(0, dtype=np.float64)
            )
            prepared.append((record, receiver, signed))
            counts.append(len(receiver))
        weights = cell_position_weights(np.asarray(counts))
        for (record, receiver, signed), weight in zip(prepared, weights):
            if not len(receiver):
                continue
            x_baseline = baseline_features(construct, record, receiver)
            structure = cache.get(record)[receiver]
            x_candidate = np.concatenate([x_baseline, structure], axis=1)
            target = np.column_stack([signed, np.abs(signed)])
            baseline_stats.add_rows(x_baseline, target, weight)
            candidate_stats.add_rows(x_candidate, target, weight)
            n_positions += len(receiver)
            n_valid_mutants += 1
    return baseline_stats, candidate_stats, {
        "n_train_cells": len(by_cell),
        "n_train_valid_mutants": n_valid_mutants,
        "n_train_qualified_positions": n_positions,
    }


def _model_to_json(model: dict[str, np.ndarray | float]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def predict_registered_held(
    univ: M2Universe,
    records: list[Any],
    cache: EnsembleFeatureCache,
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
        x_baseline = baseline_features(construct, record, receiver)
        x_candidate = np.concatenate([x_baseline, cache.get(record)], axis=1)
        baseline_prediction = predict_weighted_ridge(baseline_model, x_baseline)
        candidate_prediction = predict_weighted_ridge(candidate_model, x_candidate)
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
        raise RuntimeError("v5 probe prediction contains duplicate biological keys")
    for name in (
        "baseline_signed_delta",
        "baseline_absolute_delta",
        "candidate_signed_delta",
        "candidate_absolute_delta",
    ):
        if not np.isfinite(result[name]).all():
            raise RuntimeError(f"v5 probe produced non-finite {name}")
    return result


def run_fold(
    univ: M2Universe,
    records: list[Any],
    cache: EnsembleFeatureCache,
    fold: Any,
    out_dir: Path,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    baseline_stats, candidate_stats, counts = accumulate_train_stats(
        univ, train_records, cache
    )
    baseline_model = fit_weighted_standardized_ridge(baseline_stats, RIDGE_ALPHA)
    candidate_model = fit_weighted_standardized_ridge(candidate_stats, RIDGE_ALPHA)
    prediction = predict_registered_held(
        univ,
        held_records,
        cache,
        baseline_model,
        candidate_model,
        outer_fold=int(fold.outer_fold),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / f"v5_probe_predictions_fold{fold.outer_fold}.npz"
    model_path = out_dir / f"v5_probe_models_fold{fold.outer_fold}.json"
    np.savez_compressed(prediction_path, **prediction)
    model_path.write_text(
        json.dumps(
            {
                "ridge_alpha": RIDGE_ALPHA,
                "baseline_feature_names": list(BASELINE_FEATURE_NAMES),
                "candidate_feature_names": list(BASELINE_FEATURE_NAMES + FEATURE_NAMES),
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
        "schema_version": SCHEMA,
        "phase": "V5M2",
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "prediction_artifact": str(prediction_path),
        "model_artifact": str(model_path),
        "n_registered_prediction_rows": len(prediction["keys"]),
        **counts,
        "held_target_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }
    result_path = out_dir / f"v5_probe_fold_result_fold{fold.outer_fold}.json"
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
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    assert_probe_authority(args.repo_root.resolve())
    univ = M2Universe(args.m2_csv)
    univ.build()
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    cache = EnsembleFeatureCache(args.cache)
    try:
        for fold_id in _parse_folds(args.folds):
            result_path = args.out_dir / f"v5_probe_fold_result_fold{fold_id}.json"
            if result_path.exists():
                raise FileExistsError(f"refusing to overwrite completed fold {fold_id}")
            run_fold(univ, records, cache, fold_map[fold_id], args.out_dir)
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
