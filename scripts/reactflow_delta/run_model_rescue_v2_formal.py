#!/usr/bin/env python3
"""Run the fixed five-seed R2M4 comparison after an authorized R2M3 PASS."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import AlignedDeltaModel, aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    ConditionalScaleMixtureCalibrator,
    GlobalResidualCalibrator,
    MeanAlignedModel,
)
from scripts.reactflow_delta.run_model_rescue_m2_v1 import (
    fit_candidate,
    predict_held,
    score_predictions,
)
from scripts.reactflow_delta.run_model_rescue_v2 import (
    BASELINE,
    _make_cells,
    assert_held_target_invariance,
    fit_calibrator,
    fit_mean,
    predict_pair,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v2_formal_run.v1"
FORMAL_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v2_formal_prediction.v1"
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 40
CALIBRATION_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0


def assert_r2m4_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "R2M4":
        raise RuntimeError("formal runner is closed unless active phase is R2M4")
    if active["gate_state"]["R2M3"] != "PASS":
        raise RuntimeError("formal runner requires R2M3 PASS")
    if active["gate_state"]["R2M4"] != "IN_PROGRESS":
        raise RuntimeError("formal runner requires R2M4 IN_PROGRESS")
    if active["training_allowed"] is not True:
        raise RuntimeError("formal runner requires active training authority")
    if active["new_external_outcome_access_allowed"] is not False:
        raise RuntimeError("formal runner requires external outcomes to remain locked")


def _assert_target_invariance(
    prediction: dict[str, np.ndarray],
    predict: Callable[[list[Any]], dict[str, np.ndarray]],
    held_records: list[Any],
) -> None:
    perturbed = copy.deepcopy(held_records)
    for record in perturbed:
        record.target_reactivity = 12345.0
        record.target_error = 9876.0
        record.target_observed = not bool(record.target_observed)
        record.target_mask = np.logical_not(record.target_mask)
    repeated = predict(perturbed)
    for field in ("keys", "locations", "scales", "weights"):
        if not np.array_equal(prediction[field], repeated[field]):
            raise RuntimeError(f"held target/error/mask changed formal B1 field {field}")


def combine_five_seed_predictions(
    predictions: list[dict[str, np.ndarray]], candidate_id: str
) -> dict[str, np.ndarray]:
    if len(predictions) != 5:
        raise ValueError("formal prediction requires exactly five seeds")
    keys = predictions[0]["keys"]
    expected_components = 1 if candidate_id == BASELINE else 2
    for prediction in predictions:
        if not np.array_equal(prediction["keys"], keys):
            raise ValueError("seed prediction key universes differ")
        if prediction["locations"].shape[1] != expected_components:
            raise ValueError(
                f"{candidate_id} seed prediction has wrong component count"
            )
    if len(set(map(str, keys))) != len(keys):
        raise ValueError("formal prediction key universe contains duplicates")
    locations = np.concatenate(
        [prediction["locations"] for prediction in predictions], axis=1
    )
    scales = np.concatenate([prediction["scales"] for prediction in predictions], axis=1)
    weights = np.concatenate(
        [prediction["weights"] / 5.0 for prediction in predictions], axis=1
    )
    weights = weights / weights.sum(axis=1, keepdims=True)
    seed_point_means = np.stack(
        [
            np.sum(prediction["weights"] * prediction["locations"], axis=1)
            / prediction["weights"].sum(axis=1)
            for prediction in predictions
        ],
        axis=1,
    )
    point_mean = seed_point_means.mean(axis=1)
    mixture_mean = np.sum(weights * locations, axis=1)
    if not np.allclose(mixture_mean, point_mean, atol=1e-7, rtol=0):
        raise RuntimeError("formal mixture mean differs from five-seed point mean")
    n_rows = len(keys)
    return {
        "schema_version": np.asarray(FORMAL_PREDICTION_SCHEMA),
        "keys": keys,
        "candidate_id": np.full(n_rows, candidate_id, dtype=object),
        "seed_universe": np.asarray(SEEDS, dtype=np.int64),
        "seed_point_means": seed_point_means.astype(np.float64),
        "point_mean": point_mean.astype(np.float64),
        "locations": locations.astype(np.float64),
        "scales": scales.astype(np.float64),
        "weights": weights.astype(np.float64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
    }


def run_formal_fold(
    *,
    univ: M2Universe,
    fold: Any,
    all_records: list[Any],
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in all_records if record.puzzle in train_puzzles]
    held_records = [record for record in all_records if record.puzzle == fold.held_puzzle]
    construct_ids = sorted(
        {record.construct_id for record in train_records + held_records}
    )
    ctx_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }
    cells = _make_cells(univ, train_records, device)
    baseline_predictions: list[dict[str, np.ndarray]] = []
    candidate_predictions: list[dict[str, np.ndarray]] = []
    baseline_seed_artifacts = []
    candidate_seed_artifacts = []
    for seed in SEEDS:
        print(
            f"[R2M4] fold={fold.outer_fold} seed={seed} baseline start", flush=True
        )
        torch.manual_seed(seed)
        baseline_model = AlignedDeltaModel(k_rank=0, sparse=False).to(device)
        baseline_history = fit_candidate(
            baseline_model,
            univ,
            train_records,
            ctx_cache,
            device,
            EPOCHS,
            LEARNING_RATE,
            WEIGHT_DECAY,
            0.0,
        )
        if not np.isfinite(np.asarray(baseline_history, dtype=float)).all():
            raise RuntimeError(
                f"nonfinite formal B1 loss at fold {fold.outer_fold} seed {seed}"
            )
        baseline_prediction = predict_held(
            baseline_model, univ, held_records, ctx_cache, device
        )
        _assert_target_invariance(
            baseline_prediction,
            lambda records: predict_held(
                baseline_model, univ, records, ctx_cache, device
            ),
            held_records,
        )
        baseline_checkpoint = (
            out_dir / f"r2m4_b1_checkpoint_fold{fold.outer_fold}_seed{seed}.pt"
        )
        baseline_prediction_path = (
            out_dir / f"r2m4_b1_prediction_fold{fold.outer_fold}_seed{seed}.npz"
        )
        torch.save(baseline_model.state_dict(), baseline_checkpoint)
        np.savez_compressed(baseline_prediction_path, **baseline_prediction)
        baseline_predictions.append(baseline_prediction)
        baseline_seed_artifacts.append(
            {
                "seed": seed,
                "checkpoint": str(baseline_checkpoint),
                "prediction": str(baseline_prediction_path),
                "train_loss": baseline_history,
                "target_error_mask_invariance": True,
            }
        )
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[R2M4] fold={fold.outer_fold} seed={seed} calibrated residual start",
            flush=True,
        )
        torch.manual_seed(seed)
        mean_model = MeanAlignedModel().to(device)
        mean_history = fit_mean(
            mean_model,
            cells,
            ctx_cache,
            EPOCHS,
            LEARNING_RATE,
            WEIGHT_DECAY,
            seed,
        )
        mean_checkpoint = (
            out_dir / f"r2m4_mean_checkpoint_fold{fold.outer_fold}_seed{seed}.pt"
        )
        torch.save(mean_model.state_dict(), mean_checkpoint)
        global_calibrator = GlobalResidualCalibrator().to(device)
        mixture_calibrator = ConditionalScaleMixtureCalibrator(
            feature_dim=int(mean_model.bdirect[0].in_features)
        ).to(device)
        calibration_history = fit_calibrator(
            mixture_calibrator,
            mean_model,
            cells,
            ctx_cache,
            CALIBRATION_EPOCHS,
            LEARNING_RATE,
            WEIGHT_DECAY,
            seed,
        )
        mixture_checkpoint = (
            out_dir / f"r2m4_calibration_checkpoint_fold{fold.outer_fold}_seed{seed}.pt"
        )
        torch.save(mixture_calibrator.state_dict(), mixture_checkpoint)
        global_prediction, candidate_prediction = predict_pair(
            mean_model,
            global_calibrator,
            mixture_calibrator,
            univ,
            held_records,
            ctx_cache,
            device,
            int(fold.outer_fold),
            seed,
            mean_checkpoint,
            mean_checkpoint,
            mixture_checkpoint,
        )
        assert_held_target_invariance(
            (global_prediction, candidate_prediction),
            mean_model,
            global_calibrator,
            mixture_calibrator,
            univ,
            held_records,
            ctx_cache,
            device,
            int(fold.outer_fold),
            seed,
            mean_checkpoint,
            mean_checkpoint,
            mixture_checkpoint,
        )
        candidate_prediction_path = out_dir / (
            f"r2m4_candidate_prediction_fold{fold.outer_fold}_seed{seed}.npz"
        )
        np.savez_compressed(candidate_prediction_path, **candidate_prediction)
        candidate_predictions.append(candidate_prediction)
        candidate_seed_artifacts.append(
            {
                "seed": seed,
                "mean_checkpoint": str(mean_checkpoint),
                "calibration_checkpoint": str(mixture_checkpoint),
                "prediction": str(candidate_prediction_path),
                "mean_loss": mean_history,
                "calibration_loss": calibration_history,
                "target_error_mask_invariance": True,
            }
        )
        del mean_model, global_calibrator, mixture_calibrator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_formal = combine_five_seed_predictions(baseline_predictions, BASELINE)
    candidate_formal = combine_five_seed_predictions(
        candidate_predictions, CALIBRATED_CANDIDATE
    )
    baseline_formal_path = out_dir / f"r2m4_formal_b1_fold{fold.outer_fold}.npz"
    candidate_formal_path = out_dir / f"r2m4_formal_candidate_fold{fold.outer_fold}.npz"
    np.savez_compressed(baseline_formal_path, **baseline_formal)
    np.savez_compressed(candidate_formal_path, **candidate_formal)
    baseline_score = score_predictions(baseline_formal, univ, held_records)
    candidate_score = score_predictions(candidate_formal, univ, held_records)
    return {
        "schema_version": SCHEMA,
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "all_seed_target_error_mask_invariance": True,
        "baseline": {
            "model_id": BASELINE,
            "seed_universe": SEEDS,
            "prediction_artifact": str(baseline_formal_path),
            "seed_artifacts": baseline_seed_artifacts,
            "score": baseline_score,
        },
        "candidate": {
            "model_id": CALIBRATED_CANDIDATE,
            "seed_universe": SEEDS,
            "prediction_artifact": str(candidate_formal_path),
            "seed_artifacts": candidate_seed_artifacts,
            "score": candidate_score,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    assert_r2m4_authority(repo_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    univ.build()
    records = univ.get_records()
    splits = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = {int(value) for value in args.folds.split(",") if value}
    folds = [fold for fold in splits["folds"] if fold.outer_fold in selected]
    if not folds:
        raise ValueError("no requested outer folds")
    for fold in folds:
        result = run_formal_fold(
            univ=univ,
            fold=fold,
            all_records=records,
            device=device,
            out_dir=args.out_dir,
        )
        path = args.out_dir / f"v2_formal_fold_result_fold{fold.outer_fold}.json"
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[R2M4] fold={fold.outer_fold} artifact={path} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
