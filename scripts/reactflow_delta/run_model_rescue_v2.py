#!/usr/bin/env python3
"""Run the fixed mean-first Model Rescue v2 procedure on selected LOPO folds."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    MEAN_CANDIDATE,
    PREDICTION_SCHEMA,
    ConditionalScaleMixtureCalibrator,
    GlobalResidualCalibrator,
    MeanAlignedModel,
    assert_mean_state_unchanged,
    cell_balanced_crps,
    cell_balanced_l1,
    freeze_mean_model,
    mean_state_snapshot,
)
from scripts.reactflow_delta.run_model_rescue_m2_v1 import score_predictions
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_p3_lrso_v3 import (
    _qualified_mask,
    _target_matrix,
    _wt_filled,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v2_run.v1"
BASELINE = "b1_rfd_direct_aligned"


def _make_cells(univ: M2Universe, records: list[Any], device: str) -> list[dict[str, Any]]:
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    cells = []
    for construct_id, recs in sorted(by_construct.items()):
        target, wt_observed = _target_matrix(univ, recs)
        qualified = _qualified_mask(target, wt_observed)
        if qualified.sum() == 0:
            continue
        length = target.shape[1]
        edit = torch.tensor([record.pos for record in recs], device=device)
        distance = (torch.arange(length, device=device)[None, :] - edit[:, None]).float()
        cells.append(
            {
                "construct_id": construct_id,
                "edit": edit,
                "distance": distance,
                "refs": [record.ref for record in recs],
                "alts": [record.alt for record in recs],
                "target": torch.tensor(target, device=device),
                "prediction_mask": torch.tensor(wt_observed, device=device),
                "qualified_mask": torch.tensor(qualified, device=device),
                "wt": torch.tensor(_wt_filled(univ, construct_id), device=device),
            }
        )
    return cells


def _epoch_order(n_cells: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n_cells))
    random.Random(seed * 100_003 + epoch).shuffle(order)
    return order


def _require_finite_gradients(module: torch.nn.Module, stage: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite gradient in {stage}: {name}")


def fit_mean(
    model: MeanAlignedModel,
    cells: list[dict[str, Any]],
    ctx_cache: dict[str, Any],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for epoch in range(epochs):
        losses = []
        for index in _epoch_order(len(cells), seed, epoch):
            cell = cells[index]
            H = model.encode(ctx_cache[cell["construct_id"]])
            delta_mean = model.forward_mean(
                H,
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
            )
            loss = cell_balanced_l1(
                delta_mean,
                cell["target"],
                cell["qualified_mask"],
                cell["wt"],
            )
            optimizer.zero_grad()
            loss.backward()
            _require_finite_gradients(model, "mean")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return history


def fit_calibrator(
    calibrator: torch.nn.Module,
    mean_model: MeanAlignedModel,
    cells: list[dict[str, Any]],
    ctx_cache: dict[str, Any],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    freeze_mean_model(mean_model)
    snapshot = mean_state_snapshot(mean_model)
    calibrator.train()
    optimizer = torch.optim.Adam(
        calibrator.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for epoch in range(epochs):
        losses = []
        for index in _epoch_order(len(cells), seed + 10_000, epoch):
            cell = cells[index]
            with torch.no_grad():
                H = mean_model.encode(ctx_cache[cell["construct_id"]])
                delta_mean, features = mean_model.forward_mean_and_features(
                    H,
                    cell["edit"],
                    cell["distance"],
                    cell["refs"],
                    cell["alts"],
                    cell["prediction_mask"],
                )
            if isinstance(calibrator, GlobalResidualCalibrator):
                weights, locations, scales = calibrator(delta_mean.detach())
            else:
                weights, locations, scales = calibrator(
                    delta_mean.detach(), features.detach()
                )
            loss = cell_balanced_crps(
                weights,
                locations,
                scales,
                cell["target"],
                cell["qualified_mask"],
                cell["wt"],
            )
            optimizer.zero_grad()
            loss.backward()
            _require_finite_gradients(calibrator, "calibration")
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    assert_mean_state_unchanged(snapshot, mean_model)
    if any(parameter.grad is not None for parameter in mean_model.parameters()):
        raise RuntimeError("calibration produced a gradient on the frozen mean model")
    return history


def _prediction_arrays(
    *,
    keys: list[str],
    delta_mean: list[float],
    point_mean: list[float],
    locations: list[np.ndarray],
    scales: list[np.ndarray],
    weights: list[np.ndarray],
    candidate_id: str,
    outer_fold: int,
    seed: int,
    mean_checkpoint: Path,
    calibration_checkpoint: Path,
) -> dict[str, np.ndarray]:
    n_rows = len(keys)
    result = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "candidate_id": np.full(n_rows, candidate_id, dtype=object),
        "outer_fold": np.full(n_rows, outer_fold, dtype=np.int64),
        "seed": np.full(n_rows, seed, dtype=np.int64),
        "delta_mean": np.asarray(delta_mean, dtype=np.float64),
        "point_mean": np.asarray(point_mean, dtype=np.float64),
        "locations": np.asarray(locations, dtype=np.float64),
        "scales": np.asarray(scales, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
        "mean_checkpoint_path": np.full(n_rows, str(mean_checkpoint), dtype=object),
        "calibration_checkpoint_path": np.full(
            n_rows, str(calibration_checkpoint), dtype=object
        ),
    }
    if not np.allclose(result["weights"].sum(axis=1), 1.0, atol=1e-7, rtol=0):
        raise RuntimeError(f"{candidate_id} mixture weights do not sum to one")
    if not np.all(np.isfinite(result["scales"])) or not np.all(result["scales"] > 0):
        raise RuntimeError(f"{candidate_id} contains invalid residual scale")
    if not np.all(np.isfinite(result["locations"])):
        raise RuntimeError(f"{candidate_id} contains invalid location")
    return result


def predict_pair(
    mean_model: MeanAlignedModel,
    global_calibrator: GlobalResidualCalibrator,
    mixture_calibrator: ConditionalScaleMixtureCalibrator,
    univ: M2Universe,
    held_records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
    outer_fold: int,
    seed: int,
    mean_checkpoint: Path,
    global_checkpoint: Path,
    mixture_checkpoint: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate both candidate ledgers in one target-blind traversal."""
    by_construct: dict[str, list[Any]] = {}
    for record in held_records:
        by_construct.setdefault(record.construct_id, []).append(record)
    mean_model.eval()
    global_calibrator.eval()
    mixture_calibrator.eval()
    common_keys: list[str] = []
    common_delta: list[float] = []
    common_point: list[float] = []
    global_locations: list[np.ndarray] = []
    global_scales: list[np.ndarray] = []
    global_weights: list[np.ndarray] = []
    mixture_locations: list[np.ndarray] = []
    mixture_scales: list[np.ndarray] = []
    mixture_weights: list[np.ndarray] = []
    with torch.no_grad():
        for construct_id, recs in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            edit = torch.tensor([record.pos for record in recs], device=device)
            distance = (torch.arange(length, device=device)[None, :] - edit[:, None]).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(recs), 1)),
                device=device,
            )
            H = mean_model.encode(ctx_cache[construct_id])
            delta_mean, features = mean_model.forward_mean_and_features(
                H,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            gw, gl, gs = global_calibrator(delta_mean)
            mw, ml, ms = mixture_calibrator(delta_mean, features)
            wt = torch.tensor(_wt_filled(univ, construct_id), device=device)
            point = delta_mean + wt[None, :]
            gl = gl + wt[None, :, None]
            ml = ml + wt[None, :, None]
            for row, record in enumerate(recs):
                for position in range(length):
                    common_keys.append(_bio_key(univ, record, position))
                    common_delta.append(float(delta_mean[row, position].cpu()))
                    common_point.append(float(point[row, position].cpu()))
                    global_locations.append(gl[row, position].cpu().numpy())
                    global_scales.append(gs[row, position].cpu().numpy())
                    global_weights.append(gw[row, position].cpu().numpy())
                    mixture_locations.append(ml[row, position].cpu().numpy())
                    mixture_scales.append(ms[row, position].cpu().numpy())
                    mixture_weights.append(mw[row, position].cpu().numpy())
    global_prediction = _prediction_arrays(
        keys=common_keys,
        delta_mean=common_delta,
        point_mean=common_point,
        locations=global_locations,
        scales=global_scales,
        weights=global_weights,
        candidate_id=MEAN_CANDIDATE,
        outer_fold=outer_fold,
        seed=seed,
        mean_checkpoint=mean_checkpoint,
        calibration_checkpoint=global_checkpoint,
    )
    mixture_prediction = _prediction_arrays(
        keys=common_keys,
        delta_mean=common_delta,
        point_mean=common_point,
        locations=mixture_locations,
        scales=mixture_scales,
        weights=mixture_weights,
        candidate_id=CALIBRATED_CANDIDATE,
        outer_fold=outer_fold,
        seed=seed,
        mean_checkpoint=mean_checkpoint,
        calibration_checkpoint=mixture_checkpoint,
    )
    return global_prediction, mixture_prediction


def assert_held_target_invariance(
    expected: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    mean_model: MeanAlignedModel,
    global_calibrator: GlobalResidualCalibrator,
    mixture_calibrator: ConditionalScaleMixtureCalibrator,
    univ: M2Universe,
    held_records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
    outer_fold: int,
    seed: int,
    mean_checkpoint: Path,
    global_checkpoint: Path,
    mixture_checkpoint: Path,
) -> None:
    perturbed = copy.deepcopy(held_records)
    for record in perturbed:
        record.target_reactivity = 12345.0
        record.target_error = 9876.0
        record.target_observed = not bool(record.target_observed)
        record.target_mask = np.logical_not(record.target_mask)
    actual = predict_pair(
        mean_model,
        global_calibrator,
        mixture_calibrator,
        univ,
        perturbed,
        ctx_cache,
        device,
        outer_fold,
        seed,
        mean_checkpoint,
        global_checkpoint,
        mixture_checkpoint,
    )
    fields = (
        "keys",
        "delta_mean",
        "point_mean",
        "locations",
        "scales",
        "weights",
        "registered_status",
    )
    for candidate_index, candidate_id in enumerate((MEAN_CANDIDATE, CALIBRATED_CANDIDATE)):
        for field in fields:
            if not np.array_equal(expected[candidate_index][field], actual[candidate_index][field]):
                raise RuntimeError(
                    f"held target/error/mask changed {candidate_id} prediction field {field}"
                )


def _load_frozen_baseline(result_dir: Path | None, fold: int) -> dict[str, Any] | None:
    if result_dir is None:
        return None
    path = result_dir / f"m2_fold_result_fold{fold}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["candidates"][BASELINE]
    return {
        "model_id": BASELINE,
        "score": row["score"],
        "prediction_artifact": row["prediction_artifact"],
        "source_fold_artifact": str(path),
    }


def run_fold(
    *,
    univ: M2Universe,
    fold: Any,
    all_records: list[Any],
    device: str,
    out_dir: Path,
    mean_epochs: int,
    calibration_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    b1_result_dir: Path | None,
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
    torch.manual_seed(seed)
    mean_model = MeanAlignedModel().to(device)
    mean_history = fit_mean(
        mean_model,
        cells,
        ctx_cache,
        mean_epochs,
        learning_rate,
        weight_decay,
        seed,
    )
    mean_checkpoint = out_dir / f"v2_mean_fold{fold.outer_fold}_seed{seed}.pt"
    torch.save(mean_model.state_dict(), mean_checkpoint)
    feature_dim = int(mean_model.bdirect[0].in_features)
    global_calibrator = GlobalResidualCalibrator().to(device)
    mixture_calibrator = ConditionalScaleMixtureCalibrator(feature_dim=feature_dim).to(device)
    global_history = fit_calibrator(
        global_calibrator,
        mean_model,
        cells,
        ctx_cache,
        calibration_epochs,
        learning_rate,
        weight_decay,
        seed,
    )
    mixture_history = fit_calibrator(
        mixture_calibrator,
        mean_model,
        cells,
        ctx_cache,
        calibration_epochs,
        learning_rate,
        weight_decay,
        seed,
    )
    global_checkpoint = out_dir / f"v2_global_calibration_fold{fold.outer_fold}_seed{seed}.pt"
    mixture_checkpoint = out_dir / f"v2_mixture_calibration_fold{fold.outer_fold}_seed{seed}.pt"
    torch.save(global_calibrator.state_dict(), global_checkpoint)
    torch.save(mixture_calibrator.state_dict(), mixture_checkpoint)
    global_prediction, mixture_prediction = predict_pair(
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
        global_checkpoint,
        mixture_checkpoint,
    )
    assert_held_target_invariance(
        (global_prediction, mixture_prediction),
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
        global_checkpoint,
        mixture_checkpoint,
    )
    global_path = out_dir / f"v2_predictions_{MEAN_CANDIDATE}_fold{fold.outer_fold}_seed{seed}.npz"
    mixture_path = out_dir / f"v2_predictions_{CALIBRATED_CANDIDATE}_fold{fold.outer_fold}_seed{seed}.npz"
    np.savez_compressed(global_path, **global_prediction)
    np.savez_compressed(mixture_path, **mixture_prediction)
    point_difference = float(
        np.max(np.abs(global_prediction["point_mean"] - mixture_prediction["point_mean"]))
    )
    if point_difference > 1e-7:
        raise RuntimeError(
            f"calibration changed point mean by {point_difference}, exceeding 1e-7"
        )
    result = {
        "schema_version": SCHEMA,
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "seed": seed,
        "baseline": _load_frozen_baseline(b1_result_dir, int(fold.outer_fold)),
        "point_mean_max_abs_difference": point_difference,
        "held_target_error_mask_invariance": True,
        "candidates": {
            MEAN_CANDIDATE: {
                "score": score_predictions(global_prediction, univ, held_records),
                "prediction_artifact": str(global_path),
                "mean_checkpoint": str(mean_checkpoint),
                "calibration_checkpoint": str(global_checkpoint),
                "mean_loss": mean_history,
                "calibration_loss": global_history,
            },
            CALIBRATED_CANDIDATE: {
                "score": score_predictions(mixture_prediction, univ, held_records),
                "prediction_artifact": str(mixture_path),
                "mean_checkpoint": str(mean_checkpoint),
                "calibration_checkpoint": str(mixture_checkpoint),
                "mean_loss": mean_history,
                "calibration_loss": mixture_history,
            },
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--b1-result-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--mean-epochs", type=int, default=40)
    parser.add_argument("--calibration-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    mean_epochs = min(args.mean_epochs, 3) if args.smoke else args.mean_epochs
    calibration_epochs = (
        min(args.calibration_epochs, 3) if args.smoke else args.calibration_epochs
    )
    univ = M2Universe(args.m2_csv)
    univ.build()
    records = univ.get_records()
    splits = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = {int(value) for value in args.folds.split(",") if value}
    folds = [fold for fold in splits["folds"] if fold.outer_fold in selected]
    if not folds:
        raise ValueError("no requested outer folds")
    fold_results = []
    for fold in folds:
        print(
            f"[R2] fold={fold.outer_fold} held={fold.held_puzzle} seed={args.seed} start",
            flush=True,
        )
        result = run_fold(
            univ=univ,
            fold=fold,
            all_records=records,
            device=device,
            out_dir=args.out_dir,
            mean_epochs=mean_epochs,
            calibration_epochs=calibration_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            b1_result_dir=args.b1_result_dir,
        )
        fold_results.append(result)
        path = args.out_dir / f"v2_fold_result_fold{fold.outer_fold}_seed{args.seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[R2] fold={fold.outer_fold} artifact={path} complete", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = {
        "schema_version": SCHEMA,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY" if args.smoke else "DEVELOPMENT_CONSUMED_SCREEN"
        ),
        "seed": args.seed,
        "mean_epochs": mean_epochs,
        "calibration_epochs": calibration_epochs,
        "folds": fold_results,
        "qualification": {
            "external": "NOT_ACCESSED",
            "sota": "NOT_ESTABLISHED",
            "partial_results_must_not_change_configuration": True,
        },
    }
    path = args.out_dir / f"v2_result_seed{args.seed}.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
