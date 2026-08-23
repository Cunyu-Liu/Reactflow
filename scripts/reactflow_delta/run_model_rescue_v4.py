#!/usr/bin/env python3
"""Run fixed v4 model families on selected LOPO folds without held scoring."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v2 import (
    MeanAlignedModel,
    cell_balanced_crps,
    cell_balanced_l1,
)
from scripts.reactflow_delta.model_rescue_v4 import (
    CAPACITY_NULL,
    FOUNDATION_ONLY_CONTROL,
    PREDICTION_SCHEMA,
    PRIMARY_CANDIDATE,
    SCRATCH_CONTROL,
    MutationConditionedDualTower,
    RNAFMOnlyMean,
    V4ModelConfig,
    ZeroMeanResidualCalibrator,
    freeze_mean_model,
    trainable_parameter_count,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_p3_lrso_v3 import (
    _qualified_mask,
    _target_matrix,
    _wt_filled,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v4_fold.v1"
CORRECTED_B1 = "corrected_b1"
MODEL_UNIVERSE = (
    CORRECTED_B1,
    PRIMARY_CANDIDATE,
    SCRATCH_CONTROL,
    FOUNDATION_ONLY_CONTROL,
    CAPACITY_NULL,
)
MEAN_EPOCHS = 80
CALIBRATION_EPOCHS = 40
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 1.0


class FoundationCache:
    def __init__(self, path: Path) -> None:
        import h5py

        self.handle = h5py.File(path, "r")
        raw_ids = self.handle["row_ids"][:]
        ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_ids]
        self.index = {row_id: index for index, row_id in enumerate(ids)}
        self.lengths = self.handle["lengths"][:]
        self.embeddings = self.handle["embeddings"]

    def close(self) -> None:
        self.handle.close()

    def resolve(self, row_id: str) -> str:
        if row_id in self.index:
            return row_id
        suffix = row_id.split("_mm_", 1)[-1] if "_mm_" in row_id else row_id
        matches = [key for key in self.index if key.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"foundation cache has no unique row for {row_id}")
        return matches[0]

    def get(self, row_id: str) -> np.ndarray:
        key = self.resolve(row_id)
        index = self.index[key]
        length = int(self.lengths[index])
        return np.asarray(self.embeddings[index, :length], dtype=np.float32)


def mutant_row_id(record: Any) -> str:
    prefix = record.wt_id[:-3] if record.wt_id.endswith("_wt") else record.wt_id
    ref = record.ref.replace("U", "T")
    alt = record.alt.replace("U", "T")
    return f"{prefix}_mm_{record.design_pos}_{ref}_{alt}"


def assert_run_authority(repo_root: Path, phase: str) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected = {"V4M2": "ENGINEERING_SMOKE_ONLY", "V4M3": True, "V4M4": True}
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(f"v4 runner is closed outside active {phase}")
    if active.get("training_allowed") != expected[phase]:
        raise RuntimeError(f"v4 {phase} training authority is absent")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v4 runner requires external outcomes to remain locked")
    if active["resource_partition"]["v4_allowed_physical_gpus"] != list(range(8)):
        raise RuntimeError("v4 GPU partition changed")


def _epoch_order(n_cells: int, seed: int, epoch: int, offset: int = 0) -> list[int]:
    order = list(range(n_cells))
    random.Random(seed * 1_000_003 + epoch + offset).shuffle(order)
    return order


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    warmup = max(int(total_steps * 0.05), 1)

    def multiplier(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _finite_gradients(model: nn.Module, stage: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite {stage} gradient in {name}")


def _input_cell(
    univ: M2Universe,
    construct_id: str,
    records: list[Any],
    foundation: FoundationCache,
) -> dict[str, Any]:
    construct = univ.get_construct(construct_id)
    length = len(construct.sequence)
    seq, react, error_feature, observed, position, region = aligned_wt_ctx_tensors(
        univ, construct_id, "cpu"
    )
    position = 2.0 * position / max(length - 1, 1) - 1.0
    wt_foundation = foundation.get(records[0].wt_id)
    mutant_foundation = np.stack(
        [foundation.get(mutant_row_id(record)) for record in records]
    )
    if wt_foundation.shape != (length, 640) or mutant_foundation.shape != (
        len(records),
        length,
        640,
    ):
        raise RuntimeError(f"foundation cache shape mismatch for {construct_id}")
    return {
        "construct_id": construct_id,
        "records": records,
        "sequence_one_hot": seq,
        "wt_reactivity": react,
        "wt_error": error_feature,
        "wt_observed": observed.bool(),
        "position": position,
        "region_one_hot": region,
        "edit_idx": torch.tensor([record.full_pos for record in records]),
        "refs": [record.ref for record in records],
        "alts": [record.alt for record in records],
        "wt_foundation": torch.from_numpy(wt_foundation),
        "mutant_foundation": torch.from_numpy(mutant_foundation),
        "wt": torch.tensor(_wt_filled(univ, construct_id)),
        "b1_ctx": aligned_wt_ctx_tensors(univ, construct_id, "cpu"),
    }


def _training_cell(
    univ: M2Universe,
    construct_id: str,
    records: list[Any],
    foundation: FoundationCache,
) -> dict[str, Any] | None:
    target, wt_observed_matrix = _target_matrix(univ, records)
    qualified = _qualified_mask(target, wt_observed_matrix)
    if not bool(qualified.any()):
        return None
    value = _input_cell(univ, construct_id, records, foundation)
    value["target"] = torch.tensor(target)
    value["qualified_mask"] = torch.tensor(qualified)
    return value


def make_cells(
    univ: M2Universe, records: list[Any], foundation: FoundationCache
) -> list[dict[str, Any]]:
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    output = []
    for construct_id, cell_records in sorted(by_construct.items()):
        value = _training_cell(univ, construct_id, cell_records, foundation)
        if value is not None:
            output.append(value)
    return output


def make_prediction_cells(
    univ: M2Universe, records: list[Any], foundation: FoundationCache
) -> list[dict[str, Any]]:
    """Build held inputs without loading mutant outcomes or target masks."""
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    return [
        _input_cell(univ, construct_id, cell_records, foundation)
        for construct_id, cell_records in sorted(by_construct.items())
    ]


def _to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def build_mean_model(model_id: str, device: str) -> nn.Module:
    if model_id == CORRECTED_B1:
        model = MeanAlignedModel()
    elif model_id == PRIMARY_CANDIDATE:
        model = MutationConditionedDualTower(V4ModelConfig.primary())
    elif model_id == SCRATCH_CONTROL:
        model = MutationConditionedDualTower(V4ModelConfig.scratch())
    elif model_id == FOUNDATION_ONLY_CONTROL:
        model = RNAFMOnlyMean()
    elif model_id == CAPACITY_NULL:
        model = MutationConditionedDualTower(V4ModelConfig.capacity_null())
    else:
        raise ValueError(f"unknown frozen v4 model {model_id}")
    return model.to(device)


def _forward(
    model_id: str, model: nn.Module, cell: dict[str, Any], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    edit_idx = _to_device(cell["edit_idx"], device)
    if model_id == CORRECTED_B1:
        ctx = _to_device(cell["b1_ctx"], device)
        hidden = model.encode(ctx)
        length = hidden.shape[0]
        distance = (
            torch.arange(length, device=device)[None, :] - edit_idx[:, None]
        ).float()
        observed = _to_device(cell["wt_observed"], device)
        prediction_mask = observed[None, :].expand(len(cell["records"]), -1)
        return model.forward_mean_and_features(
            hidden,
            edit_idx,
            distance,
            cell["refs"],
            cell["alts"],
            prediction_mask,
        )
    if model_id == FOUNDATION_ONLY_CONTROL:
        return model.forward_mean_and_features(
            _to_device(cell["wt_foundation"], device),
            _to_device(cell["mutant_foundation"], device),
            edit_idx,
            cell["refs"],
            cell["alts"],
        )
    foundation_enabled = model.config.foundation_dim > 0
    return model.forward_mean_and_features(
        sequence_one_hot=_to_device(cell["sequence_one_hot"], device),
        wt_reactivity=_to_device(cell["wt_reactivity"], device),
        wt_error=_to_device(cell["wt_error"], device),
        wt_observed=_to_device(cell["wt_observed"], device),
        position=_to_device(cell["position"], device),
        region_one_hot=_to_device(cell["region_one_hot"], device),
        edit_idx=edit_idx,
        refs=cell["refs"],
        alts=cell["alts"],
        wt_foundation=(
            _to_device(cell["wt_foundation"], device) if foundation_enabled else None
        ),
        mutant_foundation=(
            _to_device(cell["mutant_foundation"], device) if foundation_enabled else None
        ),
    )


def fit_mean(
    model_id: str,
    model: nn.Module,
    cells: list[dict[str, Any]],
    *,
    device: str,
    epochs: int,
    seed: int,
) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = _scheduler(optimizer, epochs * len(cells))
    history = []
    use_amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    for epoch in range(epochs):
        losses = []
        for index in _epoch_order(len(cells), seed, epoch):
            cell = cells[index]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                mean, _features = _forward(model_id, model, cell, device)
                loss = cell_balanced_l1(
                    mean,
                    _to_device(cell["target"], device),
                    _to_device(cell["qualified_mask"], device),
                    _to_device(cell["wt"], device),
                )
            loss.backward()
            _finite_gradients(model, "mean")
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            schedule.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return history


def precompute_residual_cells(
    model_id: str,
    mean_model: nn.Module,
    cells: list[dict[str, Any]],
    device: str,
) -> list[dict[str, torch.Tensor]]:
    freeze_mean_model(mean_model)
    output = []
    with torch.no_grad():
        for cell in cells:
            mean, features = _forward(model_id, mean_model, cell, device)
            output.append(
                {
                    "mean": mean.float().cpu(),
                    "features": features.to(torch.bfloat16).cpu(),
                    "target": cell["target"],
                    "qualified_mask": cell["qualified_mask"],
                    "wt": cell["wt"],
                }
            )
    return output


def fit_calibrator(
    calibrator: ZeroMeanResidualCalibrator,
    residual_cells: list[dict[str, torch.Tensor]],
    *,
    device: str,
    epochs: int,
    seed: int,
) -> list[float]:
    calibrator.train()
    optimizer = torch.optim.AdamW(
        calibrator.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = _scheduler(optimizer, epochs * len(residual_cells))
    history = []
    for epoch in range(epochs):
        losses = []
        for index in _epoch_order(len(residual_cells), seed, epoch, offset=900_001):
            cell = residual_cells[index]
            optimizer.zero_grad(set_to_none=True)
            mean = _to_device(cell["mean"], device)
            features = _to_device(cell["features"], device).float()
            weights, locations, scales = calibrator(mean, features)
            loss = cell_balanced_crps(
                weights,
                locations,
                scales,
                _to_device(cell["target"], device),
                _to_device(cell["qualified_mask"], device),
                _to_device(cell["wt"], device),
            )
            loss.backward()
            _finite_gradients(calibrator, "calibration")
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), GRADIENT_CLIP)
            optimizer.step()
            schedule.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return history


def _prediction_arrays(
    *,
    model_id: str,
    fold: int,
    seed: int,
    keys: list[str],
    delta_mean: list[float],
    point_mean: list[float],
    locations: list[np.ndarray],
    scales: list[np.ndarray],
    weights: list[np.ndarray],
    mean_checkpoint: Path,
    calibration_checkpoint: Path,
) -> dict[str, np.ndarray]:
    n = len(keys)
    result = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "candidate_id": np.full(n, model_id, dtype=object),
        "outer_fold": np.full(n, fold, dtype=np.int64),
        "seed": np.full(n, seed, dtype=np.int64),
        "delta_mean": np.asarray(delta_mean, dtype=np.float64),
        "point_mean": np.asarray(point_mean, dtype=np.float64),
        "locations": np.asarray(locations, dtype=np.float64),
        "scales": np.asarray(scales, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "registered_status": np.full(n, "covered", dtype=object),
        "mean_checkpoint_path": np.full(n, str(mean_checkpoint), dtype=object),
        "calibration_checkpoint_path": np.full(
            n, str(calibration_checkpoint), dtype=object
        ),
    }
    if len(set(keys)) != n:
        raise RuntimeError("v4 prediction contains duplicate biological keys")
    if result["locations"].shape != (n, 2):
        raise RuntimeError("v4 residual prediction must contain two components")
    if not np.array_equal(result["locations"][:, 0], result["point_mean"]):
        raise RuntimeError("first residual location changed point mean")
    if not np.array_equal(result["locations"][:, 1], result["point_mean"]):
        raise RuntimeError("second residual location changed point mean")
    if not np.allclose(result["weights"].sum(-1), 1.0, atol=1e-7, rtol=0):
        raise RuntimeError("v4 mixture weights do not sum to one")
    if not np.isfinite(result["scales"]).all() or not (result["scales"] > 0).all():
        raise RuntimeError("v4 prediction contains invalid scales")
    return result


def predict_held(
    *,
    model_id: str,
    mean_model: nn.Module,
    calibrator: ZeroMeanResidualCalibrator,
    univ: M2Universe,
    held_records: list[Any],
    foundation: FoundationCache,
    device: str,
    fold: int,
    seed: int,
    mean_checkpoint: Path,
    calibration_checkpoint: Path,
) -> dict[str, np.ndarray]:
    cells = make_prediction_cells(univ, held_records, foundation)
    mean_model.eval()
    calibrator.eval()
    keys: list[str] = []
    delta_rows: list[float] = []
    point_rows: list[float] = []
    location_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    with torch.no_grad():
        for cell in cells:
            mean, features = _forward(model_id, mean_model, cell, device)
            component_weights, delta_locations, scales = calibrator(mean, features)
            wt = _to_device(cell["wt"], device)
            point = mean + wt[None, :]
            locations = delta_locations + wt[None, :, None]
            for row, record in enumerate(cell["records"]):
                for position in range(point.shape[1]):
                    keys.append(_bio_key(univ, record, position))
                    delta_rows.append(float(mean[row, position].cpu()))
                    point_rows.append(float(point[row, position].cpu()))
                    location_rows.append(locations[row, position].float().cpu().numpy())
                    scale_rows.append(scales[row, position].float().cpu().numpy())
                    weight_rows.append(
                        component_weights[row, position].float().cpu().numpy()
                    )
    return _prediction_arrays(
        model_id=model_id,
        fold=fold,
        seed=seed,
        keys=keys,
        delta_mean=delta_rows,
        point_mean=point_rows,
        locations=location_rows,
        scales=scale_rows,
        weights=weight_rows,
        mean_checkpoint=mean_checkpoint,
        calibration_checkpoint=calibration_checkpoint,
    )


def run_fold(
    *,
    univ: M2Universe,
    fold: Any,
    records: list[Any],
    foundation: FoundationCache,
    device: str,
    out_dir: Path,
    seed: int,
    mean_epochs: int,
    calibration_epochs: int,
    check_target_invariance: bool,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    train_cells = make_cells(univ, train_records, foundation)
    result = {
        "schema_version": SCHEMA,
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "seed": seed,
        "models": {},
        "held_score_computed": False,
        "external_outcome_accessed": False,
        "held_target_error_mask_invariance": None,
    }
    for model_index, model_id in enumerate(MODEL_UNIVERSE):
        torch.manual_seed(seed * 101 + model_index)
        model = build_mean_model(model_id, device)
        mean_parameter_count = trainable_parameter_count(model)
        mean_history = fit_mean(
            model_id, model, train_cells, device=device, epochs=mean_epochs, seed=seed
        )
        mean_checkpoint = out_dir / f"{model_id}_mean_fold{fold.outer_fold}_seed{seed}.pt"
        torch.save(model.state_dict(), mean_checkpoint)
        residual_cells = precompute_residual_cells(model_id, model, train_cells, device)
        feature_dim = int(residual_cells[0]["features"].shape[-1])
        calibrator = ZeroMeanResidualCalibrator(feature_dim=feature_dim).to(device)
        calibration_history = fit_calibrator(
            calibrator,
            residual_cells,
            device=device,
            epochs=calibration_epochs,
            seed=seed,
        )
        calibration_checkpoint = out_dir / (
            f"{model_id}_calibration_fold{fold.outer_fold}_seed{seed}.pt"
        )
        torch.save(calibrator.state_dict(), calibration_checkpoint)
        prediction = predict_held(
            model_id=model_id,
            mean_model=model,
            calibrator=calibrator,
            univ=univ,
            held_records=held_records,
            foundation=foundation,
            device=device,
            fold=int(fold.outer_fold),
            seed=seed,
            mean_checkpoint=mean_checkpoint,
            calibration_checkpoint=calibration_checkpoint,
        )
        prediction_path = out_dir / (
            f"{model_id}_predictions_fold{fold.outer_fold}_seed{seed}.npz"
        )
        np.savez_compressed(prediction_path, **prediction)
        if check_target_invariance:
            perturbed = copy.deepcopy(held_records)
            for record in perturbed:
                record.target_reactivity = 12345.0
                record.target_error = 9876.0
                record.target_observed = not bool(record.target_observed)
                record.target_mask = np.logical_not(record.target_mask)
            repeated = predict_held(
                model_id=model_id,
                mean_model=model,
                calibrator=calibrator,
                univ=univ,
                held_records=perturbed,
                foundation=foundation,
                device=device,
                fold=int(fold.outer_fold),
                seed=seed,
                mean_checkpoint=mean_checkpoint,
                calibration_checkpoint=calibration_checkpoint,
            )
            for field in ("keys", "delta_mean", "point_mean", "locations", "scales", "weights"):
                if not np.array_equal(prediction[field], repeated[field]):
                    raise RuntimeError(f"held target/error/mask changed {model_id} {field}")
        result["models"][model_id] = {
            "prediction_artifact": str(prediction_path),
            "mean_checkpoint": str(mean_checkpoint),
            "calibration_checkpoint": str(calibration_checkpoint),
            "mean_history_length": len(mean_history),
            "calibration_history_length": len(calibration_history),
            "mean_history_finite": bool(np.isfinite(mean_history).all()),
            "calibration_history_finite": bool(np.isfinite(calibration_history).all()),
            "trainable_mean_parameters": mean_parameter_count,
            "trainable_calibration_parameters": trainable_parameter_count(calibrator),
        }
        del model, calibrator, residual_cells
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["held_target_error_mask_invariance"] = bool(check_target_invariance)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=["V4M2", "V4M3", "V4M4"], required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--foundation-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    assert_run_authority(args.repo_root.resolve(), args.phase)
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(
            "launch v4 with CUDA_VISIBLE_DEVICES equal to the authorized physical GPU"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("authorized v4 GPU is unavailable")
    selected = [int(value) for value in args.folds.split(",") if value]
    if len(selected) != len(set(selected)) or not set(selected) <= set(range(20)):
        raise ValueError("v4 folds must be unique members of 0 through 19")
    if args.phase in {"V4M2", "V4M3"} and args.seed != 0:
        raise ValueError(f"{args.phase} is frozen to seed 0")
    if args.phase == "V4M4" and args.seed not in range(5):
        raise ValueError("V4M4 seed must be one of 0 through 4")
    if args.phase == "V4M2":
        if sorted(selected) != [0, 1]:
            raise ValueError("V4M2 smoke requires folds 0 and 1")
        mean_epochs = calibration_epochs = 3
    else:
        mean_epochs = MEAN_EPOCHS
        calibration_epochs = CALIBRATION_EPOCHS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    universe = M2Universe(args.m2_csv)
    universe.build()
    records = universe.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = [fold for fold in split["folds"] if fold.outer_fold in set(selected)]
    foundation = FoundationCache(args.foundation_cache)
    try:
        for fold in folds:
            print(
                f"[{args.phase}] fold={fold.outer_fold} held={fold.held_puzzle} seed={args.seed} start",
                flush=True,
            )
            result = run_fold(
                univ=universe,
                fold=fold,
                records=records,
                foundation=foundation,
                device="cuda:0",
                out_dir=args.out_dir,
                seed=args.seed,
                mean_epochs=mean_epochs,
                calibration_epochs=calibration_epochs,
                check_target_invariance=args.phase == "V4M2",
            )
            path = args.out_dir / (
                f"v4_fold_result_fold{fold.outer_fold}_seed{args.seed}.json"
            )
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"[{args.phase}] fold={fold.outer_fold} artifact={path} complete", flush=True)
    finally:
        foundation.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
