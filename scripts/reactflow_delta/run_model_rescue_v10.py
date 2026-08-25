#!/usr/bin/env python3
"""Train the frozen V10 symmetric-null and median-asymmetric residual heads."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v2 import (
    MeanAlignedModel,
    freeze_mean_model,
    gaussian_mixture_crps_torch,
)
from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache,
    predict_weighted_ridge,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    prediction_features,
    validate_cache_alignment,
)
from scripts.reactflow_delta.model_rescue_v10 import (
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
    CapacitySymmetricResidual,
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    distribution_expected_absolute_delta,
    initialize_asymmetric_from_symmetric,
    mixture_cdf_at_point,
    parameter_count,
)
from scripts.reactflow_delta.run_model_rescue_v9 import (
    TIC2A_MERGED_SCHEMA,
    TIC2A_PREDICTION_SCHEMA,
    V8_FOLD_SCHEMA,
    V8_PREDICTION_SCHEMA,
    _feature41_replay_max_difference,
    _load_reference_prediction,
    _read_json,
    _ridge_model,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_p3_lrso_v3 import _qualified_mask, _target_matrix
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


V9_MERGED_SCHEMA = "reactflow_delta.model_rescue_v9_merged.v1"
V9_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v9_prediction.v1"
HEAD_NAMES = (
    "feature41_symmetric",
    "feature41_asymmetric",
    "meanaligned_symmetric",
    "meanaligned_asymmetric",
)


def frozen_point_from_reference(
    keys: list[str], reference: dict[str, float]
) -> np.ndarray:
    missing = [key for key in keys if key not in reference]
    if missing:
        raise RuntimeError("V10 frozen point reference is missing held keys")
    return np.asarray([reference[key] for key in keys], dtype=np.float64)


def assert_run_authority(repo_root: Path, phase: str) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(f"V10 runner is closed outside active {phase}")
    if active.get("runnable_phases") != [phase]:
        raise RuntimeError(f"{phase} must be the only runnable phase")
    required = {
        "V10M1": "V10_REAL_DATA_ENGINEERING_SMOKE_ONLY",
        "V10M2": "V10_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY",
    }[phase]
    if active.get("training_allowed") != required:
        raise RuntimeError(f"{phase} training authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError(f"{phase} requires held score access closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError(f"{phase} requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError(f"{phase} requires external outcomes locked")
    if active.get("v8_point_retraining_allowed") is not False:
        raise RuntimeError("V10 cannot retrain the frozen V8 point")


def _load_mean_model(checkpoint: Path, device: str) -> MeanAlignedModel:
    model = MeanAlignedModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    freeze_mean_model(model)
    return model


def _fold_inputs(
    fold_id: int,
    v8_dir: Path,
    tic2a_merged: dict[str, Any],
    v9_merged: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    v8_path = v8_dir / f"v8_corrected_expert_fold_result_fold{fold_id}_seed0.json"
    v8_row = _read_json(v8_path)
    if v8_row.get("schema_version") != V8_FOLD_SCHEMA:
        raise ValueError(f"fold {fold_id} lacks corrected V8 schema")
    if v8_row.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise ValueError(f"fold {fold_id} lacks exact V8 target identity")
    if v8_row.get("legacy_v3_checkpoint_reused") is not False:
        raise ValueError(f"fold {fold_id} reused invalid legacy V3 state")
    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA:
        raise ValueError("V10 requires corrected TIC2A merge")
    tic_rows = {int(row["outer_fold"]): row for row in tic2a_merged["folds"]}
    if sorted(tic_rows) != list(range(20)):
        raise ValueError("V10 requires complete TIC2A fold universe")
    tic_row = tic_rows[fold_id]
    feature41_model = _ridge_model(
        _read_json(Path(tic_row["model_artifact"]))["v6_feature41"]
    )
    if v9_merged.get("schema_version") != V9_MERGED_SCHEMA or v9_merged.get(
        "status"
    ) != "V9M2_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V10 requires complete terminal V9 predictions")
    v9_rows = {int(row["outer_fold"]): row for row in v9_merged["folds"]}
    if sorted(v9_rows) != list(range(20)):
        raise ValueError("V10 requires complete V9 fold universe")
    return v8_row, tic_row, v9_rows[fold_id], feature41_model


def _calibration_cells(
    univ: M2Universe,
    records: list[Any],
    feature41_model: dict[str, Any],
    mean_model: MeanAlignedModel,
    ctx_cache: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
) -> list[dict[str, Any]]:
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    cells = []
    mean_model.eval()
    with torch.no_grad():
        for construct_id, recs in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            target, wt_observed = _target_matrix(univ, recs)
            qualified = _qualified_mask(target, wt_observed)
            length = len(construct.sequence)
            edit = torch.tensor([record.full_pos for record in recs], device=device)
            distance = (
                torch.arange(length, device=device)[None, :] - edit[:, None]
            ).float()
            prediction_mask = torch.tensor(wt_observed, device=device)
            hidden = mean_model.encode(ctx_cache[construct_id])
            mean, direct = mean_model.forward_mean_and_features(
                hidden,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            mean = mean.cpu().numpy()
            direct = direct.cpu().numpy()
            x_rows = []
            direct_rows = []
            feature41_point = []
            meanaligned_point = []
            target_delta = []
            mutant_index = []
            valid_mutants = 0
            for row, record in enumerate(recs):
                receiver = np.flatnonzero(qualified[row])
                if not len(receiver):
                    continue
                _feature30, feature41 = prediction_features(
                    construct, record, receiver, unconstrained, constrained
                )
                ridge = predict_weighted_ridge(feature41_model, feature41)[:, 0]
                signed = (
                    target[row, receiver].astype(np.float64)
                    - construct.wt_reactivity[receiver].astype(np.float64)
                )
                x_rows.append(feature41.astype(np.float32))
                direct_rows.append(direct[row, receiver].astype(np.float32))
                feature41_point.append(ridge.astype(np.float32))
                meanaligned_point.append(mean[row, receiver].astype(np.float32))
                target_delta.append(signed.astype(np.float32))
                mutant_index.append(
                    np.full(len(receiver), valid_mutants, dtype=np.int64)
                )
                valid_mutants += 1
            if valid_mutants:
                cells.append(
                    {
                        "construct_id": construct_id,
                        "feature41": np.concatenate(x_rows),
                        "direct_features": np.concatenate(direct_rows),
                        "feature41_point": np.concatenate(feature41_point),
                        "meanaligned_point": np.concatenate(meanaligned_point),
                        "target_delta": np.concatenate(target_delta),
                        "mutant_index": np.concatenate(mutant_index),
                        "n_mutants": valid_mutants,
                    }
                )
    if not cells:
        raise RuntimeError("V10 calibration produced no outer-train cells")
    return cells


def _prepare_inputs(
    cells: list[dict[str, Any]], point_field: str
) -> tuple[TrainOnlyStandardizer, list[np.ndarray]]:
    raw = [
        calibration_input(
            cell["feature41"], cell[point_field], cell["direct_features"]
        )
        for cell in cells
    ]
    standardizer = TrainOnlyStandardizer.fit(raw)
    return standardizer, [standardizer.transform_numpy(value) for value in raw]


def _mutant_balanced_crps(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    mutant_index: torch.Tensor,
    n_mutants: int,
) -> torch.Tensor:
    values = gaussian_mixture_crps_torch(locations, scales, weights, target)
    sums = torch.zeros(n_mutants, device=values.device, dtype=values.dtype)
    counts = torch.zeros(n_mutants, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, mutant_index, values)
    counts.scatter_add_(0, mutant_index, torch.ones_like(values))
    if not bool((counts > 0.0).all()):
        raise RuntimeError("V10 calibration contains an empty mutant")
    return (sums / counts).mean()


def _fit_head(
    head: CapacitySymmetricResidual | MedianAsymmetricResidual,
    cells: list[dict[str, Any]],
    inputs: list[np.ndarray],
    point_field: str,
    device: str,
    epochs: int,
    seed: int,
) -> list[float]:
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0.0)
    history = []
    for epoch in range(epochs):
        order = list(range(len(cells)))
        random.Random(seed * 100_003 + epoch).shuffle(order)
        losses = []
        for index in order:
            cell = cells[index]
            x = torch.tensor(inputs[index], device=device)
            point = torch.tensor(cell[point_field], device=device)
            target = torch.tensor(cell["target_delta"], device=device)
            mutant_index = torch.tensor(cell["mutant_index"], device=device)
            weights, locations, scales = head(point, x)
            loss = _mutant_balanced_crps(
                weights,
                locations,
                scales,
                target,
                mutant_index,
                int(cell["n_mutants"]),
            )
            optimizer.zero_grad()
            loss.backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError(f"nonfinite V10 gradient in {name}")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("V10 calibration history is incomplete or nonfinite")
    return history


def _new_heads(seed: int, device: str) -> dict[str, torch.nn.Module]:
    torch.manual_seed(seed)
    template = CapacitySymmetricResidual().to(device)
    feature41_symmetric = copy.deepcopy(template)
    meanaligned_symmetric = copy.deepcopy(template)
    feature41_asymmetric = MedianAsymmetricResidual().to(device)
    meanaligned_asymmetric = MedianAsymmetricResidual().to(device)
    initialize_asymmetric_from_symmetric(
        feature41_symmetric, feature41_asymmetric
    )
    initialize_asymmetric_from_symmetric(
        meanaligned_symmetric, meanaligned_asymmetric
    )
    return {
        "feature41_symmetric": feature41_symmetric,
        "feature41_asymmetric": feature41_asymmetric,
        "meanaligned_symmetric": meanaligned_symmetric,
        "meanaligned_asymmetric": meanaligned_asymmetric,
    }


def _load_v9_prediction(path: Path, fold_id: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as handle:
        if str(handle["schema_version"].item()) != V9_PREDICTION_SCHEMA:
            raise ValueError("V10 historical comparator is not V9 prediction schema")
        if set(map(int, handle["outer_fold"])) != {fold_id}:
            raise ValueError("V10 historical V9 fold mismatch")
        keys = list(map(str, handle["keys"]))
        return {
            "index": {key: index for index, key in enumerate(keys)},
            "weights": np.asarray(handle["meanaligned_weights"]),
            "locations": np.asarray(handle["meanaligned_locations"]),
            "scales": np.asarray(handle["meanaligned_scales"]),
            "expected_absolute_delta": np.asarray(
                handle["meanaligned_expected_absolute_delta"]
            ),
        }


def _held_prediction(
    *,
    univ: M2Universe,
    held_records: list[Any],
    feature41_model: dict[str, Any],
    mean_model: MeanAlignedModel,
    heads: dict[str, torch.nn.Module],
    feature41_standardizer: TrainOnlyStandardizer,
    meanaligned_standardizer: TrainOnlyStandardizer,
    ctx_cache: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
    fold_id: int,
    v8_prediction_path: Path,
    tic2a_prediction_path: Path,
    v9_prediction_path: Path,
) -> dict[str, np.ndarray]:
    v8_reference = _load_reference_prediction(
        v8_prediction_path,
        V8_PREDICTION_SCHEMA,
        "meanaligned_delta_mean",
        fold_id,
    )
    by_construct: dict[str, list[Any]] = {}
    for record in held_records:
        by_construct.setdefault(record.construct_id, []).append(record)
    result: dict[str, list[np.ndarray]] = {
        "feature41_point": [],
        "meanaligned_point": [],
    }
    for name in HEAD_NAMES:
        for suffix in ("weights", "locations", "scales", "expected_absolute_delta"):
            result[f"{name}_{suffix}"] = []
    keys: list[str] = []
    for head in heads.values():
        head.eval()
    mean_model.eval()
    with torch.no_grad():
        for construct_id, recs in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            receiver = np.arange(length, dtype=np.int64)
            edit = torch.tensor([record.full_pos for record in recs], device=device)
            distance = (
                torch.arange(length, device=device)[None, :] - edit[:, None]
            ).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(recs), 1)),
                device=device,
            )
            hidden = mean_model.encode(ctx_cache[construct_id])
            _mean_matrix, direct_matrix = mean_model.forward_mean_and_features(
                hidden,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            for row, record in enumerate(recs):
                record_keys = [
                    _bio_key(univ, record, position) for position in range(length)
                ]
                _feature30, feature41 = prediction_features(
                    construct, record, receiver, unconstrained, constrained
                )
                feature41_point = predict_weighted_ridge(
                    feature41_model, feature41
                )[:, 0].astype(np.float32)
                # The V8 prediction artifact is the authoritative frozen point.
                # Recomputing the same checkpoint can differ by a few float32
                # ULPs across CUDA kernels, which is enough to violate the
                # pre-frozen 1e-7 replay invariant without changing semantics.
                meanaligned_point = frozen_point_from_reference(
                    record_keys, v8_reference
                )
                direct = direct_matrix[row].cpu().numpy().astype(np.float32)
                f_input = torch.tensor(
                    feature41_standardizer.transform_numpy(
                        calibration_input(feature41, feature41_point, direct)
                    ),
                    device=device,
                )
                m_input = torch.tensor(
                    meanaligned_standardizer.transform_numpy(
                        calibration_input(feature41, meanaligned_point, direct)
                    ),
                    device=device,
                )
                point_tensors = {
                    "feature41": torch.tensor(feature41_point, device=device),
                    "meanaligned": torch.tensor(meanaligned_point, device=device),
                }
                input_tensors = {"feature41": f_input, "meanaligned": m_input}
                for name, head in heads.items():
                    point_name = "feature41" if name.startswith("feature41") else "meanaligned"
                    point = point_tensors[point_name]
                    weights, locations, scales = head(point, input_tensors[point_name])
                    if name.endswith("asymmetric"):
                        cdf = mixture_cdf_at_point(point, weights, locations, scales)
                        if not torch.allclose(
                            cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0
                        ):
                            raise RuntimeError("V10 held distribution moved its median")
                    expected_abs = distribution_expected_absolute_delta(
                        weights, locations, scales
                    )
                    result[f"{name}_weights"].append(weights.cpu().numpy())
                    result[f"{name}_locations"].append(locations.cpu().numpy())
                    result[f"{name}_scales"].append(scales.cpu().numpy())
                    result[f"{name}_expected_absolute_delta"].append(
                        expected_abs.cpu().numpy()
                    )
                result["feature41_point"].append(feature41_point)
                result["meanaligned_point"].append(meanaligned_point)
                keys.extend(record_keys)
    key_array = np.asarray(keys, dtype=object)
    feature41_point = np.concatenate(result.pop("feature41_point")).astype(np.float64)
    meanaligned_point = np.concatenate(result.pop("meanaligned_point")).astype(np.float64)
    tic_reference = _load_reference_prediction(
        tic2a_prediction_path,
        TIC2A_PREDICTION_SCHEMA,
        "v6_feature41_signed_delta",
        fold_id,
    )
    if set(keys) != set(v8_reference) or set(keys) != set(tic_reference):
        raise RuntimeError("V10 held keys do not replay V8/TIC2A")
    if not np.array_equal(
        meanaligned_point, frozen_point_from_reference(keys, v8_reference)
    ):
        raise RuntimeError("V10 MeanAligned point does not replay V8")
    if not np.allclose(
        feature41_point, [tic_reference[key] for key in keys], atol=1e-7, rtol=0.0
    ):
        raise RuntimeError("V10 feature41 point does not replay TIC2A")
    historical = _load_v9_prediction(v9_prediction_path, fold_id)
    if set(keys) != set(historical["index"]):
        raise RuntimeError("V10 keys do not replay historical V9")
    history_rows = np.asarray([historical["index"][key] for key in keys], dtype=np.int64)
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": key_array,
        "biological_scoring_key": key_array.copy(),
        "outer_fold": np.full(len(keys), fold_id, dtype=np.int64),
        "seed": np.zeros(len(keys), dtype=np.int64),
        "registered_status": np.full(len(keys), "covered", dtype=object),
        "feature41_point": feature41_point,
        "meanaligned_point": meanaligned_point,
        "historical_v9_weights": historical["weights"][history_rows].astype(np.float64),
        "historical_v9_locations": historical["locations"][history_rows].astype(np.float64),
        "historical_v9_scales": historical["scales"][history_rows].astype(np.float64),
        "historical_v9_expected_absolute_delta": historical[
            "expected_absolute_delta"
        ][history_rows].astype(np.float64),
    }
    for name, arrays in result.items():
        output[name] = np.concatenate(arrays).astype(np.float64)
    return output


def run_fold(
    *,
    univ: M2Universe,
    records: list[Any],
    fold: Any,
    device: str,
    out_dir: Path,
    v8_dir: Path,
    tic2a_merged: dict[str, Any],
    v9_merged: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    epochs: int,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    v8_row, tic2a_row, v9_row, feature41_model = _fold_inputs(
        fold_id, v8_dir, tic2a_merged, v9_merged
    )
    mean_model = _load_mean_model(Path(v8_row["meanaligned_checkpoint"]), device)
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    ctx_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in sorted(
            {record.construct_id for record in train_records + held_records}
        )
    }
    replay = _feature41_replay_max_difference(
        univ,
        held_records,
        feature41_model,
        unconstrained,
        constrained,
        Path(tic2a_row["prediction_artifact"]),
        fold_id,
    )
    if replay > 1e-7:
        raise RuntimeError("V10 feature41 replay exceeds 1e-7")
    cells = _calibration_cells(
        univ,
        train_records,
        feature41_model,
        mean_model,
        ctx_cache,
        unconstrained,
        constrained,
        device,
    )
    feature41_standardizer, feature41_inputs = _prepare_inputs(
        cells, "feature41_point"
    )
    meanaligned_standardizer, meanaligned_inputs = _prepare_inputs(
        cells, "meanaligned_point"
    )
    heads = _new_heads(seed, device)
    histories = {}
    for name in HEAD_NAMES:
        point_name = "feature41" if name.startswith("feature41") else "meanaligned"
        histories[name] = _fit_head(
            heads[name],
            cells,
            feature41_inputs if point_name == "feature41" else meanaligned_inputs,
            f"{point_name}_point",
            device,
            epochs,
            seed,
        )
    checkpoints = {}
    for name, head in heads.items():
        path = out_dir / f"v10_{name}_fold{fold_id}_seed0.pt"
        standardizer = (
            feature41_standardizer
            if name.startswith("feature41")
            else meanaligned_standardizer
        )
        torch.save(
            {
                "state_dict": head.state_dict(),
                "standardizer_mean": standardizer.mean,
                "standardizer_scale": standardizer.scale,
                "head_name": name,
            },
            path,
        )
        checkpoints[name] = str(path)
    prediction = _held_prediction(
        univ=univ,
        held_records=held_records,
        feature41_model=feature41_model,
        mean_model=mean_model,
        heads=heads,
        feature41_standardizer=feature41_standardizer,
        meanaligned_standardizer=meanaligned_standardizer,
        ctx_cache=ctx_cache,
        unconstrained=unconstrained,
        constrained=constrained,
        device=device,
        fold_id=fold_id,
        v8_prediction_path=Path(v8_row["expert_prediction_artifact"]),
        tic2a_prediction_path=Path(tic2a_row["prediction_artifact"]),
        v9_prediction_path=Path(v9_row["prediction_artifact"]),
    )
    prediction_path = out_dir / f"v10_predictions_fold{fold_id}_seed0.npz"
    np.savez_compressed(prediction_path, **prediction)
    return {
        "schema_version": FOLD_SCHEMA,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "V10M1"
            else "DEVELOPMENT_CONSUMED_PREDICTION_ONLY_SCREEN"
        ),
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "epochs": epochs,
        "v8_mean_checkpoint": str(v8_row["meanaligned_checkpoint"]),
        "feature41_model_artifact": str(tic2a_row["model_artifact"]),
        "historical_v9_prediction_artifact": str(v9_row["prediction_artifact"]),
        "checkpoints": checkpoints,
        "prediction_artifact": str(prediction_path),
        "training_histories": histories,
        "n_train_cells": len(cells),
        "n_registered_prediction_rows": len(prediction["keys"]),
        "feature41_replay_max_abs_difference": replay,
        "parameter_counts": {
            name: parameter_count(head) for name, head in heads.items()
        },
        "invariants": {
            "target_profile_identity_exact": True,
            "v8_point_replay_at_1e_7": True,
            "tic2a_feature41_replay_at_1e_7": True,
            "outer_train_only_standardization": True,
            "trained_v8_direct_features_only": True,
            "fair_feature41_and_meanaligned_head_families": True,
            "median_constraint_all_held_rows": True,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
    }


def _parse_folds(raw: str) -> list[int]:
    folds = [int(value) for value in raw.split(",") if value.strip()]
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= set(range(20)):
        raise ValueError("V10 folds must be unique members of 0 through 19")
    return sorted(folds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("V10M1", "V10M2"), required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--v9-merged-json", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    assert_run_authority(args.repo_root.resolve(), args.phase)
    folds = _parse_folds(args.folds)
    if args.seed != 0:
        raise ValueError("V10M1/V10M2 are frozen to seed 0")
    if args.phase == "V10M1" and (not set(folds) <= {0, 1} or args.epochs != 3):
        raise ValueError("V10M1 is frozen to folds0/1 and 3 epochs")
    if args.phase == "V10M2" and args.epochs != 40:
        raise ValueError("V10M2 is frozen to 40 epochs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in folds:
        result = args.out_dir / f"v10_fold_result_fold{fold_id}_seed0.json"
        if result.exists():
            raise FileExistsError(f"refusing to overwrite V10 fold {fold_id}")
    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V10 requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = [fold for fold in split["folds"] if int(fold.outer_fold) in folds]
    if len(selected) != len(folds):
        raise ValueError("one or more requested V10 folds are absent")
    tic2a_merged = _read_json(args.tic2a_merged_json)
    v9_merged = _read_json(args.v9_merged_json)
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            print(f"[{args.phase}] fold={fold_id} held={fold.held_puzzle} start", flush=True)
            result = run_fold(
                univ=univ,
                records=records,
                fold=fold,
                device=device,
                out_dir=args.out_dir,
                v8_dir=args.v8_dir,
                tic2a_merged=tic2a_merged,
                v9_merged=v9_merged,
                unconstrained=unconstrained,
                constrained=constrained,
                epochs=args.epochs,
                seed=args.seed,
                phase=args.phase,
            )
            path = args.out_dir / f"v10_fold_result_fold{fold_id}_seed0.json"
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"[{args.phase}] fold={fold_id} complete", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
