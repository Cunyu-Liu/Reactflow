#!/usr/bin/env python3
"""Run prediction-only V13 folds under the active score-blind authority."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v2 import MeanAlignedModel, freeze_mean_model
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
    CapacitySymmetricResidual,
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    distribution_expected_absolute_delta,
    initialize_asymmetric_from_symmetric,
    mixture_cdf_at_point,
    parameter_count,
)
from scripts.reactflow_delta.model_rescue_v13 import (
    MATCHED_NULL,
    EXPECTED_POINT_PARAMETERS,
    MUTANT_MICROBATCH,
    PREDICTION_SCHEMA,
    PRIMARY_CANDIDATE,
    V13PointModel,
    assert_exact_trainable_match,
    freeze_point_model,
    make_exact_matched_pair,
    method_cell_balanced_l1,
    trainable_parameter_count,
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
from scripts.reactflow_delta.run_model_rescue_v10 import (
    _fit_head as fit_v10_residual_head,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    V10_PREDICTION_SCHEMA,
    _feature41_matrix,
    _fold_sources,
    _load_authoritative_v10_feature41,
    _load_historical_v10,
    _load_v8_mean,
    _point_cells,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.validate_model_rescue_v13_contract import (
    assert_run_authority,
)


FOLD_SCHEMA = "reactflow_delta.model_rescue_v13_fold.v1"
POINT_NAMES = ("feature41", "candidate", "null")


def _parse_folds(raw: str) -> list[int]:
    folds = [int(value) for value in raw.split(",") if value.strip()]
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= set(range(20)):
        raise ValueError("V13 folds must be unique members of zero through nineteen")
    return sorted(folds)


def _epoch_order(n_cells: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n_cells))
    random.Random(seed * 100_003 + epoch).shuffle(order)
    return order


def _finite_gradients(module: torch.nn.Module, stage: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite V13 gradient in {stage}: {name}")


def fit_point_model(
    model: V13PointModel,
    cells: list[dict[str, Any]],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    *,
    epochs: int,
    seed: int,
    microbatch: int,
) -> list[float]:
    torch.manual_seed(seed + 1_300_000)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    history = []
    for epoch in range(epochs):
        losses = []
        for index in _epoch_order(len(cells), seed, epoch):
            cell = cells[index]
            point = model.forward_point(
                context_cache[cell["construct_id"]],
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
                cell["feature41_point"],
                microbatch=microbatch,
            )
            loss = method_cell_balanced_l1(
                point, cell["target"], cell["qualified_mask"], cell["wt"]
            )
            optimizer.zero_grad()
            loss.backward()
            _finite_gradients(model, "point")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("V13 point history is incomplete or nonfinite")
    return history


def _module_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _assert_unchanged(
    expected: dict[str, torch.Tensor], module: torch.nn.Module, label: str
) -> None:
    current = module.state_dict()
    for name, value in expected.items():
        if not torch.equal(value, current[name].detach().cpu()):
            raise RuntimeError(f"V13 calibration changed frozen {label}: {name}")


def _calibration_cells(
    cells: list[dict[str, Any]],
    *,
    candidate: V13PointModel,
    null: V13PointModel,
    v8_model: MeanAlignedModel,
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    microbatch: int,
) -> list[dict[str, Any]]:
    output = []
    candidate.eval()
    null.eval()
    v8_model.eval()
    with torch.no_grad():
        for source in cells:
            context = context_cache[source["construct_id"]]
            points = {
                "feature41": source["feature41_point"],
                "candidate": candidate.forward_point(
                    context,
                    source["edit"],
                    source["distance"],
                    source["refs"],
                    source["alts"],
                    source["prediction_mask"],
                    source["feature41_point"],
                    microbatch=microbatch,
                ),
                "null": null.forward_point(
                    context,
                    source["edit"],
                    source["distance"],
                    source["refs"],
                    source["alts"],
                    source["prediction_mask"],
                    source["feature41_point"],
                    microbatch=microbatch,
                ),
            }
            v8_hidden = v8_model.encode(context)
            _v8_point, direct = v8_model.forward_mean_and_features(
                v8_hidden,
                source["edit"],
                source["distance"],
                source["refs"],
                source["alts"],
                source["prediction_mask"],
            )
            qualified = source["qualified_mask"].cpu().numpy()
            target = source["target"].cpu().numpy()
            wt = source["wt"].cpu().numpy()
            point_arrays = {name: value.cpu().numpy() for name, value in points.items()}
            direct_array = direct.cpu().numpy()
            rows = {name: [] for name in POINT_NAMES}
            feature_rows = []
            direct_rows = []
            targets = []
            mutant_index = []
            valid_mutants = 0
            for mutant in range(len(qualified)):
                receiver = np.flatnonzero(qualified[mutant])
                if not len(receiver):
                    continue
                feature_rows.append(source["feature41_basis"][mutant, receiver])
                direct_rows.append(direct_array[mutant, receiver])
                for name in POINT_NAMES:
                    rows[name].append(point_arrays[name][mutant, receiver])
                targets.append(target[mutant, receiver] - wt[receiver])
                mutant_index.append(np.full(len(receiver), valid_mutants, dtype=np.int64))
                valid_mutants += 1
            if valid_mutants:
                output.append(
                    {
                        "construct_id": source["construct_id"],
                        "feature41": np.concatenate(feature_rows).astype(np.float32),
                        "direct_features": np.concatenate(direct_rows).astype(np.float32),
                        **{
                            f"{name}_point": np.concatenate(rows[name]).astype(np.float32)
                            for name in POINT_NAMES
                        },
                        "target_delta": np.concatenate(targets).astype(np.float32),
                        "mutant_index": np.concatenate(mutant_index),
                        "n_mutants": valid_mutants,
                    }
                )
    if not output:
        raise RuntimeError("V13 residual calibration produced no training cells")
    return output


def _prepare_calibration_inputs(
    cells: list[dict[str, Any]], point_name: str
) -> tuple[TrainOnlyStandardizer, list[np.ndarray]]:
    values = [
        calibration_input(
            cell["feature41"], cell[f"{point_name}_point"], cell["direct_features"]
        )
        for cell in cells
    ]
    standardizer = TrainOnlyStandardizer.fit(values)
    return standardizer, [standardizer.transform_numpy(value) for value in values]


def _new_residual_heads(
    seed: int, device: str
) -> dict[str, MedianAsymmetricResidual]:
    torch.manual_seed(seed)
    template = CapacitySymmetricResidual().to(device)
    templates = {name: copy.deepcopy(template) for name in POINT_NAMES}
    heads = {name: MedianAsymmetricResidual().to(device) for name in POINT_NAMES}
    for name in POINT_NAMES:
        initialize_asymmetric_from_symmetric(templates[name], heads[name])
    return heads


def _held_prediction(
    *,
    univ: M2Universe,
    held_records: list[Any],
    feature41_model: dict[str, Any],
    candidate: V13PointModel,
    null: V13PointModel,
    v8_model: MeanAlignedModel,
    heads: dict[str, MedianAsymmetricResidual],
    standardizers: dict[str, TrainOnlyStandardizer],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    fold_id: int,
    seed: int,
    tic2a_prediction_path: Path,
    historical_v10_path: Path,
    use_authoritative_feature41: bool,
    microbatch: int,
) -> dict[str, np.ndarray]:
    tic_reference = _load_reference_prediction(
        tic2a_prediction_path,
        TIC2A_PREDICTION_SCHEMA,
        "v6_feature41_signed_delta",
        fold_id,
    )
    historical = _load_historical_v10(historical_v10_path, fold_id)
    by_construct: dict[str, list[Any]] = {}
    for record in held_records:
        by_construct.setdefault(record.construct_id, []).append(record)

    keys: list[str] = []
    point_values = {name: [] for name in POINT_NAMES}
    distributions = {
        f"{name}_{suffix}": []
        for name in POINT_NAMES
        for suffix in ("weights", "locations", "scales", "expected_absolute_delta")
    }
    null_delta_max = 0.0
    candidate.eval()
    null.eval()
    v8_model.eval()
    for head in heads.values():
        head.eval()
    with torch.no_grad():
        for construct_id, construct_records in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            device = next(candidate.parameters()).device
            edit = torch.tensor([record.full_pos for record in construct_records], device=device)
            distance = (torch.arange(length, device=device)[None, :] - edit[:, None]).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(construct_records), 1)),
                device=device,
            )
            feature_basis, feature_matrix = _feature41_matrix(
                construct, construct_records, feature41_model, unconstrained, constrained
            )
            feature_tensor = torch.tensor(feature_matrix, device=device)
            context = context_cache[construct_id]
            candidate_matrix = candidate.forward_point(
                context,
                edit,
                distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
                feature_tensor,
                microbatch=microbatch,
            ).cpu().numpy()
            null_matrix = null.forward_point(
                context,
                edit,
                distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
                feature_tensor,
                microbatch=microbatch,
            ).cpu().numpy()
            null_wt, null_second = null.encode_paired_passes(
                context,
                edit,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                microbatch=microbatch,
            )
            null_delta_max = max(
                null_delta_max,
                float(torch.max(torch.abs(null_second - null_wt)).cpu()),
            )
            v8_hidden = v8_model.encode(context)
            _v8_point, direct_matrix = v8_model.forward_mean_and_features(
                v8_hidden,
                edit,
                distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
            )
            direct_matrix = direct_matrix.cpu().numpy()
            for mutant, record in enumerate(construct_records):
                record_keys = [_bio_key(univ, record, receiver) for receiver in range(length)]
                current = {
                    "feature41": feature_matrix[mutant].astype(np.float64),
                    "candidate": candidate_matrix[mutant].astype(np.float64),
                    "null": null_matrix[mutant].astype(np.float64),
                }
                direct = direct_matrix[mutant].astype(np.float32)
                for name in POINT_NAMES:
                    raw = calibration_input(feature_basis[mutant], current[name], direct)
                    standardized = torch.tensor(
                        standardizers[name].transform_numpy(raw), device=device
                    )
                    point = torch.tensor(current[name], device=device)
                    weights, locations, scales = heads[name](point, standardized)
                    cdf = mixture_cdf_at_point(
                        point.to(torch.float64), weights, locations, scales
                    )
                    if not torch.allclose(
                        cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0
                    ):
                        raise RuntimeError("V13 residual calibration moved the point median")
                    distributions[f"{name}_weights"].append(weights.cpu().numpy())
                    distributions[f"{name}_locations"].append(locations.cpu().numpy())
                    distributions[f"{name}_scales"].append(scales.cpu().numpy())
                    distributions[f"{name}_expected_absolute_delta"].append(
                        distribution_expected_absolute_delta(
                            weights, locations, scales
                        ).cpu().numpy()
                    )
                    point_values[name].append(current[name])
                keys.extend(record_keys)

    if len(keys) != len(set(keys)) or set(keys) != set(tic_reference):
        raise RuntimeError("V13 held biological key universe is invalid")
    feature41_point = np.concatenate(point_values["feature41"])
    if not np.allclose(
        feature41_point, [tic_reference[key] for key in keys], atol=1e-7, rtol=0.0
    ):
        raise RuntimeError("V13 feature41 point does not replay TIC2A")
    if null_delta_max > 1e-7:
        raise RuntimeError("V13 WT-replay null hidden difference exceeds 1e-7")

    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(len(keys), fold_id, dtype=np.int64),
        "seed": np.full(len(keys), seed, dtype=np.int64),
        "registered_status": np.full(len(keys), "covered", dtype=object),
        "feature41_point": feature41_point.astype(np.float64),
        "candidate_point": np.concatenate(point_values["candidate"]).astype(np.float64),
        "null_point": np.concatenate(point_values["null"]).astype(np.float64),
        "null_hidden_delta_max_abs": np.full(len(keys), null_delta_max, dtype=np.float64),
    }
    for name, values in distributions.items():
        output[name] = np.concatenate(values).astype(np.float64)

    if use_authoritative_feature41:
        if set(keys) != set(historical["index"]):
            raise RuntimeError("V13 held keys do not replay terminal V10")
        rows = np.asarray([historical["index"][key] for key in keys], dtype=np.int64)
        for suffix in ("weights", "locations", "scales", "expected_absolute_delta"):
            output[f"feature41_{suffix}"] = historical[f"feature41_{suffix}"][rows].astype(np.float64)
    return output


def run_fold(
    *,
    univ: M2Universe,
    records: list[Any],
    fold: Any,
    device: str,
    out_dir: Path,
    v8_dir: Path,
    v10_dir: Path,
    tic2a_merged: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    point_epochs: int,
    calibration_epochs: int,
    seed: int,
    phase: str,
    microbatch: int,
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    v8_row, tic_row, v10_row, feature41_model = _fold_sources(
        fold_id, v8_dir=v8_dir, v10_dir=v10_dir, tic2a_merged=tic2a_merged
    )
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    construct_ids = sorted(
        {record.construct_id for record in train_records + held_records}
    )
    context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }
    replay = _feature41_replay_max_difference(
        univ,
        held_records,
        feature41_model,
        unconstrained,
        constrained,
        Path(tic_row["prediction_artifact"]),
        fold_id,
    )
    if replay > 1e-7:
        raise RuntimeError("V13 feature41 replay exceeds 1e-7")
    cells = _point_cells(
        univ, train_records, feature41_model, unconstrained, constrained, device
    )
    candidate, null = make_exact_matched_pair(seed=seed, device=device)
    assert_exact_trainable_match(candidate, null)
    counts = {
        PRIMARY_CANDIDATE: trainable_parameter_count(candidate),
        MATCHED_NULL: trainable_parameter_count(null),
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError("V13 point parameter counts differ")
    if set(counts.values()) != {EXPECTED_POINT_PARAMETERS}:
        raise RuntimeError("V13 point parameter count differs from the frozen contract")
    candidate_history = fit_point_model(
        candidate,
        cells,
        context_cache,
        epochs=point_epochs,
        seed=seed,
        microbatch=microbatch,
    )
    null_history = fit_point_model(
        null,
        cells,
        context_cache,
        epochs=point_epochs,
        seed=seed,
        microbatch=microbatch,
    )
    candidate_checkpoint = out_dir / f"v13_candidate_point_fold{fold_id}_seed{seed}.pt"
    null_checkpoint = out_dir / f"v13_null_point_fold{fold_id}_seed{seed}.pt"
    torch.save(candidate.state_dict(), candidate_checkpoint)
    torch.save(null.state_dict(), null_checkpoint)
    freeze_point_model(candidate)
    freeze_point_model(null)
    candidate_snapshot = _module_snapshot(candidate)
    null_snapshot = _module_snapshot(null)

    v8_model = _load_v8_mean(Path(v8_row["meanaligned_checkpoint"]), device)
    calibration_cells = _calibration_cells(
        cells,
        candidate=candidate,
        null=null,
        v8_model=v8_model,
        context_cache=context_cache,
        microbatch=microbatch,
    )
    heads = _new_residual_heads(seed, device)
    standardizers: dict[str, TrainOnlyStandardizer] = {}
    histories: dict[str, list[float]] = {}
    for name in POINT_NAMES:
        standardizers[name], inputs = _prepare_calibration_inputs(
            calibration_cells, name
        )
        if name == "feature41" and phase != "V13M2" and seed == 0:
            authoritative_head, authoritative_standardizer, history = (
                _load_authoritative_v10_feature41(v10_row, device)
            )
            if not np.array_equal(
                standardizers[name].mean, authoritative_standardizer.mean
            ) or not np.array_equal(
                standardizers[name].scale, authoritative_standardizer.scale
            ):
                raise RuntimeError("V13 feature41 standardizer does not replay V10")
            heads[name] = authoritative_head
            standardizers[name] = authoritative_standardizer
            histories[name] = history
        else:
            histories[name] = fit_v10_residual_head(
                heads[name],
                calibration_cells,
                inputs,
                f"{name}_point",
                device,
                calibration_epochs,
                seed,
            )
    _assert_unchanged(candidate_snapshot, candidate, "candidate point")
    _assert_unchanged(null_snapshot, null, "null point")
    if any(parameter.grad is not None for parameter in candidate.parameters()):
        raise RuntimeError("V13 calibration produced candidate point gradients")
    if any(parameter.grad is not None for parameter in null.parameters()):
        raise RuntimeError("V13 calibration produced null point gradients")

    residual_checkpoints = {}
    for name, head in heads.items():
        path = out_dir / f"v13_{name}_asymmetric_fold{fold_id}_seed{seed}.pt"
        torch.save(
            {
                "state_dict": head.state_dict(),
                "standardizer_mean": standardizers[name].mean,
                "standardizer_scale": standardizers[name].scale,
                "point_name": name,
            },
            path,
        )
        residual_checkpoints[name] = str(path)

    prediction = _held_prediction(
        univ=univ,
        held_records=held_records,
        feature41_model=feature41_model,
        candidate=candidate,
        null=null,
        v8_model=v8_model,
        heads=heads,
        standardizers=standardizers,
        context_cache=context_cache,
        unconstrained=unconstrained,
        constrained=constrained,
        fold_id=fold_id,
        seed=seed,
        tic2a_prediction_path=Path(tic_row["prediction_artifact"]),
        historical_v10_path=Path(v10_row["prediction_artifact"]),
        use_authoritative_feature41=(phase != "V13M2" and seed == 0),
        microbatch=microbatch,
    )
    prediction_path = out_dir / f"v13_predictions_fold{fold_id}_seed{seed}.npz"
    np.savez_compressed(prediction_path, **prediction)
    return {
        "schema_version": FOLD_SCHEMA,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "V13M2"
            else "DEVELOPMENT_CONSUMED_PREDICTION_ONLY"
        ),
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "point_epochs": point_epochs,
        "calibration_epochs": calibration_epochs,
        "mutant_microbatch": microbatch,
        "point_checkpoints": {
            "candidate": str(candidate_checkpoint),
            "null": str(null_checkpoint),
        },
        "residual_checkpoints": residual_checkpoints,
        "prediction_artifact": str(prediction_path),
        "history_lengths": {
            "candidate_point": len(candidate_history),
            "null_point": len(null_history),
            **{f"{name}_residual": len(value) for name, value in histories.items()},
        },
        "point_parameter_counts": counts,
        "residual_parameter_counts": {
            name: parameter_count(head) for name, head in heads.items()
        },
        "n_train_cells": len(cells),
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "feature41_replay_max_abs_difference": replay,
        "invariants": {
            "target_profile_identity_exact": True,
            "exact_point_parameter_and_initial_state_match": True,
            "second_pass_sequence_is_only_candidate_null_difference": True,
            "candidate_exact_mutant_null_wt_replay": True,
            "null_hidden_delta_at_most_1e_7": True,
            "same_point_training_order_and_dropout_seed": True,
            "point_frozen_during_calibration": True,
            "v10_residual_family_reused": True,
            "feature41_replay_at_1e_7": True,
            "median_constraint_all_held_rows": True,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("V13M2", "V13M3", "V13M4"), required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--v10-dir", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--point-epochs", type=int, required=True)
    parser.add_argument("--calibration-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mutant-microbatch", type=int, default=MUTANT_MICROBATCH)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    assert_run_authority(repo_root, args.phase)
    folds = _parse_folds(args.folds)
    if args.phase == "V13M2":
        if args.seed != 0 or not set(folds) <= {0, 1}:
            raise ValueError("V13M2 is frozen to seed0 folds0/1")
        if (args.point_epochs, args.calibration_epochs) != (3, 3):
            raise ValueError("V13M2 is frozen to three plus three epochs")
    elif args.phase == "V13M3":
        if args.seed != 0 or (args.point_epochs, args.calibration_epochs) != (40, 40):
            raise ValueError("V13M3 is frozen to seed0 and forty plus forty epochs")
    elif args.seed not in range(5) or (
        args.point_epochs,
        args.calibration_epochs,
    ) != (40, 40):
        raise ValueError("V13M4 is frozen to seeds0-4 and forty plus forty epochs")
    if args.mutant_microbatch <= 0 or args.mutant_microbatch > MUTANT_MICROBATCH:
        raise ValueError("V13 microbatch may only be the frozen value or an OOM reduction")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in folds:
        result_path = args.out_dir / f"v13_fold_result_fold{fold_id}_seed{args.seed}.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite V13 fold {fold_id}")

    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V13 requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = [fold for fold in split["folds"] if int(fold.outer_fold) in set(folds)]
    if len(selected) != len(folds):
        raise ValueError("one or more requested V13 folds are absent")
    tic2a_merged = _read_json(args.tic2a_merged_json)
    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA:
        raise ValueError("V13 requires the corrected TIC2A merge")
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            print(
                f"[{args.phase}] fold={fold_id} held={fold.held_puzzle} seed={args.seed} start",
                flush=True,
            )
            result = run_fold(
                univ=univ,
                records=records,
                fold=fold,
                device=device,
                out_dir=args.out_dir,
                v8_dir=args.v8_dir,
                v10_dir=args.v10_dir,
                tic2a_merged=tic2a_merged,
                unconstrained=unconstrained,
                constrained=constrained,
                point_epochs=args.point_epochs,
                calibration_epochs=args.calibration_epochs,
                seed=args.seed,
                phase=args.phase,
                microbatch=args.mutant_microbatch,
            )
            result_path = args.out_dir / f"v13_fold_result_fold{fold_id}_seed{args.seed}.json"
            result_path.write_text(
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
