#!/usr/bin/env python3
"""Run score-blind Model Rescue v11 point and residual folds."""

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
    CapacitySymmetricResidual,
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    distribution_expected_absolute_delta,
    initialize_asymmetric_from_symmetric,
    mixture_cdf_at_point,
    parameter_count,
)
from scripts.reactflow_delta.model_rescue_v11 import (
    MATCHED_NULL,
    PREDICTION_SCHEMA,
    PRIMARY_CANDIDATE,
    V11PointModel,
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
    frozen_point_from_reference,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_p3_lrso_v3 import (
    _qualified_mask,
    _target_matrix,
    _wt_filled,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


FOLD_SCHEMA = "reactflow_delta.model_rescue_v11_fold.v1"
V10_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v10_prediction.v1"
POINT_NAMES = ("feature41", "anchored", "unanchored")
EXPECTED_POINT_PARAMETERS = 1_966_433


def assert_run_authority(repo_root: Path, phase: str) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(f"V11 runner is closed outside active {phase}")
    if active.get("runnable_phases") != [phase]:
        raise RuntimeError(f"{phase} must be the only runnable phase")
    required = {
        "V11M2": "V11_REAL_DATA_ENGINEERING_SMOKE_ONLY",
        "V11M3": "V11_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY",
        "V11M4": "V11_FIXED_FIVE_SEED_FORMAL_ONLY",
    }[phase]
    if active.get("training_allowed") != required:
        raise RuntimeError(f"{phase} training authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError(f"{phase} requires held scores closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError(f"{phase} requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError(f"{phase} requires external outcomes locked")
    if active.get("v10_terminal_verdict_change_allowed") is not False:
        raise RuntimeError("V11 cannot change the terminal V10 verdict")


def _parse_folds(raw: str) -> list[int]:
    folds = [int(value) for value in raw.split(",") if value.strip()]
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= set(range(20)):
        raise ValueError("V11 folds must be unique members of zero through nineteen")
    return sorted(folds)


def _load_v8_mean(checkpoint: Path, device: str) -> MeanAlignedModel:
    model = MeanAlignedModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    freeze_mean_model(model)
    return model


def _fold_sources(
    fold_id: int,
    *,
    v8_dir: Path,
    v10_dir: Path,
    tic2a_merged: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    v8 = _read_json(
        v8_dir / f"v8_corrected_expert_fold_result_fold{fold_id}_seed0.json"
    )
    if v8.get("schema_version") != V8_FOLD_SCHEMA:
        raise ValueError(f"V11 fold {fold_id} lacks corrected V8 expert schema")
    if v8.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise ValueError(f"V11 fold {fold_id} lacks exact V8 target identity")
    if v8.get("legacy_v3_checkpoint_reused") is not False:
        raise ValueError(f"V11 fold {fold_id} reused an invalid legacy expert")

    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA:
        raise ValueError("V11 requires corrected TIC2A merge")
    tic_rows = {int(row["outer_fold"]): row for row in tic2a_merged["folds"]}
    if sorted(tic_rows) != list(range(20)):
        raise ValueError("V11 requires the complete corrected TIC2A universe")
    tic = tic_rows[fold_id]
    feature41_model = _ridge_model(
        _read_json(Path(tic["model_artifact"]))["v6_feature41"]
    )

    v10 = _read_json(v10_dir / f"v10_fold_result_fold{fold_id}_seed0.json")
    if v10.get("schema_version") != "reactflow_delta.model_rescue_v10_fold.v1":
        raise ValueError(f"V11 fold {fold_id} lacks terminal V10 comparator")
    return v8, tic, v10, feature41_model


def _feature41_matrix(
    construct: Any,
    records: list[Any],
    feature41_model: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
) -> tuple[np.ndarray, np.ndarray]:
    receiver = np.arange(len(construct.sequence), dtype=np.int64)
    bases = []
    points = []
    for record in records:
        _feature30, feature41 = prediction_features(
            construct, record, receiver, unconstrained, constrained
        )
        bases.append(feature41.astype(np.float32))
        points.append(
            predict_weighted_ridge(feature41_model, feature41)[:, 0].astype(
                np.float32
            )
        )
    return np.stack(bases), np.stack(points)


def _point_cells(
    univ: M2Universe,
    records: list[Any],
    feature41_model: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
) -> list[dict[str, Any]]:
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    cells = []
    for construct_id, construct_records in sorted(by_construct.items()):
        construct = univ.get_construct(construct_id)
        target, wt_observed = _target_matrix(univ, construct_records)
        qualified = _qualified_mask(target, wt_observed)
        if not bool(qualified.any()):
            continue
        length = len(construct.sequence)
        edit = torch.tensor(
            [record.full_pos for record in construct_records], device=device
        )
        distance = (
            torch.arange(length, device=device)[None, :] - edit[:, None]
        ).float()
        feature41_basis, feature41_point = _feature41_matrix(
            construct,
            construct_records,
            feature41_model,
            unconstrained,
            constrained,
        )
        cells.append(
            {
                "construct_id": construct_id,
                "records": construct_records,
                "edit": edit,
                "distance": distance,
                "refs": [record.ref for record in construct_records],
                "alts": [record.alt for record in construct_records],
                "target": torch.tensor(target, device=device),
                "prediction_mask": torch.tensor(wt_observed, device=device),
                "qualified_mask": torch.tensor(qualified, device=device),
                "wt": torch.tensor(_wt_filled(univ, construct_id), device=device),
                "feature41_basis": feature41_basis,
                "feature41_point": torch.tensor(feature41_point, device=device),
            }
        )
    if not cells:
        raise RuntimeError("V11 point training produced no outer-train cells")
    return cells


def _epoch_order(n_cells: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n_cells))
    random.Random(seed * 100_003 + epoch).shuffle(order)
    return order


def _finite_gradients(module: torch.nn.Module, stage: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise RuntimeError(f"nonfinite V11 gradient in {stage}: {name}")


def fit_point_model(
    model: V11PointModel,
    cells: list[dict[str, Any]],
    context_cache: dict[str, Any],
    *,
    epochs: int,
    seed: int,
) -> list[float]:
    # Resetting the RNG makes candidate and null use the same dropout stream.
    torch.manual_seed(seed + 1_100_000)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    history = []
    for epoch in range(epochs):
        losses = []
        for index in _epoch_order(len(cells), seed, epoch):
            cell = cells[index]
            hidden = model.encode(context_cache[cell["construct_id"]])
            point = model.forward_point(
                hidden,
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
                cell["feature41_point"],
            )
            loss = method_cell_balanced_l1(
                point,
                cell["target"],
                cell["qualified_mask"],
                cell["wt"],
            )
            optimizer.zero_grad()
            loss.backward()
            _finite_gradients(model, "point")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("V11 point history is incomplete or nonfinite")
    return history


def _module_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _assert_unchanged(
    expected: dict[str, torch.Tensor], module: torch.nn.Module, label: str
) -> None:
    actual = module.state_dict()
    for name, value in expected.items():
        if not torch.equal(value, actual[name].detach().cpu()):
            raise RuntimeError(f"V11 calibration changed frozen {label}: {name}")


def _calibration_cells(
    cells: list[dict[str, Any]],
    *,
    anchored: V11PointModel,
    unanchored: V11PointModel,
    v8_model: MeanAlignedModel,
    v11_context_cache: dict[str, Any],
    v8_context_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    anchored.eval()
    unanchored.eval()
    v8_model.eval()
    with torch.no_grad():
        for source in cells:
            construct_id = source["construct_id"]
            anchored_hidden = anchored.encode(v11_context_cache[construct_id])
            unanchored_hidden = unanchored.encode(v11_context_cache[construct_id])
            anchored_point = anchored.forward_point(
                anchored_hidden,
                source["edit"],
                source["distance"],
                source["refs"],
                source["alts"],
                source["prediction_mask"],
                source["feature41_point"],
            )
            unanchored_point = unanchored.forward_point(
                unanchored_hidden,
                source["edit"],
                source["distance"],
                source["refs"],
                source["alts"],
                source["prediction_mask"],
                source["feature41_point"],
            )
            v8_hidden = v8_model.encode(v8_context_cache[construct_id])
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
            feature41_point = source["feature41_point"].cpu().numpy()
            anchored_values = anchored_point.cpu().numpy()
            unanchored_values = unanchored_point.cpu().numpy()
            direct_values = direct.cpu().numpy()
            feature_basis = source["feature41_basis"]
            rows = {name: [] for name in POINT_NAMES}
            features = []
            direct_rows = []
            targets = []
            mutant_index = []
            valid_mutants = 0
            for mutant in range(len(qualified)):
                receiver = np.flatnonzero(qualified[mutant])
                if not len(receiver):
                    continue
                features.append(feature_basis[mutant, receiver])
                direct_rows.append(direct_values[mutant, receiver])
                rows["feature41"].append(feature41_point[mutant, receiver])
                rows["anchored"].append(anchored_values[mutant, receiver])
                rows["unanchored"].append(unanchored_values[mutant, receiver])
                targets.append(target[mutant, receiver] - wt[receiver])
                mutant_index.append(
                    np.full(len(receiver), valid_mutants, dtype=np.int64)
                )
                valid_mutants += 1
            if valid_mutants:
                output.append(
                    {
                        "construct_id": construct_id,
                        "feature41": np.concatenate(features).astype(np.float32),
                        "direct_features": np.concatenate(direct_rows).astype(
                            np.float32
                        ),
                        "feature41_point": np.concatenate(rows["feature41"]).astype(
                            np.float32
                        ),
                        "anchored_point": np.concatenate(rows["anchored"]).astype(
                            np.float32
                        ),
                        "unanchored_point": np.concatenate(
                            rows["unanchored"]
                        ).astype(np.float32),
                        "target_delta": np.concatenate(targets).astype(np.float32),
                        "mutant_index": np.concatenate(mutant_index),
                        "n_mutants": valid_mutants,
                    }
                )
    if not output:
        raise RuntimeError("V11 residual calibration produced no training cells")
    return output


def _prepare_calibration_inputs(
    cells: list[dict[str, Any]], point_name: str
) -> tuple[TrainOnlyStandardizer, list[np.ndarray]]:
    point_field = f"{point_name}_point"
    values = [
        calibration_input(
            cell["feature41"], cell[point_field], cell["direct_features"]
        )
        for cell in cells
    ]
    standardizer = TrainOnlyStandardizer.fit(values)
    return standardizer, [
        standardizer.transform_numpy(value) for value in values
    ]


def _new_residual_heads(seed: int, device: str) -> dict[str, MedianAsymmetricResidual]:
    # Reproduce the V10 seed-0 feature41 initialization exactly: V10 first
    # creates one symmetric template, two deep copies, then two asymmetric heads.
    torch.manual_seed(seed)
    template = CapacitySymmetricResidual().to(device)
    feature41_template = copy.deepcopy(template)
    anchored_template = copy.deepcopy(template)
    unanchored_template = copy.deepcopy(template)
    feature41 = MedianAsymmetricResidual().to(device)
    _discarded_v10_second_head = MedianAsymmetricResidual().to(device)
    anchored = MedianAsymmetricResidual().to(device)
    unanchored = MedianAsymmetricResidual().to(device)
    initialize_asymmetric_from_symmetric(feature41_template, feature41)
    initialize_asymmetric_from_symmetric(anchored_template, anchored)
    initialize_asymmetric_from_symmetric(unanchored_template, unanchored)
    return {
        "feature41": feature41,
        "anchored": anchored,
        "unanchored": unanchored,
    }


def _load_historical_v10(path: Path, fold_id: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as handle:
        if str(handle["schema_version"].item()) != V10_PREDICTION_SCHEMA:
            raise ValueError("V11 historical comparator is not V10 prediction schema")
        if set(map(int, handle["outer_fold"])) != {fold_id}:
            raise ValueError("V11 historical V10 fold mismatch")
        keys = list(map(str, handle["keys"]))
        return {
            "index": {key: index for index, key in enumerate(keys)},
            "feature41_weights": np.asarray(handle["feature41_asymmetric_weights"]),
            "feature41_locations": np.asarray(handle["feature41_asymmetric_locations"]),
            "feature41_scales": np.asarray(handle["feature41_asymmetric_scales"]),
            "v10_weights": np.asarray(handle["meanaligned_asymmetric_weights"]),
            "v10_locations": np.asarray(handle["meanaligned_asymmetric_locations"]),
            "v10_scales": np.asarray(handle["meanaligned_asymmetric_scales"]),
            "v10_expected_absolute_delta": np.asarray(
                handle["meanaligned_asymmetric_expected_absolute_delta"]
            ),
        }


def _held_prediction(
    *,
    univ: M2Universe,
    held_records: list[Any],
    feature41_model: dict[str, Any],
    anchored: V11PointModel,
    unanchored: V11PointModel,
    v8_model: MeanAlignedModel,
    heads: dict[str, MedianAsymmetricResidual],
    standardizers: dict[str, TrainOnlyStandardizer],
    v11_context_cache: dict[str, Any],
    v8_context_cache: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    fold_id: int,
    seed: int,
    v8_prediction_path: Path,
    tic2a_prediction_path: Path,
    historical_v10_path: Path,
) -> dict[str, np.ndarray]:
    v8_reference = _load_reference_prediction(
        v8_prediction_path,
        V8_PREDICTION_SCHEMA,
        "meanaligned_delta_mean",
        fold_id,
    )
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
    point_values["v8"] = []
    distributions = {
        f"{name}_{suffix}": []
        for name in POINT_NAMES
        for suffix in ("weights", "locations", "scales", "expected_absolute_delta")
    }
    anchored.eval()
    unanchored.eval()
    v8_model.eval()
    for head in heads.values():
        head.eval()
    with torch.no_grad():
        for construct_id, construct_records in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            edit = torch.tensor(
                [record.full_pos for record in construct_records],
                device=next(anchored.parameters()).device,
            )
            distance = (
                torch.arange(length, device=edit.device)[None, :] - edit[:, None]
            ).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(construct_records), 1)),
                device=edit.device,
            )
            feature41_basis, feature41_matrix = _feature41_matrix(
                construct,
                construct_records,
                feature41_model,
                unconstrained,
                constrained,
            )
            feature41_tensor = torch.tensor(feature41_matrix, device=edit.device)
            anchored_hidden = anchored.encode(v11_context_cache[construct_id])
            unanchored_hidden = unanchored.encode(v11_context_cache[construct_id])
            anchored_matrix = anchored.forward_point(
                anchored_hidden,
                edit,
                distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
                feature41_tensor,
            ).cpu().numpy()
            unanchored_matrix = unanchored.forward_point(
                unanchored_hidden,
                edit,
                distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
                feature41_tensor,
            ).cpu().numpy()
            v8_hidden = v8_model.encode(v8_context_cache[construct_id])
            _v8_matrix, direct_matrix = v8_model.forward_mean_and_features(
                v8_hidden,
                edit,
                distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
            )
            direct_matrix = direct_matrix.cpu().numpy()
            for mutant, record in enumerate(construct_records):
                record_keys = [
                    _bio_key(univ, record, receiver) for receiver in range(length)
                ]
                current = {
                    "feature41": feature41_matrix[mutant].astype(np.float64),
                    "anchored": anchored_matrix[mutant].astype(np.float64),
                    "unanchored": unanchored_matrix[mutant].astype(np.float64),
                }
                v8_point = frozen_point_from_reference(record_keys, v8_reference)
                direct = direct_matrix[mutant].astype(np.float32)
                for name in POINT_NAMES:
                    raw_input = calibration_input(
                        feature41_basis[mutant], current[name], direct
                    )
                    standardized = torch.tensor(
                        standardizers[name].transform_numpy(raw_input),
                        device=edit.device,
                    )
                    point_tensor = torch.tensor(current[name], device=edit.device)
                    weights, locations, scales = heads[name](
                        point_tensor, standardized
                    )
                    cdf = mixture_cdf_at_point(
                        point_tensor.to(torch.float64), weights, locations, scales
                    )
                    if not torch.allclose(
                        cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0
                    ):
                        raise RuntimeError("V11 residual calibration moved point median")
                    expected_absolute = distribution_expected_absolute_delta(
                        weights, locations, scales
                    )
                    distributions[f"{name}_weights"].append(weights.cpu().numpy())
                    distributions[f"{name}_locations"].append(
                        locations.cpu().numpy()
                    )
                    distributions[f"{name}_scales"].append(scales.cpu().numpy())
                    distributions[f"{name}_expected_absolute_delta"].append(
                        expected_absolute.cpu().numpy()
                    )
                    point_values[name].append(current[name])
                point_values["v8"].append(v8_point)
                keys.extend(record_keys)

    if len(keys) != len(set(keys)):
        raise RuntimeError("V11 held prediction contains duplicate biological keys")
    if set(keys) != set(v8_reference) or set(keys) != set(tic_reference):
        raise RuntimeError("V11 held keys do not replay V8 and TIC2A")
    feature41_point = np.concatenate(point_values["feature41"])
    if not np.allclose(
        feature41_point,
        [tic_reference[key] for key in keys],
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("V11 feature41 point does not replay TIC2A")
    if set(keys) != set(historical["index"]):
        raise RuntimeError("V11 held keys do not replay terminal V10")
    historical_rows = np.asarray(
        [historical["index"][key] for key in keys], dtype=np.int64
    )

    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(len(keys), fold_id, dtype=np.int64),
        "seed": np.full(len(keys), seed, dtype=np.int64),
        "registered_status": np.full(len(keys), "covered", dtype=object),
        "feature41_point": feature41_point.astype(np.float64),
        "v8_point": np.concatenate(point_values["v8"]).astype(np.float64),
        "anchored_point": np.concatenate(point_values["anchored"]).astype(
            np.float64
        ),
        "unanchored_point": np.concatenate(point_values["unanchored"]).astype(
            np.float64
        ),
        "historical_v10_weights": historical["v10_weights"][historical_rows].astype(
            np.float64
        ),
        "historical_v10_locations": historical["v10_locations"][
            historical_rows
        ].astype(np.float64),
        "historical_v10_scales": historical["v10_scales"][historical_rows].astype(
            np.float64
        ),
        "historical_v10_expected_absolute_delta": historical[
            "v10_expected_absolute_delta"
        ][historical_rows].astype(np.float64),
    }
    for name, values in distributions.items():
        output[name] = np.concatenate(values).astype(np.float64)

    if seed == 0:
        for suffix in ("weights", "locations", "scales"):
            current = output[f"feature41_{suffix}"]
            expected = historical[f"feature41_{suffix}"][historical_rows]
            if not np.allclose(current, expected, atol=1e-7, rtol=0.0):
                raise RuntimeError(
                    f"V11 feature41 asymmetric replay differs for {suffix}"
                )
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
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    v8_row, tic_row, v10_row, feature41_model = _fold_sources(
        fold_id,
        v8_dir=v8_dir,
        v10_dir=v10_dir,
        tic2a_merged=tic2a_merged,
    )
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    construct_ids = sorted(
        {record.construct_id for record in train_records + held_records}
    )
    v11_context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }
    v8_context_cache = v11_context_cache
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
        raise RuntimeError("V11 feature41 replay exceeds 1e-7")

    cells = _point_cells(
        univ,
        train_records,
        feature41_model,
        unconstrained,
        constrained,
        device,
    )
    anchored, unanchored = make_exact_matched_pair(seed=seed, device=device)
    if trainable_parameter_count(anchored) != EXPECTED_POINT_PARAMETERS:
        raise RuntimeError("V11 point parameter count differs from frozen contract")
    assert_exact_trainable_match(anchored, unanchored)
    anchored_history = fit_point_model(
        anchored,
        cells,
        v11_context_cache,
        epochs=point_epochs,
        seed=seed,
    )
    unanchored_history = fit_point_model(
        unanchored,
        cells,
        v11_context_cache,
        epochs=point_epochs,
        seed=seed,
    )
    anchored_checkpoint = out_dir / (
        f"v11_anchored_point_fold{fold_id}_seed{seed}.pt"
    )
    unanchored_checkpoint = out_dir / (
        f"v11_unanchored_point_fold{fold_id}_seed{seed}.pt"
    )
    torch.save(anchored.state_dict(), anchored_checkpoint)
    torch.save(unanchored.state_dict(), unanchored_checkpoint)

    freeze_point_model(anchored)
    freeze_point_model(unanchored)
    anchored_snapshot = _module_snapshot(anchored)
    unanchored_snapshot = _module_snapshot(unanchored)
    v8_model = _load_v8_mean(Path(v8_row["meanaligned_checkpoint"]), device)
    calibration_cells = _calibration_cells(
        cells,
        anchored=anchored,
        unanchored=unanchored,
        v8_model=v8_model,
        v11_context_cache=v11_context_cache,
        v8_context_cache=v8_context_cache,
    )
    heads = _new_residual_heads(seed, device)
    standardizers: dict[str, TrainOnlyStandardizer] = {}
    calibration_inputs: dict[str, list[np.ndarray]] = {}
    histories = {}
    for name in POINT_NAMES:
        standardizers[name], calibration_inputs[name] = _prepare_calibration_inputs(
            calibration_cells, name
        )
        histories[name] = fit_v10_residual_head(
            heads[name],
            calibration_cells,
            calibration_inputs[name],
            f"{name}_point",
            device,
            calibration_epochs,
            seed,
        )
    _assert_unchanged(anchored_snapshot, anchored, "anchored point")
    _assert_unchanged(unanchored_snapshot, unanchored, "unanchored point")
    if any(parameter.grad is not None for parameter in anchored.parameters()):
        raise RuntimeError("V11 calibration produced anchored point gradients")
    if any(parameter.grad is not None for parameter in unanchored.parameters()):
        raise RuntimeError("V11 calibration produced unanchored point gradients")

    residual_checkpoints = {}
    for name, head in heads.items():
        path = out_dir / f"v11_{name}_asymmetric_fold{fold_id}_seed{seed}.pt"
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
        anchored=anchored,
        unanchored=unanchored,
        v8_model=v8_model,
        heads=heads,
        standardizers=standardizers,
        v11_context_cache=v11_context_cache,
        v8_context_cache=v8_context_cache,
        unconstrained=unconstrained,
        constrained=constrained,
        fold_id=fold_id,
        seed=seed,
        v8_prediction_path=Path(v8_row["expert_prediction_artifact"]),
        tic2a_prediction_path=Path(tic_row["prediction_artifact"]),
        historical_v10_path=Path(v10_row["prediction_artifact"]),
    )
    prediction_path = out_dir / f"v11_predictions_fold{fold_id}_seed{seed}.npz"
    np.savez_compressed(prediction_path, **prediction)
    return {
        "schema_version": FOLD_SCHEMA,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "V11M2"
            else "DEVELOPMENT_CONSUMED_PREDICTION_ONLY"
        ),
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "point_epochs": point_epochs,
        "calibration_epochs": calibration_epochs,
        "point_checkpoints": {
            "anchored": str(anchored_checkpoint),
            "unanchored": str(unanchored_checkpoint),
        },
        "v8_mean_checkpoint": str(v8_row["meanaligned_checkpoint"]),
        "feature41_model_artifact": str(tic_row["model_artifact"]),
        "historical_v10_prediction_artifact": str(v10_row["prediction_artifact"]),
        "residual_checkpoints": residual_checkpoints,
        "prediction_artifact": str(prediction_path),
        "training_histories": {
            "anchored_point": anchored_history,
            "unanchored_point": unanchored_history,
            **{f"{name}_residual": values for name, values in histories.items()},
        },
        "n_train_cells": len(cells),
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "feature41_replay_max_abs_difference": replay,
        "point_parameter_counts": {
            PRIMARY_CANDIDATE: trainable_parameter_count(anchored),
            MATCHED_NULL: trainable_parameter_count(unanchored),
        },
        "residual_parameter_counts": {
            name: parameter_count(head) for name, head in heads.items()
        },
        "invariants": {
            "target_profile_identity_exact": True,
            "exact_point_parameter_match": True,
            "fixed_skip_only_model_difference": True,
            "same_point_training_order_and_dropout_stream": True,
            "point_frozen_during_calibration": True,
            "v10_residual_family_reused": True,
            "feature41_replay_at_1e_7": True,
            "feature41_asymmetric_seed0_replay_or_not_applicable": True,
            "median_constraint_all_held_rows": True,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("V11M2", "V11M3", "V11M4"), required=True)
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
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    assert_run_authority(repo_root, args.phase)
    folds = _parse_folds(args.folds)
    if args.phase == "V11M2":
        if args.seed != 0 or not set(folds) <= {0, 1}:
            raise ValueError("V11M2 is frozen to seed0 folds0/1")
        if (args.point_epochs, args.calibration_epochs) != (3, 3):
            raise ValueError("V11M2 is frozen to three plus three epochs")
    elif args.phase == "V11M3":
        if args.seed != 0 or (args.point_epochs, args.calibration_epochs) != (40, 40):
            raise ValueError("V11M3 is frozen to seed0 and forty plus forty epochs")
    else:
        if args.seed not in range(5) or (
            args.point_epochs,
            args.calibration_epochs,
        ) != (40, 40):
            raise ValueError("V11M4 is frozen to seeds0-4 and forty plus forty epochs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in folds:
        result_path = args.out_dir / (
            f"v11_fold_result_fold{fold_id}_seed{args.seed}.json"
        )
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite V11 fold {fold_id}")

    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V11 requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=20260813
    )
    selected = [
        fold for fold in split["folds"] if int(fold.outer_fold) in set(folds)
    ]
    if len(selected) != len(folds):
        raise ValueError("one or more requested V11 folds are absent")
    tic2a_merged = _read_json(args.tic2a_merged_json)
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            print(
                f"[{args.phase}] fold={fold_id} held={fold.held_puzzle} "
                f"seed={args.seed} start",
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
            )
            result_path = args.out_dir / (
                f"v11_fold_result_fold{fold_id}_seed{args.seed}.json"
            )
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
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
