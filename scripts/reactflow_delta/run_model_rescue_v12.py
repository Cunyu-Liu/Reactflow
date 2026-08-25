#!/usr/bin/env python3
"""Run score-blind Model Rescue v12 inner-crossfit shrinkage folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache,
    fit_weighted_standardized_ridge,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    accumulate_train_stats,
    validate_cache_alignment,
)
from scripts.reactflow_delta.model_rescue_v10 import (
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    distribution_expected_absolute_delta,
    mixture_cdf_at_point,
)
from scripts.reactflow_delta.model_rescue_v11 import (
    PREDICTION_SCHEMA as V11_PREDICTION_SCHEMA,
    V11PointModel,
    freeze_point_model,
)
from scripts.reactflow_delta.model_rescue_v12 import (
    CANDIDATE,
    PREDICTION_SCHEMA,
    TASK_MATCHED_NULL,
    fit_monotone_gate,
)
from scripts.reactflow_delta.run_model_rescue_v9 import _read_json, _ridge_model
from scripts.reactflow_delta.run_model_rescue_v10 import _fit_head as fit_v10_residual_head
from scripts.reactflow_delta.run_model_rescue_v11 import (
    _feature41_matrix,
    _load_v8_mean,
    _new_residual_heads,
    _point_cells,
    fit_point_model,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.validate_model_rescue_v12_contract import (
    assert_run_authority,
)


FOLD_SCHEMA = "reactflow_delta.model_rescue_v12_fold.v1"
INNER_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v12_inner_prediction.v1"
INNER_LEDGER_SCHEMA = "reactflow_delta.model_rescue_v12_inner_crossfit_ledger.v1"


def _parse_folds(raw: str) -> list[int]:
    folds = [int(value) for value in raw.split(",") if value.strip()]
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= set(range(20)):
        raise ValueError("V12 folds must be unique members of zero through nineteen")
    return sorted(folds)


def build_inner_crossfit_ledger(fold: Any) -> list[dict[str, Any]]:
    train = set(map(str, fold.train_puzzles))
    rows = []
    seen: set[str] = set()
    for inner_fold, values in enumerate(fold.inner_groups):
        held = set(map(str, values))
        inner_train = train - held
        if not held or not inner_train or held & inner_train:
            raise RuntimeError("V12 inner split is empty or overlapping")
        if seen & held:
            raise RuntimeError("V12 inner-held puzzle appears more than once")
        seen |= held
        rows.append(
            {
                "inner_fold": inner_fold,
                "train_puzzles": sorted(inner_train),
                "held_puzzles": sorted(held),
            }
        )
    if seen != train:
        raise RuntimeError("V12 inner groups do not cover every outer-train puzzle")
    return rows


def _load_parent_row(merged: dict[str, Any], fold_id: int) -> dict[str, Any]:
    if merged.get("schema_version") != "reactflow_delta.model_rescue_v11_merged.v1":
        raise ValueError("V12 requires the complete V11 merged schema")
    if merged.get("status") != "V11M3_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V12 requires the qualified V11 complete merge")
    rows = {int(row["outer_fold"]): row for row in merged["folds"]}
    if sorted(rows) != list(range(20)):
        raise ValueError("V12 requires V11 folds0-19")
    return rows[fold_id]


def _load_parent_prediction(path: Path, fold_id: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        if str(handle["schema_version"].item()) != V11_PREDICTION_SCHEMA:
            raise ValueError("V12 parent prediction is not V11 schema")
        if set(map(int, handle["outer_fold"])) != {fold_id}:
            raise ValueError("V12 parent prediction fold mismatch")
        if set(map(int, handle["seed"])) != {0}:
            raise ValueError("V12 screen requires the authoritative V11 seed0 parent")
        result = {name: np.asarray(handle[name]) for name in handle.files}
    keys = list(map(str, result["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError("V12 parent prediction contains duplicate keys")
    if not np.array_equal(result["keys"], result["biological_scoring_key"]):
        raise ValueError("V12 parent biological scoring keys differ")
    if not np.all(np.asarray(result["registered_status"]) == "covered"):
        raise ValueError("V12 parent prediction is not fully registered")
    return result


def _fit_inner_feature41(
    univ: M2Universe,
    records: list[Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
) -> dict[str, np.ndarray | float]:
    _baseline, candidate, _counts = accumulate_train_stats(
        univ, records, unconstrained, constrained
    )
    return fit_weighted_standardized_ridge(candidate, alpha=1.0)


def _predict_inner_point(
    *,
    univ: M2Universe,
    records: list[Any],
    model: V11PointModel,
    feature41_model: dict[str, np.ndarray | float],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    outer_fold: int,
    inner_fold: int,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    values: dict[str, list[Any]] = {
        name: []
        for name in (
            "keys",
            "feature41_point",
            "parent_v11_point",
            "absolute_distance",
            "puzzle",
            "method",
            "mutant",
        )
    }
    model.eval()
    with torch.no_grad():
        for construct_id, construct_records in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            edit = torch.tensor(
                [record.full_pos for record in construct_records], device=device
            )
            signed_distance = (
                torch.arange(length, device=device)[None, :] - edit[:, None]
            ).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(construct_records), 1)),
                device=device,
            )
            _basis, feature41 = _feature41_matrix(
                construct,
                construct_records,
                feature41_model,
                unconstrained,
                constrained,
            )
            feature41_tensor = torch.tensor(feature41, device=device)
            hidden = model.encode(context_cache[construct_id])
            point = model.forward_point(
                hidden,
                edit,
                signed_distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
                feature41_tensor,
            ).cpu().numpy()
            for mutant_index, record in enumerate(construct_records):
                record_keys = [
                    _bio_key(univ, record, position) for position in range(length)
                ]
                values["keys"].extend(record_keys)
                values["feature41_point"].extend(feature41[mutant_index].tolist())
                values["parent_v11_point"].extend(point[mutant_index].tolist())
                values["absolute_distance"].extend(
                    np.abs(
                        np.arange(length, dtype=np.float64) - float(record.full_pos)
                    ).tolist()
                )
                values["puzzle"].extend([str(record.puzzle)] * length)
                values["method"].extend([str(record.method)] * length)
                values["mutant"].extend(
                    [f"{record.method}|{record.mutation_key}"] * length
                )
    keys = np.asarray(values["keys"], dtype=object)
    if len(keys) != len(set(map(str, keys))):
        raise RuntimeError("V12 inner prediction contains duplicate keys")
    n = len(keys)
    return {
        "schema_version": np.asarray(INNER_PREDICTION_SCHEMA),
        "keys": keys,
        "outer_fold": np.full(n, outer_fold, dtype=np.int64),
        "inner_fold": np.full(n, inner_fold, dtype=np.int64),
        "seed": np.full(n, seed, dtype=np.int64),
        "feature41_point": np.asarray(values["feature41_point"], dtype=np.float64),
        "parent_v11_point": np.asarray(values["parent_v11_point"], dtype=np.float64),
        "absolute_distance": np.asarray(values["absolute_distance"], dtype=np.float64),
        "puzzle": np.asarray(values["puzzle"], dtype=object),
        "method": np.asarray(values["method"], dtype=object),
        "mutant": np.asarray(values["mutant"], dtype=object),
        "registered_status": np.full(n, "covered", dtype=object),
    }


def _gate_rows_from_prediction(
    univ: M2Universe,
    records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, list[Any]]:
    index = {str(key): row for row, key in enumerate(prediction["keys"])}
    result = {
        name: []
        for name in (
            "feature41_point",
            "parent_v11_point",
            "target_delta",
            "absolute_distance",
            "puzzles",
            "methods",
            "mutants",
        )
    }
    for record in records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        qualified = construct.wt_observed & np.isfinite(target)
        for position in np.flatnonzero(qualified):
            key = _bio_key(univ, record, int(position))
            if key not in index:
                raise RuntimeError("V12 inner prediction is missing a qualified key")
            row = index[key]
            result["feature41_point"].append(float(prediction["feature41_point"][row]))
            result["parent_v11_point"].append(
                float(prediction["parent_v11_point"][row])
            )
            result["target_delta"].append(
                float(target[position] - construct.wt_reactivity[position])
            )
            result["absolute_distance"].append(
                float(prediction["absolute_distance"][row])
            )
            result["puzzles"].append(str(record.puzzle))
            result["methods"].append(str(record.method))
            result["mutants"].append(f"{record.method}|{record.mutation_key}")
    return result


def run_inner_crossfit(
    *,
    univ: M2Universe,
    outer_train_records: list[Any],
    fold: Any,
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
    out_dir: Path,
    point_epochs: int,
    gate_steps: int,
    seed: int,
) -> tuple[Any, Path]:
    ledger_rows = build_inner_crossfit_ledger(fold)
    context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in sorted({record.construct_id for record in outer_train_records})
    }
    joined = {
        name: []
        for name in (
            "feature41_point",
            "parent_v11_point",
            "target_delta",
            "absolute_distance",
            "puzzles",
            "methods",
            "mutants",
        )
    }
    executions = []
    for row in ledger_rows:
        inner_fold = int(row["inner_fold"])
        train_set = set(row["train_puzzles"])
        held_set = set(row["held_puzzles"])
        train_records = [record for record in outer_train_records if record.puzzle in train_set]
        held_records = [record for record in outer_train_records if record.puzzle in held_set]
        if set(record.puzzle for record in train_records) & set(
            record.puzzle for record in held_records
        ):
            raise RuntimeError("V12 inner-held puzzle leaked into point training")
        inner_feature41 = _fit_inner_feature41(
            univ, train_records, unconstrained, constrained
        )
        cells = _point_cells(
            univ,
            train_records,
            inner_feature41,
            unconstrained,
            constrained,
            device,
        )
        inner_seed = seed * 100_000 + int(fold.outer_fold) * 100 + inner_fold
        torch.manual_seed(inner_seed)
        model = V11PointModel(feature41_skip_multiplier=1.0).to(device)
        history = fit_point_model(
            model, cells, context_cache, epochs=point_epochs, seed=inner_seed
        )
        checkpoint = out_dir / (
            f"v12_inner_v11_outer{fold.outer_fold}_inner{inner_fold}_seed{seed}.pt"
        )
        torch.save(model.state_dict(), checkpoint)
        prediction = _predict_inner_point(
            univ=univ,
            records=held_records,
            model=model,
            feature41_model=inner_feature41,
            context_cache=context_cache,
            unconstrained=unconstrained,
            constrained=constrained,
            outer_fold=int(fold.outer_fold),
            inner_fold=inner_fold,
            seed=seed,
            device=device,
        )
        prediction_path = out_dir / (
            f"v12_inner_predictions_outer{fold.outer_fold}_inner{inner_fold}_seed{seed}.npz"
        )
        np.savez_compressed(prediction_path, **prediction)
        current = _gate_rows_from_prediction(univ, held_records, prediction)
        for name in joined:
            joined[name].extend(current[name])
        executions.append(
            {
                **row,
                "inner_seed": inner_seed,
                "point_checkpoint": str(checkpoint),
                "prediction_artifact": str(prediction_path),
                "point_training_history": history,
                "registered_prediction_rows": len(prediction["keys"]),
                "qualified_gate_rows": len(current["target_delta"]),
            }
        )
        del model, cells
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if sorted(set(joined["puzzles"])) != sorted(map(str, fold.train_puzzles)):
        raise RuntimeError("V12 inner OOF gate rows do not cover all outer-train puzzles")
    fit = fit_monotone_gate(
        feature41_point=np.asarray(joined["feature41_point"]),
        parent_v11_point=np.asarray(joined["parent_v11_point"]),
        target_delta=np.asarray(joined["target_delta"]),
        absolute_distance=np.asarray(joined["absolute_distance"]),
        puzzles=joined["puzzles"],
        methods=joined["methods"],
        mutants=joined["mutants"],
        steps=gate_steps,
        learning_rate=0.01,
        device=device,
    )
    ledger = {
        "schema_version": INNER_LEDGER_SCHEMA,
        "outer_fold": int(fold.outer_fold),
        "outer_held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "inner_folds": executions,
        "gate_parameters": fit.gate.to_dict(),
        "gate_training_history": fit.history,
        "outer_train_puzzles_covered_once": True,
        "target_values_stored": False,
        "method_used_as_gate_input": False,
        "puzzle_id_used_as_gate_input": False,
        "gate_inputs": ["absolute_distance", "absolute_feature41_point"],
    }
    ledger_path = out_dir / f"v12_inner_ledger_outer{fold.outer_fold}_seed{seed}.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return fit.gate, ledger_path


def _load_outer_parent_model(path: Path, device: str) -> V11PointModel:
    model = V11PointModel(feature41_skip_multiplier=1.0).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    freeze_point_model(model)
    model.eval()
    return model


def _candidate_calibration_cells(
    *,
    cells: list[dict[str, Any]],
    parent_model: V11PointModel,
    gate: Any,
    v8_model: Any,
    context_cache: dict[str, tuple[torch.Tensor, ...]],
) -> list[dict[str, Any]]:
    output = []
    parent_model.eval()
    gate.eval()
    v8_model.eval()
    with torch.no_grad():
        for source in cells:
            construct_id = source["construct_id"]
            parent_hidden = parent_model.encode(context_cache[construct_id])
            parent_point = parent_model.forward_point(
                parent_hidden,
                source["edit"],
                source["distance"],
                source["refs"],
                source["alts"],
                source["prediction_mask"],
                source["feature41_point"],
            )
            gate_value = gate(torch.abs(source["distance"]), source["feature41_point"])
            candidate_point = source["feature41_point"] + gate_value * (
                parent_point - source["feature41_point"]
            )
            v8_hidden = v8_model.encode(context_cache[construct_id])
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
            candidate_values = candidate_point.cpu().numpy()
            parent_values = parent_point.cpu().numpy()
            direct_values = direct.cpu().numpy()
            feature_basis = source["feature41_basis"]
            features = []
            direct_rows = []
            candidate_rows = []
            parent_rows = []
            targets = []
            mutant_index = []
            valid_mutants = 0
            for mutant in range(len(qualified)):
                receiver = np.flatnonzero(qualified[mutant])
                if not len(receiver):
                    continue
                features.append(feature_basis[mutant, receiver])
                direct_rows.append(direct_values[mutant, receiver])
                candidate_rows.append(candidate_values[mutant, receiver])
                parent_rows.append(parent_values[mutant, receiver])
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
                        "direct_features": np.concatenate(direct_rows).astype(np.float32),
                        "candidate_point": np.concatenate(candidate_rows).astype(np.float32),
                        "parent_point": np.concatenate(parent_rows).astype(np.float32),
                        "target_delta": np.concatenate(targets).astype(np.float32),
                        "mutant_index": np.concatenate(mutant_index),
                        "n_mutants": valid_mutants,
                    }
                )
    if not output:
        raise RuntimeError("V12 residual calibration produced no cells")
    return output


def _candidate_inputs(
    cells: list[dict[str, Any]],
) -> tuple[TrainOnlyStandardizer, list[np.ndarray]]:
    values = [
        calibration_input(
            cell["feature41"], cell["candidate_point"], cell["direct_features"]
        )
        for cell in cells
    ]
    standardizer = TrainOnlyStandardizer.fit(values)
    return standardizer, [standardizer.transform_numpy(value) for value in values]


def _held_distance_map(univ: M2Universe, records: list[Any]) -> dict[str, float]:
    output = {}
    for record in records:
        construct = univ.get_construct(record.construct_id)
        for position in range(len(construct.sequence)):
            key = _bio_key(univ, record, position)
            output[key] = float(abs(position - int(record.full_pos)))
    return output


def _held_candidate_distribution(
    *,
    univ: M2Universe,
    held_records: list[Any],
    parent: dict[str, np.ndarray],
    gate: Any,
    head: MedianAsymmetricResidual,
    standardizer: TrainOnlyStandardizer,
    feature41_model: dict[str, Any],
    v8_model: Any,
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
) -> dict[str, np.ndarray]:
    parent_index = {str(key): row for row, key in enumerate(parent["keys"])}
    distance_map = _held_distance_map(univ, held_records)
    keys = list(map(str, parent["keys"]))
    distance = np.asarray([distance_map[key] for key in keys], dtype=np.float64)
    feature41 = np.asarray(parent["feature41_point"], dtype=np.float64)
    parent_point = np.asarray(parent["anchored_point"], dtype=np.float64)
    with torch.no_grad():
        distance_factor, magnitude_factor = gate.factors(
            torch.tensor(distance, device=device),
            torch.tensor(feature41, device=device),
        )
        gate_value = distance_factor * magnitude_factor
        candidate_point = torch.tensor(feature41, device=device) + gate_value * (
            torch.tensor(parent_point, device=device) - torch.tensor(feature41, device=device)
        )
    by_construct: dict[str, list[Any]] = {}
    for record in held_records:
        by_construct.setdefault(record.construct_id, []).append(record)
    weights_rows = []
    location_rows = []
    scale_rows = []
    expected_rows = []
    output_key_order = []
    head.eval()
    v8_model.eval()
    with torch.no_grad():
        for construct_id, construct_records in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            edit = torch.tensor([record.full_pos for record in construct_records], device=device)
            signed_distance = (
                torch.arange(length, device=device)[None, :] - edit[:, None]
            ).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(construct_records), 1)),
                device=device,
            )
            feature_basis, _feature_point = _feature41_matrix(
                construct,
                construct_records,
                feature41_model,
                unconstrained,
                constrained,
            )
            v8_hidden = v8_model.encode(context_cache[construct_id])
            _v8_point, direct = v8_model.forward_mean_and_features(
                v8_hidden,
                edit,
                signed_distance,
                [record.ref for record in construct_records],
                [record.alt for record in construct_records],
                prediction_mask,
            )
            direct = direct.cpu().numpy()
            for mutant, record in enumerate(construct_records):
                record_keys = [_bio_key(univ, record, position) for position in range(length)]
                rows = np.asarray([parent_index[key] for key in record_keys], dtype=np.int64)
                point = candidate_point[torch.tensor(rows, device=device)]
                raw = calibration_input(feature_basis[mutant], point.cpu().numpy(), direct[mutant])
                x = torch.tensor(standardizer.transform_numpy(raw), device=device)
                mixture_weights, locations, scales = head(point, x)
                cdf = mixture_cdf_at_point(point.to(torch.float64), mixture_weights, locations, scales)
                if not torch.allclose(cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0):
                    raise RuntimeError("V12 calibration moved the gated point median")
                weights_rows.append(mixture_weights.cpu().numpy())
                location_rows.append(locations.cpu().numpy())
                scale_rows.append(scales.cpu().numpy())
                expected_rows.append(
                    distribution_expected_absolute_delta(
                        mixture_weights, locations, scales
                    ).cpu().numpy()
                )
                output_key_order.extend(record_keys)
    reorder = {key: row for row, key in enumerate(output_key_order)}
    source_rows = np.asarray([reorder[key] for key in keys], dtype=np.int64)
    return {
        "gate_distance_factor": distance_factor.cpu().numpy(),
        "gate_magnitude_factor": magnitude_factor.cpu().numpy(),
        "gate_value": gate_value.cpu().numpy(),
        "candidate_point": candidate_point.cpu().numpy(),
        "candidate_weights": np.concatenate(weights_rows)[source_rows],
        "candidate_locations": np.concatenate(location_rows)[source_rows],
        "candidate_scales": np.concatenate(scale_rows)[source_rows],
        "candidate_expected_absolute_delta": np.concatenate(expected_rows)[source_rows],
    }


def run_fold(
    *,
    univ: M2Universe,
    records: list[Any],
    fold: Any,
    parent_row: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
    out_dir: Path,
    point_epochs: int,
    gate_steps: int,
    calibration_epochs: int,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    train_set = set(fold.train_puzzles)
    outer_train_records = [record for record in records if record.puzzle in train_set]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    gate, ledger_path = run_inner_crossfit(
        univ=univ,
        outer_train_records=outer_train_records,
        fold=fold,
        unconstrained=unconstrained,
        constrained=constrained,
        device=device,
        out_dir=out_dir,
        point_epochs=point_epochs,
        gate_steps=gate_steps,
        seed=seed,
    )
    parent_prediction = _load_parent_prediction(
        Path(parent_row["prediction_artifact"]), fold_id
    )
    expected_held_keys = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    if set(map(str, parent_prediction["keys"])) != expected_held_keys:
        raise RuntimeError("V12 parent does not cover the exact held registered universe")
    parent_model = _load_outer_parent_model(
        Path(parent_row["point_checkpoints"]["anchored"]), device
    )
    feature41_model = _ridge_model(
        _read_json(Path(parent_row["feature41_model_artifact"]))["v6_feature41"]
    )
    v8_model = _load_v8_mean(Path(parent_row["v8_mean_checkpoint"]), device)
    construct_ids = sorted(
        {record.construct_id for record in outer_train_records + held_records}
    )
    context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }
    cells = _point_cells(
        univ,
        outer_train_records,
        feature41_model,
        unconstrained,
        constrained,
        device,
    )
    calibration_cells = _candidate_calibration_cells(
        cells=cells,
        parent_model=parent_model,
        gate=gate,
        v8_model=v8_model,
        context_cache=context_cache,
    )
    standardizer, inputs = _candidate_inputs(calibration_cells)
    head = _new_residual_heads(seed, device)["anchored"]
    calibration_history = fit_v10_residual_head(
        head,
        calibration_cells,
        inputs,
        "candidate_point",
        device,
        calibration_epochs,
        seed,
    )
    checkpoint_path = out_dir / f"v12_candidate_residual_fold{fold_id}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "standardizer_mean": standardizer.mean,
            "standardizer_scale": standardizer.scale,
            "point_name": "candidate",
        },
        checkpoint_path,
    )
    candidate = _held_candidate_distribution(
        univ=univ,
        held_records=held_records,
        parent=parent_prediction,
        gate=gate,
        head=head,
        standardizer=standardizer,
        feature41_model=feature41_model,
        v8_model=v8_model,
        context_cache=context_cache,
        unconstrained=unconstrained,
        constrained=constrained,
        device=device,
    )
    keys = np.asarray(parent_prediction["keys"], dtype=object)
    n = len(keys)
    prediction = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.full(n, fold_id, dtype=np.int64),
        "seed": np.full(n, seed, dtype=np.int64),
        "feature41_point": np.asarray(parent_prediction["feature41_point"], dtype=np.float64),
        "v11_parent_point": np.asarray(parent_prediction["anchored_point"], dtype=np.float64),
        "parent_weights": np.asarray(parent_prediction["anchored_weights"], dtype=np.float64),
        "parent_locations": np.asarray(parent_prediction["anchored_locations"], dtype=np.float64),
        "parent_scales": np.asarray(parent_prediction["anchored_scales"], dtype=np.float64),
        "parent_expected_absolute_delta": np.asarray(
            parent_prediction["anchored_expected_absolute_delta"], dtype=np.float64
        ),
        "feature41_weights": np.asarray(parent_prediction["feature41_weights"], dtype=np.float64),
        "feature41_locations": np.asarray(parent_prediction["feature41_locations"], dtype=np.float64),
        "feature41_scales": np.asarray(parent_prediction["feature41_scales"], dtype=np.float64),
        "feature41_expected_absolute_delta": np.asarray(
            parent_prediction["feature41_expected_absolute_delta"], dtype=np.float64
        ),
        "historical_v10_weights": np.asarray(parent_prediction["historical_v10_weights"], dtype=np.float64),
        "historical_v10_locations": np.asarray(parent_prediction["historical_v10_locations"], dtype=np.float64),
        "historical_v10_scales": np.asarray(parent_prediction["historical_v10_scales"], dtype=np.float64),
        "historical_v10_expected_absolute_delta": np.asarray(
            parent_prediction["historical_v10_expected_absolute_delta"], dtype=np.float64
        ),
        **{name: np.asarray(value, dtype=np.float64) for name, value in candidate.items()},
        "registered_status": np.full(n, "covered", dtype=object),
        "inner_crossfit_ledger_path": np.full(n, str(ledger_path), dtype=object),
    }
    prohibited = {"target", "target_error", "target_mask", "score", "crps", "signed_delta_mae"}
    if prohibited & set(prediction):
        raise RuntimeError("V12 prediction contains a prohibited target or score field")
    if not np.array_equal(prediction["v11_parent_point"], parent_prediction["anchored_point"]):
        raise RuntimeError("V12 fixed-one parent null does not exactly replay V11")
    if not np.isfinite(prediction["candidate_point"]).all():
        raise RuntimeError("V12 candidate point is non-finite")
    if not ((prediction["gate_value"] > 0.0) & (prediction["gate_value"] < 1.0)).all():
        raise RuntimeError("V12 gate left its frozen open interval")
    prediction_path = out_dir / f"v12_predictions_fold{fold_id}_seed{seed}.npz"
    np.savez_compressed(prediction_path, **prediction)
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "V12M2"
            else "DEVELOPMENT_CONSUMED_PREDICTION_ONLY"
        ),
        "candidate_id": CANDIDATE,
        "task_matched_null": TASK_MATCHED_NULL,
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "inner_point_epochs": point_epochs,
        "gate_steps": gate_steps,
        "calibration_epochs": calibration_epochs,
        "inner_crossfit_ledger": str(ledger_path),
        "gate_parameters": gate.to_dict(),
        "candidate_residual_checkpoint": str(checkpoint_path),
        "prediction_artifact": str(prediction_path),
        "authoritative_v11_parent_prediction": str(parent_row["prediction_artifact"]),
        "calibration_history": calibration_history,
        "n_registered_prediction_rows": n,
        "invariants": {
            "inner_crossfit_complete": True,
            "outer_held_target_used_for_gate_fit": False,
            "method_used_as_gate_input": False,
            "parent_v11_exact_replay": True,
            "gate_range_pass": True,
            "candidate_distribution_median_fixed": True,
            "prediction_only_artifact": True,
            "registered_prediction_coverage": 1.0,
            "failure_rate": 0.0,
            "unexpected_keys": 0,
            "partial_score_inspected": False,
            "external_outcome_accessed": False,
        },
    }
    result_path = out_dir / f"v12_fold_result_fold{fold_id}_seed{seed}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("V12M2", "V12M3", "V12M4"), required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--v11-merged-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--inner-point-epochs", type=int, required=True)
    parser.add_argument("--gate-steps", type=int, required=True)
    parser.add_argument("--calibration-epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    assert_run_authority(repo_root, args.phase)
    expected = {
        "V12M2": (3, 20, 3),
        "V12M3": (40, 500, 40),
        "V12M4": (40, 500, 40),
    }[args.phase]
    actual = (args.inner_point_epochs, args.gate_steps, args.calibration_epochs)
    if actual != expected:
        raise RuntimeError(f"V12 {args.phase} epochs or gate steps changed")
    folds = _parse_folds(args.folds)
    if args.phase == "V12M2" and not set(folds) <= {0, 1}:
        raise RuntimeError("V12M2 smoke is restricted to folds0/1")
    if args.phase == "V12M2" and args.seed != 0:
        raise RuntimeError("V12M2 smoke is restricted to seed0")
    if args.phase == "V12M3" and args.seed != 0:
        raise RuntimeError("V12M3 screen is restricted to seed0")
    if args.phase == "V12M4" and args.seed not in range(5):
        raise RuntimeError("V12M4 formal confirmation is restricted to seeds0-4")
    device = args.device if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V12 requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    merged = json.loads(args.v11_merged_json.read_text(encoding="utf-8"))
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    completed = []
    try:
        for fold_id in folds:
            result_path = args.out_dir / f"v12_fold_result_fold{fold_id}_seed{args.seed}.json"
            if result_path.exists():
                raise FileExistsError(f"V12 refuses to overwrite {result_path}")
            result = run_fold(
                univ=univ,
                records=records,
                fold=fold_map[fold_id],
                parent_row=_load_parent_row(merged, fold_id),
                unconstrained=unconstrained,
                constrained=constrained,
                device=device,
                out_dir=args.out_dir,
                point_epochs=args.inner_point_epochs,
                gate_steps=args.gate_steps,
                calibration_epochs=args.calibration_epochs,
                seed=args.seed,
                phase=args.phase,
            )
            completed.append(int(result["outer_fold"]))
    finally:
        unconstrained.close()
        constrained.close()
    print(json.dumps({"status": f"{args.phase}_RUN_COMPLETE", "folds": completed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
