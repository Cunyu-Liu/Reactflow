#!/usr/bin/env python3
"""Train fair zero-mean residual heads around frozen V9 signed means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v2 import MeanAlignedModel, freeze_mean_model
from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache,
    predict_weighted_ridge,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    CANDIDATE_PROBE_FEATURE_NAMES,
    ConstrainedFeatureCache,
    prediction_features,
    validate_cache_alignment,
)
from scripts.reactflow_delta.model_rescue_v9 import (
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
    EquiCalibratedZeroMeanMixture,
    assert_zero_mean_distribution,
    expected_absolute_delta,
    mutant_balanced_crps,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_p3_lrso_v3 import _qualified_mask, _target_matrix
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


V8_FOLD_SCHEMA = "reactflow_delta.model_rescue_v8_corrected_expert_rebuild.v1"
V8_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v8_corrected_expert_prediction.v1"
TIC2A_MERGED_SCHEMA = "reactflow_delta.target_identity_corrected_baseline_merged.v1"
TIC2A_PREDICTION_SCHEMA = (
    "reactflow_delta.target_identity_corrected_baseline_prediction.v1"
)


def assert_run_authority(repo_root: Path, phase: str) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(f"V9 runner is closed outside active {phase}")
    if active.get("runnable_phases") != [phase]:
        raise RuntimeError(f"{phase} must be the only runnable phase")
    required = {
        "V9M1": "V9_ZERO_MEAN_RESIDUAL_SMOKE_ONLY",
        "V9M2": "V9_ZERO_MEAN_RESIDUAL_TWENTY_FOLD_SCREEN_ONLY",
    }[phase]
    if active.get("training_allowed") != required:
        raise RuntimeError(f"{phase} training authority is absent")
    if active.get("held_score_read_allowed") is not False:
        raise RuntimeError(f"{phase} requires held score access closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError(f"{phase} requires partial scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError(f"{phase} requires external outcomes locked")


def _ridge_model(value: dict[str, Any]) -> dict[str, np.ndarray | float]:
    return {
        "mean_x": np.asarray(value["mean_x"], dtype=np.float64),
        "scale_x": np.asarray(value["scale_x"], dtype=np.float64),
        "mean_y": np.asarray(value["mean_y"], dtype=np.float64),
        "coefficient": np.asarray(value["coefficient"], dtype=np.float64),
        "alpha": float(value["alpha"]),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _fold_inputs(
    fold_id: int, v8_dir: Path, tic2a_merged: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray | float]]:
    v8_path = v8_dir / f"v8_corrected_expert_fold_result_fold{fold_id}_seed0.json"
    v8_row = _read_json(v8_path)
    if v8_row.get("schema_version") != V8_FOLD_SCHEMA:
        raise ValueError(f"fold {fold_id} lacks the V8 corrected schema")
    if v8_row.get("target_profile_identity") != "EXACT_PUZZLE_METHOD_MUTATION":
        raise ValueError(f"fold {fold_id} lacks exact V8 target identity")
    if v8_row.get("legacy_v3_checkpoint_reused") is not False:
        raise ValueError(f"fold {fold_id} reused a legacy V3 checkpoint")

    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA:
        raise ValueError("V9 requires the corrected TIC2A merged schema")
    rows = {
        int(row["outer_fold"]): row for row in tic2a_merged.get("folds", [])
    }
    if sorted(rows) != list(range(20)):
        raise ValueError("V9 requires complete TIC2A folds 0 through 19")
    tic2a_row = rows[fold_id]
    model_artifact = _read_json(Path(tic2a_row["model_artifact"]))
    feature_names = tuple(model_artifact.get("feature41_feature_names", []))
    if feature_names != CANDIDATE_PROBE_FEATURE_NAMES:
        raise ValueError(f"fold {fold_id} feature41 basis identity changed")
    return v8_row, tic2a_row, _ridge_model(model_artifact["v6_feature41"])


def _load_mean_model(checkpoint: Path, device: str) -> MeanAlignedModel:
    model = MeanAlignedModel().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    freeze_mean_model(model)
    return model


def _calibration_cells(
    univ: M2Universe,
    records: list[Any],
    feature41_model: dict[str, np.ndarray | float],
    mean_model: MeanAlignedModel,
    ctx_cache: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
) -> list[dict[str, np.ndarray | int | str]]:
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
            mean = mean_model.forward_mean(
                hidden,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            ).cpu().numpy()
            x_rows = []
            feature41_mean = []
            meanaligned_mean = []
            target_delta = []
            mutant_index = []
            valid_mutants = 0
            for row, record in enumerate(recs):
                receiver = np.flatnonzero(qualified[row])
                if not len(receiver):
                    continue
                _feature30, feature41 = prediction_features(
                    construct,
                    record,
                    receiver,
                    unconstrained,
                    constrained,
                )
                prediction = predict_weighted_ridge(feature41_model, feature41)
                signed = (
                    target[row, receiver].astype(np.float64)
                    - construct.wt_reactivity[receiver].astype(np.float64)
                )
                x_rows.append(feature41.astype(np.float32))
                feature41_mean.append(prediction[:, 0].astype(np.float32))
                meanaligned_mean.append(mean[row, receiver].astype(np.float32))
                target_delta.append(signed.astype(np.float32))
                mutant_index.append(
                    np.full(len(receiver), valid_mutants, dtype=np.int64)
                )
                valid_mutants += 1
            if valid_mutants:
                cells.append(
                    {
                        "construct_id": construct_id,
                        "feature41": np.concatenate(x_rows, axis=0),
                        "feature41_mean": np.concatenate(feature41_mean),
                        "meanaligned_mean": np.concatenate(meanaligned_mean),
                        "target_delta": np.concatenate(target_delta),
                        "mutant_index": np.concatenate(mutant_index),
                        "n_mutants": valid_mutants,
                    }
                )
    if not cells:
        raise RuntimeError("V9 calibration produced no train cells")
    return cells


def _fit_head(
    head: EquiCalibratedZeroMeanMixture,
    cells: list[dict[str, Any]],
    mean_field: str,
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
            feature41 = torch.tensor(cell["feature41"], device=device)
            mean = torch.tensor(cell[mean_field], device=device)
            target = torch.tensor(cell["target_delta"], device=device)
            mutant_index = torch.tensor(cell["mutant_index"], device=device)
            weights, locations, scales = head(mean, feature41)
            assert_zero_mean_distribution(mean, weights, locations)
            loss = mutant_balanced_crps(
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
                    raise RuntimeError(f"nonfinite V9 gradient in {name}")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    if len(history) != epochs or not np.isfinite(history).all():
        raise RuntimeError("V9 calibration history is incomplete or nonfinite")
    return history


def _load_reference_prediction(
    path: Path, schema: str, mean_field: str, fold_id: int
) -> dict[str, float]:
    with np.load(path, allow_pickle=True) as handle:
        if str(handle["schema_version"].item()) != schema:
            raise ValueError(f"invalid reference prediction schema in {path}")
        if set(map(int, handle["outer_fold"])) != {fold_id}:
            raise ValueError(f"reference prediction fold mismatch in {path}")
        keys = list(map(str, handle["keys"]))
        values = np.asarray(handle[mean_field], dtype=np.float64)
        if len(keys) != len(set(keys)) or values.shape != (len(keys),):
            raise ValueError(f"reference prediction keys or mean are invalid in {path}")
        return {key: float(value) for key, value in zip(keys, values)}


def _feature41_replay_max_difference(
    univ: M2Universe,
    held_records: list[Any],
    feature41_model: dict[str, np.ndarray | float],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    reference_path: Path,
    fold_id: int,
) -> float:
    reference = _load_reference_prediction(
        reference_path,
        TIC2A_PREDICTION_SCHEMA,
        "v6_feature41_signed_delta",
        fold_id,
    )
    predicted: dict[str, float] = {}
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        receiver = np.arange(len(construct.sequence), dtype=np.int64)
        _feature30, feature41 = prediction_features(
            construct, record, receiver, unconstrained, constrained
        )
        values = predict_weighted_ridge(feature41_model, feature41)[:, 0]
        for position, value in zip(receiver, values):
            key = _bio_key(univ, record, int(position))
            if key in predicted:
                raise RuntimeError("V9 feature41 replay produced a duplicate key")
            predicted[key] = float(value)
    if set(predicted) != set(reference):
        raise RuntimeError("V9 feature41 replay key universe differs from TIC2A")
    return float(max(abs(predicted[key] - reference[key]) for key in reference))


def _held_prediction(
    univ: M2Universe,
    held_records: list[Any],
    feature41_model: dict[str, np.ndarray | float],
    mean_model: MeanAlignedModel,
    baseline_head: EquiCalibratedZeroMeanMixture,
    candidate_head: EquiCalibratedZeroMeanMixture,
    ctx_cache: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    device: str,
    fold_id: int,
    v8_prediction_path: Path,
    tic2a_prediction_path: Path,
) -> dict[str, np.ndarray]:
    by_construct: dict[str, list[Any]] = {}
    for record in held_records:
        by_construct.setdefault(record.construct_id, []).append(record)
    keys = []
    baseline_mean_rows = []
    candidate_mean_rows = []
    baseline_weights = []
    baseline_locations = []
    baseline_scales = []
    baseline_abs = []
    candidate_weights = []
    candidate_locations = []
    candidate_scales = []
    candidate_abs = []
    mean_model.eval()
    baseline_head.eval()
    candidate_head.eval()
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
            candidate_mean_matrix = mean_model.forward_mean(
                hidden,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            for row, record in enumerate(recs):
                _feature30, feature41_np = prediction_features(
                    construct,
                    record,
                    receiver,
                    unconstrained,
                    constrained,
                )
                baseline_mean_np = predict_weighted_ridge(
                    feature41_model, feature41_np
                )[:, 0].astype(np.float32)
                feature41 = torch.tensor(feature41_np, device=device)
                baseline_mean = torch.tensor(baseline_mean_np, device=device)
                candidate_mean = candidate_mean_matrix[row]
                bw, bl, bs = baseline_head(baseline_mean, feature41)
                cw, cl, cs = candidate_head(candidate_mean, feature41)
                assert_zero_mean_distribution(baseline_mean, bw, bl)
                assert_zero_mean_distribution(candidate_mean, cw, cl)
                ba = expected_absolute_delta(bw, bl, bs)
                ca = expected_absolute_delta(cw, cl, cs)
                for position in range(length):
                    keys.append(_bio_key(univ, record, position))
                baseline_mean_rows.append(baseline_mean.cpu().numpy())
                candidate_mean_rows.append(candidate_mean.cpu().numpy())
                baseline_weights.append(bw.cpu().numpy())
                baseline_locations.append(bl.cpu().numpy())
                baseline_scales.append(bs.cpu().numpy())
                baseline_abs.append(ba.cpu().numpy())
                candidate_weights.append(cw.cpu().numpy())
                candidate_locations.append(cl.cpu().numpy())
                candidate_scales.append(cs.cpu().numpy())
                candidate_abs.append(ca.cpu().numpy())

    baseline_mean_array = np.concatenate(baseline_mean_rows)
    candidate_mean_array = np.concatenate(candidate_mean_rows)
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
    key_array = np.asarray(keys, dtype=object)
    if set(keys) != set(v8_reference) or set(keys) != set(tic_reference):
        raise RuntimeError("V9 held biological key universe does not replay V8/TIC2A")
    v8_mean = np.asarray([v8_reference[key] for key in keys])
    tic_mean = np.asarray([tic_reference[key] for key in keys])
    if not np.allclose(candidate_mean_array, v8_mean, atol=1e-7, rtol=0.0):
        raise RuntimeError("V9 candidate signed mean does not replay V8")
    if not np.allclose(baseline_mean_array, tic_mean, atol=1e-10, rtol=0.0):
        maximum = float(np.max(np.abs(baseline_mean_array - tic_mean)))
        raise RuntimeError(
            f"V9 feature41 signed mean does not replay TIC2A: max={maximum:.17g}"
        )
    return {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": key_array,
        "biological_scoring_key": key_array.copy(),
        "outer_fold": np.full(len(keys), fold_id, dtype=np.int64),
        "seed": np.zeros(len(keys), dtype=np.int64),
        "registered_status": np.full(len(keys), "covered", dtype=object),
        "feature41_delta_mean": baseline_mean_array.astype(np.float64),
        "feature41_weights": np.concatenate(baseline_weights).astype(np.float64),
        "feature41_locations": np.concatenate(baseline_locations).astype(np.float64),
        "feature41_scales": np.concatenate(baseline_scales).astype(np.float64),
        "feature41_expected_absolute_delta": np.concatenate(baseline_abs).astype(
            np.float64
        ),
        "meanaligned_delta_mean": candidate_mean_array.astype(np.float64),
        "meanaligned_weights": np.concatenate(candidate_weights).astype(np.float64),
        "meanaligned_locations": np.concatenate(candidate_locations).astype(
            np.float64
        ),
        "meanaligned_scales": np.concatenate(candidate_scales).astype(np.float64),
        "meanaligned_expected_absolute_delta": np.concatenate(candidate_abs).astype(
            np.float64
        ),
    }


def run_fold(
    *,
    univ: M2Universe,
    records: list[Any],
    fold: Any,
    device: str,
    out_dir: Path,
    v8_dir: Path,
    tic2a_merged: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    epochs: int,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    v8_row, tic2a_row, feature41_model = _fold_inputs(
        fold_id, v8_dir, tic2a_merged
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
    feature41_replay_max = _feature41_replay_max_difference(
        univ,
        held_records,
        feature41_model,
        unconstrained,
        constrained,
        Path(tic2a_row["prediction_artifact"]),
        fold_id,
    )
    if feature41_replay_max > 1e-10:
        raise RuntimeError(
            "V9 feature41 pre-training replay exceeds 1e-10: "
            f"max={feature41_replay_max:.17g}"
        )
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
    torch.manual_seed(seed)
    baseline_head = EquiCalibratedZeroMeanMixture().to(device)
    torch.manual_seed(seed)
    candidate_head = EquiCalibratedZeroMeanMixture().to(device)
    baseline_history = _fit_head(
        baseline_head, cells, "feature41_mean", device, epochs, seed
    )
    candidate_history = _fit_head(
        candidate_head, cells, "meanaligned_mean", device, epochs, seed
    )
    baseline_checkpoint = out_dir / f"v9_feature41_residual_fold{fold_id}_seed0.pt"
    candidate_checkpoint = out_dir / f"v9_meanaligned_residual_fold{fold_id}_seed0.pt"
    torch.save(baseline_head.state_dict(), baseline_checkpoint)
    torch.save(candidate_head.state_dict(), candidate_checkpoint)
    prediction = _held_prediction(
        univ,
        held_records,
        feature41_model,
        mean_model,
        baseline_head,
        candidate_head,
        ctx_cache,
        unconstrained,
        constrained,
        device,
        fold_id,
        Path(v8_row["expert_prediction_artifact"]),
        Path(tic2a_row["prediction_artifact"]),
    )
    prediction_path = out_dir / f"v9_predictions_fold{fold_id}_seed0.npz"
    np.savez_compressed(prediction_path, **prediction)
    return {
        "schema_version": FOLD_SCHEMA,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "V9M1"
            else "DEVELOPMENT_CONSUMED_PREDICTION_ONLY_SCREEN"
        ),
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "calibration_epochs": epochs,
        "v8_mean_checkpoint": str(v8_row["meanaligned_checkpoint"]),
        "feature41_model_artifact": str(tic2a_row["model_artifact"]),
        "baseline_calibration_checkpoint": str(baseline_checkpoint),
        "candidate_calibration_checkpoint": str(candidate_checkpoint),
        "prediction_artifact": str(prediction_path),
        "baseline_calibration_history": baseline_history,
        "candidate_calibration_history": candidate_history,
        "n_train_cells": len(cells),
        "n_registered_prediction_rows": len(prediction["keys"]),
        "feature41_replay_max_abs_difference": feature41_replay_max,
        "invariants": {
            "target_profile_identity_exact": True,
            "v8_mean_replay_at_1e_7": True,
            "tic2a_feature41_replay_at_1e_10": True,
            "identical_residual_head_class_and_budget": True,
            "both_component_locations_equal_frozen_mean": True,
            "residual_changed_signed_point_mean": False,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
    }


def _parse_folds(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)) or not set(values) <= set(range(20)):
        raise ValueError("V9 folds must be unique members of 0 through 19")
    return sorted(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("V9M1", "V9M2"), required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
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
        raise ValueError("V9M1/V9M2 are frozen to seed 0")
    if args.phase == "V9M1" and (not set(folds) <= {0, 1} or args.epochs != 3):
        raise ValueError("V9M1 shards are frozen to folds 0/1 and 3 epochs")
    if args.phase == "V9M2" and args.epochs != 40:
        raise ValueError("V9M2 is frozen to 40 epochs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in folds:
        result = args.out_dir / f"v9_fold_result_fold{fold_id}_seed0.json"
        if result.exists():
            raise FileExistsError(f"refusing to overwrite V9 fold {fold_id}")

    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V9 requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = [fold for fold in split["folds"] if int(fold.outer_fold) in folds]
    if len(selected) != len(folds):
        raise ValueError("one or more requested V9 folds are absent")
    tic2a_merged = _read_json(args.tic2a_merged_json)
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
                unconstrained=unconstrained,
                constrained=constrained,
                epochs=args.epochs,
                seed=0,
                phase=args.phase,
            )
            result_path = args.out_dir / f"v9_fold_result_fold{fold_id}_seed0.json"
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
