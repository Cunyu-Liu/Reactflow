#!/usr/bin/env python3
"""Run score-blind DeltaFlow two-stage folds (Stage 1 V14-identical point, Stage 2 CFM).

Stage 1 reuses the V14 point recipe verbatim (256/6/8/1024 architecture,
feature41 anchor, method-balanced signed-delta L1, V10 median-asymmetric
residual calibration) without masked-WT pretraining, per freeze table F4.
Both arms share the identical Stage-1 model; the only candidate/null
difference is the Stage-2 denoiser diagonal restriction.  Stage 2 fits the
rectified-flow candidate/diagonal-null pair on residual profiles
r = delta - stage1_point over outer-train qualified positions and exports
held joint samples (Euler, frozen steps, seeded draws) plus per-position
marginal mixtures for both arms.

DeltaFlow does not bind to the RND active contract; pre-registration
freeze lives in git commits (Task 7 pattern).  CUDA is mandatory and
validated before any fold artifact is created; fold outputs publish
atomically through a seven-file move set with the result marker last.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.deltaflow import (
    FLOW_CHECKPOINT_SCHEMA,
    PAIR_CHANNEL_NAMES,
    TEACHER_WIDTH,
    build_pair_channels,
    euler_flow_sample,
    flow_parameter_count,
    interpolate_flow_state,
    make_flow_pair,
    masked_flow_loss,
    samples_to_mixture,
)
from scripts.reactflow_delta.gpu_runtime import require_cuda_device
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v14 import (
    EXPECTED_DOWNSTREAM_PARAMETERS,
    V14PointModel,
    assert_exact_parameter_contract,
    assert_snapshot_equal,
    fit_point_model,
    freeze_point_model,
    module_snapshot,
    parameter_count,
)
from scripts.reactflow_delta.model_rescue_v2 import freeze_mean_model
from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache,
    baseline_features,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.run_model_rescue_v10 import (
    _fit_head as fit_v10_residual_head,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    _calibration_cells,
    _feature41_replay_max_difference,
    _fold_sources,
    _held_prediction,
    _load_v8_mean,
    _new_residual_heads,
    _parse_folds,
    _point_cells,
    _prepare_calibration_inputs,
    _read_json,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


FOLD_SCHEMA = "reactflow_delta.deltaflow_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.deltaflow_prediction.v1"
FLOW_TRAINING_SCHEMA = "reactflow_delta.deltaflow_flow_training.v1"
STAGE1_CHECKPOINT_SCHEMA = "reactflow_delta.deltaflow_stage1_point_checkpoint.v1"
V11_POINT_NAMES = ("feature41", "anchored", "unanchored")

FLOW_LEARNING_RATE = 2e-4
FLOW_WEIGHT_DECAY = 0.01
FLOW_BATCH_MUTANTS = 16
FLOW_GRADIENT_CLIP = 1.0
FLOW_TRAIN_SEED_OFFSET = 2_200_000
FLOW_DROPOUT_SEED_OFFSET = 2_300_000
SAMPLING_DRAWS = 16
SAMPLING_STEPS = 50
SAMPLING_MUTANT_BATCH = 32
SAMPLING_SEED_BASE = 2_400_000
EXPECTED_TOTAL_CONSTRUCTS = 160
EXPECTED_CANONICAL_MUTANTS = 13976
EXPECTED_CANONICAL_IDENTITY = "EXACT_PUZZLE_METHOD_MUTATION"
POINT_CHANNELS = 18
SPLIT_SEED = 20260813


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_mnt_artifact_dir(path: Path) -> Path:
    resolved = path.resolve()
    if not str(resolved).startswith("/mnt/"):
        raise RuntimeError(f"DELTAFLOW_MNT_REQUIRED: {resolved} is outside /mnt")
    return resolved


def _teacher_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        keys = sorted(handle.keys())
        if len(keys) != EXPECTED_TOTAL_CONSTRUCTS:
            raise RuntimeError("teacher embedding key count differs from 160")
        teacher = {key: np.asarray(handle[key]) for key in keys}
    for key, value in teacher.items():
        if value.ndim != 2 or value.shape[1] != TEACHER_WIDTH:
            raise RuntimeError(f"teacher embedding shape changed at {key}")
    return teacher


def _assert_tensor_cuda(tensor: torch.Tensor, label: str) -> None:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"DELTAFLOW_CUDA_REQUIRED: {label} is off CUDA")


def _finite_flow_gradients(model: torch.nn.Module, arm: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise RuntimeError(f"nonfinite DeltaFlow gradient in {arm}: {name}")


def _flow_cells(
    cells: list[dict[str, Any]],
    *,
    stage1_model: V14PointModel,
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    teacher: dict[str, np.ndarray],
    device: str,
) -> list[dict[str, Any]]:
    """Per-construct Stage-2 training cells on qualified positions.

    Residual r = (target - wt) - stage1_point is zeroed off the qualified
    mask; pair channels are built per mutant batch during training so the
    [B, 28, L, L] tensor never materializes for a whole construct.
    """

    stage1_model.eval()
    flow_cells: list[dict[str, Any]] = []
    with torch.no_grad():
        for cell in cells:
            construct_id = cell["construct_id"]
            length = int(cell["target"].shape[1])
            construct = cell["construct"]
            hidden = stage1_model.encode(context_cache[construct_id])
            stage1_point = stage1_model.forward_point(
                hidden,
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
                cell["feature41_point"],
            )
            qualified = cell["qualified_mask"]
            target = cell["target"].to(torch.float32)
            wt = cell["wt"].to(torch.float32)
            delta = torch.where(
                qualified, target - wt[None, :], torch.zeros((), device=device)
            )
            residual = torch.where(
                qualified, delta - stage1_point, torch.zeros((), device=device)
            )
            receiver = np.arange(length, dtype=np.int64)
            point_rows = np.stack(
                [
                    baseline_features(construct, record, receiver)
                    for record in cell["records"]
                ]
            ).astype(np.float32)
            if point_rows.shape != (len(cell["records"]), length, POINT_CHANNELS):
                raise RuntimeError("DeltaFlow point feature block has wrong shape")
            flow_cells.append(
                {
                    "construct_id": construct_id,
                    "n_mutants": int(len(cell["records"])),
                    "stage1_point": (stage1_point * qualified.float()).cpu().numpy(),
                    "residual": (residual * qualified.float()).cpu().numpy(),
                    "point": point_rows,
                    "mask": qualified.cpu().numpy(),
                    "teacher": teacher[construct_id].astype(np.float32),
                }
            )
    if not flow_cells:
        raise RuntimeError("DeltaFlow stage-2 training produced no outer-train cells")
    return flow_cells


def fit_flow_pair(
    candidate: torch.nn.Module,
    null: torch.nn.Module,
    flow_cells: list[dict[str, Any]],
    *,
    epochs: int,
    seed: int,
    device: str,
) -> dict[str, list[float]]:
    """Train the matched candidate/null flow pair with identical streams.

    Both arms see the same batch order, the same time/noise draws (per-arm
    generator reseeded identically), and the same dropout stream (global
    RNG reset per arm), so the only difference is the diagonal operator
    restriction.
    """

    batches: list[tuple[int, int, int]] = []
    for cell_index, cell in enumerate(flow_cells):
        for start in range(0, cell["n_mutants"], FLOW_BATCH_MUTANTS):
            stop = min(start + FLOW_BATCH_MUTANTS, cell["n_mutants"])
            batches.append((cell_index, start, stop))
    if not batches:
        raise RuntimeError("DeltaFlow flow training received no mutant batches")
    histories = {"candidate": [], "null": []}
    models = {"candidate": candidate, "null": null}
    for arm, model in models.items():
        torch.manual_seed(seed + FLOW_DROPOUT_SEED_OFFSET)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + FLOW_TRAIN_SEED_OFFSET)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=FLOW_LEARNING_RATE, weight_decay=FLOW_WEIGHT_DECAY
        )
        for epoch in range(epochs):
            order = list(range(len(batches)))
            random.Random(seed * 100_019 + epoch * 977 + 13).shuffle(order)
            losses = []
            for batch_index in order:
                cell_index, start, stop = batches[batch_index]
                cell = flow_cells[cell_index]
                mask = torch.tensor(cell["mask"][start:stop], device=device)
                point = torch.tensor(cell["point"][start:stop], device=device)
                pair = build_pair_channels(point, mask)
                teacher = torch.as_tensor(cell["teacher"], device=device)[
                    None, :, :
                ].expand(point.shape[0], -1, -1)
                stage1 = torch.tensor(
                    cell["stage1_point"][start:stop], device=device
                )
                residual = torch.tensor(
                    cell["residual"][start:stop], device=device
                )
                n_mutants = point.shape[0]
                t = torch.rand(n_mutants, device=device, generator=generator)
                noise = torch.randn(
                    n_mutants, point.shape[1], device=device, generator=generator
                )
                state = interpolate_flow_state(noise, residual, t, mask)
                velocity = model(state, point, pair, teacher, stage1, mask, t)
                loss = masked_flow_loss(velocity, residual, noise, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _finite_flow_gradients(model, arm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), FLOW_GRADIENT_CLIP)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            histories[arm].append(float(np.mean(losses)))
        if len(histories[arm]) != epochs or not np.isfinite(histories[arm]).all():
            raise RuntimeError(f"DeltaFlow {arm} flow history is incomplete or nonfinite")
    return histories


def _flow_sample_arm(
    model: torch.nn.Module,
    univ: M2Universe,
    held_records: list[Any],
    *,
    prediction: dict[str, np.ndarray],
    teacher: dict[str, np.ndarray],
    device: str,
    seed: int,
    fold_id: int,
) -> dict[str, Any]:
    """Euler-sample one arm on the held puzzle and build final marginals.

    ``prediction`` is the renamed V11-style export whose candidate point
    rows are ordered construct (sorted) -> mutant -> receiver; the same
    blocks feed the Stage-1 conditioning input so flow and point exports
    agree exactly.  Final samples = stage1_point + flow residual samples;
    the marginal mixture statistics therefore describe the full delta.
    """

    model.eval()
    by_construct: dict[str, list[Any]] = {}
    for record in held_records:
        by_construct.setdefault(record.construct_id, []).append(record)
    construct_order = sorted(by_construct)
    row_offset = 0
    construct_ids: list[str] = []
    samples_per_construct: list[np.ndarray] = []
    keys_flat: list[str] = []
    weights_rows: list[np.ndarray] = []
    location_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    expected_rows: list[np.ndarray] = []
    generator = torch.Generator(device=device)
    with torch.no_grad():
        for construct_index, construct_id in enumerate(construct_order):
            records = by_construct[construct_id]
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            n_mutants = len(records)
            stage1_block = prediction["candidate_point"][
                row_offset : row_offset + n_mutants * length
            ].reshape(n_mutants, length).astype(np.float32)
            row_offset += n_mutants * length
            mask = np.tile(construct.wt_observed.astype(bool), (n_mutants, 1))
            receiver = np.arange(length, dtype=np.int64)
            point = np.stack(
                [baseline_features(construct, record, receiver) for record in records]
            ).astype(np.float32)
            if point.shape != (n_mutants, length, POINT_CHANNELS):
                raise RuntimeError("DeltaFlow held point feature block has wrong shape")
            teacher_matrix = torch.as_tensor(teacher[construct_id], device=device)
            construct_samples = np.zeros(
                (SAMPLING_DRAWS, n_mutants, length), dtype=np.float32
            )
            for start in range(0, n_mutants, SAMPLING_MUTANT_BATCH):
                stop = min(start + SAMPLING_MUTANT_BATCH, n_mutants)
                batch_mask = torch.tensor(mask[start:stop], device=device)
                batch_point = torch.tensor(point[start:stop], device=device)
                batch_pair = build_pair_channels(batch_point, batch_mask)
                batch_teacher = teacher_matrix[None, :, :].expand(
                    stop - start, -1, -1
                )
                batch_stage1 = torch.tensor(stage1_block[start:stop], device=device)
                draws = []
                for draw in range(SAMPLING_DRAWS):
                    generator.manual_seed(
                        SAMPLING_SEED_BASE
                        + seed * 1_000_003
                        + fold_id * 7_919
                        + construct_index * 101
                        + draw * 7
                        + 3
                    )
                    draws.append(
                        euler_flow_sample(
                            model,
                            point_in=batch_point,
                            pair_in=batch_pair,
                            teacher=batch_teacher,
                            stage1=batch_stage1,
                            mask=batch_mask,
                            steps=SAMPLING_STEPS,
                            generator=generator,
                        )
                    )
                samples = torch.stack(draws, dim=0)
                final = samples + batch_stage1.to(samples.dtype)[None, :, :]
                construct_samples[:, start:stop, :] = (
                    (final * batch_mask.float()[None, :, :]).cpu().numpy()
                )
            samples_tensor = torch.tensor(construct_samples, device=device)
            mask_tensor = torch.tensor(mask, device=device)
            weights, locations, scales, expected = samples_to_mixture(
                samples_tensor, mask_tensor
            )
            construct_keys = [
                _bio_key(univ, record, receiver)
                for record in records
                for receiver in range(length)
            ]
            flat_mask = mask.reshape(-1)
            selected_keys = [
                key for key, keep in zip(construct_keys, flat_mask) if keep
            ]
            construct_ids.append(construct_id)
            samples_per_construct.append(construct_samples)
            keys_flat.extend(selected_keys)
            weights_rows.append(weights.cpu().numpy())
            location_rows.append(locations.cpu().numpy())
            scale_rows.append(scales.cpu().numpy())
            expected_rows.append(expected.cpu().numpy())
    if row_offset != len(prediction["keys"]):
        raise RuntimeError("DeltaFlow flow sampling misaligned with point export rows")
    return {
        "construct_ids": construct_ids,
        "samples": samples_per_construct,
        "keys": keys_flat,
        "weights": np.concatenate(weights_rows),
        "locations": np.concatenate(location_rows),
        "scales": np.concatenate(scale_rows),
        "expected_absolute_delta": np.concatenate(expected_rows),
    }


def _rename_v11_prediction(
    v11_prediction: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Rename the reused V11 export into the DeltaFlow schema."""

    output: dict[str, np.ndarray] = {}
    for name, value in v11_prediction.items():
        if name == "schema_version":
            output[name] = np.asarray(PREDICTION_SCHEMA)
        elif name.startswith("anchored_"):
            output[f"candidate_{name.removeprefix('anchored_')}"] = value
        elif name.startswith("unanchored_"):
            output[f"null_{name.removeprefix('unanchored_')}"] = value
        else:
            output[name] = value
    forbidden = ("anchored_", "unanchored_")
    if any(name.startswith(forbidden) for name in output):
        raise RuntimeError("DeltaFlow prediction retained a V11 candidate name")
    return output


def _artifact_paths(out_dir: Path, fold: int, seed: int) -> dict[str, Path]:
    stem = f"fold{fold}_seed{seed}"
    return {
        "result": out_dir / f"deltaflow_fold_result_{stem}.json",
        "prediction": out_dir / f"deltaflow_predictions_{stem}.npz",
        "stage1_point": out_dir / f"deltaflow_stage1_point_{stem}.pt",
        "feature41_residual": out_dir / f"deltaflow_feature41_asymmetric_{stem}.pt",
        "candidate_flow": out_dir / f"deltaflow_candidate_flow_{stem}.pt",
        "null_flow": out_dir / f"deltaflow_null_flow_{stem}.pt",
        "flow_audit": out_dir / f"deltaflow_flow_audit_{stem}.json",
    }


DFLOW_EPOCH_SCHEDULES = {
    "DFLOW1": (3, 3, 3),
    "DFLOW2": (40, 40, 100),
    "DFLOW3": (40, 40, 100),
}


def _joint_samples_dir(prediction_path: Path) -> Path:
    return prediction_path.with_name(prediction_path.stem + "_joint_samples")


def _refuse_fold_overwrite(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite DeltaFlow fold artifacts: {existing}")
    joint_dir = _joint_samples_dir(paths["prediction"])
    if joint_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite DeltaFlow joint samples directory: {joint_dir}"
        )


def _canonicalize_fold_result_paths(
    result: dict[str, Any], canonical_paths: dict[str, Path]
) -> dict[str, Any]:
    output = copy.deepcopy(result)
    output["stage1_point_checkpoint"] = str(canonical_paths["stage1_point"].resolve())
    output["residual_checkpoints"] = {
        "feature41": str(canonical_paths["feature41_residual"].resolve()),
    }
    output["flow_checkpoints"] = {
        "candidate": str(canonical_paths["candidate_flow"].resolve()),
        "null": str(canonical_paths["null_flow"].resolve()),
    }
    output["prediction_artifact"] = str(canonical_paths["prediction"].resolve())
    output["flow_audit_artifact"] = str(canonical_paths["flow_audit"].resolve())
    return output


def _publish_fold_artifacts(
    staging_paths: dict[str, Path], canonical_paths: dict[str, Path]
) -> None:
    """Publish a complete seven-file fold, with the result marker last.

    The joint-samples directory (a sidecar of the prediction npz) moves
    together with the prediction file and is rolled back in the same
    sequence on failure.
    """

    if set(staging_paths) != set(canonical_paths):
        raise RuntimeError("staging/canonical fold artifact universes differ")
    missing = [str(path) for path in staging_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"staged fold payload is incomplete: {missing}")
    staging_joint = _joint_samples_dir(staging_paths["prediction"])
    canonical_joint = _joint_samples_dir(canonical_paths["prediction"])
    if not staging_joint.is_dir():
        raise FileNotFoundError(f"staged joint samples directory is missing: {staging_joint}")
    _refuse_fold_overwrite(canonical_paths)
    publish_order = (
        "stage1_point",
        "feature41_residual",
        "candidate_flow",
        "null_flow",
        "prediction",
        "flow_audit",
        "result",
    )
    published: list[str] = []
    joint_moved = False
    try:
        for name in publish_order:
            staging_paths[name].replace(canonical_paths[name])
            published.append(name)
            if name == "prediction":
                staging_joint.replace(canonical_joint)
                joint_moved = True
    except BaseException:
        if joint_moved and canonical_joint.is_dir() and not staging_joint.exists():
            canonical_joint.replace(staging_joint)
        for name in reversed(published):
            if canonical_paths[name].is_file() and not staging_paths[name].exists():
                canonical_paths[name].replace(staging_paths[name])
        raise


def _save_flow_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    phase: str,
    arm: str,
    fold: int,
    seed: int,
    epochs: int,
    history: list[float],
) -> None:
    torch.save(
        {
            "schema_version": FLOW_CHECKPOINT_SCHEMA,
            "phase": phase,
            "arm": arm,
            "outer_fold": int(fold),
            "seed": int(seed),
            "epochs": int(epochs),
            "training_schema": FLOW_TRAINING_SCHEMA,
            "state_dict": model.state_dict(),
            "training_history": [float(value) for value in history],
            "sampling": {"steps": SAMPLING_STEPS, "draws": SAMPLING_DRAWS},
        },
        path,
    )


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
    teacher: dict[str, np.ndarray],
    point_epochs: int,
    calibration_epochs: int,
    flow_epochs: int,
    seed: int,
    phase: str,
    experiment_id: str,
    repo_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    started_at = _utc_now()
    fold_id = int(fold.outer_fold)
    paths = _artifact_paths(out_dir, fold_id, seed)
    _refuse_fold_overwrite(paths)
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
    for construct_id, context in context_cache.items():
        for tensor in context:
            _assert_tensor_cuda(tensor, label=f"construct {construct_id} context")
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
        raise RuntimeError("DeltaFlow feature41 replay exceeds 1e-7")

    cells = _point_cells(
        univ, train_records, feature41_model, unconstrained, constrained, device
    )
    for cell in cells:
        if "construct" not in cell:
            cell["construct"] = univ.get_construct(cell["construct_id"])

    torch.manual_seed(int(seed))
    stage1 = V14PointModel().to(device)
    assert_exact_parameter_contract(stage1)
    stage1_history = fit_point_model(
        stage1, cells, context_cache, epochs=point_epochs, seed=seed
    )
    print(f"[{phase}] fold={fold_id} seed={seed} stage1_point_complete", flush=True)
    if parameter_count(stage1, trainable_only=True) != EXPECTED_DOWNSTREAM_PARAMETERS:
        raise RuntimeError("DeltaFlow stage1 downstream parameter count changed")
    torch.save(
        {
            "schema_version": STAGE1_CHECKPOINT_SCHEMA,
            "phase": phase,
            "outer_fold": fold_id,
            "seed": int(seed),
            "point_epochs": int(point_epochs),
            "arch": "v14_256_6_8_1024_no_pretraining",
            "state_dict": stage1.state_dict(),
            "training_history": [float(value) for value in stage1_history],
        },
        paths["stage1_point"],
    )
    freeze_point_model(stage1)
    stage1_snapshot = module_snapshot(stage1)

    stage1_calibration_copy = copy.deepcopy(stage1)
    assert_snapshot_equal(
        stage1_snapshot, stage1_calibration_copy, "DeltaFlow stage1 calibration copy"
    )
    v8_model = _load_v8_mean(Path(v8_row["meanaligned_checkpoint"]), device)
    freeze_mean_model(v8_model)
    calibration_cells = _calibration_cells(
        cells,
        anchored=stage1,
        unanchored=stage1_calibration_copy,
        v8_model=v8_model,
        v11_context_cache=context_cache,
        v8_context_cache=context_cache,
    )
    heads = _new_residual_heads(seed, device)
    standardizers: dict[str, Any] = {}
    calibration_inputs: dict[str, list[np.ndarray]] = {}
    histories: dict[str, list[float]] = {}
    for name in V11_POINT_NAMES:
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
    assert_snapshot_equal(stage1_snapshot, stage1, "DeltaFlow stage1 point")
    if any(parameter.grad is not None for parameter in stage1.parameters()):
        raise RuntimeError("DeltaFlow calibration produced stage1 point gradients")
    print(f"[{phase}] fold={fold_id} seed={seed} calibration_complete", flush=True)
    torch.save(
        {
            "state_dict": heads["feature41"].state_dict(),
            "standardizer_mean": standardizers["feature41"].mean,
            "standardizer_scale": standardizers["feature41"].scale,
            "point_name": "feature41",
        },
        paths["feature41_residual"],
    )

    flow_cells = _flow_cells(
        cells,
        stage1_model=stage1,
        context_cache=context_cache,
        teacher=teacher,
        device=device,
    )
    candidate, null = make_flow_pair(seed=seed, device=device)
    candidate_parameters = flow_parameter_count(candidate)
    null_parameters = flow_parameter_count(null)
    if candidate_parameters != null_parameters:
        raise RuntimeError("DeltaFlow candidate/null flow parameter counts differ")
    flow_histories = fit_flow_pair(
        candidate, null, flow_cells, epochs=flow_epochs, seed=seed, device=device
    )
    print(f"[{phase}] fold={fold_id} seed={seed} flow_training_complete", flush=True)
    _save_flow_checkpoint(
        paths["candidate_flow"],
        model=candidate,
        phase=phase,
        arm="candidate",
        fold=fold_id,
        seed=seed,
        epochs=flow_epochs,
        history=flow_histories["candidate"],
    )
    _save_flow_checkpoint(
        paths["null_flow"],
        model=null,
        phase=phase,
        arm="null",
        fold=fold_id,
        seed=seed,
        epochs=flow_epochs,
        history=flow_histories["null"],
    )

    v11_prediction = _held_prediction(
        univ=univ,
        held_records=held_records,
        feature41_model=feature41_model,
        anchored=stage1,
        unanchored=stage1_calibration_copy,
        v8_model=v8_model,
        heads=heads,
        standardizers=standardizers,
        v11_context_cache=context_cache,
        v8_context_cache=context_cache,
        unconstrained=unconstrained,
        constrained=constrained,
        fold_id=fold_id,
        seed=seed,
        v8_prediction_path=Path(v8_row["expert_prediction_artifact"]),
        tic2a_prediction_path=Path(tic_row["prediction_artifact"]),
        historical_v10_path=Path(v10_row["prediction_artifact"]),
        require_v10_feature41_replay=False,
    )
    prediction = _rename_v11_prediction(v11_prediction)

    candidate_flow = _flow_sample_arm(
        candidate,
        univ,
        held_records,
        prediction=prediction,
        teacher=teacher,
        device=device,
        seed=seed,
        fold_id=fold_id,
    )
    null_flow = _flow_sample_arm(
        null,
        univ,
        held_records,
        prediction=prediction,
        teacher=teacher,
        device=device,
        seed=seed,
        fold_id=fold_id,
    )
    if candidate_flow["keys"] != null_flow["keys"]:
        raise RuntimeError("DeltaFlow candidate/null flow key alignment differs")
    if set(candidate_flow["keys"]) - set(map(str, prediction["keys"])):
        raise RuntimeError("DeltaFlow flow keys fall outside the registered universe")
    prediction["flow_keys"] = np.asarray(candidate_flow["keys"], dtype=object)
    prediction["flow_candidate_weights"] = candidate_flow["weights"].astype(np.float64)
    prediction["flow_candidate_locations"] = candidate_flow["locations"].astype(
        np.float64
    )
    prediction["flow_candidate_scales"] = candidate_flow["scales"].astype(np.float64)
    prediction["flow_candidate_expected_absolute_delta"] = candidate_flow[
        "expected_absolute_delta"
    ].astype(np.float64)
    prediction["flow_null_weights"] = null_flow["weights"].astype(np.float64)
    prediction["flow_null_locations"] = null_flow["locations"].astype(np.float64)
    prediction["flow_null_scales"] = null_flow["scales"].astype(np.float64)
    prediction["flow_null_expected_absolute_delta"] = null_flow[
        "expected_absolute_delta"
    ].astype(np.float64)
    np.savez_compressed(paths["prediction"], **prediction)
    print(f"[{phase}] fold={fold_id} seed={seed} prediction_complete", flush=True)

    n_flow_mutants = int(sum(cell["n_mutants"] for cell in flow_cells))
    flow_audit = {
        "schema_version": FLOW_TRAINING_SCHEMA,
        "phase": phase,
        "outer_fold": fold_id,
        "seed": int(seed),
        "epochs": int(flow_epochs),
        "learning_rate": FLOW_LEARNING_RATE,
        "weight_decay": FLOW_WEIGHT_DECAY,
        "batch_mutants": FLOW_BATCH_MUTANTS,
        "gradient_clip": FLOW_GRADIENT_CLIP,
        "sampling_steps": SAMPLING_STEPS,
        "sampling_draws": SAMPLING_DRAWS,
        "candidate_history": flow_histories["candidate"],
        "null_history": flow_histories["null"],
        "n_flow_train_cells": len(flow_cells),
        "n_flow_train_mutants": n_flow_mutants,
        "flow_parameter_count": candidate_parameters,
        "pair_channel_names": list(PAIR_CHANNEL_NAMES),
        "flow_keys": list(candidate_flow["keys"]),
        "construct_ids": list(candidate_flow["construct_ids"]),
        "joint_sample_shapes": [
            list(array.shape) for array in candidate_flow["samples"]
        ],
    }
    flow_precision = {
        "flow_training_dtype": str(next(candidate.parameters()).dtype),
        "euler_integration_dtype": "float32",
        "mixture_statistics_dtype": "float32",
    }
    flow_audit.update(flow_precision)
    paths["flow_audit"].write_text(
        json.dumps(flow_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joint_dir = _joint_samples_dir(paths["prediction"])
    joint_dir.mkdir(parents=True, exist_ok=False)
    for construct_id, candidate_samples, null_samples in zip(
        candidate_flow["construct_ids"],
        candidate_flow["samples"],
        null_flow["samples"],
    ):
        np.savez_compressed(
            joint_dir / f"{construct_id}.npz",
            candidate_samples=np.asarray(candidate_samples, dtype=np.float32),
            null_samples=np.asarray(null_samples, dtype=np.float32),
        )

    return {
        "schema_version": FOLD_SCHEMA,
        "experiment_id": experiment_id,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "DFLOW1"
            else "EXPOSURE_DISCLOSED_DEVELOPMENT_PREDICTION_ONLY"
        ),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "git_commit": git_commit,
        "command": list(sys.argv),
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": int(seed),
        "point_epochs": int(point_epochs),
        "calibration_epochs": int(calibration_epochs),
        "flow_epochs": int(flow_epochs),
        "training_device": device,
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "stage1_architecture": "v14_256_6_8_1024_no_pretraining",
        "stage1_point_checkpoint": str(paths["stage1_point"].resolve()),
        "v8_mean_checkpoint": str(v8_row["meanaligned_checkpoint"]),
        "feature41_model_artifact": str(tic_row["model_artifact"]),
        "historical_v10_prediction_artifact": str(v10_row["prediction_artifact"]),
        "residual_checkpoints": {
            "feature41": str(paths["feature41_residual"].resolve()),
        },
        "flow_checkpoints": {
            "candidate": str(paths["candidate_flow"].resolve()),
            "null": str(paths["null_flow"].resolve()),
        },
        "prediction_artifact": str(paths["prediction"].resolve()),
        "joint_samples_dir": str(joint_dir.resolve()),
        "flow_audit_artifact": str(paths["flow_audit"].resolve()),
        "training_histories": {
            "stage1_point": stage1_history,
            **{
                f"{ {'anchored': 'candidate', 'unanchored': 'null'}.get(name, name) }_residual": values
                for name, values in histories.items()
            },
            "candidate_flow": flow_histories["candidate"],
            "null_flow": flow_histories["null"],
        },
        "n_train_cells": len(cells),
        "n_flow_train_cells": len(flow_cells),
        "n_flow_train_mutants": n_flow_mutants,
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "feature41_replay_max_abs_difference": replay,
        "stage1_parameter_count": EXPECTED_DOWNSTREAM_PARAMETERS,
        "flow_parameter_counts": {
            "candidate": candidate_parameters,
            "null": null_parameters,
        },
        "invariants": {
            "target_profile_identity_exact": True,
            "stage1_shared_between_arms": True,
            "v14_pretraining_not_used": True,
            "same_flow_training_protocol_both_arms": True,
            "flow_diagonal_null_only_difference": True,
            "stage1_frozen_during_flow_training": True,
            "feature41_replay_at_1e_7": True,
            "median_constraint_all_held_rows": True,
            "sampling_seed_deterministic": True,
            "cuda_only_training": True,
            "held_score_computed": False,
            "partial_score_inspected": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
        "exit_code": 0,
    }


def _phase_schedule(phase: str) -> tuple[tuple[int, ...], int, int, int]:
    schedules = {
        "DFLOW1": ((0, 1), 3, 3, 3),
        "DFLOW2": (tuple(range(20)), 40, 40, 100),
        "DFLOW3": (tuple(range(20)), 40, 40, 100),
    }
    if phase not in schedules:
        raise ValueError(f"unsupported DeltaFlow phase: {phase}")
    folds, point_epochs, calibration_epochs, flow_epochs = schedules[phase]
    return folds, point_epochs, calibration_epochs, flow_epochs


def _validate_phase_request(
    *,
    phase: str,
    folds: tuple[int, ...],
    point_epochs: int,
    calibration_epochs: int,
    flow_epochs: int,
    seed: int,
) -> None:
    expected_folds, expected_point, expected_calibration, expected_flow = (
        _phase_schedule(phase)
    )
    if not set(folds) <= set(expected_folds):
        raise ValueError(f"{phase} requested folds outside the frozen universe")
    if phase == "DFLOW1":
        if seed != 0 or (point_epochs, calibration_epochs, flow_epochs) != (3, 3, 3):
            raise ValueError("DFLOW1 is frozen to seed0 folds0/1 and 3+3+3 epochs")
        return
    if phase == "DFLOW2" and seed != 0:
        raise ValueError("DFLOW2 is frozen to seed0")
    if phase == "DFLOW3" and seed not in range(5):
        raise ValueError("DFLOW3 is frozen to seeds0-4")
    if (point_epochs, calibration_epochs, flow_epochs) != (
        expected_point,
        expected_calibration,
        expected_flow,
    ):
        raise ValueError(f"{phase} epoch schedule changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("DFLOW1", "DFLOW2", "DFLOW3"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--v10-dir", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--teacher-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--point-epochs", type=int, required=True)
    parser.add_argument("--calibration-epochs", type=int, required=True)
    parser.add_argument("--flow-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    folds = _parse_folds(args.folds)
    _validate_phase_request(
        phase=args.phase,
        folds=folds,
        point_epochs=args.point_epochs,
        calibration_epochs=args.calibration_epochs,
        flow_epochs=args.flow_epochs,
        seed=args.seed,
    )
    device = require_cuda_device(args.device)
    if torch.device(device).type != "cuda":
        raise RuntimeError("CUDA_REQUIRED: DeltaFlow device resolved off CUDA")
    if args.phase == "DFLOW1":
        out_dir = args.out_dir.resolve()
        print(
            "[DFLOW1] engineering smoke: artifact dir outside /mnt allowed",
            flush=True,
        )
    else:
        out_dir = _require_mnt_artifact_dir(args.out_dir)
    for fold_id in folds:
        _refuse_fold_overwrite(_artifact_paths(out_dir, fold_id, args.seed))

    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != EXPECTED_CANONICAL_MUTANTS:
        raise RuntimeError("DeltaFlow requires the canonical mutant count 13976")
    if identity.get("canonical_mutant_full_profile_identity") != (
        EXPECTED_CANONICAL_IDENTITY
    ):
        raise RuntimeError("DeltaFlow requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=SPLIT_SEED
    )
    selected = [
        fold for fold in split["folds"] if int(fold.outer_fold) in set(folds)
    ]
    if len(selected) != len(folds):
        raise ValueError("one or more requested DeltaFlow folds are absent")
    teacher = _teacher_embeddings(args.teacher_npz)
    missing = [
        record.construct_id
        for record in records
        if record.construct_id not in teacher
    ]
    if missing:
        raise RuntimeError(f"DeltaFlow teacher embeddings miss {len(missing)} constructs")

    tic2a_merged = _read_json(args.tic2a_merged_json)
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit(repo_root)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            print(
                f"[{args.phase}] fold={fold_id} held={fold.held_puzzle} "
                f"seed={args.seed} start",
                flush=True,
            )
            canonical_paths = _artifact_paths(out_dir, fold_id, args.seed)
            with tempfile.TemporaryDirectory(
                prefix=f".deltaflow_fold{fold_id}_seed{args.seed}_",
                dir=out_dir,
            ) as staging_name:
                staging_dir = Path(staging_name)
                result = run_fold(
                    univ=univ,
                    records=records,
                    fold=fold,
                    device=device,
                    out_dir=staging_dir,
                    v8_dir=args.v8_dir,
                    v10_dir=args.v10_dir,
                    tic2a_merged=tic2a_merged,
                    unconstrained=unconstrained,
                    constrained=constrained,
                    teacher=teacher,
                    point_epochs=args.point_epochs,
                    calibration_epochs=args.calibration_epochs,
                    flow_epochs=args.flow_epochs,
                    seed=args.seed,
                    phase=args.phase,
                    experiment_id=args.experiment_id,
                    repo_root=repo_root,
                    git_commit=commit,
                )
                staging_paths = _artifact_paths(staging_dir, fold_id, args.seed)
                result = _canonicalize_fold_result_paths(result, canonical_paths)
                staging_paths["result"].write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                _publish_fold_artifacts(staging_paths, canonical_paths)
            print(f"[{args.phase}] fold={fold_id} complete", flush=True)
            torch.cuda.empty_cache()
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
