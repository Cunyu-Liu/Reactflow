#!/usr/bin/env python3
"""Implementation-only real-fold runner for a future puzzle-set amendment.

The active V14 authority cannot run this module.  A future amendment must make
``reactflow_delta_puzzle_set_meta_context`` the active task and issue the exact
training token before ``run_real_fold`` can access outer-train outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
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
from scripts.reactflow_delta.model_rescue_v13 import (
    SECOND_PASS_EXACT,
    V13PointModel,
)
from scripts.reactflow_delta.model_rescue_v14 import V14PointModel
from scripts.reactflow_delta.model_rescue_v5_probe import EnsembleFeatureCache
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    BLOCK_DIAGONAL_NULL,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    FULL_CROSS_CONSTRUCT,
    POSITION_ALIGNED_OPERATOR,
    V14_ENCODER_PREFIXES,
    fit_puzzle_set_point_model,
    make_exact_full_model_pair,
    parameter_count,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    EXPECTED_RESIDUAL_PARAMETERS,
    fit_residual_pair,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
    assemble_puzzle_training_batches,
    predict_held_puzzle_distributions,
    validate_puzzle_coordinate_frames,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    _feature41_matrix,
    _fold_sources,
    _load_v8_mean,
    _parse_folds,
    _point_cells,
)
from scripts.reactflow_delta.run_model_rescue_v9 import _read_json
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


FOLD_SCHEMA = "reactflow_delta.puzzle_set_meta_context_fold.proposed.v6"
EXPECTED_PROJECT_TASK = "reactflow_delta_puzzle_set_meta_context"
EXPECTED_TRAINING_TOKEN = "PUZZLE_SET_META_CONTEXT_REAL_DATA_TRAINING_ONLY"
RUNNABLE_PHASES = {"P1M2", "P1M3", "P1M4"}
FROZEN_PARENT_SEED = 0


def _load_frozen_v13_parent(path: Path, *, device: str) -> V13PointModel:
    model = V13PointModel(second_pass_mode=SECOND_PASS_EXACT).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def _load_v14_parent_state(path: Path, *, device: str) -> dict[str, torch.Tensor]:
    source = V14PointModel().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    source.load_state_dict(state, strict=True)
    source.eval()
    return {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name.startswith(V14_ENCODER_PREFIXES)
    }


def _assert_parent_checkpoint_identity(
    *,
    v13_checkpoint: Path,
    v14_checkpoint: Path,
    outer_fold: int,
) -> None:
    expected = {
        v13_checkpoint.name: (
            f"v13_candidate_point_fold{int(outer_fold)}_seed{FROZEN_PARENT_SEED}.pt"
        ),
        v14_checkpoint.name: (
            f"v14_candidate_point_fold{int(outer_fold)}_seed{FROZEN_PARENT_SEED}.pt"
        ),
    }
    for observed, required in expected.items():
        if observed != required:
            raise ValueError(
                f"puzzle-set parent checkpoint identity mismatch: {observed} != {required}"
            )
    if not v13_checkpoint.is_file() or not v14_checkpoint.is_file():
        raise FileNotFoundError("puzzle-set frozen parent checkpoint is absent")


def assert_real_training_authority(repo_root: Path, phase: str) -> None:
    if phase not in RUNNABLE_PHASES:
        raise ValueError(f"unsupported puzzle-set phase: {phase}")
    active_path = repo_root / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("puzzle-set real training is not the active task")
    if active.get("authority", {}).get("current_phase") != phase or active.get(
        "runnable_phases"
    ) != [phase]:
        raise RuntimeError(f"puzzle-set runner is closed outside active {phase}")
    if active.get("training_allowed") != EXPECTED_TRAINING_TOKEN or active.get(
        "candidate_model_training_allowed"
    ) != EXPECTED_TRAINING_TOKEN:
        raise RuntimeError("puzzle-set real training token is absent")
    if active.get("held_score_read_allowed") is not False or active.get(
        "partial_fold_score_read_allowed"
    ) is not False:
        raise RuntimeError("puzzle-set training requires all held scores closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("puzzle-set training requires external outcomes locked")


def prepare_real_fold(
    *,
    univ: Any,
    records: list[Any],
    fold: Any,
    feature41_model: dict[str, Any],
    unconstrained: Any,
    constrained: Any,
    v8_model: MeanAlignedModel,
    v13_parent: V13PointModel,
    v14_point_state: dict[str, torch.Tensor],
    v13_parent_checkpoint: Path,
    v14_parent_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    """Prepare outer-train batches and held outcome-blind inputs."""

    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    coordinate_frames = validate_puzzle_coordinate_frames(
        train_records + held_records, univ
    )
    all_construct_ids = sorted(
        {record.construct_id for record in train_records + held_records}
    )
    context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in all_construct_ids
    }
    cells = _point_cells(
        univ,
        train_records,
        feature41_model,
        unconstrained,
        constrained,
        device,
    )
    v13_parent.eval()
    if any(parameter.requires_grad for parameter in v13_parent.parameters()):
        raise RuntimeError("V13 parent must be frozen before puzzle-set preparation")
    with torch.no_grad():
        for cell in cells:
            context = context_cache[str(cell["construct_id"])]
            cell["parent_point"] = v13_parent.forward_point(
                context,
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
                cell["feature41_point"],
            ).detach()
    training_batches = assemble_puzzle_training_batches(
        train_records, cells, context_cache
    )
    freeze_mean_model(v8_model)
    with torch.no_grad():
        for batch in training_batches:
            for cell in batch["cells"]:
                context = batch["contexts"][int(cell["focal_construct_index"])]
                hidden = v8_model.encode(context)
                _point, direct = v8_model.forward_mean_and_features(
                    hidden,
                    cell["edit_index"],
                    cell["signed_distance"],
                    cell["refs"],
                    cell["alts"],
                    cell["prediction_mask"],
                )
                cell["direct_features"] = direct.detach().cpu().numpy().astype(
                    np.float32
                )
    by_construct: dict[str, list[Any]] = defaultdict(list)
    for record in held_records:
        by_construct[str(record.construct_id)].append(record)
    held_contexts = {
        construct_id: context_cache[construct_id]
        for construct_id in sorted(by_construct)
    }
    held_feature41 = {}
    held_parent_point = {}
    held_feature41_basis = {}
    held_direct_features = {}
    for construct_id, construct_records in sorted(by_construct.items()):
        construct_records.sort(
            key=lambda record: (
                int(record.design_pos),
                str(record.ref),
                str(record.alt),
            )
        )
        construct = univ.get_construct(construct_id)
        basis, matrix = _feature41_matrix(
            construct,
            construct_records,
            feature41_model,
            unconstrained,
            constrained,
        )
        held_feature41[construct_id] = matrix
        held_feature41_basis[construct_id] = basis
        length = len(construct.sequence)
        edit = torch.tensor(
            [int(record.full_pos) for record in construct_records], device=device
        )
        distance = (
            torch.arange(length, device=device)[None, :] - edit[:, None]
        ).float()
        prediction_mask = torch.tensor(
            np.tile(
                np.asarray(construct.wt_observed, dtype=bool),
                (len(construct_records), 1),
            ),
            device=device,
        )
        with torch.no_grad():
            hidden = v8_model.encode(held_contexts[construct_id])
            _point, direct = v8_model.forward_mean_and_features(
                hidden,
                edit,
                distance,
                [str(record.ref) for record in construct_records],
                [str(record.alt) for record in construct_records],
                prediction_mask,
            )
            parent = v13_parent.forward_point(
                held_contexts[construct_id],
                edit,
                distance,
                [str(record.ref) for record in construct_records],
                [str(record.alt) for record in construct_records],
                prediction_mask,
                torch.tensor(matrix, device=device),
            )
        held_direct_features[construct_id] = (
            direct.detach().cpu().numpy().astype(np.float32)
        )
        held_parent_point[construct_id] = (
            parent.detach().cpu().numpy().astype(np.float32)
        )
    return {
        "training_batches": training_batches,
        "held_records": held_records,
        "held_contexts": held_contexts,
        "held_feature41": held_feature41,
        "held_parent_point": held_parent_point,
        "held_feature41_basis": held_feature41_basis,
        "held_direct_features": held_direct_features,
        "v14_point_state": v14_point_state,
        "frozen_parent_checkpoints": {
            "v13_point": str(v13_parent_checkpoint),
            "v14_encoder": str(v14_parent_checkpoint),
        },
        "coordinate_frames": coordinate_frames,
    }


def _initial_parent_replay_max_difference(
    model: Any,
    puzzle_batches: list[dict[str, Any]],
) -> float:
    model.eval()
    maximum = 0.0
    with torch.no_grad():
        for batch in puzzle_batches:
            contexts = batch["contexts"]
            hidden = model.encode_puzzle_set(contexts)
            observed = [context[3].bool() for context in contexts]
            reactivity = [context[1] for context in contexts]
            mixed = model.meta_context.mix_construct_tokens(
                hidden, observed, reactivity
            )
            for cell in batch["cells"]:
                point, _residual = model.forward_from_encoded(
                    hidden,
                    mixed,
                    int(cell["focal_construct_index"]),
                    cell["edit_index"],
                    cell["signed_distance"],
                    cell["refs"],
                    cell["alts"],
                    cell["feature41_point"],
                    cell["parent_point"],
                    cell["prediction_mask"],
                )
                expected = cell["parent_point"].masked_fill(
                    ~cell["prediction_mask"], 0.0
                )
                maximum = max(
                    maximum,
                    float(torch.max(torch.abs(point - expected)).detach().cpu()),
                )
    return maximum


def run_prepared_fold(
    *,
    univ: Any,
    prepared: dict[str, Any],
    outer_fold: int,
    held_puzzle: str,
    phase: str,
    seed: int,
    point_epochs: int,
    calibration_epochs: int,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Fit exact matched arms and emit one target-free fold artifact."""

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / f"puzzle_set_fold_result_fold{outer_fold}_seed{seed}.json"
    prediction_path = out_dir / (
        f"puzzle_set_predictions_fold{outer_fold}_seed{seed}.npz"
    )
    candidate_point_checkpoint = out_dir / (
        f"puzzle_set_candidate_point_fold{outer_fold}_seed{seed}.pt"
    )
    null_point_checkpoint = out_dir / (
        f"puzzle_set_null_point_fold{outer_fold}_seed{seed}.pt"
    )
    candidate_residual_checkpoint = out_dir / (
        f"puzzle_set_candidate_residual_fold{outer_fold}_seed{seed}.pt"
    )
    null_residual_checkpoint = out_dir / (
        f"puzzle_set_null_residual_fold{outer_fold}_seed{seed}.pt"
    )
    existing = [
        path
        for path in (
            fold_path,
            prediction_path,
            candidate_point_checkpoint,
            null_point_checkpoint,
            candidate_residual_checkpoint,
            null_residual_checkpoint,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite puzzle-set fold: {existing}")

    candidate, null = make_exact_full_model_pair(
        seed=seed,
        v14_point_state=prepared["v14_point_state"],
        device=device,
    )
    initial_replay = {
        "candidate": _initial_parent_replay_max_difference(
            candidate, prepared["training_batches"]
        ),
        "null": _initial_parent_replay_max_difference(
            null, prepared["training_batches"]
        ),
    }
    if max(initial_replay.values()) > 1e-7:
        raise RuntimeError("puzzle-set initialization does not replay V13 parent")
    point_parameter_counts = {
        "candidate": parameter_count(candidate),
        "null": parameter_count(null),
    }
    point_trainable_counts = {
        "candidate": parameter_count(candidate, trainable_only=True),
        "null": parameter_count(null, trainable_only=True),
    }
    candidate_history = fit_puzzle_set_point_model(
        candidate,
        prepared["training_batches"],
        epochs=point_epochs,
        seed=seed,
    )
    null_history = fit_puzzle_set_point_model(
        null,
        prepared["training_batches"],
        epochs=point_epochs,
        seed=seed,
    )
    torch.save(candidate.state_dict(), candidate_point_checkpoint)
    torch.save(null.state_dict(), null_point_checkpoint)
    residual = fit_residual_pair(
        prepared["training_batches"],
        candidate=candidate,
        null=null,
        epochs=calibration_epochs,
        seed=seed,
        device=device,
    )
    residual_checkpoints = {}
    for name, path in (
        ("candidate", candidate_residual_checkpoint),
        ("null", null_residual_checkpoint),
    ):
        torch.save(
            {
                "state_dict": residual["heads"][name].state_dict(),
                "standardizer_mean": residual["standardizers"][name].mean,
                "standardizer_scale": residual["standardizers"][name].scale,
                "point_name": name,
            },
            path,
        )
        residual_checkpoints[name] = str(path)
    prediction = predict_held_puzzle_distributions(
        univ=univ,
        held_records=prepared["held_records"],
        context_cache=prepared["held_contexts"],
        feature41_by_construct=prepared["held_feature41"],
        parent_point_by_construct=prepared["held_parent_point"],
        feature41_basis_by_construct=prepared["held_feature41_basis"],
        direct_features_by_construct=prepared["held_direct_features"],
        candidate=candidate,
        null=null,
        residual_heads=residual["heads"],
        standardizers=residual["standardizers"],
        outer_fold=outer_fold,
        seed=seed,
    )
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA or set(
        prediction
    ) & FORBIDDEN_PREDICTION_FIELDS:
        raise RuntimeError("puzzle-set fold prediction schema is invalid")
    np.savez_compressed(prediction_path, **prediction)
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "P1M2"
            else "POST_HOC_DEVELOPMENT_PREDICTION_ONLY"
        ),
        "outer_fold": int(outer_fold),
        "held_puzzle": str(held_puzzle),
        "seed": int(seed),
        "point_epochs": int(point_epochs),
        "calibration_epochs": int(calibration_epochs),
        "candidate_connectivity": FULL_CROSS_CONSTRUCT,
        "null_connectivity": BLOCK_DIAGONAL_NULL,
        "cross_construct_operator": POSITION_ALIGNED_OPERATOR,
        "candidate_parameter_count": point_parameter_counts["candidate"],
        "null_parameter_count": point_parameter_counts["null"],
        "candidate_trainable_parameter_count": point_trainable_counts["candidate"],
        "null_trainable_parameter_count": point_trainable_counts["null"],
        "frozen_parent_seed": FROZEN_PARENT_SEED,
        "frozen_parent_checkpoints": prepared["frozen_parent_checkpoints"],
        "initial_parent_replay_max_abs_difference": initial_replay,
        "n_validated_puzzle_coordinate_frames": len(
            prepared.get("coordinate_frames", {})
        ),
        "training_histories": {
            "candidate_point": candidate_history,
            "null_point": null_history,
            "candidate_residual": residual["histories"]["candidate"],
            "null_residual": residual["histories"]["null"],
        },
        "point_checkpoints": {
            "candidate": str(candidate_point_checkpoint),
            "null": str(null_point_checkpoint),
        },
        "residual_checkpoints": residual_checkpoints,
        "prediction_artifact": str(prediction_path),
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "n_calibration_cells": int(residual["n_calibration_cells"]),
        "n_outer_train_puzzles": len(prepared["training_batches"]),
        "point_optimizer_steps_each": (
            int(point_epochs) * len(prepared["training_batches"])
        ),
        "residual_optimizer_steps_each": (
            int(calibration_epochs) * len(prepared["training_batches"])
        ),
        "residual_parameter_counts": {
            "candidate": EXPECTED_RESIDUAL_PARAMETERS,
            "null": EXPECTED_RESIDUAL_PARAMETERS,
        },
        "invariants": {
            "outcome_blind_puzzle_set_inputs": True,
            "exact_parameter_and_initialization_match": True,
            "candidate_full_cross_construct_attention": True,
            "null_block_diagonal_attention": True,
            "puzzle_balanced_training": True,
            "position_aligned_cross_construct_attention": True,
            "leave_one_construct_alignment_statistics": True,
            "matched_null_self_only_alignment_statistics": True,
            "puzzle_coordinate_frames_validated": True,
            "frozen_v13_point_parent": True,
            "frozen_v14_context_encoder": True,
            "zero_initialized_parent_replay_at_1e_7": True,
            "point_frozen_during_calibration": True,
            "v10_residual_family_reused": True,
            "puzzle_balanced_residual_calibration": True,
            "median_constraint_all_held_rows": True,
            "prediction_target_free": True,
            "held_score_computed": False,
            "external_outcome_accessed": False,
        },
    }
    fold_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_real_fold(
    *,
    repo_root: Path,
    univ: Any,
    records: list[Any],
    fold: Any,
    feature41_model: dict[str, Any],
    unconstrained: Any,
    constrained: Any,
    v8_model: MeanAlignedModel,
    v13_parent_checkpoint: Path,
    v14_parent_checkpoint: Path,
    phase: str,
    seed: int,
    point_epochs: int,
    calibration_epochs: int,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    assert_real_training_authority(repo_root, phase)
    _assert_parent_checkpoint_identity(
        v13_checkpoint=v13_parent_checkpoint,
        v14_checkpoint=v14_parent_checkpoint,
        outer_fold=int(fold.outer_fold),
    )
    v13_parent = _load_frozen_v13_parent(
        v13_parent_checkpoint, device=device
    )
    v14_point_state = _load_v14_parent_state(
        v14_parent_checkpoint, device=device
    )
    prepared = prepare_real_fold(
        univ=univ,
        records=records,
        fold=fold,
        feature41_model=feature41_model,
        unconstrained=unconstrained,
        constrained=constrained,
        v8_model=v8_model,
        v13_parent=v13_parent,
        v14_point_state=v14_point_state,
        v13_parent_checkpoint=v13_parent_checkpoint,
        v14_parent_checkpoint=v14_parent_checkpoint,
        device=device,
    )
    return run_prepared_fold(
        univ=univ,
        prepared=prepared,
        outer_fold=int(fold.outer_fold),
        held_puzzle=str(fold.held_puzzle),
        phase=phase,
        seed=seed,
        point_epochs=point_epochs,
        calibration_epochs=calibration_epochs,
        device=device,
        out_dir=out_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=sorted(RUNNABLE_PHASES), required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--v10-dir", type=Path, required=True)
    parser.add_argument("--v13-dir", type=Path, required=True)
    parser.add_argument("--v14-dir", type=Path, required=True)
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
    assert_real_training_authority(repo_root, args.phase)
    folds = _parse_folds(args.folds)
    schedule = (args.point_epochs, args.calibration_epochs)
    if args.phase == "P1M2":
        if args.seed != 0 or not set(folds) <= {0, 1} or schedule != (3, 3):
            raise ValueError("P1M2 is frozen to seed0 folds0/1 and 3+3 epochs")
    elif args.phase == "P1M3":
        if args.seed != 0 or schedule != (40, 40):
            raise ValueError("P1M3 is frozen to seed0 and 40+40 epochs")
    elif args.seed not in range(5) or schedule != (40, 40):
        raise ValueError("P1M4 is frozen to seeds0-4 and 40+40 epochs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in folds:
        result_path = args.out_dir / (
            f"puzzle_set_fold_result_fold{fold_id}_seed{args.seed}.json"
        )
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite puzzle-set fold {fold_id}")
    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13_976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("puzzle-set runner requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = [
        fold for fold in split["folds"] if int(fold.outer_fold) in set(folds)
    ]
    if len(selected) != len(folds):
        raise ValueError("one or more requested puzzle-set folds are absent")

    tic2a_merged = _read_json(args.tic2a_merged_json)
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            v8_row, _tic_row, _v10_row, feature41_model = _fold_sources(
                fold_id,
                v8_dir=args.v8_dir,
                v10_dir=args.v10_dir,
                tic2a_merged=tic2a_merged,
            )
            v8_model = _load_v8_mean(Path(v8_row["meanaligned_checkpoint"]), device)
            print(
                f"[{args.phase}] fold={fold_id} held={fold.held_puzzle} "
                f"seed={args.seed} start",
                flush=True,
            )
            result = run_real_fold(
                repo_root=repo_root,
                univ=univ,
                records=records,
                fold=fold,
                feature41_model=feature41_model,
                unconstrained=unconstrained,
                constrained=constrained,
                v8_model=v8_model,
                v13_parent_checkpoint=(
                    args.v13_dir
                    / f"v13_candidate_point_fold{fold_id}_seed{FROZEN_PARENT_SEED}.pt"
                ),
                v14_parent_checkpoint=(
                    args.v14_dir
                    / f"v14_candidate_point_fold{fold_id}_seed{FROZEN_PARENT_SEED}.pt"
                ),
                phase=args.phase,
                seed=args.seed,
                point_epochs=args.point_epochs,
                calibration_epochs=args.calibration_epochs,
                device=device,
                out_dir=args.out_dir,
            )
            if result["candidate_parameter_count"] != EXPECTED_TOTAL_PARAMETERS or (
                result["candidate_trainable_parameter_count"]
                != EXPECTED_TRAINABLE_PARAMETERS
            ):
                raise RuntimeError("puzzle-set frozen parameter count changed")
            print(f"[{args.phase}] fold={fold_id} complete", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
