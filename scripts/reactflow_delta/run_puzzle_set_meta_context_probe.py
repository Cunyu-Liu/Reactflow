#!/usr/bin/env python3
"""Implementation-only real-fold runner for a future puzzle-set amendment.

The active V14 authority cannot run this module.  A future amendment must make
``reactflow_delta_puzzle_set_meta_context`` the active task and issue the exact
training token before ``run_real_fold`` can access outer-train outcomes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.puzzle_set_meta_context import (
    BLOCK_DIAGONAL_NULL,
    FULL_CROSS_CONSTRUCT,
    fit_puzzle_set_point_model,
    make_exact_full_model_pair,
    parameter_count,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
    assemble_puzzle_training_batches,
    predict_held_puzzle_points,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    _feature41_matrix,
    _point_cells,
)


FOLD_SCHEMA = "reactflow_delta.puzzle_set_meta_context_fold.proposed.v1"
EXPECTED_PROJECT_TASK = "reactflow_delta_puzzle_set_meta_context"
EXPECTED_TRAINING_TOKEN = "PUZZLE_SET_META_CONTEXT_REAL_DATA_TRAINING_ONLY"


def assert_real_training_authority(repo_root: Path) -> None:
    active_path = repo_root / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("puzzle-set real training is not the active task")
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
    device: str,
) -> dict[str, Any]:
    """Prepare outer-train batches and held outcome-blind inputs."""

    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
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
    training_batches = assemble_puzzle_training_batches(
        train_records, cells, context_cache
    )
    by_construct: dict[str, list[Any]] = defaultdict(list)
    for record in held_records:
        by_construct[str(record.construct_id)].append(record)
    held_contexts = {
        construct_id: context_cache[construct_id]
        for construct_id in sorted(by_construct)
    }
    held_feature41 = {}
    for construct_id, construct_records in sorted(by_construct.items()):
        construct_records.sort(
            key=lambda record: (
                int(record.design_pos),
                str(record.ref),
                str(record.alt),
            )
        )
        construct = univ.get_construct(construct_id)
        _basis, matrix = _feature41_matrix(
            construct,
            construct_records,
            feature41_model,
            unconstrained,
            constrained,
        )
        held_feature41[construct_id] = matrix
    return {
        "training_batches": training_batches,
        "held_records": held_records,
        "held_contexts": held_contexts,
        "held_feature41": held_feature41,
    }


def run_prepared_fold(
    *,
    univ: Any,
    prepared: dict[str, Any],
    outer_fold: int,
    held_puzzle: str,
    seed: int,
    epochs: int,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Fit exact matched arms and emit one target-free fold artifact."""

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / f"puzzle_set_fold_result_fold{outer_fold}_seed{seed}.json"
    prediction_path = out_dir / (
        f"puzzle_set_predictions_fold{outer_fold}_seed{seed}.npz"
    )
    candidate_checkpoint = out_dir / (
        f"puzzle_set_candidate_fold{outer_fold}_seed{seed}.pt"
    )
    null_checkpoint = out_dir / f"puzzle_set_null_fold{outer_fold}_seed{seed}.pt"
    existing = [
        path
        for path in (
            fold_path,
            prediction_path,
            candidate_checkpoint,
            null_checkpoint,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite puzzle-set fold: {existing}")

    candidate, null = make_exact_full_model_pair(seed=seed, device=device)
    candidate_history = fit_puzzle_set_point_model(
        candidate,
        prepared["training_batches"],
        epochs=epochs,
        seed=seed,
    )
    null_history = fit_puzzle_set_point_model(
        null,
        prepared["training_batches"],
        epochs=epochs,
        seed=seed,
    )
    torch.save(candidate.state_dict(), candidate_checkpoint)
    torch.save(null.state_dict(), null_checkpoint)
    prediction = predict_held_puzzle_points(
        univ=univ,
        held_records=prepared["held_records"],
        context_cache=prepared["held_contexts"],
        feature41_by_construct=prepared["held_feature41"],
        candidate=candidate,
        null=null,
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
        "evidence_status": "IMPLEMENTATION_PROBE_ONLY_NO_SCIENTIFIC_AUTHORITY",
        "outer_fold": int(outer_fold),
        "held_puzzle": str(held_puzzle),
        "seed": int(seed),
        "epochs": int(epochs),
        "candidate_connectivity": FULL_CROSS_CONSTRUCT,
        "null_connectivity": BLOCK_DIAGONAL_NULL,
        "candidate_parameter_count": parameter_count(candidate),
        "null_parameter_count": parameter_count(null),
        "candidate_history": candidate_history,
        "null_history": null_history,
        "candidate_checkpoint": str(candidate_checkpoint),
        "null_checkpoint": str(null_checkpoint),
        "prediction_artifact": str(prediction_path),
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "invariants": {
            "outcome_blind_puzzle_set_inputs": True,
            "exact_parameter_and_initialization_match": True,
            "candidate_full_cross_construct_attention": True,
            "null_block_diagonal_attention": True,
            "puzzle_balanced_training": True,
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
    seed: int,
    epochs: int,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    assert_real_training_authority(repo_root)
    prepared = prepare_real_fold(
        univ=univ,
        records=records,
        fold=fold,
        feature41_model=feature41_model,
        unconstrained=unconstrained,
        constrained=constrained,
        device=device,
    )
    return run_prepared_fold(
        univ=univ,
        prepared=prepared,
        outer_fold=int(fold.outer_fold),
        held_puzzle=str(fold.held_puzzle),
        seed=seed,
        epochs=epochs,
        device=device,
        out_dir=out_dir,
    )
