#!/usr/bin/env python3
"""Run the frozen post-V14 branch-5 cross-construct route probe.

The runner is deliberately inert unless a future, branch-5-specific authority
is installed.  It recomputes the frozen V13 parent from its same-fold
checkpoint, uses only the same-fold frozen V14 encoder for non-focal WT
summaries, and writes target-free held-puzzle predictions.  Mutant outcomes are
used only for the nineteen outer-train puzzles when fitting the two fixed ridge
arms.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v5_probe import (
    EnsembleFeatureCache,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.model_rescue_v13 import (
    EXPECTED_POINT_PARAMETERS as V13_POINT_PARAMETERS,
    SECOND_PASS_EXACT,
    V13PointModel,
    freeze_point_model as freeze_v13_point,
)
from scripts.reactflow_delta.model_rescue_v14 import (
    EXPECTED_ENCODER_PARAMETERS as V14_ENCODER_PARAMETERS,
    V14PointModel,
    encoder_parameters as v14_encoder_parameters,
    freeze_point_model as freeze_v14_point,
)
from scripts.reactflow_delta.post_v14_branch5_route_probe import (
    ALIGNED_SHIFT,
    HIDDEN_WIDTH,
    MATCHED_NULL_SHIFT,
    PROBE_FEATURE_WIDTH,
    RAW_SUMMARY_WIDTH,
    RIDGE_ALPHA,
    ProbeRidgeStats,
    fit_probe_ridge,
    nonfocal_linear_summary,
    predict_probe_ridge,
    puzzle_method_balanced_weights,
    source_receiver_features,
    zero_preserving_v14_content_hidden,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    validate_puzzle_coordinate_frames,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    SafeTIC2AFold,
    load_tic2a_safe_registry,
    read_json_object,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    _feature41_matrix,
    _point_cells,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


EXPECTED_PROJECT_TASK = "reactflow_delta_post_v14_branch5_route_probe"
PREDICTION_PHASE = "B5RP1"
PREDICTION_TOKEN = "POST_V14_BRANCH5_LINEAR_CROSS_CONSTRUCT_ROUTE_PREDICTION_ONLY"
FOLD_SCHEMA = "reactflow_delta.puzzle_set_branch5_route_probe_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.puzzle_set_branch5_route_probe_prediction.v1"
RIDGE_SCHEMA = "reactflow_delta.puzzle_set_branch5_route_probe_ridge.v1"
EXPECTED_FOLDS = list(range(20))
EXPECTED_SEED = 0
EXPECTED_PARENT_STATE = {
    "v14_status": "TERMINAL_V14M3_TOP_JOURNAL_SCREEN_FAIL",
    "post_v14_first_matching_branch_id": "5",
    "post_v14_route_classification": "INDEPENDENT_CONSTRUCT_TRANSFER_LIMITED",
    "v14m4_path_allowed": False,
}
SOURCE_MANIFEST_SCHEMA = "reactflow_delta.post_v14_branch5_safe_source_manifest.v1"
SOURCE_MANIFEST_STATUS = "POST_V14_BRANCH5_SAFE_SOURCE_MANIFEST_PASS"
SOURCE_MANIFEST_TOP_FIELDS = {"schema_version", "status", "parent_state", "folds"}
FROZEN_RUNTIME_PATHS = {
    "v13_checkpoint_dir": Path(
        "/mnt/cunyuliu/reactflow_delta_model_rescue_v13/v13m3_screen_seed0"
    ),
    "v14_checkpoint_dir": Path(
        "/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0"
    ),
    "source_manifest_path": Path(
        "/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/"
        "source_binding/post_v14_branch5_safe_source_manifest.json"
    ),
    "m2_csv_path": Path(
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "openknot_m2/OK7a_M2_data.v4.5.2.csv"
    ),
    "tic2a_merged_registry_path": Path(
        "/mnt/cunyuliu/reactflow_delta_target_identity_correction/"
        "tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json"
    ),
    "unconstrained_feature_cache_path": Path(
        "/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/"
        "ensemble_delta_cache.h5"
    ),
    "constrained_feature_cache_path": Path(
        "/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/"
        "constrained_cache.h5"
    ),
    "prediction_dir": Path(
        "/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/b5rp1_seed0"
    ),
    "complete_unscored_merge_path": Path(
        "/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/b5rp1_seed0/"
        "puzzle_set_branch5_probe_complete_unscored_merge.json"
    ),
    "complete_score_path": Path(
        "/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/b5rp1_seed0/"
        "puzzle_set_branch5_probe_complete_score.json"
    ),
    "qualification_path": Path(
        "/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/b5rp1_seed0/"
        "puzzle_set_branch5_probe_qualification.json"
    ),
}
V14_HIDDEN_SOURCE = "V14_ENCODE_REAL_MINUS_V14_ENCODE_COORDINATE_ONLY_REFERENCE"
STANDARDIZATION_INACTIVE_STD_THRESHOLD = 1.0e-8
FORBIDDEN_SOURCE_MANIFEST_FIELDS = {
    "target",
    "held_target",
    "error",
    "held_error",
    "mask",
    "held_mask",
    "target_error",
    "target_mask",
    "qualified_target_mask",
    "loss",
    "training_history",
    "training_histories",
    "score",
    "per_puzzle_effect",
    "gate",
    "gates",
}
FORBIDDEN_PREDICTION_FIELDS = {
    "target",
    "held_target",
    "target_error",
    "held_target_error",
    "target_mask",
    "qualified_mask",
    "qualified_target_mask",
    "held_qualified_target_mask",
    "loss",
    "score",
    "mae",
    "per_puzzle_effect",
}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_prediction(path: Path, value: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **value)
    os.replace(temporary, path)


def _parse_folds(raw: str) -> list[int]:
    folds = [int(value) for value in raw.split(",") if value.strip()]
    if (
        not folds
        or len(folds) != len(set(folds))
        or not set(folds) <= set(EXPECTED_FOLDS)
    ):
        raise ValueError(
            "branch5 folds must be unique members of zero through nineteen"
        )
    return sorted(folds)


def _assert_parent_state(active: dict[str, Any]) -> None:
    parent = active.get("parent_state")
    if not isinstance(parent, dict) or any(
        parent.get(name) != value for name, value in EXPECTED_PARENT_STATE.items()
    ):
        raise RuntimeError(
            "branch5 authority lacks the exact terminal V14 branch-5 parent"
        )


def assert_frozen_runtime_paths(
    authority: Any,
    *,
    required_fields: tuple[str, ...],
    cli_paths: dict[str, Path] | None = None,
) -> None:
    """Require exact authority bindings and, when supplied, exact CLI paths."""

    if not isinstance(authority, dict):
        raise RuntimeError("branch5 active authority must be one mapping")
    unknown = set(required_fields) - set(FROZEN_RUNTIME_PATHS)
    if unknown:
        raise ValueError(f"unknown branch5 frozen runtime fields: {sorted(unknown)}")
    for field in required_fields:
        expected = FROZEN_RUNTIME_PATHS[field]
        observed = Path(str(authority.get(field, "")))
        if not observed.is_absolute() or observed != expected:
            raise RuntimeError(
                f"branch5 active authority {field} differs from the frozen path"
            )
    if cli_paths is None:
        return
    if set(cli_paths) != set(required_fields):
        raise RuntimeError("branch5 CLI path field universe is incomplete")
    for field, raw_path in cli_paths.items():
        observed = Path(raw_path)
        if not observed.is_absolute() or observed != FROZEN_RUNTIME_PATHS[field]:
            raise RuntimeError(f"branch5 CLI {field} differs from active authority")


def assert_run_authority(
    repo_root: Path,
    *,
    source_manifest: Path | None = None,
    m2_csv: Path | None = None,
    tic2a_merged_registry: Path | None = None,
    unconstrained_feature_cache: Path | None = None,
    constrained_feature_cache: Path | None = None,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the exact future B5RP1 authority is active."""

    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("branch5 route probe is not the active project")
    if active.get("authority", {}).get("current_phase") != PREDICTION_PHASE:
        raise RuntimeError("branch5 runner is closed outside B5RP1")
    if active.get("runnable_phases") != [PREDICTION_PHASE]:
        raise RuntimeError("B5RP1 must be the only runnable phase")
    if (
        active.get("training_allowed") != PREDICTION_TOKEN
        or active.get("candidate_model_training_allowed") != PREDICTION_TOKEN
    ):
        raise RuntimeError("branch5 prediction-only training token is absent")
    if (
        active.get("held_score_read_allowed") is not False
        or active.get("partial_fold_score_read_allowed") is not False
        or active.get("new_external_outcome_access_allowed") is not False
    ):
        raise RuntimeError(
            "branch5 training requires held and external outcomes closed"
        )
    authority = active.get("authority", {})
    required_paths = (
        "source_manifest_path",
        "m2_csv_path",
        "tic2a_merged_registry_path",
        "unconstrained_feature_cache_path",
        "constrained_feature_cache_path",
        "prediction_dir",
    )
    provided = {
        "source_manifest_path": source_manifest,
        "m2_csv_path": m2_csv,
        "tic2a_merged_registry_path": tic2a_merged_registry,
        "unconstrained_feature_cache_path": unconstrained_feature_cache,
        "constrained_feature_cache_path": constrained_feature_cache,
        "prediction_dir": prediction_dir,
    }
    assert_frozen_runtime_paths(
        authority,
        required_fields=required_paths,
        cli_paths=(
            {name: value for name, value in provided.items() if value is not None}
            if any(value is not None for value in provided.values())
            else None
        ),
    )
    if authority.get("source_manifest_status") != SOURCE_MANIFEST_STATUS:
        raise RuntimeError("branch5 active safe-source binding is absent")
    _assert_parent_state(active)
    return active


def _manifest_contains_forbidden_fields(value: Any) -> bool:
    if isinstance(value, dict):
        if set(value) & FORBIDDEN_SOURCE_MANIFEST_FIELDS:
            return True
        return any(_manifest_contains_forbidden_fields(item) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_contains_forbidden_fields(item) for item in value)
    return False


def _load_source_registry(
    path: Path,
    *,
    expected_checkpoint_dirs: dict[str, Path] | None = None,
) -> dict[int, dict[str, Any]]:
    """Load only the score/prediction/history-free post-terminal source projection."""

    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(
            "branch5 safe source manifest must be one absolute file"
        )
    manifest = read_json_object(path)
    if (
        set(manifest) != SOURCE_MANIFEST_TOP_FIELDS
        or manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("status") != SOURCE_MANIFEST_STATUS
        or manifest.get("parent_state") != EXPECTED_PARENT_STATE
        or _manifest_contains_forbidden_fields(manifest)
    ):
        raise ValueError(
            "branch5 safe source manifest identity or field boundary failed"
        )
    rows: dict[int, dict[str, Any]] = {}
    expected_fields = {
        "outer_fold",
        "held_puzzle",
        "seed",
        "v13_source_phase",
        "v13_candidate_checkpoint",
        "v14_source_phase",
        "v14_arm",
        "v14_candidate_checkpoint",
        "held_score_closed_at_projection",
        "external_outcome_accessed",
    }
    for row in manifest.get("folds", []):
        fold = int(row.get("outer_fold", -1))
        v13_checkpoint = Path(str(row.get("v13_candidate_checkpoint", "")))
        v14_checkpoint = Path(str(row.get("v14_candidate_checkpoint", "")))
        if (
            set(row) != expected_fields
            or fold in rows
            or fold not in EXPECTED_FOLDS
            or str(row.get("held_puzzle")) != f"P{fold + 1:02d}"
            or int(row.get("seed", -1)) != EXPECTED_SEED
            or row.get("v13_source_phase") != "V13M3"
            or row.get("v14_source_phase") != "V14M3"
            or row.get("v14_arm") != "CANDIDATE"
            or row.get("held_score_closed_at_projection") is not True
            or row.get("external_outcome_accessed") is not False
            or v13_checkpoint.name != f"v13_candidate_point_fold{fold}_seed0.pt"
            or v14_checkpoint.name != f"v14_candidate_point_fold{fold}_seed0.pt"
            or not v13_checkpoint.is_absolute()
            or not v14_checkpoint.is_absolute()
            or not v13_checkpoint.is_file()
            or not v14_checkpoint.is_file()
        ):
            raise ValueError("branch5 safe source registry row is not exact")
        if expected_checkpoint_dirs is not None and (
            v13_checkpoint.parent != expected_checkpoint_dirs["v13_checkpoint_dir"]
            or v14_checkpoint.parent != expected_checkpoint_dirs["v14_checkpoint_dir"]
        ):
            raise ValueError(
                "branch5 safe source registry checkpoint directory differs from frozen binding"
            )
        rows[fold] = row
    if sorted(rows) != EXPECTED_FOLDS:
        raise ValueError("branch5 safe source registry is not folds0-19")
    return rows


def _load_v13_parent(path: Path, device: str) -> V13PointModel:
    model = V13PointModel(second_pass_mode=SECOND_PASS_EXACT).to(device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True), strict=True
    )
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != V13_POINT_PARAMETERS
    ):
        raise RuntimeError("branch5 V13 parent parameter count changed")
    freeze_v13_point(model)
    return model


def _load_v14_encoder(path: Path, device: str) -> V14PointModel:
    model = V14PointModel().to(device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True), strict=True
    )
    if sum(parameter.numel() for parameter in v14_encoder_parameters(model)) != (
        V14_ENCODER_PARAMETERS
    ):
        raise RuntimeError("branch5 V14 encoder parameter count changed")
    freeze_v14_point(model)
    return model


def _checkpoint_sources(
    *,
    fold_id: int,
    held_puzzle: str,
    source_row: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if str(source_row.get("held_puzzle")) != str(held_puzzle):
        raise ValueError("branch5 safe source held-puzzle identity differs")
    v13 = Path(str(source_row.get("v13_candidate_checkpoint", "")))
    v14 = Path(str(source_row.get("v14_candidate_checkpoint", "")))
    expected_v14_name = f"v14_candidate_point_fold{fold_id}_seed0.pt"
    if v14.name != expected_v14_name:
        raise ValueError("branch5 V14 checkpoint filename or arm differs")
    if v13.name != f"v13_candidate_point_fold{fold_id}_seed0.pt":
        raise ValueError("branch5 V13 checkpoint filename differs")
    if not v13.is_file() or not v14.is_file():
        raise FileNotFoundError("branch5 same-fold parent checkpoint is absent")
    return {
        "v13_point_checkpoint": {
            "path": str(v13),
            "role": "FROZEN_SAME_OUTER_FOLD_V13_POINT_MODEL",
            "outer_fold": fold_id,
            "seed": EXPECTED_SEED,
        },
        "v14_encoder_checkpoint": {
            "path": str(v14),
            "role": "FROZEN_SAME_OUTER_FOLD_V14_OUTCOME_BLIND_ENCODER",
            "outer_fold": fold_id,
            "seed": EXPECTED_SEED,
        },
    }


def _records_by_puzzle_construct(
    records: Iterable[Any],
) -> dict[str, dict[str, list[Any]]]:
    result: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        result[str(record.puzzle)][str(record.construct_id)].append(record)
    return {
        puzzle: {
            construct: sorted(
                values,
                key=lambda row: (int(row.design_pos), str(row.ref), str(row.alt)),
            )
            for construct, values in sorted(constructs.items())
        }
        for puzzle, constructs in sorted(result.items())
    }


def _encode_puzzle(
    *,
    model: V14PointModel,
    construct_ids: list[str],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(construct_ids) != 8:
        raise ValueError("branch5 requires exactly eight constructs per puzzle")
    hidden = []
    reactivity = []
    observed = []
    model.eval()
    with torch.no_grad():
        for construct_id in construct_ids:
            context = context_cache[construct_id]
            encoded = zero_preserving_v14_content_hidden(model, context)
            if encoded.shape != (len(context[0]), HIDDEN_WIDTH):
                raise RuntimeError("branch5 V14 hidden width changed")
            if not torch.isfinite(encoded).all():
                raise RuntimeError("branch5 V14 content-contrast hidden is nonfinite")
            hidden.append(encoded)
            reactivity.append(context[1])
            observed.append(context[3].bool())
    return torch.stack(hidden), torch.stack(reactivity), torch.stack(observed)


def add_weighted_grid_to_stats(
    stats: ProbeRidgeStats,
    *,
    summary: np.ndarray,
    edit_index: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
) -> None:
    """Accumulate [source,receiver] sufficient statistics without a huge X grid."""

    receiver = np.asarray(summary, dtype=np.float64)
    edit = np.asarray(edit_index, dtype=np.int64)
    y = np.asarray(residual, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if receiver.ndim != 2 or receiver.shape[1] != RAW_SUMMARY_WIDTH:
        raise ValueError("branch5 grid receiver summary has invalid shape")
    source = receiver[edit]
    if y.shape != w.shape or y.shape != (len(edit), len(receiver)):
        raise ValueError("branch5 grid residual or weight is misaligned")
    positive = w > 0
    if not bool(positive.any()) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("branch5 grid requires finite nonnegative weights")
    if not np.isfinite(y[positive]).all():
        raise ValueError("branch5 qualified residuals must be finite")
    safe_y = np.where(positive, y, 0.0)
    row_weight = w.sum(axis=1)
    column_weight = w.sum(axis=0)
    weighted_y = w * safe_y
    sum_x = np.concatenate([source.T @ row_weight, receiver.T @ column_weight])
    sum_x2 = np.concatenate(
        [(source**2).T @ row_weight, (receiver**2).T @ column_weight]
    )
    source_source = source.T @ (row_weight[:, None] * source)
    receiver_receiver = receiver.T @ (column_weight[:, None] * receiver)
    source_receiver = source.T @ (w @ receiver)
    xtx = np.block(
        [[source_source, source_receiver], [source_receiver.T, receiver_receiver]]
    )
    xty = np.concatenate(
        [source.T @ weighted_y.sum(axis=1), receiver.T @ weighted_y.sum(axis=0)]
    )
    stats.sum_weight += float(w.sum())
    stats.sum_x += sum_x
    stats.sum_x2 += sum_x2
    stats.xtx += xtx
    stats.sum_y += float(weighted_y.sum())
    stats.xty += xty


def _fit_probe_models(
    *,
    train_records: list[Any],
    cells: list[dict[str, Any]],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    v14_encoder: V14PointModel,
) -> tuple[dict[str, dict[str, np.ndarray | float]], dict[str, int]]:
    cells_by_construct = {str(cell["construct_id"]): cell for cell in cells}
    grouped = _records_by_puzzle_construct(train_records)
    puzzle_cells: list[list[np.ndarray]] = []
    puzzle_cell_ids: list[list[str]] = []
    for puzzle, constructs in grouped.items():
        construct_ids = sorted(constructs)
        if len(construct_ids) != 8:
            raise ValueError(
                f"branch5 train puzzle {puzzle} does not have eight constructs"
            )
        eligible = [value for value in construct_ids if value in cells_by_construct]
        if not eligible:
            raise ValueError(f"branch5 train puzzle {puzzle} has no supervised cell")
        puzzle_cells.append(
            [
                cells_by_construct[value]["qualified_mask"].cpu().numpy()
                for value in eligible
            ]
        )
        puzzle_cell_ids.append(eligible)
    weights = puzzle_method_balanced_weights(puzzle_cells)
    stats = {
        "aligned": ProbeRidgeStats.zeros(),
        "shift17": ProbeRidgeStats.zeros(),
    }
    n_rows = 0
    for (puzzle, constructs), cell_ids, cell_weights in zip(
        grouped.items(), puzzle_cell_ids, weights
    ):
        construct_ids = sorted(constructs)
        hidden, reactivity, observed = _encode_puzzle(
            model=v14_encoder,
            construct_ids=construct_ids,
            context_cache=context_cache,
        )
        for construct_id, weight in zip(cell_ids, cell_weights):
            cell = cells_by_construct[construct_id]
            focal = construct_ids.index(construct_id)
            target_delta = (
                cell["target"].cpu().numpy() - cell["wt"].cpu().numpy()[None, :]
            )
            parent = cell["parent_point"].cpu().numpy()
            residual = target_delta - parent
            for name, shift in (
                ("aligned", ALIGNED_SHIFT),
                ("shift17", MATCHED_NULL_SHIFT),
            ):
                summary = (
                    nonfocal_linear_summary(
                        hidden,
                        reactivity,
                        observed,
                        focal_index=focal,
                        shift=shift,
                    )
                    .cpu()
                    .numpy()
                )
                add_weighted_grid_to_stats(
                    stats[name],
                    summary=summary,
                    edit_index=cell["edit"].cpu().numpy(),
                    residual=residual,
                    weight=weight,
                )
            n_rows += int(np.count_nonzero(weight))
    models = {name: fit_probe_ridge(value) for name, value in stats.items()}
    if (
        not np.isclose(stats["aligned"].sum_weight, n_rows, atol=1e-6, rtol=0.0)
        or not np.isclose(
            stats["aligned"].sum_weight,
            stats["shift17"].sum_weight,
            atol=1e-9,
            rtol=0.0,
        )
        or not np.isclose(
            stats["aligned"].sum_y,
            stats["shift17"].sum_y,
            atol=1e-9,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "branch5 aligned/null target or hierarchical weight universe differs"
        )
    return models, {
        "n_outer_train_puzzles": len(grouped),
        "n_outer_train_supervised_cells": len(cells),
        "n_outer_train_qualified_rows": n_rows,
    }


def _held_prediction(
    *,
    univ: M2Universe,
    held_records: list[Any],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    feature41_model: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    v13_parent: V13PointModel,
    v14_encoder: V14PointModel,
    ridge_models: dict[str, dict[str, np.ndarray | float]],
    fold_id: int,
) -> dict[str, np.ndarray]:
    grouped = _records_by_puzzle_construct(held_records)
    if len(grouped) != 1:
        raise ValueError("branch5 held prediction requires exactly one puzzle")
    _puzzle, constructs = next(iter(grouped.items()))
    construct_ids = sorted(constructs)
    hidden, reactivity, observed = _encode_puzzle(
        model=v14_encoder,
        construct_ids=construct_ids,
        context_cache=context_cache,
    )
    keys: list[str] = []
    values: dict[str, list[np.ndarray]] = {
        "parent_point": [],
        "aligned_point": [],
        "shift17_point": [],
    }
    v13_parent.eval()
    with torch.no_grad():
        for focal, construct_id in enumerate(construct_ids):
            records = constructs[construct_id]
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            _basis, feature41 = _feature41_matrix(
                construct, records, feature41_model, unconstrained, constrained
            )
            device = next(v13_parent.parameters()).device
            edit = torch.tensor([int(row.full_pos) for row in records], device=device)
            distance = (
                torch.arange(length, device=device)[None, :] - edit[:, None]
            ).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(records), 1)),
                device=device,
            )
            parent = (
                v13_parent.forward_point(
                    context_cache[construct_id],
                    edit,
                    distance,
                    [str(row.ref) for row in records],
                    [str(row.alt) for row in records],
                    prediction_mask,
                    torch.tensor(feature41, device=device),
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            predictions = {"parent_point": parent}
            for name, shift in (
                ("aligned", ALIGNED_SHIFT),
                ("shift17", MATCHED_NULL_SHIFT),
            ):
                summary = nonfocal_linear_summary(
                    hidden,
                    reactivity,
                    observed,
                    focal_index=focal,
                    shift=shift,
                )
                features = source_receiver_features(summary, edit).cpu().numpy()
                increment = predict_probe_ridge(
                    ridge_models[name], features.reshape(-1, PROBE_FEATURE_WIDTH)
                ).reshape(len(records), length)
                point = parent + increment
                point[~prediction_mask.cpu().numpy()] = 0.0
                same = np.asarray(
                    [
                        str(row.ref).replace("T", "U") == str(row.alt).replace("T", "U")
                        for row in records
                    ]
                )
                point[same] = 0.0
                predictions[f"{name}_point"] = point
            for mutant, record in enumerate(records):
                keys.extend(
                    _bio_key(univ, record, position) for position in range(length)
                )
                for name in values:
                    values[name].append(predictions[name][mutant])
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(len(keys), fold_id, dtype=np.int64),
        "seed": np.full(len(keys), EXPECTED_SEED, dtype=np.int64),
        "registered_status": np.full(len(keys), "covered", dtype=object),
        **{
            name: np.concatenate(rows).astype(np.float64)
            for name, rows in values.items()
        },
    }
    if set(output) & FORBIDDEN_PREDICTION_FIELDS:
        raise RuntimeError("branch5 held prediction contains target-side fields")
    if len(keys) != len(set(keys)) or not all(
        np.isfinite(value).all()
        for value in output.values()
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu"
    ):
        raise RuntimeError("branch5 held prediction key or numeric integrity failed")
    return output


def _serializable_model(model: dict[str, np.ndarray | float]) -> dict[str, Any]:
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for name, value in model.items()
    }


def run_fold(
    *,
    univ: M2Universe,
    records: list[Any],
    fold: Any,
    device: str,
    out_dir: Path,
    m2_csv_path: Path,
    source_manifest_path: Path,
    source_row: dict[str, Any],
    tic2a_registry_path: Path,
    tic2a_source: SafeTIC2AFold,
    unconstrained_cache_path: Path,
    constrained_cache_path: Path,
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    sources = _checkpoint_sources(
        fold_id=fold_id,
        held_puzzle=str(fold.held_puzzle),
        source_row=source_row,
    )
    output_paths = {
        "fold": out_dir / f"puzzle_set_branch5_probe_fold{fold_id}_seed0.json",
        "prediction": out_dir
        / f"puzzle_set_branch5_probe_predictions_fold{fold_id}_seed0.npz",
        "ridge": out_dir / f"puzzle_set_branch5_probe_ridge_fold{fold_id}_seed0.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            f"branch5 refuses to overwrite fold {fold_id}: {existing}"
        )
    train_puzzles = set(fold.train_puzzles)
    train_records = [row for row in records if row.puzzle in train_puzzles]
    held_records = [row for row in records if row.puzzle == fold.held_puzzle]
    frames = validate_puzzle_coordinate_frames(train_records + held_records, univ)
    construct_ids = sorted(
        {str(row.construct_id) for row in train_records + held_records}
    )
    context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }
    if int(tic2a_source.row.get("outer_fold", -1)) != fold_id or str(
        tic2a_source.row.get("held_puzzle")
    ) != str(fold.held_puzzle):
        raise ValueError("branch5 TIC2A safe source differs from the outer fold")
    feature41_model = tic2a_source.feature41_model
    v13_parent = _load_v13_parent(Path(sources["v13_point_checkpoint"]["path"]), device)
    v14_encoder = _load_v14_encoder(
        Path(sources["v14_encoder_checkpoint"]["path"]), device
    )
    cells = _point_cells(
        univ, train_records, feature41_model, unconstrained, constrained, device
    )
    with torch.no_grad():
        for cell in cells:
            cell["parent_point"] = v13_parent.forward_point(
                context_cache[str(cell["construct_id"])],
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
                cell["feature41_point"],
            ).detach()
    ridge_models, train_counts = _fit_probe_models(
        train_records=train_records,
        cells=cells,
        context_cache=context_cache,
        v14_encoder=v14_encoder,
    )
    prediction = _held_prediction(
        univ=univ,
        held_records=held_records,
        context_cache={
            construct_id: context_cache[construct_id]
            for construct_id in sorted({str(row.construct_id) for row in held_records})
        },
        feature41_model=feature41_model,
        unconstrained=unconstrained,
        constrained=constrained,
        v13_parent=v13_parent,
        v14_encoder=v14_encoder,
        ridge_models=ridge_models,
        fold_id=fold_id,
    )
    _atomic_write_prediction(output_paths["prediction"], prediction)
    ridge_artifact = {
        "schema_version": RIDGE_SCHEMA,
        "outer_fold": fold_id,
        "seed": EXPECTED_SEED,
        "feature_width": PROBE_FEATURE_WIDTH,
        "arms_fit_independently": True,
        "weighted_outer_train_standardization_per_arm": True,
        "intercept_unpenalized": True,
        "ridge_alpha": RIDGE_ALPHA,
        "shifts": {"aligned": ALIGNED_SHIFT, "shift17": MATCHED_NULL_SHIFT},
        "models": {
            name: _serializable_model(model) for name, model in ridge_models.items()
        },
        "fit_target": "SIGNED_DELTA_MINUS_FROZEN_V13_POINT",
        "v14_hidden_source": V14_HIDDEN_SOURCE,
        "reference_preserves_position_and_region": True,
        "reference_zeros_sequence_reactivity_precision_observed": True,
        "shift17_applied_only_after_content_contrast": True,
        "inactive_std_threshold_lt": STANDARDIZATION_INACTIVE_STD_THRESHOLD,
        "inactive_std_replacement_scale": 1.0,
        "constant_feature_standardized_value": 0.0,
        "sample_weight_hierarchy": (
            "PUZZLE_TO_METHOD_CELL_TO_MUTANT_TO_QUALIFIED_POSITION"
        ),
        **train_counts,
        "held_puzzle_target_accessed": False,
        "partial_score_computed": False,
        "external_outcome_accessed": False,
    }
    _atomic_write_json(output_paths["ridge"], ridge_artifact)
    return {
        "schema_version": FOLD_SCHEMA,
        "phase": PREDICTION_PHASE,
        "status": "BRANCH5_ROUTE_PROBE_FOLD_PREDICTION_PASS",
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": EXPECTED_SEED,
        "source_provenance": {
            **sources,
            "m2_csv": {
                "path": str(m2_csv_path),
                "role": "TRAIN_TARGET_AND_OUTCOME_BLIND_CONTEXT_SOURCE",
                "scope": "GLOBAL",
            },
            "tic2a_merged_registry": {
                "path": str(tic2a_registry_path),
                "role": "STRICT_TARGET_FREE_FEATURE41_REGISTRY",
                "scope": "GLOBAL",
            },
            "unconstrained_feature_cache": {
                "path": str(unconstrained_cache_path),
                "role": "OUTCOME_BLIND_FEATURE41_UNCONSTRAINED_CACHE",
                "scope": "GLOBAL",
            },
            "constrained_feature_cache": {
                "path": str(constrained_cache_path),
                "role": "OUTCOME_BLIND_FEATURE41_CONSTRAINED_CACHE",
                "scope": "GLOBAL",
            },
            "tic2a_feature41_model_artifact": {
                "path": str(tic2a_source.model_path),
                "role": "FROZEN_OUTER_FOLD_TIC2A_FEATURE41_MODEL_ONLY",
                "outer_fold": fold_id,
                "source_phase": "TIC2A",
                "held_target_used_for_prediction": False,
                "held_score_computed": False,
                "partial_score_inspected": False,
                "external_outcome_accessed": False,
            },
            "safe_source_manifest": {
                "path": str(source_manifest_path),
                "role": "POST_V14_BRANCH5_SCORE_PREDICTION_HISTORY_FREE_SOURCE_PROJECTION",
            },
        },
        "ridge_model_artifact": str(output_paths["ridge"]),
        "prediction_artifact": str(output_paths["prediction"]),
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "coordinate_frame_count": len(frames),
        **train_counts,
        "invariants": {
            "target_profile_identity_exact": True,
            "samefold_v13_parent_recomputed_from_checkpoint": True,
            "samefold_v14_encoder_recomputed_from_checkpoint": True,
            "v14_hidden_is_zero_preserving_content_contrast": True,
            "v14_coordinate_only_reference_preserves_position_and_region": True,
            "v14_coordinate_only_reference_zeros_biological_streams": True,
            "shift17_applied_only_after_v14_content_contrast": True,
            "tic2a_feature41_loaded_from_strict_target_free_projection": True,
            "seven_nonfocal_constructs_only": True,
            "aligned_and_shift17_fit_independently": True,
            "outer_train_weighted_standardization": True,
            "ridge_alpha_one_unpenalized_intercept": True,
            "held_prediction_full_registered_universe": True,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "partial_score_inspected": False,
            "external_outcome_accessed": False,
            "model_or_threshold_selection_performed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)

    source_manifest = args.source_manifest.resolve()
    folds = _parse_folds(args.folds)
    out_dir = args.out_dir.resolve()
    m2_csv_path = args.m2_csv.resolve()
    tic2a_registry_path = args.tic2a_merged_json.resolve()
    unconstrained_cache_path = args.unconstrained_cache.resolve()
    constrained_cache_path = args.constrained_cache.resolve()
    assert_run_authority(
        args.repo_root.resolve(),
        source_manifest=source_manifest,
        m2_csv=m2_csv_path,
        tic2a_merged_registry=tic2a_registry_path,
        unconstrained_feature_cache=unconstrained_cache_path,
        constrained_feature_cache=constrained_cache_path,
        prediction_dir=out_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(m2_csv_path)
    identity = univ.build()
    if (
        identity.get("n_canonical_mutant_full_profiles") != 13976
        or identity.get("canonical_mutant_full_profile_identity")
        != "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("branch5 requires exact canonical mutant identity")
    records = univ.get_records()
    split = build_split_v4(sorted({row.puzzle for row in records}), seed=20260813)
    selected = [row for row in split["folds"] if int(row.outer_fold) in set(folds)]
    if len(selected) != len(folds):
        raise ValueError("branch5 requested fold is absent")
    source_rows = _load_source_registry(
        source_manifest,
        expected_checkpoint_dirs={
            "v13_checkpoint_dir": FROZEN_RUNTIME_PATHS["v13_checkpoint_dir"],
            "v14_checkpoint_dir": FROZEN_RUNTIME_PATHS["v14_checkpoint_dir"],
        },
    )
    tic2a_sources = load_tic2a_safe_registry(tic2a_registry_path)
    unconstrained = EnsembleFeatureCache(unconstrained_cache_path)
    constrained = ConstrainedFeatureCache(constrained_cache_path)
    validate_cache_alignment(unconstrained, constrained)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            result = run_fold(
                univ=univ,
                records=records,
                fold=fold,
                device=device,
                out_dir=out_dir,
                m2_csv_path=m2_csv_path,
                source_manifest_path=source_manifest,
                source_row=source_rows[fold_id],
                tic2a_registry_path=tic2a_registry_path,
                tic2a_source=tic2a_sources[fold_id],
                unconstrained_cache_path=unconstrained_cache_path,
                constrained_cache_path=constrained_cache_path,
                unconstrained=unconstrained,
                constrained=constrained,
            )
            path = out_dir / f"puzzle_set_branch5_probe_fold{fold_id}_seed0.json"
            _atomic_write_json(path, result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
