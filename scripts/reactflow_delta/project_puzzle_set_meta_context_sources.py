#!/usr/bin/env python3
"""Project the exact seven-source manifest for Puzzle-Set P1 activation.

The production entrypoint accepts only a repository root.  It derives the
canonical active pointer and every source path from that pointer, checks those
paths against the frozen V5 machine contract, and never opens historical score
or wide fold-result artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from scripts.reactflow_delta.preflight_puzzle_set_meta_context_sources import (
    EXPECTED_PARAMETER_COUNTS,
    inspect_checkpoint_parameter_counts,
    inspect_feature_caches,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    EXPECTED_CONTRACT_ID,
    EXPECTED_FOLDS,
    FOLD_SCOPED_INPUT_SOURCES,
    FROZEN_INPUT_SOURCE_SPEC,
    SOURCE_BINDING_STATUS,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    SafeTIC2AFold,
    load_tic2a_safe_registry,
    validate_source_manifest,
)


EXPECTED_PROJECT_TASK = "reactflow_delta_puzzle_set_meta_context"
PROJECTION_PHASE = "P1M1"
PROJECTION_AUTHORITY_STATE = "SOURCE_MANIFEST_PROJECTION_ONLY"
PENDING_SOURCE_BINDING_STATUS = "REALIZED_PATHS_ROLES_AND_COUNTS_PENDING"
EXPECTED_PARENT_STATUS = "TERMINAL_V14M3_TOP_JOURNAL_SCREEN_FAIL"
EXPECTED_CACHE_ALIGNMENT = {
    "biological_key_universe_equal": True,
    "registered_mutants": 13_976,
    "receiver_length": 177,
    "unconstrained_width": 12,
    "constrained_cache_width": 12,
    "constrained_probe_width": 11,
}
FROZEN_COMMON_PATH_FIELDS = (
    "source_manifest_path",
    "v8_checkpoint_dir",
    "v13_checkpoint_dir",
    "v14_checkpoint_dir",
    "tic2a_merged_registry_path",
    "unconstrained_feature_cache_path",
    "constrained_feature_cache_path",
)
ROUTER_BRANCHES = {
    "3": {
        "classification": "CAPACITY_WITHOUT_PRETRAINING_INCREMENT",
        "probe_requirement": "NOT_APPLICABLE",
        "probe_status": "NOT_APPLICABLE",
    },
    "4": {
        "classification": "PRETRAINING_SIGNAL_INSUFFICIENT_FOR_TRANSFER",
        "probe_requirement": "NOT_APPLICABLE",
        "probe_status": "NOT_APPLICABLE",
    },
    "5": {
        "classification": "INDEPENDENT_CONSTRUCT_TRANSFER_LIMITED",
        "probe_requirement": "REQUIRED",
        "probe_status": "EXACT_PASS",
    },
}


@dataclass(frozen=True)
class ProjectionAuthority:
    """The exact safe path universe derived from the canonical active pointer."""

    active_path: Path
    machine_contract_path: Path
    output_path: Path
    v8_dir: Path
    v13_dir: Path
    v14_dir: Path
    tic2a_registry: Path
    unconstrained_cache: Path
    constrained_cache: Path


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one mapping")
    return value


def _canonical_repo_file(repo_root: Path, relative: str, label: str) -> Path:
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent from the selected repository")
    return path


def _assert_closed_projection_authority(active: Mapping[str, Any]) -> None:
    authority = active.get("authority")
    if (
        active.get("project_task_id") != EXPECTED_PROJECT_TASK
        or not isinstance(authority, Mapping)
        or authority.get("current_phase") != PROJECTION_PHASE
        or authority.get("current_authority_state") != PROJECTION_AUTHORITY_STATE
        or active.get("runnable_phases") != [PROJECTION_PHASE]
    ):
        raise RuntimeError("Puzzle-Set source projection is closed outside P1M1")
    for field in (
        "training_allowed",
        "candidate_model_training_allowed",
        "held_score_read_allowed",
        "partial_fold_score_read_allowed",
        "new_external_outcome_access_allowed",
    ):
        if active.get(field) is not False:
            raise RuntimeError(f"Puzzle-Set source projection requires {field}=false")
    if authority.get("source_binding_status") != PENDING_SOURCE_BINDING_STATUS:
        raise RuntimeError(
            "Puzzle-Set P1M1 requires the pending source-binding authority"
        )


def _assert_parent_route(active: Mapping[str, Any]) -> None:
    parent = active.get("parent_state")
    if not isinstance(parent, Mapping):
        raise RuntimeError("Puzzle-Set P1M1 parent route is absent")
    branch = str(parent.get("post_v14_first_matching_branch_id", ""))
    expected = ROUTER_BRANCHES.get(branch)
    if (
        expected is None
        or parent.get("v14_status") != EXPECTED_PARENT_STATUS
        or parent.get("v14m4_path_allowed") is not False
        or parent.get("post_v14_route_classification") != expected["classification"]
        or parent.get("post_v14_route_probe_requirement")
        != expected["probe_requirement"]
        or parent.get("post_v14_route_probe_status") != expected["probe_status"]
    ):
        raise RuntimeError(
            "Puzzle-Set P1M1 requires one exact eligible post-V14 parent route"
        )


def _absolute_path(value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise RuntimeError(f"Puzzle-Set {label} must be one absolute path")
    return path


def load_projection_authority(repo_root: Path) -> ProjectionAuthority:
    """Read only the canonical pointer and bind it to the frozen machine paths."""

    root = repo_root.resolve()
    active_path = _canonical_repo_file(
        root,
        "configs/reactflow_delta/active_contract.yaml",
        "Puzzle-Set canonical active pointer",
    )
    machine_contract_path = _canonical_repo_file(
        root,
        "configs/reactflow_delta/puzzle_set_meta_context_v5_amendment.yaml",
        "Puzzle-Set V5 machine contract",
    )
    active = _read_yaml_mapping(active_path, "active contract")
    machine = _read_yaml_mapping(machine_contract_path, "Puzzle-Set V5 contract")
    _assert_closed_projection_authority(active)
    _assert_parent_route(active)
    if machine.get("contract_id") != EXPECTED_CONTRACT_ID:
        raise RuntimeError("Puzzle-Set V5 machine contract identity changed")
    future_authority = machine.get("future_p1_runtime_authority")
    if (
        not isinstance(future_authority, Mapping)
        or future_authority.get("project_task_id") != EXPECTED_PROJECT_TASK
    ):
        raise RuntimeError("Puzzle-Set frozen future project authority changed")
    common = future_authority.get("common_source_paths")
    authority = active.get("authority")
    if not isinstance(common, Mapping) or not isinstance(authority, Mapping):
        raise RuntimeError("Puzzle-Set frozen future source authority is absent")
    if common.get("source_binding_status") != SOURCE_BINDING_STATUS:
        raise RuntimeError("Puzzle-Set frozen future source binding status changed")

    realized: dict[str, Path] = {}
    for field in FROZEN_COMMON_PATH_FIELDS:
        frozen_path = _absolute_path(common.get(field), f"contract {field}")
        active_path_value = _absolute_path(authority.get(field), f"active {field}")
        if active_path_value != frozen_path:
            raise RuntimeError(
                f"Puzzle-Set active {field} differs from the frozen future authority"
            )
        realized[field] = active_path_value

    return ProjectionAuthority(
        active_path=active_path,
        machine_contract_path=machine_contract_path,
        output_path=realized["source_manifest_path"],
        v8_dir=realized["v8_checkpoint_dir"],
        v13_dir=realized["v13_checkpoint_dir"],
        v14_dir=realized["v14_checkpoint_dir"],
        tic2a_registry=realized["tic2a_merged_registry_path"],
        unconstrained_cache=realized["unconstrained_feature_cache_path"],
        constrained_cache=realized["constrained_feature_cache_path"],
    )


def _require_source_path(path: Path, *, directory: bool, label: str) -> None:
    valid = path.is_absolute() and (path.is_dir() if directory else path.is_file())
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(
            f"Puzzle-Set {label} must be one existing absolute {kind}"
        )


def _source_record(
    source_id: str,
    *,
    path: Path,
    outer_fold: int | None,
) -> dict[str, Any]:
    expected = FROZEN_INPUT_SOURCE_SPEC[source_id]
    return {
        "path": str(path),
        "role": expected["role"],
        "used_in_candidate_prediction": expected["used_in_candidate_prediction"],
        "outer_fold": outer_fold,
        "seed": expected["seed"],
        "realized_parameter_count": expected["realized_parameter_count"],
        "trainable_in_p1": expected["trainable_in_p1"],
    }


def _atomic_write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        validate_source_manifest(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def project_source_manifest(
    repo_root: Path,
    *,
    checkpoint_inspector: Callable[..., dict[str, int | None]] = (
        inspect_checkpoint_parameter_counts
    ),
    cache_inspector: Callable[..., dict[str, int | bool]] = inspect_feature_caches,
    registry_loader: Callable[[Path], dict[int, SafeTIC2AFold]] = (
        load_tic2a_safe_registry
    ),
) -> dict[str, Any]:
    """Validate the exact 20-fold source universe and atomically bind it."""

    bound = load_projection_authority(repo_root)
    if bound.output_path.exists():
        raise FileExistsError(
            f"Puzzle-Set source manifest already exists: {bound.output_path}"
        )
    for path, label in (
        (bound.v8_dir, "V8 checkpoint directory"),
        (bound.v13_dir, "V13 checkpoint directory"),
        (bound.v14_dir, "V14 checkpoint directory"),
    ):
        _require_source_path(path, directory=True, label=label)
    for path, label in (
        (bound.tic2a_registry, "TIC2A safe registry"),
        (bound.unconstrained_cache, "unconstrained feature cache"),
        (bound.constrained_cache, "constrained feature cache"),
    ):
        _require_source_path(path, directory=False, label=label)

    cache_alignment = cache_inspector(
        unconstrained_cache=bound.unconstrained_cache,
        constrained_cache=bound.constrained_cache,
    )
    if cache_alignment != EXPECTED_CACHE_ALIGNMENT:
        raise ValueError("Puzzle-Set frozen feature-cache alignment changed")
    tic2a = registry_loader(bound.tic2a_registry)
    if tuple(sorted(tic2a)) != EXPECTED_FOLDS:
        raise ValueError("Puzzle-Set TIC2A registry is not exact folds0-19")

    global_sources = {
        "tic2a_merged_registry": _source_record(
            "tic2a_merged_registry", path=bound.tic2a_registry, outer_fold=None
        ),
        "unconstrained_feature_cache": _source_record(
            "unconstrained_feature_cache",
            path=bound.unconstrained_cache,
            outer_fold=None,
        ),
        "constrained_feature_cache": _source_record(
            "constrained_feature_cache",
            path=bound.constrained_cache,
            outer_fold=None,
        ),
    }
    rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        v8_checkpoint = bound.v8_dir / f"v8_corrected_mean_fold{fold}_seed0.pt"
        v13_checkpoint = bound.v13_dir / f"v13_candidate_point_fold{fold}_seed0.pt"
        v14_checkpoint = bound.v14_dir / f"v14_candidate_point_fold{fold}_seed0.pt"
        for checkpoint, label in (
            (v8_checkpoint, "V8 MeanAligned checkpoint"),
            (v13_checkpoint, "V13 point checkpoint"),
            (v14_checkpoint, "V14 encoder-source checkpoint"),
        ):
            _require_source_path(checkpoint, directory=False, label=label)
        observed_counts = checkpoint_inspector(
            v8_checkpoint=v8_checkpoint,
            v13_checkpoint=v13_checkpoint,
            v14_checkpoint=v14_checkpoint,
        )
        if observed_counts != EXPECTED_PARAMETER_COUNTS:
            raise ValueError(
                f"Puzzle-Set fold {fold} checkpoint parameter counts changed: "
                f"{observed_counts}"
            )

        tic2a_fold = tic2a[fold]
        held_puzzle = f"P{fold + 1:02d}"
        if (
            tic2a_fold.row.get("outer_fold") != fold
            or tic2a_fold.row.get("held_puzzle") != held_puzzle
            or tic2a_fold.ridge_parameter_counts
            != {
                "predictive_parameter_count": 84,
                "stored_fitted_scalar_count": 166,
            }
        ):
            raise ValueError(f"Puzzle-Set fold {fold} TIC2A identity or count changed")
        _require_source_path(
            tic2a_fold.model_path,
            directory=False,
            label="TIC2A feature41 model",
        )
        if tic2a_fold.model_path.name != f"tic2a_corrected_models_fold{fold}.json":
            raise ValueError(f"Puzzle-Set fold {fold} TIC2A model path changed")

        sources = {
            "v13_point_checkpoint": _source_record(
                "v13_point_checkpoint", path=v13_checkpoint, outer_fold=fold
            ),
            "v14_encoder_checkpoint": _source_record(
                "v14_encoder_checkpoint", path=v14_checkpoint, outer_fold=fold
            ),
            "v8_meanaligned_checkpoint": _source_record(
                "v8_meanaligned_checkpoint", path=v8_checkpoint, outer_fold=fold
            ),
            "tic2a_feature41_model_artifact": _source_record(
                "tic2a_feature41_model_artifact",
                path=tic2a_fold.model_path,
                outer_fold=fold,
            ),
            **global_sources,
        }
        if set(sources) != set(FROZEN_INPUT_SOURCE_SPEC) or any(
            source_id in FOLD_SCOPED_INPUT_SOURCES
            and sources[source_id]["outer_fold"] != fold
            for source_id in sources
        ):
            raise RuntimeError(f"Puzzle-Set fold {fold} source universe changed")
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": held_puzzle,
                "seed": 0,
                "sources": sources,
            }
        )

    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "status": SOURCE_MANIFEST_STATUS,
        "contract_id": EXPECTED_CONTRACT_ID,
        "binding_status": SOURCE_BINDING_STATUS,
        "folds": rows,
    }
    _atomic_write_manifest(bound.output_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project the exact Puzzle-Set P1M1 seven-source manifest"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest = project_source_manifest(repo_root)
    authority = load_projection_authority(repo_root)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "binding_status": manifest["binding_status"],
                "folds": len(manifest["folds"]),
                "output": str(authority.output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
