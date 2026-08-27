#!/usr/bin/env python3
"""Project the terminal, score-free checkpoint binding for branch-5.

This command is intentionally narrower than the branch-5 runner.  It can run
only after the active authority records the exact terminal V14M3 branch-5
route.  It opens the two frozen checkpoint families and writes only their
same-fold paths; it never reads parent fold results, predictions, training
histories, scientific scores, or external outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from scripts.reactflow_delta.model_rescue_v13 import (
    EXPECTED_POINT_PARAMETERS as V13_POINT_PARAMETERS,
    SECOND_PASS_EXACT,
    V13PointModel,
)
from scripts.reactflow_delta.model_rescue_v14 import (
    EXPECTED_ENCODER_PARAMETERS as V14_ENCODER_PARAMETERS,
    EXPECTED_TOTAL_PARAMETERS as V14_TOTAL_PARAMETERS,
    V14PointModel,
    assert_exact_parameter_contract as assert_v14_parameter_contract,
    encoder_parameters as v14_encoder_parameters,
)
from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_FOLDS,
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    EXPECTED_SEED,
    FROZEN_RUNTIME_PATHS,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    assert_frozen_runtime_paths,
)


DEFAULT_OUTPUT_PATH = FROZEN_RUNTIME_PATHS["source_manifest_path"]
PROJECTION_PHASE = "B5RP0"
PENDING_SOURCE_MANIFEST_STATUS = (
    "POST_V14_BRANCH5_SAFE_SOURCE_MANIFEST_PENDING_PROJECTION"
)
EXPECTED_CHECKPOINT_COUNTS = {
    "v13_point_parameters": V13_POINT_PARAMETERS,
    "v14_encoder_parameters": V14_ENCODER_PARAMETERS,
    "v14_total_parameters": V14_TOTAL_PARAMETERS,
}


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain one checkpoint state dict")
    return state


def inspect_checkpoint_pair(
    *, v13_checkpoint: Path, v14_checkpoint: Path
) -> dict[str, int]:
    """Strictly replay both frozen architectures and return parameter counts."""

    v13 = V13PointModel(second_pass_mode=SECOND_PASS_EXACT)
    v13.load_state_dict(_load_state_dict(v13_checkpoint), strict=True)
    if v13.second_pass_mode != SECOND_PASS_EXACT:
        raise RuntimeError("projected V13 checkpoint is not the exact-mutant model")

    v14 = V14PointModel()
    v14.load_state_dict(_load_state_dict(v14_checkpoint), strict=True)
    assert_v14_parameter_contract(v14)

    return {
        "v13_point_parameters": sum(
            parameter.numel() for parameter in v13.parameters()
        ),
        "v14_encoder_parameters": sum(
            parameter.numel() for parameter in v14_encoder_parameters(v14)
        ),
        "v14_total_parameters": sum(
            parameter.numel() for parameter in v14.parameters()
        ),
    }


def _load_active_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("active contract must contain one mapping")
    return value


def _assert_terminal_projection_authority(active: dict[str, Any]) -> None:
    if active.get("parent_state") != EXPECTED_PARENT_STATE:
        raise RuntimeError(
            "branch5 source projection requires the exact terminal V14M3 route"
        )
    if (
        active.get("training_allowed") is not False
        or active.get("candidate_model_training_allowed") is not False
        or active.get("held_score_read_allowed") is not False
        or active.get("partial_fold_score_read_allowed") is not False
        or active.get("new_external_outcome_access_allowed") is not False
    ):
        raise RuntimeError(
            "branch5 source projection requires training, held, partial, and external outcomes closed"
        )


def assert_projection_cli_authority(
    active_contract: Path,
    *,
    v13_checkpoint_dir: Path,
    v14_checkpoint_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind the real projector CLI to the one frozen terminal authority."""

    active = _load_active_contract(active_contract)
    _assert_terminal_projection_authority(active)
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("branch5 source projection is not the active project")
    if active.get("authority", {}).get("current_phase") != PROJECTION_PHASE:
        raise RuntimeError("branch5 source projection is closed outside B5RP0")
    if active.get("runnable_phases") != [PROJECTION_PHASE]:
        raise RuntimeError("B5RP0 must be the only runnable projection phase")
    authority = active.get("authority")
    assert_frozen_runtime_paths(
        authority,
        required_fields=(
            "v13_checkpoint_dir",
            "v14_checkpoint_dir",
            "source_manifest_path",
        ),
        cli_paths={
            "v13_checkpoint_dir": v13_checkpoint_dir,
            "v14_checkpoint_dir": v14_checkpoint_dir,
            "source_manifest_path": output_path,
        },
    )
    if authority.get("source_manifest_status") != PENDING_SOURCE_MANIFEST_STATUS:
        raise RuntimeError(
            "branch5 source projection requires the pending manifest status"
        )
    return active


def _manifest_binding(active: dict[str, Any]) -> Path:
    """Resolve the single active/future binding, falling back to the fixed path."""

    candidates: list[tuple[str, Any, Any]] = []
    authority = active.get("authority")
    if isinstance(authority, dict) and authority.get("source_manifest_path"):
        candidates.append(
            (
                "authority",
                authority.get("source_manifest_path"),
                authority.get("source_manifest_status"),
            )
        )
    future = active.get("branch_5_route_probe_specification", {}).get(
        "future_runtime_authority", {}
    )
    if isinstance(future, dict) and future.get("source_manifest_path"):
        candidates.append(
            (
                "future_runtime_authority",
                future.get("source_manifest_path"),
                future.get("source_manifest_status"),
            )
        )

    observed_paths: set[Path] = set()
    for label, raw_path, status in candidates:
        path = Path(str(raw_path))
        if not path.is_absolute() or status not in {
            PENDING_SOURCE_MANIFEST_STATUS,
            SOURCE_MANIFEST_STATUS,
        }:
            raise RuntimeError(f"branch5 {label} source-manifest binding is invalid")
        observed_paths.add(path)
    if len(observed_paths) > 1:
        raise RuntimeError(
            "branch5 active and future source-manifest bindings conflict"
        )
    return next(iter(observed_paths), DEFAULT_OUTPUT_PATH)


def _require_checkpoint_directory(path: Path, label: str) -> None:
    if not path.is_absolute() or not path.is_dir():
        raise FileNotFoundError(
            f"branch5 {label} checkpoint directory must be one existing absolute directory"
        )


def project_safe_source_manifest(
    *,
    active_contract: Path,
    v13_checkpoint_dir: Path,
    v14_checkpoint_dir: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    checkpoint_inspector: Callable[..., dict[str, int]] = inspect_checkpoint_pair,
) -> dict[str, Any]:
    """Validate all 40 checkpoints, then create the one terminal manifest."""

    active = _load_active_contract(active_contract)
    _assert_terminal_projection_authority(active)
    bound_output = _manifest_binding(active)
    if not output_path.is_absolute() or output_path != bound_output:
        raise RuntimeError(
            "branch5 output path differs from the active/future manifest binding"
        )
    if output_path.exists():
        raise FileExistsError(f"branch5 source manifest already exists: {output_path}")

    _require_checkpoint_directory(v13_checkpoint_dir, "V13M3")
    _require_checkpoint_directory(v14_checkpoint_dir, "V14M3")

    rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        v13_checkpoint = v13_checkpoint_dir / f"v13_candidate_point_fold{fold}_seed0.pt"
        v14_checkpoint = v14_checkpoint_dir / f"v14_candidate_point_fold{fold}_seed0.pt"
        for path in (v13_checkpoint, v14_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(
                    f"branch5 fold {fold} frozen checkpoint is absent: {path}"
                )
        observed_counts = checkpoint_inspector(
            v13_checkpoint=v13_checkpoint,
            v14_checkpoint=v14_checkpoint,
        )
        if observed_counts != EXPECTED_CHECKPOINT_COUNTS:
            raise ValueError(
                f"branch5 fold {fold} checkpoint architecture or parameter count changed: "
                f"{observed_counts}"
            )
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "seed": EXPECTED_SEED,
                "v13_source_phase": "V13M3",
                "v13_candidate_checkpoint": str(v13_checkpoint),
                "v14_source_phase": "V14M3",
                "v14_arm": "CANDIDATE",
                "v14_candidate_checkpoint": str(v14_checkpoint),
                "held_score_closed_at_projection": True,
                "external_outcome_accessed": False,
            }
        )

    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "status": SOURCE_MANIFEST_STATUS,
        "parent_state": dict(EXPECTED_PARENT_STATE),
        "folds": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project the terminal-safe post-V14 branch-5 source manifest"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--v13-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--v14-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    active_contract = (
        repo_root / "configs/reactflow_delta/active_contract.yaml"
    ).resolve()
    if not active_contract.is_file():
        raise FileNotFoundError(
            "branch5 projector requires the active pointer in the selected repository"
        )
    v13_checkpoint_dir = args.v13_checkpoint_dir.resolve()
    v14_checkpoint_dir = args.v14_checkpoint_dir.resolve()
    output_path = args.output.resolve()
    assert_projection_cli_authority(
        active_contract,
        v13_checkpoint_dir=v13_checkpoint_dir,
        v14_checkpoint_dir=v14_checkpoint_dir,
        output_path=output_path,
    )
    manifest = project_safe_source_manifest(
        active_contract=active_contract,
        v13_checkpoint_dir=v13_checkpoint_dir,
        v14_checkpoint_dir=v14_checkpoint_dir,
        output_path=output_path,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "folds": len(manifest["folds"]),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
