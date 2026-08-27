from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from scripts.reactflow_delta.preflight_puzzle_set_meta_context_sources import (
    EXPECTED_PARAMETER_COUNTS,
)
from scripts.reactflow_delta.project_puzzle_set_meta_context_sources import (
    EXPECTED_CACHE_ALIGNMENT,
    EXPECTED_PARENT_STATUS,
    EXPECTED_PROJECT_TASK,
    PENDING_SOURCE_BINDING_STATUS,
    PROJECTION_AUTHORITY_STATE,
    PROJECTION_PHASE,
    ROUTER_BRANCHES,
    load_projection_authority,
    parse_args,
    project_source_manifest,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    EXPECTED_CONTRACT_ID,
    FROZEN_INPUT_SOURCE_SPEC,
    SOURCE_BINDING_STATUS,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    SafeTIC2AFold,
    validate_source_manifest,
)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _active(common: dict[str, str], branch: str = "3") -> dict[str, Any]:
    route = ROUTER_BRANCHES[branch]
    return {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "authority": {
            "current_phase": PROJECTION_PHASE,
            "current_authority_state": PROJECTION_AUTHORITY_STATE,
            "source_binding_status": PENDING_SOURCE_BINDING_STATUS,
            **common,
        },
        "parent_state": {
            "v14_status": EXPECTED_PARENT_STATUS,
            "v14m4_path_allowed": False,
            "post_v14_first_matching_branch_id": branch,
            "post_v14_route_classification": route["classification"],
            "post_v14_route_probe_requirement": route["probe_requirement"],
            "post_v14_route_probe_status": route["probe_status"],
        },
        "runnable_phases": [PROJECTION_PHASE],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }


def _fixture(tmp_path: Path, *, branch: str = "3") -> dict[str, Any]:
    repo_root = (tmp_path / "repo").resolve()
    source_root = (tmp_path / "sources").resolve()
    v8_dir = source_root / "v8"
    v13_dir = source_root / "v13"
    v14_dir = source_root / "v14"
    tic2a_dir = source_root / "tic2a"
    for directory in (v8_dir, v13_dir, v14_dir, tic2a_dir):
        directory.mkdir(parents=True)
    models: dict[int, Path] = {}
    for fold in range(20):
        (v8_dir / f"v8_corrected_mean_fold{fold}_seed0.pt").touch()
        (v13_dir / f"v13_candidate_point_fold{fold}_seed0.pt").touch()
        (v14_dir / f"v14_candidate_point_fold{fold}_seed0.pt").touch()
        model_path = tic2a_dir / f"tic2a_corrected_models_fold{fold}.json"
        model_path.write_text("{}\n", encoding="utf-8")
        models[fold] = model_path
    registry = tic2a_dir / "tic2a_corrected_merged_unscored.json"
    registry.write_text("{}\n", encoding="utf-8")
    unconstrained_cache = source_root / "ensemble_delta_cache.h5"
    constrained_cache = source_root / "constrained_cache.h5"
    unconstrained_cache.touch()
    constrained_cache.touch()
    output = (
        tmp_path / "artifacts/source_binding/puzzle_set_source_manifest.json"
    ).resolve()
    common = {
        "source_manifest_path": str(output),
        "source_binding_status": SOURCE_BINDING_STATUS,
        "v8_checkpoint_dir": str(v8_dir),
        "v13_checkpoint_dir": str(v13_dir),
        "v14_checkpoint_dir": str(v14_dir),
        "m2_csv_path": str(source_root / "must_not_be_opened.csv"),
        "tic2a_merged_registry_path": str(registry),
        "unconstrained_feature_cache_path": str(unconstrained_cache),
        "constrained_feature_cache_path": str(constrained_cache),
        "v13_historical_bundle_path": str(source_root / "must_not_be_opened.json"),
    }
    active_common = {
        key: value
        for key, value in common.items()
        if key
        in {
            "source_manifest_path",
            "v8_checkpoint_dir",
            "v13_checkpoint_dir",
            "v14_checkpoint_dir",
            "tic2a_merged_registry_path",
            "unconstrained_feature_cache_path",
            "constrained_feature_cache_path",
        }
    }
    _write_yaml(
        repo_root / "configs/reactflow_delta/active_contract.yaml",
        _active(active_common, branch),
    )
    _write_yaml(
        repo_root / "configs/reactflow_delta/puzzle_set_meta_context_v5_amendment.yaml",
        {
            "contract_id": EXPECTED_CONTRACT_ID,
            "future_p1_runtime_authority": {
                "project_task_id": EXPECTED_PROJECT_TASK,
                "common_source_paths": common,
            },
        },
    )
    return {
        "repo_root": repo_root,
        "v8_dir": v8_dir,
        "v13_dir": v13_dir,
        "v14_dir": v14_dir,
        "registry": registry,
        "unconstrained_cache": unconstrained_cache,
        "constrained_cache": constrained_cache,
        "output": output,
        "models": models,
    }


def _fake_checkpoint_inspector(**_paths: Path) -> dict[str, int]:
    return dict(EXPECTED_PARAMETER_COUNTS)


def _fake_cache_inspector(**_paths: Path) -> dict[str, int | bool]:
    return dict(EXPECTED_CACHE_ALIGNMENT)


def _registry_loader(
    paths: dict[str, Any]
) -> Callable[[Path], dict[int, SafeTIC2AFold]]:
    def load(path: Path) -> dict[int, SafeTIC2AFold]:
        assert path == paths["registry"]
        return {
            fold: SafeTIC2AFold(
                row={"outer_fold": fold, "held_puzzle": f"P{fold + 1:02d}"},
                model_path=paths["models"][fold],
                feature41_model={},
                ridge_parameter_counts={
                    "predictive_parameter_count": 84,
                    "stored_fitted_scalar_count": 166,
                },
            )
            for fold in range(20)
        }

    return load


def _project(paths: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return project_source_manifest(
        paths["repo_root"],
        checkpoint_inspector=overrides.get(
            "checkpoint_inspector", _fake_checkpoint_inspector
        ),
        cache_inspector=overrides.get("cache_inspector", _fake_cache_inspector),
        registry_loader=overrides.get("registry_loader", _registry_loader(paths)),
    )


@pytest.mark.parametrize("branch", ["3", "4", "5"])
def test_projector_writes_exact_validator_compatible_twenty_fold_manifest(
    tmp_path: Path, branch: str
) -> None:
    paths = _fixture(tmp_path, branch=branch)
    manifest = _project(paths)
    rows = validate_source_manifest(paths["output"])

    assert set(manifest) == {
        "schema_version",
        "status",
        "contract_id",
        "binding_status",
        "folds",
    }
    assert manifest["schema_version"] == SOURCE_MANIFEST_SCHEMA
    assert manifest["status"] == SOURCE_MANIFEST_STATUS
    assert manifest["contract_id"] == EXPECTED_CONTRACT_ID
    assert manifest["binding_status"] == SOURCE_BINDING_STATUS
    assert tuple(sorted(rows)) == tuple(range(20))
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == manifest

    row = rows[7]
    assert row["outer_fold"] == 7
    assert row["held_puzzle"] == "P08"
    assert row["seed"] == 0
    assert set(row["sources"]) == set(FROZEN_INPUT_SOURCE_SPEC)
    for source_id, source in row["sources"].items():
        expected = FROZEN_INPUT_SOURCE_SPEC[source_id]
        assert source["role"] == expected["role"]
        assert (
            source["used_in_candidate_prediction"]
            is expected["used_in_candidate_prediction"]
        )
        assert source["seed"] == expected["seed"]
        assert (
            source["realized_parameter_count"] == expected["realized_parameter_count"]
        )
        assert source["trainable_in_p1"] is False
        expected_fold = (
            7
            if source_id.endswith("checkpoint")
            or source_id == ("tic2a_feature41_model_artifact")
            else None
        )
        assert source["outer_fold"] == expected_fold


def test_entrypoint_uses_only_canonical_repo_active_pointer(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    arbitrary = tmp_path / "arbitrary_active.yaml"
    arbitrary.write_text(
        yaml.safe_dump(_active({})),
        encoding="utf-8",
    )
    active_path = paths["repo_root"] / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    active["authority"]["current_phase"] = "P1M2"
    _write_yaml(active_path, active)

    with pytest.raises(RuntimeError, match="closed outside P1M1"):
        load_projection_authority(paths["repo_root"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--repo-root",
                str(paths["repo_root"]),
                "--active-contract",
                str(arbitrary),
            ]
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda active: active["authority"].__setitem__(
            "current_authority_state", "TRAINING_OPEN"
        ),
        lambda active: active.__setitem__("project_task_id", "wrong_task"),
        lambda active: active.__setitem__("runnable_phases", ["P1M1", "P1M2"]),
        lambda active: active.__setitem__("training_allowed", "OPEN"),
        lambda active: active.__setitem__("candidate_model_training_allowed", True),
        lambda active: active.__setitem__("held_score_read_allowed", True),
        lambda active: active.__setitem__("partial_fold_score_read_allowed", True),
        lambda active: active.__setitem__("new_external_outcome_access_allowed", True),
        lambda active: active["parent_state"].__setitem__(
            "v14_status", "TERMINAL_V14M4_FAIL"
        ),
        lambda active: active["parent_state"].__setitem__("v14m4_path_allowed", True),
        lambda active: active["parent_state"].__setitem__(
            "post_v14_route_probe_status", "NOT_RUN"
        ),
        lambda active: active["parent_state"].__setitem__(
            "post_v14_first_matching_branch_id", "6"
        ),
        lambda active: active["authority"].__setitem__(
            "source_binding_status", SOURCE_BINDING_STATUS
        ),
    ],
)
def test_projector_rejects_nonprojection_or_ineligible_parent_authority(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    paths = _fixture(tmp_path, branch="5")
    active_path = paths["repo_root"] / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    mutate(active)
    _write_yaml(active_path, active)
    with pytest.raises(RuntimeError):
        _project(paths)
    assert not paths["output"].exists()


def test_projector_rejects_active_path_outside_frozen_future_authority(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    active_path = paths["repo_root"] / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    active["authority"]["v14_checkpoint_dir"] = str(tmp_path / "different_v14")
    _write_yaml(active_path, active)
    with pytest.raises(RuntimeError, match="v14_checkpoint_dir differs"):
        _project(paths)
    assert not paths["output"].exists()


def test_projector_rejects_changed_frozen_future_source_status(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    contract_path = (
        paths["repo_root"]
        / "configs/reactflow_delta/puzzle_set_meta_context_v5_amendment.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    contract["future_p1_runtime_authority"]["common_source_paths"][
        "source_binding_status"
    ] = "PENDING"
    _write_yaml(contract_path, contract)
    with pytest.raises(RuntimeError, match="source binding status changed"):
        _project(paths)
    assert not paths["output"].exists()


def test_projector_rejects_missing_fold_or_changed_realized_parameter_count(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    missing = paths["v13_dir"] / "v13_candidate_point_fold9_seed0.pt"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="V13 point checkpoint"):
        _project(paths)
    assert not paths["output"].exists()

    missing.touch()

    def wrong_count(**_paths: Path) -> dict[str, int]:
        counts = dict(EXPECTED_PARAMETER_COUNTS)
        counts["v14_encoder_checkpoint"] += 1
        return counts

    with pytest.raises(ValueError, match="parameter counts changed"):
        _project(paths, checkpoint_inspector=wrong_count)
    assert not paths["output"].exists()


def test_projector_rejects_tic2a_fold_or_model_path_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    registry = _registry_loader(paths)(paths["registry"])
    registry.pop(19)
    with pytest.raises(ValueError, match="not exact folds0-19"):
        _project(paths, registry_loader=lambda _path: registry)
    assert not paths["output"].exists()

    registry = _registry_loader(paths)(paths["registry"])
    wrong_model = paths["models"][4].with_name("wrong_name.json")
    wrong_model.touch()
    original = registry[4]
    registry[4] = SafeTIC2AFold(
        row=original.row,
        model_path=wrong_model,
        feature41_model=original.feature41_model,
        ridge_parameter_counts=original.ridge_parameter_counts,
    )
    with pytest.raises(ValueError, match="TIC2A model path changed"):
        _project(paths, registry_loader=lambda _path: registry)
    assert not paths["output"].exists()


def test_projector_ignores_wide_v10_v11_v12_score_and_result_decoys(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    decoys = [
        paths["v8_dir"] / "v10_fold_result_fold0_seed0.json",
        paths["v13_dir"] / "v11_complete_score.json",
        paths["v14_dir"] / "v12_fold_result_fold0_seed0.json",
    ]
    for decoy in decoys:
        decoy.write_text("this must never be parsed", encoding="utf-8")
    checkpoint_paths: list[Path] = []

    def inspect(**source_paths: Path) -> dict[str, int]:
        checkpoint_paths.extend(source_paths.values())
        return dict(EXPECTED_PARAMETER_COUNTS)

    _project(paths, checkpoint_inspector=inspect)
    assert len(checkpoint_paths) == 60
    assert all(path.suffix == ".pt" for path in checkpoint_paths)
    assert all(
        decoy.read_text(encoding="utf-8") == "this must never be parsed"
        for decoy in decoys
    )


def test_projector_fails_closed_without_overwrite_and_finalizes_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.reactflow_delta.project_puzzle_set_meta_context_sources as projector

    paths = _fixture(tmp_path)

    def interrupted_replace(_source: Path, _destination: Path) -> None:
        assert not paths["output"].exists()
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(projector.os, "replace", interrupted_replace)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _project(paths)
    assert not paths["output"].exists()
    assert list(paths["output"].parent.glob("*.tmp")) == []

    monkeypatch.undo()
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _project(paths)
    assert paths["output"].read_text(encoding="utf-8") == "existing\n"
