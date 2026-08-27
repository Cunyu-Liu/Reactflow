from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.reactflow_delta.project_post_v14_branch5_safe_sources import (
    EXPECTED_CHECKPOINT_COUNTS,
    PENDING_SOURCE_MANIFEST_STATUS,
    PROJECTION_PHASE,
    assert_projection_cli_authority,
    inspect_checkpoint_pair,
    project_safe_source_manifest,
)
from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    FROZEN_RUNTIME_PATHS,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    _load_source_registry,
)


def _terminal_active_authority() -> dict[str, Any]:
    return {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "parent_state": dict(EXPECTED_PARENT_STATE),
        "runnable_phases": [PROJECTION_PHASE],
        "authority": {
            "current_phase": PROJECTION_PHASE,
            "v13_checkpoint_dir": str(FROZEN_RUNTIME_PATHS["v13_checkpoint_dir"]),
            "v14_checkpoint_dir": str(FROZEN_RUNTIME_PATHS["v14_checkpoint_dir"]),
            "source_manifest_path": str(FROZEN_RUNTIME_PATHS["source_manifest_path"]),
            "source_manifest_status": PENDING_SOURCE_MANIFEST_STATUS,
        },
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }


def test_projector_cli_is_bound_to_frozen_active_paths(tmp_path: Path) -> None:
    active_contract = tmp_path / "active.yaml"
    active_contract.write_text(
        yaml.safe_dump(_terminal_active_authority()), encoding="utf-8"
    )
    assert_projection_cli_authority(
        active_contract,
        v13_checkpoint_dir=FROZEN_RUNTIME_PATHS["v13_checkpoint_dir"],
        v14_checkpoint_dir=FROZEN_RUNTIME_PATHS["v14_checkpoint_dir"],
        output_path=FROZEN_RUNTIME_PATHS["source_manifest_path"],
    )

    with pytest.raises(RuntimeError, match="CLI v14_checkpoint_dir differs"):
        assert_projection_cli_authority(
            active_contract,
            v13_checkpoint_dir=FROZEN_RUNTIME_PATHS["v13_checkpoint_dir"],
            v14_checkpoint_dir=(tmp_path / "v14").resolve(),
            output_path=FROZEN_RUNTIME_PATHS["source_manifest_path"],
        )

    active = _terminal_active_authority()
    active["authority"]["source_manifest_path"] = "/mnt/cunyuliu/wrong.json"
    active_contract.write_text(yaml.safe_dump(active), encoding="utf-8")
    with pytest.raises(RuntimeError, match="active authority source_manifest_path"):
        assert_projection_cli_authority(
            active_contract,
            v13_checkpoint_dir=FROZEN_RUNTIME_PATHS["v13_checkpoint_dir"],
            v14_checkpoint_dir=FROZEN_RUNTIME_PATHS["v14_checkpoint_dir"],
            output_path=FROZEN_RUNTIME_PATHS["source_manifest_path"],
        )

    active = _terminal_active_authority()
    active["authority"]["source_manifest_status"] = SOURCE_MANIFEST_STATUS
    active_contract.write_text(yaml.safe_dump(active), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pending manifest status"):
        assert_projection_cli_authority(
            active_contract,
            v13_checkpoint_dir=FROZEN_RUNTIME_PATHS["v13_checkpoint_dir"],
            v14_checkpoint_dir=FROZEN_RUNTIME_PATHS["v14_checkpoint_dir"],
            output_path=FROZEN_RUNTIME_PATHS["source_manifest_path"],
        )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    output = (tmp_path / "binding" / "safe.json").resolve()
    active_contract = tmp_path / "active.yaml"
    active_contract.write_text(
        yaml.safe_dump(
            {
                "parent_state": dict(EXPECTED_PARENT_STATE),
                "training_allowed": False,
                "candidate_model_training_allowed": False,
                "held_score_read_allowed": False,
                "partial_fold_score_read_allowed": False,
                "new_external_outcome_access_allowed": False,
                "branch_5_route_probe_specification": {
                    "future_runtime_authority": {
                        "source_manifest_path": str(output),
                        "source_manifest_status": SOURCE_MANIFEST_STATUS,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    v13_dir = (tmp_path / "v13").resolve()
    v14_dir = (tmp_path / "v14").resolve()
    v13_dir.mkdir()
    v14_dir.mkdir()
    for fold in range(20):
        (v13_dir / f"v13_candidate_point_fold{fold}_seed0.pt").touch()
        (v14_dir / f"v14_candidate_point_fold{fold}_seed0.pt").touch()
    return {
        "active_contract": active_contract,
        "v13_checkpoint_dir": v13_dir,
        "v14_checkpoint_dir": v14_dir,
        "output_path": output,
    }


def _fake_inspector(**_paths: Path) -> dict[str, int]:
    return dict(EXPECTED_CHECKPOINT_COUNTS)


def test_projector_writes_exact_runtime_compatible_twenty_fold_manifest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    calls: list[tuple[str, str]] = []

    def inspector(*, v13_checkpoint: Path, v14_checkpoint: Path) -> dict[str, int]:
        calls.append((v13_checkpoint.name, v14_checkpoint.name))
        return dict(EXPECTED_CHECKPOINT_COUNTS)

    manifest = project_safe_source_manifest(
        **paths,
        checkpoint_inspector=inspector,
    )
    assert set(manifest) == {"schema_version", "status", "parent_state", "folds"}
    assert manifest["schema_version"] == SOURCE_MANIFEST_SCHEMA
    assert manifest["status"] == SOURCE_MANIFEST_STATUS
    assert manifest["parent_state"] == EXPECTED_PARENT_STATE
    assert len(manifest["folds"]) == len(calls) == 20
    assert calls[0] == (
        "v13_candidate_point_fold0_seed0.pt",
        "v14_candidate_point_fold0_seed0.pt",
    )
    assert calls[-1] == (
        "v13_candidate_point_fold19_seed0.pt",
        "v14_candidate_point_fold19_seed0.pt",
    )

    row = manifest["folds"][7]
    assert set(row) == {
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
    assert row == {
        "outer_fold": 7,
        "held_puzzle": "P08",
        "seed": 0,
        "v13_source_phase": "V13M3",
        "v13_candidate_checkpoint": str(
            paths["v13_checkpoint_dir"] / "v13_candidate_point_fold7_seed0.pt"
        ),
        "v14_source_phase": "V14M3",
        "v14_arm": "CANDIDATE",
        "v14_candidate_checkpoint": str(
            paths["v14_checkpoint_dir"] / "v14_candidate_point_fold7_seed0.pt"
        ),
        "held_score_closed_at_projection": True,
        "external_outcome_accessed": False,
    }
    assert json.loads(paths["output_path"].read_text(encoding="utf-8")) == manifest
    assert sorted(_load_source_registry(paths["output_path"])) == list(range(20))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda active: active["parent_state"].__setitem__(
            "post_v14_first_matching_branch_id", "4"
        ),
        lambda active: active["parent_state"].__setitem__("v14m4_path_allowed", True),
        lambda active: active.__setitem__("training_allowed", "OPEN"),
        lambda active: active.__setitem__("held_score_read_allowed", True),
        lambda active: active.__setitem__("new_external_outcome_access_allowed", True),
    ],
)
def test_projector_rejects_nonterminal_or_open_outcome_authority(
    tmp_path: Path, mutation
) -> None:
    paths = _fixture(tmp_path)
    active = yaml.safe_load(paths["active_contract"].read_text(encoding="utf-8"))
    mutation(active)
    paths["active_contract"].write_text(yaml.safe_dump(active), encoding="utf-8")
    with pytest.raises(RuntimeError):
        project_safe_source_manifest(
            **paths,
            checkpoint_inspector=_fake_inspector,
        )
    assert not paths["output_path"].exists()


def test_projector_requires_exact_bound_output_and_never_overwrites(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    wrong = (tmp_path / "different.json").resolve()
    with pytest.raises(RuntimeError, match="differs"):
        project_safe_source_manifest(
            **{**paths, "output_path": wrong},
            checkpoint_inspector=_fake_inspector,
        )
    assert not wrong.exists()

    paths["output_path"].parent.mkdir(parents=True)
    paths["output_path"].write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        project_safe_source_manifest(
            **paths,
            checkpoint_inspector=_fake_inspector,
        )
    assert paths["output_path"].read_text(encoding="utf-8") == "existing"


def test_projector_rejects_missing_checkpoint_or_changed_parameter_count(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    missing = paths["v14_checkpoint_dir"] / "v14_candidate_point_fold3_seed0.pt"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="fold 3"):
        project_safe_source_manifest(
            **paths,
            checkpoint_inspector=_fake_inspector,
        )
    assert not paths["output_path"].exists()

    missing.touch()

    def wrong_count(**_paths: Path) -> dict[str, int]:
        value = dict(EXPECTED_CHECKPOINT_COUNTS)
        value["v14_encoder_parameters"] += 1
        return value

    with pytest.raises(ValueError, match="parameter count changed"):
        project_safe_source_manifest(
            **paths,
            checkpoint_inspector=wrong_count,
        )
    assert not paths["output_path"].exists()


def test_default_inspector_uses_weights_only_and_strict_architectures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.reactflow_delta.project_post_v14_branch5_safe_sources as projector

    events: list[tuple[Any, ...]] = []

    class Parameter:
        def __init__(self, count: int) -> None:
            self.count = count

        def numel(self) -> int:
            return self.count

    class FakeV13:
        def __init__(self, *, second_pass_mode: str) -> None:
            self.second_pass_mode = second_pass_mode
            events.append(("v13_init", second_pass_mode))

        def load_state_dict(self, state: dict, *, strict: bool) -> None:
            events.append(("v13_load", state, strict))

        def parameters(self):
            return iter([Parameter(projector.V13_POINT_PARAMETERS)])

    class FakeV14:
        def load_state_dict(self, state: dict, *, strict: bool) -> None:
            events.append(("v14_load", state, strict))

        def parameters(self):
            return iter([Parameter(projector.V14_TOTAL_PARAMETERS)])

    def fake_load(path: Path, **kwargs: Any) -> dict[str, str]:
        events.append(("torch_load", path.name, kwargs))
        return {"checkpoint": path.name}

    monkeypatch.setattr(projector, "V13PointModel", FakeV13)
    monkeypatch.setattr(projector, "V14PointModel", FakeV14)
    monkeypatch.setattr(projector.torch, "load", fake_load)
    monkeypatch.setattr(
        projector,
        "assert_v14_parameter_contract",
        lambda model: events.append(("v14_contract", model)),
    )
    monkeypatch.setattr(
        projector,
        "v14_encoder_parameters",
        lambda _model: iter([Parameter(projector.V14_ENCODER_PARAMETERS)]),
    )

    counts = inspect_checkpoint_pair(
        v13_checkpoint=tmp_path / "v13.pt",
        v14_checkpoint=tmp_path / "v14.pt",
    )
    assert counts == EXPECTED_CHECKPOINT_COUNTS
    assert ("v13_load", {"checkpoint": "v13.pt"}, True) in events
    assert ("v14_load", {"checkpoint": "v14.pt"}, True) in events
    load_events = [event for event in events if event[0] == "torch_load"]
    assert load_events == [
        (
            "torch_load",
            "v13.pt",
            {"map_location": "cpu", "weights_only": True},
        ),
        (
            "torch_load",
            "v14.pt",
            {"map_location": "cpu", "weights_only": True},
        ),
    ]


def test_projector_entrypoint_reads_only_repo_active_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.reactflow_delta.project_post_v14_branch5_safe_sources as projector

    repo_root = tmp_path / "repo"
    active_contract = repo_root / "configs/reactflow_delta/active_contract.yaml"
    active_contract.parent.mkdir(parents=True)
    active_contract.write_text("project_task_id: fixed\n", encoding="utf-8")
    v13_dir = tmp_path / "v13"
    v14_dir = tmp_path / "v14"
    v13_dir.mkdir()
    v14_dir.mkdir()
    output = tmp_path / "manifest.json"
    observed: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        projector,
        "parse_args",
        lambda: SimpleNamespace(
            repo_root=repo_root,
            v13_checkpoint_dir=v13_dir,
            v14_checkpoint_dir=v14_dir,
            output=output,
        ),
    )

    def fake_assert(path: Path, **_kwargs: Path) -> dict[str, Any]:
        observed.append(("assert", path))
        return {}

    def fake_project(*, active_contract: Path, **_kwargs: Path) -> dict[str, Any]:
        observed.append(("project", active_contract))
        return {"status": SOURCE_MANIFEST_STATUS, "folds": [{}] * 20}

    monkeypatch.setattr(projector, "assert_projection_cli_authority", fake_assert)
    monkeypatch.setattr(projector, "project_safe_source_manifest", fake_project)
    projector.main()

    expected = active_contract.resolve()
    assert observed == [("assert", expected), ("project", expected)]
