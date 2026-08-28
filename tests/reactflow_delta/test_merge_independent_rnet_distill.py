from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.reactflow_delta.merge_independent_rnet_distill as merger
from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EVIDENCE_STATUS,
    EXPECTED_PREDICTION_FIELDS,
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
    PRETRAIN_FILENAMES,
)


def _write_prediction(path: Path, *, fold: int, include_target: bool = False) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name in EXPECTED_PREDICTION_FIELDS:
        if name == "schema_version":
            arrays[name] = np.asarray(PREDICTION_SCHEMA)
        elif name in {"keys", "biological_scoring_key"}:
            arrays[name] = np.asarray([f"P{fold}:key"], dtype=object)
        elif name == "registered_status":
            arrays[name] = np.asarray(["covered"], dtype=object)
        elif name == "outer_fold":
            arrays[name] = np.asarray([fold], dtype=np.int64)
        elif name == "seed":
            arrays[name] = np.asarray([0], dtype=np.int64)
        elif name.endswith(("weights", "locations", "scales")):
            arrays[name] = np.ones((1, 2), dtype=np.float64)
        else:
            arrays[name] = np.ones(1, dtype=np.float64)
    if include_target:
        arrays["target"] = np.ones(1, dtype=np.float64)
    np.savez_compressed(path, **arrays)


def _write_pretrain_files(pretrain_dir: Path) -> None:
    pretrain_dir.mkdir(parents=True)
    for name in PRETRAIN_FILENAMES.values():
        (pretrain_dir / name).write_bytes(b"frozen")


def _write_fold(
    input_dir: Path,
    *,
    phase: str,
    fold: int,
    experiment_id: str | None = None,
    git_commit: str = "a" * 40,
) -> None:
    paths = merger._expected_paths(input_dir, fold, 0)
    _write_prediction(paths["prediction"], fold=fold)
    for name, path in paths.items():
        if name not in {"result", "prediction"}:
            path.write_bytes(b"checkpoint")
    point_epochs, calibration_epochs = (3, 3) if phase == "RND2" else (40, 40)
    row = {
        "schema_version": FOLD_SCHEMA,
        "experiment_id": (
            experiment_id or merger.EXPECTED_EXPERIMENT_ID[phase]
        ),
        "phase": phase,
        "evidence_status": EVIDENCE_STATUS[phase],
        "metric_eligibility": EVIDENCE_STATUS[phase],
        "started_at_utc": "2026-08-28T00:00:00+00:00",
        "finished_at_utc": "2026-08-28T00:01:00+00:00",
        "git_commit": git_commit,
        "command": ["python", "-m", "runner"],
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "seed": 0,
        "point_epochs": point_epochs,
        "calibration_epochs": calibration_epochs,
        "training_device": "cuda:0",
        "gpu_name": "test cuda",
        "contract_paths": {
            "active": "/home/project/active_contract.yaml",
            "machine": "/home/project/contract.yaml",
            "ledger": "/home/project/ledger.yaml",
        },
        "split": {
            "name": "split_v4_lopo_puzzle",
            "seed": 20260813,
            "fold_universe": list(range(20)),
        },
        "pretraining_checkpoints": {
            name: str((merger.PRETRAIN_DIR / filename).resolve())
            for name, filename in PRETRAIN_FILENAMES.items()
        },
        "point_checkpoints": {
            "candidate": str(paths["candidate_point"].resolve()),
            "null": str(paths["null_point"].resolve()),
        },
        "residual_checkpoints": {
            "feature41": str(paths["feature41_residual"].resolve()),
            "candidate": str(paths["candidate_residual"].resolve()),
            "null": str(paths["null_residual"].resolve()),
        },
        "prediction_artifact": str(paths["prediction"].resolve()),
        "training_histories": {
            "candidate_point": [1.0] * point_epochs,
            "null_point": [1.0] * point_epochs,
            "feature41_residual": [1.0] * calibration_epochs,
            "candidate_residual": [1.0] * calibration_epochs,
            "null_residual": [1.0] * calibration_epochs,
        },
        "n_train_cells": 1,
        "n_registered_prediction_rows": 1,
        "feature41_replay_max_abs_difference": 0.0,
        "point_parameter_counts": {"candidate": 5_117_105, "null": 5_117_105},
        "residual_parameter_counts": {
            "feature41": 63_748,
            "candidate": 63_748,
            "null": 63_748,
        },
        "invariants": dict(merger.EXPECTED_INVARIANTS),
        "exit_code": 0,
    }
    paths["result"].write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("phase", "expected_folds"),
    (("RND2", (0, 1)), ("RND3", tuple(range(20)))),
)
def test_complete_target_free_merge_passes_with_exact_experiment_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_folds: tuple[int, ...],
) -> None:
    input_dir = tmp_path / phase.lower()
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    for fold in expected_folds:
        _write_fold(input_dir, phase=phase, fold=fold)
    merged = merger.merge_folds(input_dir, phase)
    assert frozenset(merged) == merger.EXPECTED_MERGED_FIELDS
    assert merged["status"] == merger.STATUS[phase]
    assert [row["outer_fold"] for row in merged["folds"]] == list(expected_folds)
    assert {row["experiment_id"] for row in merged["folds"]} == {
        merger.EXPECTED_EXPERIMENT_ID[phase]
    }
    assert merged["merge_integrity"] == merger.MERGE_INTEGRITY


def test_merge_rejects_missing_or_unexpected_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_fold(input_dir, phase="RND2", fold=0)
    with pytest.raises(FileNotFoundError, match="missing fold result"):
        merger.merge_folds(input_dir, "RND2")
    (input_dir / "rnet_distill_fold_result_fold20_seed0.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unexpected independent RNet artifacts"):
        merger.merge_folds(input_dir, "RND2")


def test_merge_rejects_prediction_target_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    for fold in (0, 1):
        _write_fold(input_dir, phase="RND2", fold=fold)
    _write_prediction(
        merger._expected_paths(input_dir, 0, 0)["prediction"],
        fold=0,
        include_target=True,
    )
    with pytest.raises(RuntimeError, match="prediction checks failed"):
        merger.merge_folds(input_dir, "RND2")


def test_main_rejects_mismatched_experiment_id_without_writing_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_fold(
        input_dir,
        phase="RND2",
        fold=0,
        experiment_id=merger.EXPECTED_EXPERIMENT_ID["RND3"],
    )
    _write_fold(input_dir, phase="RND2", fold=1)
    monkeypatch.setattr(merger, "assert_run_authority", lambda *_: None)
    monkeypatch.setattr(merger, "validate_merge_cli_binding", lambda *_: {})
    out_json = input_dir / merger.MERGE_FILENAME

    with pytest.raises(RuntimeError, match="fold 0 experiment_id differs"):
        merger.main(
            [
                "--repo-root",
                str(tmp_path),
                "--input-dir",
                str(input_dir),
                "--phase",
                "RND2",
                "--out-json",
                str(out_json),
            ]
        )

    assert not out_json.exists()


def test_main_rejects_mixed_git_commits_without_writing_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_fold(input_dir, phase="RND2", fold=0, git_commit="a" * 40)
    _write_fold(input_dir, phase="RND2", fold=1, git_commit="b" * 40)
    monkeypatch.setattr(merger, "assert_run_authority", lambda *_: None)
    monkeypatch.setattr(merger, "validate_merge_cli_binding", lambda *_: {})
    out_json = input_dir / merger.MERGE_FILENAME

    with pytest.raises(RuntimeError, match="git_commit differs across folds"):
        merger.main(
            [
                "--repo-root",
                str(tmp_path),
                "--input-dir",
                str(input_dir),
                "--phase",
                "RND2",
                "--out-json",
                str(out_json),
            ]
        )

    assert not out_json.exists()


@pytest.mark.parametrize("git_commit", ("", "g" * 40, "a" * 39))
def test_merge_rejects_invalid_git_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_commit: str
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_fold(input_dir, phase="RND2", fold=0, git_commit=git_commit)
    _write_fold(input_dir, phase="RND2", fold=1)

    with pytest.raises(
        RuntimeError, match="git_commit is not a 40-character hex commit"
    ):
        merger.merge_folds(input_dir, "RND2")


def test_merger_binds_input_and_output_to_active_phase(tmp_path: Path) -> None:
    canonical_input = tmp_path / "artifacts/rnd2"
    authority = {
        "current_phase": "RND2",
        "m2_csv_path": str(tmp_path / "data/m2.csv"),
        "pretraining_dir": str(tmp_path / "artifacts/rnd1"),
        "historical_v8_dir": str(tmp_path / "artifacts/v8"),
        "historical_v10_dir": str(tmp_path / "artifacts/v10"),
        "tic2a_merged_registry_path": str(tmp_path / "artifacts/tic2a.json"),
        "unconstrained_feature_cache_path": str(tmp_path / "artifacts/u.h5"),
        "constrained_feature_cache_path": str(tmp_path / "artifacts/c.h5"),
        "smoke_prediction_dir": str(canonical_input),
        "screen_prediction_dir": str(tmp_path / "artifacts/rnd3"),
    }
    active_path = tmp_path / "configs/reactflow_delta/active_contract.yaml"
    active_path.parent.mkdir(parents=True)
    active_path.write_text(
        yaml.safe_dump({"authority": authority}, sort_keys=False), encoding="utf-8"
    )
    canonical_output = canonical_input / merger.MERGE_FILENAME
    binding = merger.validate_merge_cli_binding(
        tmp_path, "RND2", canonical_input, canonical_output
    )
    assert binding["out_json"] == str(canonical_output.resolve())
    with pytest.raises(RuntimeError, match="input_dir differs"):
        merger.validate_merge_cli_binding(
            tmp_path, "RND2", tmp_path / "wrong", canonical_output
        )
    with pytest.raises(RuntimeError, match="out_json differs"):
        merger.validate_merge_cli_binding(
            tmp_path, "RND2", canonical_input, tmp_path / "wrong.json"
        )
