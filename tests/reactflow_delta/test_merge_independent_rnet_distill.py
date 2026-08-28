from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def _write_fold(input_dir: Path, *, phase: str, fold: int) -> None:
    paths = merger._expected_paths(input_dir, fold, 0)
    _write_prediction(paths["prediction"], fold=fold)
    for name, path in paths.items():
        if name not in {"result", "prediction"}:
            path.write_bytes(b"checkpoint")
    point_epochs, calibration_epochs = (3, 3) if phase == "RND2" else (40, 40)
    row = {
        "schema_version": FOLD_SCHEMA,
        "experiment_id": f"{phase}_TEST",
        "phase": phase,
        "evidence_status": EVIDENCE_STATUS[phase],
        "metric_eligibility": EVIDENCE_STATUS[phase],
        "started_at_utc": "2026-08-28T00:00:00+00:00",
        "finished_at_utc": "2026-08-28T00:01:00+00:00",
        "git_commit": "a" * 40,
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


def test_rnd2_complete_target_free_merge_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    for fold in (0, 1):
        _write_fold(input_dir, phase="RND2", fold=fold)
    merged = merger.merge_folds(input_dir, "RND2")
    assert frozenset(merged) == merger.EXPECTED_MERGED_FIELDS
    assert merged["status"] == merger.STATUS["RND2"]
    assert [row["outer_fold"] for row in merged["folds"]] == [0, 1]
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
