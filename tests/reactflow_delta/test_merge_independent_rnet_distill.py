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


def _write_prediction(
    path: Path,
    *,
    fold: int,
    seed: int = 0,
    keys: tuple[str, ...] | None = None,
    include_target: bool = False,
) -> None:
    keys = keys or (f"P{fold}:key",)
    n_rows = len(keys)
    arrays: dict[str, np.ndarray] = {}
    for name in EXPECTED_PREDICTION_FIELDS:
        if name == "schema_version":
            arrays[name] = np.asarray(PREDICTION_SCHEMA)
        elif name in {"keys", "biological_scoring_key"}:
            arrays[name] = np.asarray(keys, dtype=object)
        elif name == "registered_status":
            arrays[name] = np.full(n_rows, "covered", dtype=object)
        elif name == "outer_fold":
            arrays[name] = np.full(n_rows, fold, dtype=np.int64)
        elif name == "seed":
            arrays[name] = np.full(n_rows, seed, dtype=np.int64)
        elif name.endswith(("weights", "locations", "scales")):
            arrays[name] = np.ones((n_rows, 2), dtype=np.float64)
        else:
            arrays[name] = np.ones(n_rows, dtype=np.float64)
    if include_target:
        arrays["target"] = np.ones(n_rows, dtype=np.float64)
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
    seed: int = 0,
    keys: tuple[str, ...] | None = None,
    held_puzzle: str | None = None,
    experiment_id: str | None = None,
    git_commit: str = "a" * 40,
) -> None:
    keys = keys or (f"P{fold}:key",)
    paths = merger._expected_paths(input_dir, fold, seed)
    _write_prediction(paths["prediction"], fold=fold, seed=seed, keys=keys)
    for name, path in paths.items():
        if name not in {"result", "prediction"}:
            path.write_bytes(b"checkpoint")
    point_epochs, calibration_epochs = merger.EXPECTED_SCHEDULE[phase]
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
        "held_puzzle": held_puzzle or f"P{fold + 1:02d}",
        "seed": seed,
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
        "n_registered_prediction_rows": len(keys),
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


def _write_rnd6p_universe(input_dir: Path, *, git_commit: str = "a" * 40) -> None:
    for seed in range(5):
        for fold in range(20):
            _write_fold(
                input_dir,
                phase="RND6P",
                fold=fold,
                seed=seed,
                keys=(f"P{fold}:a", f"P{fold}:b"),
                git_commit=git_commit,
            )


def _prediction_keys(path: str | Path) -> tuple[str, ...]:
    with np.load(path, allow_pickle=True) as handle:
        return tuple(map(str, handle["keys"]))


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


def test_rnd6p_complete_100_fold_seed_merge_allows_same_keys_across_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd6_formal_seeds0_4"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_rnd6p_universe(input_dir)

    merged = merger.merge_folds(input_dir, "RND6P")

    assert merged["status"] == "RND6P_COMPLETE_UNSCORED_FORMAL_MERGE_PASS"
    assert len(merged["folds"]) == 100
    assert {
        (int(row["outer_fold"]), int(row["seed"])) for row in merged["folds"]
    } == {(fold, seed) for fold in range(20) for seed in range(5)}
    assert {row["git_commit"] for row in merged["folds"]} == {"a" * 40}
    for fold in range(20):
        key_orders = {
            _prediction_keys(row["prediction_artifact"])
            for row in merged["folds"]
            if int(row["outer_fold"]) == fold
        }
        assert key_orders == {(f"P{fold}:a", f"P{fold}:b")}


def test_rnd6p_merge_rejects_missing_and_unexpected_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd6_formal_seeds0_4"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_rnd6p_universe(input_dir)
    missing = merger._expected_paths(input_dir, 19, 4)["result"]
    original = missing.read_bytes()
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="missing fold result"):
        merger.merge_folds(input_dir, "RND6P")

    missing.write_bytes(original)
    (input_dir / "rnet_distill_fold_result_fold20_seed0.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unexpected independent RNet artifacts"):
        merger.merge_folds(input_dir, "RND6P")

    (input_dir / "rnet_distill_fold_result_fold20_seed0.json").unlink()
    changed_path = merger._expected_paths(input_dir, 0, 4)["result"]
    changed = json.loads(changed_path.read_text(encoding="utf-8"))
    changed["git_commit"] = "b" * 40
    changed_path.write_text(
        json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="git_commit differs across fold-seed runs"):
        merger.merge_folds(input_dir, "RND6P")


def test_rnd6p_merge_rejects_duplicate_keys_within_one_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd6_formal_seeds0_4"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_rnd6p_universe(input_dir)
    duplicate_path = merger._expected_paths(input_dir, 1, 0)["prediction"]
    _write_prediction(
        duplicate_path,
        fold=1,
        seed=0,
        keys=("P0:a", "P0:b"),
    )

    with pytest.raises(RuntimeError, match="repeat across folds within seed 0"):
        merger.merge_folds(input_dir, "RND6P")


@pytest.mark.parametrize("drift", ("held_puzzle", "key_order", "row_count"))
def test_rnd6p_merge_rejects_fold_identity_drift_across_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    input_dir = tmp_path / "rnd6_formal_seeds0_4"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    _write_rnd6p_universe(input_dir)
    paths = merger._expected_paths(input_dir, 0, 1)
    row = json.loads(paths["result"].read_text(encoding="utf-8"))
    if drift == "held_puzzle":
        row["held_puzzle"] = "PX"
    elif drift == "key_order":
        _write_prediction(
            paths["prediction"],
            fold=0,
            seed=1,
            keys=("P0:b", "P0:a"),
        )
    else:
        row["n_registered_prediction_rows"] = 1
        _write_prediction(
            paths["prediction"],
            fold=0,
            seed=1,
            keys=("P0:a",),
        )
    paths["result"].write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        RuntimeError, match="held puzzle, key order, or row count differs across seeds"
    ):
        merger.merge_folds(input_dir, "RND6P")


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

    with pytest.raises(RuntimeError, match="git_commit differs across fold-seed runs"):
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


def test_existing_merge_validation_is_exact_and_never_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    for fold in (0, 1):
        _write_fold(input_dir, phase="RND2", fold=fold)
    monkeypatch.setattr(merger, "assert_run_authority", lambda *_: None)
    monkeypatch.setattr(merger, "validate_merge_cli_binding", lambda *_: {})
    out_json = input_dir / merger.MERGE_FILENAME
    args = [
        "--repo-root",
        str(tmp_path),
        "--input-dir",
        str(input_dir),
        "--phase",
        "RND2",
        "--out-json",
        str(out_json),
    ]

    assert merger.main(args) == 0
    original = out_json.read_bytes()
    assert merger.main([*args, "--validate-existing"]) == 0
    assert out_json.read_bytes() == original

    changed = json.loads(out_json.read_text(encoding="utf-8"))
    changed["status"] = "STALE"
    out_json.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    stale_bytes = out_json.read_bytes()
    with pytest.raises(RuntimeError, match="differs from the exact fold artifacts"):
        merger.main([*args, "--validate-existing"])
    assert out_json.read_bytes() == stale_bytes


def test_existing_truncated_merge_is_rejected_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    for fold in (0, 1):
        _write_fold(input_dir, phase="RND2", fold=fold)
    monkeypatch.setattr(merger, "assert_run_authority", lambda *_: None)
    monkeypatch.setattr(merger, "validate_merge_cli_binding", lambda *_: {})
    out_json = input_dir / merger.MERGE_FILENAME
    out_json.write_text('{"schema_version":', encoding="utf-8")
    original = out_json.read_bytes()

    with pytest.raises(json.JSONDecodeError):
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
                "--validate-existing",
            ]
        )

    assert out_json.read_bytes() == original


def test_atomic_merge_write_failure_leaves_no_canonical_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "rnd2"
    input_dir.mkdir()
    pretrain_dir = tmp_path / "pretrain"
    _write_pretrain_files(pretrain_dir)
    monkeypatch.setattr(merger, "PRETRAIN_DIR", pretrain_dir)
    for fold in (0, 1):
        _write_fold(input_dir, phase="RND2", fold=fold)
    monkeypatch.setattr(merger, "assert_run_authority", lambda *_: None)
    monkeypatch.setattr(merger, "validate_merge_cli_binding", lambda *_: {})
    out_json = input_dir / merger.MERGE_FILENAME
    args = [
        "--repo-root",
        str(tmp_path),
        "--input-dir",
        str(input_dir),
        "--phase",
        "RND2",
        "--out-json",
        str(out_json),
    ]
    original_write_text = Path.write_text

    def fail_temporary_write(
        path: Path, data: str, *positional: object, **keywords: object
    ) -> int:
        if path.name.startswith(f".{merger.MERGE_FILENAME}."):
            original_write_text(path, data[:16], encoding="utf-8")
            raise OSError("injected merge write failure")
        return original_write_text(path, data, *positional, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_temporary_write)
        with pytest.raises(OSError, match="injected merge write failure"):
            merger.main(args)

    assert not out_json.exists()
    assert not list(input_dir.glob(f".{merger.MERGE_FILENAME}.*.tmp"))
    assert merger.main(args) == 0
    assert out_json.is_file()


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


def test_rnd6p_merger_binds_exact_formal_input_and_output(tmp_path: Path) -> None:
    canonical_input = tmp_path / "artifacts/rnd6_formal_seeds0_4"
    authority = {
        "current_phase": "RND6P",
        "m2_csv_path": str(tmp_path / "data/m2.csv"),
        "pretraining_dir": str(tmp_path / "artifacts/rnd1"),
        "historical_v8_dir": str(tmp_path / "artifacts/v8"),
        "historical_v10_dir": str(tmp_path / "artifacts/v10"),
        "tic2a_merged_registry_path": str(tmp_path / "artifacts/tic2a.json"),
        "unconstrained_feature_cache_path": str(tmp_path / "artifacts/u.h5"),
        "constrained_feature_cache_path": str(tmp_path / "artifacts/c.h5"),
        "formal_prediction_dir": str(canonical_input),
    }
    active_path = tmp_path / "configs/reactflow_delta/active_contract.yaml"
    active_path.parent.mkdir(parents=True)
    active_path.write_text(
        yaml.safe_dump({"authority": authority}, sort_keys=False), encoding="utf-8"
    )
    canonical_output = canonical_input / merger.MERGE_FILENAME

    binding = merger.validate_merge_cli_binding(
        tmp_path, "RND6P", canonical_input, canonical_output
    )

    assert binding == {
        "phase": "RND6P",
        "input_dir": str(canonical_input.resolve()),
        "out_json": str(canonical_output.resolve()),
    }
