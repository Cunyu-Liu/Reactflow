from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.assemble_puzzle_set_meta_context_formal import (
    ASSEMBLY_STATUS,
    FORMAL_PREDICTION_SCHEMA,
    SCHEMA as ASSEMBLY_SCHEMA,
    _REQUIRED_MERGE_INTEGRITY_TRUE,
    assemble,
)
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import MERGED_SCHEMA
from scripts.reactflow_delta.puzzle_set_meta_context_data import PREDICTION_SCHEMA
import scripts.reactflow_delta.score_puzzle_set_meta_context_formal as formal_module
from scripts.reactflow_delta.score_puzzle_set_meta_context_formal import (
    ASSEMBLY_SCHEMA,
    ASSEMBLY_STATUS,
    EXPECTED_PHASE,
    EXPECTED_PROJECT_TASK,
    EXPECTED_SCORE_TOKEN,
    _load_prediction,
    assert_score_authority,
    score_formal,
)


def _write_prediction(path: Path, *, schema: str, fold: int, seed: int) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(schema),
        keys=np.asarray(["key0"], dtype=object),
        outer_fold=np.asarray([fold], dtype=np.int64),
        seed=np.asarray([seed], dtype=np.int64),
    )


def _context_retention_summary() -> dict[str, object]:
    return {
        "candidate_pretraining_established_all_runs": True,
        "candidate_retention_positive_all_runs": True,
        "null_pretraining_established_all_runs": True,
        "null_retention_positive_all_runs": True,
        "fold_seed_diagnostics": [],
        "selection_performed": False,
        "mutant_outcome_used": False,
        "held_puzzle_accessed": False,
    }


def _write_source_prediction(
    directory: Path,
    *,
    fold: int,
    seed: int,
    rerun_shift: float,
) -> dict[str, object]:
    key = f"key-{fold}"
    candidate_point = np.asarray([0.1 * fold + seed + rerun_shift])
    null_point = np.asarray([10.0 + 0.1 * fold + seed + rerun_shift])
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray([key], dtype=object),
        "biological_scoring_key": np.asarray([key], dtype=object),
        "outer_fold": np.asarray([fold], dtype=np.int64),
        "seed": np.asarray([seed], dtype=np.int64),
        "registered_status": np.asarray(["covered"], dtype=object),
        "feature41_point": np.asarray([0.0]),
        "parent_point": np.asarray([0.05]),
        "candidate_point": candidate_point,
        "null_point": null_point,
    }
    for name, point in (("candidate", candidate_point), ("null", null_point)):
        payload[f"{name}_weights"] = np.asarray([[0.5, 0.5]])
        payload[f"{name}_locations"] = np.stack([point, point], axis=1)
        payload[f"{name}_scales"] = np.asarray([[0.1, 0.2]])
        payload[f"{name}_expected_absolute_delta"] = np.abs(point)
    path = directory / f"source_{fold}_{seed}.npz"
    np.savez_compressed(path, **payload)
    return {
        "outer_fold": fold,
        "seed": seed,
        "prediction_artifact": str(path),
        "n_registered_prediction_rows": 1,
    }


def _complete_merge(directory: Path, *, rerun_shift: float = 0.0) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    source_rows = [
        _write_source_prediction(
            directory,
            fold=fold,
            seed=seed,
            rerun_shift=rerun_shift,
        )
        for fold in range(20)
        for seed in range(5)
    ]
    integrity = {name: True for name in _REQUIRED_MERGE_INTEGRITY_TRUE}
    integrity.update(
        {"partial_scores_inspected": False, "external_outcome_accessed": False}
    )
    return {
        "schema_version": MERGED_SCHEMA,
        "status": "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS",
        "phase": "P1M4",
        "expected_folds": list(range(20)),
        "expected_seeds": list(range(5)),
        "expected_pretraining_epochs": 200,
        "expected_point_epochs": 40,
        "expected_calibration_epochs": 40,
        "folds": source_rows,
        "context_retention_summary": _context_retention_summary(),
        "merge_integrity": integrity,
    }


def _tic2a_merge() -> dict[str, object]:
    return {
        "schema_version": formal_module.TIC2A_MERGED_SCHEMA,
        "status": "TIC2A_COMPLETE_UNSCORED_MERGE_PASS",
        "folds": [
            {"outer_fold": fold, "prediction_artifact": "unused"} for fold in range(20)
        ],
    }


def test_formal_loader_distinguishes_ensemble_from_constituent_seed(
    tmp_path: Path,
) -> None:
    ensemble = tmp_path / "ensemble.npz"
    source = tmp_path / "source.npz"
    _write_prediction(ensemble, schema="formal", fold=3, seed=-1)
    _write_prediction(source, schema="source", fold=3, seed=2)
    assert (
        _load_prediction(ensemble, schema="formal", fold=3, seed=None)["seed"].item()
        == -1
    )
    assert _load_prediction(source, schema="source", fold=3, seed=2)["seed"].item() == 2
    with pytest.raises(ValueError, match="seed mismatch"):
        _load_prediction(source, schema="source", fold=3, seed=None)


def test_formal_score_authority_is_exact_and_training_closed(tmp_path: Path) -> None:
    active = {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "authority": {"current_phase": EXPECTED_PHASE},
        "runnable_phases": [EXPECTED_PHASE],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": EXPECTED_SCORE_TOKEN,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(active), encoding="utf-8"
    )
    assert_score_authority(tmp_path)
    active["candidate_model_training_allowed"] = "still-open"
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(active), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="training must be closed"):
        assert_score_authority(tmp_path)


def test_formal_scorer_rejects_a_nonformal_merge_before_target_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="complete 100-run merge"):
        score_formal(
            {
                "schema_version": ASSEMBLY_SCHEMA,
                "status": ASSEMBLY_STATUS,
                "phase": "P1M4",
            },
            {},
            {},
            {},
            tmp_path / "unused.csv",
        )


def test_formal_scorer_scores_ten_component_mixture_not_seed_score_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged = _complete_merge(tmp_path / "rerun_a")
    assembly = assemble(merged, tmp_path / "formal_a")
    tic2a = _tic2a_merge()

    class _Universe:
        def __init__(self, _path: Path) -> None:
            self.records = [SimpleNamespace(puzzle=f"P{fold}") for fold in range(20)]

        def build(self) -> dict[str, str]:
            return {
                "canonical_mutant_full_profile_identity": (
                    "EXACT_PUZZLE_METHOD_MUTATION"
                )
            }

        def get_records(self) -> list[SimpleNamespace]:
            return self.records

    component_counts = []

    def _fake_score(_univ, _held, prediction, _absolute):
        component_counts.append(int(prediction["candidate_weights"].shape[1]))
        return {}

    monkeypatch.setattr(formal_module, "M2Universe", _Universe)
    monkeypatch.setattr(
        formal_module,
        "build_split_v4",
        lambda *_args, **_kwargs: {
            "folds": [
                SimpleNamespace(outer_fold=fold, held_puzzle=f"P{fold}")
                for fold in range(20)
            ]
        },
    )
    monkeypatch.setattr(
        formal_module,
        "_v13_reference_rows",
        lambda _score: {fold: {"held_puzzle": f"P{fold}"} for fold in range(20)},
    )
    monkeypatch.setattr(formal_module, "_load_tic2a_absolute", lambda *_args: {})
    monkeypatch.setattr(formal_module, "score_fold", _fake_score)
    monkeypatch.setattr(formal_module, "_add_frozen_references", lambda *_args: None)

    result = score_formal(assembly, merged, tic2a, {}, tmp_path / "unused.csv")
    assert result["status"] == "PUZZLE_SET_M4_COMPLETE_FORMAL_SCORE_PASS"
    assert result["context_retention_summary"] == _context_retention_summary()
    assert (
        result["formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources"]
        is True
    )
    assert component_counts.count(10) == 20
    assert component_counts.count(2) == 100


def test_formal_scorer_rejects_assembly_from_a_different_legal_rerun_before_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged_a = _complete_merge(tmp_path / "rerun_a", rerun_shift=0.0)
    assembly_a = assemble(merged_a, tmp_path / "formal_a")
    merged_b = _complete_merge(tmp_path / "rerun_b", rerun_shift=0.25)
    # Establish that rerun B independently forms a legal formal assembly; the
    # rejected condition is cross-pairing legal assembly A with legal merge B.
    assemble(merged_b, tmp_path / "formal_b")

    target_accessed = False

    class _ForbiddenUniverse:
        def __init__(self, _path: Path) -> None:
            nonlocal target_accessed
            target_accessed = True
            raise AssertionError("target universe opened before source reconstruction")

    monkeypatch.setattr(formal_module, "M2Universe", _ForbiddenUniverse)

    with pytest.raises(
        ValueError, match="does not exactly derive from merged source artifacts"
    ):
        score_formal(
            assembly_a,
            merged_b,
            _tic2a_merge(),
            {},
            tmp_path / "must-not-open.csv",
        )
    assert target_accessed is False
