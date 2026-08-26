from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.reactflow_delta.assemble_puzzle_set_meta_context_formal import (
    ASSEMBLY_STATUS,
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    FORMAL_PREDICTION_SCHEMA,
    SCHEMA,
    _REQUIRED_MERGE_INTEGRITY_TRUE,
    assemble,
    assemble_fold,
    assemble_fold_prediction_arrays,
)
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v9 import expected_absolute_delta
from scripts.reactflow_delta.puzzle_set_meta_context_data import PREDICTION_SCHEMA


def _write_prediction(
    directory: Path,
    *,
    fold: int,
    seed: int,
    reverse_keys: bool = False,
    feature41_shift: float = 0.0,
    parent_shift: float = 0.0,
) -> dict[str, object]:
    order = np.asarray([1, 0] if reverse_keys else [0, 1])
    keys = np.asarray([f"fold{fold}-row0", f"fold{fold}-row1"], dtype=object)[order]
    feature41_point = (np.asarray([100.0 + fold, -100.0 - fold]) + feature41_shift)[
        order
    ]
    parent_point = (np.asarray([200.0 + fold, -200.0 - fold]) + parent_shift)[order]
    candidate_point = np.asarray([float(seed) + fold, float(seed) + fold + 0.5])[order]
    null_point = np.asarray([10.0 + seed + fold, 10.5 + seed + fold])[order]
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.full(2, fold, dtype=np.int64),
        "seed": np.full(2, seed, dtype=np.int64),
        "registered_status": np.full(2, "covered", dtype=object),
        "feature41_point": feature41_point,
        "parent_point": parent_point,
        "candidate_point": candidate_point,
        "null_point": null_point,
    }
    narrow_weight = 0.2 + 0.05 * seed
    for name, point in (("candidate", candidate_point), ("null", null_point)):
        payload[f"{name}_weights"] = np.tile(
            [narrow_weight, 1.0 - narrow_weight], (2, 1)
        )
        payload[f"{name}_locations"] = np.stack([point, point], axis=1)
        payload[f"{name}_scales"] = np.tile(
            [0.1 + 0.01 * seed, 0.2 + 0.01 * seed], (2, 1)
        )
        payload[f"{name}_expected_absolute_delta"] = np.zeros(2)
    path = directory / f"prediction_fold{fold}_seed{seed}.npz"
    np.savez_compressed(path, **payload)
    return {
        "outer_fold": fold,
        "seed": seed,
        "prediction_artifact": str(path),
        "n_registered_prediction_rows": 2,
    }


def _fold_rows(directory: Path, *, fold: int = 0) -> list[dict[str, object]]:
    return [
        _write_prediction(directory, fold=fold, seed=seed) for seed in EXPECTED_SEEDS
    ]


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


def _complete_merge(directory: Path) -> dict[str, object]:
    rows = []
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            # Tiny source differences remain within the frozen replay tolerance.
            # The formal artifact must nevertheless retain seed 0 exactly.
            rows.append(
                _write_prediction(
                    directory,
                    fold=fold,
                    seed=seed,
                    feature41_shift=seed * 1e-8,
                    parent_shift=seed * 1e-8,
                )
            )
    integrity = {name: True for name in _REQUIRED_MERGE_INTEGRITY_TRUE}
    integrity.update(
        {"partial_scores_inspected": False, "external_outcome_accessed": False}
    )
    return {
        "schema_version": MERGED_SCHEMA,
        "status": "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS",
        "phase": "P1M4",
        "expected_folds": list(EXPECTED_FOLDS),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_pretraining_epochs": 200,
        "expected_point_epochs": 40,
        "expected_calibration_epochs": 40,
        "folds": rows,
        "context_retention_summary": _context_retention_summary(),
        "merge_integrity": integrity,
    }


def test_formal_assembly_uses_complete_universe_and_equal_seed_mixture(
    tmp_path: Path,
) -> None:
    merged = _complete_merge(tmp_path)
    result = assemble(merged, tmp_path / "formal")

    assert result["schema_version"] == SCHEMA
    assert result["status"] == ASSEMBLY_STATUS
    assert result["source_run_count"] == 100
    assert len(result["folds"]) == 20
    assert result["best_seed_selection_performed"] is False
    assert result["score_computed"] is False

    fold0 = result["folds"][0]
    assert fold0["seeds"] == [0, 1, 2, 3, 4]
    assert fold0["components_per_distribution"] == 10
    assert fold0["equal_seed_weight"] == 0.2
    with np.load(fold0["prediction_artifact"], allow_pickle=True) as prediction:
        assert str(prediction["schema_version"].item()) == FORMAL_PREDICTION_SCHEMA
        assert np.array_equal(prediction["feature41_point"], [100.0, -100.0])
        assert np.array_equal(prediction["parent_point"], [200.0, -200.0])
        assert np.allclose(prediction["candidate_point"], [2.0, 2.5])
        assert np.allclose(prediction["null_point"], [12.0, 12.5])
        for name in ("candidate", "null"):
            weights = prediction[f"{name}_weights"]
            locations = prediction[f"{name}_locations"]
            scales = prediction[f"{name}_scales"]
            assert weights.shape == locations.shape == scales.shape == (2, 10)
            assert np.allclose(weights.sum(axis=1), 1.0)
            for seed in EXPECTED_SEEDS:
                assert np.allclose(weights[:, 2 * seed : 2 * seed + 2].sum(axis=1), 0.2)
            expected_absolute = expected_absolute_delta(
                torch.as_tensor(weights, dtype=torch.float64),
                torch.as_tensor(locations, dtype=torch.float64),
                torch.as_tensor(scales, dtype=torch.float64),
            ).numpy()
            assert np.allclose(
                prediction[f"{name}_expected_absolute_delta"], expected_absolute
            )
        forbidden_fragments = ("target", "score", "best_seed", "selected_seed")
        assert not any(
            fragment in field
            for field in prediction.files
            for fragment in forbidden_fragments
        )


def test_persisted_formal_fold_exactly_matches_the_pure_array_assembly(
    tmp_path: Path,
) -> None:
    rows = _fold_rows(tmp_path)
    sources = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        with np.load(row["prediction_artifact"], allow_pickle=True) as handle:
            sources.append(
                {name: np.asarray(handle[name]).copy() for name in handle.files}
            )
    expected = assemble_fold_prediction_arrays(sources, fold=0)
    assembled = assemble_fold(rows, fold=0, out_dir=tmp_path / "formal")
    with np.load(assembled["prediction_artifact"], allow_pickle=True) as handle:
        observed = {name: np.asarray(handle[name]).copy() for name in handle.files}

    assert set(observed) == set(expected)
    for name in expected:
        assert observed[name].shape == expected[name].shape
        assert observed[name].dtype == expected[name].dtype
        assert np.array_equal(observed[name], expected[name])


def test_formal_fold_rejects_missing_or_duplicate_seed(tmp_path: Path) -> None:
    rows = _fold_rows(tmp_path)
    with pytest.raises(ValueError, match="unique seeds0-4"):
        assemble_fold(rows[:-1], fold=0, out_dir=tmp_path / "missing")

    duplicate = [*rows[:-1], dict(rows[0])]
    with pytest.raises(ValueError, match="unique seeds0-4"):
        assemble_fold(duplicate, fold=0, out_dir=tmp_path / "duplicate")


def test_formal_fold_rejects_cross_seed_key_order_change(tmp_path: Path) -> None:
    rows = _fold_rows(tmp_path)
    rows[-1] = _write_prediction(tmp_path, fold=0, seed=4, reverse_keys=True)
    with pytest.raises(ValueError, match="key order differs"):
        assemble_fold(rows, fold=0, out_dir=tmp_path / "formal")


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("feature41_point", {"feature41_shift": 1e-5}),
        ("parent_point", {"parent_shift": 1e-5}),
    ],
)
def test_formal_fold_rejects_frozen_point_drift(
    tmp_path: Path, field: str, kwargs: dict[str, float]
) -> None:
    rows = _fold_rows(tmp_path)
    rows[-1] = _write_prediction(tmp_path, fold=0, seed=4, **kwargs)
    with pytest.raises(ValueError, match=f"{field} differs by seed"):
        assemble_fold(rows, fold=0, out_dir=tmp_path / "formal")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", "P1M3"),
        ("expected_pretraining_epochs", 199),
        ("expected_point_epochs", 39),
        ("expected_calibration_epochs", 39),
    ],
)
def test_formal_assembly_rejects_non_p1m4_or_changed_training_freeze(
    tmp_path: Path, field: str, value: object
) -> None:
    merged = _complete_merge(tmp_path)
    merged[field] = value
    with pytest.raises(ValueError, match="complete P1M4 merge|changed frozen"):
        assemble(merged, tmp_path / "formal")


def test_formal_assembly_rejects_incomplete_or_duplicated_run_universe(
    tmp_path: Path,
) -> None:
    merged = _complete_merge(tmp_path)
    missing = copy.deepcopy(merged)
    missing["folds"] = missing["folds"][:-1]
    with pytest.raises(ValueError, match="unique 20-fold x five-seed universe"):
        assemble(missing, tmp_path / "missing")

    duplicated = copy.deepcopy(merged)
    duplicated["folds"][-1] = copy.deepcopy(duplicated["folds"][0])
    with pytest.raises(ValueError, match="unique 20-fold x five-seed universe"):
        assemble(duplicated, tmp_path / "duplicated")


def test_formal_assembly_rejects_failed_merge_integrity(tmp_path: Path) -> None:
    merged = _complete_merge(tmp_path)
    merged["merge_integrity"][
        "mutant_outcome_excluded_from_pretraining_all_runs"
    ] = False
    with pytest.raises(ValueError, match="failed integrity check"):
        assemble(merged, tmp_path / "formal")


def test_formal_assembly_rejects_incomplete_input_provenance(tmp_path: Path) -> None:
    merged = _complete_merge(tmp_path)
    merged["merge_integrity"]["complete_frozen_input_provenance_all_runs"] = False
    with pytest.raises(ValueError, match="failed integrity check"):
        assemble(merged, tmp_path / "formal")
