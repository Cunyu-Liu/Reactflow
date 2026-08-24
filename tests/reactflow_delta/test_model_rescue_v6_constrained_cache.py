from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("RNA")

from scripts.reactflow_delta.build_model_rescue_v5_ensemble_cache import (
    build_cache as build_unconstrained_cache,
    fold_ensemble,
)
from scripts.reactflow_delta.build_model_rescue_v6_constrained_cache import (
    build_cache,
    fold_constrained_ensemble,
    load_outcome_blind_inputs,
    shape_constraint_vector,
)
from scripts.reactflow_delta.model_rescue_v6_schema import (
    FEATURE_NAMES,
    METADATA_COLUMNS,
    MISSING_REACTIVITY,
)
from scripts.reactflow_delta.qualify_model_rescue_v6_constrained_cache import (
    qualify_cache,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": "P01_method_wt",
            "sequence": "GGGAAACCC",
            "puzzle": "P01",
            "method": "method",
            "sub_start": 1,
            "mutA": np.nan,
            **{f"reactivity_{i + 1:04d}": value for i, value in enumerate(
                [0.1, -0.2, 0.3, np.nan, 0.5, 0.6, 0.2, 0.1, 0.4]
            )},
        },
        {
            "id": "P01_method_mm_3_A_G",
            "sequence": "GGGGAACCC",
            "puzzle": "P01",
            "method": "method",
            "sub_start": 1,
            "mutA": 4,
            **{f"reactivity_{i + 1:04d}": 900.0 + i for i in range(9)},
        },
        {
            "id": "P02_method_wt",
            "sequence": "CCCAAAGGG",
            "puzzle": "P02",
            "method": "method",
            "sub_start": 1,
            "mutA": np.nan,
            **{f"reactivity_{i + 1:04d}": np.nan for i in range(9)},
        },
        {
            "id": "P02_method_mm_3_A_G",
            "sequence": "CCCGAAGGG",
            "puzzle": "P02",
            "method": "method",
            "sub_start": 1,
            "mutA": 4,
            **{f"reactivity_{i + 1:04d}": -900.0 - i for i in range(9)},
        },
    ]


def _frame() -> pd.DataFrame:
    columns = [*METADATA_COLUMNS, *(f"reactivity_{i + 1:04d}" for i in range(9))]
    return pd.DataFrame(_rows(), columns=columns)


def test_wt_only_loader_does_not_materialize_mutant_reactivity(tmp_path) -> None:
    path = tmp_path / "m2.csv"
    _frame().to_csv(path, index=False)
    metadata, columns, profiles = load_outcome_blind_inputs(path)
    assert len(metadata) == 4
    assert len(columns) == 9
    assert set(profiles) == {"P01_method_wt", "P02_method_wt"}
    assert profiles["P01_method_wt"][0] == pytest.approx(0.1)
    assert np.isnan(profiles["P02_method_wt"]).all()


def test_constraint_policy_clamps_negative_and_preserves_missing() -> None:
    vector, policy = shape_constraint_vector(np.asarray([0.2, -0.4, np.nan]))
    assert vector == [MISSING_REACTIVITY, 0.2, 0.0, MISSING_REACTIVITY]
    assert policy == {
        "observed_positions": 2,
        "missing_positions": 1,
        "negative_positions_clamped": 1,
        "all_missing": False,
    }


def test_all_missing_constraint_is_exact_unconstrained_fallback() -> None:
    sequence = "GGGAAACCC"
    constrained = fold_constrained_ensemble(sequence, np.full(len(sequence), np.nan))
    unconstrained = fold_ensemble(sequence)
    assert np.array_equal(constrained[0], unconstrained[0])
    assert np.array_equal(constrained[1], unconstrained[1])
    assert constrained[2] == unconstrained[2]


def test_observed_constraint_changes_ensemble_without_changing_shape() -> None:
    sequence = "GGGAAACCC"
    profile = np.asarray([0.1, 0.0, 0.3, np.nan, 0.5, 0.6, 0.2, 0.1, 0.4])
    constrained = fold_constrained_ensemble(sequence, profile)
    unconstrained = fold_ensemble(sequence)
    assert constrained[0].shape == unconstrained[0].shape == (9, 9)
    assert not np.array_equal(constrained[0], unconstrained[0])


def test_cache_is_invariant_to_mutant_outcome_values(tmp_path) -> None:
    first = _frame()
    second = _frame()
    mutant_rows = ~second["id"].str.endswith("_wt")
    reactivity_columns = [column for column in second if column.startswith("reactivity_")]
    second.loc[mutant_rows, reactivity_columns] = 123456.0
    paths = []
    for name, frame in (("first", first), ("second", second)):
        csv_path = tmp_path / f"{name}.csv"
        cache_path = tmp_path / f"{name}.h5"
        manifest_path = tmp_path / f"{name}.json"
        frame.to_csv(csv_path, index=False)
        build_cache(csv_path, cache_path, manifest_path, workers=1, max_constructs=None)
        paths.append(cache_path)
    with h5py.File(paths[0], "r") as first_cache, h5py.File(
        paths[1], "r"
    ) as second_cache:
        assert np.array_equal(first_cache["features"][:], second_cache["features"][:])


def test_small_cache_qualifies_against_unconstrained_reference(tmp_path) -> None:
    csv_path = tmp_path / "m2.csv"
    unconstrained_path = tmp_path / "unconstrained.h5"
    unconstrained_manifest = tmp_path / "unconstrained.json"
    constrained_path = tmp_path / "constrained.h5"
    constrained_manifest = tmp_path / "constrained.json"
    _frame().to_csv(csv_path, index=False)
    build_unconstrained_cache(
        csv_path,
        unconstrained_path,
        unconstrained_manifest,
        workers=1,
        max_constructs=None,
    )
    manifest = build_cache(
        csv_path,
        constrained_path,
        constrained_manifest,
        workers=1,
        max_constructs=None,
    )
    assert manifest["mutant_reactivity_rows_read"] == 0
    assert manifest["all_missing_construct_ids"] == ["P02_method"]
    result = qualify_cache(
        constrained_path,
        constrained_manifest,
        unconstrained_path,
        csv_path,
        expected_constructs=2,
        expected_mutants=2,
        expected_length=9,
    )
    assert result["status"] == "V6M1_OUTCOME_BLIND_CONSTRAINED_CACHE_PASS"
    assert all(result["checks"].values())
    assert result["comparison_counts"]["observed_constructs"] == 1
    assert result["comparison_counts"]["all_missing_rows"] == 1
    with h5py.File(constrained_path, "r") as handle:
        assert handle["features"].shape == (2, 9, len(FEATURE_NAMES))
