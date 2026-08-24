from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("RNA")

from scripts.reactflow_delta.build_model_rescue_v5_ensemble_cache import (
    FEATURE_NAMES,
    SOURCE_COLUMNS,
    build_construct_groups,
    ensemble_delta_features,
    fold_ensemble,
)
from scripts.reactflow_delta.qualify_model_rescue_v5_ensemble_cache import (
    qualify_cache,
)


def _frame(mutant_sequence: str = "GGGAAACCC") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "P01_method_wt",
                "sequence": "GGGAAACCC",
                "puzzle": "P01",
                "method": "method",
                "sub_start": 1,
                "mutA": np.nan,
            },
            {
                "id": "P01_method_mm_3_A_G",
                "sequence": mutant_sequence,
                "puzzle": "P01",
                "method": "method",
                "sub_start": 1,
                "mutA": 4,
            },
        ],
        columns=SOURCE_COLUMNS,
    )


def test_construct_group_uses_corrected_full_coordinate_and_exact_mutant() -> None:
    groups = build_construct_groups(_frame("GGGGAACCC"))
    assert len(groups) == 1
    mutant = groups[0]["mutants"][0]
    assert mutant["design_pos"] == 3
    assert mutant["full_pos"] == 3
    assert mutant["ref"] == "A"
    assert mutant["alt"] == "G"


def test_construct_group_rejects_mutation_at_wrong_full_coordinate() -> None:
    with pytest.raises(ValueError, match="exact one-base mutant at full_pos"):
        build_construct_groups(_frame("GGAGAACCC"))


def test_ensemble_delta_features_are_zero_for_identical_ensembles() -> None:
    bpp, entropy, energy = fold_ensemble("GGGAAACCC")
    features = ensemble_delta_features(
        bpp, entropy, energy, bpp.copy(), entropy.copy(), energy, full_pos=3
    )
    assert features.shape == (9, len(FEATURE_NAMES))
    zero_columns = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    assert np.array_equal(features[:, zero_columns], np.zeros((9, len(zero_columns))))
    assert np.array_equal(features[:, 3], features[:, 4])


def test_exact_mutant_ensemble_features_are_finite_and_nontrivial() -> None:
    wt_bpp, wt_entropy, wt_energy = fold_ensemble("GGGAAACCC")
    mut_bpp, mut_entropy, mut_energy = fold_ensemble("GGGGAACCC")
    features = ensemble_delta_features(
        wt_bpp,
        wt_entropy,
        wt_energy,
        mut_bpp,
        mut_entropy,
        mut_energy,
        full_pos=3,
    )
    assert features.shape == (9, 12)
    assert np.isfinite(features).all()
    assert np.count_nonzero(features) > 0
    assert np.all(features[:, 5] >= 0)
    assert np.all(features[:, 6] >= 0)


def test_small_real_cache_qualifier_checks_exact_universe(tmp_path) -> None:
    from scripts.reactflow_delta.build_model_rescue_v5_ensemble_cache import build_cache

    csv_path = tmp_path / "metadata.csv"
    cache_path = tmp_path / "cache.h5"
    manifest_path = tmp_path / "manifest.json"
    _frame("GGGGAACCC").to_csv(csv_path, index=False)
    manifest = build_cache(
        csv_path,
        cache_path,
        manifest_path,
        workers=1,
        max_constructs=None,
    )
    assert manifest["outcome_columns_read"] is False
    result = qualify_cache(
        cache_path,
        manifest_path,
        csv_path,
        expected_constructs=1,
        expected_mutants=1,
        expected_length=9,
    )
    assert result["status"] == "V5M1_OUTCOME_BLIND_ENSEMBLE_CACHE_PASS"
    assert all(result["checks"].values())
