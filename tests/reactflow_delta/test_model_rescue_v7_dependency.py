from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.reactflow_delta.model_rescue_v7_dependency import (
    dependency_features_from_acgu_logits,
    exact_mutant_sequence,
    normalize_rna_sequence,
)
from scripts.reactflow_delta.model_rescue_v7_schema import FEATURE_NAMES


def test_exact_mutant_sequence_uses_correct_full_coordinate() -> None:
    assert normalize_rna_sequence("ACGT") == "ACGU"
    assert exact_mutant_sequence("GGGAAACCC", 3, "A", "G") == "GGGGAACCC"
    with pytest.raises(ValueError, match="reference"):
        exact_mutant_sequence("GGGAAACCC", 3, "C", "G")


def test_dependency_features_match_published_signed_log_odds_transform() -> None:
    sequence = "ACGU"
    wt_logits = np.zeros((4, 4), dtype=np.float32)
    mutant_logits = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -0.5, 0.25, 0.0],
            [-0.1, 0.2, 0.8, -0.4],
            [0.4, 0.1, -0.2, 0.7],
        ],
        dtype=np.float32,
    )
    features = dependency_features_from_acgu_logits(
        wt_logits, mutant_logits, sequence, source_pos=0
    )
    assert features.shape == (4, len(FEATURE_NAMES))
    np.testing.assert_array_equal(features[0], np.zeros(len(FEATURE_NAMES)))

    wt_prob = np.exp(wt_logits) / np.exp(wt_logits).sum(axis=1, keepdims=True)
    mut_prob = np.exp(mutant_logits) / np.exp(mutant_logits).sum(axis=1, keepdims=True)
    expected = np.log2(mut_prob / (1.0 - mut_prob)) - np.log2(
        wt_prob / (1.0 - wt_prob)
    )
    np.testing.assert_allclose(features[1:, :4], expected[1:], atol=1e-6, rtol=0.0)
    receiver_indices = np.asarray([0, 1, 2, 3])
    np.testing.assert_allclose(
        features[1:, 4], expected[np.arange(4), receiver_indices][1:], atol=1e-6
    )
    np.testing.assert_allclose(
        features[1:, 5], np.max(np.abs(expected), axis=1)[1:], atol=1e-6
    )


def test_dependency_transform_rejects_length_or_nonfinite_drift() -> None:
    with pytest.raises(ValueError, match="shape"):
        dependency_features_from_acgu_logits(
            np.zeros((2, 4)), np.zeros((2, 4)), "ACG", source_pos=0
        )
    bad = np.zeros((3, 4))
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dependency_features_from_acgu_logits(
            bad, np.zeros((3, 4)), "ACG", source_pos=0
        )


def _frame(mutant_outcome: float) -> pd.DataFrame:
    rows = []
    for method in ("methodA", "methodB"):
        prefix = f"P01_{method}"
        rows.extend(
            [
                {
                    "id": f"{prefix}_wt",
                    "sequence": "GGGAAACCC",
                    "puzzle": "P01",
                    "method": method,
                    "sub_start": 1,
                    "mutA": np.nan,
                    "reactivity_0001": 0.1,
                },
                {
                    "id": f"{prefix}_mm_3_A_G",
                    "sequence": "GGGGAACCC",
                    "puzzle": "P01",
                    "method": method,
                    "sub_start": 1,
                    "mutA": 4,
                    "reactivity_0001": mutant_outcome,
                },
            ]
        )
    return pd.DataFrame(rows)


class _FakeInferer:
    def __call__(self, sequences: list[str], *, batch_size: int) -> dict[str, np.ndarray]:
        assert batch_size > 0
        mapping = {base: index for index, base in enumerate("ACGU")}
        output = {}
        for sequence in sequences:
            counts = np.bincount(
                [mapping[base] for base in sequence], minlength=4
            ).astype(np.float32)
            values = np.empty((len(sequence), 4), dtype=np.float32)
            for position, base in enumerate(sequence):
                values[position] = 0.03 * (position + 1) * counts
                values[position, mapping[base]] += 0.5
            output[sequence] = values
        return output


def test_small_cache_is_outcome_invariant_and_reuses_exact_sequence_duplicates(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    from scripts.reactflow_delta.build_model_rescue_v7_dependency_cache import build_cache
    from scripts.reactflow_delta.qualify_model_rescue_v7_dependency_cache import (
        qualify_cache,
    )

    artifacts = []
    for name, outcome in (("first", 999.0), ("second", -999.0)):
        csv_path = tmp_path / f"{name}.csv"
        cache_path = tmp_path / f"{name}.h5"
        manifest_path = tmp_path / f"{name}.json"
        _frame(outcome).to_csv(csv_path, index=False)
        manifest = build_cache(
            csv_path,
            cache_path,
            manifest_path,
            inferer=_FakeInferer(),
            batch_size=2,
            max_constructs=None,
            model_code_root=None,
            weights_path=None,
            attention_backend="TEST",
        )
        assert manifest["mutant_reactivity_rows_read"] == 0
        assert manifest["n_unique_dependency_edges"] == 1
        assert manifest["n_dependency_edge_reuse_rows"] == 1
        assert manifest["n_unique_wt_sequences"] == 1
        assert manifest["n_unique_mutant_sequences"] == 1
        assert manifest["n_unique_inference_sequences"] == 2
        qualification = qualify_cache(
            cache_path,
            manifest_path,
            csv_path,
            expected_constructs=2,
            expected_mutants=2,
            expected_length=9,
        )
        assert qualification["status"] == (
            "V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_PASS"
        )
        assert all(qualification["checks"].values())
        artifacts.append(cache_path)

    with h5py.File(artifacts[0], "r") as first, h5py.File(
        artifacts[1], "r"
    ) as second:
        np.testing.assert_array_equal(first["features"][:], second["features"][:])
        np.testing.assert_array_equal(first["features"][0], first["features"][1])
