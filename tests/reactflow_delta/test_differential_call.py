"""Tests for T-D1.9 frozen differential caller
(v3 §6.6 step 12 Δreactivity; §7.2 replicate-aware + FDR; §7.3 continuous-only).
"""

from __future__ import annotations

import math

import pytest

from reactflow.delta.data import (
    DIFFERENTIAL_FDR_ALPHA,
    _benjamini_hochberg,
    _normal_cdf,
    build_pair_delta_reactivity,
    compute_delta_reactivity,
    frozen_differential_call,
)


# -- _normal_cdf / _benjamini_hochberg helpers -------------------------------


class TestHelpers:
    def test_normal_cdf_known(self) -> None:
        assert _normal_cdf(0.0) == pytest.approx(0.5)
        assert _normal_cdf(1.0) == pytest.approx(0.8413447460685429)
        assert _normal_cdf(-1.0) == pytest.approx(0.15865525393145707)

    def test_bh_empty(self) -> None:
        threshold_p, rejected = _benjamini_hochberg([], 0.05)
        assert threshold_p is None
        assert rejected == set()

    def test_bh_no_rejection(self) -> None:
        # both p above BH cutoff
        threshold_p, rejected = _benjamini_hochberg([0.4, 0.5], 0.05)
        assert threshold_p is None
        assert rejected == set()

    def test_bh_rejects_both(self) -> None:
        # p = [0.0000634, 0.0027]; both <= k/2 * 0.05
        threshold_p, rejected = _benjamini_hochberg([0.0027, 0.0000634], 0.05)
        assert threshold_p == pytest.approx(0.0027)
        assert rejected == {0, 1}

    def test_bh_rejects_one(self) -> None:
        # p = [0.01, 0.5]; k=1: 0.01 <= 0.025 yes; k=2: 0.5 <= 0.05 no → reject only idx 0
        threshold_p, rejected = _benjamini_hochberg([0.01, 0.5], 0.05)
        assert threshold_p == pytest.approx(0.01)
        assert rejected == {0}


# -- compute_delta_reactivity ------------------------------------------------


class TestComputeDeltaReactivity:
    def test_known_values(self) -> None:
        assert compute_delta_reactivity([1.0, 2.0, 3.0], [4.0, 2.0, 0.0]) == [3.0, 0.0, -3.0]

    def test_missing_yields_none(self) -> None:
        assert compute_delta_reactivity([1.0, None, 3.0], [4.0, 2.0, None]) == [3.0, None, None]

    def test_missing_wt_yields_none(self) -> None:
        assert compute_delta_reactivity([None, 2.0], [4.0, 2.0]) == [None, 0.0]

    def test_unequal_length_uses_min(self) -> None:
        assert compute_delta_reactivity([1.0, 2.0, 3.0], [4.0, 2.0]) == [3.0, 0.0]

    def test_empty(self) -> None:
        assert compute_delta_reactivity([], []) == []

    def test_non_finite_yields_none(self) -> None:
        result = compute_delta_reactivity([1.0, float("nan")], [4.0, float("inf")])
        assert result[0] == pytest.approx(3.0)
        assert result[1] is None


# -- frozen_differential_call ------------------------------------------------


class TestFrozenDifferentialCall:
    def test_replicate_aware_with_threshold(self) -> None:
        delta = [0.5, 2.0, 0.1]
        result = frozen_differential_call(delta, noise_threshold=1.0, has_replicates=True)
        assert result["significant_mask"] == [0, 1, 0]
        assert result["significant_count"] == 1
        assert result["caller_status"] == "replicate_aware"

    def test_no_replicate_no_significance(self) -> None:
        delta = [0.5, 5.0, 0.1]
        result = frozen_differential_call(delta, noise_threshold=1.0, has_replicates=False)
        assert result["significant_mask"] == [0, 0, 0]
        assert result["significant_count"] == 0
        assert result["caller_status"] == "no_replicate_continuous_only"

    def test_replicate_no_threshold(self) -> None:
        delta = [0.5, 5.0]
        result = frozen_differential_call(delta, noise_threshold=None, has_replicates=True)
        assert result["significant_mask"] == [0, 0]
        assert result["caller_status"] == "no_threshold"

    def test_threshold_strict_inequality(self) -> None:
        # |Δr| == threshold should NOT be significant (strict >)
        delta = [1.0]
        result = frozen_differential_call(delta, noise_threshold=1.0, has_replicates=True)
        assert result["significant_mask"] == [0]

    def test_missing_delta_not_significant(self) -> None:
        delta = [None, 5.0]
        result = frozen_differential_call(delta, noise_threshold=1.0, has_replicates=True)
        assert result["significant_mask"] == [0, 1]

    def test_z_scores_with_variance(self) -> None:
        delta = [1.0, 2.0]
        result = frozen_differential_call(
            delta, noise_threshold=10.0, measurement_variance=1.0, has_replicates=True
        )
        assert result["z_scores"][0] == pytest.approx(1.0)
        assert result["z_scores"][1] == pytest.approx(2.0)

    def test_z_scores_none_without_variance(self) -> None:
        delta = [1.0, 2.0]
        result = frozen_differential_call(delta, has_replicates=True)
        assert result["z_scores"] == [None, None]

    def test_z_scores_missing_delta_is_none(self) -> None:
        delta = [1.0, None]
        result = frozen_differential_call(delta, measurement_variance=1.0, has_replicates=True)
        assert result["z_scores"][0] == pytest.approx(1.0)
        assert result["z_scores"][1] is None

    def test_fdr_rejection_replicate_aware(self) -> None:
        # z=[3,4], var=1.0 → p≈[0.0027, 0.0000634]; both rejected at α=0.05
        delta = [3.0, 4.0]
        result = frozen_differential_call(
            delta, noise_threshold=10.0, measurement_variance=1.0, has_replicates=True
        )
        assert result["fdr_significant_mask"] == [1, 1]
        assert result["fdr_threshold_p"] == pytest.approx(0.0027, abs=1e-4)

    def test_fdr_no_rejection(self) -> None:
        # z=[1,2] → p≈[0.317, 0.0455]; neither rejected at α=0.05
        delta = [1.0, 2.0]
        result = frozen_differential_call(
            delta, noise_threshold=10.0, measurement_variance=1.0, has_replicates=True
        )
        assert result["fdr_significant_mask"] == [0, 0]
        assert result["fdr_threshold_p"] is None

    def test_fdr_not_computed_without_replicates(self) -> None:
        delta = [3.0, 4.0]
        result = frozen_differential_call(
            delta, measurement_variance=1.0, has_replicates=False
        )
        assert result["fdr_significant_mask"] == [0, 0]
        assert result["fdr_threshold_p"] is None

    def test_zero_variance_no_zscores(self) -> None:
        delta = [1.0, 2.0]
        result = frozen_differential_call(
            delta, measurement_variance=0.0, has_replicates=True
        )
        assert result["z_scores"] == [None, None]

    def test_alpha_constant(self) -> None:
        assert DIFFERENTIAL_FDR_ALPHA == 0.05


# -- build_pair_delta_reactivity ---------------------------------------------


class TestBuildPairDeltaReactivity:
    def test_raw_only(self) -> None:
        result = build_pair_delta_reactivity(
            [1.0, 2.0, 3.0], [4.0, 2.0, 0.0],
            noise_threshold=1.0, has_replicates=True,
        )
        assert result["delta_reactivity_raw"] == [3.0, 0.0, -3.0]
        assert result["delta_reactivity_normalized"] is None
        # call on raw layer: |3|>1, |0|<=1, |-3|>1
        assert result["significant_mask"] == [1, 0, 1]
        assert result["caller_status"] == "replicate_aware"

    def test_with_normalized_layer(self) -> None:
        result = build_pair_delta_reactivity(
            [1.0, 2.0], [4.0, 2.0],
            wt_reactivity_normalized=[0.0, 0.5],
            mut_reactivity_normalized=[1.0, 0.5],
            noise_threshold=0.2, has_replicates=True,
        )
        assert result["delta_reactivity_raw"] == [3.0, 0.0]
        assert result["delta_reactivity_normalized"] == [1.0, 0.0]
        # call on normalized: |1.0|>0.2 yes, |0.0|<=0.2 no
        assert result["significant_mask"] == [1, 0]

    def test_missing_propagated(self) -> None:
        result = build_pair_delta_reactivity(
            [1.0, None], [None, 2.0],
        )
        assert result["delta_reactivity_raw"] == [None, None]

    def test_no_replicate_status(self) -> None:
        result = build_pair_delta_reactivity([1.0], [3.0], has_replicates=False)
        assert result["caller_status"] == "no_replicate_continuous_only"
        assert result["significant_mask"] == [0]
