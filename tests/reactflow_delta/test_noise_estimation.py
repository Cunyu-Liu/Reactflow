"""Tests for T-D1.8 study/probe measurement-noise estimation
(v3 §6.6 step 10; §7.2 with-replicates, §7.3 without-replicates; v3.1 §2.2/§4).
"""

from __future__ import annotations

import math

import pytest

from reactflow.delta.data import (
    CONTROL_NOISE_THRESHOLD_MIN_VALUES,
    CONTROL_NOISE_THRESHOLD_PERCENTILE,
    NOISE_ESTIMATION_MIN_OVERLAP,
    NOISE_ESTIMATION_MIN_REPLICATES,
    estimate_error_variance,
    estimate_pair_noise,
    estimate_replicate_noise,
    freeze_control_noise_threshold,
)


# -- estimate_replicate_noise ------------------------------------------------


class TestEstimateReplicateNoise:
    def test_known_two_replicates(self) -> None:
        # per-pos: [1,3]->var2, [2,2]->var0, [3,1]->var2 ; mean var = 4/3
        result = estimate_replicate_noise([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        assert result["n_replicates"] == 2
        assert result["n_positions"] == 3
        assert result["noise_variance"] == pytest.approx(4.0 / 3.0)
        assert result["noise_std"] == pytest.approx(math.sqrt(4.0 / 3.0))

    def test_three_replicates(self) -> None:
        # pos0: [1,2,3] mean2 var=((1+0+1)/2)=1 ; pos1: [4,4,4] var0
        result = estimate_replicate_noise([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
        assert result["n_replicates"] == 3
        assert result["n_positions"] == 2
        assert result["noise_variance"] == pytest.approx(0.5)  # (1 + 0) / 2

    def test_single_replicate_returns_none(self) -> None:
        result = estimate_replicate_noise([[1.0, 2.0, 3.0]])
        assert result["noise_std"] is None
        assert result["noise_variance"] is None
        assert result["n_replicates"] == 1
        assert result["n_positions"] == 0

    def test_empty_input(self) -> None:
        result = estimate_replicate_noise([])
        assert result["noise_std"] is None
        assert result["n_replicates"] == 0

    def test_excludes_missing(self) -> None:
        # pos0: [1,3] var2 ; pos1: [None,None] skipped ; pos2: [3,1] var2
        result = estimate_replicate_noise(
            [[1.0, None, 3.0], [3.0, None, 1.0]]
        )
        assert result["n_positions"] == 2
        assert result["noise_variance"] == pytest.approx(2.0)

    def test_below_min_overlap(self) -> None:
        # only 1 position has >=2 non-missing -> below default min_overlap=2
        result = estimate_replicate_noise([[1.0, None], [3.0, None]])
        assert result["noise_std"] is None
        assert result["n_positions"] == 1

    def test_excludes_non_finite(self) -> None:
        result = estimate_replicate_noise(
            [[1.0, float("nan"), 3.0], [3.0, float("inf"), 1.0]]
        )
        # pos0: [1,3] var2 ; pos1: skipped ; pos2: [3,1] var2
        assert result["n_positions"] == 2
        assert result["noise_variance"] == pytest.approx(2.0)

    def test_unequal_lengths_uses_min(self) -> None:
        # lengths 3 and 2 -> compare positions 0,1 only
        result = estimate_replicate_noise([[1.0, 2.0, 3.0], [3.0, 2.0]])
        assert result["n_positions"] == 2
        # pos0 [1,3] var2, pos1 [2,2] var0 -> mean 1.0
        assert result["noise_variance"] == pytest.approx(1.0)


# -- estimate_error_variance -------------------------------------------------


class TestEstimateErrorVariance:
    def test_known_values(self) -> None:
        # (0.01 + 0.04 + 0.04) / 3 = 0.03
        assert estimate_error_variance([0.1, 0.2, 0.2]) == pytest.approx(0.03)

    def test_excludes_none(self) -> None:
        # (0.01 + 0.04) / 2 = 0.025
        assert estimate_error_variance([0.1, None, 0.2]) == pytest.approx(0.025)

    def test_all_missing_returns_none(self) -> None:
        assert estimate_error_variance([None, None]) is None

    def test_empty_returns_none(self) -> None:
        assert estimate_error_variance([]) is None

    def test_excludes_non_finite(self) -> None:
        assert estimate_error_variance([0.1, float("nan"), float("inf")]) == pytest.approx(0.01)


# -- freeze_control_noise_threshold ------------------------------------------


class TestFreezeControlNoiseThreshold:
    def test_known_95th_percentile_n10(self) -> None:
        values = [0.1 * (i + 1) for i in range(10)]  # 0.1..1.0
        threshold = freeze_control_noise_threshold(values)
        # nearest-rank 95th of 10 = rank 10 = 1.0
        assert threshold == pytest.approx(1.0)

    def test_below_min_values_returns_none(self) -> None:
        values = [0.1, 0.2, 0.3]
        assert freeze_control_noise_threshold(values) is None

    def test_custom_percentile(self) -> None:
        values = [float(i) for i in range(1, 21)]  # 1..20, n=20
        # 50th percentile nearest-rank = rank 10 = 10.0
        threshold = freeze_control_noise_threshold(values, percentile=50.0)
        assert threshold == pytest.approx(10.0)

    def test_custom_min_values(self) -> None:
        values = [0.1, 0.2, 0.3]
        assert freeze_control_noise_threshold(values, min_values=3) == pytest.approx(0.3)

    def test_excludes_none_and_non_finite(self) -> None:
        values = [0.1 * (i + 1) for i in range(10)] + [None, float("nan")]
        threshold = freeze_control_noise_threshold(values)
        assert threshold == pytest.approx(1.0)

    def test_constants(self) -> None:
        assert CONTROL_NOISE_THRESHOLD_PERCENTILE == 95.0
        assert CONTROL_NOISE_THRESHOLD_MIN_VALUES == 10
        assert NOISE_ESTIMATION_MIN_REPLICATES == 2
        assert NOISE_ESTIMATION_MIN_OVERLAP == 2


# -- estimate_pair_noise -----------------------------------------------------


class TestEstimatePairNoise:
    def test_both_replicate(self) -> None:
        wt = {"noise_variance": 2.0}
        mut = {"noise_variance": 3.0}
        result = estimate_pair_noise(wt, mut)
        assert result["measurement_variance"] == pytest.approx(5.0)
        assert result["replicate_noise_estimate"] == pytest.approx(math.sqrt(5.0))
        assert result["source"] == "replicate"
        assert result["wt_variance"] == pytest.approx(2.0)
        assert result["mut_variance"] == pytest.approx(3.0)

    def test_wt_replicate_mut_error(self) -> None:
        wt = {"noise_variance": 2.0}
        result = estimate_pair_noise(wt, None, mut_error_variance=0.5)
        assert result["measurement_variance"] == pytest.approx(2.5)
        assert result["replicate_noise_estimate"] == pytest.approx(math.sqrt(2.0))
        assert result["source"] == "replicate"
        assert result["mut_variance"] == pytest.approx(0.5)

    def test_both_error_only(self) -> None:
        result = estimate_pair_noise(None, None, wt_error_variance=0.1, mut_error_variance=0.2)
        assert result["measurement_variance"] == pytest.approx(0.3)
        assert result["replicate_noise_estimate"] is None
        assert result["source"] == "upstream_error"

    def test_no_info(self) -> None:
        result = estimate_pair_noise(None, None)
        assert result["measurement_variance"] is None
        assert result["replicate_noise_estimate"] is None
        assert result["source"] == "none"

    def test_replicate_prefers_over_error(self) -> None:
        # replicate variance should override upstream error for the member
        wt = {"noise_variance": 2.0}
        result = estimate_pair_noise(wt, None, wt_error_variance=99.0, mut_error_variance=0.5)
        assert result["wt_variance"] == pytest.approx(2.0)  # not 99.0
        assert result["measurement_variance"] == pytest.approx(2.5)

    def test_one_member_only(self) -> None:
        result = estimate_pair_noise(None, None, wt_error_variance=0.4)
        assert result["measurement_variance"] == pytest.approx(0.4)
        assert result["replicate_noise_estimate"] is None
        assert result["source"] == "upstream_error"

    def test_replicate_none_variance_falls_back(self) -> None:
        # replicate dict present but noise_variance None -> fall back to error
        wt = {"noise_variance": None}
        result = estimate_pair_noise(wt, None, wt_error_variance=0.7)
        assert result["wt_variance"] == pytest.approx(0.7)
        assert result["measurement_variance"] == pytest.approx(0.7)
        assert result["source"] == "upstream_error"
