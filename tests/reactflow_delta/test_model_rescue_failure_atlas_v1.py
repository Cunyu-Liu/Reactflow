from __future__ import annotations

import numpy as np

from scripts.reactflow_delta.run_model_rescue_failure_atlas_v1 import (
    distance_bin,
    magnitude_bin,
    mixture_moments,
    shapley_mean_scale_crps,
)


def test_fixed_response_and_distance_bins_cover_contract_boundaries():
    assert magnitude_bin(np.array([0.0, 0.05, 0.051, 0.2, 0.5, 0.501])).tolist() == [
        "near_zero_le_0.05",
        "near_zero_le_0.05",
        "small_0.05_0.20",
        "small_0.05_0.20",
        "medium_0.20_0.50",
        "tail_gt_0.50",
    ]
    assert distance_bin(np.array([0, 1, -5, 6, -20, 21, -50, 51])).tolist() == [
        "edit_site_0",
        "near_1_5",
        "near_1_5",
        "mid_6_20",
        "mid_6_20",
        "far_21_50",
        "far_21_50",
        "distal_gt_50",
    ]


def test_mixture_moments_include_between_seed_variance():
    locs = [np.array([0.0]), np.array([2.0])]
    scales = [np.array([1.0]), np.array([1.0])]
    mean, sd = mixture_moments(locs, scales)
    assert np.allclose(mean, [1.0])
    assert np.allclose(sd, [np.sqrt(2.0)])


def test_shapley_mean_scale_contributions_sum_to_total_crps_gain():
    y = np.array([0.2, -0.4, 1.0])
    loc0 = [np.array([0.0, 0.0, 0.0]), np.array([0.1, -0.1, 0.2])]
    loc1 = [np.array([0.2, -0.2, 0.7]), np.array([0.2, -0.3, 0.8])]
    scale0 = [np.full(3, 0.8), np.full(3, 0.9)]
    scale1 = [np.full(3, 0.3), np.full(3, 0.4)]
    split = shapley_mean_scale_crps(loc0, scale0, loc1, scale1, y)
    assert np.allclose(split["mean_gain"] + split["scale_gain"], split["total_gain"])
    assert np.allclose(split["total_gain"], split["rank0"] - split["rankpos"])
