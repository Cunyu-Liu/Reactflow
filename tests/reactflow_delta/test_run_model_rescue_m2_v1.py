from __future__ import annotations

import numpy as np

from scripts.reactflow_delta.run_model_rescue_m2_v1 import score_predictions, summarize


class _Construct:
    sequence = "AC"
    wt_observed = np.array([True, True])
    wt_reactivity = np.array([0.0, 0.2])


class _Record:
    puzzle = "P01"
    method = "M"
    construct_id = "P01_M"
    wt_id = "P01_M_wt"
    design_pos = 0
    full_pos = 0
    ref = "A"
    alt = "G"


class _Universe:
    def get_construct(self, _construct_id):
        return _Construct()

    def mutant_full_profile(self, *_args):
        return np.array([0.1, 0.4]), np.array([0.1, 0.1])


def test_two_fold_summary_requires_both_primary_directions():
    def candidate(crps: float, delta: float):
        return {"score": {"crps": crps, "signed_delta_mae": delta}}

    folds = [
        {
            "candidates": {
                "b1_rfd_direct_aligned": candidate(0.20, 0.25),
                "l2_aligned_rank2": candidate(0.19, 0.26),
                "sparse_delta_mdn_h0": candidate(0.19, 0.24),
                "sparse_delta_mdn_h01": candidate(0.21, 0.23),
            }
        },
        {
            "candidates": {
                "b1_rfd_direct_aligned": candidate(0.22, 0.27),
                "l2_aligned_rank2": candidate(0.21, 0.28),
                "sparse_delta_mdn_h0": candidate(0.20, 0.25),
                "sparse_delta_mdn_h01": candidate(0.23, 0.25),
            }
        },
    ]
    result = summarize(folds, smoke=True)["candidates"]
    assert result["sparse_delta_mdn_h0"]["directional_both_primary"]
    assert not result["l2_aligned_rank2"]["directional_both_primary"]
    assert not result["sparse_delta_mdn_h01"]["directional_both_primary"]
    assert np.isclose(result["sparse_delta_mdn_h0"]["crps_gain_vs_b1"], 0.015)


def test_score_reports_fractional_full_output_coverage_and_failures():
    keys = np.array(
        [
            "openknot_m2|P01|M|P01_M|0|A>G|0",
            "openknot_m2|P01|M|P01_M|0|A>G|1",
        ],
        dtype=object,
    )
    prediction = {
        "keys": keys,
        "locations": np.array([[0.1], [0.4]]),
        "scales": np.array([[0.1], [0.1]]),
        "weights": np.ones((2, 1)),
    }
    result = score_predictions(prediction, _Universe(), [_Record()])
    assert result["registered_prediction_coverage"] == 1.0
    assert result["failure_rate"] == 0.0
    assert result["n_registered_prediction_keys_expected"] == 2

    prediction["locations"][1, 0] = np.nan
    failed = score_predictions(prediction, _Universe(), [_Record()])
    assert failed["registered_prediction_coverage"] == 1.0
    assert failed["failure_rate"] == 0.5
