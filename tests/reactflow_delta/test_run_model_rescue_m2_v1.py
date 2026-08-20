from __future__ import annotations

import numpy as np

from scripts.reactflow_delta.run_model_rescue_m2_v1 import summarize


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
