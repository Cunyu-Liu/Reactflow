from __future__ import annotations

import numpy as np
import pytest

from scripts.reactflow_delta.run_model_rescue_m3_v1 import (
    choose_from_inner,
    combine_seed_predictions,
    eligible_specs,
)


def test_m3_fails_closed_without_m2_pass():
    with pytest.raises(ValueError):
        eligible_specs({"overall_status": "M2_NO_RESCUE_CANDIDATE"})


def test_sparse_lambda_variants_are_both_exposed_to_inner_selection():
    specs = eligible_specs(
        {
            "overall_status": "M2_SCREEN_PASS",
            "m3_eligible_families": [
                "b1_rfd_direct_aligned",
                "sparse_delta_mdn_inner_selected_lambda",
            ],
        }
    )
    assert [x.model_id for x in specs] == [
        "b1_rfd_direct_aligned",
        "sparse_delta_mdn_h0",
        "sparse_delta_mdn_h01",
    ]


def test_inner_selection_requires_neither_primary_worse_than_b1():
    inner = {
        "b1_rfd_direct_aligned": {"crps": 0.20, "signed_delta_mae": 0.25},
        "l2_aligned_rank2": {"crps": 0.19, "signed_delta_mae": 0.26},
        "sparse_delta_mdn_h0": {"crps": 0.19, "signed_delta_mae": 0.24},
    }
    selected = choose_from_inner(inner, list(inner))
    assert selected["selected_candidate"] == "sparse_delta_mdn_h0"
    assert selected["selected_comparator"] == "b1_rfd_direct_aligned"


def test_five_seed_predictions_form_one_normalized_mixture():
    rows = []
    for seed in range(5):
        rows.append(
            {
                "keys": np.array(["k"], dtype=object),
                "locations": np.array([[float(seed), float(seed + 1)]]),
                "scales": np.ones((1, 2)),
                "weights": np.array([[0.25, 0.75]]),
            }
        )
    mixture = combine_seed_predictions(rows)
    assert mixture["locations"].shape == (1, 10)
    assert np.allclose(mixture["weights"].sum(axis=1), 1.0)
    assert np.allclose(mixture["weights"][0, :2], [0.05, 0.15])
