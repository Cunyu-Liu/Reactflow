from __future__ import annotations

import numpy as np
import pytest

from scripts.reactflow_delta.run_model_rescue_m3_v1 import (
    choose_from_inner,
    combine_seed_predictions,
    eligible_specs,
    score_wt_anchor_signed_delta_mae,
)


class _Construct:
    sequence = "AC"
    wt_observed = np.array([True, True])
    wt_reactivity = np.array([0.0, 0.2])


class _Record:
    puzzle = "P01"
    method = "M"
    construct_id = "P01_M"
    wt_id = "P01_M_wt"
    pos = 0
    ref = "A"
    alt = "G"


class _Universe:
    def get_construct(self, _construct_id):
        return _Construct()

    def mutant_full_profile(self, *_args):
        return np.array([0.1, 0.5]), np.array([0.1, 0.1])


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


def test_wt_anchor_signed_delta_mae_uses_method_balanced_evaluator():
    score = score_wt_anchor_signed_delta_mae(_Universe(), [_Record()])
    assert np.isclose(score, np.mean([0.1, 0.3]))
