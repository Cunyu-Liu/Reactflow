from __future__ import annotations

import numpy as np
import pytest

from scripts.reactflow_delta.model_rescue_v3 import CANDIDATE
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE
from scripts.reactflow_delta.run_model_rescue_v3_formal import (
    combine_five_seed_predictions,
)


def _prediction(components: int, offset: float) -> dict[str, np.ndarray]:
    locations = np.full((2, components), offset)
    weights = np.full((2, components), 1.0 / components)
    return {
        "keys": np.asarray(["k0", "k1"], dtype=object),
        "locations": locations,
        "scales": np.full((2, components), 0.2),
        "weights": weights,
    }


def test_five_seed_candidate_mixture_preserves_equal_seed_mass() -> None:
    predictions = [_prediction(2, float(seed)) for seed in range(5)]
    combined = combine_five_seed_predictions(predictions, CANDIDATE)
    assert combined["locations"].shape == (2, 10)
    np.testing.assert_allclose(combined["weights"].sum(axis=1), 1.0)
    for seed in range(5):
        np.testing.assert_allclose(
            combined["weights"][:, 2 * seed : 2 * seed + 2].sum(axis=1), 0.2
        )
    np.testing.assert_allclose(combined["point_mean"], 2.0)


def test_five_seed_combiner_rejects_key_or_component_mismatch() -> None:
    predictions = [_prediction(1, float(seed)) for seed in range(5)]
    combined = combine_five_seed_predictions(predictions, BASELINE)
    assert combined["locations"].shape == (2, 5)
    predictions[-1]["keys"] = np.asarray(["wrong", "k1"], dtype=object)
    with pytest.raises(ValueError, match="key universes"):
        combine_five_seed_predictions(predictions, BASELINE)
