from __future__ import annotations

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v10 import parameter_count
from scripts.reactflow_delta.puzzle_set_meta_context import (
    make_exact_full_model_pair,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    EXPECTED_RESIDUAL_PARAMETERS,
    calibrated_distribution,
    fit_residual_pair,
    make_exact_residual_pair,
)


def _context(length: int, observed: bool = True):
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.ones(length)
    observed_mask = torch.full((length,), float(observed))
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed_mask, position, region


def _training_batch(*, seven_cells: bool = False):
    contexts = [
        _context(4, observed=index != 0 or not seven_cells) for index in range(8)
    ]
    cells = []
    start = 1 if seven_cells else 0
    for focal in range(start, 8):
        edit = torch.tensor([focal % 4])
        distance = torch.arange(4)[None, :] - edit[:, None]
        cells.append(
            {
                "focal_construct_index": focal,
                "construct_id": f"construct{focal}",
                "edit_index": edit,
                "signed_distance": distance.float(),
                "refs": ["A"],
                "alts": ["G"],
                "feature41_point": torch.zeros(1, 4),
                "prediction_mask": torch.ones(1, 4, dtype=torch.bool),
                "target": torch.full((1, 4), float(focal + 1) / 10.0),
                "qualified_mask": torch.ones(1, 4, dtype=torch.bool),
                "wt": torch.zeros(4),
                "feature41_basis": np.zeros((1, 4, 41), dtype=np.float32),
                "direct_features": np.zeros((1, 4, 201), dtype=np.float32),
            }
        )
    return {
        "puzzle": "P20" if seven_cells else "P01",
        "contexts": contexts,
        "cells": cells,
    }


def test_residual_heads_are_exact_v10_family_matches() -> None:
    candidate, null = make_exact_residual_pair(seed=7, device="cpu")
    assert parameter_count(candidate) == EXPECTED_RESIDUAL_PARAMETERS
    assert parameter_count(null) == EXPECTED_RESIDUAL_PARAMETERS
    for left, right in zip(candidate.parameters(), null.parameters()):
        assert torch.equal(left, right)


def test_residual_fit_freezes_both_point_models_and_supports_p20_context() -> None:
    candidate, null = make_exact_full_model_pair(seed=11)
    candidate_before = {
        name: value.detach().clone() for name, value in candidate.state_dict().items()
    }
    null_before = {
        name: value.detach().clone() for name, value in null.state_dict().items()
    }
    fitted = fit_residual_pair(
        [_training_batch(seven_cells=True)],
        candidate=candidate,
        null=null,
        epochs=1,
        seed=0,
        device="cpu",
    )
    assert fitted["n_calibration_cells"] == 7
    assert set(fitted["histories"]) == {"candidate", "null"}
    assert all(len(history) == 1 for history in fitted["histories"].values())
    assert all(not parameter.requires_grad for parameter in candidate.parameters())
    assert all(not parameter.requires_grad for parameter in null.parameters())
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, candidate_before[name])
    for name, value in null.state_dict().items():
        assert torch.equal(value, null_before[name])


def test_calibrated_distribution_preserves_point_median() -> None:
    candidate, _null = make_exact_full_model_pair(seed=21)
    fitted = fit_residual_pair(
        [_training_batch()],
        candidate=candidate,
        null=make_exact_full_model_pair(seed=21)[1],
        epochs=1,
        seed=0,
        device="cpu",
    )
    point = np.asarray([-0.2, 0.0, 0.3], dtype=np.float64)
    distribution = calibrated_distribution(
        point=point,
        feature41=np.zeros((3, 41), dtype=np.float32),
        direct_features=np.zeros((3, 201), dtype=np.float32),
        head=fitted["heads"]["candidate"],
        standardizer=fitted["standardizers"]["candidate"],
    )
    weights = torch.tensor(distribution["weights"])
    locations = torch.tensor(distribution["locations"])
    scales = torch.tensor(distribution["scales"])
    point_tensor = torch.tensor(point)
    cdf = torch.sum(
        weights
        * torch.special.ndtr((point_tensor[:, None] - locations) / scales),
        dim=-1,
    )
    assert torch.allclose(cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0)
    assert np.all(distribution["scales"] > 0.0)
    assert np.allclose(distribution["weights"].sum(axis=1), 1.0)
