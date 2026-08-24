from __future__ import annotations

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v2 import gaussian_mixture_crps_torch
from scripts.reactflow_delta.model_rescue_v10 import (
    INPUT_WIDTH,
    CapacitySymmetricResidual,
    MedianAsymmetricResidual,
    TrainOnlyStandardizer,
    calibration_input,
    initialize_asymmetric_from_symmetric,
    mixture_cdf_at_point,
    parameter_count,
)


def test_v10_parameter_counts_match_frozen_contract() -> None:
    assert parameter_count(CapacitySymmetricResidual()) == 63491
    assert parameter_count(MedianAsymmetricResidual()) == 63748


def test_v10_asymmetric_initialization_is_exact_symmetric_nested_null() -> None:
    torch.manual_seed(17)
    symmetric = CapacitySymmetricResidual().double()
    asymmetric = MedianAsymmetricResidual().double()
    initialize_asymmetric_from_symmetric(symmetric, asymmetric)
    inputs = torch.randn(128, INPUT_WIDTH, dtype=torch.float64)
    point = torch.randn(128, dtype=torch.float64)
    sw, sl, ss = symmetric(point, inputs)
    aw, al, ass = asymmetric(point, inputs)
    assert torch.equal(sw, aw)
    assert torch.equal(ss, ass)
    assert torch.equal(sl, al)
    target = torch.randn(128, dtype=torch.float64)
    assert torch.equal(
        gaussian_mixture_crps_torch(sl, ss, sw, target),
        gaussian_mixture_crps_torch(al, ass, aw, target),
    )


def test_v10_asymmetric_cdf_constraint_and_gradients_are_finite() -> None:
    torch.manual_seed(19)
    model = MedianAsymmetricResidual()
    inputs = torch.randn(1024, INPUT_WIDTH)
    point = torch.randn(1024)
    weights, locations, scales = model(point, inputs)
    cdf = mixture_cdf_at_point(point, weights, locations, scales)
    assert torch.allclose(cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0)
    target = torch.randn(1024)
    loss = gaussian_mixture_crps_torch(
        locations, scales, weights, target
    ).mean()
    loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert float(model.output_layer.weight.grad[3].abs().sum()) > 0.0
    assert float(model.output_layer.bias.grad[3].abs()) > 0.0


def test_v10_train_only_standardization_and_input_width() -> None:
    feature41 = np.arange(5 * 41, dtype=np.float64).reshape(5, 41)
    point = np.linspace(-1.0, 1.0, 5)
    direct = np.arange(5 * 201, dtype=np.float64).reshape(5, 201)
    values = calibration_input(feature41, point, direct)
    assert values.shape == (5, INPUT_WIDTH)
    standardizer = TrainOnlyStandardizer.fit([values[:3]])
    train = standardizer.transform_numpy(values[:3])
    assert np.allclose(train.mean(axis=0)[np.std(values[:3], axis=0) > 0], 0.0)
    held = standardizer.transform_numpy(values[3:])
    assert held.shape == (2, INPUT_WIDTH)
