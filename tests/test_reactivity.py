import math

import pytest

from reactflow.constraints import dotbracket_to_matrix
from reactflow.reactivity import (
    ReactivityForwardOperator,
    ReactivityParameters,
    fit_weighted_affine_calibration,
    masked_unit_weights,
    reactivity_consistency_loss,
    weighted_mean,
    weighted_mse,
    weighted_pearson,
)


def test_default_parameters_encode_probe_specific_chemistry():
    params = ReactivityParameters.defaults()

    assert params.coefficient("DMS", "A")[0] > 0
    assert params.coefficient("DMS", "G") == (0.0, 0.0, 0.0)
    assert params.coefficient("2A3", "U")[0] > 0
    with pytest.raises(ValueError, match="invalid RNA base"):
        params.coefficient("DMS", "N")


def test_forward_operator_from_structure_matches_expected_profile():
    sequence = "GGGAAACCC"
    matrix = dotbracket_to_matrix("(((...)))")
    operator = ReactivityForwardOperator()

    profile = operator.from_structure(sequence, matrix, "2A3")

    assert profile[3] > profile[0]
    assert profile[4] > profile[1]
    assert len(profile) == len(sequence)


def test_forward_operator_from_expectations_clamps_probabilities():
    operator = ReactivityForwardOperator()

    profile = operator.from_expectations(
        "AC",
        unpaired_prob=(-1.0, 2.0),
        edge_prob=(2.0, -1.0),
        probe="DMS",
    )

    assert profile[0] == pytest.approx(0.11)
    assert profile[1] == pytest.approx(1.01)


def test_forward_operator_from_expectations_rejects_bad_lengths():
    operator = ReactivityForwardOperator()

    with pytest.raises(ValueError, match="unpaired_prob length"):
        operator.from_expectations("AC", (1.0,), None, "2A3")
    with pytest.raises(ValueError, match="edge_prob length"):
        operator.from_expectations("AC", (1.0, 0.0), (0.0,), "2A3")


def test_weighted_statistics_and_calibration():
    predicted = (1.0, 2.0, 3.0)
    target = (3.0, 5.0, 7.0)
    weights = (1.0, 1.0, 1.0)

    assert weighted_mean(predicted, weights) == pytest.approx(2.0)
    assert weighted_pearson(predicted, target, weights) == pytest.approx(1.0)
    alpha, gamma = fit_weighted_affine_calibration(predicted, target, weights)
    assert alpha == pytest.approx(2.0)
    assert gamma == pytest.approx(1.0)
    assert weighted_mse((3.0, 5.0, 7.0), target, weights) == 0.0


def test_weighted_statistics_reject_empty_effective_weights():
    with pytest.raises(ValueError, match="positive"):
        weighted_mean((1.0,), (0.0,))
    with pytest.raises(ValueError, match="same length"):
        weighted_pearson((1.0,), (1.0, 2.0), (1.0,))
    with pytest.raises(ValueError, match="no finite"):
        fit_weighted_affine_calibration((math.nan,), (1.0,), (1.0,))
    with pytest.raises(ValueError, match="positive"):
        weighted_mse((1.0,), (1.0,), (0.0,))


def test_weighted_pearson_zero_variance_returns_zero_and_singular_calibration_falls_back():
    assert weighted_pearson((1.0, 1.0), (2.0, 3.0), (1.0, 1.0)) == 0.0

    alpha, gamma = fit_weighted_affine_calibration((1.0, 1.0), (2.0, 4.0), (1.0, 1.0))

    assert alpha == 1.0
    assert gamma == pytest.approx(2.0)


def test_reactivity_consistency_loss_is_zero_for_perfect_nonconstant_match():
    target = (0.1, 0.5, 1.0, 0.2)
    weights = (1.0, 1.0, 1.0, 1.0)

    loss = reactivity_consistency_loss(target, target, weights)

    assert loss.total == pytest.approx(0.0)
    assert loss.magnitude == pytest.approx(0.0)
    assert loss.shape == pytest.approx(0.0)
    assert loss.alpha == pytest.approx(1.0)
    assert loss.gamma == pytest.approx(0.0)


def test_reactivity_consistency_loss_handles_scale_calibration():
    predicted = (1.0, 2.0, 3.0)
    target = (3.0, 5.0, 7.0)
    weights = (1.0, 1.0, 1.0)

    loss = reactivity_consistency_loss(predicted, target, weights)

    assert loss.magnitude == pytest.approx(0.0)
    assert loss.shape == pytest.approx(0.0)
    assert loss.alpha == pytest.approx(2.0)
    assert loss.gamma == pytest.approx(1.0)


def test_reactivity_consistency_loss_rejects_bad_lengths():
    with pytest.raises(ValueError, match="same length"):
        reactivity_consistency_loss((1.0,), (1.0, 2.0), (1.0,))


def test_masked_unit_weights_use_probe_base_mask_and_finite_values():
    target = (0.1, 0.2, math.nan, 0.4)

    assert masked_unit_weights("ACGU", "DMS", target) == (1.0, 1.0, 0.0, 0.0)
    assert masked_unit_weights("ACGU", "2A3", target) == (1.0, 1.0, 0.0, 1.0)
