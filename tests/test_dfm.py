import math
import random

import pytest

from reactflow.dfm import (
    conditional_rate_matrix,
    cross_entropy_from_logits,
    denoising_cross_entropy,
    euler_step_distribution,
    log_softmax,
    mixture_path_marginal,
    mixture_path_time_derivative,
    posterior_transition_rates,
    sample_path_index,
    softmax,
    softmax_cross_entropy_gradient,
    uniform_source,
)


def _numeric_gradient(logits, target_index, epsilon=1e-6):
    """Central finite-difference gradient of cross-entropy w.r.t. logits."""

    grad = []
    for k in range(len(logits)):
        plus = list(logits)
        minus = list(logits)
        plus[k] += epsilon
        minus[k] -= epsilon
        derivative = (cross_entropy_from_logits(plus, target_index) - cross_entropy_from_logits(minus, target_index)) / (2 * epsilon)
        grad.append(derivative)
    return grad


def test_softmax_is_normalized_and_shift_invariant():
    logits = [2.0, -1.0, 0.5]
    probs = softmax(logits)

    assert sum(probs) == pytest.approx(1.0)
    shifted = softmax([value + 10.0 for value in logits])
    for base, moved in zip(probs, shifted):
        assert base == pytest.approx(moved)


def test_log_softmax_matches_log_of_softmax():
    logits = [0.3, 1.2, -0.7, 2.0]
    log_probs = log_softmax(logits)
    probs = softmax(logits)

    for lp, p in zip(log_probs, probs):
        assert lp == pytest.approx(math.log(p))


def test_softmax_and_log_softmax_reject_empty():
    with pytest.raises(ValueError, match="non-empty"):
        softmax([])
    with pytest.raises(ValueError, match="non-empty"):
        log_softmax([])


def test_cross_entropy_gradient_matches_finite_difference():
    logits = [0.5, -1.5, 2.0, 0.1]
    for target in range(len(logits)):
        analytic = softmax_cross_entropy_gradient(logits, target)
        numeric = _numeric_gradient(logits, target)
        for a, n in zip(analytic, numeric):
            assert a == pytest.approx(n, abs=1e-6)


def test_cross_entropy_gradient_equals_softmax_minus_onehot():
    logits = [1.0, 0.0, -1.0]
    grad = softmax_cross_entropy_gradient(logits, 1)
    probs = softmax(logits)

    assert grad[0] == pytest.approx(probs[0])
    assert grad[1] == pytest.approx(probs[1] - 1.0)
    assert grad[2] == pytest.approx(probs[2])


def test_cross_entropy_helpers_reject_out_of_range_targets():
    with pytest.raises(ValueError, match="out of range"):
        cross_entropy_from_logits([0.0, 1.0], 5)
    with pytest.raises(ValueError, match="out of range"):
        softmax_cross_entropy_gradient([0.0, 1.0], -1)


def test_uniform_source_and_mixture_path_endpoints():
    source = uniform_source(4)
    assert source == pytest.approx((0.25, 0.25, 0.25, 0.25))

    at_zero = mixture_path_marginal(0.0, 2, source)
    at_one = mixture_path_marginal(1.0, 2, source)

    assert at_zero == pytest.approx(source)
    assert at_one == pytest.approx((0.0, 0.0, 1.0, 0.0))
    for t in (0.1, 0.5, 0.9):
        assert sum(mixture_path_marginal(t, 2, source)) == pytest.approx(1.0)


def test_mixture_path_time_derivative_is_onehot_minus_source():
    source = uniform_source(3)
    derivative = mixture_path_time_derivative(1, source)

    assert derivative == pytest.approx((-1.0 / 3, 2.0 / 3, -1.0 / 3))


def test_mixture_path_validation():
    source = uniform_source(3)
    with pytest.raises(ValueError, match="t must lie"):
        mixture_path_marginal(1.5, 0, source)
    with pytest.raises(ValueError, match="out of range"):
        mixture_path_marginal(0.5, 9, source)
    with pytest.raises(ValueError, match="out of range"):
        mixture_path_time_derivative(9, source)
    with pytest.raises(ValueError, match="positive"):
        uniform_source(0)


def test_sample_path_index_is_deterministic_and_in_range():
    source = uniform_source(5)
    rng = random.Random(0)
    draws = [sample_path_index(0.5, 2, source, rng=rng) for _ in range(50)]

    assert all(0 <= value < 5 for value in draws)
    # At t close to 1 the path collapses onto the data index.
    rng2 = random.Random(1)
    near_one = [sample_path_index(0.999, 3, source, rng=rng2) for _ in range(20)]
    assert near_one.count(3) >= 18


def test_conditional_rate_matrix_rows_sum_to_zero_and_flux_into_data():
    source = uniform_source(3)
    rates = conditional_rate_matrix(0.5, 0, source)

    for row in rates:
        assert sum(row) == pytest.approx(0.0, abs=1e-12)
    # Only transitions toward the data class receive positive off-diagonal rate.
    assert rates[1][0] > 0
    assert rates[2][0] > 0
    assert rates[0][1] == pytest.approx(0.0)
    assert rates[0][2] == pytest.approx(0.0)


def test_conditional_rate_matrix_satisfies_master_equation():
    source = uniform_source(4)
    t = 0.4
    rates = conditional_rate_matrix(t, 1, source)
    marginal = mixture_path_marginal(t, 1, source)
    derivative = mixture_path_time_derivative(1, source)

    for j in range(4):
        flux = sum(rates[z][j] * marginal[z] for z in range(4))
        assert flux == pytest.approx(derivative[j], abs=1e-12)


def test_conditional_rate_matrix_rejects_t_at_one():
    source = uniform_source(3)
    with pytest.raises(ValueError, match="t in"):
        conditional_rate_matrix(1.0, 0, source)


def test_posterior_transition_rates_match_delta_posterior():
    source = uniform_source(3)
    t = 0.3
    posterior = (1.0, 0.0, 0.0)
    row = posterior_transition_rates(t, 1, posterior, source)
    reference = conditional_rate_matrix(t, 0, source)[1]

    for value, ref in zip(row, reference):
        assert value == pytest.approx(ref, abs=1e-12)


def test_posterior_transition_rates_validation():
    source = uniform_source(3)
    with pytest.raises(ValueError, match="same length"):
        posterior_transition_rates(0.5, 0, (1.0, 0.0), source)
    with pytest.raises(ValueError, match="out of range"):
        posterior_transition_rates(0.5, 9, (1.0, 0.0, 0.0), source)
    with pytest.raises(ValueError, match="t in"):
        posterior_transition_rates(1.0, 0, (1.0, 0.0, 0.0), source)


def test_euler_step_moves_mass_toward_data_class():
    source = uniform_source(3)
    t = 0.2
    rates = conditional_rate_matrix(t, 0, source)
    current = mixture_path_marginal(t, 0, source)
    stepped = euler_step_distribution(current, rates, 0.05)

    assert sum(stepped) == pytest.approx(1.0)
    assert stepped[0] > current[0]


def test_euler_step_validation():
    source = uniform_source(3)
    rates = conditional_rate_matrix(0.2, 0, source)
    with pytest.raises(ValueError, match="square"):
        euler_step_distribution((1.0, 0.0), rates, 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        euler_step_distribution((1 / 3, 1 / 3, 1 / 3), rates, -0.1)


def test_denoising_cross_entropy_averages_positions():
    logit_rows = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    loss = denoising_cross_entropy(logit_rows, [0, 1])
    manual = 0.5 * (cross_entropy_from_logits(logit_rows[0], 0) + cross_entropy_from_logits(logit_rows[1], 1))

    assert loss == pytest.approx(manual)
    with pytest.raises(ValueError, match="same length"):
        denoising_cross_entropy(logit_rows, [0])
    with pytest.raises(ValueError, match="at least one"):
        denoising_cross_entropy([], [])
