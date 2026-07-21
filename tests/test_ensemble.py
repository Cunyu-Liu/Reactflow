"""Tests for the variance-aware (heteroscedastic) reactivity observation model."""

import math

import pytest

from reactflow.ensemble import (
    heteroscedastic_reactivity_logit_gradient,
    heteroscedastic_reactivity_nll,
    structural_variance,
)


def _softmax(row):
    max_v = max(row)
    exps = [math.exp(v - max_v) for v in row]
    total = sum(exps)
    return [v / total for v in exps]


def test_structural_variance_matches_bernoulli_formula_and_floor():
    q = (0.0, 0.5, 1.0)
    a = (2.0, 2.0, 2.0)
    variances = structural_variance(q, a, beta=1.0, tau_squared=0.01)
    # beta a^2 q(1-q) + tau^2
    assert variances[0] == pytest.approx(0.01)
    assert variances[1] == pytest.approx(1.0 * 4.0 * 0.25 + 0.01)
    assert variances[2] == pytest.approx(0.01)


def test_structural_variance_floor_and_negative_beta():
    # tau below the floor is lifted to the floor so the NLL log-term stays finite.
    variances = structural_variance((0.5,), (1.0,), beta=0.0, tau_squared=0.0)
    assert variances[0] == pytest.approx(1e-6)
    with pytest.raises(ValueError, match="beta"):
        structural_variance((0.5,), (1.0,), beta=-1.0, tau_squared=1.0)


def test_nll_reduces_to_scaled_mse_plus_constant_when_beta_zero():
    q = (0.2, 0.8)
    target = (0.5, 0.4)
    weights = (1.0, 1.0)
    a = (1.0, 1.0)
    c = (0.0, 0.0)
    alpha, gamma, tau2 = 1.0, 0.0, 0.25
    nll = heteroscedastic_reactivity_nll(
        q, target, weights, a, c, alpha=alpha, gamma=gamma, beta=0.0, tau_squared=tau2
    )
    # With beta=0: v=tau2 constant, ell = mean_i (mu-r)^2/(2 tau2) + 0.5 log tau2.
    manual = 0.0
    for qi, ri in zip(q, target):
        mu = qi
        manual += (mu - ri) ** 2 / (2 * tau2) + 0.5 * math.log(tau2)
    manual /= 2
    assert nll == pytest.approx(manual)


def test_nll_zero_when_no_valid_weights():
    nll = heteroscedastic_reactivity_nll(
        (0.5,), (float("nan"),), (0.0,), (1.0,), (0.0,),
        alpha=1.0, gamma=0.0, beta=1.0, tau_squared=0.1,
    )
    assert nll == 0.0


@pytest.mark.parametrize("beta", [0.0, 0.7])
def test_hetero_logit_gradient_matches_finite_difference(beta):
    logits = [[0.4, -0.1, 0.2], [-0.3, 0.5, 0.0], [0.1, 0.2, -0.4]]
    marginals = [_softmax(row) for row in logits]
    target = (0.7, 0.35, 0.5)
    weights = (1.0, 0.5, 2.0)
    a_values = (1.0, 0.8, 1.2)
    c_values = (0.02, 0.0, 0.01)
    alpha, gamma, tau2, lam = 1.1, 0.05, 0.2, 1.3

    analytic = heteroscedastic_reactivity_logit_gradient(
        marginals, target, weights, a_values, c_values,
        alpha=alpha, gamma=gamma, beta=beta, tau_squared=tau2, lambda_calib=lam,
    )

    def loss(rows):
        m = [_softmax(r) for r in rows]
        q = [row[0] for row in m]
        return lam * heteroscedastic_reactivity_nll(
            q, target, weights, a_values, c_values,
            alpha=alpha, gamma=gamma, beta=beta, tau_squared=tau2,
        )

    eps = 1e-6
    for i in range(len(logits)):
        for k in range(len(logits[i])):
            plus = [list(r) for r in logits]
            minus = [list(r) for r in logits]
            plus[i][k] += eps
            minus[i][k] -= eps
            numeric = (loss(plus) - loss(minus)) / (2 * eps)
            assert analytic[i][k] == pytest.approx(numeric, abs=1e-6)


def test_hetero_gradient_shape_validation():
    marginals = [[0.5, 0.3, 0.2]]
    with pytest.raises(ValueError, match="sequence length"):
        heteroscedastic_reactivity_logit_gradient(
            marginals, (0.5, 0.5), (1.0,), (1.0,), (0.0,),
            alpha=1.0, gamma=0.0, beta=1.0, tau_squared=0.1, lambda_calib=1.0,
        )
    with pytest.raises(ValueError, match="beta"):
        heteroscedastic_reactivity_logit_gradient(
            marginals, (0.5,), (1.0,), (1.0,), (0.0,),
            alpha=1.0, gamma=0.0, beta=-1.0, tau_squared=0.1, lambda_calib=1.0,
        )
