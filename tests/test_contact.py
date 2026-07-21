"""Tests for the contact-map denoising auxiliary."""

import math

import pytest

from reactflow.contact import (
    contact_denoising_bce,
    contact_denoising_logit_gradient,
    soft_contact_matrix,
)


def _softmax(row):
    max_value = max(row)
    exps = [math.exp(value - max_value) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def _marginals(logits):
    return [_softmax(row) for row in logits]


def test_soft_contact_matrix_symmetrizes_partner_marginals():
    marginals = (
        (0.1, 0.0, 0.2, 0.7),
        (0.4, 0.3, 0.0, 0.3),
        (0.5, 0.6, 0.1, 0.0),
    )

    matrix = soft_contact_matrix(marginals)

    assert matrix[0][0] == 0.0
    assert matrix[0][2] == pytest.approx(0.5 * (0.7 + 0.6))
    assert matrix[2][0] == pytest.approx(matrix[0][2])


def test_contact_denoising_bce_balances_positive_and_negative_sets():
    marginals = (
        (0.1, 0.0, 0.2, 0.7),
        (0.4, 0.3, 0.0, 0.3),
        (0.5, 0.6, 0.1, 0.0),
    )
    target = (
        (0, 0, 1),
        (0, 0, 0),
        (1, 0, 0),
    )
    legal = (
        (False, True, True),
        (True, False, True),
        (True, True, False),
    )

    loss = contact_denoising_bce(marginals, target, legal, negative_weight=0.25)

    p02 = 0.5 * (0.7 + 0.6)
    p01 = 0.5 * (0.2 + 0.3)
    p12 = 0.5 * (0.3 + 0.1)
    expected = -math.log(p02) + 0.25 * ((-math.log(1.0 - p01) - math.log(1.0 - p12)) / 2.0)
    assert loss == pytest.approx(expected)


def test_contact_denoising_bce_reweights_long_range_candidates():
    marginals = (
        (0.10, 0.00, 0.20, 0.30, 0.40),
        (0.20, 0.30, 0.00, 0.25, 0.25),
        (0.30, 0.20, 0.20, 0.00, 0.30),
        (0.40, 0.10, 0.20, 0.30, 0.00),
    )
    target = (
        (0, 1, 0, 1),
        (1, 0, 0, 0),
        (0, 0, 0, 0),
        (1, 0, 0, 0),
    )
    legal = tuple(tuple(i != j for j in range(4)) for i in range(4))

    loss = contact_denoising_bce(
        marginals,
        target,
        legal,
        negative_weight=0.5,
        long_range_min_distance=3,
        long_range_weight=4.0,
    )

    p01 = 0.5 * (0.20 + 0.30)
    p03 = 0.5 * (0.40 + 0.10)
    pos_expected = (-math.log(p01) + 4.0 * -math.log(p03)) / 5.0
    neg_terms = []
    neg_weights = []
    for i, j, prob in [
        (0, 2, 0.5 * (0.30 + 0.20)),
        (1, 2, 0.5 * (0.25 + 0.20)),
        (1, 3, 0.5 * (0.25 + 0.20)),
        (2, 3, 0.5 * (0.30 + 0.30)),
    ]:
        weight = 4.0 if (j - i) >= 3 else 1.0
        neg_terms.append(weight * -math.log(1.0 - prob))
        neg_weights.append(weight)
    expected = pos_expected + 0.5 * sum(neg_terms) / sum(neg_weights)
    assert loss == pytest.approx(expected)


def test_contact_denoising_rejects_illegal_target_pair():
    marginals = (
        (0.5, 0.0, 0.2),
        (0.5, 0.3, 0.0),
    )
    target = (
        (0, 1),
        (1, 0),
    )
    legal = (
        (False, False),
        (False, False),
    )

    with pytest.raises(ValueError, match="not legal"):
        contact_denoising_bce(marginals, target, legal)


def test_contact_denoising_logit_gradient_matches_finite_difference():
    logits = [
        [0.4, -0.2, 0.1, 0.3],
        [-0.1, 0.5, 0.0, -0.4],
        [0.2, 0.3, -0.5, 0.1],
    ]
    target = (
        (0, 0, 1),
        (0, 0, 0),
        (1, 0, 0),
    )
    legal = (
        (False, True, True),
        (True, False, True),
        (True, True, False),
    )
    lam = 1.7
    negative_weight = 0.4
    analytic = contact_denoising_logit_gradient(
        _marginals(logits),
        target,
        legal,
        lambda_contact=lam,
        negative_weight=negative_weight,
        long_range_min_distance=2,
        long_range_weight=3.0,
    )

    def loss(rows):
        return lam * contact_denoising_bce(
            _marginals(rows),
            target,
            legal,
            negative_weight=negative_weight,
            long_range_min_distance=2,
            long_range_weight=3.0,
        )

    eps = 1e-6
    for i in range(len(logits)):
        for k in range(len(logits[i])):
            plus = [list(row) for row in logits]
            minus = [list(row) for row in logits]
            plus[i][k] += eps
            minus[i][k] -= eps
            numeric = (loss(plus) - loss(minus)) / (2.0 * eps)
            assert analytic[i][k] == pytest.approx(numeric, abs=1e-6)


def test_contact_gradient_zero_when_disabled():
    marginals = _marginals([[0.0, 0.1, 0.2], [0.2, -0.1, 0.0]])
    target = ((0, 0), (0, 0))
    legal = ((False, True), (True, False))

    grads = contact_denoising_logit_gradient(marginals, target, legal, lambda_contact=0.0)

    assert grads == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_contact_input_validation_edges():
    marginals = _marginals([[0.0, 0.1, 0.2], [0.2, -0.1, 0.0]])
    target = ((0, 0), (0, 0))
    legal = ((False, True), (True, False))

    with pytest.raises(ValueError, match="at least one"):
        soft_contact_matrix(())
    with pytest.raises(ValueError, match="L\\+1"):
        soft_contact_matrix(((0.5, 0.5), (0.5, 0.3, 0.2)))
    with pytest.raises(ValueError, match="non-negative"):
        contact_denoising_bce(marginals, target, legal, negative_weight=-0.1)
    with pytest.raises(ValueError, match="positive"):
        contact_denoising_bce(marginals, target, legal, long_range_min_distance=0)
    with pytest.raises(ValueError, match="non-negative"):
        contact_denoising_bce(marginals, target, legal, long_range_weight=-0.1)
    with pytest.raises(ValueError, match="eps"):
        contact_denoising_bce(marginals, target, legal, eps=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        contact_denoising_logit_gradient(marginals, target, legal, lambda_contact=1.0, negative_weight=-0.1)
    with pytest.raises(ValueError, match="positive"):
        contact_denoising_logit_gradient(marginals, target, legal, lambda_contact=1.0, long_range_min_distance=0)
    with pytest.raises(ValueError, match="non-negative"):
        contact_denoising_logit_gradient(marginals, target, legal, lambda_contact=1.0, long_range_weight=-0.1)
    with pytest.raises(ValueError, match="eps"):
        contact_denoising_logit_gradient(marginals, target, legal, lambda_contact=1.0, eps=0.0)
    with pytest.raises(ValueError, match="match marginals"):
        contact_denoising_bce(marginals, ((0, 0),), legal)
    with pytest.raises(ValueError, match="marginal row"):
        contact_denoising_bce(((0.5, 0.5), (0.2, 0.3, 0.5)), target, legal)
    with pytest.raises(ValueError, match="target pair row"):
        contact_denoising_bce(marginals, ((0,), (0,)), legal)
    with pytest.raises(ValueError, match="legal pair row"):
        contact_denoising_bce(marginals, target, ((False,), (True, False)))


def test_contact_loss_handles_degenerate_candidate_sets():
    marginals = _marginals([[0.0, 0.1, 0.2], [0.2, -0.1, 0.0]])

    no_legal = ((False, False), (False, False))
    no_pairs = ((0, 0), (0, 0))
    assert contact_denoising_bce(marginals, no_pairs, no_legal) == 0.0
    assert contact_denoising_logit_gradient(marginals, no_pairs, no_legal, lambda_contact=1.0) == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    only_positive = ((0, 1), (1, 0))
    legal_pair = ((False, True), (True, False))
    assert contact_denoising_bce(marginals, only_positive, legal_pair) > 0.0

    # With only a negative candidate and zero negative weight, the auxiliary is a no-op.
    assert contact_denoising_bce(marginals, no_pairs, legal_pair, negative_weight=0.0) == 0.0
    assert contact_denoising_logit_gradient(
        marginals,
        no_pairs,
        legal_pair,
        lambda_contact=1.0,
        negative_weight=0.0,
    ) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
