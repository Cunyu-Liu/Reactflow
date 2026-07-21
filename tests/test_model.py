import random

import pytest

from reactflow.dfm import softmax_cross_entropy_gradient
from reactflow.model import (
    DenoiserGradients,
    DenoiserParameters,
    PairwiseDenoiser,
    marginal_pair_matrix,
    sgd_update,
    unpaired_probabilities,
)


def _legal_sequence_and_targets():
    """Return a sequence with a clean nested structure and legal targets.

    Using a legal nested structure as the cross-entropy targets is essential:
    illegal targets point at ``-inf`` logits, making the loss explode and
    destroying finite-difference precision.  Here ``partner = {0:11, 1:10, 2:9,
    3:8}`` is a canonical hairpin so every target class is a legal pair.
    """

    sequence = "GGGGAAAACCCC"
    partner = {0: 11, 1: 10, 2: 9, 3: 8}
    targets = []
    for i in range(len(sequence)):
        if i in partner:
            targets.append(partner[i] + 1)
        elif i in partner.values():
            match = next(k for k, v in partner.items() if v == i)
            targets.append(match + 1)
        else:
            targets.append(0)
    return sequence, targets


def _features(sequence, hidden_seed=0):
    """Build simple deterministic feature vectors (F = 5) for the tests."""

    bases = ("A", "C", "G", "U")
    rng = random.Random(hidden_seed)
    features = []
    for base in sequence.upper():
        one_hot = [1.0 if base == b else 0.0 for b in bases]
        features.append(tuple(one_hot + [rng.random()]))
    return tuple(features)


def _total_loss(model, sequence, features, targets):
    """Scalar denoising cross-entropy summed over positions."""

    from reactflow.dfm import cross_entropy_from_logits

    forward = model.forward(sequence, features)
    return sum(cross_entropy_from_logits(forward.logits[i], targets[i]) for i in range(len(sequence)))


def _flat_param_refs(params):
    """Yield (getter, setter) closures over every scalar parameter."""

    refs = []
    for r, row in enumerate(params.input_weight):
        for c in range(len(row)):
            refs.append((("input_weight", r, c),))
    for r in range(len(params.input_bias)):
        refs.append((("input_bias", r),))
    for r, row in enumerate(params.pair_matrix):
        for c in range(len(row)):
            refs.append((("pair_matrix", r, c),))
    refs.append((("pair_compat",),))
    for r in range(len(params.unpaired_weight)):
        refs.append((("unpaired_weight", r),))
    refs.append((("unpaired_bias",),))
    return refs


def _get(params, key):
    if key[0] in {"input_weight", "pair_matrix"}:
        return getattr(params, key[0])[key[1]][key[2]]
    if key[0] in {"input_bias", "unpaired_weight"}:
        return getattr(params, key[0])[key[1]]
    return getattr(params, key[0])


def _set(params, key, value):
    if key[0] in {"input_weight", "pair_matrix"}:
        getattr(params, key[0])[key[1]][key[2]] = value
    elif key[0] in {"input_bias", "unpaired_weight"}:
        getattr(params, key[0])[key[1]] = value
    else:
        setattr(params, key[0], value)


def _grad(grad, key):
    if key[0] in {"input_weight", "pair_matrix"}:
        return getattr(grad, key[0])[key[1]][key[2]]
    if key[0] in {"input_bias", "unpaired_weight"}:
        return getattr(grad, key[0])[key[1]]
    return getattr(grad, key[0])


def test_forward_marginals_are_normalized_and_mask_illegal_pairs():
    sequence, _ = _legal_sequence_and_targets()
    params = DenoiserParameters.random_init(5, 6, seed=3)
    model = PairwiseDenoiser(params)
    features = _features(sequence)

    forward = model.forward(sequence, features)

    for i, row in enumerate(forward.marginals):
        assert sum(row) == pytest.approx(1.0)
        for j in range(len(sequence)):
            if not forward.legal_pair[i][j]:
                assert row[j + 1] == pytest.approx(0.0, abs=1e-9)


def test_handwritten_backprop_matches_finite_difference():
    sequence, targets = _legal_sequence_and_targets()
    params = DenoiserParameters.random_init(5, 4, seed=7)
    model = PairwiseDenoiser(params)
    features = _features(sequence)

    forward = model.forward(sequence, features)
    grad_logits = [list(softmax_cross_entropy_gradient(forward.logits[i], targets[i])) for i in range(len(sequence))]
    analytic = model.backward(forward, features, grad_logits)

    epsilon = 1e-6
    max_rel = 0.0
    for (key,) in _flat_param_refs(params):
        original = _get(params, key)
        _set(params, key, original + epsilon)
        plus = _total_loss(model, sequence, features, targets)
        _set(params, key, original - epsilon)
        minus = _total_loss(model, sequence, features, targets)
        _set(params, key, original)
        numeric = (plus - minus) / (2 * epsilon)
        analytic_value = _grad(analytic, key)
        denom = max(1.0, abs(numeric), abs(analytic_value))
        max_rel = max(max_rel, abs(numeric - analytic_value) / denom)

    assert max_rel < 1e-5


def test_backward_grad_features_matches_finite_difference():
    """The dL/dfeat_i signal must match central finite differences.

    This is the load-bearing gradient for the C5 frozen-encoder adapter: the
    adapter produces feature slots and needs ``dL/dfeat_i`` to train ``W, b`` by
    hand-written backprop.  We perturb every feature component and compare.
    """

    sequence, targets = _legal_sequence_and_targets()
    params = DenoiserParameters.random_init(5, 4, seed=13)
    model = PairwiseDenoiser(params)
    features = [list(row) for row in _features(sequence, hidden_seed=5)]

    forward = model.forward(sequence, features)
    grad_logits = [list(softmax_cross_entropy_gradient(forward.logits[i], targets[i])) for i in range(len(sequence))]
    analytic = model.backward(forward, features, grad_logits)

    assert analytic.grad_features is not None
    assert len(analytic.grad_features) == len(sequence)

    epsilon = 1e-6
    max_rel = 0.0
    for i in range(len(sequence)):
        for f in range(len(features[i])):
            original = features[i][f]
            features[i][f] = original + epsilon
            plus = _total_loss(model, sequence, features, targets)
            features[i][f] = original - epsilon
            minus = _total_loss(model, sequence, features, targets)
            features[i][f] = original
            numeric = (plus - minus) / (2 * epsilon)
            analytic_value = analytic.grad_features[i][f]
            denom = max(1.0, abs(numeric), abs(analytic_value))
            max_rel = max(max_rel, abs(numeric - analytic_value) / denom)

    assert max_rel < 1e-5


def test_random_init_is_deterministic_and_validates():
    a = DenoiserParameters.random_init(5, 4, seed=1)
    b = DenoiserParameters.random_init(5, 4, seed=1)

    assert a.input_weight == b.input_weight
    assert a.pair_matrix == b.pair_matrix
    assert a.pair_compat == 0.5
    assert a.hidden_size == 4
    assert a.feature_size == 5
    with pytest.raises(ValueError, match="positive"):
        DenoiserParameters.random_init(0, 4)


def test_forward_rejects_wrong_feature_shapes():
    sequence = "GGGGAAAACCCC"
    params = DenoiserParameters.random_init(5, 4, seed=0)
    model = PairwiseDenoiser(params)

    with pytest.raises(ValueError, match="features length"):
        model.forward(sequence, _features("GGG"))
    with pytest.raises(ValueError, match="wrong dimension"):
        model.forward(sequence, tuple((0.0,) for _ in sequence))


def test_backward_rejects_wrong_grad_shapes():
    sequence, targets = _legal_sequence_and_targets()
    params = DenoiserParameters.random_init(5, 4, seed=0)
    model = PairwiseDenoiser(params)
    features = _features(sequence)
    forward = model.forward(sequence, features)

    with pytest.raises(ValueError, match="grad_logits length"):
        model.backward(forward, features, [[0.0]])
    bad = [[0.0] for _ in sequence]
    with pytest.raises(ValueError, match="length L\\+1"):
        model.backward(forward, features, bad)


def test_sgd_update_decreases_denoising_loss():
    sequence, targets = _legal_sequence_and_targets()
    params = DenoiserParameters.random_init(5, 6, seed=11)
    model = PairwiseDenoiser(params)
    features = _features(sequence)

    before = _total_loss(model, sequence, features, targets)
    for _ in range(15):
        forward = model.forward(sequence, features)
        grad_logits = [list(softmax_cross_entropy_gradient(forward.logits[i], targets[i])) for i in range(len(sequence))]
        gradients = model.backward(forward, features, grad_logits)
        sgd_update(params, gradients, 0.3)
    after = _total_loss(model, sequence, features, targets)

    assert after < before


def test_sgd_update_rejects_non_positive_learning_rate():
    params = DenoiserParameters.random_init(5, 4, seed=0)
    zero_grad = DenoiserGradients(
        input_weight=[[0.0 for _ in row] for row in params.input_weight],
        input_bias=[0.0 for _ in params.input_bias],
        pair_matrix=[[0.0 for _ in row] for row in params.pair_matrix],
        pair_compat=0.0,
        unpaired_weight=[0.0 for _ in params.unpaired_weight],
        unpaired_bias=0.0,
    )
    with pytest.raises(ValueError, match="learning_rate"):
        sgd_update(params, zero_grad, 0.0)


def test_marginal_pair_matrix_is_symmetric_with_zero_diagonal():
    sequence, _ = _legal_sequence_and_targets()
    params = DenoiserParameters.random_init(5, 4, seed=2)
    model = PairwiseDenoiser(params)
    features = _features(sequence)
    marginals = model.marginals(sequence, features)

    matrix = marginal_pair_matrix(marginals)
    size = len(sequence)
    for i in range(size):
        assert matrix[i][i] == 0.0
        for j in range(size):
            assert matrix[i][j] == pytest.approx(matrix[j][i])

    q = unpaired_probabilities(marginals)
    assert all(0.0 <= value <= 1.0 for value in q)
