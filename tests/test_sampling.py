import pytest

from reactflow.model import DenoiserParameters, PairwiseDenoiser
from reactflow.sampling import (
    ensemble_unpaired_probability,
    euler_transition_kernel,
    pairing_frequency_matrix,
    sample_structure,
    sample_structures,
)
from reactflow.train import FEATURE_SIZE


def _denoiser(seed: int = 0) -> PairwiseDenoiser:
    """Build a small deterministic denoiser directly from initialized weights.

    Training convergence is exercised in ``test_train.py``; the sampler tests
    only need a valid forward pass, so we skip the training loop for speed.
    """

    params = DenoiserParameters.random_init(FEATURE_SIZE, 8, seed=seed, scale=0.2)
    return PairwiseDenoiser(params, min_loop=3)


def test_euler_transition_kernel_matches_closed_form_and_normalizes():
    # Rate row already sums to zero: T = 1[j=z] + dt * R, dt = 1.
    kernel = euler_transition_kernel(0, [-0.5, 0.3, 0.2], 1.0)

    assert kernel == pytest.approx((0.5, 0.3, 0.2))
    assert sum(kernel) == pytest.approx(1.0)


def test_euler_transition_kernel_clamps_negative_self_transition_and_renormalizes():
    # Large dt drives the self-transition negative: base row [-1, 1, 1] -> clamp -> renorm.
    kernel = euler_transition_kernel(0, [-2.0, 1.0, 1.0], 1.0)

    assert kernel == pytest.approx((0.0, 0.5, 0.5))
    assert sum(kernel) == pytest.approx(1.0)


def test_euler_transition_kernel_rejects_bad_arguments():
    with pytest.raises(ValueError, match="out of range"):
        euler_transition_kernel(5, [0.0, 0.0, 0.0], 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        euler_transition_kernel(0, [0.0, 0.0], -0.1)
    # Every mass clamps to zero -> the row cannot be normalized.
    with pytest.raises(ValueError, match="non-positive mass"):
        euler_transition_kernel(1, [0.0, -1.0, 0.0], 2.0)


def test_sample_structure_is_legal_across_seeds():
    model = _denoiser()
    sequence = "GGGAAACCC"

    for seed in range(6):
        structure = sample_structure(model, sequence, num_steps=8, seed=seed)

        assert structure.validation.valid
        assert structure.sequence == sequence
        assert len(structure.partner_classes) == len(sequence)
        assert len(structure.pair_matrix) == len(sequence)
        assert all(len(row) == len(sequence) for row in structure.pair_matrix)
        assert structure.num_steps == 8


def test_sample_structure_rejects_bad_arguments():
    model = _denoiser()
    with pytest.raises(ValueError, match="num_steps"):
        sample_structure(model, "GGGAAACCC", num_steps=0)
    with pytest.raises(ValueError, match="non-empty"):
        sample_structure(model, "", num_steps=4)


def test_sample_structures_is_deterministic_and_all_legal():
    model = _denoiser()
    sequence = "GGGAAACCC"

    first = sample_structures(model, sequence, num_samples=8, num_steps=8, seed=0)
    second = sample_structures(model, sequence, num_samples=8, num_steps=8, seed=0)

    assert len(first) == 8
    assert all(structure.validation.valid for structure in first)
    # Same master seed reproduces the whole ensemble bit-for-bit.
    assert [s.partner_classes for s in first] == [s.partner_classes for s in second]
    assert [s.pair_matrix for s in first] == [s.pair_matrix for s in second]

    with pytest.raises(ValueError, match="at least 1"):
        sample_structures(model, sequence, num_samples=0)


def test_pairing_frequency_matrix_is_symmetric_probability_with_zero_diagonal():
    model = _denoiser()
    sequence = "GGGAAACCC"
    structures = sample_structures(model, sequence, num_samples=10, num_steps=8, seed=3)

    frequency = pairing_frequency_matrix(structures)
    size = len(sequence)

    assert len(frequency) == size
    for i in range(size):
        assert frequency[i][i] == 0.0
        for j in range(size):
            assert 0.0 <= frequency[i][j] <= 1.0
            assert frequency[i][j] == pytest.approx(frequency[j][i])


def test_ensemble_unpaired_probability_complements_pairing_mass():
    model = _denoiser()
    sequence = "GGGAAACCC"
    structures = sample_structures(model, sequence, num_samples=10, num_steps=8, seed=5)

    frequency = pairing_frequency_matrix(structures)
    unpaired = ensemble_unpaired_probability(structures)
    size = len(sequence)

    assert len(unpaired) == size
    for i in range(size):
        paired_mass = sum(frequency[i][j] for j in range(size) if j != i)
        assert unpaired[i] == pytest.approx(min(1.0, max(0.0, 1.0 - paired_mass)))
        assert 0.0 <= unpaired[i] <= 1.0


def test_frequency_matrix_rejects_empty_and_ragged_ensembles():
    with pytest.raises(ValueError, match="at least one"):
        pairing_frequency_matrix([])

    model = _denoiser()
    good = sample_structures(model, "GGGAAACCC", num_samples=2, num_steps=6, seed=0)
    other = sample_structures(model, "GGGAAACC", num_samples=1, num_steps=6, seed=0)
    with pytest.raises(ValueError, match="same length"):
        pairing_frequency_matrix(list(good) + list(other))
