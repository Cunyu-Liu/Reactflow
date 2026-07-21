import pytest

from reactflow.constraints import validate_pair_matrix
from reactflow.synthetic import (
    SyntheticSample,
    make_dataset,
    make_sample,
)


def test_make_sample_produces_legal_nested_structure():
    sample = make_sample(stem=4, loop=4, probe="2A3", seed=0)

    assert isinstance(sample, SyntheticSample)
    assert len(sample.sequence) == 12
    assert sample.dotbracket == "((((....))))"
    validation = validate_pair_matrix(sample.sequence, sample.pair_matrix, allow_pseudoknot=False)
    assert validation.valid
    # Closing pairs are canonical/wobble by construction.
    assert len(sample.reactivity) == len(sample.sequence)
    assert len(sample.weights) == len(sample.sequence)


def test_partner_classes_encode_pairs_and_unpaired():
    sample = make_sample(stem=4, loop=4, probe="2A3", seed=0)

    # Position 0 pairs with the last index -> class = last_index + 1.
    assert sample.partner_classes[0] == len(sample.sequence)
    # Loop positions are unpaired -> class 0.
    assert sample.partner_classes[4] == 0
    assert sample.partner_classes[5] == 0


def test_make_sample_is_deterministic():
    a = make_sample(stem=4, loop=4, probe="2A3", seed=5)
    b = make_sample(stem=4, loop=4, probe="2A3", seed=5)

    assert a == b


def test_make_sample_validates_arguments():
    with pytest.raises(ValueError, match="stem must be positive"):
        make_sample(stem=0, loop=4)
    with pytest.raises(ValueError, match="loop at least 3"):
        make_sample(stem=4, loop=2)


def test_make_dataset_produces_distinct_reproducible_samples():
    dataset = make_dataset(count=4, stem=4, loop=4, probe="2A3", seed=0)

    assert len(dataset) == 4
    sequences = {sample.sequence for sample in dataset}
    assert len(sequences) == 4  # distinct sequences

    again = make_dataset(count=4, stem=4, loop=4, probe="2A3", seed=0)
    assert dataset == again

    with pytest.raises(ValueError, match="count must be positive"):
        make_dataset(count=0)


def test_reactivity_is_perturbed_forward_operator_output():
    from reactflow.reactivity import ReactivityForwardOperator

    sample = make_sample(stem=4, loop=4, probe="2A3", seed=0, noise=0.0)
    clean = ReactivityForwardOperator().from_structure(sample.sequence, sample.pair_matrix, "2A3")

    # With zero noise the reactivity is exactly the forward operator output.
    for observed, expected in zip(sample.reactivity, clean):
        assert observed == pytest.approx(expected)
