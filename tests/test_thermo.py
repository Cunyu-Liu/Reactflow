import math

import pytest

from reactflow.constraints import dotbracket_to_matrix, matrix_to_pairs
from reactflow.thermo import (
    crossing_count,
    energy_guided_scores,
    guided_projection,
    monte_carlo_unpaired_prior,
    nested_partition_table,
    pair_energy,
    sample_nested_structure,
    structure_energy,
)


def test_pair_energy_and_structure_energy_reward_canonical_pairs():
    matrix = dotbracket_to_matrix("(((...)))")

    assert pair_energy("G", "C") < pair_energy("G", "U") < 0
    assert math.isinf(pair_energy("A", "C"))
    assert structure_energy("GGGAAACCC", matrix) == pytest.approx(3 * (-3.0 + 0.2))


def test_structure_energy_rejects_invalid_structure():
    matrix = ((0, 1), (1, 0))

    with pytest.raises(ValueError, match="min_loop"):
        structure_energy("GC", matrix)


def test_crossing_count_detects_pseudoknot_crossings():
    assert crossing_count(((0, 5), (2, 7), (8, 9))) == 1


def test_energy_guided_scores_boost_favorable_pairs_and_mask_illegal_pairs():
    scores = [[0.0 for _ in range(9)] for _ in range(9)]

    guided = energy_guided_scores("GGGAAACCC", scores, eta=1.0)

    assert guided[0][8] > 0.0
    assert guided[3][4] == float("-inf")
    assert guided[0][8] == guided[8][0]


def test_nested_partition_sampling_produces_valid_nested_structures():
    sequence = "GGGAAACCC"
    partition = nested_partition_table(sequence)
    matrix = sample_nested_structure(sequence, partition, rng=__import__("random").Random(7))

    assert partition[0][len(sequence) - 1] > 1.0
    assert all(sum(row) <= 1 for row in matrix)


def test_monte_carlo_unpaired_prior_is_bounded_and_deterministic():
    first = monte_carlo_unpaired_prior("GGGAAACCC", samples=32, seed=123)
    second = monte_carlo_unpaired_prior("GGGAAACCC", samples=32, seed=123)

    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first)
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_unpaired_prior("GGG", samples=0)


def test_guided_projection_bugmap_all_negative_scores_should_not_force_pairs():
    sequence = "GGGAAACCC"
    scores = [[-10.0 for _ in sequence] for _ in sequence]

    matrix = guided_projection(sequence, scores, eta=0.0)

    assert matrix_to_pairs(matrix) == ()
