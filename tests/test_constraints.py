import pytest

from reactflow.constraints import (
    allowed_pair_mask,
    dotbracket_to_matrix,
    edge_context_indicators,
    is_allowed_pair,
    matrix_to_pairs,
    project_greedy_matching,
    unpaired_indicators,
    validate_pair_matrix,
    would_cross,
)


def test_allowed_pairs_cover_canonical_and_wobble():
    assert is_allowed_pair("G", "C")
    assert is_allowed_pair("G", "U")
    assert not is_allowed_pair("G", "U", allow_wobble=False)
    assert not is_allowed_pair("A", "C")


def test_dotbracket_round_trip_to_pairs_and_validation():
    matrix = dotbracket_to_matrix("(((...)))")

    assert matrix_to_pairs(matrix) == ((0, 8), (1, 7), (2, 6))
    result = validate_pair_matrix("GGGAAACCC", matrix)
    assert result.valid
    assert result.pair_count == 3


@pytest.mark.parametrize("dotbracket", [")..", "((.", "..x"])
def test_dotbracket_rejects_unbalanced_or_unsupported_input(dotbracket):
    with pytest.raises(ValueError):
        dotbracket_to_matrix(dotbracket)


def test_validate_pair_matrix_reports_multiple_violation_types():
    matrix = (
        (1, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 1),
        (0, 0, 1, 0),
    )

    result = validate_pair_matrix("AAAA", matrix, min_loop=0)

    assert not result.valid
    text = ";".join(result.violations)
    assert "diagonal" in text
    assert "not symmetric" in text
    assert "not canonical" in text
    assert "position 2 has 2 partners" in text


def test_validate_pair_matrix_reports_length_and_pseudoknot_violations():
    matrix = (
        (0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0),
    )

    result = validate_pair_matrix("GAGAACAC", matrix, min_loop=1, allow_pseudoknot=False)
    length_result = validate_pair_matrix("GAGA", matrix, min_loop=1)

    assert not result.valid
    assert any("pseudoknot" in item for item in result.violations)
    assert any("sequence length" in item for item in length_result.violations)


def test_pair_matrix_helpers_reject_non_square_inputs():
    with pytest.raises(ValueError, match="square"):
        validate_pair_matrix("AC", ((0, 1),))
    with pytest.raises(ValueError, match="square"):
        unpaired_indicators(((0, 1),))


def test_allowed_pair_mask_applies_loop_and_chemistry_constraints():
    mask = allowed_pair_mask("GGGAAACCC", min_loop=3)

    assert mask[0][8]
    assert not mask[0][1]
    assert not mask[3][4]


def test_project_greedy_matching_selects_highest_non_conflicting_pairs():
    sequence = "GGGAAACCC"
    scores = [[0.0 for _ in sequence] for _ in sequence]
    scores[0][8] = scores[8][0] = 3.0
    scores[1][7] = scores[7][1] = 2.0
    scores[2][6] = scores[6][2] = 1.0
    scores[0][7] = scores[7][0] = 1.5

    matrix = project_greedy_matching(sequence, scores, min_score=0.1)

    assert matrix_to_pairs(matrix) == ((0, 8), (1, 7), (2, 6))
    assert validate_pair_matrix(sequence, matrix).valid


def test_project_greedy_matching_rejects_size_mismatch():
    with pytest.raises(ValueError, match="size differ"):
        project_greedy_matching("AC", ((0.0,),))


def test_project_greedy_matching_can_disallow_pseudoknots():
    sequence = "GAGAACAC"
    scores = [[0.0 for _ in sequence] for _ in sequence]
    scores[0][5] = scores[5][0] = 2.0
    scores[2][7] = scores[7][2] = 1.5

    matrix = project_greedy_matching(sequence, scores, min_loop=1, allow_pseudoknot=False, min_score=0.1)

    assert matrix_to_pairs(matrix) == ((0, 5),)
    assert would_cross(((0, 5),), 2, 7)


def test_unpaired_and_edge_context_indicators():
    matrix = dotbracket_to_matrix("((....))")

    unpaired = unpaired_indicators(matrix)
    edge = edge_context_indicators(matrix)

    assert unpaired == pytest.approx((0, 0, 1, 1, 1, 1, 0, 0))
    assert edge[1] == 1.0
    assert edge[0] == 1.0
