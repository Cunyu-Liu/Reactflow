from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.reactflow_delta.audit_puzzle_set_wt_alignment import (
    SCHEMA,
    audit_wt_alignment,
    region_alignment_summary,
)


@dataclass
class _Construct:
    puzzle: str
    method: str
    sequence: str
    wt_reactivity: np.ndarray
    wt_observed: np.ndarray
    design_start: int = 0
    design_end: int = 12


class _Universe:
    def __init__(self, constructs):
        self.constructs = {
            f"{value.puzzle}_{value.method}": value for value in constructs
        }


def _constructs(puzzle: str) -> list[_Construct]:
    base = np.asarray(
        [0.0, 1.2, -0.4, 0.8, 2.0, -1.1, 0.3, 1.6, -0.7, 0.5, 1.8, -0.2]
    )
    output = []
    for method in range(8):
        values = base + (method - 3.5) * 0.01
        output.append(
            _Construct(
                puzzle=puzzle,
                method=f"method{method}",
                sequence=("ACGU" * 3)[method % 4 :] + ("ACGU" * 3)[: method % 4],
                wt_reactivity=values,
                wt_observed=np.ones(12, dtype=bool),
            )
        )
    return output


def test_registered_position_signal_exceeds_wrong_position_controls() -> None:
    result = region_alignment_summary(
        _constructs("P01"), 0, 12, shifts=(1, 3, 5, 17)
    )
    assert result["effective_shift_controls"] == [1, 3, 5, 5]
    assert result["pair_alignment_increment"] > 0.0
    assert result["consensus_alignment_increment"] > 0.0
    assert result["n_pair_correlations"] == 28
    assert result["n_consensus_correlations"] == 8


def test_audit_is_wt_only_and_requires_the_exact_puzzle_universe() -> None:
    constructs = _constructs("P01") + _constructs("P02")
    result = audit_wt_alignment(
        _Universe(constructs), expected_puzzles=2, shifts=(1, 3, 5, 7)
    )
    assert result["schema_version"] == SCHEMA
    assert result["mutant_outcome_used"] is False
    assert result["external_outcome_accessed"] is False
    assert result["summary"]["design_pair_alignment_increment_positive_puzzles"] == 2
    assert (
        result["summary"]["design_consensus_alignment_increment_positive_puzzles"]
        == 2
    )
    try:
        audit_wt_alignment(
            _Universe(constructs), expected_puzzles=20, shifts=(1, 3, 5, 7)
        )
    except ValueError as error:
        assert "puzzle universe is incomplete" in str(error)
    else:
        raise AssertionError("WT alignment audit accepted an incomplete universe")
