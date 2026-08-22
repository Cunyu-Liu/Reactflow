from __future__ import annotations

import pytest

from scripts.reactflow_delta.analyze_model_rescue_v3 import (
    summarize_mutant_balanced,
    summarize_position_pooled,
)


def test_mutant_balanced_hierarchy_does_not_overweight_mutants_with_more_positions():
    mutant_values = {("P01", "M1"): [0.0, 1.0]}
    pooled_values = {("P01", "M1"): (3.0, 4)}

    nested = summarize_mutant_balanced(mutant_values)
    pooled = summarize_position_pooled(pooled_values)

    assert nested["mean"] == pytest.approx(0.5)
    assert pooled["mean"] == pytest.approx(0.75)


def test_both_hierarchies_remain_method_balanced():
    mutant_values = {
        ("P01", "M1"): [0.0, 0.0],
        ("P01", "M2"): [1.0],
    }
    pooled_values = {
        ("P01", "M1"): (0.0, 100),
        ("P01", "M2"): (1.0, 1),
    }

    assert summarize_mutant_balanced(mutant_values)["mean"] == pytest.approx(0.5)
    assert summarize_position_pooled(pooled_values)["mean"] == pytest.approx(0.5)
