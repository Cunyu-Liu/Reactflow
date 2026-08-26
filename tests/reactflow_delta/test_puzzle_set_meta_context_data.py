from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from scripts.reactflow_delta.puzzle_set_meta_context import (
    make_exact_full_model_pair,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
    assemble_puzzle_training_batches,
    predict_held_puzzle_points,
)


@dataclass
class _Record:
    puzzle: str
    method: str
    construct_id: str
    design_pos: int
    full_pos: int
    ref: str = "A"
    alt: str = "G"


@dataclass
class _Construct:
    sequence: str
    wt_observed: np.ndarray


class _Universe:
    def __init__(self, constructs):
        self.constructs = constructs

    def get_construct(self, construct_id):
        return self.constructs[construct_id]


def _context(length: int, observed: bool = True):
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.ones(length)
    observed_mask = torch.full((length,), float(observed))
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed_mask, position, region


def _cell(construct_id: str, length: int = 4):
    edit = torch.tensor([1])
    return {
        "construct_id": construct_id,
        "edit": edit,
        "distance": (torch.arange(length)[None, :] - edit[:, None]).float(),
        "refs": ["A"],
        "alts": ["G"],
        "feature41_point": torch.zeros(1, length),
        "prediction_mask": torch.ones(1, length, dtype=torch.bool),
        "target": torch.zeros(1, length),
        "qualified_mask": torch.ones(1, length, dtype=torch.bool),
        "wt": torch.zeros(length),
    }


def test_training_assembler_builds_equal_eight_construct_puzzle_batches() -> None:
    records = []
    cells = []
    contexts = {}
    for puzzle in ("P01", "P02"):
        for method in range(8):
            construct_id = f"{puzzle}_method{method}"
            records.append(_Record(puzzle, f"method{method}", construct_id, 1, 1))
            cells.append(_cell(construct_id))
            contexts[construct_id] = _context(4)
    batches = assemble_puzzle_training_batches(records, cells, contexts)
    assert [batch["puzzle"] for batch in batches] == ["P01", "P02"]
    assert all(len(batch["contexts"]) == 8 for batch in batches)
    assert all(
        [cell["focal_construct_index"] for cell in batch["cells"]]
        == list(range(8))
        for batch in batches
    )


def test_training_assembler_rejects_incomplete_puzzle_set() -> None:
    records = [
        _Record("P01", f"method{index}", f"P01_method{index}", 1, 1)
        for index in range(7)
    ]
    cells = [_cell(record.construct_id) for record in records]
    contexts = {record.construct_id: _context(4) for record in records}
    try:
        assemble_puzzle_training_batches(records, cells, contexts)
    except ValueError as error:
        assert "instead of eight" in str(error)
    else:
        raise AssertionError("assembler accepted an incomplete puzzle set")


def test_held_prediction_is_complete_target_free_and_feature41_replaying() -> None:
    records = []
    constructs = {}
    contexts = {}
    feature41 = {}
    for method in range(8):
        construct_id = f"P20_method{method}"
        records.append(_Record("P20", f"method{method}", construct_id, 1, 1))
        observed = method != 0
        constructs[construct_id] = _Construct(
            sequence="ACGU", wt_observed=np.full(4, observed, dtype=bool)
        )
        contexts[construct_id] = _context(4, observed=observed)
        feature41[construct_id] = np.full((1, 4), method / 10.0, dtype=np.float32)
    candidate, null = make_exact_full_model_pair(seed=91)
    prediction = predict_held_puzzle_points(
        univ=_Universe(constructs),
        held_records=records,
        context_cache=contexts,
        feature41_by_construct=feature41,
        candidate=candidate,
        null=null,
        outer_fold=19,
        seed=0,
    )
    assert str(prediction["schema_version"].item()) == PREDICTION_SCHEMA
    assert len(prediction["keys"]) == 32
    assert len(set(map(str, prediction["keys"]))) == 32
    assert set(prediction["registered_status"]) == {"covered"}
    assert not (set(prediction) & FORBIDDEN_PREDICTION_FIELDS)
    # Zero-initialized candidate/null replay feature41 on observed positions.
    # P20_method0 is the registered zero-observed construct and is output as
    # zero by the prediction mask rather than by fabricating WT observations.
    for method in range(8):
        start = method * 4
        stop = start + 4
        expected = np.zeros(4) if method == 0 else np.full(4, method / 10.0)
        assert np.allclose(
            prediction["candidate_point"][start:stop], expected, atol=1e-7, rtol=0.0
        )
        assert np.allclose(
            prediction["null_point"][start:stop], expected, atol=1e-7, rtol=0.0
        )


def test_held_prediction_rejects_nonexact_context_universe() -> None:
    records = [
        _Record("P20", f"method{index}", f"P20_method{index}", 1, 1)
        for index in range(8)
    ]
    constructs = {
        record.construct_id: _Construct("ACGU", np.ones(4, dtype=bool))
        for record in records
    }
    contexts = {record.construct_id: _context(4) for record in records[:-1]}
    feature41 = {
        record.construct_id: np.zeros((1, 4), dtype=np.float32)
        for record in records
    }
    candidate, null = make_exact_full_model_pair(seed=101)
    try:
        predict_held_puzzle_points(
            univ=_Universe(constructs),
            held_records=records,
            context_cache=contexts,
            feature41_by_construct=feature41,
            candidate=candidate,
            null=null,
            outer_fold=19,
            seed=0,
        )
    except ValueError as error:
        assert "universe is not exact" in str(error)
    else:
        raise AssertionError("prediction accepted a missing WT context")
