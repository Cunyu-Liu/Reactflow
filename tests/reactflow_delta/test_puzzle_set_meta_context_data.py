from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v14 import V14PointModel
from scripts.reactflow_delta.model_rescue_v10 import (
    TrainOnlyStandardizer,
    calibration_input,
)


def _full_pair(seed: int):
    torch.manual_seed(1400)
    source = V14PointModel()
    return make_exact_full_model_pair(
        seed=seed, v14_point_state=source.state_dict()
    )
from scripts.reactflow_delta.puzzle_set_meta_context import (
    make_exact_full_model_pair,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    make_exact_residual_pair,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import (
    FORBIDDEN_PREDICTION_FIELDS,
    POINT_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
    assemble_puzzle_training_batches,
    predict_held_puzzle_distributions,
    predict_held_puzzle_points,
    validate_puzzle_coordinate_frames,
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
    design_start: int = 1
    design_end: int = 3


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
        "parent_point": torch.zeros(1, length),
        "prediction_mask": torch.ones(1, length, dtype=torch.bool),
        "target": torch.zeros(1, length),
        "qualified_mask": torch.ones(1, length, dtype=torch.bool),
        "wt": torch.zeros(length),
        "feature41_basis": np.zeros((1, length, 41), dtype=np.float32),
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
        assert "constructs instead of eight" in str(error)
    else:
        raise AssertionError("assembler accepted an incomplete puzzle set")


def test_coordinate_audit_accepts_shared_frame_and_rejects_shifted_design() -> None:
    records = [
        _Record("P01", f"method{index}", f"P01_method{index}", 1, 1)
        for index in range(8)
    ]
    constructs = {
        record.construct_id: _Construct("ACGU", np.ones(4, dtype=bool))
        for record in records
    }
    frames = validate_puzzle_coordinate_frames(records, _Universe(constructs))
    assert frames == {"P01": (4, 1, 3)}
    constructs[records[-1].construct_id].design_start = 0
    try:
        validate_puzzle_coordinate_frames(records, _Universe(constructs))
    except ValueError as error:
        assert "share length/design coordinates" in str(error)
    else:
        raise AssertionError("coordinate audit accepted a shifted construct")


def test_zero_outcome_construct_remains_context_without_fake_supervision() -> None:
    records = [
        _Record("P20", f"method{index}", f"P20_method{index}", 1, 1)
        for index in range(8)
    ]
    cells = [_cell(record.construct_id) for record in records[1:]]
    contexts = {
        record.construct_id: _context(4, observed=index != 0)
        for index, record in enumerate(records)
    }
    batch = assemble_puzzle_training_batches(records, cells, contexts)[0]
    assert len(batch["contexts"]) == 8
    assert len(batch["cells"]) == 7
    assert {cell["construct_id"] for cell in batch["cells"]} == {
        record.construct_id for record in records[1:]
    }


def test_held_prediction_is_complete_target_free_and_feature41_replaying() -> None:
    records = []
    constructs = {}
    contexts = {}
    feature41 = {}
    parent = {}
    for method in range(8):
        construct_id = f"P20_method{method}"
        records.append(_Record("P20", f"method{method}", construct_id, 1, 1))
        observed = method != 0
        constructs[construct_id] = _Construct(
            sequence="ACGU", wt_observed=np.full(4, observed, dtype=bool)
        )
        contexts[construct_id] = _context(4, observed=observed)
        feature41[construct_id] = np.full((1, 4), method / 10.0, dtype=np.float32)
        parent[construct_id] = np.full(
            (1, 4), method / 10.0 + 0.05, dtype=np.float32
        )
    candidate, null = _full_pair(91)
    prediction = predict_held_puzzle_points(
        univ=_Universe(constructs),
        held_records=records,
        context_cache=contexts,
        feature41_by_construct=feature41,
        parent_point_by_construct=parent,
        candidate=candidate,
        null=null,
        outer_fold=19,
        seed=0,
    )
    assert str(prediction["schema_version"].item()) == POINT_PREDICTION_SCHEMA
    assert len(prediction["keys"]) == 32
    assert len(set(map(str, prediction["keys"]))) == 32
    assert set(prediction["registered_status"]) == {"covered"}
    assert not (set(prediction) & FORBIDDEN_PREDICTION_FIELDS)
    # Zero-initialized candidate/null replay the frozen parent on observed positions.
    # P20_method0 is the registered zero-observed construct and is output as
    # zero by the prediction mask rather than by fabricating WT observations.
    for method in range(8):
        start = method * 4
        stop = start + 4
        expected = (
            np.zeros(4)
            if method == 0
            else np.full(4, method / 10.0 + 0.05)
        )
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
    parent = {
        record.construct_id: np.zeros((1, 4), dtype=np.float32)
        for record in records
    }
    candidate, null = _full_pair(101)
    try:
        predict_held_puzzle_points(
            univ=_Universe(constructs),
            held_records=records,
            context_cache=contexts,
            feature41_by_construct=feature41,
            parent_point_by_construct=parent,
            candidate=candidate,
            null=null,
            outer_fold=19,
            seed=0,
        )
    except ValueError as error:
        assert "universe is not exact" in str(error)
    else:
        raise AssertionError("prediction accepted a missing WT context")


def test_held_distribution_is_target_free_and_preserves_each_point_median() -> None:
    records = []
    constructs = {}
    contexts = {}
    feature41_point = {}
    parent_point = {}
    feature41_basis = {}
    direct_features = {}
    for method in range(8):
        construct_id = f"P01_method{method}"
        records.append(_Record("P01", f"method{method}", construct_id, 1, 1))
        constructs[construct_id] = _Construct(
            sequence="ACGU", wt_observed=np.ones(4, dtype=bool)
        )
        contexts[construct_id] = _context(4)
        feature41_point[construct_id] = np.zeros((1, 4), dtype=np.float32)
        parent_point[construct_id] = np.zeros((1, 4), dtype=np.float32)
        feature41_basis[construct_id] = np.zeros((1, 4, 41), dtype=np.float32)
        direct_features[construct_id] = np.zeros((1, 4, 201), dtype=np.float32)
    candidate, null = _full_pair(111)
    candidate_head, null_head = make_exact_residual_pair(seed=0, device="cpu")
    raw = calibration_input(
        np.zeros((1, 41)), np.zeros(1), np.zeros((1, 201))
    )
    standardizer = TrainOnlyStandardizer.fit([raw])
    prediction = predict_held_puzzle_distributions(
        univ=_Universe(constructs),
        held_records=records,
        context_cache=contexts,
        feature41_by_construct=feature41_point,
        parent_point_by_construct=parent_point,
        feature41_basis_by_construct=feature41_basis,
        direct_features_by_construct=direct_features,
        candidate=candidate,
        null=null,
        residual_heads={"candidate": candidate_head, "null": null_head},
        standardizers={"candidate": standardizer, "null": standardizer},
        outer_fold=0,
        seed=0,
    )
    assert str(prediction["schema_version"].item()) == PREDICTION_SCHEMA
    assert not (set(prediction) & FORBIDDEN_PREDICTION_FIELDS)
    for name in ("candidate", "null"):
        weights = torch.tensor(prediction[f"{name}_weights"])
        locations = torch.tensor(prediction[f"{name}_locations"])
        scales = torch.tensor(prediction[f"{name}_scales"])
        point = torch.tensor(prediction[f"{name}_point"])
        cdf = torch.sum(
            weights * torch.special.ndtr((point[:, None] - locations) / scales),
            dim=-1,
        )
        assert torch.allclose(
            cdf, torch.full_like(cdf, 0.5), atol=3e-6, rtol=0.0
        )
