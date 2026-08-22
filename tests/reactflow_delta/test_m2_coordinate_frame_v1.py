from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.run_p2_v3 import _bio_key


def _write_fixture(path: Path, *, sub_start: int = 5) -> Path:
    sequence = "AAAACCCCGGGG"
    mutant = sequence[:5] + "G" + sequence[6:]
    common = {
        "experiment_type": "2A3_MaP",
        "dataset_name": "coordinate-fixture",
        "puzzle": "P01",
        "method": "Eterna",
        "sub_start": sub_start,
        "sub_end": 8,
        "design_length": 4,
        "design_sequence": sequence[4:8],
        "target_structure": "",
        "M2_structure": "",
    }
    rows = [
        {
            **common,
            "id": "P01_Eterna_wt",
            "sequence": sequence,
            "mutA": 0,
        },
        {
            **common,
            "id": "P01_Eterna_mm_1_C_G",
            "sequence": mutant,
            "mutA": 2,
        },
    ]
    for row_idx, row in enumerate(rows):
        for position in range(1, len(sequence) + 1):
            row[f"reactivity_{position:04d}"] = row_idx + position / 100.0
            row[f"reactivity_error_{position:04d}"] = 0.1
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_design_and_full_coordinates_are_explicit_and_correct(tmp_path: Path) -> None:
    universe = M2Universe(_write_fixture(tmp_path / "m2.csv"))
    ledger = universe.build()
    record = universe.get_records()[0]
    construct = universe.get_construct(record.construct_id)

    assert not hasattr(record, "pos")
    assert record.design_pos == 1
    assert record.full_pos == 5
    assert construct.design_start == 4
    assert construct.design_end == 8
    assert construct.sequence[record.full_pos] == record.ref == "C"
    assert record.region == "design_region"
    assert record.wt_reactivity == pytest.approx(0.06)
    assert record.target_reactivity == pytest.approx(1.06)
    assert ledger["coordinate_frame"]["formula_matches_raw_diff"] == 1
    assert ledger["coordinate_frame"]["mutA_equals_design_pos_plus_one"] == 1

    target, error = universe.mutant_full_profile(
        record.wt_id, record.design_pos, record.ref, record.alt
    )
    assert target is not None and error is not None
    assert target[record.full_pos] == pytest.approx(1.06)
    assert _bio_key(universe, record, record.full_pos) == (
        "openknot_m2|P01|Eterna|P01_Eterna|1|C>G|5"
    )


def test_coordinate_validation_rejects_wrong_full_sequence_offset(
    tmp_path: Path,
) -> None:
    universe = M2Universe(
        _write_fixture(tmp_path / "wrong-offset.csv", sub_start=4)
    )
    with pytest.raises(ValueError, match="coordinate validation failed"):
        universe.build()
