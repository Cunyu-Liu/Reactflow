from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.reactflow_delta.model_rescue_v4 import (
    PRIMARY_CANDIDATE,
    MutationConditionedDualTower,
    V4ModelConfig,
)
from scripts.reactflow_delta import run_model_rescue_v4 as R


def _tiny_model() -> MutationConditionedDualTower:
    return MutationConditionedDualTower(
        V4ModelConfig(
            d_model=32,
            heads=4,
            wt_blocks=1,
            response_blocks=1,
            ff_dim=64,
            pair_dim=16,
            pair_heads=4,
            pair_blocks=1,
            foundation_dim=12,
            dropout=0.0,
            max_relative_distance=8,
        )
    )


def _training_cell():
    length, batch, width = 6, 2, 12
    base = torch.tensor([0, 1, 2, 3, 0, 1])
    sequence = torch.nn.functional.one_hot(base, 4).float()
    wt_foundation = torch.randn(length, width)
    mutant_foundation = wt_foundation[None].repeat(batch, 1, 1)
    mutant_foundation[0, 1] += 0.2
    mutant_foundation[1, 4] -= 0.2
    return {
        "records": [SimpleNamespace(), SimpleNamespace()],
        "sequence_one_hot": sequence,
        "wt_reactivity": torch.randn(length),
        "wt_error": torch.rand(length),
        "wt_observed": torch.ones(length, dtype=torch.bool),
        "position": torch.linspace(-1, 1, length),
        "region_one_hot": torch.nn.functional.one_hot(
            torch.tensor([0, 0, 1, 1, 0, 0]), 2
        ).float(),
        "edit_idx": torch.tensor([1, 4]),
        "refs": ["C", "A"],
        "alts": ["A", "G"],
        "wt_foundation": wt_foundation,
        "mutant_foundation": mutant_foundation,
        "target": torch.randn(batch, length),
        "qualified_mask": torch.ones(batch, length, dtype=torch.bool),
        "wt": torch.randn(length),
        "b1_ctx": (),
    }


def test_mean_training_runs_one_exact_cell_epoch_with_finite_history() -> None:
    model = _tiny_model()
    history = R.fit_mean(
        PRIMARY_CANDIDATE,
        model,
        [_training_cell()],
        device="cpu",
        epochs=1,
        seed=0,
    )
    assert len(history) == 1
    assert np.isfinite(history).all()


def test_held_prediction_cell_builder_never_loads_target_matrix(monkeypatch) -> None:
    records = [SimpleNamespace(construct_id="P01_m1")]

    def forbidden(*args, **kwargs):
        raise AssertionError("held target loader was called")

    monkeypatch.setattr(R, "_target_matrix", forbidden)
    monkeypatch.setattr(
        R,
        "_input_cell",
        lambda univ, construct_id, cell_records, foundation: {
            "construct_id": construct_id,
            "records": cell_records,
        },
    )
    cells = R.make_prediction_cells(object(), records, object())
    assert cells == [{"construct_id": "P01_m1", "records": records}]


def test_prediction_schema_is_target_free_and_zero_mean_centered(tmp_path) -> None:
    point = [1.2, 2.3]
    arrays = R._prediction_arrays(
        model_id=PRIMARY_CANDIDATE,
        fold=0,
        seed=0,
        keys=["a", "b"],
        delta_mean=[0.2, 0.3],
        point_mean=point,
        locations=[np.array([1.2, 1.2]), np.array([2.3, 2.3])],
        scales=[np.array([0.1, 0.2]), np.array([0.2, 0.3])],
        weights=[np.array([0.4, 0.6]), np.array([0.5, 0.5])],
        mean_checkpoint=tmp_path / "mean.pt",
        calibration_checkpoint=tmp_path / "cal.pt",
    )
    assert {"target", "target_error", "qualified_target_mask", "score"}.isdisjoint(arrays)
    assert np.array_equal(arrays["locations"][:, 0], arrays["point_mean"])
    assert np.array_equal(arrays["locations"][:, 1], arrays["point_mean"])


def test_v4m1_active_contract_cannot_run_real_training() -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(RuntimeError, match="closed outside active V4M2"):
        R.assert_run_authority(root, "V4M2")
