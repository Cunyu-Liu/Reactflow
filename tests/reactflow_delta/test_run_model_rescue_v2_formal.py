from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.model_rescue_v2 import CALIBRATED_CANDIDATE
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE
from scripts.reactflow_delta.run_model_rescue_v2_formal import (
    assert_r2m4_authority,
    combine_five_seed_predictions,
)


def _prediction(seed: int, components: int) -> dict[str, np.ndarray]:
    keys = np.asarray(["k1", "k2"], dtype=object)
    center = np.asarray([0.1 + 0.01 * seed, 0.2])
    locations = np.repeat(center[:, None], components, axis=1)
    return {
        "keys": keys,
        "locations": locations,
        "scales": np.full((2, components), 0.3),
        "weights": np.full((2, components), 1.0 / components),
    }


def test_formal_combiner_builds_fixed_five_and_ten_component_mixtures():
    baseline = combine_five_seed_predictions(
        [_prediction(seed, 1) for seed in range(5)], BASELINE
    )
    candidate = combine_five_seed_predictions(
        [_prediction(seed, 2) for seed in range(5)], CALIBRATED_CANDIDATE
    )
    assert baseline["locations"].shape == (2, 5)
    assert candidate["locations"].shape == (2, 10)
    np.testing.assert_allclose(baseline["weights"], 0.2, atol=1e-12)
    np.testing.assert_allclose(candidate["weights"].sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        (candidate["weights"] * candidate["locations"]).sum(axis=1),
        candidate["point_mean"],
        atol=1e-12,
    )
    np.testing.assert_array_equal(candidate["seed_universe"], np.arange(5))


def test_formal_combiner_rejects_missing_seed_or_key_mismatch():
    with pytest.raises(ValueError, match="exactly five seeds"):
        combine_five_seed_predictions(
            [_prediction(seed, 2) for seed in range(4)], CALIBRATED_CANDIDATE
        )
    predictions = [_prediction(seed, 2) for seed in range(5)]
    predictions[-1]["keys"] = np.asarray(["k1", "different"], dtype=object)
    with pytest.raises(ValueError, match="key universes differ"):
        combine_five_seed_predictions(predictions, CALIBRATED_CANDIDATE)


def _write_active(root: Path, phase: str, r2m3: str, r2m4: str) -> None:
    path = root / "configs/reactflow_delta"
    path.mkdir(parents=True)
    (path / "active_contract.yaml").write_text(
        yaml.safe_dump(
            {
                "authority": {"current_phase": phase},
                "gate_state": {"R2M3": r2m3, "R2M4": r2m4},
                "training_allowed": True,
                "new_external_outcome_access_allowed": False,
            }
        ),
        encoding="utf-8",
    )


def test_formal_runner_is_closed_before_r2m4_authority(tmp_path):
    _write_active(tmp_path, "R2M3", "IN_PROGRESS", "NOT_RUN")
    with pytest.raises(RuntimeError, match="active phase is R2M4"):
        assert_r2m4_authority(tmp_path)


def test_formal_runner_opens_only_for_r2m3_pass_and_r2m4_active(tmp_path):
    _write_active(tmp_path, "R2M4", "PASS", "IN_PROGRESS")
    assert_r2m4_authority(tmp_path)
