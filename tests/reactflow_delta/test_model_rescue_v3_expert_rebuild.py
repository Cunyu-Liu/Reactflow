from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.run_model_rescue_v3_expert_rebuild import (
    PREDICTION_SCHEMA,
    _save_expert_prediction,
    assert_expert_rebuild_authority,
)
from scripts.reactflow_delta.qualify_model_rescue_v3_expert_rebuild import (
    check_fold_result,
)


def _write_authority(root: Path, **updates) -> None:
    config = root / "configs" / "reactflow_delta"
    config.mkdir(parents=True, exist_ok=True)
    active = {
        "authority": {"current_phase": "R3C3"},
        "runnable_phases": ["R3C3"],
        "training_allowed": "CORRECTED_B1_AND_MEANALIGNED_REBUILD_ONLY",
        "new_external_outcome_access_allowed": False,
        **updates,
    }
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(active), encoding="utf-8"
    )


def test_expert_rebuild_authority_is_exact_and_external_locked(tmp_path: Path) -> None:
    _write_authority(tmp_path)
    assert_expert_rebuild_authority(tmp_path)
    _write_authority(tmp_path, training_allowed=True)
    with pytest.raises(RuntimeError, match="authority is absent"):
        assert_expert_rebuild_authority(tmp_path)
    _write_authority(tmp_path, new_external_outcome_access_allowed=True)
    with pytest.raises(RuntimeError, match="external outcomes"):
        assert_expert_rebuild_authority(tmp_path)


def test_expert_prediction_is_prediction_only_and_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    prediction = {
        "keys": np.asarray(["k0", "k1"], dtype=object),
        "b1_delta_mean": np.asarray([0.1, 0.2]),
        "meanaligned_delta_mean": np.asarray([0.3, 0.4]),
    }
    path = tmp_path / "prediction.npz"
    _save_expert_prediction(path, prediction, fold=2, seed=0)
    with np.load(path, allow_pickle=True) as stored:
        assert str(stored["schema_version"]) == PREDICTION_SCHEMA
        assert set(stored.files) == {
            "schema_version",
            "keys",
            "b1_delta_mean",
            "meanaligned_delta_mean",
            "outer_fold",
            "seed",
        }
        assert not any("target" in name or "score" in name for name in stored.files)
    prediction["keys"] = np.asarray(["k0", "k0"], dtype=object)
    with pytest.raises(RuntimeError, match="duplicate keys"):
        _save_expert_prediction(path, prediction, fold=2, seed=0)


def test_expert_fold_qualification_requires_exact_key_universe(tmp_path: Path) -> None:
    b1 = tmp_path / "b1.pt"
    mean = tmp_path / "mean.pt"
    b1.touch()
    mean.touch()
    prediction = tmp_path / "prediction.npz"
    _save_expert_prediction(
        prediction,
        {
            "keys": np.asarray(["k0", "k1"], dtype=object),
            "b1_delta_mean": np.asarray([0.1, 0.2]),
            "meanaligned_delta_mean": np.asarray([0.3, 0.4]),
        },
        fold=2,
        seed=0,
    )
    row = {
        "schema_version": "reactflow_delta.model_rescue_v3_corrected_expert_rebuild.v1",
        "outer_fold": 2,
        "seed": 0,
        "epochs": 40,
        "held_score_computed": False,
        "external_outcome_accessed": False,
        "b1_checkpoint": str(b1),
        "meanaligned_checkpoint": str(mean),
        "expert_prediction_artifact": str(prediction),
        "b1_train_loss": [0.1] * 40,
        "meanaligned_train_loss": [0.2] * 40,
    }
    assert all(check_fold_result(row, {"k0", "k1"}).values())
    assert check_fold_result(row, {"k0", "k2"})[
        "prediction_key_universe_exact"
    ] is False
