from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.qualify_model_rescue_v3_expert_rebuild import (
    check_fold_result,
)
from scripts.reactflow_delta.run_model_rescue_v3_expert_rebuild import (
    PREDICTION_SCHEMA as V3_PREDICTION_SCHEMA,
    _save_expert_prediction,
)
from scripts.reactflow_delta.run_model_rescue_v8_expert_rebuild import (
    PREDICTION_SCHEMA,
    SCHEMA,
    assert_v8m1_authority,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_authority(root: Path) -> None:
    path = root / "configs/reactflow_delta/active_contract.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "authority": {"current_phase": "V8M1"},
                "runnable_phases": ["V8M1"],
                "training_allowed": (
                    "TARGET_IDENTITY_CORRECTED_B1_AND_MEANALIGNED_FRESH_REBUILD_ONLY"
                ),
                "held_score_read_allowed": False,
                "partial_fold_score_read_allowed": False,
                "new_external_outcome_access_allowed": False,
                "legacy_v3_expert_reuse_allowed": False,
            }
        ),
        encoding="utf-8",
    )


def test_v8m1_authority_is_narrow_and_fail_closed(tmp_path: Path) -> None:
    _write_authority(tmp_path)
    assert_v8m1_authority(tmp_path)
    active = yaml.safe_load(
        (tmp_path / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    active["held_score_read_allowed"] = True
    (tmp_path / "configs/reactflow_delta/active_contract.yaml").write_text(
        yaml.safe_dump(active)
    )
    try:
        assert_v8m1_authority(tmp_path)
    except RuntimeError as exc:
        assert "held scores closed" in str(exc)
    else:
        raise AssertionError("V8M1 authority accepted held score access")


def test_expert_prediction_helper_supports_separate_v8_namespace(
    tmp_path: Path,
) -> None:
    prediction = {
        "keys": np.asarray(["k0", "k1"], dtype=object),
        "b1_delta_mean": np.asarray([0.1, 0.2]),
        "meanaligned_delta_mean": np.asarray([0.3, 0.4]),
    }
    v8_path = tmp_path / "v8.npz"
    _save_expert_prediction(
        v8_path,
        prediction,
        fold=2,
        seed=0,
        prediction_schema=PREDICTION_SCHEMA,
    )
    with np.load(v8_path, allow_pickle=True) as stored:
        assert str(stored["schema_version"]) == PREDICTION_SCHEMA
        assert set(stored.files) == {
            "schema_version",
            "keys",
            "b1_delta_mean",
            "meanaligned_delta_mean",
            "outer_fold",
            "seed",
        }
    v3_path = tmp_path / "v3.npz"
    _save_expert_prediction(v3_path, prediction, fold=2, seed=0)
    with np.load(v3_path, allow_pickle=True) as stored:
        assert str(stored["schema_version"]) == V3_PREDICTION_SCHEMA


def test_v8_fold_qualification_rejects_schema_or_reuse_drift(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "prediction.npz"
    _save_expert_prediction(
        prediction_path,
        {
            "keys": np.asarray(["k0"], dtype=object),
            "b1_delta_mean": np.asarray([0.1]),
            "meanaligned_delta_mean": np.asarray([0.2]),
        },
        fold=0,
        seed=0,
        prediction_schema=PREDICTION_SCHEMA,
    )
    b1 = tmp_path / "b1.pt"
    mean = tmp_path / "mean.pt"
    torch.save({}, b1)
    torch.save({}, mean)
    row = {
        "schema_version": SCHEMA,
        "outer_fold": 0,
        "seed": 0,
        "epochs": 40,
        "held_score_computed": False,
        "external_outcome_accessed": False,
        "b1_checkpoint": str(b1),
        "meanaligned_checkpoint": str(mean),
        "expert_prediction_artifact": str(prediction_path),
        "b1_train_loss": [1.0] * 40,
        "meanaligned_train_loss": [1.0] * 40,
    }
    checks = check_fold_result(
        row,
        {"k0"},
        result_schema=SCHEMA,
        prediction_schema=PREDICTION_SCHEMA,
    )
    assert all(checks.values())
    row["schema_version"] = "legacy"
    checks = check_fold_result(
        row,
        {"k0"},
        result_schema=SCHEMA,
        prediction_schema=PREDICTION_SCHEMA,
    )
    assert checks["result_schema"] is False


def test_v8_uses_exact_construct_identity_without_suffix_fallback() -> None:
    source = inspect.getsource(M2Universe.mutant_full_profile)
    assert "endswith" not in source
    assert "cands" not in source
    assert "wt_id.replace" in source


def test_v8_controller_runs_fresh_twenty_fold_qualification_without_scores() -> None:
    controller = (
        ROOT
        / "scripts/reactflow_delta/run_model_rescue_v8_expert_controller.sh"
    )
    subprocess.run(["bash", "-n", str(controller)], check=True)
    text = controller.read_text(encoding="utf-8")
    assert "run_model_rescue_v8_expert_rebuild" in text
    assert "qualify_model_rescue_v8_expert_rebuild" in text
    assert "v8_corrected_expert_fold_result_fold" in text
    assert "v3_corrected_experts" not in text
    assert "score_model_rescue" not in text
    assert "0 4 8 12 16" in text
    assert "3 7 11 15 19" in text
