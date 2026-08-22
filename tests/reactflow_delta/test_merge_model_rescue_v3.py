from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reactflow_delta.merge_model_rescue_v3 import merge_screen_folds
from scripts.reactflow_delta.model_rescue_v3 import CANDIDATE
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE
from scripts.reactflow_delta.run_model_rescue_v3 import SCHEMA


def _write_fold(root: Path, fold: int) -> None:
    artifacts = []
    for name in ("base_pred", "base_ckpt", "pred", "b1", "mean", "cal", "inner"):
        path = root / f"{name}_{fold}"
        path.write_text("x", encoding="utf-8")
        artifacts.append(path)
    row = {
        "schema_version": SCHEMA,
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "seed": 0,
        "baseline": {
            "model_id": BASELINE,
            "prediction_artifact": str(artifacts[0]),
            "checkpoint": str(artifacts[1]),
        },
        "candidate": {
            "candidate_id": CANDIDATE,
            "prediction_artifact": str(artifacts[2]),
            "b1_checkpoint": str(artifacts[3]),
            "meanaligned_checkpoint": str(artifacts[4]),
            "calibration_checkpoint": str(artifacts[5]),
            "inner_crossfit_ledger": str(artifacts[6]),
        },
        "invariants": {
            "held_target_error_mask_invariance": True,
            "inner_crossfit_complete": True,
            "method_used_as_gate_input": False,
            "residual_changed_point_mean": False,
        },
    }
    (root / f"v3_fold_result_fold{fold}_seed0.json").write_text(
        json.dumps(row), encoding="utf-8"
    )


def test_merge_requires_complete_unique_v3_fold_universe(tmp_path: Path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    result = merge_screen_folds(tmp_path)
    assert result["merge_integrity"]["fold_ids"] == list(range(20))
    (tmp_path / "v3_fold_result_fold19_seed0.json").unlink()
    with pytest.raises(ValueError, match="exactly folds"):
        merge_screen_folds(tmp_path)
