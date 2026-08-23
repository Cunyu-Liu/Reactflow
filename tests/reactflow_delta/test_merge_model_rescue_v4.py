from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reactflow_delta.merge_model_rescue_v4 import EXPECTED_MODELS, merge_fold_results


def _write_fold(root: Path, fold: int, seed: int = 0) -> None:
    models = {}
    for model in EXPECTED_MODELS:
        artifacts = {}
        for field, suffix in (
            ("prediction_artifact", "npz"),
            ("mean_checkpoint", "pt"),
            ("calibration_checkpoint", "pt"),
        ):
            path = root / f"{model}_fold{fold}_seed{seed}_{field}.{suffix}"
            path.write_text("fixture", encoding="utf-8")
            artifacts[field] = str(path)
        models[model] = artifacts
    row = {
        "schema_version": "reactflow_delta.model_rescue_v4_fold.v1",
        "outer_fold": fold,
        "seed": seed,
        "held_puzzle": f"P{fold + 1:02d}",
        "models": models,
        "held_score_computed": False,
        "external_outcome_accessed": False,
    }
    (root / f"v4_fold_result_fold{fold}_seed{seed}.json").write_text(
        json.dumps(row), encoding="utf-8"
    )


def test_merge_requires_exact_complete_unscored_seed0_universe(tmp_path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    merged = merge_fold_results(tmp_path, phase="V4M3")
    assert merged["merge_integrity"]["complete_fold_seed_universe"] is True
    assert merged["merge_integrity"]["per_fold_held_scores_absent"] is True
    assert len(merged["folds"]) == 20


def test_merge_rejects_missing_fold_before_any_qualification(tmp_path) -> None:
    for fold in range(19):
        _write_fold(tmp_path, fold)
    with pytest.raises(ValueError, match="incomplete v4 fold universe"):
        merge_fold_results(tmp_path, phase="V4M3")


def test_merge_rejects_per_fold_score_field_state(tmp_path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    path = tmp_path / "v4_fold_result_fold3_seed0.json"
    row = json.loads(path.read_text())
    row["held_score_computed"] = True
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="must not compute held scores"):
        merge_fold_results(tmp_path, phase="V4M3")
