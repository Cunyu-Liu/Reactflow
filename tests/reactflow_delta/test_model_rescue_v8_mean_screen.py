from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.merge_model_rescue_v8_mean_screen import (
    QUALIFICATION_SCHEMA,
    merge_folds,
)
from scripts.reactflow_delta.qualify_model_rescue_v8_mean_screen import qualify
from scripts.reactflow_delta.run_model_rescue_v8_expert_rebuild import (
    PREDICTION_SCHEMA,
    SCHEMA as FOLD_SCHEMA,
)
from scripts.reactflow_delta.score_model_rescue_v8_mean_screen import (
    SCHEMA as SCORE_SCHEMA,
    assert_score_authority,
)


def _write_v8_fold(directory: Path, fold: int) -> None:
    keys = np.asarray([f"key-{fold}-0", f"key-{fold}-1"], dtype=object)
    prediction = directory / f"prediction-{fold}.npz"
    np.savez_compressed(
        prediction,
        schema_version=np.asarray(PREDICTION_SCHEMA),
        keys=keys,
        b1_delta_mean=np.asarray([0.1, -0.2]),
        meanaligned_delta_mean=np.asarray([0.09, -0.18]),
        outer_fold=np.full(2, fold, dtype=np.int64),
        seed=np.zeros(2, dtype=np.int64),
    )
    b1 = directory / f"b1-{fold}.pt"
    mean = directory / f"mean-{fold}.pt"
    torch.save({}, b1)
    torch.save({}, mean)
    row = {
        "schema_version": FOLD_SCHEMA,
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "seed": 0,
        "epochs": 40,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "canonical_mutant_full_profiles": 13976,
        "held_score_computed": False,
        "external_outcome_accessed": False,
        "legacy_v3_checkpoint_reused": False,
        "legacy_v3_prediction_reused": False,
        "b1_checkpoint": str(b1),
        "meanaligned_checkpoint": str(mean),
        "expert_prediction_artifact": str(prediction),
    }
    (directory / f"v8_corrected_expert_fold_result_fold{fold}_seed0.json").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


def _write_qualification(path: Path, status: str = "V8M1_CORRECTED_EXPERT_REBUILD_PASS") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": QUALIFICATION_SCHEMA,
                "status": status,
                "target_profile_identity_exact": True,
                "scores_read": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_v8m2_merge_requires_complete_fresh_prediction_only_universe(
    tmp_path: Path,
) -> None:
    qualification = tmp_path / "qualification.json"
    _write_qualification(qualification)
    for fold in range(20):
        _write_v8_fold(tmp_path, fold)
    merged = merge_folds(tmp_path, qualification)
    assert merged["status"] == "V8M2_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged["merge_integrity"]["target_identity_exact"] is True
    assert merged["merge_integrity"]["legacy_v3_reuse"] is False
    assert merged["merge_integrity"]["held_scores_absent"] is True


def test_v8m2_merge_rejects_missing_fold_or_nonpass_qualification(
    tmp_path: Path,
) -> None:
    qualification = tmp_path / "qualification.json"
    _write_qualification(qualification)
    for fold in range(19):
        _write_v8_fold(tmp_path, fold)
    with pytest.raises(ValueError, match="incomplete"):
        merge_folds(tmp_path, qualification)
    _write_v8_fold(tmp_path, 19)
    _write_qualification(qualification, "V8M1_CORRECTED_EXPERT_REBUILD_FAIL")
    with pytest.raises(ValueError, match="exact V8M1 PASS"):
        merge_folds(tmp_path, qualification)


def test_v8m2_score_authority_is_complete_score_only(tmp_path: Path) -> None:
    active = {
        "authority": {"current_phase": "V8M2"},
        "runnable_phases": ["V8M2"],
        "held_score_read_allowed": True,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "training_allowed": False,
    }
    path = tmp_path / "configs/reactflow_delta/active_contract.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(active), encoding="utf-8")
    assert_score_authority(tmp_path)
    active["partial_fold_score_read_allowed"] = True
    path.write_text(yaml.safe_dump(active), encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial"):
        assert_score_authority(tmp_path)


def _score_fixture(
    *, feature41: float, b1: float, meanaligned: float, absolute_meanaligned: float
) -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
                "feature41_signed_delta_mae": feature41,
                "b1_signed_delta_mae": b1,
                "meanaligned_signed_delta_mae": meanaligned,
                "feature41_absolute_delta_mae": 0.15,
                "b1_absolute_delta_mae": 0.151,
                "meanaligned_absolute_delta_mae": absolute_meanaligned,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "V8M2_COMPLETE_MEAN_SCREEN_SCORE_PASS",
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "scores": rows,
    }


def test_v8m2_qualifier_requires_both_comparators_and_absolute_guardrail() -> None:
    passed = qualify(
        _score_fixture(
            feature41=0.20,
            b1=0.205,
            meanaligned=0.195,
            absolute_meanaligned=0.1505,
        )
    )
    assert passed["status"] == "V8M2_MEAN_SIGNAL_ELIGIBLE"
    assert passed["gate_passed"] is True
    weak_vs_feature41 = qualify(
        _score_fixture(
            feature41=0.20,
            b1=0.205,
            meanaligned=0.199,
            absolute_meanaligned=0.1505,
        )
    )
    assert weak_vs_feature41["status"] == "V8M2_MEAN_SIGNAL_NOT_ELIGIBLE"
    assert weak_vs_feature41["gates"][
        "signed_relative_gain_vs_feature41_ge_1pct"
    ] is False
    bad_absolute = qualify(
        _score_fixture(
            feature41=0.20,
            b1=0.205,
            meanaligned=0.195,
            absolute_meanaligned=0.151,
        )
    )
    assert bad_absolute["gates"][
        "absolute_relative_gain_vs_feature41_ge_minus_0_5pct"
    ] is False


def test_v8m2_controller_merges_before_single_complete_score() -> None:
    root = Path(__file__).resolve().parents[2]
    controller = root / "scripts/reactflow_delta/run_model_rescue_v8_mean_screen.sh"
    subprocess.run(["bash", "-n", str(controller)], check=True)
    text = controller.read_text(encoding="utf-8")
    merge_at = text.index("merge_model_rescue_v8_mean_screen")
    score_at = text.index("score_model_rescue_v8_mean_screen")
    qualify_at = text.index("qualify_model_rescue_v8_mean_screen")
    assert merge_at < score_at < qualify_at
    assert "tic2a_corrected_merged_unscored.json" in text
    assert "v8m2_complete_mean_screen_scores.json" in text
    assert "fold_result" not in text
