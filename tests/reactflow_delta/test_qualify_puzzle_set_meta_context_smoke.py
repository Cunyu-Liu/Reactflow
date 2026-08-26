from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    MERGED_SCHEMA,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    EXPECTED_RESIDUAL_PARAMETERS,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import PREDICTION_SCHEMA
from scripts.reactflow_delta.puzzle_set_meta_context_pretraining import (
    EXPECTED_DECODER_PARAMETERS,
    EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS,
)
from scripts.reactflow_delta.qualify_puzzle_set_meta_context_smoke import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    REQUIRED_INTEGRITY_FALSE,
    REQUIRED_INTEGRITY_TRUE,
    main,
    qualify,
)


def _write_prediction(directory: Path, fold: int) -> Path:
    keys = np.asarray([f"fold{fold}-row0", f"fold{fold}-row1"], dtype=object)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.full(2, fold, dtype=np.int64),
        "seed": np.zeros(2, dtype=np.int64),
        "registered_status": np.full(2, "covered", dtype=object),
        "feature41_point": np.zeros(2),
        "parent_point": np.zeros(2),
        "candidate_point": np.zeros(2),
        "null_point": np.zeros(2),
    }
    for arm in ("candidate", "null"):
        arrays[f"{arm}_weights"] = np.full((2, 2), 0.5)
        arrays[f"{arm}_locations"] = np.zeros((2, 2))
        arrays[f"{arm}_scales"] = np.tile([0.1, 0.2], (2, 1))
        arrays[f"{arm}_expected_absolute_delta"] = np.zeros(2)
    path = directory / f"prediction_fold{fold}.npz"
    np.savez_compressed(path, **arrays)
    return path


def _merged(tmp_path: Path) -> dict[str, object]:
    integrity = {name: True for name in REQUIRED_INTEGRITY_TRUE}
    integrity.update({name: False for name in REQUIRED_INTEGRITY_FALSE})
    rows = []
    for fold in (0, 1):
        rows.append(
            {
                "phase": "P1M2",
                "evidence_status": "ENGINEERING_SMOKE_ONLY",
                "outer_fold": fold,
                "seed": 0,
                "pretraining_epochs": 3,
                "point_epochs": 3,
                "calibration_epochs": 3,
                "candidate_parameter_count": EXPECTED_PARAMETER_COUNT,
                "null_parameter_count": EXPECTED_PARAMETER_COUNT,
                "candidate_trainable_parameter_count": (
                    EXPECTED_TRAINABLE_PARAMETER_COUNT
                ),
                "null_trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
                "residual_parameter_counts": {
                    "candidate": EXPECTED_RESIDUAL_PARAMETERS,
                    "null": EXPECTED_RESIDUAL_PARAMETERS,
                },
                "candidate_specific_trainable_parameter_counts": {
                    "candidate": (
                        EXPECTED_TRAINABLE_PARAMETER_COUNT
                        + EXPECTED_RESIDUAL_PARAMETERS
                    ),
                    "null": (
                        EXPECTED_TRAINABLE_PARAMETER_COUNT
                        + EXPECTED_RESIDUAL_PARAMETERS
                    ),
                },
                "pretraining_decoder_parameter_counts": {
                    "candidate": EXPECTED_DECODER_PARAMETERS,
                    "null": EXPECTED_DECODER_PARAMETERS,
                },
                "prediction_artifact": str(_write_prediction(tmp_path, fold)),
                "n_registered_prediction_rows": 2,
                "invariants": {
                    "prediction_target_free": True,
                    "held_score_computed": False,
                    "external_outcome_accessed": False,
                },
            }
        )
    return {
        "schema_version": MERGED_SCHEMA,
        "status": "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS",
        "phase": "P1M2",
        "expected_folds": [0, 1],
        "expected_seeds": [0],
        "expected_pretraining_epochs": 3,
        "expected_point_epochs": 3,
        "expected_calibration_epochs": 3,
        "expected_parameter_count_each": EXPECTED_PARAMETER_COUNT,
        "expected_trainable_parameter_count_each": (EXPECTED_TRAINABLE_PARAMETER_COUNT),
        "expected_residual_parameter_count_each": EXPECTED_RESIDUAL_PARAMETERS,
        "expected_candidate_specific_trainable_parameter_count_each": (
            EXPECTED_TRAINABLE_PARAMETER_COUNT + EXPECTED_RESIDUAL_PARAMETERS
        ),
        "expected_pretraining_decoder_parameter_count_each": (
            EXPECTED_DECODER_PARAMETERS
        ),
        "expected_pretraining_trainable_parameter_count_each": (
            EXPECTED_PRETRAINING_TRAINABLE_PARAMETERS
        ),
        "folds": rows,
        "context_retention_gate_required": False,
        "context_retention_gate_passed": False,
        "context_retention_summary": {
            "candidate_pretraining_established_all_runs": False,
            "candidate_retention_positive_all_runs": False,
            "selection_performed": False,
            "mutant_outcome_used": False,
            "held_puzzle_accessed": False,
        },
        "merge_integrity": integrity,
    }


def test_smoke_qualifier_accepts_exact_target_free_engineering_merge(
    tmp_path: Path,
) -> None:
    result = qualify(_merged(tmp_path))
    assert result["status"] == "P1M2_ENGINEERING_SMOKE_PASS"
    assert result["gate_passed"] is True
    assert result["p1m3_activation_eligible"] is True
    assert result["p1m3_authorized"] is False
    assert result["scientific_score_computed"] is False
    assert result["held_target_read"] is False
    assert result["scientific_score_fields_found"] == []
    assert all(result["gates"].values())


def test_smoke_retention_direction_is_report_only(tmp_path: Path) -> None:
    merged = _merged(tmp_path)
    assert merged["context_retention_gate_passed"] is False
    assert qualify(merged)["gate_passed"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "phase",
        "fold_universe",
        "schedule",
        "parameter_count",
        "integrity",
        "fold_evidence",
        "scientific_score",
    ),
)
def test_smoke_qualifier_fails_closed_on_protocol_changes(
    tmp_path: Path, mutation: str
) -> None:
    merged = _merged(tmp_path)
    if mutation == "phase":
        merged["phase"] = "P1M3"
    elif mutation == "fold_universe":
        merged["expected_folds"] = [0]
    elif mutation == "schedule":
        merged["expected_point_epochs"] = 40
    elif mutation == "parameter_count":
        merged["expected_parameter_count_each"] = EXPECTED_PARAMETER_COUNT + 1
    elif mutation == "integrity":
        merged["merge_integrity"]["prediction_only_schema"] = False
    elif mutation == "fold_evidence":
        merged["folds"][0]["evidence_status"] = "POST_HOC_DEVELOPMENT_PREDICTION_ONLY"
    elif mutation == "scientific_score":
        merged["folds"][0]["candidate_crps"] = 0.1
    result = qualify(merged)
    assert result["status"] == "P1M2_ENGINEERING_SMOKE_FAIL"
    assert result["gate_passed"] is False
    assert result["p1m3_activation_eligible"] is False
    assert result["p1m3_authorized"] is False


def test_smoke_qualifier_reopens_prediction_and_rejects_target_field(
    tmp_path: Path,
) -> None:
    merged = _merged(tmp_path)
    row = merged["folds"][0]
    path = Path(row["prediction_artifact"])
    with np.load(path, allow_pickle=True) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}
    arrays["target"] = np.zeros(2)
    np.savez_compressed(path, **arrays)
    result = qualify(merged)
    assert result["status"] == "P1M2_ENGINEERING_SMOKE_FAIL"
    assert result["gates"]["complete_prediction_only_integrity"] is False


def test_smoke_qualifier_refuses_to_overwrite(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.json"
    merged_path.write_text(json.dumps(_merged(tmp_path)), encoding="utf-8")
    out_path = tmp_path / "qualification.json"
    out_path.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        main(
            [
                "--merged-json",
                str(merged_path),
                "--out-json",
                str(out_path),
            ]
        )
    assert out_path.read_text(encoding="utf-8") == "preserve me"


def test_smoke_qualifier_source_has_no_scientific_scorer_dependency() -> None:
    source = Path(
        "scripts/reactflow_delta/qualify_puzzle_set_meta_context_smoke.py"
    ).read_text(encoding="utf-8")
    assert "score_puzzle_set_meta_context" not in source
