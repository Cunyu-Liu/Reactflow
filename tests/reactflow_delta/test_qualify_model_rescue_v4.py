from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.reactflow_delta.qualify_model_rescue_v4 import (
    BASELINE,
    CAPACITY_NULL,
    FOUNDATION_ONLY,
    PRIMARY,
    PUBLISHED,
    SCRATCH,
    qualify_complete_scores,
    qualify_engineering_smoke,
    qualify_foundation_cache,
)


def _row(crps: float, mae: float, coverage68: float = 0.68, coverage95: float = 0.95):
    return {
        "crps": crps,
        "signed_delta_mae": mae,
        "coverage68": coverage68,
        "coverage95": coverage95,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "n_unexpected_prediction_keys": 0,
    }


def _passing_scores():
    return {
        BASELINE: [_row(0.20, 0.30) for _ in range(20)],
        PRIMARY: [_row(0.18, 0.27) for _ in range(20)],
        SCRATCH: [_row(0.195, 0.295) for _ in range(20)],
        FOUNDATION_ONLY: [_row(0.19, 0.285) for _ in range(20)],
        CAPACITY_NULL: [_row(0.19, 0.285) for _ in range(20)],
        PUBLISHED: [_row(0.195, 0.292) for _ in range(20)],
    }


def test_top_journal_screen_pass_requires_dual_metric_and_attribution_gates() -> None:
    result = qualify_complete_scores(_passing_scores(), phase="V4M3")
    assert result["overall_status"] == "V4M3_SCREEN_PASS"
    assert result["v4m4_authorized"] is True
    assert all(result["gates"].values())
    assert result["publication_ready"] is False
    assert result["external_replication"] == "NOT_ESTABLISHED"


def test_four_percent_mean_gain_fails_even_when_all_puzzles_are_positive() -> None:
    scores = _passing_scores()
    scores[PRIMARY] = [_row(0.192, 0.288) for _ in range(20)]
    result = qualify_complete_scores(scores, phase="V4M3")
    assert result["overall_status"] == "MODEL_RESCUE_V4_FAIL"
    assert result["gates"]["dual_metric_top_journal_development"] is False
    assert result["v4m4_authorized"] is False


def test_missing_task_matched_published_comparator_cannot_generate_pass() -> None:
    scores = _passing_scores()
    del scores[PUBLISHED]
    result = qualify_complete_scores(scores, phase="V4M3")
    assert result["overall_status"] == "MODEL_RESCUE_V4_FAIL"
    assert result["gates"]["task_matched_published_comparator"] is False
    assert result["versus_task_matched_published"] is None


def test_integrity_or_coverage_regression_blocks_pass() -> None:
    scores = copy.deepcopy(_passing_scores())
    scores[PRIMARY][0]["registered_prediction_coverage"] = 0.99
    scores[PRIMARY] = [dict(row, coverage68=0.64) for row in scores[PRIMARY]]
    result = qualify_complete_scores(scores, phase="V4M3")
    assert result["overall_status"] == "MODEL_RESCUE_V4_FAIL"
    assert result["gates"]["prediction_integrity"] is False
    assert result["gates"]["coverage_calibration"] is False


def _write_smoke_prediction(path, model_id):
    point = np.array([1.0, 2.0])
    np.savez_compressed(
        path,
        keys=np.array(["a", "b"], dtype=object),
        biological_scoring_key=np.array(["a", "b"], dtype=object),
        candidate_id=np.full(2, model_id, dtype=object),
        point_mean=point,
        locations=np.stack([point, point], axis=1),
        scales=np.full((2, 2), 0.2),
        weights=np.full((2, 2), 0.5),
        registered_status=np.full(2, "covered", dtype=object),
    )


def _write_smoke_fold(root, fold):
    models = {}
    for model_id in {BASELINE, PRIMARY, SCRATCH, FOUNDATION_ONLY, CAPACITY_NULL}:
        prediction = root / f"{model_id}_{fold}.npz"
        mean = root / f"{model_id}_{fold}_mean.pt"
        calibration = root / f"{model_id}_{fold}_cal.pt"
        _write_smoke_prediction(prediction, model_id)
        mean.write_text("checkpoint")
        calibration.write_text("checkpoint")
        count = 35_331_841 if model_id == PRIMARY else 36_001_281 if model_id == CAPACITY_NULL else 1
        models[model_id] = {
            "prediction_artifact": str(prediction),
            "mean_checkpoint": str(mean),
            "calibration_checkpoint": str(calibration),
            "mean_history_length": 3,
            "calibration_history_length": 3,
            "mean_history_finite": True,
            "calibration_history_finite": True,
            "trainable_mean_parameters": count,
        }
    row = {
        "outer_fold": fold,
        "seed": 0,
        "models": models,
        "held_score_computed": False,
        "external_outcome_accessed": False,
        "held_target_error_mask_invariance": True,
    }
    (root / f"v4_fold_result_fold{fold}_seed0.json").write_text(json.dumps(row))


def test_engineering_smoke_qualifies_without_reading_scientific_scores(tmp_path) -> None:
    _write_smoke_fold(tmp_path, 0)
    _write_smoke_fold(tmp_path, 1)
    result = qualify_engineering_smoke(tmp_path)
    assert result["overall_status"] == "V4M2_REAL_DATA_ENGINEERING_SMOKE_PASS"
    assert result["v4m3_authorized"] is True
    assert result["held_scores_computed"] is False
    assert result["scientific_interpretation_prohibited"] is True


def test_foundation_cache_qualifies_complete_frozen_outcome_blind_artifact(
    tmp_path,
) -> None:
    h5py = pytest.importorskip("h5py")
    cache = tmp_path / "rnafm.h5"
    with h5py.File(cache, "w") as handle:
        handle.create_dataset("row_ids", data=np.asarray([b"a", b"b"]))
        handle.create_dataset("lengths", data=np.asarray([3, 3], dtype=np.int32))
        handle.create_dataset(
            "embeddings", data=np.zeros((2, 3, 640), dtype=np.float16)
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "reactflow_delta.model_rescue_v4_rnafm_cache.v1",
                "evidence_status": "OUTCOME_BLIND_FROZEN_FOUNDATION_INPUT_ONLY",
                "official_repository": "https://github.com/ml4bio/RNA-FM",
                "official_repository_commit": "348951516e0963d22bbb33b3c9fc18c89081d38e",
                "official_checkpoint_source": "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth",
                "checkpoint_path_used": "/frozen/RNA-FM_pretrained.pth",
                "package_source_root": "/frozen/RNA-FM",
                "foundation_parameter_count": 99_000_000,
                "foundation_trainable_parameter_count": 0,
                "csv_columns_read": ["id", "puzzle", "method", "sequence"],
                "mutant_outcome_columns_loaded": False,
                "external_outcome_accessed": False,
                "exact_openknot_pretraining_overlap": "UNKNOWN_NOT_ASSERTED",
                "representation_layer": 12,
                "representation_width": 640,
                "n_sequences": 2,
                "max_sequence_length": 3,
            }
        ),
        encoding="utf-8",
    )

    result = qualify_foundation_cache(cache, manifest)

    assert result["overall_status"] == "V4M1_IMPLEMENTATION_AND_FOUNDATION_CACHE_PASS"
    assert result["v4m2_authorized"] is True
    assert result["held_scores_computed"] is False
