from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    MEAN_CANDIDATE,
)
from scripts.reactflow_delta.qualify_model_rescue_v2 import qualify_screen, qualify_smoke


def _score(crps: float, delta: float) -> dict:
    return {
        "crps": crps,
        "signed_delta_mae": delta,
        "coverage68": 0.68,
        "coverage95": 0.95,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "n_unexpected_prediction_keys": 0,
    }


def _fold(index: int, *, mean_delta: float = 0.196, calibrated_crps: float = 0.19) -> dict:
    mean_score = _score(0.205, mean_delta)
    calibrated_score = _score(calibrated_crps, mean_delta)
    return {
        "outer_fold": index,
        "held_puzzle": f"P{index + 1:02d}",
        "seed": 0,
        "baseline": {
            "model_id": "b1_rfd_direct_aligned",
            "score": _score(0.20, 0.20),
        },
        "point_mean_max_abs_difference": 0.0,
        "candidates": {
            MEAN_CANDIDATE: {"score": mean_score},
            CALIBRATED_CANDIDATE: {"score": calibrated_score},
        },
    }


def test_screen_pass_requires_both_pre_frozen_gates():
    result = qualify_screen([_fold(index) for index in range(20)])
    assert result["mean_gate"]["status"] == "MEAN_GATE_PASS"
    assert result["calibration_gate"]["status"] == "CALIBRATION_GATE_PASS"
    assert result["overall_status"] == "R2M3_SCREEN_PASS"
    assert result["r2m4_authorized"] is True


def test_mean_gate_fails_below_one_percent_even_if_direction_is_positive():
    result = qualify_screen(
        [_fold(index, mean_delta=0.199, calibrated_crps=0.19) for index in range(20)]
    )
    assert result["mean_gate"]["checks"]["mean_signed_delta_mae_gain_positive"]
    assert not result["mean_gate"]["checks"][
        "signed_delta_mae_relative_gain_at_least_1pct"
    ]
    assert result["overall_status"] == "MODEL_RESCUE_V2_FAIL"


def test_calibration_gate_fails_if_point_mean_differs():
    folds = [_fold(index) for index in range(20)]
    folds[3]["point_mean_max_abs_difference"] = 2e-7
    result = qualify_screen(folds)
    assert not result["calibration_gate"]["checks"]["point_mean_identical_atol_1e_7"]
    assert result["overall_status"] == "MODEL_RESCUE_V2_FAIL"


def test_qualifier_rejects_incomplete_or_duplicate_fold_universe():
    with pytest.raises(ValueError, match="exactly folds 0 through 19"):
        qualify_screen([_fold(index) for index in range(19)])
    folds = [_fold(index) for index in range(20)]
    folds[-1]["outer_fold"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        qualify_screen(folds)


def _write_prediction(path, candidate: str, components: int) -> None:
    point = np.asarray([0.1, 0.2])
    np.savez_compressed(
        path,
        keys=np.asarray(["k1", "k2"], dtype=object),
        biological_scoring_key=np.asarray(["k1", "k2"], dtype=object),
        candidate_id=np.full(2, candidate, dtype=object),
        outer_fold=np.zeros(2, dtype=int),
        seed=np.zeros(2, dtype=int),
        delta_mean=point.copy(),
        point_mean=point.copy(),
        locations=np.tile(point[:, None], (1, components)),
        scales=np.full((2, components), 0.3),
        weights=np.full((2, components), 1.0 / components),
        registered_status=np.full(2, "covered", dtype=object),
        mean_checkpoint_path=np.full(2, "mean.pt", dtype=object),
        calibration_checkpoint_path=np.full(2, "cal.pt", dtype=object),
    )


def test_smoke_qualifier_opens_r2m3_only_for_real_artifact_invariants(tmp_path):
    folds = []
    for index in (0, 1):
        mean_path = tmp_path / f"mean{index}.npz"
        calibrated_path = tmp_path / f"calibrated{index}.npz"
        _write_prediction(mean_path, MEAN_CANDIDATE, 1)
        _write_prediction(calibrated_path, CALIBRATED_CANDIDATE, 2)
        folds.append(
            {
                "outer_fold": index,
                "held_puzzle": f"P{index + 1:02d}",
                "seed": 0,
                "point_mean_max_abs_difference": 0.0,
                "held_target_error_mask_invariance": True,
                "candidates": {
                    MEAN_CANDIDATE: {
                        "score": _score(0.2, 0.2),
                        "prediction_artifact": str(mean_path),
                        "mean_loss": [0.3, 0.2, 0.1],
                        "calibration_loss": [0.3, 0.2, 0.1],
                    },
                    CALIBRATED_CANDIDATE: {
                        "score": _score(0.19, 0.2),
                        "prediction_artifact": str(calibrated_path),
                        "mean_loss": [0.3, 0.2, 0.1],
                        "calibration_loss": [0.3, 0.2, 0.1],
                    },
                },
            }
        )
    result = qualify_smoke(folds)
    assert result["overall_status"] == "R2M2_REAL_DATA_ENGINEERING_SMOKE_PASS"
    assert result["r2m3_authorized"] is True
    json.dumps(result)
