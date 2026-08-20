from __future__ import annotations

from scripts.reactflow_delta.qualify_model_rescue_m2_v1 import qualify


def _score(crps: float, delta: float):
    return {
        "score": {
            "crps": crps,
            "signed_delta_mae": delta,
            "registered_prediction_coverage": 1.0,
            "failure_rate": 0.0,
            "n_unexpected_prediction_keys": 0,
        }
    }


def _fold(index: int, sparse_h0_delta: float = 0.19):
    return {
        "outer_fold": index,
        "held_puzzle": f"P{index + 1:02d}",
        "candidates": {
            "b1_rfd_direct_aligned": _score(0.20, 0.20),
            "l2_aligned_rank2": _score(0.19, 0.19),
            "sparse_delta_mdn_h0": _score(0.18, sparse_h0_delta),
            "sparse_delta_mdn_h01": _score(0.21, 0.18),
        },
    }


def test_qualification_requires_both_metrics_and_preserves_inner_lambda_selection():
    result = qualify([_fold(i) for i in range(20)])
    assert result["candidate_results"]["l2_aligned_rank2"]["status"] == "M2_SCREEN_ELIGIBLE"
    assert result["candidate_results"]["sparse_delta_mdn_h0"]["status"] == "M2_SCREEN_ELIGIBLE"
    assert result["candidate_results"]["sparse_delta_mdn_h01"]["status"] == "M2_SCREEN_FAIL"
    assert result["lambda_policy"] == "M3_INNER_SELECT_BETWEEN_0_AND_0.1"
    assert "sparse_delta_mdn_inner_selected_lambda" in result["m3_eligible_families"]


def test_missing_fold_or_nonfinite_coverage_fails_screen():
    folds = [_fold(i) for i in range(19)]
    folds[0]["candidates"]["sparse_delta_mdn_h0"]["score"]["registered_prediction_coverage"] = 0.99
    result = qualify(folds)
    sparse = result["candidate_results"]["sparse_delta_mdn_h0"]
    assert not sparse["checks"]["twenty_folds_complete"]
    assert not sparse["checks"]["registered_prediction_coverage_100pct"]
    assert sparse["status"] == "M2_SCREEN_FAIL"
