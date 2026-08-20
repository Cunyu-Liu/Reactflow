from __future__ import annotations

from scripts.reactflow_delta.qualify_model_rescue_m3_v1 import qualify


def _score(crps: float, delta: float):
    return {
        "crps": crps,
        "signed_delta_mae": delta,
        "coverage68": 0.68,
        "coverage95": 0.95,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
    }


def _fold(index: int, crps_gain: float = 0.01, delta_gain: float = 0.01):
    comparator = _score(0.20, 0.25)
    candidate = _score(0.20 - crps_gain, 0.25 - delta_gain)
    return {
        "outer_fold": index,
        "held_puzzle": f"P{index + 1:02d}",
        "selection": {
            "selected_candidate": "sparse_delta_mdn_h01",
            "selected_comparator": "b1_rfd_direct_aligned",
        },
        "outer_scores": {
            "sparse_delta_mdn_h01": candidate,
            "b1_rfd_direct_aligned": comparator,
        },
        "effects": {
            "crps_gain": crps_gain,
            "signed_delta_mae_gain": delta_gain,
            "signed_delta_mae_gain_vs_wt_anchor": 0.02,
        },
    }


def test_m3_gate_passes_only_complete_dual_practical_effect():
    result = qualify([_fold(i) for i in range(20)])
    assert result["gate"]["status"] == "POST_HOC_DEVELOPMENT_PASS"
    assert all(result["gate"]["checks"].values())


def test_m3_gate_fails_when_only_crps_improves():
    result = qualify([_fold(i, crps_gain=0.01, delta_gain=-0.001) for i in range(20)])
    assert result["gate"]["status"] == "METHOD_RESCUE_FAIL"
    assert not result["gate"]["checks"]["delta_mae_ci"]


def test_m3_gate_does_not_decide_from_incomplete_folds():
    result = qualify([_fold(i) for i in range(19)])
    assert result["gate"]["status"] == "METHOD_RESCUE_FAIL"
    assert result["gate"]["next_route"] == "M3_INCOMPLETE_DO_NOT_DECIDE"
