import math

import pytest

from reactflow.probing import (
    ProfilePrediction,
    aggregate_full_profiles,
    fit_probe_calibration,
)
from reactflow.synthetic import make_sample
from reactflow.train import sample_from_cache_obj, sample_to_cache_obj


def _profile(source_id, predicted, target, *, source="real_profile", probe="DMS", snr=None, quality=None):
    return ProfilePrediction(
        source_id=source_id,
        probe=probe,
        predicted=predicted,
        target=target,
        weights=[1.0] * len(target),
        reactivity_source=source,
        length=len(target),
        snr=snr,
        quality=quality,
    )


def test_probe_calibration_is_validation_locked_and_test_fit_rejected():
    records = [_profile("a", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0])]
    calibration = fit_probe_calibration(records, split="validation")
    assert calibration["DMS"].alpha == pytest.approx(2.0)
    assert calibration["DMS"].gamma == pytest.approx(1.0)
    with pytest.raises(ValueError, match="train/validation"):
        fit_probe_calibration(records, split="test")


def test_full_profile_aggregation_uses_all_real_profiles_and_excludes_proxy():
    validation = [_profile("v", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0])]
    calibration = fit_probe_calibration(validation, split="validation")
    records = [
        _profile("a", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0], snr=2.0, quality="pass"),
        _profile("b", [1.0, 2.0, 3.0], [3.0, 5.0, 7.0]),
        _profile("proxy", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], source="structure_forward_proxy"),
    ]
    result = aggregate_full_profiles(records, calibration)
    assert result["main"]["profile_count"] == 2
    assert result["main"]["valid_position_count"] == 6
    assert result["main"]["calibrated_mae"] == pytest.approx(0.0)
    assert result["diagnostic_proxy"] == {"profile_count": 1, "included_in_main": False}
    assert result["strata"]["quality"]["missing"]["profile_count"] == 1


def test_unknown_probe_reports_raw_metrics_but_no_calibrated_metric():
    records = [_profile("x", [0.0, 1.0, 2.0], [0.1, 1.1, 2.1], probe="2A3")]
    result = aggregate_full_profiles(records, {})
    assert result["main"]["raw_mae"] == pytest.approx(0.1)
    assert result["main"]["calibrated_mae"] is None
    assert result["main"]["calibrated_profile_count"] == 0


def test_cache_round_trip_preserves_probing_and_window_metadata():
    sample = make_sample(stem=4, loop=4, probe="2A3", seed=4)
    enriched = sample.__class__(
        **{
            **sample.__dict__,
            "reactivity_source": "real_profile",
            "reactivity_error": tuple(0.1 for _ in sample.sequence),
            "reactivity_snr": 3.5,
            "reactivity_quality": "pass",
            "parent_source_id": "parent",
            "window_start": 10,
            "window_end": 22,
            "parent_length": 100,
        }
    )
    restored = sample_from_cache_obj(sample_to_cache_obj(enriched))
    assert restored.reactivity_source == "real_profile"
    assert restored.reactivity_error == pytest.approx(enriched.reactivity_error)
    assert restored.reactivity_snr == pytest.approx(3.5)
    assert restored.reactivity_quality == "pass"
    assert (restored.window_start, restored.window_end, restored.parent_length) == (10, 22, 100)
