"""Leakage-safe full-profile probing calibration and aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

from reactflow.evaluate import spearman_correlation
from reactflow.reactivity import fit_weighted_affine_calibration, weighted_pearson


@dataclass(frozen=True)
class ProfilePrediction:
    source_id: str
    probe: str
    predicted: Sequence[float]
    target: Sequence[float]
    weights: Sequence[float]
    reactivity_source: str
    length: int
    snr: Optional[float] = None
    quality: Optional[str] = None


@dataclass(frozen=True)
class ProbeCalibration:
    probe: str
    alpha: float
    gamma: float
    valid_position_count: int
    fitted_split: str = "validation"


def _valid(record: ProfilePrediction) -> list[tuple[float, float, float]]:
    if not (len(record.predicted) == len(record.target) == len(record.weights)):
        raise ValueError("predicted, target and weights must have the same length")
    return [
        (float(pred), float(target), float(weight))
        for pred, target, weight in zip(record.predicted, record.target, record.weights)
        if math.isfinite(float(pred))
        and math.isfinite(float(target))
        and math.isfinite(float(weight))
        and float(weight) > 0.0
    ]


def fit_probe_calibration(
    records: Sequence[ProfilePrediction],
    *,
    split: str,
) -> Dict[str, ProbeCalibration]:
    """Fit one affine mapping per probe; test splits are explicitly rejected."""

    if str(split).lower() not in {"validation", "val", "train"}:
        raise ValueError("probing calibration may only be fit on train/validation")
    grouped: Dict[str, list[tuple[float, float, float]]] = {}
    for record in records:
        if record.reactivity_source != "real_profile":
            continue
        grouped.setdefault(record.probe, []).extend(_valid(record))
    result: Dict[str, ProbeCalibration] = {}
    for probe, values in grouped.items():
        if not values:
            continue
        predicted, target, weights = zip(*values)
        alpha, gamma = fit_weighted_affine_calibration(predicted, target, weights)
        result[probe] = ProbeCalibration(
            probe=probe,
            alpha=alpha,
            gamma=gamma,
            valid_position_count=len(values),
            fitted_split="validation" if str(split).lower() in {"validation", "val"} else "train",
        )
    return result


def calibration_manifest(calibrations: Mapping[str, ProbeCalibration]) -> dict:
    return {
        "schema_version": 1,
        "fit_scope": "probe_type_pooled",
        "test_refit_allowed": False,
        "probes": {probe: asdict(value) for probe, value in sorted(calibrations.items())},
    }


def _length_bucket(length: int) -> str:
    for bound in (64, 128, 256, 512, 1024):
        if length <= bound:
            return f"len_le_{bound}"
    return "len_gt_1024"


def _aggregate(
    records: Sequence[ProfilePrediction],
    calibrations: Mapping[str, ProbeCalibration],
) -> dict:
    pooled_pred: list[float] = []
    pooled_target: list[float] = []
    pooled_weight: list[float] = []
    pooled_calibrated: list[float] = []
    calibrated_target: list[float] = []
    calibrated_weight: list[float] = []
    profile_pearson: list[float] = []
    profile_spearman: list[float] = []
    calibrated_profiles = 0
    for record in records:
        valid = _valid(record)
        if not valid:
            continue
        pred, target, weights = zip(*valid)
        pooled_pred.extend(pred)
        pooled_target.extend(target)
        pooled_weight.extend(weights)
        if len(valid) >= 3:
            profile_pearson.append(weighted_pearson(pred, target, weights))
            profile_spearman.append(spearman_correlation(pred, target))
        calibration = calibrations.get(record.probe)
        if calibration is not None:
            calibrated_profiles += 1
            pooled_calibrated.extend(calibration.alpha * value + calibration.gamma for value in pred)
            calibrated_target.extend(target)
            calibrated_weight.extend(weights)

    def weighted_error(predicted: Sequence[float], target: Sequence[float], weights: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return None, None
        mae = sum(w * abs(p - t) for p, t, w in zip(predicted, target, weights)) / total_weight
        rmse = math.sqrt(sum(w * (p - t) ** 2 for p, t, w in zip(predicted, target, weights)) / total_weight)
        return mae, rmse

    raw_mae, raw_rmse = weighted_error(pooled_pred, pooled_target, pooled_weight)
    calibrated_mae, calibrated_rmse = weighted_error(
        pooled_calibrated, calibrated_target, calibrated_weight
    )
    return {
        "profile_count": len(records),
        "valid_position_count": len(pooled_pred),
        "correlation_profile_count": len(profile_pearson),
        "profile_macro_pearson": sum(profile_pearson) / len(profile_pearson) if profile_pearson else None,
        "profile_macro_spearman": sum(profile_spearman) / len(profile_spearman) if profile_spearman else None,
        "position_pooled_pearson": weighted_pearson(pooled_pred, pooled_target, pooled_weight) if pooled_pred else None,
        "position_pooled_spearman": spearman_correlation(pooled_pred, pooled_target) if pooled_pred else None,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "calibrated_mae": calibrated_mae,
        "calibrated_rmse": calibrated_rmse,
        "calibrated_profile_count": calibrated_profiles,
    }


def aggregate_full_profiles(
    records: Sequence[ProfilePrediction],
    calibrations: Mapping[str, ProbeCalibration],
) -> dict:
    """Aggregate every real profile and keep proxy records diagnostic-only."""

    real = [record for record in records if record.reactivity_source == "real_profile"]
    proxy = [record for record in records if record.reactivity_source == "structure_forward_proxy"]
    by_probe: Dict[str, list[ProfilePrediction]] = {}
    by_length: Dict[str, list[ProfilePrediction]] = {}
    by_quality: Dict[str, list[ProfilePrediction]] = {}
    by_snr: Dict[str, list[ProfilePrediction]] = {}
    for record in real:
        by_probe.setdefault(record.probe, []).append(record)
        by_length.setdefault(_length_bucket(record.length), []).append(record)
        by_quality.setdefault(record.quality or "missing", []).append(record)
        snr_label = "missing" if record.snr is None else ("snr_lt_1" if record.snr < 1 else "snr_ge_1")
        by_snr.setdefault(snr_label, []).append(record)
    return {
        "schema_version": 1,
        "main": _aggregate(real, calibrations),
        "strata": {
            "probe": {key: _aggregate(value, calibrations) for key, value in sorted(by_probe.items())},
            "length": {key: _aggregate(value, calibrations) for key, value in sorted(by_length.items())},
            "quality": {key: _aggregate(value, calibrations) for key, value in sorted(by_quality.items())},
            "snr": {key: _aggregate(value, calibrations) for key, value in sorted(by_snr.items())},
        },
        "diagnostic_proxy": {
            "profile_count": len(proxy),
            "included_in_main": False,
        },
        "excluded_unknown_source_count": sum(
            1 for record in records if record.reactivity_source not in {"real_profile", "structure_forward_proxy"}
        ),
    }
