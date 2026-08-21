#!/usr/bin/env python3
"""Mechanically qualify the frozen Model Rescue v2 seed-0 screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    MEAN_CANDIDATE,
)
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE


SCHEMA = "reactflow_delta.model_rescue_v2_screen_qualification.v1"
SMOKE_SCHEMA = "reactflow_delta.model_rescue_v2_smoke_qualification.v1"
FORMAL_SCHEMA = "reactflow_delta.model_rescue_v2_formal_qualification.v1"


def _load_folds(path: Path, phase: str) -> list[dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        folds = data.get("folds")
        if not isinstance(folds, list):
            raise ValueError("result JSON does not contain a folds list")
        return folds
    pattern = (
        "v2_formal_fold_result_fold*.json"
        if phase == "R2M4"
        else "v2_fold_result_fold*_seed0.json"
    )
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no seed-0 per-fold artifacts below {path}")
    return [json.loads(file.read_text(encoding="utf-8")) for file in files]


def _paired_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        low = high = float("nan")
    else:
        half = float(
            student_t.ppf(0.975, len(array) - 1)
            * array.std(ddof=1)
            / math.sqrt(len(array))
        )
        low, high = mean - half, mean + half
    return {
        "n": len(array),
        "mean": mean,
        "ci95": [low, high],
        "positive_puzzles": int((array > 0).sum()),
        "per_puzzle": array.tolist(),
    }


def _leave_one_positive(values: list[float]) -> bool:
    array = np.asarray(values, dtype=float)
    return len(array) >= 2 and all(
        float(np.delete(array, index).mean()) > 0.0 for index in range(len(array))
    )


def _formal_prediction_checks(
    path: Path, candidate: str, components: int
) -> tuple[dict[str, bool], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    required = {
        "keys",
        "candidate_id",
        "seed_universe",
        "seed_point_means",
        "point_mean",
        "locations",
        "scales",
        "weights",
        "registered_status",
    }
    prohibited = {
        "target",
        "target_error",
        "target_mask",
        "qualified_target_mask",
        "score",
    }
    keys = prediction.get("keys", np.empty(0, dtype=object))
    locations = prediction.get("locations", np.empty((0, 0)))
    scales = prediction.get("scales", np.empty((0, 0)))
    weights = prediction.get("weights", np.empty((0, 0)))
    point_mean = prediction.get("point_mean", np.empty(0))
    seed_point_means = prediction.get("seed_point_means", np.empty((0, 0)))
    n_rows = len(keys)
    shapes_valid = (
        locations.shape == (n_rows, components)
        and scales.shape == (n_rows, components)
        and weights.shape == (n_rows, components)
        and point_mean.shape == (n_rows,)
        and seed_point_means.shape == (n_rows, 5)
    )
    mixture_mean_valid = shapes_valid and np.allclose(
        np.sum(weights * locations, axis=1), point_mean, atol=1e-7, rtol=0
    )
    seed_layout_valid = False
    if shapes_valid and candidate == CALIBRATED_CANDIDATE:
        seed_layout_valid = all(
            np.allclose(locations[:, 2 * seed], seed_point_means[:, seed], atol=1e-7, rtol=0)
            and np.allclose(
                locations[:, 2 * seed + 1], seed_point_means[:, seed], atol=1e-7, rtol=0
            )
            and np.allclose(
                weights[:, 2 * seed] + weights[:, 2 * seed + 1],
                0.2,
                atol=1e-7,
                rtol=0,
            )
            for seed in range(5)
        )
    elif shapes_valid and candidate == BASELINE:
        seed_layout_valid = np.allclose(locations, seed_point_means, atol=1e-7, rtol=0)
        seed_layout_valid = bool(seed_layout_valid) and bool(
            np.allclose(weights, 0.2, atol=1e-7, rtol=0)
        )
    checks = {
        "required_fields_present": required.issubset(prediction),
        "prohibited_target_fields_absent": prohibited.isdisjoint(prediction),
        "nonempty_unique_key_universe": n_rows > 0 and len(set(map(str, keys))) == n_rows,
        "candidate_id_constant": n_rows > 0
        and set(map(str, prediction.get("candidate_id", []))) == {candidate},
        "seed_universe_exact_0_to_4": np.array_equal(
            prediction.get("seed_universe"), np.arange(5)
        ),
        "registered_status_covered": n_rows > 0
        and set(map(str, prediction.get("registered_status", []))) == {"covered"},
        "formal_component_shapes": shapes_valid,
        "positive_finite_scales": shapes_valid
        and bool(np.isfinite(scales).all())
        and bool((scales > 0).all()),
        "weights_sum_to_one": shapes_valid
        and bool(np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0)),
        "mixture_mean_equals_reported_point_mean": mixture_mean_valid,
        "five_seed_component_layout": seed_layout_valid,
    }
    return {name: bool(value) for name, value in checks.items()}, prediction


def _score(row: dict[str, Any], candidate: str) -> dict[str, Any]:
    return row["candidates"][candidate]["score"]


def _baseline_score(row: dict[str, Any]) -> dict[str, Any]:
    baseline = row.get("baseline")
    if not baseline or baseline.get("model_id") != BASELINE:
        raise ValueError(f"fold {row.get('outer_fold')} has no frozen {BASELINE} comparator")
    return baseline["score"]


def _integrity(score: dict[str, Any]) -> dict[str, bool]:
    return {
        "registered_prediction_coverage_100pct": float(
            score.get("registered_prediction_coverage", float("nan"))
        )
        == 1.0,
        "failure_rate_zero": float(score.get("failure_rate", float("nan"))) == 0.0,
        "unexpected_prediction_keys_zero": int(
            score.get("n_unexpected_prediction_keys", -1)
        )
        == 0,
    }


def _prediction_checks(path: Path, candidate: str, components: int) -> tuple[dict[str, bool], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    prohibited = {
        "target",
        "target_error",
        "target_mask",
        "qualified_target_mask",
        "score",
    }
    n_rows = len(prediction.get("keys", []))
    required = {
        "biological_scoring_key",
        "candidate_id",
        "outer_fold",
        "seed",
        "delta_mean",
        "point_mean",
        "locations",
        "scales",
        "weights",
        "registered_status",
        "mean_checkpoint_path",
        "calibration_checkpoint_path",
    }
    checks = {
        "required_prediction_fields_present": required.issubset(prediction),
        "prohibited_target_fields_absent": prohibited.isdisjoint(prediction),
        "nonempty_prediction_ledger": n_rows > 0,
        "biological_key_matches_scorer_key": np.array_equal(
            prediction.get("keys"), prediction.get("biological_scoring_key")
        ),
        "candidate_id_constant": n_rows > 0
        and set(map(str, prediction.get("candidate_id", []))) == {candidate},
        "registered_status_covered": n_rows > 0
        and set(map(str, prediction.get("registered_status", []))) == {"covered"},
        "component_count": prediction.get("locations", np.empty((0, 0))).shape
        == (n_rows, components)
        and prediction.get("scales", np.empty((0, 0))).shape == (n_rows, components)
        and prediction.get("weights", np.empty((0, 0))).shape == (n_rows, components),
        "locations_equal_point_mean": n_rows > 0
        and np.allclose(
            prediction.get("locations"),
            prediction.get("point_mean", np.empty(0))[:, None],
            atol=1e-7,
            rtol=0,
        ),
        "weights_sum_to_one": n_rows > 0
        and np.allclose(
            prediction.get("weights", np.empty((0, 0))).sum(axis=1),
            1.0,
            atol=1e-7,
            rtol=0,
        ),
        "positive_finite_scales": n_rows > 0
        and np.isfinite(prediction.get("scales", np.empty(0))).all()
        and (prediction.get("scales", np.empty(0)) > 0).all(),
    }
    return checks, prediction


def qualify_smoke(folds: list[dict[str, Any]]) -> dict[str, Any]:
    fold_ids = [int(row["outer_fold"]) for row in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate outer fold result")
    if sorted(fold_ids) != [0, 1] or len(folds) != 2:
        raise ValueError("R2M2 smoke requires exactly folds 0 and 1")
    fold_results = []
    for row in sorted(folds, key=lambda item: int(item["outer_fold"])):
        if int(row.get("seed", -1)) != 0:
            raise ValueError("R2M2 smoke accepts seed 0 only")
        candidate_checks = {}
        predictions = {}
        for candidate, components in ((MEAN_CANDIDATE, 1), (CALIBRATED_CANDIDATE, 2)):
            candidate_row = row["candidates"][candidate]
            artifact_checks, prediction = _prediction_checks(
                Path(candidate_row["prediction_artifact"]), candidate, components
            )
            histories = candidate_row["mean_loss"] + candidate_row["calibration_loss"]
            score_checks = _integrity(candidate_row["score"])
            checks = {
                **artifact_checks,
                **score_checks,
                "three_or_fewer_epochs_each_stage": len(candidate_row["mean_loss"]) <= 3
                and len(candidate_row["calibration_loss"]) <= 3,
                "finite_training_losses": len(histories) > 0
                and bool(np.isfinite(np.asarray(histories, dtype=float)).all()),
            }
            candidate_checks[candidate] = {
                name: bool(value) for name, value in checks.items()
            }
            predictions[candidate] = prediction
        pair_checks = {
            "candidate_keys_identical": np.array_equal(
                predictions[MEAN_CANDIDATE]["keys"],
                predictions[CALIBRATED_CANDIDATE]["keys"],
            ),
            "candidate_delta_mean_identical": np.array_equal(
                predictions[MEAN_CANDIDATE]["delta_mean"],
                predictions[CALIBRATED_CANDIDATE]["delta_mean"],
            ),
            "candidate_point_mean_identical": np.array_equal(
                predictions[MEAN_CANDIDATE]["point_mean"],
                predictions[CALIBRATED_CANDIDATE]["point_mean"],
            ),
            "held_target_error_mask_invariance": row.get(
                "held_target_error_mask_invariance"
            )
            is True,
            "reported_point_difference_atol_1e_7": float(
                row.get("point_mean_max_abs_difference", float("inf"))
            )
            <= 1e-7,
        }
        pair_checks = {name: bool(value) for name, value in pair_checks.items()}
        passed = all(pair_checks.values()) and all(
            all(checks.values()) for checks in candidate_checks.values()
        )
        fold_results.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "held_puzzle": row["held_puzzle"],
                "candidate_checks": candidate_checks,
                "pair_checks": pair_checks,
                "status": "ENGINEERING_SMOKE_FOLD_PASS" if passed else "ENGINEERING_SMOKE_FOLD_FAIL",
            }
        )
    passed = all(row["status"] == "ENGINEERING_SMOKE_FOLD_PASS" for row in fold_results)
    return {
        "schema_version": SMOKE_SCHEMA,
        "evidence_status": "ENGINEERING_SMOKE_ONLY",
        "folds": fold_results,
        "overall_status": "R2M2_REAL_DATA_ENGINEERING_SMOKE_PASS" if passed else "R2M2_FAIL",
        "r2m3_authorized": passed,
        "scientific_interpretation_prohibited": True,
    }


def qualify_screen(folds: list[dict[str, Any]]) -> dict[str, Any]:
    fold_ids = [int(row["outer_fold"]) for row in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate outer fold result")
    if any(int(row.get("seed", -1)) != 0 for row in folds):
        raise ValueError("R2M3 screen accepts seed 0 only")
    complete = len(folds) == 20 and sorted(fold_ids) == list(range(20))
    if not complete:
        raise ValueError("R2M3 qualification requires exactly folds 0 through 19")

    mean_delta_gains: list[float] = []
    calibrated_crps_gains: list[float] = []
    point_differences: list[float] = []
    candidate_delta_differences: list[float] = []
    baseline_delta: list[float] = []
    mean_integrity: list[dict[str, bool]] = []
    calibration_integrity: list[dict[str, bool]] = []
    per_fold: list[dict[str, Any]] = []
    for row in sorted(folds, key=lambda item: int(item["outer_fold"])):
        missing = {MEAN_CANDIDATE, CALIBRATED_CANDIDATE} - set(row["candidates"])
        if missing:
            raise ValueError(f"fold {row['outer_fold']} missing candidates {sorted(missing)}")
        base = _baseline_score(row)
        mean = _score(row, MEAN_CANDIDATE)
        calibrated = _score(row, CALIBRATED_CANDIDATE)
        delta_gain = float(base["signed_delta_mae"] - mean["signed_delta_mae"])
        crps_gain = float(base["crps"] - calibrated["crps"])
        point_difference = float(row.get("point_mean_max_abs_difference", float("inf")))
        delta_difference = abs(
            float(mean["signed_delta_mae"]) - float(calibrated["signed_delta_mae"])
        )
        mean_delta_gains.append(delta_gain)
        calibrated_crps_gains.append(crps_gain)
        point_differences.append(point_difference)
        candidate_delta_differences.append(delta_difference)
        baseline_delta.append(float(base["signed_delta_mae"]))
        mean_integrity.append(_integrity(mean))
        calibration_integrity.append(_integrity(calibrated))
        per_fold.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "held_puzzle": row["held_puzzle"],
                "mean_signed_delta_mae_gain_vs_b1": delta_gain,
                "calibrated_crps_gain_vs_b1": crps_gain,
                "point_mean_max_abs_difference": point_difference,
                "candidate_signed_delta_mae_abs_difference": delta_difference,
            }
        )

    mean_gain = float(np.mean(mean_delta_gains))
    crps_gain = float(np.mean(calibrated_crps_gains))
    baseline_delta_mean = float(np.mean(baseline_delta))
    relative_mean_gain = mean_gain / baseline_delta_mean
    mean_checks = {
        "mean_signed_delta_mae_gain_positive": mean_gain > 0.0,
        "signed_delta_mae_relative_gain_at_least_1pct": relative_mean_gain >= 0.01,
        "signed_delta_mae_positive_puzzles_at_least_12": int(
            np.sum(np.asarray(mean_delta_gains) > 0.0)
        )
        >= 12,
        "registered_prediction_coverage_100pct": all(
            item["registered_prediction_coverage_100pct"] for item in mean_integrity
        ),
        "failure_rate_zero": all(item["failure_rate_zero"] for item in mean_integrity),
        "unexpected_prediction_keys_zero": all(
            item["unexpected_prediction_keys_zero"] for item in mean_integrity
        ),
    }
    calibration_checks = {
        "point_mean_identical_atol_1e_7": max(point_differences) <= 1e-7,
        "signed_delta_mae_identical_atol_1e_7": max(candidate_delta_differences) <= 1e-7,
        "mean_crps_gain_positive": crps_gain > 0.0,
        "crps_positive_puzzles_at_least_12": int(
            np.sum(np.asarray(calibrated_crps_gains) > 0.0)
        )
        >= 12,
        "signed_delta_positive_puzzles_inherited_at_least_12": int(
            np.sum(np.asarray(mean_delta_gains) > 0.0)
        )
        >= 12,
        "registered_prediction_coverage_100pct": all(
            item["registered_prediction_coverage_100pct"]
            for item in calibration_integrity
        ),
        "failure_rate_zero": all(
            item["failure_rate_zero"] for item in calibration_integrity
        ),
        "unexpected_prediction_keys_zero": all(
            item["unexpected_prediction_keys_zero"]
            for item in calibration_integrity
        ),
    }
    mean_pass = all(mean_checks.values())
    calibration_pass = all(calibration_checks.values())
    return {
        "schema_version": SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_SCREEN_NOT_CONFIRMATION",
        "fold_integrity": {
            "n_fold_artifacts": len(folds),
            "unique_fold_ids": sorted(fold_ids),
            "complete_0_through_19": complete,
            "seed": 0,
        },
        "mean_gate": {
            "candidate": MEAN_CANDIDATE,
            "mean_signed_delta_mae_gain_vs_b1": mean_gain,
            "relative_signed_delta_mae_gain": relative_mean_gain,
            "positive_puzzles": int(np.sum(np.asarray(mean_delta_gains) > 0.0)),
            "checks": mean_checks,
            "status": "MEAN_GATE_PASS" if mean_pass else "MEAN_GATE_FAIL",
        },
        "calibration_gate": {
            "candidate": CALIBRATED_CANDIDATE,
            "mean_crps_gain_vs_b1": crps_gain,
            "positive_puzzles": int(np.sum(np.asarray(calibrated_crps_gains) > 0.0)),
            "max_point_mean_abs_difference": max(point_differences),
            "max_signed_delta_mae_abs_difference": max(candidate_delta_differences),
            "checks": calibration_checks,
            "status": (
                "CALIBRATION_GATE_PASS" if calibration_pass else "CALIBRATION_GATE_FAIL"
            ),
        },
        "per_fold": per_fold,
        "overall_status": (
            "R2M3_SCREEN_PASS" if mean_pass and calibration_pass else "MODEL_RESCUE_V2_FAIL"
        ),
        "r2m4_authorized": bool(mean_pass and calibration_pass),
        "prohibited_interpretation": [
            "EXTERNAL_REPLICATION",
            "SOTA",
            "MECHANISM",
            "PUBLICATION_PASS",
        ],
    }


def qualify_formal(folds: list[dict[str, Any]]) -> dict[str, Any]:
    folds = sorted(folds, key=lambda row: int(row["outer_fold"]))
    fold_ids = [int(row["outer_fold"]) for row in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate R2M4 outer fold result")
    if len(folds) != 20 or fold_ids != list(range(20)):
        raise ValueError("R2M4 qualification requires exactly folds 0 through 19")

    crps_effects: list[float] = []
    delta_effects: list[float] = []
    baseline_crps: list[float] = []
    baseline_delta: list[float] = []
    coverage68_worsening: list[float] = []
    coverage95_worsening: list[float] = []
    artifact_integrity: list[bool] = []
    score_integrity: list[bool] = []
    seed_invariance: list[bool] = []
    per_fold: list[dict[str, Any]] = []
    for row in folds:
        baseline = row.get("baseline", {})
        candidate = row.get("candidate", {})
        if baseline.get("model_id") != BASELINE:
            raise ValueError(f"fold {row['outer_fold']} has wrong formal baseline")
        if candidate.get("model_id") != CALIBRATED_CANDIDATE:
            raise ValueError(f"fold {row['outer_fold']} has wrong formal candidate")
        if baseline.get("seed_universe") != list(range(5)):
            raise ValueError(f"fold {row['outer_fold']} baseline seed universe is not 0..4")
        if candidate.get("seed_universe") != list(range(5)):
            raise ValueError(f"fold {row['outer_fold']} candidate seed universe is not 0..4")
        baseline_checks, baseline_prediction = _formal_prediction_checks(
            Path(baseline["prediction_artifact"]), BASELINE, 5
        )
        candidate_checks, candidate_prediction = _formal_prediction_checks(
            Path(candidate["prediction_artifact"]), CALIBRATED_CANDIDATE, 10
        )
        keys_identical = np.array_equal(
            baseline_prediction["keys"], candidate_prediction["keys"]
        )
        artifact_ok = (
            all(baseline_checks.values())
            and all(candidate_checks.values())
            and bool(keys_identical)
        )
        artifact_integrity.append(artifact_ok)
        baseline_score = baseline["score"]
        candidate_score = candidate["score"]
        baseline_integrity = _integrity(baseline_score)
        candidate_integrity = _integrity(candidate_score)
        score_ok = all(baseline_integrity.values()) and all(candidate_integrity.values())
        score_integrity.append(score_ok)
        seed_invariance.append(row.get("all_seed_target_error_mask_invariance") is True)

        crps_gain = float(baseline_score["crps"] - candidate_score["crps"])
        delta_gain = float(
            baseline_score["signed_delta_mae"] - candidate_score["signed_delta_mae"]
        )
        crps_effects.append(crps_gain)
        delta_effects.append(delta_gain)
        baseline_crps.append(float(baseline_score["crps"]))
        baseline_delta.append(float(baseline_score["signed_delta_mae"]))
        worsening68 = abs(float(candidate_score["coverage68"]) - 0.68) - abs(
            float(baseline_score["coverage68"]) - 0.68
        )
        worsening95 = abs(float(candidate_score["coverage95"]) - 0.95) - abs(
            float(baseline_score["coverage95"]) - 0.95
        )
        coverage68_worsening.append(worsening68)
        coverage95_worsening.append(worsening95)
        per_fold.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "held_puzzle": row["held_puzzle"],
                "crps_gain_vs_b1": crps_gain,
                "signed_delta_mae_gain_vs_b1": delta_gain,
                "baseline_artifact_checks": baseline_checks,
                "candidate_artifact_checks": candidate_checks,
                "key_universe_identical": bool(keys_identical),
                "all_seed_target_error_mask_invariance": seed_invariance[-1],
            }
        )

    crps = _paired_ci(crps_effects)
    delta = _paired_ci(delta_effects)
    mean_baseline_crps = float(np.mean(baseline_crps))
    mean_baseline_delta = float(np.mean(baseline_delta))
    crps_required_gain = max(0.003, 0.02 * mean_baseline_crps)
    delta_relative_gain = delta["mean"] / max(mean_baseline_delta, 1e-12)
    normalized = 0.5 * np.asarray(crps_effects) / max(mean_baseline_crps, 1e-12)
    normalized += 0.5 * np.asarray(delta_effects) / max(mean_baseline_delta, 1e-12)
    max_fraction = float(
        np.max(np.abs(normalized)) / max(np.sum(np.abs(normalized)), 1e-12)
    )
    coverage68_worsening_pp = 100.0 * float(np.mean(coverage68_worsening))
    coverage95_worsening_pp = 100.0 * float(np.mean(coverage95_worsening))
    checks = {
        "twenty_outer_folds_complete": True,
        "baseline_and_candidate_seed_universe_exact_0_to_4": True,
        "formal_prediction_artifacts_valid": all(artifact_integrity),
        "score_key_coverage_integrity": all(score_integrity),
        "all_seed_target_error_mask_invariance": all(seed_invariance),
        "crps_ci95_lower_positive": crps["ci95"][0] > 0.0,
        "crps_gain_at_least_max_0_003_or_2pct": crps["mean"] >= crps_required_gain,
        "signed_delta_mae_ci95_lower_positive": delta["ci95"][0] > 0.0,
        "signed_delta_mae_relative_gain_at_least_1pct": delta_relative_gain >= 0.01,
        "crps_positive_puzzles_at_least_14": crps["positive_puzzles"] >= 14,
        "signed_delta_mae_positive_puzzles_at_least_12": delta["positive_puzzles"] >= 12,
        "crps_leave_one_puzzle_effect_positive": _leave_one_positive(crps_effects),
        "signed_delta_leave_one_puzzle_effect_positive": _leave_one_positive(
            delta_effects
        ),
        "max_single_puzzle_effect_fraction_at_most_0_25": max_fraction <= 0.25,
        "coverage68_absolute_error_worsening_at_most_2pp": coverage68_worsening_pp
        <= 2.0,
        "coverage95_absolute_error_worsening_at_most_2pp": coverage95_worsening_pp
        <= 2.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    crps_only = (
        checks["crps_ci95_lower_positive"]
        and checks["crps_gain_at_least_max_0_003_or_2pct"]
        and not (
            checks["signed_delta_mae_ci95_lower_positive"]
            and checks["signed_delta_mae_relative_gain_at_least_1pct"]
        )
    )
    return {
        "schema_version": FORMAL_SCHEMA,
        "evidence_status": "POST_HOC_DEVELOPMENT_FORMAL_NOT_EXTERNAL_CONFIRMATION",
        "fold_integrity": {"fold_ids": fold_ids, "complete_0_through_19": True},
        "crps_effect": crps,
        "signed_delta_mae_effect": delta,
        "mean_baseline_crps": mean_baseline_crps,
        "mean_baseline_signed_delta_mae": mean_baseline_delta,
        "required_crps_gain": crps_required_gain,
        "relative_signed_delta_mae_gain": delta_relative_gain,
        "max_single_puzzle_effect_fraction": max_fraction,
        "coverage68_absolute_error_worsening_pp": coverage68_worsening_pp,
        "coverage95_absolute_error_worsening_pp": coverage95_worsening_pp,
        "checks": checks,
        "per_fold": per_fold,
        "overall_status": (
            "R2M4_POST_HOC_DEVELOPMENT_PASS" if passed else "MODEL_RESCUE_V2_FAIL"
        ),
        "route": (
            "BENCHMARK_WITH_MEAN_FIRST_DEVELOPMENT_BASELINE"
            if passed
            else "BENCHMARK_ROUTE_LOCKED"
        ),
        "model_qualification": (
            "POST_HOC_DEVELOPMENT_PASS"
            if passed
            else "CALIBRATION_BASELINE_ONLY"
            if crps_only
            else "MODEL_RESCUE_V2_FAIL"
        ),
        "prohibited_interpretation": [
            "EXTERNAL_REPLICATION",
            "SOTA",
            "MECHANISM",
            "PUBLICATION_PASS",
        ],
    }
def render_markdown(result: dict[str, Any]) -> str:
    if result["schema_version"] == SMOKE_SCHEMA:
        return "\n".join(
            [
                "# ReactFlow-Delta Model Rescue v2 real-data engineering smoke",
                "",
                f"Overall status: `{result['overall_status']}`.",
                "",
                "This artifact establishes engineering invariants only. Smoke scores are not eligible for model selection or scientific interpretation.",
                "",
            ]
        )
    if result["schema_version"] == FORMAL_SCHEMA:
        crps = result["crps_effect"]
        delta = result["signed_delta_mae_effect"]
        lines = [
            "# ReactFlow-Delta Model Rescue v2 five-seed formal qualification",
            "",
            f"Overall status: `{result['overall_status']}`; route: `{result['route']}`.",
            "",
            f"- CRPS gain: {crps['mean']:+.8f}, 95% Student-t CI [{crps['ci95'][0]:+.8f}, {crps['ci95'][1]:+.8f}], positive puzzles {crps['positive_puzzles']}/20.",
            f"- signed-delta MAE gain: {delta['mean']:+.8f}, 95% Student-t CI [{delta['ci95'][0]:+.8f}, {delta['ci95'][1]:+.8f}], positive puzzles {delta['positive_puzzles']}/20.",
            "",
            "## Gate checks",
            "",
        ]
        lines.extend(f"- `{name}`: `{value}`" for name, value in result["checks"].items())
        lines.extend(
            [
                "",
                "This consumed-development formal result cannot establish external replication, SOTA, mechanism, or publication readiness.",
                "",
            ]
        )
        return "\n".join(lines)
    mean = result["mean_gate"]
    calibration = result["calibration_gate"]
    return "\n".join(
        [
            "# ReactFlow-Delta Model Rescue v2 seed-0 screen qualification",
            "",
            f"Overall status: `{result['overall_status']}`.",
            f"Evidence status: `{result['evidence_status']}`.",
            "",
            "| gate | mean effect | relative effect | positive puzzles | status |",
            "|---|---:|---:|---:|---|",
            f"| Mean | {mean['mean_signed_delta_mae_gain_vs_b1']:+.8f} signed-delta MAE | {mean['relative_signed_delta_mae_gain']:.3%} | {mean['positive_puzzles']}/20 | `{mean['status']}` |",
            f"| Calibration | {calibration['mean_crps_gain_vs_b1']:+.8f} CRPS | n/a | {calibration['positive_puzzles']}/20 | `{calibration['status']}` |",
            "",
            f"Maximum Candidate A/B point-mean difference: `{calibration['max_point_mean_abs_difference']:.3e}`.",
            "",
            "This consumed-development screen can authorize the frozen five-seed confirmation only. It does not establish external replication, SOTA, mechanism, or publication readiness.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--phase", choices=["R2M2", "R2M3", "R2M4"], default="R2M3")
    args = parser.parse_args(argv)
    folds = _load_folds(args.input, args.phase)
    if args.phase == "R2M2":
        result = qualify_smoke(folds)
    elif args.phase == "R2M3":
        result = qualify_screen(folds)
    else:
        result = qualify_formal(folds)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["overall_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
