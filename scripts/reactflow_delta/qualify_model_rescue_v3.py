#!/usr/bin/env python3
"""Mechanically qualify Model Rescue v3 artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from scripts.reactflow_delta.model_rescue_v3 import (
    CANDIDATE,
    INNER_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE


SMOKE_SCHEMA = "reactflow_delta.model_rescue_v3_smoke_qualification.v1"
SCREEN_SCHEMA = "reactflow_delta.model_rescue_v3_screen_qualification.v1"
FORMAL_SCHEMA = "reactflow_delta.model_rescue_v3_formal_qualification.v1"


def _load_folds(path: Path, phase: str) -> list[dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        folds = data.get("folds")
        if not isinstance(folds, list):
            raise ValueError("result JSON does not contain a folds list")
        return folds
    pattern = (
        "v3_formal_fold_result_fold*.json"
        if phase == "R3M4"
        else "v3_fold_result_fold*_seed0.json"
    )
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no v3 fold artifacts below {path}")
    return [json.loads(file.read_text(encoding="utf-8")) for file in files]


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


def _prediction_checks(path: Path) -> dict[str, bool]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    required = {
        "schema_version",
        "keys",
        "biological_scoring_key",
        "candidate_id",
        "outer_fold",
        "seed",
        "b1_delta_mean",
        "meanaligned_delta_mean",
        "expert_disagreement",
        "gate_threshold",
        "gate_alpha_low",
        "gate_alpha_high",
        "gate_alpha_applied",
        "delta_mean",
        "point_mean",
        "locations",
        "scales",
        "weights",
        "registered_status",
        "b1_checkpoint_path",
        "meanaligned_checkpoint_path",
        "calibration_checkpoint_path",
        "inner_crossfit_ledger_path",
    }
    prohibited = {
        "target",
        "target_error",
        "target_mask",
        "qualified_target_mask",
        "qualified_mask",
        "score",
        "method",
        "design_method",
    }
    keys = prediction.get("keys", np.empty(0, dtype=object))
    n = len(keys)
    b1 = prediction.get("b1_delta_mean", np.empty(0))
    mean = prediction.get("meanaligned_delta_mean", np.empty(0))
    disagreement = prediction.get("expert_disagreement", np.empty(0))
    threshold = prediction.get("gate_threshold", np.empty(0))
    alpha_low = prediction.get("gate_alpha_low", np.empty(0))
    alpha_high = prediction.get("gate_alpha_high", np.empty(0))
    alpha = prediction.get("gate_alpha_applied", np.empty(0))
    delta = prediction.get("delta_mean", np.empty(0))
    point = prediction.get("point_mean", np.empty(0))
    locations = prediction.get("locations", np.empty((0, 0)))
    scales = prediction.get("scales", np.empty((0, 0)))
    weights = prediction.get("weights", np.empty((0, 0)))
    shapes = (
        all(array.shape == (n,) for array in (b1, mean, disagreement, threshold, alpha_low, alpha_high, alpha, delta, point))
        and locations.shape == (n, 2)
        and scales.shape == (n, 2)
        and weights.shape == (n, 2)
    )
    expected_disagreement = np.abs(b1 - mean) if shapes else np.empty(0)
    expected_alpha = (
        np.where(expected_disagreement > threshold, alpha_high, alpha_low)
        if shapes
        else np.empty(0)
    )
    expected_delta = b1 + expected_alpha * (mean - b1) if shapes else np.empty(0)
    mixture_mean = (
        np.sum(weights * locations, axis=1) / weights.sum(axis=1)
        if shapes
        else np.empty(0)
    )
    checks = {
        "schema_version_exact": str(prediction.get("schema_version", "")) == PREDICTION_SCHEMA,
        "required_fields_present": required.issubset(prediction),
        "prohibited_target_and_method_fields_absent": prohibited.isdisjoint(prediction),
        "nonempty_unique_key_universe": n > 0 and len(set(map(str, keys))) == n,
        "biological_key_matches_scorer_key": np.array_equal(
            keys, prediction.get("biological_scoring_key")
        ),
        "candidate_id_constant": n > 0
        and set(map(str, prediction.get("candidate_id", []))) == {CANDIDATE},
        "registered_status_covered": n > 0
        and set(map(str, prediction.get("registered_status", []))) == {"covered"},
        "component_and_vector_shapes": shapes,
        "expert_disagreement_replays": shapes
        and bool(np.allclose(disagreement, expected_disagreement, atol=1e-7, rtol=0)),
        "gate_alpha_replays": shapes
        and bool(np.allclose(alpha, expected_alpha, atol=1e-7, rtol=0)),
        "alpha_convex": shapes and bool(((0 <= alpha) & (alpha <= 1)).all()),
        "blended_delta_replays": shapes
        and bool(np.allclose(delta, expected_delta, atol=1e-7, rtol=0)),
        "locations_equal_point_mean": shapes
        and bool(np.allclose(locations, point[:, None], atol=1e-7, rtol=0)),
        "mixture_mean_equals_point_mean": shapes
        and bool(np.allclose(mixture_mean, point, atol=1e-7, rtol=0)),
        "weights_sum_to_one": shapes
        and bool(np.allclose(weights.sum(1), 1.0, atol=1e-7, rtol=0)),
        "positive_finite_scales": shapes
        and bool(np.isfinite(scales).all())
        and bool((scales > 0).all()),
        "single_gate_parameter_set": shapes
        and len(set(threshold.tolist())) == 1
        and len(set(alpha_low.tolist())) == 1
        and len(set(alpha_high.tolist())) == 1,
    }
    return {name: bool(value) for name, value in checks.items()}


def _inner_ledger_checks(path: Path, outer_held: str, gate: dict[str, float]) -> dict[str, bool]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    rows = ledger.get("inner_folds", [])
    held = [puzzle for row in rows for puzzle in row.get("held_puzzles", [])]
    train_union = {puzzle for row in rows for puzzle in row.get("train_puzzles", [])}
    row_disjoint = all(
        not (set(row.get("held_puzzles", [])) & set(row.get("train_puzzles", [])))
        for row in rows
    )
    inner_prediction_ok = True
    all_keys: list[str] = []
    checkpoints_exist = True
    for row in rows:
        checkpoints_exist &= Path(row.get("b1_checkpoint", "")).is_file()
        checkpoints_exist &= Path(row.get("meanaligned_checkpoint", "")).is_file()
        prediction_path = Path(row.get("prediction_artifact", ""))
        if not prediction_path.is_file():
            inner_prediction_ok = False
            continue
        with np.load(prediction_path, allow_pickle=True) as handle:
            fields = set(handle.files)
            keys = list(map(str, handle.get("keys", [])))
            inner_prediction_ok &= str(handle.get("schema_version", "")) == INNER_PREDICTION_SCHEMA
            inner_prediction_ok &= {
                "keys",
                "b1_delta_mean",
                "meanaligned_delta_mean",
                "outer_fold",
                "inner_fold",
                "seed",
            }.issubset(fields)
            inner_prediction_ok &= {
                "target",
                "target_error",
                "target_mask",
                "qualified_mask",
                "method",
                "score",
            }.isdisjoint(fields)
            inner_prediction_ok &= len(keys) > 0 and len(keys) == len(set(keys))
            all_keys.extend(keys)
    recorded_gate = ledger.get("gate", {})
    gate_matches = all(
        math.isclose(float(recorded_gate.get(name, float("nan"))), float(gate[name]), rel_tol=0, abs_tol=1e-12)
        for name in ("threshold", "alpha_low", "alpha_high", "quantile")
    )
    checks = {
        "four_inner_folds": len(rows) == 4,
        "inner_fold_ids_exact": sorted(row.get("inner_fold") for row in rows) == list(range(4)),
        "held_train_disjoint_each_fold": row_disjoint,
        "outer_held_never_inner_train_or_held": outer_held not in set(held) | train_union,
        "nineteen_outer_train_puzzles_held_once": len(held) == 19 and len(set(held)) == 19,
        "inner_prediction_key_universe_disjoint": len(all_keys) > 0 and len(all_keys) == len(set(all_keys)),
        "inner_prediction_artifacts_target_blind": inner_prediction_ok,
        "inner_checkpoints_exist": checkpoints_exist,
        "gate_matches_fold_result": gate_matches,
        "target_values_not_stored": ledger.get("target_values_stored") is False,
        "method_not_gate_input": ledger.get("method_used_as_gate_input") is False,
        "hierarchy_weights_sum_one": math.isclose(
            float(ledger.get("coverage", {}).get("hierarchy_weight_sum", float("nan"))),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ),
    }
    return {name: bool(value) for name, value in checks.items()}


def _fold_checks(row: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    candidate = row["candidate"]
    prediction = _prediction_checks(Path(candidate["prediction_artifact"]))
    inner = _inner_ledger_checks(
        Path(candidate["inner_crossfit_ledger"]), row["held_puzzle"], candidate["gate"]
    )
    integrity = _integrity(candidate["score"])
    invariants = {
        "held_target_error_mask_invariance": row.get("invariants", {}).get(
            "held_target_error_mask_invariance"
        )
        is True,
        "inner_crossfit_complete": row.get("invariants", {}).get(
            "inner_crossfit_complete"
        )
        is True,
        "method_not_gate_input": row.get("invariants", {}).get(
            "method_used_as_gate_input"
        )
        is False,
        "residual_did_not_change_point_mean": row.get("invariants", {}).get(
            "residual_changed_point_mean"
        )
        is False,
    }
    if smoke:
        baseline = row.get("baseline", {})
        b1_history = baseline.get("b1_train_loss", [])
        mean_history = baseline.get("meanaligned_train_loss", [])
        invariants["outer_experts_not_reused"] = baseline.get(
            "reused_exact_seed0_outer_expert"
        ) is False
        invariants["outer_expert_epochs_at_most_three"] = (
            0 < len(b1_history) <= 3 and 0 < len(mean_history) <= 3
        )
        invariants["finite_outer_expert_losses"] = bool(
            np.isfinite(np.asarray(b1_history + mean_history, dtype=float)).all()
        )
        invariants["residual_epochs_at_most_three"] = len(
            candidate.get("residual_calibration_loss", [])
        ) <= 3
        invariants["finite_residual_loss"] = bool(
            np.isfinite(
                np.asarray(candidate.get("residual_calibration_loss", []), dtype=float)
            ).all()
        )
    return {
        "prediction": prediction,
        "inner_crossfit": inner,
        "score_integrity": integrity,
        "invariants": invariants,
        "passed": all(prediction.values())
        and all(inner.values())
        and all(integrity.values())
        and all(invariants.values()),
    }


def qualify_smoke(
    folds: list[dict[str, Any]], phase: str = "R3M2"
) -> dict[str, Any]:
    if phase not in {"R3M2", "R3C2"}:
        raise ValueError(f"{phase} is not an engineering-smoke phase")
    fold_ids = sorted(int(row["outer_fold"]) for row in folds)
    if fold_ids != [0, 1] or len(folds) != 2:
        raise ValueError(f"{phase} smoke requires exactly folds 0 and 1")
    if any(int(row.get("seed", -1)) != 0 for row in folds):
        raise ValueError(f"{phase} smoke accepts seed 0 only")
    rows = []
    for row in sorted(folds, key=lambda value: int(value["outer_fold"])):
        checks = _fold_checks(row, smoke=True)
        rows.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "held_puzzle": row["held_puzzle"],
                **checks,
            }
        )
    passed = all(row["passed"] for row in rows)
    pass_status = (
        "R3C2_CORRECTED_REAL_DATA_ENGINEERING_SMOKE_PASS"
        if phase == "R3C2"
        else "R3M2_REAL_DATA_ENGINEERING_SMOKE_PASS"
    )
    return {
        "schema_version": SMOKE_SCHEMA,
        "phase": phase,
        "evidence_status": "ENGINEERING_SMOKE_ONLY",
        "folds": rows,
        "overall_status": pass_status if passed else f"{phase}_FAIL",
        "r3c3_authorized": passed if phase == "R3C2" else False,
        "r3m3_authorized": passed if phase == "R3M2" else False,
        "scientific_interpretation_prohibited": True,
    }


def qualify_screen(folds: list[dict[str, Any]]) -> dict[str, Any]:
    fold_ids = [int(row["outer_fold"]) for row in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate outer fold result")
    if sorted(fold_ids) != list(range(20)) or len(folds) != 20:
        raise ValueError("R3M3 qualification requires exactly folds 0 through 19")
    if any(int(row.get("seed", -1)) != 0 for row in folds):
        raise ValueError("R3M3 accepts seed 0 only")
    delta_effects = []
    crps_effects = []
    baseline_delta = []
    fold_integrity = []
    per_fold = []
    for row in sorted(folds, key=lambda value: int(value["outer_fold"])):
        if row.get("baseline", {}).get("model_id") != BASELINE:
            raise ValueError(f"fold {row['outer_fold']} has wrong baseline")
        if row.get("candidate", {}).get("candidate_id") != CANDIDATE:
            raise ValueError(f"fold {row['outer_fold']} has wrong candidate")
        checks = _fold_checks(row, smoke=False)
        base = row["baseline"]["score"]
        candidate = row["candidate"]["score"]
        delta_gain = float(base["signed_delta_mae"] - candidate["signed_delta_mae"])
        crps_gain = float(base["crps"] - candidate["crps"])
        delta_effects.append(delta_gain)
        crps_effects.append(crps_gain)
        baseline_delta.append(float(base["signed_delta_mae"]))
        baseline_ok = all(_integrity(base).values())
        fold_integrity.append(checks["passed"] and baseline_ok)
        per_fold.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "held_puzzle": row["held_puzzle"],
                "signed_delta_mae_gain_vs_b1": delta_gain,
                "crps_gain_vs_b1": crps_gain,
                "artifact_and_crossfit_pass": checks["passed"],
                "baseline_integrity_pass": baseline_ok,
            }
        )
    mean_delta_gain = float(np.mean(delta_effects))
    relative_delta_gain = mean_delta_gain / float(np.mean(baseline_delta))
    mean_crps_gain = float(np.mean(crps_effects))
    mean_checks = {
        "signed_delta_mae_gain_positive": mean_delta_gain > 0,
        "signed_delta_mae_relative_gain_at_least_1pct": relative_delta_gain >= 0.01,
        "signed_delta_positive_puzzles_at_least_12": int(
            np.sum(np.asarray(delta_effects) > 0)
        )
        >= 12,
        "all_fold_prediction_crossfit_and_score_integrity": all(fold_integrity),
    }
    calibration_checks = {
        "mean_crps_gain_positive": mean_crps_gain > 0,
        "crps_positive_puzzles_at_least_12": int(np.sum(np.asarray(crps_effects) > 0))
        >= 12,
        "signed_delta_positive_puzzles_inherited_at_least_12": int(
            np.sum(np.asarray(delta_effects) > 0)
        )
        >= 12,
        "all_fold_zero_mean_and_target_invariance": all(fold_integrity),
    }
    mean_pass = all(mean_checks.values())
    calibration_pass = all(calibration_checks.values())
    return {
        "schema_version": SCREEN_SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_SCREEN_NOT_CONFIRMATION",
        "mean_gate": {
            "candidate": CANDIDATE,
            "mean_signed_delta_mae_gain_vs_b1": mean_delta_gain,
            "relative_signed_delta_mae_gain": relative_delta_gain,
            "positive_puzzles": int(np.sum(np.asarray(delta_effects) > 0)),
            "checks": mean_checks,
            "status": "MEAN_GATE_PASS" if mean_pass else "MEAN_GATE_FAIL",
        },
        "calibration_gate": {
            "candidate": CANDIDATE,
            "mean_crps_gain_vs_b1": mean_crps_gain,
            "positive_puzzles": int(np.sum(np.asarray(crps_effects) > 0)),
            "checks": calibration_checks,
            "status": "CALIBRATION_GATE_PASS" if calibration_pass else "CALIBRATION_GATE_FAIL",
        },
        "per_fold": per_fold,
        "overall_status": (
            "R3M3_SCREEN_PASS" if mean_pass and calibration_pass else "MODEL_RESCUE_V3_FAIL"
        ),
        "r3m4_authorized": bool(mean_pass and calibration_pass),
        "prohibited_interpretation": [
            "EXTERNAL_REPLICATION",
            "SOTA",
            "MECHANISM",
            "PUBLICATION_PASS",
        ],
    }


def _paired_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    half = float(
        student_t.ppf(0.975, len(array) - 1)
        * array.std(ddof=1)
        / math.sqrt(len(array))
    )
    return {
        "n": len(array),
        "mean": mean,
        "ci95": [mean - half, mean + half],
        "positive_puzzles": int((array > 0).sum()),
        "per_puzzle": array.tolist(),
    }


def _leave_one_positive(values: list[float]) -> bool:
    array = np.asarray(values, dtype=float)
    return all(float(np.delete(array, i).mean()) > 0 for i in range(len(array)))


def _formal_prediction_checks(path: Path, candidate: str, components: int) -> dict[str, bool]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    keys = prediction.get("keys", np.empty(0, dtype=object))
    n = len(keys)
    locations = prediction.get("locations", np.empty((0, 0)))
    scales = prediction.get("scales", np.empty((0, 0)))
    weights = prediction.get("weights", np.empty((0, 0)))
    point = prediction.get("point_mean", np.empty(0))
    seed_point = prediction.get("seed_point_means", np.empty((0, 0)))
    shapes = (
        locations.shape == scales.shape == weights.shape == (n, components)
        and point.shape == (n,)
        and seed_point.shape == (n, 5)
    )
    seed_layout = False
    if shapes and candidate == CANDIDATE:
        seed_layout = all(
            np.allclose(locations[:, 2 * seed], seed_point[:, seed], atol=1e-7, rtol=0)
            and np.allclose(locations[:, 2 * seed + 1], seed_point[:, seed], atol=1e-7, rtol=0)
            and np.allclose(
                weights[:, 2 * seed] + weights[:, 2 * seed + 1], 0.2, atol=1e-7, rtol=0
            )
            for seed in range(5)
        )
    elif shapes and candidate == BASELINE:
        seed_layout = bool(np.allclose(locations, seed_point, atol=1e-7, rtol=0))
        seed_layout &= bool(np.allclose(weights, 0.2, atol=1e-7, rtol=0))
    prohibited = {"target", "target_error", "target_mask", "qualified_mask", "score"}
    checks = {
        "target_fields_absent": prohibited.isdisjoint(prediction),
        "unique_nonempty_keys": n > 0 and len(set(map(str, keys))) == n,
        "candidate_constant": set(map(str, prediction.get("candidate_id", []))) == {candidate},
        "seed_universe_0_to_4": np.array_equal(prediction.get("seed_universe"), np.arange(5)),
        "component_shapes": shapes,
        "positive_finite_scales": shapes and bool(np.isfinite(scales).all()) and bool((scales > 0).all()),
        "weights_sum_one": shapes and bool(np.allclose(weights.sum(1), 1, atol=1e-7, rtol=0)),
        "mixture_mean_is_point": shapes
        and bool(np.allclose(np.sum(weights * locations, axis=1), point, atol=1e-7, rtol=0)),
        "five_seed_layout": bool(seed_layout),
    }
    return {name: bool(value) for name, value in checks.items()}


def qualify_formal(folds: list[dict[str, Any]]) -> dict[str, Any]:
    folds = sorted(folds, key=lambda row: int(row["outer_fold"]))
    if [int(row["outer_fold"]) for row in folds] != list(range(20)):
        raise ValueError("R3M4 requires exactly folds 0 through 19")
    crps_effects: list[float] = []
    delta_effects: list[float] = []
    baseline_crps: list[float] = []
    baseline_delta: list[float] = []
    worsening68: list[float] = []
    worsening95: list[float] = []
    integrity = []
    per_fold = []
    for row in folds:
        baseline = row["baseline"]
        candidate = row["candidate"]
        if baseline.get("model_id") != BASELINE or candidate.get("model_id") != CANDIDATE:
            raise ValueError("formal fold has wrong model identity")
        if baseline.get("seed_universe") != list(range(5)) or candidate.get("seed_universe") != list(range(5)):
            raise ValueError("formal seed universe is not 0..4")
        base_art = _formal_prediction_checks(Path(baseline["prediction_artifact"]), BASELINE, 5)
        cand_art = _formal_prediction_checks(Path(candidate["prediction_artifact"]), CANDIDATE, 10)
        base_score = baseline["score"]
        cand_score = candidate["score"]
        ok = (
            all(base_art.values())
            and all(cand_art.values())
            and all(_integrity(base_score).values())
            and all(_integrity(cand_score).values())
            and row.get("all_seed_target_error_mask_invariance") is True
            and row.get("all_seed_inner_crossfit_complete") is True
        )
        integrity.append(ok)
        crps_gain = float(base_score["crps"] - cand_score["crps"])
        delta_gain = float(base_score["signed_delta_mae"] - cand_score["signed_delta_mae"])
        crps_effects.append(crps_gain)
        delta_effects.append(delta_gain)
        baseline_crps.append(float(base_score["crps"]))
        baseline_delta.append(float(base_score["signed_delta_mae"]))
        worsening68.append(abs(float(cand_score["coverage68"]) - 0.68) - abs(float(base_score["coverage68"]) - 0.68))
        worsening95.append(abs(float(cand_score["coverage95"]) - 0.95) - abs(float(base_score["coverage95"]) - 0.95))
        per_fold.append({"outer_fold": int(row["outer_fold"]), "held_puzzle": row["held_puzzle"], "crps_gain_vs_b1": crps_gain, "signed_delta_mae_gain_vs_b1": delta_gain, "integrity": ok})
    crps = _paired_ci(crps_effects)
    delta = _paired_ci(delta_effects)
    mean_base_crps = float(np.mean(baseline_crps))
    mean_base_delta = float(np.mean(baseline_delta))
    required_crps = max(0.003, 0.02 * mean_base_crps)
    relative_delta = delta["mean"] / mean_base_delta
    normalized = 0.5 * np.asarray(crps_effects) / mean_base_crps
    normalized += 0.5 * np.asarray(delta_effects) / mean_base_delta
    influence = float(np.max(np.abs(normalized)) / np.sum(np.abs(normalized)))
    worsening68_pp = 100 * float(np.mean(worsening68))
    worsening95_pp = 100 * float(np.mean(worsening95))
    checks = {
        "twenty_outer_folds_complete": True,
        "formal_artifact_crossfit_and_score_integrity": all(integrity),
        "crps_ci95_lower_positive": crps["ci95"][0] > 0,
        "crps_gain_at_least_max_0_003_or_2pct": crps["mean"] >= required_crps,
        "signed_delta_mae_ci95_lower_positive": delta["ci95"][0] > 0,
        "signed_delta_mae_relative_gain_at_least_1pct": relative_delta >= 0.01,
        "crps_positive_puzzles_at_least_14": crps["positive_puzzles"] >= 14,
        "signed_delta_mae_positive_puzzles_at_least_12": delta["positive_puzzles"] >= 12,
        "crps_leave_one_puzzle_effect_positive": _leave_one_positive(crps_effects),
        "signed_delta_leave_one_puzzle_effect_positive": _leave_one_positive(delta_effects),
        "max_single_puzzle_effect_fraction_at_most_0_25": influence <= 0.25,
        "coverage68_absolute_error_worsening_at_most_2pp": worsening68_pp <= 2.0,
        "coverage95_absolute_error_worsening_at_most_2pp": worsening95_pp <= 2.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "schema_version": FORMAL_SCHEMA,
        "evidence_status": "POST_HOC_DEVELOPMENT_FORMAL_NOT_EXTERNAL_CONFIRMATION",
        "crps_effect": crps,
        "signed_delta_mae_effect": delta,
        "required_crps_gain": required_crps,
        "relative_signed_delta_mae_gain": relative_delta,
        "max_single_puzzle_effect_fraction": influence,
        "coverage68_absolute_error_worsening_pp": worsening68_pp,
        "coverage95_absolute_error_worsening_pp": worsening95_pp,
        "checks": checks,
        "per_fold": per_fold,
        "overall_status": "R2M4_POST_HOC_DEVELOPMENT_PASS" if passed else "MODEL_RESCUE_V3_FAIL",
        "r3_phase_status": "R3M4_ORIGINAL_R2M4_GATE_PASS" if passed else "R3M4_FAIL",
        "model_qualification": "POST_HOC_DEVELOPMENT_PASS" if passed else "MODEL_RESCUE_V3_FAIL",
        "prohibited_interpretation": ["EXTERNAL_REPLICATION", "SOTA", "MECHANISM", "PUBLICATION_PASS"],
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ReactFlow-Delta Model Rescue v3 qualification",
        "",
        f"Overall status: `{result['overall_status']}`.",
        f"Evidence status: `{result['evidence_status']}`.",
        "",
    ]
    if result["schema_version"] == SCREEN_SCHEMA:
        lines.extend(
            [
                f"- signed-delta MAE gain: {result['mean_gate']['mean_signed_delta_mae_gain_vs_b1']:+.8f} ({result['mean_gate']['relative_signed_delta_mae_gain']:.3%}), positive puzzles {result['mean_gate']['positive_puzzles']}/20, `{result['mean_gate']['status']}`.",
                f"- CRPS gain: {result['calibration_gate']['mean_crps_gain_vs_b1']:+.8f}, positive puzzles {result['calibration_gate']['positive_puzzles']}/20, `{result['calibration_gate']['status']}`.",
            ]
        )
    elif result["schema_version"] == FORMAL_SCHEMA:
        lines.extend(
            [
                f"- CRPS gain {result['crps_effect']['mean']:+.8f}, CI {result['crps_effect']['ci95']}.",
                f"- signed-delta MAE gain {result['signed_delta_mae_effect']['mean']:+.8f}, CI {result['signed_delta_mae_effect']['ci95']}.",
            ]
        )
    lines.extend(["", "This result does not establish external replication, SOTA, mechanism, or publication readiness.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["R3M2", "R3C2", "R3M3", "R3M4"], default="R3M3"
    )
    args = parser.parse_args(argv)
    folds = _load_folds(args.input, args.phase)
    result = (
        qualify_smoke(folds, phase=args.phase)
        if args.phase in {"R3M2", "R3C2"}
        else qualify_screen(folds)
        if args.phase == "R3M3"
        else qualify_formal(folds)
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["overall_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
