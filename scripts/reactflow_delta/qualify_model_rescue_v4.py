#!/usr/bin/env python3
"""Mechanical top-journal development qualification for complete v4 scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


SCHEMA = "reactflow_delta.model_rescue_v4_qualification.v1"
FOUNDATION_SCHEMA = "reactflow_delta.model_rescue_v4_foundation_qualification.v1"
SMOKE_SCHEMA = "reactflow_delta.model_rescue_v4_engineering_smoke_qualification.v1"
BASELINE = "corrected_b1"
PRIMARY = "v4_dual_tower_rnafm"
SCRATCH = "v4_dual_tower_scratch"
FOUNDATION_ONLY = "v4_rnafm_only"
CAPACITY_NULL = "v4_capacity_matched_sequence_null"
PUBLISHED = "task_matched_published_comparator"
METRICS = ("crps", "signed_delta_mae")
INTERNAL_MODELS = {BASELINE, PRIMARY, SCRATCH, FOUNDATION_ONLY, CAPACITY_NULL}


def qualify_foundation_cache(cache_path: Path, manifest_path: Path) -> dict[str, Any]:
    import h5py

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "cache_schema_exact": manifest.get("schema_version")
        == "reactflow_delta.model_rescue_v4_rnafm_cache.v1",
        "outcome_blind_evidence_status": manifest.get("evidence_status")
        == "OUTCOME_BLIND_FROZEN_FOUNDATION_INPUT_ONLY",
        "official_repository_exact": manifest.get("official_repository")
        == "https://github.com/ml4bio/RNA-FM",
        "official_repository_commit_exact": manifest.get("official_repository_commit")
        == "348951516e0963d22bbb33b3c9fc18c89081d38e",
        "official_checkpoint_source_exact": manifest.get("official_checkpoint_source")
        == "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth",
        "explicit_checkpoint_path_recorded": bool(manifest.get("checkpoint_path_used")),
        "foundation_parameter_count_plausible": int(
            manifest.get("foundation_parameter_count", 0)
        )
        > 90_000_000,
        "foundation_fully_frozen": int(
            manifest.get("foundation_trainable_parameter_count", -1)
        )
        == 0,
        "outcome_blind_columns_exact": manifest.get("csv_columns_read")
        == ["id", "puzzle", "method", "sequence"],
        "mutant_outcomes_absent": manifest.get("mutant_outcome_columns_loaded") is False,
        "external_outcomes_absent": manifest.get("external_outcome_accessed") is False,
        "pretraining_overlap_not_overclaimed": manifest.get(
            "exact_openknot_pretraining_overlap"
        )
        == "UNKNOWN_NOT_ASSERTED",
        "representation_layer_exact": int(manifest.get("representation_layer", -1)) == 12,
        "representation_width_exact": int(manifest.get("representation_width", -1))
        == 640,
    }
    with h5py.File(cache_path, "r") as handle:
        row_ids = handle["row_ids"][:]
        lengths = handle["lengths"][:]
        embeddings = handle["embeddings"]
        decoded = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in row_ids
        ]
        expected_rows = int(manifest.get("n_sequences", -1))
        expected_max_length = int(manifest.get("max_sequence_length", -1))
        checks.update(
            {
                "nonempty_unique_row_universe": len(decoded) > 0
                and len(decoded) == len(set(decoded)),
                "row_count_matches_manifest": len(decoded) == expected_rows,
                "length_count_matches_rows": len(lengths) == len(decoded),
                "positive_lengths_within_matrix": len(lengths) > 0
                and bool((lengths > 0).all())
                and int(lengths.max()) == expected_max_length,
                "embedding_shape_complete": embeddings.shape
                == (expected_rows, expected_max_length, 640),
            }
        )
    passed = all(checks.values())
    return {
        "schema_version": FOUNDATION_SCHEMA,
        "evidence_status": "ENGINEERING_FOUNDATION_CACHE_ONLY",
        "checks": checks,
        "overall_status": (
            "V4M1_IMPLEMENTATION_AND_FOUNDATION_CACHE_PASS"
            if passed
            else "MODEL_RESCUE_V4_IMPLEMENTATION_FAIL"
        ),
        "v4m2_authorized": passed,
        "scientific_interpretation_prohibited": True,
        "held_scores_computed": False,
        "external_outcome_accessed": False,
    }


def _smoke_prediction_checks(path: Path, model_id: str) -> dict[str, bool]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    required = {
        "keys",
        "biological_scoring_key",
        "candidate_id",
        "point_mean",
        "locations",
        "scales",
        "weights",
        "registered_status",
    }
    prohibited = {"target", "target_error", "qualified_target_mask", "score"}
    keys = prediction.get("keys", np.empty(0, dtype=object))
    n = len(keys)
    locations = prediction.get("locations", np.empty((0, 0)))
    scales = prediction.get("scales", np.empty((0, 0)))
    weights = prediction.get("weights", np.empty((0, 0)))
    point = prediction.get("point_mean", np.empty(0))
    return {
        "required_fields_present": required <= set(prediction),
        "target_side_fields_absent": prohibited.isdisjoint(prediction),
        "nonempty_unique_keys": n > 0 and len(set(map(str, keys))) == n,
        "biological_keys_identical": np.array_equal(
            keys, prediction.get("biological_scoring_key")
        ),
        "candidate_id_constant": n > 0
        and set(map(str, prediction.get("candidate_id", []))) == {model_id},
        "all_registered_rows_covered": n > 0
        and set(map(str, prediction.get("registered_status", []))) == {"covered"},
        "two_component_shapes": locations.shape == (n, 2)
        and scales.shape == (n, 2)
        and weights.shape == (n, 2),
        "zero_mean_locations": locations.shape == (n, 2)
        and np.array_equal(locations[:, 0], point)
        and np.array_equal(locations[:, 1], point),
        "positive_finite_scales": scales.shape == (n, 2)
        and bool(np.isfinite(scales).all())
        and bool((scales > 0).all()),
        "weights_sum_to_one": weights.shape == (n, 2)
        and bool(np.allclose(weights.sum(-1), 1.0, atol=1e-7, rtol=0)),
    }


def qualify_engineering_smoke(fold_dir: Path) -> dict[str, Any]:
    paths = sorted(fold_dir.glob("v4_fold_result_fold*_seed0.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    fold_ids = [int(row.get("outer_fold", -1)) for row in rows]
    if sorted(fold_ids) != [0, 1] or len(set(fold_ids)) != 2:
        raise ValueError("V4M2 smoke requires exactly unique folds 0 and 1")
    fold_results = []
    for row in sorted(rows, key=lambda value: int(value["outer_fold"])):
        if int(row.get("seed", -1)) != 0:
            raise ValueError("V4M2 smoke is frozen to seed 0")
        if row.get("held_score_computed") is not False:
            raise ValueError("V4M2 smoke cannot contain held scores")
        if row.get("external_outcome_accessed") is not False:
            raise ValueError("V4M2 smoke reports external outcome access")
        models = row.get("models", {})
        if set(models) != INTERNAL_MODELS:
            raise ValueError("V4M2 smoke does not contain the frozen model universe")
        model_checks = {}
        for model_id, value in models.items():
            prediction_path = Path(value.get("prediction_artifact", ""))
            checks = {
                "mean_checkpoint_present": Path(value.get("mean_checkpoint", "")).is_file(),
                "calibration_checkpoint_present": Path(
                    value.get("calibration_checkpoint", "")
                ).is_file(),
                "prediction_artifact_present": prediction_path.is_file(),
                "mean_epochs_exact_three": int(value.get("mean_history_length", -1)) == 3,
                "calibration_epochs_exact_three": int(
                    value.get("calibration_history_length", -1)
                )
                == 3,
                "mean_history_finite": value.get("mean_history_finite") is True,
                "calibration_history_finite": value.get("calibration_history_finite")
                is True,
                "positive_mean_parameter_count": int(
                    value.get("trainable_mean_parameters", 0)
                )
                > 0,
            }
            if prediction_path.is_file():
                checks.update(_smoke_prediction_checks(prediction_path, model_id))
            model_checks[model_id] = checks
        primary_count = int(models[PRIMARY]["trainable_mean_parameters"])
        null_count = int(models[CAPACITY_NULL]["trainable_mean_parameters"])
        capacity_ratio = abs(primary_count - null_count) / primary_count
        fold_checks = {
            "held_target_error_mask_invariance": row.get(
                "held_target_error_mask_invariance"
            )
            is True,
            "primary_parameter_range_35m_to_45m": 35_000_000
            <= primary_count
            <= 45_000_000,
            "capacity_null_within_five_percent": capacity_ratio <= 0.05,
        }
        passed = all(fold_checks.values()) and all(
            all(checks.values()) for checks in model_checks.values()
        )
        fold_results.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "model_checks": model_checks,
                "fold_checks": fold_checks,
                "primary_parameter_count": primary_count,
                "capacity_null_parameter_count": null_count,
                "capacity_relative_difference": capacity_ratio,
                "status": "V4M2_FOLD_PASS" if passed else "V4M2_FOLD_FAIL",
            }
        )
    passed = all(row["status"] == "V4M2_FOLD_PASS" for row in fold_results)
    return {
        "schema_version": SMOKE_SCHEMA,
        "evidence_status": "ENGINEERING_SMOKE_ONLY",
        "folds": fold_results,
        "overall_status": (
            "V4M2_REAL_DATA_ENGINEERING_SMOKE_PASS"
            if passed
            else "MODEL_RESCUE_V4_ENGINEERING_FAIL"
        ),
        "v4m3_authorized": passed,
        "scientific_interpretation_prohibited": True,
        "held_scores_computed": False,
        "external_outcome_accessed": False,
    }


def paired_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if len(array) != 20 or not np.isfinite(array).all():
        raise ValueError("paired development effect requires 20 finite puzzle values")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    half = float(student_t.ppf(0.975, len(array) - 1) * standard_error)
    leave_one = [float(np.delete(array, index).mean()) for index in range(len(array))]
    total = abs(float(array.sum()))
    max_fraction = float(np.max(np.abs(array)) / total) if total > 0 else float("inf")
    return {
        "mean": mean,
        "ci95": [mean - half, mean + half],
        "positive_puzzles": int((array > 0).sum()),
        "leave_one_puzzle_means": leave_one,
        "leave_one_puzzle_all_positive": all(value > 0 for value in leave_one),
        "max_single_puzzle_effect_fraction": max_fraction,
        "per_puzzle": array.tolist(),
    }


def _effects(
    scores: dict[str, list[dict[str, float]]], comparator: str, candidate: str, metric: str
) -> list[float]:
    return [
        float(base[metric]) - float(model[metric])
        for base, model in zip(scores[comparator], scores[candidate])
    ]


def _integrity(scores: list[dict[str, float]]) -> bool:
    return all(
        float(row.get("registered_prediction_coverage", float("nan"))) == 1.0
        and float(row.get("failure_rate", float("nan"))) == 0.0
        and int(row.get("n_unexpected_prediction_keys", -1)) == 0
        for row in scores
    )


def qualify_complete_scores(
    scores: dict[str, list[dict[str, float]]], *, phase: str
) -> dict[str, Any]:
    required = {BASELINE, PRIMARY, SCRATCH, FOUNDATION_ONLY, CAPACITY_NULL}
    if not required <= set(scores):
        raise ValueError(f"qualification scores missing models {sorted(required - set(scores))}")
    if any(len(scores[model]) != 20 for model in required):
        raise ValueError("every frozen v4 model requires 20 puzzle scores")
    base_means = {
        metric: float(np.mean([row[metric] for row in scores[BASELINE]]))
        for metric in METRICS
    }
    versus_b1 = {
        metric: paired_summary(_effects(scores, BASELINE, PRIMARY, metric))
        for metric in METRICS
    }
    for metric in METRICS:
        versus_b1[metric]["relative_gain"] = (
            versus_b1[metric]["mean"] / base_means[metric]
        )
    versus_null = {
        metric: paired_summary(_effects(scores, CAPACITY_NULL, PRIMARY, metric))
        for metric in METRICS
    }
    versus_foundation = {
        metric: paired_summary(_effects(scores, FOUNDATION_ONLY, PRIMARY, metric))
        for metric in METRICS
    }
    published_available = PUBLISHED in scores and len(scores[PUBLISHED]) == 20
    versus_published = (
        {
            metric: paired_summary(_effects(scores, PUBLISHED, PRIMARY, metric))
            for metric in METRICS
        }
        if published_available
        else None
    )

    metric_gate = all(
        versus_b1[metric]["relative_gain"] >= 0.05
        and versus_b1[metric]["ci95"][0] > 0.0
        and versus_b1[metric]["positive_puzzles"] >= 16
        and versus_b1[metric]["leave_one_puzzle_all_positive"]
        and versus_b1[metric]["max_single_puzzle_effect_fraction"] <= 0.20
        for metric in METRICS
    )
    attribution_gate = all(
        versus_null[metric]["ci95"][0] > 0.0
        and versus_foundation[metric]["ci95"][0] > 0.0
        for metric in METRICS
    )
    published_gate = published_available and all(
        versus_published[metric]["ci95"][0] > 0.0 for metric in METRICS
    )
    integrity_gate = all(_integrity(scores[model]) for model in required)
    coverage_gate = True
    coverage_details = {}
    for nominal, field in ((0.68, "coverage68"), (0.95, "coverage95")):
        baseline_error = abs(
            float(np.mean([row[field] for row in scores[BASELINE]])) - nominal
        )
        primary_error = abs(
            float(np.mean([row[field] for row in scores[PRIMARY]])) - nominal
        )
        worsening = primary_error - baseline_error
        coverage_details[field] = {
            "baseline_absolute_error": baseline_error,
            "primary_absolute_error": primary_error,
            "worsening": worsening,
            "max_allowed_worsening": 0.01,
        }
        coverage_gate = coverage_gate and worsening <= 0.01

    overall_pass = bool(
        metric_gate
        and attribution_gate
        and published_gate
        and integrity_gate
        and coverage_gate
    )
    status = (
        "V4M3_SCREEN_PASS"
        if overall_pass and phase == "V4M3"
        else "V4M4_HIGH_EFFECT_POST_HOC_DEVELOPMENT_PASS"
        if overall_pass and phase == "V4M4"
        else "MODEL_RESCUE_V4_FAIL"
    )
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "evidence_status": "DEVELOPMENT_CONSUMED",
        "overall_status": status,
        "v4m4_authorized": status == "V4M3_SCREEN_PASS",
        "versus_corrected_b1": versus_b1,
        "versus_capacity_matched_null": versus_null,
        "versus_rnafm_only": versus_foundation,
        "versus_task_matched_published": versus_published,
        "gates": {
            "dual_metric_top_journal_development": metric_gate,
            "architecture_attribution": attribution_gate,
            "task_matched_published_comparator": published_gate,
            "prediction_integrity": integrity_gate,
            "coverage_calibration": coverage_gate,
        },
        "coverage": coverage_details,
        "scratch_ablation_reported": len(scores[SCRATCH]) == 20,
        "external_replication": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "mechanism": "NOT_ESTABLISHED",
        "publication_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", type=Path)
    parser.add_argument("--fold-dir", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["V4M1", "V4M2", "V4M3", "V4M4"], required=True
    )
    args = parser.parse_args(argv)
    if args.phase == "V4M1":
        if args.cache_path is None or args.cache_manifest is None:
            raise ValueError(
                "V4M1 qualification requires --cache-path and --cache-manifest"
            )
        result = qualify_foundation_cache(args.cache_path, args.cache_manifest)
    elif args.phase == "V4M2":
        if args.fold_dir is None:
            raise ValueError("V4M2 qualification requires --fold-dir")
        result = qualify_engineering_smoke(args.fold_dir)
    else:
        if args.score_json is None:
            raise ValueError(f"{args.phase} qualification requires --score-json")
        scores = json.loads(args.score_json.read_text(encoding="utf-8"))["scores"]
        result = qualify_complete_scores(scores, phase=args.phase)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["overall_status"], "result": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
