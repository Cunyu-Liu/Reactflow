#!/usr/bin/env python3
"""Run the pre-registered post-V9 residual-shape diagnostics.

This descriptive analysis cannot alter the terminal V9 verdict or authorize
V9M4. Every inferential unit is one held puzzle and every within-puzzle
statistic uses method -> mutant -> qualified-position balancing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v9 import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v9_post_gate_residual_diagnostic.v1"
MERGED_SCHEMA = "reactflow_delta.model_rescue_v9_merged.v1"


def assert_diagnostic_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V9M6":
        raise RuntimeError("post-V9 diagnostics require terminal V9M6")
    if active.get("training_allowed") is not False:
        raise RuntimeError("post-V9 diagnostics cannot run with training open")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("complete held outcomes are not authorized")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("external outcomes must remain locked")
    if active["gate_state"].get("V9M4") != "PERMANENTLY_NOT_AUTHORIZED":
        raise RuntimeError("V9M4 must remain closed")


def hierarchical_weights(methods: np.ndarray, mutants: np.ndarray) -> np.ndarray:
    """Equal method, mutant-within-method and position-within-mutant weights."""
    methods = np.asarray(methods, dtype=object)
    mutants = np.asarray(mutants, dtype=object)
    if methods.ndim != 1 or mutants.shape != methods.shape or not len(methods):
        raise ValueError("hierarchical weights require aligned nonempty vectors")
    unique_methods = sorted(set(map(str, methods)))
    weights = np.zeros(len(methods), dtype=np.float64)
    for method in unique_methods:
        method_rows = np.flatnonzero(methods == method)
        method_mutants = sorted(set(map(str, mutants[method_rows])))
        for mutant in method_mutants:
            rows = method_rows[mutants[method_rows] == mutant]
            weights[rows] = 1.0 / (
                len(unique_methods) * len(method_mutants) * len(rows)
            )
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("hierarchical diagnostic weights do not sum to one")
    return weights


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1 or not len(values):
        raise ValueError("weighted quantile requires aligned nonempty vectors")
    if not 0.0 <= quantile <= 1.0 or np.any(weights < 0.0):
        raise ValueError("invalid weighted quantile arguments")
    total = float(weights.sum())
    if not total > 0.0:
        raise ValueError("weighted quantile has zero total weight")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / total
    index = min(
        int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1
    )
    return float(values[order[index]])


def weighted_correlation(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> float:
    weights = weights / weights.sum()
    x_centered = x - np.sum(weights * x)
    y_centered = y - np.sum(weights * y)
    covariance = float(np.sum(weights * x_centered * y_centered))
    denominator = math.sqrt(
        float(np.sum(weights * x_centered**2))
        * float(np.sum(weights * y_centered**2))
    )
    return covariance / denominator if denominator > 0.0 else float("nan")


def residual_statistics(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mean = float(np.sum(weights * values))
    q10 = weighted_quantile(values, weights, 0.10)
    median = weighted_quantile(values, weights, 0.50)
    q90 = weighted_quantile(values, weights, 0.90)
    width = q90 - q10
    asymmetry = (q90 + q10 - 2.0 * median) / width if width > 0.0 else 0.0
    return {
        "mean": mean,
        "median": median,
        "mean_minus_median": mean - median,
        "q10": q10,
        "q50": median,
        "q90": q90,
        "normalized_quantile_asymmetry": float(asymmetry),
    }


def puzzle_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        raise ValueError("puzzle summary has no finite values")
    mean = float(finite.mean())
    half = (
        float(
            student_t.ppf(0.975, len(finite) - 1)
            * finite.std(ddof=1)
            / math.sqrt(len(finite))
        )
        if len(finite) > 1
        else 0.0
    )
    direction = 1.0 if mean > 0.0 else -1.0 if mean < 0.0 else 0.0
    count = int(np.sum(finite * direction > 0.0)) if direction else 0
    leave_one_out = (
        [float(np.delete(finite, index).mean()) for index in range(len(finite))]
        if len(finite) > 1
        else [mean]
    )
    return {
        "n_independent_puzzles": int(len(finite)),
        "mean": mean,
        "ci95": [mean - half, mean + half],
        "global_direction": int(direction),
        "puzzles_in_global_direction": count,
        "leave_one_puzzle_out": leave_one_out,
        "leave_one_puzzle_out_same_direction": (
            all(value * direction > 0.0 for value in leave_one_out)
            if direction
            else False
        ),
        "per_puzzle": array.tolist(),
    }


def _distance_band(distance: int) -> str:
    value = abs(int(distance))
    if value == 0:
        return "B_edit_0"
    if value <= 3:
        return "B_near_1_3"
    if value <= 10:
        return "B_mid_4_10"
    if value <= 25:
        return "B_far_11_25"
    return "B_vfar_26plus"


def _load_prediction(path: Path, fold_id: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V9 prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold_id}:
        raise ValueError(f"prediction fold mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate prediction key in {path}")
    return prediction


def _collect_puzzle_rows(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    index = {str(key): row for row, key in enumerate(prediction["keys"])}
    names = (
        "method", "mutant", "region", "distance_band", "target_delta",
        "observed_absolute_delta", "feature41_residual", "meanaligned_residual",
        "feature41_expected_absolute_delta", "meanaligned_expected_absolute_delta",
        "feature41_crps", "meanaligned_crps", "feature41_narrow_weight",
        "meanaligned_narrow_weight", "feature41_narrow_scale",
        "meanaligned_narrow_scale", "feature41_wide_scale",
        "meanaligned_wide_scale",
    )
    output: dict[str, list[Any]] = {name: [] for name in names}
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        positions = np.flatnonzero(construct.wt_observed & np.isfinite(target))
        if not len(positions):
            continue
        keys = [_bio_key(univ, record, int(position)) for position in positions]
        if any(key not in index for key in keys):
            raise ValueError("diagnostic target/prediction key universe is incomplete")
        rows = np.asarray([index[key] for key in keys], dtype=np.int64)
        signed = target[positions] - construct.wt_reactivity[positions]
        f_weights = prediction["feature41_weights"][rows]
        f_locations = prediction["feature41_locations"][rows]
        f_scales = prediction["feature41_scales"][rows]
        m_weights = prediction["meanaligned_weights"][rows]
        m_locations = prediction["meanaligned_locations"][rows]
        m_scales = prediction["meanaligned_scales"][rows]
        f_crps = weighted_gaussian_mixture_crps(
            f_locations, f_scales, f_weights, signed
        )
        m_crps = weighted_gaussian_mixture_crps(
            m_locations, m_scales, m_weights, signed
        )
        mutant = f"{record.construct_id}|{record.mutation_key}"
        for local, position in enumerate(positions):
            output["method"].append(record.method)
            output["mutant"].append(mutant)
            output["region"].append(str(construct.region_map[position]))
            output["distance_band"].append(
                _distance_band(int(position) - record.full_pos)
            )
            output["target_delta"].append(float(signed[local]))
            output["observed_absolute_delta"].append(abs(float(signed[local])))
            output["feature41_residual"].append(
                float(
                    signed[local]
                    - prediction["feature41_delta_mean"][rows[local]]
                )
            )
            output["meanaligned_residual"].append(
                float(
                    signed[local]
                    - prediction["meanaligned_delta_mean"][rows[local]]
                )
            )
            output["feature41_expected_absolute_delta"].append(
                float(prediction["feature41_expected_absolute_delta"][rows[local]])
            )
            output["meanaligned_expected_absolute_delta"].append(
                float(prediction["meanaligned_expected_absolute_delta"][rows[local]])
            )
            output["feature41_crps"].append(float(f_crps[local]))
            output["meanaligned_crps"].append(float(m_crps[local]))
            output["feature41_narrow_weight"].append(float(f_weights[local, 0]))
            output["meanaligned_narrow_weight"].append(float(m_weights[local, 0]))
            output["feature41_narrow_scale"].append(float(f_scales[local, 0]))
            output["meanaligned_narrow_scale"].append(float(m_scales[local, 0]))
            output["feature41_wide_scale"].append(float(f_scales[local, 1]))
            output["meanaligned_wide_scale"].append(float(m_scales[local, 1]))
    object_fields = {"method", "mutant", "region", "distance_band"}
    return {
        name: np.asarray(values, dtype=object if name in object_fields else np.float64)
        for name, values in output.items()
    }


def _subset_statistics(
    puzzles: list[dict[str, np.ndarray]],
    model: str,
    field: str | None = None,
    value: str | None = None,
) -> dict[str, Any]:
    per_puzzle = []
    relations = []
    for rows in puzzles:
        mask = np.ones(len(rows["method"]), dtype=bool)
        if field is not None:
            mask &= rows[field] == value
        if not bool(mask.any()):
            continue
        weights = hierarchical_weights(rows["method"][mask], rows["mutant"][mask])
        residual = rows[f"{model}_residual"][mask].astype(np.float64)
        per_puzzle.append(residual_statistics(residual, weights))
        predicted_abs = rows[f"{model}_expected_absolute_delta"][mask].astype(
            np.float64
        )
        observed_abs = rows["observed_absolute_delta"][mask].astype(np.float64)
        relations.append(
            {
                "predicted_absolute_mean": float(np.sum(weights * predicted_abs)),
                "observed_absolute_mean": float(np.sum(weights * observed_abs)),
                "absolute_mean_bias": float(
                    np.sum(weights * (predicted_abs - observed_abs))
                ),
                "absolute_mae": float(
                    np.sum(weights * np.abs(predicted_abs - observed_abs))
                ),
                "crps": float(
                    np.sum(
                        weights * rows[f"{model}_crps"][mask].astype(np.float64)
                    )
                ),
                "predicted_observed_absolute_correlation": weighted_correlation(
                    predicted_abs, observed_abs, weights
                ),
                "narrow_weight": float(
                    np.sum(
                        weights
                        * rows[f"{model}_narrow_weight"][mask].astype(np.float64)
                    )
                ),
                "narrow_scale": float(
                    np.sum(
                        weights
                        * rows[f"{model}_narrow_scale"][mask].astype(np.float64)
                    )
                ),
                "wide_scale": float(
                    np.sum(
                        weights
                        * rows[f"{model}_wide_scale"][mask].astype(np.float64)
                    )
                ),
            }
        )
    if not per_puzzle:
        raise ValueError("diagnostic stratum has no puzzle rows")
    result = {
        name: puzzle_summary([row[name] for row in per_puzzle])
        for name in per_puzzle[0]
    }
    result["distribution_and_magnitude"] = {
        name: puzzle_summary([row[name] for row in relations])
        for name in relations[0]
    }
    return result


def _asymmetry_criterion(model_result: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for name in ("mean_minus_median", "normalized_quantile_asymmetry"):
        summary = model_result[name]
        lower, upper = summary["ci95"]
        ci_excludes_zero = lower > 0.0 or upper < 0.0
        checks[name] = {
            "ci95_excludes_zero": ci_excludes_zero,
            "puzzles_in_global_direction_ge_14": (
                summary["puzzles_in_global_direction"] >= 14
            ),
            "eligible": (
                ci_excludes_zero
                and summary["puzzles_in_global_direction"] >= 14
            ),
        }
    eligible = any(value["eligible"] for value in checks.values())
    return {
        "median_constrained_asymmetric_amendment_eligible": eligible,
        "qualifying_statistics": [
            name for name, value in checks.items() if value["eligible"]
        ],
        "checks": checks,
    }


def diagnose(merged: dict[str, Any], m2_csv: Path) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V9M2_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("post-V9 diagnostics require the complete V9M2 merge")
    if len(merged.get("folds", [])) != 20:
        raise ValueError("post-V9 diagnostics require exactly 20 folds")
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("post-V9 diagnostics require exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    split_by_fold = {int(row.outer_fold): row for row in split["folds"]}
    merged_by_fold = {int(row["outer_fold"]): row for row in merged["folds"]}
    if sorted(merged_by_fold) != list(range(20)):
        raise ValueError("post-V9 diagnostic fold universe is incomplete")
    puzzles = []
    puzzle_ids = []
    for fold_id in range(20):
        held = split_by_fold[fold_id].held_puzzle
        held_records = [record for record in records if record.puzzle == held]
        prediction = _load_prediction(
            Path(merged_by_fold[fold_id]["prediction_artifact"]), fold_id
        )
        puzzles.append(_collect_puzzle_rows(univ, held_records, prediction))
        puzzle_ids.append(str(held))
    models = {
        model: _subset_statistics(puzzles, model)
        for model in ("feature41", "meanaligned")
    }
    methods = sorted({str(value) for rows in puzzles for value in rows["method"]})
    regions = sorted({str(value) for rows in puzzles for value in rows["region"]})
    bands = [
        "B_edit_0", "B_near_1_3", "B_mid_4_10", "B_far_11_25",
        "B_vfar_26plus",
    ]
    strata = {}
    for field, values in (
        ("method", methods), ("region", regions), ("distance_band", bands)
    ):
        strata[field] = {
            value: {
                model: _subset_statistics(puzzles, model, field, value)
                for model in ("feature41", "meanaligned")
            }
            for value in values
        }
    return {
        "schema_version": SCHEMA,
        "status": "V9_POST_GATE_RESIDUAL_DIAGNOSTIC_COMPLETE",
        "evidence_status": "POST_HOC_DIAGNOSTIC_NOT_V9_GATE_EVIDENCE",
        "puzzle_ids_in_fold_order": puzzle_ids,
        "independent_unit": "HELD_PUZZLE",
        "n_independent_units": 20,
        "within_puzzle_estimand": (
            "EQUAL_METHOD_THEN_MUTANT_THEN_QUALIFIED_POSITION"
        ),
        "models": models,
        "strata": strata,
        "candidate_decision": _asymmetry_criterion(models["meanaligned"]),
        "v9_verdict_changed": False,
        "v9m4_authorized": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_diagnostic_authority(args.repo_root.resolve())
    result = diagnose(
        json.loads(args.merged_json.read_text(encoding="utf-8")), args.m2_csv
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
