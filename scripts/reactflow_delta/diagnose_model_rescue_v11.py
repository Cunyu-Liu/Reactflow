#!/usr/bin/env python3
"""Run the predeclared post-V11 diagnostics after complete score access."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v11 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.qualify_model_rescue_v11 import (
    SCHEMA as QUALIFICATION_SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v11 import (
    SCHEMA as SCORE_SCHEMA,
    _load_prediction,
    merged_integrity_pass,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v11_post_screen_diagnostics.v1"
LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95)


def assert_diagnostic_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V11M3":
        raise RuntimeError("post-V11 diagnostics are closed outside V11M3")
    if active.get("runnable_phases") != ["V11M3"]:
        raise RuntimeError("V11M3 must remain the only runnable phase")
    if active.get("training_allowed") is not False:
        raise RuntimeError("post-V11 diagnostics require training closed")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("post-V11 diagnostics require complete-score authority")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial V11 diagnostics remain prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("post-V11 diagnostics require external outcomes locked")


def method_balanced_weights(
    methods: np.ndarray,
    mutants: np.ndarray,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    """Return method→mutant→position weights for one held puzzle."""

    methods = np.asarray(methods, dtype=object)
    mutants = np.asarray(mutants, dtype=object)
    if methods.shape != mutants.shape or methods.ndim != 1:
        raise ValueError("method and mutant labels must be aligned vectors")
    mask = (
        np.ones(len(methods), dtype=bool)
        if selected is None
        else np.asarray(selected, dtype=bool)
    )
    if mask.shape != methods.shape or not mask.any():
        raise ValueError("a diagnostic bin must contain at least one position")
    output = np.zeros(len(methods), dtype=np.float64)
    method_values = sorted(set(map(str, methods[mask])))
    for method in method_values:
        method_mask = mask & (methods == method)
        mutant_values = sorted(set(map(str, mutants[method_mask])))
        for mutant in mutant_values:
            position_mask = method_mask & (mutants == mutant)
            output[position_mask] = (
                1.0
                / len(method_values)
                / len(mutant_values)
                / int(position_mask.sum())
            )
    if not np.isclose(output.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("method-balanced diagnostic weights do not sum to one")
    return output


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    x_centered = x - np.sum(weights * x)
    y_centered = y - np.sum(weights * y)
    denominator = math.sqrt(
        float(np.sum(weights * x_centered**2))
        * float(np.sum(weights * y_centered**2))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(weights * x_centered * y_centered) / denominator)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    index = int(np.searchsorted(np.cumsum(ordered_weights), 0.5, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def directional_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) != 20:
        return {
            "n_puzzles": int(len(array)),
            "confirmatory": False,
            "reason": "REQUIRES_ALL_TWENTY_PUZZLES",
        }
    mean = float(array.mean())
    half = float(student_t.ppf(0.975, 19) * array.std(ddof=1) / math.sqrt(20))
    positive = int((array > 0.0).sum())
    negative = int((array < 0.0).sum())
    return {
        "n_puzzles": 20,
        "confirmatory": True,
        "mean": mean,
        "ci95": [mean - half, mean + half],
        "positive_puzzles": positive,
        "negative_puzzles": negative,
        "stable_nonzero_direction": (
            (mean - half > 0.0 or mean + half < 0.0)
            and max(positive, negative) >= 14
        ),
        "per_puzzle": array.tolist(),
    }


def convergence_diagnostic(folds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in sorted(folds, key=lambda item: int(item["outer_fold"])):
        histories = row.get("training_histories", {})
        current = {"outer_fold": int(row["outer_fold"])}
        for name in ("anchored_point", "unanchored_point"):
            history = np.asarray(histories.get(name, []), dtype=np.float64)
            if history.shape != (40,) or not np.isfinite(history).all():
                raise ValueError(f"V11 {name} history must contain forty finite epochs")
            prior = float(history[30:35].mean())
            recent = float(history[35:40].mean())
            current[f"{name}_last_window_relative_decrease"] = (
                (prior - recent) / prior
            )
        rows.append(current)
    if [row["outer_fold"] for row in rows] != list(range(20)):
        raise ValueError("V11 convergence diagnostic requires folds0-19")
    anchored = np.asarray(
        [row["anchored_point_last_window_relative_decrease"] for row in rows]
    )
    return {
        "folds": rows,
        "anchored_median_last_window_relative_decrease": float(np.median(anchored)),
        "anchored_folds_ge_1pct": int((anchored >= 0.01).sum()),
        "schedule_visibly_unfinished": (
            float(np.median(anchored)) >= 0.01 and int((anchored >= 0.01).sum()) >= 14
        ),
    }


def _observations(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    index = {str(key): row for row, key in enumerate(prediction["keys"])}
    output: dict[str, list[Any]] = {
        name: []
        for name in (
            "method",
            "mutant",
            "region",
            "distance",
            "target",
            "feature41_point",
            "anchored_point",
            "unanchored_point",
        )
    }
    for name in ("feature41", "anchored", "unanchored"):
        for suffix in ("weights", "locations", "scales"):
            output[f"{name}_{suffix}"] = []
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        positions = np.flatnonzero(construct.wt_observed & np.isfinite(target))
        mutant = f"{record.method}|{record.mutation_key}"
        for position in positions:
            key = _bio_key(univ, record, int(position))
            if key not in index:
                raise ValueError("V11 diagnostic prediction universe is incomplete")
            row = index[key]
            output["method"].append(record.method)
            output["mutant"].append(mutant)
            output["region"].append(str(construct.region_map[position]))
            output["distance"].append(abs(int(position) - int(record.full_pos)))
            output["target"].append(
                float(target[position] - construct.wt_reactivity[position])
            )
            for name in ("feature41", "anchored", "unanchored"):
                output[f"{name}_point"].append(float(prediction[f"{name}_point"][row]))
                for suffix in ("weights", "locations", "scales"):
                    output[f"{name}_{suffix}"].append(
                        np.asarray(prediction[f"{name}_{suffix}"][row], dtype=np.float64)
                    )
    result = {}
    for name, values in output.items():
        if name in ("method", "mutant", "region"):
            result[name] = np.asarray(values, dtype=object)
        elif name.endswith(("_weights", "_locations", "_scales")):
            result[name] = np.stack(values)
        else:
            result[name] = np.asarray(values, dtype=np.float64)
    return result


def _distribution_crps(observations: dict[str, np.ndarray], name: str) -> np.ndarray:
    return weighted_gaussian_mixture_crps(
        observations[f"{name}_locations"],
        observations[f"{name}_scales"],
        observations[f"{name}_weights"],
        observations["target"],
    )


def point_residual_diagnostic(observations: dict[str, np.ndarray]) -> dict[str, Any]:
    weights = method_balanced_weights(
        observations["method"], observations["mutant"]
    )
    residual = observations["target"] - observations["feature41_point"]
    anchored = observations["anchored_point"] - observations["feature41_point"]
    unanchored = observations["unanchored_point"]
    target_amplitude = weighted_median(np.abs(residual), weights)
    return {
        "zero_residual_mae": float(np.sum(weights * np.abs(residual))),
        "anchored_residual_mae": float(np.sum(weights * np.abs(residual - anchored))),
        "unanchored_residual_mae": float(
            np.sum(weights * np.abs(residual - unanchored))
        ),
        "anchored_residual_association": weighted_correlation(
            residual, anchored, weights
        ),
        "unanchored_residual_association": weighted_correlation(
            residual, unanchored, weights
        ),
        "anchored_amplitude_ratio": (
            weighted_median(np.abs(anchored), weights) / target_amplitude
            if target_amplitude > 0.0
            else float("nan")
        ),
        "unanchored_amplitude_ratio": (
            weighted_median(np.abs(unanchored), weights) / target_amplitude
            if target_amplitude > 0.0
            else float("nan")
        ),
    }


def _regime_masks(observations: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    magnitude = np.abs(observations["feature41_point"])
    distance = observations["distance"]
    return {
        "feature41_absolute": {
            "0_0.05": (magnitude < 0.05),
            "0.05_0.10": (magnitude >= 0.05) & (magnitude < 0.10),
            "0.10_0.20": (magnitude >= 0.10) & (magnitude < 0.20),
            "0.20_inf": magnitude >= 0.20,
        },
        "absolute_distance": {
            "0": distance == 0,
            "1_5": (distance >= 1) & (distance <= 5),
            "6_20": (distance >= 6) & (distance <= 20),
            "gt20": distance > 20,
        },
        "region": {
            "design": observations["region"] == "design_region",
            "other": observations["region"] != "design_region",
        },
        "method": {
            method: observations["method"] == method
            for method in sorted(set(map(str, observations["method"])))
        },
    }


def regime_diagnostic(observations: dict[str, np.ndarray]) -> dict[str, Any]:
    target = observations["target"]
    feature41_error = np.abs(target - observations["feature41_point"])
    anchored_error = np.abs(target - observations["anchored_point"])
    feature41_crps = _distribution_crps(observations, "feature41")
    anchored_crps = _distribution_crps(observations, "anchored")
    result = {}
    for dimension, masks in _regime_masks(observations).items():
        result[dimension] = {}
        for label, selected in masks.items():
            if not selected.any():
                result[dimension][label] = {"available": False}
                continue
            weights = method_balanced_weights(
                observations["method"], observations["mutant"], selected
            )
            result[dimension][label] = {
                "available": True,
                "n_positions": int(selected.sum()),
                "n_methods": len(set(map(str, observations["method"][selected]))),
                "feature41_residual_mae": float(np.sum(weights * feature41_error)),
                "anchored_residual_mae": float(np.sum(weights * anchored_error)),
                "feature41_crps": float(np.sum(weights * feature41_crps)),
                "anchored_crps": float(np.sum(weights * anchored_crps)),
            }
    return result


def distribution_diagnostic(observations: dict[str, np.ndarray]) -> dict[str, Any]:
    balanced = method_balanced_weights(observations["method"], observations["mutant"])
    result = {}
    for name in ("feature41", "anchored", "unanchored"):
        mixture_weights = observations[f"{name}_weights"]
        locations = observations[f"{name}_locations"]
        scales = observations[f"{name}_scales"]
        target = observations["target"]
        component_cdf = ndtr((target[:, None] - locations) / scales)
        cdf = np.sum(mixture_weights * component_cdf, axis=1)
        current: dict[str, Any] = {
            "pit_mean": float(np.sum(balanced * cdf)),
            "pit_variance": float(
                np.sum(balanced * (cdf - np.sum(balanced * cdf)) ** 2)
            ),
        }
        for level in LEVELS:
            lower = (1.0 - level) / 2.0
            upper = 1.0 - lower
            tag = int(round(level * 100))
            covered = (cdf >= lower) & (cdf <= upper)
            current[f"coverage{tag}"] = float(np.sum(balanced * covered))
            if level == 0.90:
                current["lower_tail_miss90"] = float(np.sum(balanced * (cdf < lower)))
                current["upper_tail_miss90"] = float(np.sum(balanced * (cdf > upper)))
        expected_scale = np.sum(mixture_weights * scales, axis=1)
        absolute_error = np.abs(target - observations[f"{name}_point"])
        current["scale_absolute_error_association"] = weighted_correlation(
            expected_scale, absolute_error, balanced
        )
        point_component_cdf = ndtr(
            (observations[f"{name}_point"][:, None] - locations) / scales
        )
        current["median_allocation_absolute_error_association"] = weighted_correlation(
            point_component_cdf[:, 0], absolute_error, balanced
        )
        result[name] = current
    return result


def _paired_summary(rows: list[dict[str, Any]], baseline: str, candidate: str) -> dict[str, Any]:
    effects = [float(row[baseline]) - float(row[candidate]) for row in rows]
    summary = directional_summary(effects)
    if summary.get("confirmatory"):
        baseline_mean = float(np.mean([row[baseline] for row in rows]))
        summary.update(
            {
                "baseline_mean": baseline_mean,
                "candidate_mean": float(np.mean([row[candidate] for row in rows])),
                "relative_gain": summary["mean"] / baseline_mean,
            }
        )
    return summary


def summarize_regimes(folds: list[dict[str, Any]]) -> dict[str, Any]:
    universe: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for fold in folds:
        for dimension, bins in fold["regimes"].items():
            for label, row in bins.items():
                if row.get("available") is True:
                    universe.setdefault((dimension, label), []).append(row)
    result: dict[str, Any] = {}
    for (dimension, label), rows in sorted(universe.items()):
        result.setdefault(dimension, {})[label] = {
            "n_puzzles": len(rows),
            "residual_mae": _paired_summary(
                rows, "feature41_residual_mae", "anchored_residual_mae"
            ),
            "task_crps": _paired_summary(rows, "feature41_crps", "anchored_crps"),
        }
    return result


def summarize_distribution(folds: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in ("feature41", "anchored", "unanchored"):
        rows = [fold["distribution"][name] for fold in folds]
        current = {
            "pit_mean_minus_half": directional_summary(
                [row["pit_mean"] - 0.5 for row in rows]
            ),
            "pit_variance_minus_uniform": directional_summary(
                [row["pit_variance"] - 1.0 / 12.0 for row in rows]
            ),
            "lower_minus_upper_tail_miss90": directional_summary(
                [row["lower_tail_miss90"] - row["upper_tail_miss90"] for row in rows]
            ),
            "scale_absolute_error_association": directional_summary(
                [row["scale_absolute_error_association"] for row in rows]
            ),
            "median_allocation_absolute_error_association": directional_summary(
                [
                    row["median_allocation_absolute_error_association"]
                    for row in rows
                ]
            ),
        }
        for level in LEVELS:
            tag = int(round(level * 100))
            current[f"coverage{tag}_minus_nominal"] = directional_summary(
                [row[f"coverage{tag}"] - level for row in rows]
            )
        result[name] = current
    anchored = result["anchored"]
    stable_tail_or_asymmetry = any(
        anchored[name].get("stable_nonzero_direction") is True
        for name in (
            "pit_mean_minus_half",
            "lower_minus_upper_tail_miss90",
        )
    )
    return {
        "models": result,
        "anchored_stable_asymmetry_or_tail_signal": stable_tail_or_asymmetry,
    }


def diagnose_complete(
    merged: dict[str, Any],
    scores: dict[str, Any],
    qualification: dict[str, Any],
    m2_csv: Path,
) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V11M3_COMPLETE_UNSCORED_MERGE_PASS"
    ) or not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("post-V11 diagnostics require the qualified complete merge")
    if scores.get("schema_version") != SCORE_SCHEMA or scores.get("status") != (
        "V11M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("post-V11 diagnostics require the complete score artifact")
    if qualification.get("schema_version") != QUALIFICATION_SCHEMA or qualification.get(
        "status"
    ) not in ("V11M3_TOP_JOURNAL_SCREEN_PASS", "V11M3_TOP_JOURNAL_SCREEN_FAIL"):
        raise ValueError("post-V11 diagnostics require the mechanical screen verdict")
    merged_rows = {int(row["outer_fold"]): row for row in merged["folds"]}
    score_rows = {int(row["outer_fold"]): row for row in scores["scores"]}
    if sorted(merged_rows) != list(range(20)) or sorted(score_rows) != list(range(20)):
        raise ValueError("post-V11 diagnostics require folds0-19")

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("post-V11 diagnostics require exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    folds = []
    for fold_id in range(20):
        fold = fold_map[fold_id]
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        prediction = _load_prediction(
            Path(merged_rows[fold_id]["prediction_artifact"]), fold_id
        )
        observations = _observations(univ, held_records, prediction)
        residual = point_residual_diagnostic(observations)
        score = score_rows[fold_id]
        if not np.isclose(
            residual["zero_residual_mae"],
            float(score["feature41_signed_delta_mae"]),
            atol=1e-10,
            rtol=0.0,
        ) or not np.isclose(
            residual["anchored_residual_mae"],
            float(score["anchored_signed_delta_mae"]),
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError("post-V11 diagnostic estimator does not replay headline MAE")
        folds.append(
            {
                "outer_fold": fold_id,
                "held_puzzle": str(fold.held_puzzle),
                "residual": residual,
                "regimes": regime_diagnostic(observations),
                "distribution": distribution_diagnostic(observations),
            }
        )

    zero_vs_anchored = _paired_summary(
        [fold["residual"] for fold in folds],
        "zero_residual_mae",
        "anchored_residual_mae",
    )
    null_vs_anchored = _paired_summary(
        [fold["residual"] for fold in folds],
        "unanchored_residual_mae",
        "anchored_residual_mae",
    )
    anchor_signal = (
        zero_vs_anchored.get("ci95", [float("-inf")])[0] > 0.0
        and null_vs_anchored.get("ci95", [float("-inf")])[0] > 0.0
        and zero_vs_anchored.get("positive_puzzles", 0) >= 14
        and null_vs_anchored.get("positive_puzzles", 0) >= 14
    )
    convergence = convergence_diagnostic(list(merged_rows.values()))
    distribution = summarize_distribution(folds)
    return {
        "schema_version": SCHEMA,
        "phase": "POST_V11M3_DIAGNOSTIC_ONLY",
        "status": "POST_V11_DIAGNOSTICS_COMPLETE",
        "screen_status": qualification["status"],
        "folds": folds,
        "point_residual_summary": {
            "zero_vs_anchored": zero_vs_anchored,
            "unanchored_vs_anchored": null_vs_anchored,
            "anchor_specific_residual_signal": anchor_signal,
            "anchored_association": directional_summary(
                [fold["residual"]["anchored_residual_association"] for fold in folds]
            ),
            "unanchored_association": directional_summary(
                [fold["residual"]["unanchored_residual_association"] for fold in folds]
            ),
        },
        "convergence": convergence,
        "regime_summary": summarize_regimes(folds),
        "distribution_summary": distribution,
        "decision_support": {
            "open_v11m4_only_if_exact_screen_pass": qualification["status"]
            == "V11M3_TOP_JOURNAL_SCREEN_PASS",
            "anchor_specific_residual_signal": anchor_signal,
            "convergence_only_route_supported": convergence[
                "schedule_visibly_unfinished"
            ],
            "distribution_only_route_has_stable_diagnostic": distribution[
                "anchored_stable_asymmetry_or_tail_signal"
            ],
            "new_model_authorized": False,
        },
        "independent_unit": "HELD_PUZZLE_N20",
        "partial_scores_inspected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--qualification-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_diagnostic_authority(args.repo_root.resolve())
    result = diagnose_complete(
        json.loads(args.merged_json.read_text(encoding="utf-8")),
        json.loads(args.score_json.read_text(encoding="utf-8")),
        json.loads(args.qualification_json.read_text(encoding="utf-8")),
        args.m2_csv,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
