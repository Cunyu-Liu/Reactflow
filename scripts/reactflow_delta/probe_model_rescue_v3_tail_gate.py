#!/usr/bin/env python3
"""Probe legal outcome-blind tail gates between B1 and MeanAligned experts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.probe_model_rescue_v3_blend import (
    MutantExample,
    load_examples,
    mixture_mean,
    shifted_residual_locations,
)


SCHEMA = "reactflow_delta.model_rescue_v3_tail_gate_probe.v1"
QUANTILES = (0.50, 0.65, 0.80, 0.90, 0.95)
FOUR_BIN_QUANTILES = (0.70, 0.85, 0.95)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or not 0.0 <= q <= 1.0:
        raise ValueError("weighted_quantile received an empty input or invalid q")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0:
        raise ValueError("weighted_quantile requires positive total weight")
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def fit_convex_l1_alpha(
    target: np.ndarray,
    b1: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Exact weighted-L1 convex blend via the weighted median of breakpoints."""
    target = np.asarray(target, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    mean = np.asarray(mean, dtype=float)
    weights = np.asarray(weights, dtype=float)
    difference = mean - b1
    informative = (np.abs(difference) > 1e-12) & (weights > 0)
    if not bool(informative.any()):
        return 1.0
    breakpoints = (target[informative] - b1[informative]) / difference[informative]
    effective_weights = weights[informative] * np.abs(difference[informative])
    alpha = weighted_quantile(breakpoints, effective_weights, 0.5)
    return float(np.clip(alpha, 0.0, 1.0))


def _objective(
    target: np.ndarray,
    b1: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
    alpha: np.ndarray,
) -> float:
    prediction = b1 + alpha * (mean - b1)
    return float(np.sum(weights * np.abs(target - prediction)) / np.sum(weights))


def _flat_train_rows(
    examples: list[MutantExample], held_puzzle: str
) -> dict[str, np.ndarray]:
    mutant_counts = Counter(
        (example.puzzle, example.method)
        for example in examples
        if example.puzzle != held_puzzle
    )
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for example in examples:
        if example.puzzle == held_puzzle:
            continue
        n_mutants = mutant_counts[(example.puzzle, example.method)]
        n_positions = len(example.true_delta)
        rows["target"].append(example.true_delta)
        rows["b1"].append(example.b1_delta)
        rows["mean"].append(example.mean_delta)
        rows["weight"].append(
            np.full(n_positions, 1.0 / (n_mutants * n_positions), dtype=float)
        )
    return {name: np.concatenate(parts) for name, parts in rows.items()}


def fit_global_gate(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    alpha = fit_convex_l1_alpha(
        rows["target"], rows["b1"], rows["mean"], rows["weight"]
    )
    return {"family": "global", "alpha": alpha}


def fit_two_bin_gate(
    rows: dict[str, np.ndarray], *, feature: str
) -> dict[str, Any]:
    if feature == "b1_magnitude":
        values = np.abs(rows["b1"])
    elif feature == "expert_disagreement":
        values = np.abs(rows["b1"] - rows["mean"])
    else:
        raise ValueError(f"unknown gate feature: {feature}")
    best: tuple[float, float, dict[str, Any]] | None = None
    for q in QUANTILES:
        threshold = weighted_quantile(values, rows["weight"], q)
        high = values > threshold
        alpha = np.empty(len(values), dtype=float)
        fitted: list[float] = []
        for selected in (~high, high):
            if not bool(selected.any()):
                value = fit_convex_l1_alpha(
                    rows["target"], rows["b1"], rows["mean"], rows["weight"]
                )
            else:
                value = fit_convex_l1_alpha(
                    rows["target"][selected],
                    rows["b1"][selected],
                    rows["mean"][selected],
                    rows["weight"][selected],
                )
            alpha[selected] = value
            fitted.append(value)
        objective = _objective(
            rows["target"], rows["b1"], rows["mean"], rows["weight"], alpha
        )
        spec = {
            "family": "two_bin",
            "feature": feature,
            "quantile": q,
            "threshold": threshold,
            "alpha_low": fitted[0],
            "alpha_high": fitted[1],
        }
        ranking = (objective, abs(q - 0.8), spec)
        if best is None or ranking[:2] < best[:2]:
            best = ranking
    assert best is not None
    return best[2]


def fit_four_bin_gate(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    magnitude = np.abs(rows["b1"])
    disagreement = np.abs(rows["b1"] - rows["mean"])
    best: tuple[float, float, dict[str, Any]] | None = None
    global_alpha = fit_convex_l1_alpha(
        rows["target"], rows["b1"], rows["mean"], rows["weight"]
    )
    for magnitude_q in FOUR_BIN_QUANTILES:
        magnitude_threshold = weighted_quantile(
            magnitude, rows["weight"], magnitude_q
        )
        for disagreement_q in FOUR_BIN_QUANTILES:
            disagreement_threshold = weighted_quantile(
                disagreement, rows["weight"], disagreement_q
            )
            bin_id = 2 * (magnitude > magnitude_threshold).astype(int) + (
                disagreement > disagreement_threshold
            ).astype(int)
            alpha = np.empty(len(magnitude), dtype=float)
            fitted: list[float] = []
            for current_bin in range(4):
                selected = bin_id == current_bin
                if bool(selected.any()):
                    value = fit_convex_l1_alpha(
                        rows["target"][selected],
                        rows["b1"][selected],
                        rows["mean"][selected],
                        rows["weight"][selected],
                    )
                else:
                    value = global_alpha
                alpha[selected] = value
                fitted.append(value)
            objective = _objective(
                rows["target"], rows["b1"], rows["mean"], rows["weight"], alpha
            )
            spec = {
                "family": "four_bin",
                "magnitude_quantile": magnitude_q,
                "magnitude_threshold": magnitude_threshold,
                "disagreement_quantile": disagreement_q,
                "disagreement_threshold": disagreement_threshold,
                "alphas": fitted,
            }
            ranking = (
                objective,
                abs(magnitude_q - 0.85) + abs(disagreement_q - 0.85),
                spec,
            )
            if best is None or ranking[:2] < best[:2]:
                best = ranking
    assert best is not None
    return best[2]


def apply_gate(spec: dict[str, Any], b1: np.ndarray, mean: np.ndarray) -> np.ndarray:
    if spec["family"] == "global":
        return np.full(len(b1), float(spec["alpha"]), dtype=float)
    magnitude = np.abs(b1)
    disagreement = np.abs(b1 - mean)
    if spec["family"] == "two_bin":
        values = magnitude if spec["feature"] == "b1_magnitude" else disagreement
        return np.where(
            values > float(spec["threshold"]),
            float(spec["alpha_high"]),
            float(spec["alpha_low"]),
        )
    if spec["family"] == "four_bin":
        bin_id = 2 * (magnitude > float(spec["magnitude_threshold"])).astype(int) + (
            disagreement > float(spec["disagreement_threshold"])
        ).astype(int)
        return np.asarray(spec["alphas"], dtype=float)[bin_id]
    raise ValueError(f"unknown gate family: {spec['family']}")


def _paired_summary(
    baseline_by_puzzle: dict[str, float], candidate_by_puzzle: dict[str, float]
) -> dict[str, Any]:
    puzzles = sorted(set(baseline_by_puzzle) & set(candidate_by_puzzle))
    gains = {
        puzzle: float(baseline_by_puzzle[puzzle] - candidate_by_puzzle[puzzle])
        for puzzle in puzzles
    }
    baseline_mean = float(np.mean([baseline_by_puzzle[p] for p in puzzles]))
    candidate_mean = float(np.mean([candidate_by_puzzle[p] for p in puzzles]))
    gain = baseline_mean - candidate_mean
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_gain": gain,
        "relative_gain": gain / baseline_mean,
        "positive_puzzles": int(sum(value > 0 for value in gains.values())),
        "per_puzzle_gain": gains,
    }


def evaluate(
    examples: list[MutantExample], specs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    cell_metrics: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in examples:
        alpha = apply_gate(specs[example.puzzle], example.b1_delta, example.mean_delta)
        blend = example.b1_delta + alpha * (example.mean_delta - example.b1_delta)
        locations = shifted_residual_locations(
            example.residual_locations, example.mean_delta, blend
        )
        losses = {
            "b1_mae": np.abs(example.true_delta - example.b1_delta),
            "candidate_mae": np.abs(example.true_delta - blend),
            "b1_crps": weighted_gaussian_mixture_crps(
                example.b1_locations,
                example.b1_scales,
                example.b1_weights,
                example.target,
            ),
            "candidate_crps": weighted_gaussian_mixture_crps(
                locations,
                example.residual_scales,
                example.residual_weights,
                example.target,
            ),
        }
        for metric, values in losses.items():
            cell_metrics[(example.puzzle, example.method)][metric].append(
                float(np.mean(values))
            )
    puzzle_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (puzzle, _method), metrics in cell_metrics.items():
        for metric, values in metrics.items():
            puzzle_metrics[puzzle][metric].append(float(np.mean(values)))
    macro = {
        puzzle: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for puzzle, metrics in puzzle_metrics.items()
    }
    return {
        "signed_delta_mae": _paired_summary(
            {p: values["b1_mae"] for p, values in macro.items()},
            {p: values["candidate_mae"] for p, values in macro.items()},
        ),
        "crps": _paired_summary(
            {p: values["b1_crps"] for p, values in macro.items()},
            {p: values["candidate_crps"] for p, values in macro.items()},
        ),
    }


def run_probe(m2_csv: Path, merged_result: Path) -> dict[str, Any]:
    examples, center_error = load_examples(m2_csv, merged_result)
    puzzles = sorted({example.puzzle for example in examples})
    fitters = {
        "global_exact_l1": fit_global_gate,
        "b1_magnitude_two_bin": lambda rows: fit_two_bin_gate(
            rows, feature="b1_magnitude"
        ),
        "expert_disagreement_two_bin": lambda rows: fit_two_bin_gate(
            rows, feature="expert_disagreement"
        ),
        "magnitude_disagreement_four_bin": fit_four_bin_gate,
    }
    variants: dict[str, Any] = {}
    for name, fitter in fitters.items():
        specs = {
            puzzle: fitter(_flat_train_rows(examples, puzzle)) for puzzle in puzzles
        }
        variants[name] = {"specs": specs, "metrics": evaluate(examples, specs)}
    return {
        "schema_version": SCHEMA,
        "evidence_status": "POST_HOC_OOF_DIAGNOSTIC_NOT_GATE_EVIDENCE",
        "input": {
            "m2_csv": str(m2_csv),
            "merged_result": str(merged_result),
            "folds": 20,
            "seed": 0,
            "legal_gate_inputs": [
                "absolute_b1_signed_delta_prediction",
                "absolute_b1_meanaligned_prediction_disagreement",
            ],
            "forbidden_gate_inputs": ["design_method", "held_target", "held_mask"],
        },
        "invariants": {
            "max_original_residual_center_error": center_error,
            "alpha_range": [0.0, 1.0],
            "held_puzzle_excluded_from_gate_fit": True,
        },
        "variants": variants,
        "interpretation_boundary": {
            "may_select_v3_direction": True,
            "may_claim_gate_pass": False,
            "may_retroactively_change_v2": False,
            "reason": (
                "the base OOF experts are meta-cross-fold artifacts; a kept gate must "
                "be fitted inside each outer-training fold and evaluated by a new "
                "end-to-end run"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--merged-result", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_probe(args.m2_csv, args.merged_result)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        name: {
            "signed_delta_relative_gain_percent": 100
            * row["metrics"]["signed_delta_mae"]["relative_gain"],
            "signed_delta_positive_puzzles": row["metrics"]["signed_delta_mae"][
                "positive_puzzles"
            ],
            "crps_gain": row["metrics"]["crps"]["mean_gain"],
            "crps_positive_puzzles": row["metrics"]["crps"]["positive_puzzles"],
        }
        for name, row in result["variants"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
