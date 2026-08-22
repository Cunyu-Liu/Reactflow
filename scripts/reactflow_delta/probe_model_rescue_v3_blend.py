#!/usr/bin/env python3
"""Probe train-puzzle-only blending of B1 and MeanAligned OOF experts.

This is a diagnostic, not a qualification run.  The base OOF experts used to fit a
leave-one-puzzle blend were produced by different outer models, some of which saw the
probe's held puzzle.  A promising result must therefore be rebuilt inside every outer
training fold before it can count as scientific evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    MEAN_CANDIDATE,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key


SCHEMA = "reactflow_delta.model_rescue_v3_blend_probe.v1"
CENTER_ATOL = 1e-6


@dataclass(frozen=True)
class MutantExample:
    puzzle: str
    method: str
    true_delta: np.ndarray
    target: np.ndarray
    b1_delta: np.ndarray
    mean_delta: np.ndarray
    b1_locations: np.ndarray
    b1_scales: np.ndarray
    b1_weights: np.ndarray
    residual_locations: np.ndarray
    residual_scales: np.ndarray
    residual_weights: np.ndarray


def shifted_residual_locations(
    locations: np.ndarray,
    old_mean: np.ndarray,
    new_mean: np.ndarray,
) -> np.ndarray:
    """Translate a zero-mean residual distribution without changing its shape."""
    locations = np.asarray(locations, dtype=float)
    old_mean = np.asarray(old_mean, dtype=float)
    new_mean = np.asarray(new_mean, dtype=float)
    return locations + (new_mean - old_mean)[:, None]


def mixture_mean(
    locations: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return np.sum(np.asarray(locations, dtype=float) * weights, axis=1) / np.sum(
        weights, axis=1
    )


def build_cell_loss_curves(
    examples: Iterable[MutantExample], alpha_grid: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    """Return exact position->mutant->method-cell L1 curves."""
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    alpha = np.asarray(alpha_grid, dtype=float)[:, None]
    for example in examples:
        blended = example.b1_delta[None, :] + alpha * (
            example.mean_delta - example.b1_delta
        )[None, :]
        grouped[(example.puzzle, example.method)].append(
            np.mean(np.abs(example.true_delta[None, :] - blended), axis=1)
        )
    return {
        cell: np.mean(np.stack(mutant_curves, axis=0), axis=0)
        for cell, mutant_curves in grouped.items()
    }


def select_alpha_from_curves(
    curves: dict[tuple[str, str], np.ndarray],
    alpha_grid: np.ndarray,
    *,
    excluded_puzzle: str,
    method: str | None,
) -> float:
    """Select alpha without using any cell from ``excluded_puzzle``."""
    eligible = [
        curve
        for (puzzle, cell_method), curve in curves.items()
        if puzzle != excluded_puzzle and (method is None or cell_method == method)
    ]
    if not eligible:
        raise ValueError(
            f"no train-puzzle cells for excluded={excluded_puzzle!r}, method={method!r}"
        )
    objective = np.mean(np.stack(eligible, axis=0), axis=0)
    minimum = float(np.min(objective))
    tied = np.flatnonzero(np.isclose(objective, minimum, atol=1e-12, rtol=0.0))
    # Prefer the incumbent MeanAligned expert under an exact numerical tie.
    best = min(tied.tolist(), key=lambda i: (abs(float(alpha_grid[i]) - 1.0), i))
    return float(alpha_grid[best])


def _load_prediction(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        return {name: handle[name] for name in handle.files}


def load_examples(m2_csv: Path, merged_result: Path) -> tuple[list[MutantExample], float]:
    merged = json.loads(merged_result.read_text(encoding="utf-8"))
    folds = sorted(merged["folds"], key=lambda row: int(row["outer_fold"]))
    if [int(row["outer_fold"]) for row in folds] != list(range(20)):
        raise ValueError("blend probe requires complete folds 0 through 19")

    universe = M2Universe(m2_csv)
    universe.build()
    records_by_puzzle: dict[str, list[Any]] = defaultdict(list)
    for record in universe.get_records():
        records_by_puzzle[record.puzzle].append(record)

    examples: list[MutantExample] = []
    max_residual_center_error = 0.0
    for fold in folds:
        puzzle = str(fold["held_puzzle"])
        baseline = _load_prediction(fold["baseline"]["prediction_artifact"])
        mean = _load_prediction(
            fold["candidates"][MEAN_CANDIDATE]["prediction_artifact"]
        )
        calibrated = _load_prediction(
            fold["candidates"][CALIBRATED_CANDIDATE]["prediction_artifact"]
        )
        indices = {
            "baseline": {str(key): i for i, key in enumerate(baseline["keys"])},
            "mean": {str(key): i for i, key in enumerate(mean["keys"])},
            "calibrated": {
                str(key): i for i, key in enumerate(calibrated["keys"])
            },
        }
        if not (
            set(indices["baseline"])
            == set(indices["mean"])
            == set(indices["calibrated"])
        ):
            raise ValueError(f"prediction key mismatch in {puzzle}")

        for record in records_by_puzzle[puzzle]:
            construct = universe.get_construct(record.construct_id)
            target, _error = universe.mutant_full_profile(
                record.wt_id, record.pos, record.ref, record.alt
            )
            if target is None:
                continue
            qualified = construct.wt_observed & np.isfinite(target)
            positions = np.flatnonzero(qualified)
            if len(positions) == 0:
                continue
            keys = [_bio_key(universe, record, int(position)) for position in positions]
            b_rows = np.asarray([indices["baseline"][key] for key in keys], dtype=int)
            m_rows = np.asarray([indices["mean"][key] for key in keys], dtype=int)
            c_rows = np.asarray([indices["calibrated"][key] for key in keys], dtype=int)
            wt = construct.wt_reactivity[positions]
            b1_point = mixture_mean(
                baseline["locations"][b_rows], baseline["weights"][b_rows]
            )
            residual_point = mixture_mean(
                calibrated["locations"][c_rows], calibrated["weights"][c_rows]
            )
            mean_point = wt + mean["delta_mean"][m_rows]
            max_residual_center_error = max(
                max_residual_center_error,
                float(np.max(np.abs(residual_point - mean_point))),
            )
            examples.append(
                MutantExample(
                    puzzle=puzzle,
                    method=record.method,
                    true_delta=target[positions] - wt,
                    target=target[positions],
                    b1_delta=b1_point - wt,
                    mean_delta=mean["delta_mean"][m_rows],
                    b1_locations=baseline["locations"][b_rows],
                    b1_scales=baseline["scales"][b_rows],
                    b1_weights=baseline["weights"][b_rows],
                    residual_locations=calibrated["locations"][c_rows],
                    residual_scales=calibrated["scales"][c_rows],
                    residual_weights=calibrated["weights"][c_rows],
                )
            )
    # The frozen artifacts store the authoritative point_mean in float64 but their
    # component locations originate from float32 tensors.  Preserve and report that
    # quantization error; reject only a discrepancy larger than its observed scale.
    if max_residual_center_error > CENTER_ATOL:
        raise ValueError(
            "calibrated residual is not centered on MeanAligned point mean: "
            f"max error {max_residual_center_error}"
        )
    return examples, max_residual_center_error


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
    mean_gain = baseline_mean - candidate_mean
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_gain": mean_gain,
        "relative_gain": mean_gain / baseline_mean,
        "positive_puzzles": int(sum(value > 0 for value in gains.values())),
        "per_puzzle_gain": gains,
    }


def evaluate_variant(
    examples: list[MutantExample],
    alphas: dict[tuple[str, str | None], float],
    *,
    method_specific: bool,
) -> dict[str, Any]:
    cell_mutants: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in examples:
        key = (
            example.puzzle,
            example.method if method_specific else None,
        )
        alpha = alphas[key]
        blend_delta = example.b1_delta + alpha * (
            example.mean_delta - example.b1_delta
        )
        locations = shifted_residual_locations(
            example.residual_locations,
            example.mean_delta,
            blend_delta,
        )
        candidate_point = mixture_mean(locations, example.residual_weights)
        expected_point = (
            mixture_mean(example.residual_locations, example.residual_weights)
            + blend_delta
            - example.mean_delta
        )
        if not np.allclose(candidate_point, expected_point, atol=1e-10, rtol=0.0):
            raise AssertionError("residual translation changed its zero-mean constraint")
        losses = {
            "b1_mae": np.abs(example.true_delta - example.b1_delta),
            "candidate_mae": np.abs(example.true_delta - blend_delta),
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
            cell_mutants[(example.puzzle, example.method)][metric].append(
                float(np.mean(values))
            )

    puzzle_methods: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (puzzle, _method), metrics in cell_mutants.items():
        for metric, mutant_losses in metrics.items():
            puzzle_methods[puzzle][metric].append(float(np.mean(mutant_losses)))
    per_puzzle = {
        puzzle: {
            metric: float(np.mean(method_values))
            for metric, method_values in metrics.items()
        }
        for puzzle, metrics in puzzle_methods.items()
    }
    return {
        "signed_delta_mae": _paired_summary(
            {p: row["b1_mae"] for p, row in per_puzzle.items()},
            {p: row["candidate_mae"] for p, row in per_puzzle.items()},
        ),
        "crps": _paired_summary(
            {p: row["b1_crps"] for p, row in per_puzzle.items()},
            {p: row["candidate_crps"] for p, row in per_puzzle.items()},
        ),
    }


def run_probe(
    m2_csv: Path,
    merged_result: Path,
    *,
    alpha_step: float,
) -> dict[str, Any]:
    if alpha_step <= 0 or alpha_step > 1:
        raise ValueError("alpha_step must be in (0, 1]")
    alpha_grid = np.linspace(0.0, 1.0, int(round(1.0 / alpha_step)) + 1)
    examples, center_error = load_examples(m2_csv, merged_result)
    curves = build_cell_loss_curves(examples, alpha_grid)
    puzzles = sorted({example.puzzle for example in examples})
    methods_by_puzzle: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        methods_by_puzzle[example.puzzle].add(example.method)

    global_alphas = {
        (puzzle, None): select_alpha_from_curves(
            curves, alpha_grid, excluded_puzzle=puzzle, method=None
        )
        for puzzle in puzzles
    }
    method_alphas = {
        (puzzle, method): select_alpha_from_curves(
            curves, alpha_grid, excluded_puzzle=puzzle, method=method
        )
        for puzzle in puzzles
        for method in sorted(methods_by_puzzle[puzzle])
    }
    return {
        "schema_version": SCHEMA,
        "evidence_status": "POST_HOC_OOF_DIAGNOSTIC_NOT_GATE_EVIDENCE",
        "input": {
            "m2_csv": str(m2_csv),
            "merged_result": str(merged_result),
            "folds": 20,
            "seed": 0,
            "alpha_grid": [float(value) for value in alpha_grid],
        },
        "invariants": {
            "max_original_residual_center_error": center_error,
            "candidate_residual_shape_unchanged": True,
            "candidate_residual_location_translated_only": True,
        },
        "variants": {
            "global_lopo_alpha": {
                "alphas": {p: global_alphas[(p, None)] for p in puzzles},
                "metrics": evaluate_variant(
                    examples, global_alphas, method_specific=False
                ),
            },
            "method_lopo_alpha": {
                "alphas": {
                    puzzle: {
                        method: method_alphas[(puzzle, method)]
                        for method in sorted(methods_by_puzzle[puzzle])
                    }
                    for puzzle in puzzles
                },
                "metrics": evaluate_variant(
                    examples, method_alphas, method_specific=True
                ),
            },
        },
        "interpretation_boundary": {
            "may_select_v3_direction": True,
            "may_claim_gate_pass": False,
            "may_retroactively_change_v2": False,
            "reason": (
                "other-puzzle OOF experts were trained in different outer folds and may "
                "have indirectly seen the probe-held puzzle; a kept gate must be fitted "
                "inside each outer-training fold and rerun end to end"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--merged-result", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    args = parser.parse_args(argv)
    result = run_probe(
        args.m2_csv,
        args.merged_result,
        alpha_step=args.alpha_step,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        name: {
            "signed_delta_relative_gain_percent": 100.0
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
