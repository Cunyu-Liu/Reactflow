#!/usr/bin/env python3
"""Diagnose Model Rescue v2 OOF errors under implemented and contracted estimands."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.model_rescue_v2 import (
    CALIBRATED_CANDIDATE,
    MEAN_CANDIDATE,
)
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE
from scripts.reactflow_delta.run_p2_v3 import _bio_key


SCHEMA = "reactflow_delta.model_rescue_v3_oof_diagnosis.v1"
METRICS = ("b1_mae", "candidate_mae", "b1_crps", "candidate_crps")


def _cell_summary(cell_values: dict[tuple[str, str], float]) -> dict[str, Any]:
    puzzle_methods: dict[str, list[float]] = defaultdict(list)
    for (puzzle, _method), value in cell_values.items():
        puzzle_methods[puzzle].append(float(value))
    per_puzzle = {
        puzzle: float(np.mean(values)) for puzzle, values in sorted(puzzle_methods.items())
    }
    return {
        "mean": float(np.mean(list(per_puzzle.values()))),
        "per_puzzle": per_puzzle,
        "n_puzzles": len(per_puzzle),
        "n_cells": len(cell_values),
    }


def summarize_mutant_balanced(
    mutant_values: dict[tuple[str, str], list[float]],
) -> dict[str, Any]:
    """Mutant -> puzzle-method cell -> puzzle macro hierarchy."""
    cells = {
        cell: float(np.mean(values)) for cell, values in mutant_values.items() if values
    }
    return _cell_summary(cells)


def summarize_position_pooled(
    position_totals: dict[tuple[str, str], tuple[float, int]],
) -> dict[str, Any]:
    """Current evaluator_v2 behavior: positions are pooled inside each method."""
    cells = {
        cell: total / count
        for cell, (total, count) in position_totals.items()
        if count > 0
    }
    return _cell_summary(cells)


def _paired_gain(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    puzzles = sorted(set(baseline["per_puzzle"]) & set(candidate["per_puzzle"]))
    gains = {
        puzzle: baseline["per_puzzle"][puzzle] - candidate["per_puzzle"][puzzle]
        for puzzle in puzzles
    }
    baseline_mean = float(np.mean([baseline["per_puzzle"][p] for p in puzzles]))
    mean_gain = float(np.mean(list(gains.values())))
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": float(np.mean([candidate["per_puzzle"][p] for p in puzzles])),
        "mean_gain": mean_gain,
        "relative_gain": mean_gain / baseline_mean,
        "positive_puzzles": int(sum(value > 0 for value in gains.values())),
        "per_puzzle_gain": gains,
    }


def _distance_group(distance: np.ndarray) -> np.ndarray:
    absolute = np.abs(distance)
    return np.select(
        [absolute == 0, absolute <= 5, absolute <= 20],
        ["edit_site", "near_1_5", "mid_6_20"],
        default="far_21_plus",
    )


def _magnitude_group(delta: np.ndarray) -> np.ndarray:
    absolute = np.abs(delta)
    return np.select(
        [absolute <= 0.05, absolute <= 0.20],
        ["near_zero_le_0_05", "moderate_0_05_0_20"],
        default="large_gt_0_20",
    )


class Collector:
    def __init__(self) -> None:
        self.pooled: dict[str, dict[tuple[str, str], list[float]]] = {
            metric: defaultdict(lambda: [0.0, 0.0]) for metric in METRICS
        }
        self.mutants: dict[str, dict[tuple[str, str], list[float]]] = {
            metric: defaultdict(list) for metric in METRICS
        }
        self.groups: dict[
            str, dict[str, dict[str, dict[tuple[str, str], list[float]]]]
        ] = defaultdict(
            lambda: {
                metric: defaultdict(lambda: defaultdict(list)) for metric in METRICS
            }
        )

    def add(
        self,
        *,
        puzzle: str,
        method: str,
        losses: dict[str, np.ndarray],
        group_arrays: dict[str, np.ndarray],
    ) -> None:
        cell = (puzzle, method)
        n_positions = len(next(iter(losses.values())))
        for metric, values in losses.items():
            self.pooled[metric][cell][0] += float(values.sum())
            self.pooled[metric][cell][1] += float(n_positions)
            self.mutants[metric][cell].append(float(values.mean()))
        for group_type, labels in group_arrays.items():
            for label in np.unique(labels):
                selected = labels == label
                if not bool(selected.any()):
                    continue
                for metric, values in losses.items():
                    self.groups[group_type][metric][str(label)][cell].append(
                        float(values[selected].mean())
                    )

    def _summary(self, hierarchy: str, metric: str) -> dict[str, Any]:
        if hierarchy == "position_pooled":
            totals = {
                cell: (values[0], int(values[1]))
                for cell, values in self.pooled[metric].items()
            }
            return summarize_position_pooled(totals)
        return summarize_mutant_balanced(self.mutants[metric])

    def finish(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for hierarchy in ("position_pooled", "mutant_balanced"):
            b1_mae = self._summary(hierarchy, "b1_mae")
            candidate_mae = self._summary(hierarchy, "candidate_mae")
            b1_crps = self._summary(hierarchy, "b1_crps")
            candidate_crps = self._summary(hierarchy, "candidate_crps")
            result[hierarchy] = {
                "signed_delta_mae": _paired_gain(b1_mae, candidate_mae),
                "crps": _paired_gain(b1_crps, candidate_crps),
            }

        breakdown: dict[str, Any] = {}
        for group_type, metrics in self.groups.items():
            labels = sorted(metrics["b1_mae"])
            breakdown[group_type] = {}
            for label in labels:
                b1_mae = summarize_mutant_balanced(metrics["b1_mae"][label])
                candidate_mae = summarize_mutant_balanced(
                    metrics["candidate_mae"][label]
                )
                b1_crps = summarize_mutant_balanced(metrics["b1_crps"][label])
                candidate_crps = summarize_mutant_balanced(
                    metrics["candidate_crps"][label]
                )
                breakdown[group_type][label] = {
                    "signed_delta_mae": _paired_gain(b1_mae, candidate_mae),
                    "crps": _paired_gain(b1_crps, candidate_crps),
                }
        result["mutant_balanced_breakdown"] = breakdown
        return result


def _load_prediction(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        return {name: handle[name] for name in handle.files}


def analyze(m2_csv: Path, merged_result: Path) -> dict[str, Any]:
    merged = json.loads(merged_result.read_text(encoding="utf-8"))
    folds = sorted(merged["folds"], key=lambda row: int(row["outer_fold"]))
    if [int(row["outer_fold"]) for row in folds] != list(range(20)):
        raise ValueError("diagnosis requires complete folds 0 through 19")

    univ = M2Universe(m2_csv)
    univ.build()
    records_by_puzzle: dict[str, list[Any]] = defaultdict(list)
    for record in univ.get_records():
        records_by_puzzle[record.puzzle].append(record)

    collector = Collector()
    expected_pooled: dict[str, dict[str, float]] = {
        "b1_mae": {},
        "candidate_mae": {},
        "b1_crps": {},
        "candidate_crps": {},
    }
    for fold in folds:
        puzzle = str(fold["held_puzzle"])
        baseline = _load_prediction(fold["baseline"]["prediction_artifact"])
        mean = _load_prediction(
            fold["candidates"][MEAN_CANDIDATE]["prediction_artifact"]
        )
        calibrated = _load_prediction(
            fold["candidates"][CALIBRATED_CANDIDATE]["prediction_artifact"]
        )
        baseline_index = {str(key): index for index, key in enumerate(baseline["keys"])}
        mean_index = {str(key): index for index, key in enumerate(mean["keys"])}
        calibrated_index = {
            str(key): index for index, key in enumerate(calibrated["keys"])
        }
        if set(baseline_index) != set(mean_index) or set(mean_index) != set(
            calibrated_index
        ):
            raise ValueError(f"prediction key mismatch in {puzzle}")

        expected_pooled["b1_mae"][puzzle] = float(
            fold["baseline"]["score"]["signed_delta_mae"]
        )
        expected_pooled["candidate_mae"][puzzle] = float(
            fold["candidates"][MEAN_CANDIDATE]["score"]["signed_delta_mae"]
        )
        expected_pooled["b1_crps"][puzzle] = float(fold["baseline"]["score"]["crps"])
        expected_pooled["candidate_crps"][puzzle] = float(
            fold["candidates"][CALIBRATED_CANDIDATE]["score"]["crps"]
        )

        for record in records_by_puzzle[puzzle]:
            construct = univ.get_construct(record.construct_id)
            target, _error = univ.mutant_full_profile(
                record.wt_id, record.design_pos, record.ref, record.alt
            )
            if target is None:
                continue
            qualified = construct.wt_observed & np.isfinite(target)
            positions = np.flatnonzero(qualified)
            if len(positions) == 0:
                continue
            keys = [_bio_key(univ, record, int(position)) for position in positions]
            baseline_rows = np.asarray([baseline_index[key] for key in keys], dtype=int)
            mean_rows = np.asarray([mean_index[key] for key in keys], dtype=int)
            calibrated_rows = np.asarray(
                [calibrated_index[key] for key in keys], dtype=int
            )
            true_delta = target[positions] - construct.wt_reactivity[positions]
            b1_point = np.sum(
                baseline["weights"][baseline_rows]
                * baseline["locations"][baseline_rows],
                axis=1,
            ) / np.sum(baseline["weights"][baseline_rows], axis=1)
            b1_delta = b1_point - construct.wt_reactivity[positions]
            candidate_delta = mean["delta_mean"][mean_rows]
            b1_crps = weighted_gaussian_mixture_crps(
                baseline["locations"][baseline_rows],
                baseline["scales"][baseline_rows],
                baseline["weights"][baseline_rows],
                target[positions],
            )
            candidate_crps = weighted_gaussian_mixture_crps(
                calibrated["locations"][calibrated_rows],
                calibrated["scales"][calibrated_rows],
                calibrated["weights"][calibrated_rows],
                target[positions],
            )
            collector.add(
                puzzle=puzzle,
                method=record.method,
                losses={
                    "b1_mae": np.abs(true_delta - b1_delta),
                    "candidate_mae": np.abs(true_delta - candidate_delta),
                    "b1_crps": b1_crps,
                    "candidate_crps": candidate_crps,
                },
                group_arrays={
                    "distance": _distance_group(positions - record.full_pos),
                    "target_magnitude": _magnitude_group(true_delta),
                    "readout_region": construct.region_map[positions],
                    "method": np.full(len(positions), record.method, dtype=object),
                    "mutation": np.full(
                        len(positions), f"{record.ref}>{record.alt}", dtype=object
                    ),
                },
            )

    result = collector.finish()
    pooled = result["position_pooled"]
    observed = {
        "b1_mae": pooled["signed_delta_mae"]["baseline_mean"],
        "candidate_mae": pooled["signed_delta_mae"]["candidate_mean"],
        "b1_crps": pooled["crps"]["baseline_mean"],
        "candidate_crps": pooled["crps"]["candidate_mean"],
    }
    recorded = {
        metric: float(np.mean(list(per_puzzle.values())))
        for metric, per_puzzle in expected_pooled.items()
    }
    matching = {
        metric: abs(observed[metric] - recorded[metric]) for metric in observed
    }
    return {
        "schema_version": SCHEMA,
        "evidence_status": "COMPLETED_DEVELOPMENT_OOF_DIAGNOSIS",
        "input": {
            "m2_csv": str(m2_csv),
            "merged_result": str(merged_result),
            "folds": 20,
            "seed": 0,
        },
        "estimand_finding": {
            "contracted_hierarchy": "position_to_mutant_to_method_cell_to_puzzle",
            "implemented_evaluator_v2_hierarchy": "position_pooled_within_method_cell_to_puzzle",
            "implementation_mismatch_confirmed": True,
            "recomputed_vs_recorded_max_abs_difference": max(matching.values()),
            "recomputed_matches_recorded_atol_1e_10": max(matching.values()) <= 1e-10,
        },
        "results": result,
        "interpretation_boundary": {
            "retroactive_v2_pass_allowed": False,
            "new_evaluator_and_model_experiment_required": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--merged-result", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.m2_csv, args.merged_result)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "out_json": str(args.out_json),
        "implemented_mean_relative_gain": result["results"]["position_pooled"][
            "signed_delta_mae"
        ]["relative_gain"],
        "contracted_mean_relative_gain": result["results"]["mutant_balanced"][
            "signed_delta_mae"
        ]["relative_gain"],
        "implemented_crps_gain": result["results"]["position_pooled"]["crps"][
            "mean_gain"
        ],
        "contracted_crps_gain": result["results"]["mutant_balanced"]["crps"][
            "mean_gain"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
