#!/usr/bin/env python3
"""Audit position-aligned WT signal without using mutant outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.reactflow_delta.m2_universe_v1 import M2Universe


SCHEMA = "reactflow_delta.puzzle_set_wt_alignment_audit.proposed.v1"
FIXED_SHIFT_CONTROLS = (1, 17, 43, 89)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.shape != right.shape
        or left.ndim != 1
        or len(left) < 8
        or float(np.std(left)) == 0.0
        or float(np.std(right)) == 0.0
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def region_alignment_summary(
    constructs: Sequence[Any],
    start: int,
    end: int,
    *,
    shifts: Sequence[int] = FIXED_SHIFT_CONTROLS,
) -> dict[str, Any]:
    """Compare registered positions with fixed wrong-position controls."""

    if len(constructs) != 8 or end - start < 8:
        raise ValueError("WT alignment audit requires eight nontrivial constructs")
    width = end - start
    if any(int(shift) <= 0 or int(shift) % width == 0 for shift in shifts):
        raise ValueError("WT alignment controls must remain nonzero after wrapping")
    effective_shifts = tuple(int(shift) % width for shift in shifts)
    pair_same: list[float] = []
    pair_shifted: list[float] = []
    consensus_same: list[float] = []
    consensus_shifted: list[float] = []
    rescued_missing: list[float] = []
    for left_index, left in enumerate(constructs):
        left_values = np.asarray(left.wt_reactivity[start:end], dtype=np.float64)
        left_mask = np.asarray(left.wt_observed[start:end], dtype=bool)
        others = [
            construct
            for index, construct in enumerate(constructs)
            if index != left_index
        ]
        other_values = np.stack(
            [np.asarray(value.wt_reactivity[start:end], dtype=np.float64) for value in others]
        )
        other_masks = np.stack(
            [np.asarray(value.wt_observed[start:end], dtype=bool) for value in others]
        )
        counts = other_masks.sum(axis=0)
        sums = np.where(other_masks, other_values, 0.0).sum(axis=0)
        consensus = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums),
            where=counts > 0,
        )
        valid = left_mask & (counts > 0)
        value = _correlation(left_values[valid], consensus[valid])
        if value is not None:
            consensus_same.append(value)
        for shift in effective_shifts:
            shifted = np.roll(consensus, int(shift))
            shifted_valid = left_mask & np.roll(counts > 0, int(shift))
            value = _correlation(
                left_values[shifted_valid], shifted[shifted_valid]
            )
            if value is not None:
                consensus_shifted.append(value)
        missing = ~left_mask
        if bool(missing.any()):
            rescued_missing.append(float(np.mean((counts > 0)[missing])))

        for right in constructs[left_index + 1 :]:
            right_values = np.asarray(
                right.wt_reactivity[start:end], dtype=np.float64
            )
            right_mask = np.asarray(right.wt_observed[start:end], dtype=bool)
            valid = left_mask & right_mask
            value = _correlation(left_values[valid], right_values[valid])
            if value is not None:
                pair_same.append(value)
            for shift in effective_shifts:
                shifted_values = np.roll(right_values, int(shift))
                shifted_mask = np.roll(right_mask, int(shift))
                valid = left_mask & shifted_mask
                value = _correlation(
                    left_values[valid], shifted_values[valid]
                )
                if value is not None:
                    pair_shifted.append(value)

    return {
        "effective_shift_controls": list(effective_shifts),
        "pair_same_mean": float(np.mean(pair_same)) if pair_same else None,
        "pair_shift_control_mean": (
            float(np.mean(pair_shifted)) if pair_shifted else None
        ),
        "pair_alignment_increment": (
            float(np.mean(pair_same) - np.mean(pair_shifted))
            if pair_same and pair_shifted
            else None
        ),
        "consensus_same_mean": (
            float(np.mean(consensus_same)) if consensus_same else None
        ),
        "consensus_shift_control_mean": (
            float(np.mean(consensus_shifted)) if consensus_shifted else None
        ),
        "consensus_alignment_increment": (
            float(np.mean(consensus_same) - np.mean(consensus_shifted))
            if consensus_same and consensus_shifted
            else None
        ),
        "missing_positions_rescued_fraction": (
            float(np.mean(rescued_missing)) if rescued_missing else None
        ),
        "n_pair_correlations": len(pair_same),
        "n_consensus_correlations": len(consensus_same),
    }


def audit_wt_alignment(
    universe: Any,
    *,
    expected_puzzles: int = 20,
    shifts: Sequence[int] = FIXED_SHIFT_CONTROLS,
) -> dict[str, Any]:
    by_puzzle: dict[str, list[Any]] = defaultdict(list)
    for construct in universe.constructs.values():
        by_puzzle[str(construct.puzzle)].append(construct)
    if len(by_puzzle) != int(expected_puzzles):
        raise ValueError("WT alignment audit puzzle universe is incomplete")

    rows = []
    for puzzle in sorted(by_puzzle):
        constructs = sorted(by_puzzle[puzzle], key=lambda value: value.method)
        if len(constructs) != 8:
            raise ValueError(f"WT alignment audit puzzle {puzzle} lacks eight constructs")
        starts = {int(value.design_start) for value in constructs}
        ends = {int(value.design_end) for value in constructs}
        lengths = {len(value.sequence) for value in constructs}
        if len(starts) != 1 or len(ends) != 1 or len(lengths) != 1:
            raise ValueError(f"WT alignment audit puzzle {puzzle} changed coordinates")
        start = next(iter(starts))
        end = next(iter(ends))
        length = next(iter(lengths))
        sequence_identity = []
        for index, left in enumerate(constructs):
            for right in constructs[index + 1 :]:
                sequence_identity.append(
                    float(np.mean([a == b for a, b in zip(left.sequence, right.sequence)]))
                )
        rows.append(
            {
                "puzzle": puzzle,
                "mean_pair_sequence_identity": float(np.mean(sequence_identity)),
                "full": region_alignment_summary(
                    constructs, 0, length, shifts=shifts
                ),
                "design": region_alignment_summary(
                    constructs, start, end, shifts=shifts
                ),
            }
        )

    summary: dict[str, Any] = {"fixed_shift_controls": list(map(int, shifts))}
    for region in ("full", "design"):
        for field in (
            "pair_alignment_increment",
            "consensus_alignment_increment",
            "missing_positions_rescued_fraction",
        ):
            values = [
                float(row[region][field])
                for row in rows
                if row[region][field] is not None
            ]
            summary[f"{region}_{field}_mean"] = (
                float(np.mean(values)) if values else None
            )
            summary[f"{region}_{field}_positive_puzzles"] = sum(
                value > 0.0 for value in values
            )
            summary[f"{region}_{field}_n"] = len(values)
    summary["mean_pair_sequence_identity"] = float(
        np.mean([row["mean_pair_sequence_identity"] for row in rows])
    )
    return {
        "schema_version": SCHEMA,
        "status": "WT_INPUT_ALIGNMENT_DIAGNOSTIC_COMPLETE",
        "evidence_status": "DEVELOPMENT_WT_INPUT_DIAGNOSTIC_ONLY",
        "scope": "WT_SEQUENCE_REACTIVITY_MASK_ONLY",
        "mutant_outcome_used": False,
        "external_outcome_accessed": False,
        "summary": summary,
        "per_puzzle": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError("WT alignment audit refuses to overwrite its artifact")
    universe = M2Universe(args.m2_csv)
    universe.build()
    result = audit_wt_alignment(universe)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
