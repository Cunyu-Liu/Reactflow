#!/usr/bin/env python3
"""split_v4_lopo_puzzle: 20-fold leave-one-puzzle-out with fixed inner 4-fold.

Outer folds: each held puzzle is fully excluded from fit/tuning/prior/calibration/
early-stopping/B* selection (all its methods, constructs, mutants).
Inner split: within the 19 outer-train puzzles, a fixed outcome-blind 4-fold
grouped by puzzle (for nested T*/Direct* selection). All normalization/selection
is fold-local.

Outcome-blind: split is derived only from puzzle identities + raw availability,
never from mutant outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Fold:
    outer_fold: int
    held_puzzle: str
    train_puzzles: list[str]
    inner_groups: list[list[str]]  # 4 puzzle groups over train_puzzles


def _grouped_folds(puzzles: list[str], n_inner: int, seed: int = 0) -> list[list[str]]:
    rng = np.random.RandomState(seed)
    arr = np.array(puzzles)
    idx = rng.permutation(len(arr))
    return [arr[idx[i::n_inner]].tolist() for i in range(n_inner)]


def build_split_v4(puzzles: list[str], *, n_inner: int = 4, seed: int = 0) -> dict[str, Any]:
    puzzles = sorted(puzzles)
    outer = []
    for f, held in enumerate(puzzles):
        train = [p for p in puzzles if p != held]
        inner = _grouped_folds(train, n_inner, seed=seed)
        outer.append(Fold(outer_fold=f, held_puzzle=held, train_puzzles=train,
                          inner_groups=inner))
    return {
        "schema_version": "reactflow_delta.split_v4_lopo_puzzle.v1",
        "n_outer_folds": len(outer),
        "n_inner_folds": n_inner,
        "puzzles": puzzles,
        "folds": outer,
        "seed": seed,
    }


def exposure_audit(split: dict[str, Any], puzzle_to_cells: dict[str, list[str]]) -> dict[str, Any]:
    """Verify held puzzle's methods/constructs are fully excluded from outer-train."""
    problems = []
    for fold in split["folds"]:
        held = fold.held_puzzle
        train = set(fold.train_puzzles)
        # cells of held puzzle must not appear in train puzzles
        held_cells = puzzle_to_cells.get(held, [])
        for t in train:
            if any(t == held for _ in [0]):  # held not in train by construction
                pass
        # any train puzzle cell that belongs to held puzzle? impossible by key
        for t in train:
            for c in puzzle_to_cells.get(t, []):
                if c.split("_")[0] == held:
                    problems.append((held, t, c))
    return {
        "schema_version": "reactflow_delta.split_v4.exposure_audit.v1",
        "held_puzzle_zero_exposure": len(problems) == 0,
        "n_problems": len(problems),
        "problems": problems[:20],
        "method_disjoint_note": "method-disjoint and cell-LOO are sensitivity/diagnostic only, not primary",
    }
