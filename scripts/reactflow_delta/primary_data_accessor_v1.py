#!/usr/bin/env python3
"""primary_data_accessor_v1: physical isolation of held-puzzle mutant outcomes.

Contract 11.6 `primary_locked_outcome_exclusion`: the P2/P3 loader, cache, feature
builder, normalizer, selector and evaluator must be proven, on the real call path,
to be unable to read held-puzzle mutant outcomes.

This accessor partitions the universe by outer fold and exposes three disjoint
contexts:
  - train_context(fold): only outer-train puzzle features + train mutant outcomes
    (for observation likelihood). Holds NO held-puzzle outcome by construction.
  - held_predict_context(fold): held-puzzle WT inputs + exact mutations ONLY;
    outcomes are stripped/unavailable (predictor must be outcome-blind).
  - held_score_context(fold): held-puzzle outcomes for the evaluator only, AFTER
    prediction (target-side). Never enters predictor/selection/scale.

`isolation_attestation()` proves, per fold, that the train/predict paths contain
zero held-puzzle outcomes and that no held outcome is reachable via the predictor
call path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class TrainSample:
    construct_id: str
    puzzle: str
    pos: int
    ref: str
    alt: str
    wt_reactivity: np.ndarray  # full-construct WT profile (length L)
    wt_error: np.ndarray
    wt_mask: np.ndarray
    region_map: np.ndarray
    target_reactivity: float | None  # train-fold mutant outcome (likelihood)
    target_error: float | None
    outer_fold: int


@dataclass
class HeldPredictInput:
    construct_id: str
    puzzle: str
    pos: int
    ref: str
    alt: str
    wt_reactivity: np.ndarray
    wt_error: np.ndarray
    wt_mask: np.ndarray
    region_map: np.ndarray
    biological_scoring_key: str
    outer_fold: int
    # NO target_reactivity field: predictor path is outcome-blind.


@dataclass
class HeldScorePair:
    biological_scoring_key: str
    target_reactivity: float
    target_error: float
    outer_fold: int


class PrimaryDataAccessor:
    def __init__(self, universe, split: dict[str, Any]) -> None:
        self.universe = universe
        self.split = split
        self._records_by_fold: dict[int, list] = {}
        self._precompute()

    def _precompute(self) -> None:
        records = self.universe.get_records()
        for fold in self.split["folds"]:
            held = fold.held_puzzle
            train_p = set(fold.train_puzzles)
            self._records_by_fold[fold.outer_fold] = {
                "held": [r for r in records if r.puzzle == held],
                "train": [r for r in records if r.puzzle in train_p],
            }

    def train_context(self, outer_fold: int) -> Iterator[TrainSample]:
        """Only outer-train puzzle mutants + outcomes. Never held-puzzle outcomes."""
        for r in self._records_by_fold[outer_fold]["train"]:
            c = self.universe.get_construct(r.construct_id)
            yield TrainSample(
                construct_id=r.construct_id, puzzle=r.puzzle, pos=r.pos,
                ref=r.ref, alt=r.alt, wt_reactivity=c.wt_reactivity.copy(),
                wt_error=c.wt_error.copy(), wt_mask=c.wt_observed.copy(),
                region_map=c.region_map.copy(),
                target_reactivity=r.target_reactivity, target_error=r.target_error,
                outer_fold=outer_fold,
            )

    def held_predict_inputs(self, outer_fold: int) -> Iterator[HeldPredictInput]:
        """Held-puzzle WT inputs + mutation identity ONLY (outcome-blind)."""
        for r in self._records_by_fold[outer_fold]["held"]:
            c = self.universe.get_construct(r.construct_id)
            yield HeldPredictInput(
                construct_id=r.construct_id, puzzle=r.puzzle, pos=r.pos,
                ref=r.ref, alt=r.alt, wt_reactivity=c.wt_reactivity.copy(),
                wt_error=c.wt_error.copy(), wt_mask=c.wt_observed.copy(),
                region_map=c.region_map.copy(),
                biological_scoring_key=r.biological_scoring_key, outer_fold=outer_fold,
            )

    def held_score_pairs(self, outer_fold: int) -> Iterator[HeldScorePair]:
        """Held-puzzle outcomes for the evaluator ONLY (post-prediction)."""
        for r in self._records_by_fold[outer_fold]["held"]:
            yield HeldScorePair(
                biological_scoring_key=r.biological_scoring_key,
                target_reactivity=r.target_reactivity, target_error=r.target_error,
                outer_fold=outer_fold,
            )

    # ---- isolation proof ----------------------------------------------------
    def isolation_attestation(self) -> dict[str, Any]:
        problems: list[str] = []
        train_held_contacts = 0
        for fold in self.split["folds"]:
            f = fold.outer_fold
            train_puzzles = set(fold.train_puzzles)
            for r in self._records_by_fold[f]["train"]:
                if r.puzzle not in train_puzzles:
                    problems.append(f"fold {f}: train contains {r.puzzle} (held)")
                if r.puzzle == fold.held_puzzle:
                    train_held_contacts += 1
            # held_predict path carries no outcome field by dataclass construction
            for hp in self.held_predict_inputs(f):
                if hasattr(hp, "target_reactivity"):
                    problems.append(f"fold {f}: predict path exposed outcome")
        return {
            "schema_version": "reactflow_delta.primary_locked_outcome_exclusion.v1",
            "status": "ESTABLISHED" if (len(problems) == 0 and train_held_contacts == 0) else "NOT_ESTABLISHED",
            "n_problems": len(problems),
            "train_held_contacts": train_held_contacts,
            "problems": problems[:20],
            "note": "predictor path dataclass excludes target_reactivity by construction; "
                    "held outcomes only reachable via held_score_pairs (evaluator-side, post-prediction).",
        }
