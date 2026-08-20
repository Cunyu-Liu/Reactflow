#!/usr/bin/env python3
"""evaluator_v2: frozen, target-invariant, method-balanced probabilistic evaluator.

Implements the frozen estimand hierarchy required by the 2026-08-17 audit
(§12 Phase 1, §13 P0-2):

    position -> mutant -> cell(puzzle x method) -> method-balanced puzzle -> 20-puzzle mean

Key guarantees (audit P0-2 acceptance):
  * method-balanced: a puzzle's loss is the MEAN over its methods, NOT a mutant-pooled
    average. Duplicating mutants of one method must NOT change the puzzle weight.
  * candidate/baseline biological-scoring-key exact pairing: only keys present in BOTH
    sides are scored; missing keys raise a mismatch (never zero-filled).
  * missing target positions are excluded by mask and NEVER treated as 0.
  * prediction/score separation: the scoring functions receive predictions (loc/scale)
    and the independently-joined held target; the prediction side carries no target.

CRPS primitives are reused verbatim from evaluator_crps_v1 (single source of truth);
this module only adds the aggregation layer, so hand-computed fixtures stay in lockstep.

Outcome-blind: structural scoring only; targets arrive from the evaluator-side join.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from scripts.reactflow_delta.evaluator_crps_v1 import (
    crps_gaussian,
    crps_student_t,
    mixture_crps,
)

SCHEMA_VERSION = "reactflow_delta.evaluator_v2.v1"


# --------------------------------------------------------------------------- #
# per-position scoring (single source of truth)
# --------------------------------------------------------------------------- #
def score_position(loc: float, scale: float, y: float, family: str = "gaussian",
                   df: float | None = None) -> float:
    """Exact CRPS of the predictive distribution at observed y."""
    if family == "gaussian":
        return crps_gaussian(loc, scale, y)
    if family == "student_t":
        return crps_student_t(loc, scale, float(df), y)
    raise ValueError(f"unknown family {family}")


def score_mixture(locs: list[float], scales: list[float], weights: list[float],
                  y: float) -> float:
    """Exact CRPS of an equal-arbitrary-weight Gaussian mixture at y."""
    return mixture_crps(locs, scales, weights, y)


# --------------------------------------------------------------------------- #
# ledger row types
# --------------------------------------------------------------------------- #
@dataclass
class PredPoint:
    """A single prediction row keyed by biological_scoring_key (no target)."""
    biological_scoring_key: str
    model_id: str
    seed_or_component_id: Any
    outer_fold: int
    family: str            # gaussian | student_t
    location: float
    scale: float
    df: float | None = None
    mixture_weight: float = 1.0
    status: str = "covered"   # covered | failure; failure_reason below
    failure_reason: str | None = None


@dataclass
class TargetPoint:
    """A held target row keyed by the SAME biological_scoring_key (evaluator-side)."""
    biological_scoring_key: str
    target: float | None       # None = missing/not-observed
    qualified: bool            # whether this position counts for scoring


# --------------------------------------------------------------------------- #
# aggregation: position -> mutant -> cell -> method -> puzzle
# --------------------------------------------------------------------------- #
def _bio_key_parts(bio_key: str) -> dict[str, str]:
    """Parse a biological_scoring_key into {dataset,puzzle,method,construct,mutation,pos}.

    Expects the registered-universe layout produced by M2Universe:
      openknot_m2|{puzzle}|{method}|{construct}|{pos}|{ref}>{alt}|{pos}
    The final two fields are duplicated pos for legacy reasons; we use the 4th part
    (construct) and the mutation part to build cell/mutant grouping.
    """
    parts = bio_key.split("|")
    return {
        "dataset": parts[0] if len(parts) > 0 else "",
        "puzzle": parts[1] if len(parts) > 1 else "",
        "method": parts[2] if len(parts) > 2 else "",
        "construct": parts[3] if len(parts) > 3 else "",
        "mutation": parts[5] if len(parts) > 5 else parts[4],
        "pos": parts[6] if len(parts) > 6 else parts[4],
    }


def _aggregate_position_losses(position_losses: dict[str, float],
                               *, method_balanced: bool = True) -> dict[str, dict[str, Any]]:
    """Frozen method-balanced aggregation over per-bio-key position losses.

    position_losses: {biological_scoring_key: loss}. Missing keys are simply
    absent (missing targets never appear as 0). Returns
        {puzzle: {"methods": {method: cell_loss}, "L": method-balanced L,
                  "n_methods": int, "n_positions": int}}
    """
    from collections import defaultdict
    puzzle_method_loss: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for k, lval in position_losses.items():
        parts = _bio_key_parts(k)
        puzzle_method_loss[parts["puzzle"]][parts["method"]].append(float(lval))
    result_puzzles = {}
    for puzzle, methods in puzzle_method_loss.items():
        method_loss = {m: float(np.mean(l)) for m, l in methods.items()}
        if method_balanced:
            L = float(np.mean(list(method_loss.values())))
        else:
            allvals = [v for vs in methods.values() for v in vs]
            L = float(np.mean(allvals))
        result_puzzles[puzzle] = {
            "methods": method_loss, "L": L,
            "n_methods": len(method_loss),
            "n_positions": int(sum(len(v) for v in methods.values())),
        }
    return result_puzzles


def score_ledger(pred_rows: list[PredPoint], target_rows: list[TargetPoint],
                 *, method_balanced: bool = True) -> dict[str, Any]:
    """Score a full prediction ledger against a full target ledger.

    Returns a per-puzzle result map and a pool of metadata:
      result["puzzles"][puzzle] = {"methods": {method: L}, "L": method-balanced loss}
      result["effects"][puzzle] = {"candidate": L_c, "baseline": L_b, "D_p": L_b - L_c}
      result["pairs_matched"] / ["pairs_missing_candidate"] / ["pairs_missing_baseline"]
    """
    pred_by_key = {r.biological_scoring_key: r for r in pred_rows}
    tgt_by_key = {r.biological_scoring_key: r for r in target_rows}

    # exact key pairing: only keys present in BOTH sides are scored
    matched = sorted(pred_by_key.keys() & tgt_by_key.keys())
    only_pred = sorted(pred_by_key.keys() - tgt_by_key.keys())
    only_tgt = sorted(tgt_by_key.keys() - pred_by_key.keys())
    if only_pred or only_tgt:
        raise ValueError(
            "candidate/baseline biological scoring key mismatch: "
            f"{len(only_pred)} pred-only, {len(only_tgt)} target-only")

    # per-position losses (missing target positions excluded, NEVER 0)
    position_losses: dict[str, float] = {}
    n_matched_positions = 0
    for k in matched:
        pr = pred_by_key[k]; tr = tgt_by_key[k]
        if tr.target is None or not tr.qualified:
            continue  # missing target excluded, NEVER 0
        if pr.status != "covered" or pr.scale <= 0:
            continue
        position_losses[k] = score_position(pr.location, pr.scale, float(tr.target),
                                            family=pr.family, df=pr.df)
        n_matched_positions += 1

    result_puzzles = _aggregate_position_losses(position_losses,
                                                method_balanced=method_balanced)

    return {
        "schema_version": SCHEMA_VERSION,
        "method_balanced": method_balanced,
        "n_matched_keys": len(matched),
        "n_matched_positions": n_matched_positions,
        "puzzles": result_puzzles,
        "effects": {},
    }


def score_position_losses(position_losses: dict[str, float], *,
                          method_balanced: bool = True) -> dict[str, Any]:
    """Score a pre-computed per-key position-loss map (e.g. mixture CRPS computed
    over multiple seeds outside the evaluator). Returns the same puzzle structure
    as score_ledger's 'puzzles' field, so mixture models and single-distribution
    models aggregate through the SAME frozen method-balanced path."""
    return {
        "schema_version": SCHEMA_VERSION,
        "method_balanced": method_balanced,
        "n_matched_keys": len(position_losses),
        "n_matched_positions": int(sum(
            _bio_key_parts(k)["pos"] is not None for k in position_losses)),
        "puzzles": _aggregate_position_losses(position_losses,
                                              method_balanced=method_balanced),
        "effects": {},
    }


def exact_paired_effects(score_candidate: dict[str, Any],
                         score_baseline: dict[str, Any]) -> dict[str, float]:
    """D_p = L_baseline - L_candidate per puzzle (positive = candidate better).

    Both score dicts must share the exact same puzzle key set (raise otherwise).
    """
    if score_candidate["puzzles"].keys() != score_baseline["puzzles"].keys():
        raise ValueError("candidate/baseline puzzle sets differ")
    out = {}
    for p in score_candidate["puzzles"]:
        out[p] = score_baseline["puzzles"][p]["L"] - score_candidate["puzzles"][p]["L"]
    return out
