#!/usr/bin/env python3
"""DeltaFlow joint-structure metric scorer (Task 6.5, freeze table F3).

All estimator semantics are frozen to the Phase 1 Task 2 coupling audit so
that the F3 anchors stay directly comparable:

- distal positions: |receiver - edit| > 2 AND the observed delta is valid;
- shape correlation: per-mutant Pearson (gated) and Spearman (reported)
  between the ensemble-mean predicted delta profile and the observed delta
  profile over that mutant's distal positions, pairwise-complete with a
  minimum of ten positions (Phase 1 min-obs rule);
- distal localization: the observed argmax of |delta| over distal
  positions must fall within the top-1/top-3 positions of the ensemble
  mean |prediction|; the random baselines are mean(1/n_distal) and
  mean(min(3, n_distal)/n_distal) exactly as in Phase 1;
- energy score: proper multivariate score ES = mean_x||y - x|| -
  0.5*mean_{x != x'}||x - x'|| over the distal positions of one mutant,
  with the forecast ensemble F being the model's joint draws; the null
  re-uses the same draws with columns permuted independently (marginals
  kept, cross-position joint destroyed), averaged over seeded repeats;
  the reported gain is (ES_null / ES_obs - 1) * 100 percent;
- joint coverage: PCA top-2 variance fraction of the stacked generated
  ensemble (rows = mutant x draw sample profiles restricted to that
  mutant's distal-valid positions, zero elsewhere, column-centered),
  pooled as the median over constructs.

Gate values follow freeze table F3 verbatim: shape PASS 0.60 / MARGINAL
0.546; localization PASS 3x / MARGINAL 2x the recomputed random baseline;
energy PASS >= 1.0 percent median gain; coverage PASS in [0.30, 0.50].
MARGINAL never counts toward any gate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


JOINT_METRICS_SCHEMA = "reactflow_delta.deltaflow_joint_metrics.v1"

DISTAL_K = 2
MIN_CORRELATION_PAIRS = 10
MIN_LOCALIZATION_POSITIONS = 1
MIN_ENERGY_POSITIONS = 2
MIN_ENERGY_DRAWS = 2
DEFAULT_B_PERM = 50

SHAPE_PASS = 0.60
SHAPE_MARGINAL = 0.546
LOCALIZATION_PASS_MULTIPLIER = 3.0
LOCALIZATION_MARGINAL_MULTIPLIER = 2.0
ENERGY_PASS_PERCENT = 1.0
PCA2_PASS_RANGE = (0.30, 0.50)

SHAPE_FLOOR_ANCHOR = 0.546
RANDOM_TOP1_ANCHOR = 0.0219
RANDOM_TOP3_ANCHOR = 0.0362
ENERGY_ANCHOR_PERCENT = 1.05
PCA2_OBSERVED_ANCHOR = 0.397
PCA2_NULL_ANCHOR = 0.215


def _finite_or_nan(value: float) -> float:
    if value is None:
        return float("nan")
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        stop = start
        while (
            stop + 1 < len(sorted_values)
            and sorted_values[stop + 1] == sorted_values[start]
        ):
            stop += 1
        average = (start + stop) / 2.0 + 1.0
        ranks[order[start : stop + 1]] = average
        start = stop + 1
    return ranks


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else float("nan")


def pearson_masked(x: np.ndarray, y: np.ndarray, *, min_pairs: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < min_pairs:
        return float("nan")
    return _corrcoef(x[keep], y[keep])


def spearman_masked(x: np.ndarray, y: np.ndarray, *, min_pairs: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < min_pairs:
        return float("nan")
    return _corrcoef(_average_ranks(x[keep]), _average_ranks(y[keep]))


def distal_positions(
    edit_position: int, length: int, valid: np.ndarray, *, k: int = DISTAL_K
) -> np.ndarray:
    positions = np.arange(length)
    return (np.abs(positions - int(edit_position)) > k) & np.asarray(valid, dtype=bool)


def mutant_shape_correlation(
    samples: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    predicted = np.asarray(samples, dtype=np.float64).mean(axis=0)
    observed = np.asarray(observed, dtype=np.float64)
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(positions) < MIN_CORRELATION_PAIRS:
        return float("nan"), float("nan")
    pearson = _corrcoef(predicted[positions], observed[positions])
    spearman = _corrcoef(
        _average_ranks(predicted[positions]), _average_ranks(observed[positions])
    )
    return pearson, spearman


def mutant_localization(
    samples: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> tuple[int, int, int]:
    predicted = np.asarray(samples, dtype=np.float64).mean(axis=0)
    observed = np.asarray(observed, dtype=np.float64)
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    n_distal = len(positions)
    if n_distal < MIN_LOCALIZATION_POSITIONS:
        return 0, 0, n_distal
    predicted_magnitude = np.abs(predicted[positions])
    observed_magnitude = np.abs(observed[positions])
    observed_rank = int(np.argmax(observed_magnitude))
    observed_argmax = positions[observed_rank]
    predicted_order = positions[np.argsort(-predicted_magnitude, kind="mergesort")]
    top1_hit = int(predicted_order[0] == observed_argmax)
    top3_hit = int(observed_argmax in set(predicted_order[:3].tolist()))
    return top1_hit, top3_hit, n_distal


def _energy_from_pairwise(distance_to_y: np.ndarray, pairwise: np.ndarray) -> float:
    n_draws = pairwise.shape[0]
    spread = (pairwise.sum() - np.trace(pairwise)) / (n_draws * (n_draws - 1))
    return float(distance_to_y.mean() - 0.5 * spread)


def energy_score_mutant(
    samples: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    *,
    b_perm: int,
    seed: int,
) -> dict[str, float]:
    forecast = np.asarray(samples, dtype=np.float64)[:, np.asarray(mask, dtype=bool)]
    y = np.asarray(observed, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    n_draws, n_positions = forecast.shape
    if n_draws < MIN_ENERGY_DRAWS or n_positions < MIN_ENERGY_POSITIONS:
        return {
            "es_obs": float("nan"),
            "es_null": float("nan"),
            "gain_percent": float("nan"),
        }
    distance_to_y = np.linalg.norm(y[None, :] - forecast, axis=1)
    pairwise = np.linalg.norm(
        forecast[:, None, :] - forecast[None, :, :], axis=-1
    )
    es_obs = _energy_from_pairwise(distance_to_y, pairwise)
    rng = np.random.default_rng(int(seed))
    null_values = []
    for _ in range(int(b_perm)):
        permuted = np.empty_like(forecast)
        for column in range(n_positions):
            permuted[:, column] = forecast[rng.permutation(n_draws), column]
        distance_null = np.linalg.norm(y[None, :] - permuted, axis=1)
        pairwise_null = np.linalg.norm(
            permuted[:, None, :] - permuted[None, :, :], axis=-1
        )
        null_values.append(_energy_from_pairwise(distance_null, pairwise_null))
    es_null = float(np.mean(null_values))
    if es_obs > 0.0:
        gain_percent = (es_null / es_obs - 1.0) * 100.0
    else:
        gain_percent = float("nan")
    return {
        "es_obs": float(es_obs),
        "es_null": es_null,
        "gain_percent": _finite_or_nan(gain_percent),
    }


def pca2_variance_fraction(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return float("nan")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = float((singular**2).sum())
    if total <= 0.0:
        return float("nan")
    return float((singular[:2] ** 2).sum() / total)


def construct_pca2(
    samples_by_mutant: list[np.ndarray],
    masks_by_mutant: list[np.ndarray],
    length: int,
) -> float:
    rows = []
    for samples, mask in zip(samples_by_mutant, masks_by_mutant):
        samples = np.asarray(samples, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        for draw in range(samples.shape[0]):
            row = np.zeros(length, dtype=np.float64)
            row[mask] = samples[draw][mask]
            rows.append(row)
    if len(rows) < 2:
        return float("nan")
    return pca2_variance_fraction(np.stack(rows, axis=0))


def classify_shape(value: float) -> str:
    if not math.isfinite(value):
        return "FAIL"
    if value >= SHAPE_PASS:
        return "PASS"
    if value >= SHAPE_MARGINAL:
        return "MARGINAL"
    return "FAIL"


def classify_localization(rate: float, baseline: float) -> str:
    if not math.isfinite(rate) or baseline <= 0.0:
        return "FAIL"
    if rate >= LOCALIZATION_PASS_MULTIPLIER * baseline:
        return "PASS"
    if rate >= LOCALIZATION_MARGINAL_MULTIPLIER * baseline:
        return "MARGINAL"
    return "FAIL"


def classify_energy(gain_percent: float) -> str:
    if not math.isfinite(gain_percent):
        return "FAIL"
    return "PASS" if gain_percent >= ENERGY_PASS_PERCENT else "FAIL"


def classify_pca2(value: float) -> str:
    if not math.isfinite(value):
        return "FAIL"
    low, high = PCA2_PASS_RANGE
    return "PASS" if low <= value <= high else "FAIL"


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(np.median(finite))


def _summary(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"n": 0, "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "n": len(finite),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
    }


def score_arm(
    *,
    arm: str,
    mutants: list[dict[str, Any]],
    b_perm: int = DEFAULT_B_PERM,
    seed: int = 0,
) -> dict[str, Any]:
    """Score one arm over held mutants.

    Each mutant dict must carry:
      construct_id, edit_position, length,
      observed_delta  [L] (NaN outside qualified),
      samples         [S, L] joint draws of the full delta,
      distal_mask     [L] bool (distal AND observed-valid).
    """

    pearson_values: list[float] = []
    spearman_values: list[float] = []
    top1_hits = 0
    top3_hits = 0
    localization_trials = 0
    random_top1_weighted = 0.0
    random_top3_weighted = 0.0
    localization_mutants = 0
    energy_gains: list[float] = []
    energy_mutants = 0
    per_construct_pca2: list[float] = []
    by_construct: dict[str, dict[str, Any]] = {}
    for index, mutant in enumerate(mutants):
        construct_id = str(mutant["construct_id"])
        samples = np.asarray(mutant["samples"], dtype=np.float64)
        observed = np.asarray(mutant["observed_delta"], dtype=np.float64)
        distal = np.asarray(mutant["distal_mask"], dtype=bool)
        pearson, spearman = mutant_shape_correlation(samples, observed, distal)
        if math.isfinite(pearson):
            pearson_values.append(pearson)
        if math.isfinite(spearman):
            spearman_values.append(spearman)
        top1_hit, top3_hit, n_distal = mutant_localization(samples, observed, distal)
        top1_hits += top1_hit
        top3_hits += top3_hit
        localization_trials += 1 if n_distal >= MIN_LOCALIZATION_POSITIONS else 0
        if n_distal >= MIN_LOCALIZATION_POSITIONS:
            localization_mutants += 1
            random_top1_weighted += 1.0 / n_distal
            random_top3_weighted += min(3.0, n_distal) / n_distal
        energy = energy_score_mutant(
            samples, observed, distal, b_perm=b_perm, seed=seed * 100_003 + index
        )
        if math.isfinite(energy["gain_percent"]):
            energy_gains.append(energy["gain_percent"])
            energy_mutants += 1
        slot = by_construct.setdefault(
            construct_id,
            {
                "samples": [],
                "masks": [],
                "length": int(mutant["length"]),
                "n_mutants": 0,
            },
        )
        slot["samples"].append(samples)
        slot["masks"].append(distal)
        slot["n_mutants"] += 1
    for construct_id, slot in by_construct.items():
        value = construct_pca2(slot["samples"], slot["masks"], slot["length"])
        if math.isfinite(value):
            per_construct_pca2.append(value)
    if localization_mutants:
        random_top1 = random_top1_weighted / localization_mutants
        random_top3 = random_top3_weighted / localization_mutants
        top1_rate = top1_hits / localization_mutants
        top3_rate = top3_hits / localization_mutants
    else:
        random_top1 = float("nan")
        random_top3 = float("nan")
        top1_rate = float("nan")
        top3_rate = float("nan")
    shape_median = _median(pearson_values)
    spearman_median = _median(spearman_values)
    energy_median = _median(energy_gains)
    pca2_median = _median(per_construct_pca2)
    return {
        "schema_version": JOINT_METRICS_SCHEMA,
        "arm": arm,
        "n_mutants_total": len(mutants),
        "shape": {
            "pearson_median": shape_median,
            "pearson_quartiles": _summary(pearson_values),
            "spearman_median": spearman_median,
            "spearman_quartiles": _summary(spearman_values),
            "floor_anchor": SHAPE_FLOOR_ANCHOR,
            "verdict": classify_shape(shape_median),
        },
        "localization": {
            "top1_rate": top1_rate,
            "top3_rate": top3_rate,
            "random_top1_baseline": random_top1,
            "random_top3_baseline": random_top3,
            "n_evaluated_mutants": localization_mutants,
            "anchor_random_top1": RANDOM_TOP1_ANCHOR,
            "anchor_random_top3": RANDOM_TOP3_ANCHOR,
            "verdict_top1": classify_localization(top1_rate, random_top1),
            "verdict_top3": classify_localization(top3_rate, random_top3),
        },
        "energy": {
            "gain_percent_median": energy_median,
            "gain_percent_quartiles": _summary(energy_gains),
            "n_evaluated_mutants": energy_mutants,
            "anchor_gain_percent": ENERGY_ANCHOR_PERCENT,
            "verdict": classify_energy(energy_median),
        },
        "coverage": {
            "pca2_median": pca2_median,
            "pca2_quartiles": _summary(per_construct_pca2),
            "n_constructs": len(per_construct_pca2),
            "anchor_observed": PCA2_OBSERVED_ANCHOR,
            "anchor_null": PCA2_NULL_ANCHOR,
            "verdict": classify_pca2(pca2_median),
        },
        "gate_inputs": {
            "shape_pearson_median": shape_median,
            "top1_rate": top1_rate,
            "top3_rate": top3_rate,
            "random_top1_baseline": random_top1,
            "random_top3_baseline": random_top3,
            "energy_gain_percent_median": energy_median,
            "pca2_median": pca2_median,
        },
        "parameters": {
            "b_perm": int(b_perm),
            "seed": int(seed),
            "distal_k": DISTAL_K,
            "min_correlation_pairs": MIN_CORRELATION_PAIRS,
        },
    }


def compare_arms(candidate: dict[str, Any], null: dict[str, Any]) -> dict[str, Any]:
    if candidate["arm"] == null["arm"]:
        raise ValueError("candidate/null arms must differ")
    gate_c = candidate["gate_inputs"]
    gate_n = null["gate_inputs"]

    def _delta(name: str) -> float:
        left = float(gate_c[name])
        right = float(gate_n[name])
        if not (math.isfinite(left) and math.isfinite(right)):
            return float("nan")
        return left - right

    return {
        "schema_version": JOINT_METRICS_SCHEMA,
        "candidate": candidate,
        "null": null,
        "deltas": {
            "shape_pearson_median": _delta("shape_pearson_median"),
            "top1_rate": _delta("top1_rate"),
            "top3_rate": _delta("top3_rate"),
            "energy_gain_percent_median": _delta("energy_gain_percent_median"),
            "pca2_median": _delta("pca2_median"),
        },
        "frozen_anchors": {
            "shape_floor": SHAPE_FLOOR_ANCHOR,
            "random_top1": RANDOM_TOP1_ANCHOR,
            "random_top3": RANDOM_TOP3_ANCHOR,
            "energy_gain_percent": ENERGY_ANCHOR_PERCENT,
            "pca2_observed": PCA2_OBSERVED_ANCHOR,
            "pca2_null": PCA2_NULL_ANCHOR,
            "shape_pass": SHAPE_PASS,
            "shape_marginal": SHAPE_MARGINAL,
            "localization_pass_multiplier": LOCALIZATION_PASS_MULTIPLIER,
            "localization_marginal_multiplier": LOCALIZATION_MARGINAL_MULTIPLIER,
            "energy_pass_percent": ENERGY_PASS_PERCENT,
            "pca2_pass_range": list(PCA2_PASS_RANGE),
        },
    }
