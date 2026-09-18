#!/usr/bin/env python3
"""DeltaFlow qualifier-once (Task 9.2): mechanical F2 + F3 judgement.

Executes the frozen decision rules exactly once on a complete score
artifact (score_deltaflow.py output).  No re-scoring, no threshold
adjustment; every number and rule is traceable to the Task 7 freeze
commit (7bc771e) and the Phase 1 freeze table F2/F3.

F2 primary-gate conditions (all four metrics, candidate vs null):
  1. assembly-level paired 95% CI of the per-fold improvement direction
     (N=20 folds, t-distribution df=19, t_0.975 = 2.093024);
  2. 2/2 seeds individually positive (per-seed 20-fold aggregate; amended 2026-09-19);
  3. assembly-level aggregate improvement percent >= MDE tier.

Both MDE tiers are computed and registered: independent tier
(sigma_assembly = sigma_d / sqrt(2), primary; amended 2026-09-19) and conservative tier
(rho=1 audit); Step 2 measures realized rho/sigma_assembly and the
realized-tier MDE; Step 3 grades PASS (strong) / PASS with caveat /
FAIL.

F3 joint-structure metrics: verdicts come from the per-seed scorer
outputs (already classified by deltaflow_joint_metrics against the F3
constants); the qualifier reports the median-across-seeds value and its
grade for the five joint metrics, separately from the primary gate
(MARGINAL never counts toward any gate).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "reactflow_delta.deltaflow_qualification.v1"
SCORE_SCHEMA = "reactflow_delta.deltaflow_score.v1"
T975_DF19 = 2.093024
SEED_UNIVERSE = (0, 1)  # 2-seed final per amendment 2026-09-19 (no further seeds will be run)
K_INDEPENDENT = 1.0 / math.sqrt(len(SEED_UNIVERSE))
K_CONSERVATIVE = 1.0

# independent-tier MDE amended for 2 seeds: frozen values x sqrt(5/2) = 1.58114
# (deltaflow_amendment_2seed_final.md, 2026-09-19); conservative tier (rho=1) unchanged.
PRIMARY_METRICS = (
    ("signed_delta_mae", 1.85, 2.61),
    ("point_absolute_delta_mae", 2.74, 3.86),
    ("crps", 1.14, 1.62),
    ("distribution_absolute_delta_mae", 1.28, 1.80),
)
CANDIDATE_PREFIX = "flow_candidate"
NULL_PREFIX = "flow_null"
JOINT_GATE_KEYS = (
    ("shape_pearson_median", "shape"),
    ("top1_rate", "localization_top1"),
    ("top3_rate", "localization_top3"),
    ("energy_gain_percent_median", "energy"),
    ("pca2_median", "coverage"),
)


def _load_score(path: Path) -> dict[str, Any]:
    score = json.loads(path.read_text(encoding="utf-8"))
    if score.get("schema_version") != SCORE_SCHEMA:
        raise ValueError(f"score schema mismatch in {path}")
    if score.get("status") != "DELTAFLOW_COMPLETE_SCORE_PASS":
        raise ValueError("score artifact is not a complete pass")
    if len(score["scores"]) != 20:
        raise ValueError("score artifact must cover folds 0-19")
    return score


def _metric_values(score: dict[str, Any], label: str, suffix: str) -> np.ndarray:
    values = np.asarray(
        [fold[f"{label}_{suffix}"] for fold in score["scores"]], dtype=np.float64
    )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"nonfinite or nonpositive metric values for {label}_{suffix}")
    return values


def _paired_improvement_percent(
    candidate: np.ndarray, null: np.ndarray
) -> np.ndarray:
    return (null - candidate) / null * 100.0


def _t_ci(improvements: np.ndarray) -> tuple[float, float, float]:
    mean = float(improvements.mean())
    std = float(improvements.std(ddof=1))
    half = T975_DF19 * std / math.sqrt(len(improvements))
    return mean, mean - half, mean + half


def _primary_gate(
    score: dict[str, Any], metric: str, mde_independent: float, mde_conservative: float
) -> dict[str, Any]:
    candidate = _metric_values(score, CANDIDATE_PREFIX, metric)
    null = _metric_values(score, NULL_PREFIX, metric)
    improvements = _paired_improvement_percent(candidate, null)
    mean, low, high = _t_ci(improvements)
    per_seed_positive = {}
    per_seed_means = {}
    for seed in SEED_UNIVERSE:
        seed_candidate = _metric_values(
            score, f"{CANDIDATE_PREFIX}_seed{seed}", metric
        )
        seed_null = _metric_values(score, f"{NULL_PREFIX}_seed{seed}", metric)
        seed_improvement = _paired_improvement_percent(seed_candidate, seed_null)
        per_seed_means[seed] = float(seed_improvement.mean())
        per_seed_positive[seed] = bool(seed_improvement.mean() > 0.0)
    n_positive = sum(per_seed_positive.values())

    sigma_d = float(improvements.std(ddof=1))
    sigma_assembly_independent = sigma_d * K_INDEPENDENT
    sigma_assembly_conservative = sigma_d * K_CONSERVATIVE
    realized = float(np.std([per_seed_means[s] for s in SEED_UNIVERSE], ddof=1))
    kappa = 2.801585
    mde_realized = kappa * realized / math.sqrt(20)
    baseline = float(null.mean())

    def conditions(mde: float) -> dict[str, Any]:
        return {
            "ci_lower_positive": bool(low > 0.0),
            "seeds_positive": f"{n_positive}/{len(SEED_UNIVERSE)}",
            "seeds_positive_pass": bool(n_positive >= len(SEED_UNIVERSE)),
            "aggregate_improvement_percent": mean,
            "mde_percent": mde,
            "mde_pass": bool(mean >= mde),
        }

    independent = conditions(mde_independent)
    conservative = conditions(mde_conservative)
    realized_conditions = {
        **conditions(mde_realized / baseline * 100.0 * math.sqrt(20) / math.sqrt(20)),
    }
    realized_conditions["mde_percent"] = float(
        mde_realized / baseline * 100.0
    )
    independent_pass = (
        independent["ci_lower_positive"]
        and independent["seeds_positive_pass"]
        and independent["mde_pass"]
    )
    conservative_pass = (
        conservative["ci_lower_positive"]
        and conservative["seeds_positive_pass"]
        and conservative["mde_pass"]
    )
    realized_pass = (
        realized_conditions["ci_lower_positive"]
        and realized_conditions["seeds_positive_pass"]
        and realized_conditions["mde_pass"]
    )
    if not independent_pass:
        verdict = "FAIL"
    elif realized_pass:
        verdict = "PASS (strong)"
    else:
        verdict = "PASS with caveat"
    return {
        "metric": metric,
        "candidate_mean": float(candidate.mean()),
        "null_mean": float(null.mean()),
        "improvement_percent": {
            "mean": mean,
            "ci95": [low, high],
            "per_fold": [float(v) for v in improvements],
        },
        "per_seed": {
            str(seed): {
                "aggregate_improvement_percent": per_seed_means[seed],
                "positive": per_seed_positive[seed],
            }
            for seed in SEED_UNIVERSE
        },
        "realized_sigma_assembly": realized,
        "sigma_d": sigma_d,
        "sigma_assembly_independent": sigma_assembly_independent,
        "sigma_assembly_conservative": sigma_assembly_conservative,
        "tiers": {
            "independent": independent,
            "conservative": conservative,
            "realized": realized_conditions,
        },
        "independent_pass": independent_pass,
        "conservative_pass": conservative_pass,
        "realized_pass": realized_pass,
        "verdict": verdict,
    }


def _joint_summary(score: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, label in JOINT_GATE_KEYS:
        per_seed_values = []
        for fold in score["scores"]:
            for seed_entry in fold["joint_metrics_by_seed"].values():
                delta = seed_entry["deltas"][key]
                if isinstance(delta, (int, float)) and math.isfinite(delta):
                    per_seed_values.append(float(delta))
        if per_seed_values:
            output[label] = {
                "candidate_minus_null_median": float(np.median(per_seed_values)),
                "n_values": len(per_seed_values),
            }
    return output


def qualify(score_path: Path) -> dict[str, Any]:
    score = _load_score(score_path)
    primary = [
        _primary_gate(score, metric, mde_ind, mde_cons)
        for metric, mde_ind, mde_cons in PRIMARY_METRICS
    ]
    overall = "PASS" if all(row["verdict"].startswith("PASS") for row in primary) else "FAIL"
    return {
        "schema_version": SCHEMA,
        "status": "DELTAFLOW_QUALIFIER_ONCE_EXECUTED",
        "score_artifact": str(score_path.resolve()),
        "primary_metrics": primary,
        "primary_verdict": overall,
        "joint_metrics": _joint_summary(score),
        "joint_metrics_note": (
            "F3 joint-structure metrics are reported per the frozen scorer; "
            "they are separate from the primary gate and MARGINAL never counts."
        ),
        "rule_source": "Task 7 freeze commit 7bc771e / Phase 1 freeze table F2-F3 / 2-seed amendment deltaflow_amendment_2seed_final (2026-09-19)",
        "thresholds_frozen": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(args.score_json)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_verdict": result["primary_verdict"],
                "result": str(args.out_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
