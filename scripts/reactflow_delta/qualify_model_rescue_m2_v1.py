#!/usr/bin/env python3
"""Qualify the frozen M2 20-fold screen without promoting it to confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


SCHEMA = "reactflow_delta.model_rescue_m2_qualification.v1"
BASELINE = "b1_rfd_direct_aligned"
CANDIDATES = [
    "l2_aligned_rank2",
    "sparse_delta_mdn_h0",
    "sparse_delta_mdn_h01",
]


def _load_folds(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        folds = data.get("folds")
        if not isinstance(folds, list):
            raise ValueError("result JSON does not contain a folds list")
        return folds
    files = sorted(path.glob("m2_fold_result_fold*.json"))
    if not files:
        raise FileNotFoundError(f"no per-fold result artifacts below {path}")
    return [json.loads(file.read_text(encoding="utf-8")) for file in files]


def _paired_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        ci = [None, None]
    else:
        half = float(student_t.ppf(0.975, n - 1) * arr.std(ddof=1) / math.sqrt(n))
        ci = [mean - half, mean + half]
    return {
        "n": n,
        "mean": mean,
        "ci95_descriptive": ci,
        "positive_puzzles": int((arr > 0).sum()),
        "nonnegative_puzzles": int((arr >= 0).sum()),
        "per_puzzle": arr.tolist(),
    }


def qualify(folds: list[dict[str, Any]]) -> dict[str, Any]:
    fold_ids = [int(row["outer_fold"]) for row in folds]
    unique_ids = sorted(set(fold_ids))
    complete = len(folds) == 20 and unique_ids == list(range(20))
    if len(unique_ids) != len(fold_ids):
        raise ValueError("duplicate outer fold result")
    required_models = {BASELINE, *CANDIDATES}
    for row in folds:
        missing = required_models - set(row["candidates"])
        if missing:
            raise ValueError(f"fold {row['outer_fold']} missing candidates {sorted(missing)}")

    candidate_results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        crps_effects = []
        delta_effects = []
        coverage_checks = []
        failure_checks = []
        unexpected_checks = []
        per_fold = []
        for row in sorted(folds, key=lambda x: int(x["outer_fold"])):
            baseline = row["candidates"][BASELINE]["score"]
            score = row["candidates"][candidate]["score"]
            crps_gain = float(baseline["crps"] - score["crps"])
            delta_gain = float(baseline["signed_delta_mae"] - score["signed_delta_mae"])
            crps_effects.append(crps_gain)
            delta_effects.append(delta_gain)
            coverage = float(score.get("registered_prediction_coverage", float("nan")))
            failure = float(score.get("failure_rate", float("nan")))
            unexpected = int(score.get("n_unexpected_prediction_keys", -1))
            coverage_checks.append(coverage == 1.0)
            failure_checks.append(failure == 0.0)
            unexpected_checks.append(unexpected == 0)
            per_fold.append(
                {
                    "outer_fold": int(row["outer_fold"]),
                    "held_puzzle": row["held_puzzle"],
                    "crps_gain_vs_b1": crps_gain,
                    "signed_delta_mae_gain_vs_b1": delta_gain,
                    "registered_prediction_coverage": coverage,
                    "failure_rate": failure,
                    "n_unexpected_prediction_keys": unexpected,
                }
            )
        crps = _paired_summary(crps_effects)
        delta = _paired_summary(delta_effects)
        checks = {
            "twenty_folds_complete": complete,
            "mean_crps_gain_nonnegative": crps["mean"] >= 0.0,
            "mean_signed_delta_mae_gain_nonnegative": delta["mean"] >= 0.0,
            "crps_positive_puzzles_min_10": crps["positive_puzzles"] >= 10,
            "signed_delta_mae_positive_puzzles_min_10": delta["positive_puzzles"] >= 10,
            "registered_prediction_coverage_100pct": all(coverage_checks),
            "failure_rate_zero": all(failure_checks),
            "unexpected_prediction_keys_zero": all(unexpected_checks),
        }
        candidate_results[candidate] = {
            "crps": crps,
            "signed_delta_mae": delta,
            "checks": checks,
            "status": "M2_SCREEN_ELIGIBLE" if all(checks.values()) else "M2_SCREEN_FAIL",
            "per_fold": per_fold,
        }

    sparse_eligible = any(
        candidate_results[candidate]["status"] == "M2_SCREEN_ELIGIBLE"
        for candidate in ["sparse_delta_mdn_h0", "sparse_delta_mdn_h01"]
    )
    l2_eligible = candidate_results["l2_aligned_rank2"]["status"] == "M2_SCREEN_ELIGIBLE"
    m3_families = [BASELINE]
    if l2_eligible:
        m3_families.append("l2_aligned_rank2")
    if sparse_eligible:
        m3_families.append("sparse_delta_mdn_inner_selected_lambda")
    return {
        "schema_version": SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_SCREEN_NOT_CONFIRMATION",
        "fold_integrity": {
            "n_fold_artifacts": len(folds),
            "unique_fold_ids": unique_ids,
            "complete_0_through_19": complete,
        },
        "candidate_results": candidate_results,
        "m3_eligible_families": m3_families,
        "lambda_policy": (
            "M3_INNER_SELECT_BETWEEN_0_AND_0.1"
            if sparse_eligible
            else "SPARSE_DELTA_FAMILY_EXCLUDED"
        ),
        "overall_status": (
            "M2_SCREEN_PASS" if len(m3_families) > 1 else "M2_NO_RESCUE_CANDIDATE"
        ),
        "prohibited_interpretation": [
            "PROSPECTIVE_PASS",
            "EXTERNAL_REPLICATION",
            "SOTA",
            "SELECT_LAMBDA_FROM_OUTER_M2_EFFECT",
        ],
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ReactFlow-Delta M2 20-fold screen qualification",
        "",
        f"Status: `{result['overall_status']}`. Evidence: `{result['evidence_status']}`.",
        "",
        "| candidate | CRPS gain vs B1 | CRPS positive puzzles | signed-delta MAE gain vs B1 | delta positive puzzles | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for candidate, row in result["candidate_results"].items():
        lines.append(
            f"| {candidate} | {row['crps']['mean']:+.8f} | {row['crps']['positive_puzzles']}/20 | "
            f"{row['signed_delta_mae']['mean']:+.8f} | {row['signed_delta_mae']['positive_puzzles']}/20 | "
            f"`{row['status']}` |"
        )
    lines += [
        "",
        "M3 eligible families: " + ", ".join(f"`{x}`" for x in result["m3_eligible_families"]) + ".",
        "",
        f"SparseDelta lambda policy: `{result['lambda_policy']}`.",
        "",
        "This consumed-development screen can remove failed families only. It does not establish prospective, external, mechanism, or SOTA evidence.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(_load_folds(args.input))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["overall_status"], "out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0 if result["fold_integrity"]["complete_0_through_19"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
