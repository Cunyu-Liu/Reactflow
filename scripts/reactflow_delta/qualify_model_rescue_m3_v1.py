#!/usr/bin/env python3
"""Apply the frozen dual-primary internal gate to M3 nested OOF results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from scripts.reactflow_delta.model_rescue_gate_v1 import resolve_internal_gate


SCHEMA = "reactflow_delta.model_rescue_m3_qualification.v1"


def _load_folds(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["folds"]
    files = sorted(path.glob("m3_fold_result_fold*.json"))
    if not files:
        raise FileNotFoundError(f"no M3 fold artifacts below {path}")
    return [json.loads(file.read_text(encoding="utf-8")) for file in files]


def _ci(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        low = high = float("nan")
    else:
        half = float(student_t.ppf(0.975, n - 1) * arr.std(ddof=1) / math.sqrt(n))
        low, high = mean - half, mean + half
    return {"n": n, "mean": mean, "ci95": [low, high], "positive_puzzles": int((arr > 0).sum())}


def _leave_one_positive(values: list[float]) -> bool:
    arr = np.asarray(values, dtype=float)
    return len(arr) >= 2 and all(np.delete(arr, i).mean() > 0 for i in range(len(arr)))


def qualify(folds: list[dict[str, Any]]) -> dict[str, Any]:
    folds = sorted(folds, key=lambda row: int(row["outer_fold"]))
    fold_ids = [int(row["outer_fold"]) for row in folds]
    complete = len(folds) == 20 and fold_ids == list(range(20))
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("duplicate M3 outer fold")

    crps_effects = []
    delta_effects = []
    wt_effects = []
    baseline_crps = []
    baseline_delta = []
    coverages = []
    failures = []
    calibration_candidate = []
    calibration_comparator = []
    selected_candidates = Counter()
    selected_comparators = Counter()
    per_fold = []
    for row in folds:
        selected = row["selection"]["selected_candidate"]
        comparator = row["selection"]["selected_comparator"]
        selected_candidates[selected] += 1
        selected_comparators[comparator] += 1
        candidate_score = row["outer_scores"][selected]
        comparator_score = row["outer_scores"][comparator]
        crps_gain = float(row["effects"]["crps_gain"])
        delta_gain = float(row["effects"]["signed_delta_mae_gain"])
        wt_gain = float(row["effects"]["signed_delta_mae_gain_vs_wt_anchor"])
        crps_effects.append(crps_gain)
        delta_effects.append(delta_gain)
        wt_effects.append(wt_gain)
        baseline_crps.append(float(comparator_score["crps"]))
        baseline_delta.append(float(comparator_score["signed_delta_mae"]))
        coverages.append(float(candidate_score["registered_prediction_coverage"]))
        failures.append(float(candidate_score["failure_rate"]))
        candidate_cal = 0.5 * (
            abs(float(candidate_score["coverage68"]) - 0.68)
            + abs(float(candidate_score["coverage95"]) - 0.95)
        )
        comparator_cal = 0.5 * (
            abs(float(comparator_score["coverage68"]) - 0.68)
            + abs(float(comparator_score["coverage95"]) - 0.95)
        )
        calibration_candidate.append(candidate_cal)
        calibration_comparator.append(comparator_cal)
        per_fold.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "held_puzzle": row["held_puzzle"],
                "selected_candidate": selected,
                "selected_comparator": comparator,
                "crps_gain": crps_gain,
                "signed_delta_mae_gain": delta_gain,
                "signed_delta_mae_gain_vs_wt_anchor": wt_gain,
                "registered_prediction_coverage": coverages[-1],
                "failure_rate": failures[-1],
            }
        )

    crps = _ci(crps_effects)
    delta = _ci(delta_effects)
    wt = _ci(wt_effects)
    mean_baseline_crps = float(np.mean(baseline_crps))
    mean_baseline_delta = float(np.mean(baseline_delta))
    normalized = 0.5 * np.asarray(crps_effects) / max(mean_baseline_crps, 1e-12)
    normalized += 0.5 * np.asarray(delta_effects) / max(mean_baseline_delta, 1e-12)
    max_fraction = float(np.max(np.abs(normalized)) / max(np.sum(np.abs(normalized)), 1e-12))
    calibration_worsening_pp = 100.0 * (
        float(np.mean(calibration_candidate)) - float(np.mean(calibration_comparator))
    )
    gate = resolve_internal_gate(
        crps_gain=crps["mean"],
        crps_ci_low=crps["ci95"][0],
        baseline_crps=mean_baseline_crps,
        delta_mae_gain=delta["mean"],
        delta_mae_ci_low=delta["ci95"][0],
        baseline_delta_mae=mean_baseline_delta,
        beats_wt_anchor=wt["mean"] > 0 and wt["positive_puzzles"] >= 12,
        crps_positive_puzzles=crps["positive_puzzles"],
        delta_mae_positive_puzzles=delta["positive_puzzles"],
        leave_one_puzzle_positive=_leave_one_positive(crps_effects) and _leave_one_positive(delta_effects),
        max_single_puzzle_fraction=max_fraction,
        prediction_coverage=min(coverages) if coverages else 0.0,
        failure_rate=max(failures) if failures else 1.0,
        coverage_error_worsening_pp=calibration_worsening_pp,
    )
    if not complete:
        gate["checks"]["twenty_outer_folds_complete"] = False
        gate["status"] = "METHOD_RESCUE_FAIL"
        gate["next_route"] = "M3_INCOMPLETE_DO_NOT_DECIDE"
    else:
        gate["checks"]["twenty_outer_folds_complete"] = True
        if not all(gate["checks"].values()):
            gate["status"] = "METHOD_RESCUE_FAIL"
            gate["next_route"] = "BENCHMARK_ROUTE_LOCKED"
    return {
        "schema_version": SCHEMA,
        "evidence_status": "POST_HOC_DEVELOPMENT_NESTED_NOT_CONFIRMATION",
        "fold_integrity": {"fold_ids": fold_ids, "complete_0_through_19": complete},
        "crps_effect": crps,
        "signed_delta_mae_effect": delta,
        "signed_delta_mae_vs_wt_anchor": wt,
        "mean_comparator_crps": mean_baseline_crps,
        "mean_comparator_signed_delta_mae": mean_baseline_delta,
        "selection_frequency": {
            "candidate": dict(selected_candidates),
            "comparator": dict(selected_comparators),
        },
        "max_single_puzzle_effect_fraction": max_fraction,
        "calibration_error_worsening_pp": calibration_worsening_pp,
        "gate": gate,
        "per_fold": per_fold,
        "publication_boundary": {
            "sota": "NOT_ESTABLISHED",
            "external": "NOT_ESTABLISHED",
            "mechanism": "NOT_ESTABLISHED",
            "allowed_if_pass": "POST_HOC_DEVELOPMENT_METHOD_RESCUE_PASS",
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    crps = result["crps_effect"]
    delta = result["signed_delta_mae_effect"]
    wt = result["signed_delta_mae_vs_wt_anchor"]
    lines = [
        "# ReactFlow-Delta M3 nested internal qualification",
        "",
        f"Gate: `{result['gate']['status']}`; next route: `{result['gate']['next_route']}`.",
        "",
        f"- CRPS gain: {crps['mean']:+.8f}, 95% CI [{crps['ci95'][0]:+.8f}, {crps['ci95'][1]:+.8f}], positive puzzles {crps['positive_puzzles']}/20.",
        f"- signed-delta MAE gain: {delta['mean']:+.8f}, 95% CI [{delta['ci95'][0]:+.8f}, {delta['ci95'][1]:+.8f}], positive puzzles {delta['positive_puzzles']}/20.",
        f"- signed-delta MAE gain vs WT anchor: {wt['mean']:+.8f}, positive puzzles {wt['positive_puzzles']}/20.",
        "",
        "## Gate checks",
        "",
    ]
    for key, value in result["gate"]["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "This is nested but consumed-development evidence. It cannot establish prospective performance, external replication, mechanism, or SOTA.",
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
    print(json.dumps({"status": result["gate"]["status"], "next_route": result["gate"]["next_route"]}, indent=2))
    return 0 if result["fold_integrity"]["complete_0_through_19"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
