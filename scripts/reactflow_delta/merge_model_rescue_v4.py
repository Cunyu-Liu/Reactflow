#!/usr/bin/env python3
"""Merge a complete v4 fold universe before any scientific score is computed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "reactflow_delta.model_rescue_v4_merged.v1"
FOLD_SCHEMA = "reactflow_delta.model_rescue_v4_fold.v1"
EXPECTED_MODELS = {
    "corrected_b1",
    "v4_dual_tower_rnafm",
    "v4_dual_tower_scratch",
    "v4_rnafm_only",
    "v4_capacity_matched_sequence_null",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def merge_fold_results(input_dir: Path, *, phase: str) -> dict[str, Any]:
    if phase == "V4M3":
        pattern = "v4_fold_result_fold*_seed0.json"
        expected_seeds = [0]
    elif phase == "V4M4":
        pattern = "v4_fold_result_fold*_seed*.json"
        expected_seeds = [0, 1, 2, 3, 4]
    else:
        raise ValueError("merge phase must be V4M3 or V4M4")
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no v4 fold artifacts below {input_dir}")
    rows = [_read(path) for path in paths]
    observed = [(int(row.get("outer_fold", -1)), int(row.get("seed", -1))) for row in rows]
    if len(set(observed)) != len(observed):
        raise ValueError("duplicate v4 fold/seed result")
    expected = {(fold, seed) for seed in expected_seeds for fold in range(20)}
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(f"incomplete v4 fold universe; missing={missing[:10]} extra={extra[:10]}")
    for row in rows:
        if row.get("schema_version") != FOLD_SCHEMA:
            raise ValueError("unexpected v4 fold schema")
        if row.get("held_score_computed") is not False:
            raise ValueError("per-fold v4 runner must not compute held scores")
        if row.get("external_outcome_accessed") is not False:
            raise ValueError("v4 fold artifact reports external outcome access")
        models = row.get("models")
        if not isinstance(models, dict) or set(models) != EXPECTED_MODELS:
            raise ValueError("v4 fold does not contain the frozen model universe")
        for model_id, model in models.items():
            for field in ("prediction_artifact", "mean_checkpoint", "calibration_checkpoint"):
                artifact = Path(model.get(field, ""))
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"fold={row['outer_fold']} seed={row['seed']} model={model_id} "
                        f"missing {field}: {artifact}"
                    )
    ordered = sorted(rows, key=lambda row: (int(row["seed"]), int(row["outer_fold"])))
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "evidence_status": "DEVELOPMENT_CONSUMED_COMPLETE_UNIVERSE_UNSCORED",
        "seeds": expected_seeds,
        "outer_folds": list(range(20)),
        "model_universe": sorted(EXPECTED_MODELS),
        "folds": ordered,
        "merge_integrity": {
            "n_fold_seed_rows": len(ordered),
            "complete_fold_seed_universe": True,
            "unique_fold_seed_rows": True,
            "all_referenced_artifacts_present": True,
            "per_fold_held_scores_absent": True,
            "partial_scores_inspected_before_merge": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--phase", choices=["V4M3", "V4M4"], required=True)
    args = parser.parse_args(argv)
    merged = merge_fold_results(args.input_dir, phase=args.phase)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": f"{args.phase}_COMPLETE_UNSCORED_MERGE_PASS",
                "n_fold_seed_rows": len(merged["folds"]),
                "result": str(args.out_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
