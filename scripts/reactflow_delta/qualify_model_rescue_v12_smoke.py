#!/usr/bin/env python3
"""Engineering-only qualifier for the two-fold V12 real-data smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.reactflow_delta.model_rescue_v12 import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_model_rescue_v12 import FOLD_SCHEMA, INNER_LEDGER_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v12_smoke_qualification.v1"


def qualify(out_dir: Path, seed: int = 0) -> dict[str, object]:
    rows = []
    for fold in (0, 1):
        path = out_dir / f"v12_fold_result_fold{fold}_seed{seed}.json"
        if not path.is_file():
            raise ValueError(f"V12 smoke is missing fold{fold}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != "V12M2":
            raise ValueError("V12 smoke fold schema or phase mismatch")
        invariants = row.get("invariants", {})
        required = {
            "inner_crossfit_complete": True,
            "outer_held_target_used_for_gate_fit": False,
            "method_used_as_gate_input": False,
            "parent_v11_exact_replay": True,
            "gate_range_pass": True,
            "candidate_distribution_median_fixed": True,
            "prediction_only_artifact": True,
            "registered_prediction_coverage": 1.0,
            "failure_rate": 0.0,
            "unexpected_keys": 0,
            "partial_score_inspected": False,
            "external_outcome_accessed": False,
        }
        if any(invariants.get(name) != value for name, value in required.items()):
            raise ValueError("V12 smoke invariant failed")
        ledger = json.loads(Path(row["inner_crossfit_ledger"]).read_text())
        if ledger.get("schema_version") != INNER_LEDGER_SCHEMA:
            raise ValueError("V12 smoke inner ledger schema mismatch")
        if ledger.get("outer_train_puzzles_covered_once") is not True:
            raise ValueError("V12 smoke inner coverage failed")
        if ledger.get("target_values_stored") is not False:
            raise ValueError("V12 smoke inner ledger stored target values")
        with np.load(Path(row["prediction_artifact"]), allow_pickle=True) as handle:
            if str(handle["schema_version"].item()) != PREDICTION_SCHEMA:
                raise ValueError("V12 smoke prediction schema mismatch")
            prohibited = {"target", "target_error", "target_mask", "score", "crps", "signed_delta_mae"}
            if prohibited & set(handle.files):
                raise ValueError("V12 smoke prediction contains targets or scores")
            if not np.array_equal(handle["keys"], handle["biological_scoring_key"]):
                raise ValueError("V12 smoke biological key columns differ")
            if not np.isfinite(handle["candidate_point"]).all():
                raise ValueError("V12 smoke candidate point is non-finite")
            if not ((handle["gate_value"] > 0) & (handle["gate_value"] < 1)).all():
                raise ValueError("V12 smoke gate range failed")
        rows.append(row)
    return {
        "schema_version": SCHEMA,
        "status": "V12M2_ENGINEERING_SMOKE_PASS",
        "evidence_status": "ENGINEERING_SMOKE_ONLY",
        "folds": [0, 1],
        "seed": seed,
        "scientific_score_computed": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = qualify(args.out_dir, args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
