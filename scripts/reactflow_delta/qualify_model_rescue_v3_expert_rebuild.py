#!/usr/bin/env python3
"""Qualify the complete corrected-coordinate R3C3 expert rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.run_model_rescue_v3_expert_rebuild import (
    PREDICTION_SCHEMA,
    SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


QUALIFICATION_SCHEMA = (
    "reactflow_delta.model_rescue_v3_corrected_expert_qualification.v1"
)


def check_fold_result(
    row: dict[str, Any], expected_keys: set[str]
) -> dict[str, bool]:
    prediction_path = Path(row["expert_prediction_artifact"])
    checks = {
        "result_schema": row.get("schema_version") == SCHEMA,
        "seed_zero": int(row.get("seed", -1)) == 0,
        "epochs_forty": int(row.get("epochs", -1)) == 40,
        "held_score_not_computed": row.get("held_score_computed") is False,
        "external_outcome_not_accessed": row.get("external_outcome_accessed") is False,
        "b1_checkpoint_exists": Path(row["b1_checkpoint"]).is_file(),
        "meanaligned_checkpoint_exists": Path(
            row["meanaligned_checkpoint"]
        ).is_file(),
        "prediction_artifact_exists": prediction_path.is_file(),
        "b1_history_complete_finite": len(row.get("b1_train_loss", [])) == 40
        and bool(np.isfinite(np.asarray(row.get("b1_train_loss", []))).all()),
        "meanaligned_history_complete_finite": len(
            row.get("meanaligned_train_loss", [])
        )
        == 40
        and bool(
            np.isfinite(np.asarray(row.get("meanaligned_train_loss", []))).all()
        ),
    }
    if not prediction_path.is_file():
        return checks
    with np.load(prediction_path, allow_pickle=True) as stored:
        keys = [str(value) for value in stored["keys"]]
        required = {
            "schema_version",
            "keys",
            "b1_delta_mean",
            "meanaligned_delta_mean",
            "outer_fold",
            "seed",
        }
        checks.update(
            {
                "prediction_schema": str(stored["schema_version"])
                == PREDICTION_SCHEMA,
                "prediction_only_fields": set(stored.files) == required,
                "prediction_keys_unique": len(keys) == len(set(keys)),
                "prediction_key_universe_exact": set(keys) == expected_keys,
                "prediction_means_finite": bool(
                    np.isfinite(stored["b1_delta_mean"]).all()
                    and np.isfinite(stored["meanaligned_delta_mean"]).all()
                ),
                "prediction_fold_exact": bool(
                    (stored["outer_fold"] == int(row["outer_fold"])).all()
                ),
                "prediction_seed_zero": bool((stored["seed"] == 0).all()),
            }
        )
    return checks


def qualify(input_dir: Path, m2_csv: Path) -> dict[str, Any]:
    universe = M2Universe(m2_csv)
    universe.build()
    records = universe.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=20260813
    )
    fold_by_id = {int(fold.outer_fold): fold for fold in split["folds"]}
    paths = sorted(input_dir.glob("v3_corrected_expert_fold_result_fold*_seed0.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ids = [int(row["outer_fold"]) for row in rows]
    universe_complete = sorted(ids) == list(range(20)) and len(ids) == len(set(ids))
    fold_results = []
    if universe_complete:
        for row in sorted(rows, key=lambda value: int(value["outer_fold"])):
            fold_id = int(row["outer_fold"])
            fold = fold_by_id[fold_id]
            held_records = [
                record for record in records if record.puzzle == fold.held_puzzle
            ]
            expected = {
                _bio_key(universe, record, position)
                for record in held_records
                for position in range(
                    len(universe.get_construct(record.construct_id).sequence)
                )
            }
            checks = check_fold_result(row, expected)
            fold_results.append(
                {
                    "outer_fold": fold_id,
                    "held_puzzle": fold.held_puzzle,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    passed = universe_complete and all(row["passed"] for row in fold_results)
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_EXPERT_REBUILD_ONLY",
        "fold_universe_complete": universe_complete,
        "folds": fold_results,
        "overall_status": (
            "R3C3_CORRECTED_EXPERT_REBUILD_PASS"
            if passed
            else "R3C3_CORRECTED_EXPERT_REBUILD_FAIL"
        ),
        "r3m3_authorized": passed,
        "scores_read": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = qualify(args.input_dir, args.m2_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["overall_status"]}, indent=2))
    return 0 if result["overall_status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
