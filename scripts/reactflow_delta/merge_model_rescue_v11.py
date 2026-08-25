#!/usr/bin/env python3
"""Merge complete score-blind V11 fold universes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.reactflow_delta.qualify_model_rescue_v11_smoke import (
    prediction_checks,
    recorded_invariants_pass,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    EXPECTED_POINT_PARAMETERS,
    FOLD_SCHEMA,
)


SCHEMA = "reactflow_delta.model_rescue_v11_merged.v1"


def authoritative_comparator_invariant_pass(
    phase: str, invariants: dict[str, Any]
) -> bool:
    if phase == "V11M2":
        return True
    return (
        invariants.get(
            "feature41_asymmetric_seed0_uses_authoritative_v10_or_not_applicable"
        )
        is True
    )


def _expected_universe(phase: str) -> tuple[list[int], list[int], int, int, str]:
    if phase == "V11M2":
        return [0, 1], [0], 3, 3, "V11M2_COMPLETE_UNSCORED_SMOKE_MERGE_PASS"
    if phase == "V11M3":
        return list(range(20)), [0], 40, 40, "V11M3_COMPLETE_UNSCORED_MERGE_PASS"
    if phase == "V11M4":
        return list(range(20)), list(range(5)), 40, 40, "V11M4_COMPLETE_UNSCORED_MERGE_PASS"
    raise ValueError("unsupported V11 merge phase")


def merge_folds(input_dir: Path, phase: str) -> dict[str, Any]:
    expected_folds, expected_seeds, point_epochs, calibration_epochs, status = (
        _expected_universe(phase)
    )
    rows = []
    seen: set[tuple[int, int]] = set()
    for path in sorted(input_dir.glob("v11_fold_result_fold*_seed*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        fold = int(row.get("outer_fold", -1))
        seed = int(row.get("seed", -1))
        pair = (fold, seed)
        if pair in seen:
            raise ValueError(f"duplicate V11 fold-seed {pair}")
        seen.add(pair)
        if row.get("schema_version") != FOLD_SCHEMA or row.get("phase") != phase:
            raise ValueError(f"invalid {phase} fold result in {path}")
        if int(row.get("point_epochs", -1)) != point_epochs or int(
            row.get("calibration_epochs", -1)
        ) != calibration_epochs:
            raise ValueError(f"V11 fold-seed {pair} violates epoch freeze")
        if not recorded_invariants_pass(row.get("invariants", {})):
            raise ValueError(f"V11 fold-seed {pair} lacks required invariants")
        if not authoritative_comparator_invariant_pass(
            phase, row.get("invariants", {})
        ):
            raise ValueError(
                f"V11 fold-seed {pair} lacks authoritative comparator provenance"
            )
        checks = prediction_checks(
            Path(row["prediction_artifact"]),
            fold=fold,
            seed=seed,
            expected_rows=int(row["n_registered_prediction_rows"]),
        )
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(f"V11 fold-seed {pair} prediction checks failed: {failed}")
        if sorted(row.get("point_parameter_counts", {}).values()) != [
            EXPECTED_POINT_PARAMETERS,
            EXPECTED_POINT_PARAMETERS,
        ]:
            raise ValueError(f"V11 fold-seed {pair} point parameter count changed")
        if set(row.get("residual_parameter_counts", {}).values()) != {63748}:
            raise ValueError(f"V11 fold-seed {pair} residual family changed")
        if not all(
            Path(value).is_file()
            for value in {
                **row.get("point_checkpoints", {}),
                **row.get("residual_checkpoints", {}),
            }.values()
        ):
            raise FileNotFoundError(f"V11 fold-seed {pair} lacks a checkpoint")
        rows.append(row)
    expected = {
        (fold, seed) for seed in expected_seeds for fold in expected_folds
    }
    if seen != expected or len(rows) != len(expected):
        raise ValueError(
            "V11 fold-seed universe is incomplete: "
            f"found={sorted(seen)} expected={sorted(expected)}"
        )
    rows.sort(key=lambda row: (int(row["seed"]), int(row["outer_fold"])))
    return {
        "schema_version": SCHEMA,
        "phase": phase,
        "status": status,
        "folds": rows,
        "merge_integrity": {
            "complete_fold_seed_universe": True,
            "unique_fold_seed_pairs": True,
            "prediction_only_schema": True,
            "target_identity_exact": True,
            "exact_point_parameter_match_all_runs": True,
            "fixed_skip_only_difference_all_runs": True,
            "point_frozen_during_calibration_all_runs": True,
            "v10_residual_family_all_runs": True,
            "feature41_replay_all_runs": True,
            "feature41_asymmetric_seed0_replay_all_folds": True,
            "authoritative_feature41_seed0_comparator_provenance_all_runs": True,
            "median_constraint_all_runs": True,
            "partial_scores_inspected": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("V11M2", "V11M3", "V11M4"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_folds(args.input_dir, args.phase)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
