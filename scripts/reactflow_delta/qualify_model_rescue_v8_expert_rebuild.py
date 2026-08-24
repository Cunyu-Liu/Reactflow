#!/usr/bin/env python3
"""Qualify the complete fresh target-identity-corrected V8M1 experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.qualify_model_rescue_v3_expert_rebuild import (
    check_fold_result,
)
from scripts.reactflow_delta.run_model_rescue_v8_expert_rebuild import (
    PREDICTION_SCHEMA,
    SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


QUALIFICATION_SCHEMA = (
    "reactflow_delta.model_rescue_v8_corrected_expert_qualification.v1"
)


def qualify(input_dir: Path, m2_csv: Path) -> dict:
    universe = M2Universe(m2_csv)
    identity = universe.build()
    identity_exact = (
        identity.get("n_canonical_mutant_full_profiles") == 13976
        and identity.get("canonical_mutant_full_profile_identity")
        == "EXACT_PUZZLE_METHOD_MUTATION"
    )
    records = universe.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=20260813
    )
    fold_by_id = {int(fold.outer_fold): fold for fold in split["folds"]}
    paths = sorted(input_dir.glob("v8_corrected_expert_fold_result_fold*_seed0.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ids = [int(row["outer_fold"]) for row in rows]
    universe_complete = sorted(ids) == list(range(20)) and len(ids) == len(set(ids))
    fold_results = []
    if universe_complete and identity_exact:
        for row in sorted(rows, key=lambda value: int(value["outer_fold"])):
            fold_id = int(row["outer_fold"])
            fold = fold_by_id[fold_id]
            held_records = [r for r in records if r.puzzle == fold.held_puzzle]
            expected = {
                _bio_key(universe, record, position)
                for record in held_records
                for position in range(
                    len(universe.get_construct(record.construct_id).sequence)
                )
            }
            checks = check_fold_result(
                row,
                expected,
                result_schema=SCHEMA,
                prediction_schema=PREDICTION_SCHEMA,
            )
            checks.update(
                {
                    "target_profile_identity_exact": row.get(
                        "target_profile_identity"
                    )
                    == "EXACT_PUZZLE_METHOD_MUTATION",
                    "canonical_profile_count_exact": int(
                        row.get("canonical_mutant_full_profiles", -1)
                    )
                    == 13976,
                    "legacy_v3_checkpoint_not_reused": row.get(
                        "legacy_v3_checkpoint_reused"
                    )
                    is False,
                    "legacy_v3_prediction_not_reused": row.get(
                        "legacy_v3_prediction_reused"
                    )
                    is False,
                }
            )
            fold_results.append(
                {
                    "outer_fold": fold_id,
                    "held_puzzle": fold.held_puzzle,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    passed = identity_exact and universe_complete and all(
        row["passed"] for row in fold_results
    )
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": (
            "V8M1_CORRECTED_EXPERT_REBUILD_PASS"
            if passed
            else "V8M1_CORRECTED_EXPERT_REBUILD_FAIL"
        ),
        "target_profile_identity_exact": identity_exact,
        "fold_universe_complete": universe_complete,
        "folds": fold_results,
        "scores_read": False,
        "external_outcome_accessed": False,
        "v8m2_authorized": passed,
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
    print(json.dumps({"status": result["status"]}, indent=2))
    return 0 if result["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
