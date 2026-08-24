#!/usr/bin/env python3
"""Post-Gate magnitude-bias diagnosis for the complete failed V8M2 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.score_model_rescue_v8_mean_screen import (
    _load_feature41_prediction,
    _load_v8_prediction,
    _tic2a_fold_map,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v8_magnitude_bias_diagnostic.v1"


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 20:
        raise ValueError("V8 magnitude diagnosis requires all 20 puzzles")
    return {
        "n_puzzles": 20,
        "target_absolute_delta_mean": _mean(rows, "target_absolute_delta"),
        "meanaligned_predicted_absolute_mean": _mean(
            rows, "meanaligned_predicted_absolute"
        ),
        "feature41_predicted_absolute_mean": _mean(
            rows, "feature41_predicted_absolute"
        ),
        "meanaligned_absolute_bias": _mean(rows, "meanaligned_absolute_bias"),
        "feature41_absolute_bias": _mean(rows, "feature41_absolute_bias"),
        "meanaligned_underprediction_puzzles": int(
            sum(float(row["meanaligned_absolute_bias"]) < 0.0 for row in rows)
        ),
        "feature41_underprediction_puzzles": int(
            sum(float(row["feature41_absolute_bias"]) < 0.0 for row in rows)
        ),
    }


def diagnose(
    v8_merged: dict[str, Any], tic2a_merged: dict[str, Any], m2_csv: Path
) -> dict[str, Any]:
    if v8_merged.get("status") != "V8M2_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("magnitude diagnosis requires the complete V8 merge")
    v8_rows = {int(row["outer_fold"]): row for row in v8_merged["folds"]}
    if sorted(v8_rows) != list(range(20)):
        raise ValueError("magnitude diagnosis requires V8 folds 0 through 19")
    tic2a_rows = _tic2a_fold_map(tic2a_merged)
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("magnitude diagnosis requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    puzzle_rows = []
    for fold_id in range(20):
        fold = folds[fold_id]
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        v8_prediction = _load_v8_prediction(
            Path(v8_rows[fold_id]["expert_prediction_artifact"]), fold_id
        )
        feature41_prediction = _load_feature41_prediction(
            Path(tic2a_rows[fold_id]["prediction_artifact"]), fold_id
        )
        v8_index = {
            str(key): index for index, key in enumerate(v8_prediction["keys"])
        }
        feature41_index = {
            str(key): index
            for index, key in enumerate(feature41_prediction["keys"])
        }
        values: dict[str, dict[str, float]] = {
            "target_absolute_delta": {},
            "meanaligned_predicted_absolute": {},
            "feature41_predicted_absolute": {},
            "meanaligned_absolute_bias": {},
            "feature41_absolute_bias": {},
        }
        for record in held_records:
            construct = univ.get_construct(record.construct_id)
            target, _error = univ.mutant_full_profile(
                record.wt_id, record.design_pos, record.ref, record.alt
            )
            if target is None:
                continue
            qualified = construct.wt_observed.astype(bool) & np.isfinite(target)
            for position in np.flatnonzero(qualified):
                key = _bio_key(univ, record, int(position))
                truth = abs(
                    float(target[position] - construct.wt_reactivity[position])
                )
                meanaligned = abs(
                    float(
                        v8_prediction["meanaligned_delta_mean"][v8_index[key]]
                    )
                )
                feature41 = float(
                    feature41_prediction["v6_feature41_absolute_delta"][
                        feature41_index[key]
                    ]
                )
                values["target_absolute_delta"][key] = truth
                values["meanaligned_predicted_absolute"][key] = meanaligned
                values["feature41_predicted_absolute"][key] = feature41
                values["meanaligned_absolute_bias"][key] = meanaligned - truth
                values["feature41_absolute_bias"][key] = feature41 - truth
        puzzle_rows.append(
            {
                "outer_fold": fold_id,
                "held_puzzle": str(fold.held_puzzle),
                **{name: _puzzle_macro(data) for name, data in values.items()},
            }
        )
    return {
        "schema_version": SCHEMA,
        "status": "V8M2_POST_GATE_MAGNITUDE_BIAS_DIAGNOSIS_COMPLETE",
        "evidence_status": "POST_HOC_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "puzzles": puzzle_rows,
        "summary": summarize(puzzle_rows),
        "v8_gate_changed": False,
        "model_or_threshold_selected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-merged-json", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = diagnose(
        json.loads(args.v8_merged_json.read_text(encoding="utf-8")),
        json.loads(args.tic2a_merged_json.read_text(encoding="utf-8")),
        args.m2_csv,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
