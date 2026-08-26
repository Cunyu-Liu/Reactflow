#!/usr/bin/env python3
"""Join targets once and score the complete post-V13 diagnostic universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta import evaluator_v2 as E
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_post_v13_route_diagnostics import (
    SCHEMA as MERGED_SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_post_v13_route_diagnostics import (
    PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.post_v13_route_diagnostic_score.v1"
SCORE_TOKEN = "PV13D_COMPLETE_MERGE_SCORE_ONCE_ONLY"


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active.get("authority", {}).get("current_phase") != "PV13D3":
        raise RuntimeError("post-V13 scorer is closed outside PV13D3")
    if active.get("held_score_read_allowed") != SCORE_TOKEN:
        raise RuntimeError("post-V13 complete score-once authority is absent")
    if active.get("training_allowed") is not False:
        raise RuntimeError("post-V13 scoring requires fixed-ridge fitting closed")
    if active.get("candidate_model_training_allowed") is not False:
        raise RuntimeError("post-V13 scoring requires neural training closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("post-V13 partial scores must remain prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("post-V13 scorer requires external outcomes locked")


def _load_prediction(path: Path, outer_fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction.get("schema_version", np.asarray("")).item()) != (
        PREDICTION_SCHEMA
    ):
        raise ValueError(f"invalid post-V13 prediction schema in {path}")
    keys = list(map(str, prediction["keys"]))
    if keys != list(map(str, prediction["biological_scoring_key"])):
        raise ValueError(f"post-V13 prediction keys disagree in {path}")
    if len(keys) != len(set(keys)):
        raise ValueError(f"post-V13 prediction keys are duplicated in {path}")
    if set(map(int, prediction["outer_fold"])) != {outer_fold}:
        raise ValueError(f"post-V13 prediction outer fold differs in {path}")
    for field in PREDICTION_FIELDS:
        if prediction[field].shape != (len(keys),) or not np.isfinite(
            prediction[field]
        ).all():
            raise ValueError(f"post-V13 prediction {field} is invalid in {path}")
    forbidden = {
        "target",
        "target_error",
        "target_mask",
        "qualified_target_mask",
        "score",
        "loss",
        "mae",
        "crps",
    }
    if set(prediction) & forbidden:
        raise ValueError(f"post-V13 prediction contains target-side fields in {path}")
    return prediction


def puzzle_macro(losses: dict[str, float]) -> float:
    mutant_losses: dict[tuple[str, str, str], list[float]] = {}
    puzzles: set[str] = set()
    for key, loss in losses.items():
        parts = E._bio_key_parts(key)
        puzzle = parts["puzzle"]
        method = parts["method"]
        raw = key.split("|")
        if len(raw) < 7:
            raise ValueError(f"invalid post-V13 biological scoring key {key}")
        mutation = f"{raw[3]}|{raw[4]}|{raw[5]}"
        puzzles.add(puzzle)
        mutant_losses.setdefault((puzzle, method, mutation), []).append(float(loss))
    if len(puzzles) != 1 or not mutant_losses:
        raise ValueError("one post-V13 fold must contain one scored held puzzle")
    method_losses: dict[str, list[float]] = {}
    for (_puzzle, method, _mutation), values in mutant_losses.items():
        method_losses.setdefault(method, []).append(float(np.mean(values)))
    return float(np.mean([float(np.mean(values)) for values in method_losses.values()]))


def score_fold(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, Any]:
    expected_keys = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    key_to_index = {str(key): index for index, key in enumerate(prediction["keys"])}
    predicted_keys = set(key_to_index)
    metric_names = (
        "baseline_signed_delta",
        "noise_aware_signed_delta",
        "coherent_signed_delta",
        "baseline_point_absolute_delta",
        "noise_aware_point_absolute_delta",
        "coherent_point_absolute_delta",
        "baseline_dedicated_absolute_delta",
        "noise_aware_dedicated_absolute_delta",
    )
    losses: dict[str, dict[str, float]] = {name: {} for name in metric_names}
    n_qualified = 0
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
            if key not in key_to_index:
                raise ValueError("qualified post-V13 target lacks a prediction")
            index = key_to_index[key]
            signed = float(target[position] - construct.wt_reactivity[position])
            absolute = abs(signed)
            baseline_signed = float(prediction["baseline_signed_delta"][index])
            noise_signed = float(prediction["noise_aware_signed_delta"][index])
            coherent_signed = float(prediction["coherent_signed_delta"][index])
            losses["baseline_signed_delta"][key] = abs(signed - baseline_signed)
            losses["noise_aware_signed_delta"][key] = abs(signed - noise_signed)
            losses["coherent_signed_delta"][key] = abs(signed - coherent_signed)
            losses["baseline_point_absolute_delta"][key] = abs(
                absolute - abs(baseline_signed)
            )
            losses["noise_aware_point_absolute_delta"][key] = abs(
                absolute - abs(noise_signed)
            )
            losses["coherent_point_absolute_delta"][key] = abs(
                absolute - abs(coherent_signed)
            )
            losses["baseline_dedicated_absolute_delta"][key] = abs(
                absolute - float(prediction["baseline_absolute_delta"][index])
            )
            losses["noise_aware_dedicated_absolute_delta"][key] = abs(
                absolute - float(prediction["noise_aware_absolute_delta"][index])
            )
            n_qualified += 1

    unexpected = predicted_keys - expected_keys
    missing = expected_keys - predicted_keys
    result = {f"{name}_mae": puzzle_macro(values) for name, values in losses.items()}
    result.update(
        {
            "n_qualified_positions": n_qualified,
            "n_registered_expected": len(expected_keys),
            "n_registered_observed": len(predicted_keys),
            "registered_prediction_coverage": len(expected_keys & predicted_keys)
            / max(len(expected_keys), 1),
            "failure_rate": len(missing) / max(len(expected_keys), 1),
            "n_unexpected_prediction_keys": len(unexpected),
        }
    )
    return result


def score_complete_merged(merged: dict[str, Any], m2_csv: Path) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "PV13D2_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("post-V13 scorer requires one complete unscored merge")
    integrity = merged.get("merge_integrity", {})
    required = (
        "complete_fold_universe",
        "unique_folds",
        "prediction_only_fields",
        "corrected_feature41_replay_all_folds",
        "held_scores_absent",
    )
    if not all(integrity.get(name) is True for name in required) or any(
        integrity.get(name) is True
        for name in (
            "partial_score_inspected",
            "model_or_threshold_selection_performed",
            "external_outcome_accessed",
        )
    ):
        raise ValueError("post-V13 scorer rejects a contaminated or incomplete merge")

    univ = M2Universe(m2_csv)
    univ.build()
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    rows: list[dict[str, Any]] = []
    for fold_row in merged["folds"]:
        fold_id = int(fold_row["outer_fold"])
        fold = fold_map[fold_id]
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        prediction = _load_prediction(Path(fold_row["prediction_artifact"]), fold_id)
        score = score_fold(univ, held_records, prediction)
        score["outer_fold"] = fold_id
        score["held_puzzle"] = str(fold.held_puzzle)
        rows.append(score)
    rows.sort(key=lambda row: int(row["outer_fold"]))
    if [int(row["outer_fold"]) for row in rows] != list(range(20)):
        raise ValueError("post-V13 score fold universe changed")
    return {
        "schema_version": SCHEMA,
        "phase": "PV13D3",
        "status": "PV13D3_COMPLETE_SCORE_PASS",
        "scores": rows,
        "target_join_after_complete_merge": True,
        "corrected_feature41_replay_all_folds": True,
        "partial_fold_scores_inspected": False,
        "model_or_threshold_selection_performed": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_score_authority(args.repo_root.resolve())
    merged = json.loads(args.merged_json.read_text(encoding="utf-8"))
    result = score_complete_merged(merged, args.m2_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
