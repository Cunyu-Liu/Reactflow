#!/usr/bin/env python3
"""Join targets and score V5M2 only after the complete fold merge."""

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
from scripts.reactflow_delta.merge_model_rescue_v5_probe import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.run_model_rescue_v5_probe import PREDICTION_SCHEMA
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v5_probe_score.v1"
PREDICTION_FIELDS = (
    "baseline_signed_delta",
    "candidate_signed_delta",
    "baseline_absolute_delta",
    "candidate_absolute_delta",
)


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V5M2":
        raise RuntimeError("v5 probe scorer is closed outside active V5M2")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("complete V5M2 score access has not been opened")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial V5M2 scores must remain prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v5 probe scorer requires external outcomes locked")


def _load_prediction(path: Path, outer_fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid v5 prediction schema in {path}")
    keys = prediction["keys"]
    if not np.array_equal(keys, prediction["biological_scoring_key"]):
        raise ValueError(f"v5 prediction keys disagree in {path}")
    if len(set(map(str, keys))) != len(keys):
        raise ValueError(f"v5 prediction keys are not unique in {path}")
    if set(map(int, prediction["outer_fold"])) != {outer_fold}:
        raise ValueError(f"v5 prediction outer fold mismatch in {path}")
    for field in PREDICTION_FIELDS:
        if prediction[field].shape != (len(keys),) or not np.isfinite(prediction[field]).all():
            raise ValueError(f"v5 prediction {field} is invalid in {path}")
    forbidden = {"target", "target_error", "target_mask", "score"}
    if not forbidden.isdisjoint(prediction):
        raise ValueError(f"v5 prediction contains target-side fields in {path}")
    return prediction


def _puzzle_macro(losses: dict[str, float]) -> float:
    """Apply the frozen position -> mutant -> method -> puzzle estimand.

    ``evaluator_v2.score_position_losses`` balances methods but pools positions
    directly within a method.  That is only equivalent to the registered v5
    estimand when every mutant has the same qualified-position count.  The real
    M2 masks differ, so v5 performs the missing mutant-level reduction here.
    """
    mutant_losses: dict[tuple[str, str, str], list[float]] = {}
    puzzles: set[str] = set()
    for key, loss in losses.items():
        parts = E._bio_key_parts(key)
        puzzle = parts["puzzle"]
        method = parts["method"]
        mutation = parts["mutation"]
        puzzles.add(puzzle)
        mutant_losses.setdefault((puzzle, method, mutation), []).append(float(loss))
    if len(puzzles) != 1 or not mutant_losses:
        raise ValueError("one V5M2 fold must contain exactly one scored held puzzle")
    method_losses: dict[str, list[float]] = {}
    for (_puzzle, method, _mutation), values in mutant_losses.items():
        method_losses.setdefault(method, []).append(float(np.mean(values)))
    return float(
        np.mean(
            [float(np.mean(values)) for values in method_losses.values()]
        )
    )


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
    losses: dict[str, dict[str, float]] = {field: {} for field in PREDICTION_FIELDS}
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
                raise ValueError("qualified V5M2 target is missing from prediction ledger")
            index = key_to_index[key]
            signed = float(target[position] - construct.wt_reactivity[position])
            absolute = abs(signed)
            losses["baseline_signed_delta"][key] = abs(
                signed - float(prediction["baseline_signed_delta"][index])
            )
            losses["candidate_signed_delta"][key] = abs(
                signed - float(prediction["candidate_signed_delta"][index])
            )
            losses["baseline_absolute_delta"][key] = abs(
                absolute - float(prediction["baseline_absolute_delta"][index])
            )
            losses["candidate_absolute_delta"][key] = abs(
                absolute - float(prediction["candidate_absolute_delta"][index])
            )
            n_qualified += 1
    unexpected = predicted_keys - expected_keys
    missing = expected_keys - predicted_keys
    return {
        "baseline_signed_delta_mae": _puzzle_macro(losses["baseline_signed_delta"]),
        "candidate_signed_delta_mae": _puzzle_macro(losses["candidate_signed_delta"]),
        "baseline_absolute_delta_mae": _puzzle_macro(losses["baseline_absolute_delta"]),
        "candidate_absolute_delta_mae": _puzzle_macro(losses["candidate_absolute_delta"]),
        "n_qualified_positions": n_qualified,
        "n_registered_expected": len(expected_keys),
        "n_registered_observed": len(predicted_keys),
        "registered_prediction_coverage": len(expected_keys & predicted_keys)
        / max(len(expected_keys), 1),
        "failure_rate": len(missing) / max(len(expected_keys), 1),
        "n_unexpected_prediction_keys": len(unexpected),
    }


def score_complete_merged(merged: dict[str, Any], m2_csv: Path) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA:
        raise ValueError("v5 scorer requires the complete merged schema")
    integrity = merged.get("merge_integrity", {})
    if integrity.get("complete_fold_universe") is not True:
        raise ValueError("v5 scorer cannot access an incomplete fold universe")
    univ = M2Universe(m2_csv)
    univ.build()
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    rows = []
    for fold_row in merged["folds"]:
        fold_id = int(fold_row["outer_fold"])
        fold = fold_map[fold_id]
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        prediction = _load_prediction(Path(fold_row["prediction_artifact"]), fold_id)
        score = score_fold(univ, held_records, prediction)
        score["outer_fold"] = fold_id
        score["held_puzzle"] = fold.held_puzzle
        rows.append(score)
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "V5M2",
        "status": "V5M2_COMPLETE_SCORE_PASS",
        "scores": rows,
        "target_join_after_complete_merge": True,
        "partial_fold_scores_inspected": False,
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
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
