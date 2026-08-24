#!/usr/bin/env python3
"""Join corrected targets only after the complete TIC2A prediction merge."""

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

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_target_identity_corrected_baselines import (
    SCHEMA as MERGED_SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_target_identity_corrected_baselines import (
    PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.target_identity_corrected_baseline_score.v1"


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "TIC2A":
        raise RuntimeError("corrected baseline scorer is closed outside TIC2A")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("complete corrected baseline score access is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial corrected baseline scores must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("corrected baseline scorer requires external outcomes locked")


def _load_prediction(path: Path, outer_fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid corrected prediction schema in {path}")
    keys = prediction["keys"]
    if not np.array_equal(keys, prediction["biological_scoring_key"]):
        raise ValueError(f"corrected prediction keys disagree in {path}")
    if len(set(map(str, keys))) != len(keys):
        raise ValueError(f"corrected prediction keys are not unique in {path}")
    if set(map(int, prediction["outer_fold"])) != {outer_fold}:
        raise ValueError(f"corrected prediction outer fold mismatch in {path}")
    for field in PREDICTION_FIELDS:
        if prediction[field].shape != (len(keys),) or not np.isfinite(
            prediction[field]
        ).all():
            raise ValueError(f"corrected prediction {field} is invalid in {path}")
    forbidden = {
        "target",
        "target_error",
        "target_mask",
        "qualified_target_mask",
        "score",
        "mae",
        "crps",
    }
    if not forbidden.isdisjoint(prediction):
        raise ValueError(f"corrected prediction contains target-side fields in {path}")
    return prediction


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
                raise ValueError("qualified corrected target is missing from prediction")
            index = key_to_index[key]
            signed = float(target[position] - construct.wt_reactivity[position])
            absolute = abs(signed)
            for field in PREDICTION_FIELDS:
                expected = absolute if field.endswith("absolute_delta") else signed
                losses[field][key] = abs(expected - float(prediction[field][index]))
            n_qualified += 1
    unexpected = predicted_keys - expected_keys
    missing = expected_keys - predicted_keys
    result = {
        f"{field}_mae": _puzzle_macro(values) for field, values in losses.items()
    }
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
    if merged.get("schema_version") != MERGED_SCHEMA:
        raise ValueError("corrected scorer requires the complete merged schema")
    integrity = merged.get("merge_integrity", {})
    required = (
        "complete_fold_universe",
        "target_identity_exact",
        "v5_v6_feature30_replay_all_folds",
    )
    if any(integrity.get(name) is not True for name in required):
        raise ValueError("corrected scorer requires complete exact-identity merge")
    univ = M2Universe(m2_csv)
    ledger = univ.build()
    if ledger.get("n_canonical_mutant_full_profiles") != 13976:
        raise RuntimeError("corrected scorer requires 13,976 canonical profiles")
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
        score["held_puzzle"] = str(fold.held_puzzle)
        rows.append(score)
    rows.sort(key=lambda row: int(row["outer_fold"]))
    return {
        "schema_version": SCHEMA,
        "phase": "TIC2A",
        "status": "TIC2A_COMPLETE_CORRECTED_SCORE_PASS",
        "scores": rows,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "legacy_prediction_reused": False,
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
