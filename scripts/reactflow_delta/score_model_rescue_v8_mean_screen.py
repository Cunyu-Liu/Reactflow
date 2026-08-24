#!/usr/bin/env python3
"""Join corrected targets once for the complete frozen V8M2 mean screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v8_mean_screen import (
    SCHEMA as V8_MERGED_SCHEMA,
)
from scripts.reactflow_delta.run_model_rescue_v8_expert_rebuild import (
    PREDICTION_SCHEMA as V8_PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v8_mean_screen_score.v1"
TIC2A_MERGED_SCHEMA = "reactflow_delta.target_identity_corrected_baseline_merged.v1"
TIC2A_PREDICTION_SCHEMA = (
    "reactflow_delta.target_identity_corrected_baseline_prediction.v1"
)


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V8M2":
        raise RuntimeError("V8 mean screen scoring is closed outside V8M2")
    if active.get("runnable_phases") != ["V8M2"]:
        raise RuntimeError("V8M2 must be the only runnable phase")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("complete V8M2 score access is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial V8M2 scores must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V8M2 requires external outcomes locked")
    if active.get("training_allowed") is not False:
        raise RuntimeError("training must be closed during V8M2 scoring")


def _load_v8_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != V8_PREDICTION_SCHEMA:
        raise ValueError(f"invalid V8 prediction schema in {path}")
    keys = prediction["keys"]
    if len(set(map(str, keys))) != len(keys):
        raise ValueError(f"V8 prediction keys are duplicated in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold}:
        raise ValueError(f"V8 prediction fold mismatch in {path}")
    for field in ("b1_delta_mean", "meanaligned_delta_mean"):
        if prediction[field].shape != (len(keys),) or not np.isfinite(
            prediction[field]
        ).all():
            raise ValueError(f"invalid V8 {field} in {path}")
    return prediction


def _load_feature41_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != TIC2A_PREDICTION_SCHEMA:
        raise ValueError(f"invalid TIC2A prediction schema in {path}")
    keys = prediction["keys"]
    if not np.array_equal(keys, prediction["biological_scoring_key"]):
        raise ValueError(f"TIC2A biological keys disagree in {path}")
    if len(set(map(str, keys))) != len(keys):
        raise ValueError(f"TIC2A prediction keys are duplicated in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold}:
        raise ValueError(f"TIC2A prediction fold mismatch in {path}")
    for field in (
        "v6_feature41_signed_delta",
        "v6_feature41_absolute_delta",
    ):
        if prediction[field].shape != (len(keys),) or not np.isfinite(
            prediction[field]
        ).all():
            raise ValueError(f"invalid TIC2A {field} in {path}")
    return prediction


def _tic2a_fold_map(merged: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if merged.get("schema_version") != TIC2A_MERGED_SCHEMA:
        raise ValueError("V8M2 requires the frozen TIC2A merged schema")
    if merged.get("status") != "TIC2A_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V8M2 requires a complete unscored TIC2A merge")
    integrity = merged.get("merge_integrity", {})
    for name in (
        "complete_fold_universe",
        "prediction_only_fields",
        "target_identity_exact",
    ):
        if integrity.get(name) is not True:
            raise ValueError(f"TIC2A merge lacks {name}")
    result = {int(row["outer_fold"]): row for row in merged.get("folds", [])}
    if sorted(result) != list(range(20)) or len(merged.get("folds", [])) != 20:
        raise ValueError("TIC2A fold universe is not exactly 0 through 19")
    return result


def score_fold(
    univ: M2Universe,
    held_records: list[Any],
    v8_prediction: dict[str, np.ndarray],
    feature41_prediction: dict[str, np.ndarray],
) -> dict[str, Any]:
    v8_index = {str(key): i for i, key in enumerate(v8_prediction["keys"])}
    feature41_index = {
        str(key): i for i, key in enumerate(feature41_prediction["keys"])
    }
    expected_keys = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    if set(v8_index) != expected_keys or set(feature41_index) != expected_keys:
        raise ValueError("V8M2 requires exact, identical registered key universes")

    fields = (
        "feature41_signed_delta",
        "b1_signed_delta",
        "meanaligned_signed_delta",
        "feature41_absolute_delta",
        "b1_absolute_delta",
        "meanaligned_absolute_delta",
    )
    losses: dict[str, dict[str, float]] = {name: {} for name in fields}
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
            vi = v8_index[key]
            fi = feature41_index[key]
            signed = float(target[position] - construct.wt_reactivity[position])
            absolute = abs(signed)
            b1 = float(v8_prediction["b1_delta_mean"][vi])
            meanaligned = float(v8_prediction["meanaligned_delta_mean"][vi])
            feature41_signed = float(
                feature41_prediction["v6_feature41_signed_delta"][fi]
            )
            feature41_absolute = float(
                feature41_prediction["v6_feature41_absolute_delta"][fi]
            )
            predicted = {
                "feature41_signed_delta": feature41_signed,
                "b1_signed_delta": b1,
                "meanaligned_signed_delta": meanaligned,
                "feature41_absolute_delta": feature41_absolute,
                "b1_absolute_delta": abs(b1),
                "meanaligned_absolute_delta": abs(meanaligned),
            }
            for field, value in predicted.items():
                truth = absolute if field.endswith("absolute_delta") else signed
                losses[field][key] = abs(truth - value)
            n_qualified += 1

    result = {f"{name}_mae": _puzzle_macro(values) for name, values in losses.items()}
    result.update(
        {
            "n_qualified_positions": n_qualified,
            "n_registered_expected": len(expected_keys),
            "n_registered_v8": len(v8_index),
            "n_registered_feature41": len(feature41_index),
            "registered_prediction_coverage": 1.0,
            "failure_rate": 0.0,
            "n_unexpected_prediction_keys": 0,
        }
    )
    return result


def score_complete(
    v8_merged: dict[str, Any], tic2a_merged: dict[str, Any], m2_csv: Path
) -> dict[str, Any]:
    if v8_merged.get("schema_version") != V8_MERGED_SCHEMA:
        raise ValueError("V8M2 scorer requires the V8 complete merged schema")
    if v8_merged.get("status") != "V8M2_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V8M2 scorer requires one complete unscored V8 merge")
    integrity = v8_merged.get("merge_integrity", {})
    for name in (
        "complete_fold_universe",
        "prediction_only_fields",
        "target_identity_exact",
        "fresh_checkpoints_all_folds",
    ):
        if integrity.get(name) is not True:
            raise ValueError(f"V8 merge lacks {name}")
    v8_rows = {int(row["outer_fold"]): row for row in v8_merged["folds"]}
    if sorted(v8_rows) != list(range(20)) or len(v8_merged["folds"]) != 20:
        raise ValueError("V8 fold universe is not exactly 0 through 19")
    tic2a_rows = _tic2a_fold_map(tic2a_merged)

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V8M2 requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    rows = []
    for fold_id in range(20):
        fold = fold_map[fold_id]
        v8_row = v8_rows[fold_id]
        tic2a_row = tic2a_rows[fold_id]
        if str(v8_row["held_puzzle"]) != str(fold.held_puzzle):
            raise ValueError(f"V8 held puzzle mismatch for fold {fold_id}")
        if str(tic2a_row["held_puzzle"]) != str(fold.held_puzzle):
            raise ValueError(f"TIC2A held puzzle mismatch for fold {fold_id}")
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        score = score_fold(
            univ,
            held_records,
            _load_v8_prediction(Path(v8_row["expert_prediction_artifact"]), fold_id),
            _load_feature41_prediction(
                Path(tic2a_row["prediction_artifact"]), fold_id
            ),
        )
        score["outer_fold"] = fold_id
        score["held_puzzle"] = str(fold.held_puzzle)
        rows.append(score)
    return {
        "schema_version": SCHEMA,
        "phase": "V8M2",
        "status": "V8M2_COMPLETE_MEAN_SCREEN_SCORE_PASS",
        "scores": rows,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_both_complete_merges": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_selection_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--v8-merged-json", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_score_authority(args.repo_root.resolve())
    result = score_complete(
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
