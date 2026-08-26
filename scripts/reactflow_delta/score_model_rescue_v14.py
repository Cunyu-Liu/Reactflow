#!/usr/bin/env python3
"""Score the one complete V14M3 universe after score-blind merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v14 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v14 import PREDICTION_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v10 import _load_tic2a_absolute
from scripts.reactflow_delta.score_model_rescue_v12 import SCHEMA as V12_SCORE_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v13 import score_fold
from scripts.reactflow_delta.score_model_rescue_v9 import TIC2A_MERGED_SCHEMA
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v14_score.v1"


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V14M3" or active.get(
        "runnable_phases"
    ) != ["V14M3"]:
        raise RuntimeError("V14 scorer is closed outside complete V14M3 authority")
    if active.get("training_allowed") is not False or active.get(
        "candidate_model_training_allowed"
    ) is not False:
        raise RuntimeError("V14 training must be closed before scientific scoring")
    if active.get("held_score_read_allowed") != "V14_COMPLETE_MERGE_SCORE_ONCE_ONLY":
        raise RuntimeError("V14 complete score authority is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V14 partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V14 external outcomes must remain locked")


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    required_true = (
        "complete_fold_seed_universe",
        "unique_fold_seed_pairs",
        "prediction_only_schema",
        "target_identity_exact",
        "outer_train_wt_only_pretraining_all_runs",
        "zero_observed_constructs_excluded_all_runs",
        "held_puzzle_wt_excluded_all_runs",
        "mutant_outcome_excluded_from_pretraining_all_runs",
        "exact_initial_and_parameter_match_all_runs",
        "candidate_encoder_changed_all_runs",
        "null_unchanged_before_supervision_all_runs",
        "residual_head_equal_before_supervision_all_runs",
        "decoder_frozen_downstream_all_runs",
        "point_frozen_during_calibration_all_runs",
        "v10_residual_family_all_runs",
        "feature41_replay_all_runs",
        "median_constraint_all_runs",
    )
    required_false = ("partial_scores_inspected", "external_outcome_accessed")
    return all(integrity.get(name) is True for name in required_true) and all(
        integrity.get(name) is False for name in required_false
    )


def _load_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: np.asarray(handle[name]) for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V14 prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold} or set(
        map(int, prediction["seed"])
    ) != {0}:
        raise ValueError(f"V14 fold or seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate V14 keys in {path}")
    return prediction


def _v12_parent_rows(score: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if score.get("schema_version") != V12_SCORE_SCHEMA or score.get("status") != (
        "V12M3_COMPLETE_SCORE_PASS"
    ):
        raise ValueError("V14 scorer requires the frozen complete V12 score")
    rows = {int(row["outer_fold"]): row for row in score.get("scores", [])}
    if sorted(rows) != list(range(20)):
        raise ValueError("V14 scorer requires complete V12 folds0-19")
    return rows


def score_complete(
    merged: dict[str, Any],
    tic2a_merged: dict[str, Any],
    v12_score: dict[str, Any],
    m2_csv: Path,
) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V14M3_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V14 scorer requires one complete V14M3 merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("V14 merged integrity is not qualified")
    if tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA or tic2a_merged.get(
        "status"
    ) != "TIC2A_COMPLETE_UNSCORED_MERGE_PASS":
        raise ValueError("V14 scorer requires the corrected TIC2A merge")
    v14_rows = {int(row["outer_fold"]): row for row in merged["folds"]}
    tic_rows = {int(row["outer_fold"]): row for row in tic2a_merged["folds"]}
    parent_rows = _v12_parent_rows(v12_score)
    if sorted(v14_rows) != list(range(20)) or sorted(tic_rows) != list(range(20)):
        raise ValueError("V14 scorer requires folds0-19 in both universes")

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("V14 scorer requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    rows = []
    for fold_id in range(20):
        fold = folds[fold_id]
        parent = parent_rows[fold_id]
        if str(parent.get("held_puzzle")) != str(fold.held_puzzle):
            raise ValueError("V14/V12 held puzzle alignment differs")
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        score = score_fold(
            univ,
            held_records,
            _load_prediction(Path(v14_rows[fold_id]["prediction_artifact"]), fold_id),
            _load_tic2a_absolute(Path(tic_rows[fold_id]["prediction_artifact"]), fold_id),
        )
        score.update(
            {
                "outer_fold": fold_id,
                "held_puzzle": str(fold.held_puzzle),
                "terminal_v12_signed_delta_mae": float(parent["candidate_signed_delta_mae"]),
                "terminal_v11_point_absolute_delta_mae": float(parent["parent_v11_point_absolute_delta_mae"]),
                "terminal_v12_crps": float(parent["candidate_crps"]),
                "terminal_v10_distribution_absolute_delta_mae": float(parent["historical_v10_distribution_absolute_delta_mae"]),
            }
        )
        rows.append(score)
    return {
        "schema_version": SCHEMA,
        "phase": "V14M3",
        "status": "V14M3_COMPLETE_SCORE_PASS",
        "scores": rows,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "terminal_parent_metrics_from_frozen_complete_v12_score": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--v12-score-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_score_authority(args.repo_root.resolve())
    result = score_complete(
        json.loads(args.merged_json.read_text(encoding="utf-8")),
        json.loads(args.tic2a_merged_json.read_text(encoding="utf-8")),
        json.loads(args.v12_score_json.read_text(encoding="utf-8")),
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
