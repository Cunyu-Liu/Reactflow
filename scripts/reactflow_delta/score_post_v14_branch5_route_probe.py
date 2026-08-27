#!/usr/bin/env python3
"""Join held M2 targets once, after the complete branch-5 prediction merge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_post_v14_branch5_route_probe import (
    EXPECTED_PREDICTION_FIELDS,
    SCHEMA as MERGED_SCHEMA,
    STATUS as MERGED_STATUS,
)
from scripts.reactflow_delta.run_post_v14_branch5_route_probe import (
    EXPECTED_FOLDS,
    EXPECTED_PARENT_STATE,
    EXPECTED_PROJECT_TASK,
    EXPECTED_SEED,
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
    assert_frozen_runtime_paths,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCORE_PHASE = "B5RP2"
SCORE_TOKEN = "POST_V14_BRANCH5_COMPLETE_MERGE_SCORE_ONCE_ONLY"
SCHEMA = "reactflow_delta.puzzle_set_branch5_route_probe_score.v1"
COMPLETE_STATUS = "BRANCH5_ROUTE_PROBE_COMPLETE_SCORE_PASS"
INDETERMINATE_STATUS = "BRANCH5_ROUTE_PROBE_COMPLETE_SCORE_INDETERMINATE"
POINT_FIELDS = ("parent_point", "aligned_point", "shift17_point")
EXPECTED_MERGED_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "expected_folds",
    "expected_seed",
    "folds",
    "merge_integrity",
}


def assert_score_authority(
    repo_root: Path,
    *,
    merged_json: Path | None = None,
    m2_csv: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("branch5 scorer is not the active project")
    if active.get("authority", {}).get("current_phase") != SCORE_PHASE:
        raise RuntimeError("branch5 scorer is closed outside B5RP2")
    if active.get("runnable_phases") != [SCORE_PHASE]:
        raise RuntimeError("B5RP2 must be the only runnable phase")
    if (
        active.get("training_allowed") is not False
        or active.get("candidate_model_training_allowed") is not False
    ):
        raise RuntimeError("branch5 training must close before held scoring")
    if active.get("held_score_read_allowed") != SCORE_TOKEN:
        raise RuntimeError("branch5 complete score-once token is absent")
    if (
        active.get("partial_fold_score_read_allowed") is not False
        or active.get("new_external_outcome_access_allowed") is not False
    ):
        raise RuntimeError("branch5 partial or external score access is open")
    parent = active.get("parent_state", {})
    if any(parent.get(name) != value for name, value in EXPECTED_PARENT_STATE.items()):
        raise RuntimeError("branch5 scorer parent route is not exact")
    provided = {
        "complete_unscored_merge_path": merged_json,
        "m2_csv_path": m2_csv,
        "complete_score_path": out_json,
    }
    assert_frozen_runtime_paths(
        active.get("authority"),
        required_fields=tuple(provided),
        cli_paths=(
            {name: value for name, value in provided.items() if value is not None}
            if any(value is not None for value in provided.values())
            else None
        ),
    )
    return active


def merged_integrity_pass(integrity: dict[str, Any]) -> bool:
    required_true = (
        "complete_fold_universe",
        "unique_fold_ids",
        "prediction_only_schema",
        "prediction_key_universe_unique_per_fold",
        "samefold_parent_provenance_all_folds",
        "samefold_v14_content_contrast_all_folds",
        "single_complete_safe_source_registry",
        "single_complete_tic2a_safe_registry",
        "global_input_provenance_consistent_all_folds",
        "tic2a_safe_feature41_projection_all_folds",
        "ridge_protocol_exact_all_folds",
        "target_profile_identity_exact",
    )
    required_false = (
        "partial_scores_inspected",
        "external_outcome_accessed",
        "model_or_threshold_selection_performed",
    )
    return (
        set(integrity) == set(required_true) | set(required_false)
        and all(integrity.get(name) is True for name in required_true)
        and all(integrity.get(name) is False for name in required_false)
    )


def _load_prediction(path: Path, fold: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        names = set(handle.files)
        if names != EXPECTED_PREDICTION_FIELDS or names & FORBIDDEN_PREDICTION_FIELDS:
            raise ValueError(f"branch5 prediction field universe changed in {path}")
        result = {name: np.asarray(handle[name]) for name in names}
    if str(result["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"branch5 prediction schema changed in {path}")
    keys = list(map(str, result["keys"]))
    row_fields = (
        "biological_scoring_key",
        "outer_fold",
        "seed",
        "registered_status",
        *POINT_FIELDS,
    )
    if (
        not all(result[name].shape == (len(keys),) for name in row_fields)
        or keys != list(map(str, result["biological_scoring_key"]))
        or len(keys) != len(set(keys))
        or set(map(int, result["outer_fold"])) != {fold}
        or set(map(int, result["seed"])) != {EXPECTED_SEED}
        or set(map(str, result["registered_status"])) != {"covered"}
    ):
        raise ValueError(f"branch5 prediction identity changed in {path}")
    for name in POINT_FIELDS:
        if result[name].shape != (len(keys),) or not np.isfinite(result[name]).all():
            raise ValueError(f"branch5 prediction {name} is invalid in {path}")
    return result


def score_fold(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, Any]:
    keys = list(map(str, prediction["keys"]))
    index = {key: row for row, key in enumerate(keys)}
    expected = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    observed = set(index)
    coverage = len(expected & observed) / max(len(expected), 1)
    failure = len(expected - observed) / max(len(expected), 1)
    unexpected = len(observed - expected)
    integrity = {
        "n_registered_expected": len(expected),
        "n_registered_observed": len(observed),
        "registered_prediction_coverage": coverage,
        "failure_rate": failure,
        "n_unexpected_prediction_keys": unexpected,
    }
    if observed != expected:
        return {**integrity, "score_integrity_pass": False}

    losses: dict[str, dict[str, float]] = {
        f"{name}_{metric}": {}
        for name in ("parent", "aligned", "shift17")
        for metric in ("signed", "point_absolute")
    }
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
            row = index[key]
            signed = float(target[position] - construct.wt_reactivity[position])
            absolute = abs(signed)
            for name in ("parent", "aligned", "shift17"):
                point = float(prediction[f"{name}_point"][row])
                losses[f"{name}_signed"][key] = abs(signed - point)
                losses[f"{name}_point_absolute"][key] = abs(absolute - abs(point))
            n_qualified += 1
    metrics = {
        f"{name}_signed_delta_mae": _puzzle_macro(losses[f"{name}_signed"])
        for name in ("parent", "aligned", "shift17")
    }
    metrics.update(
        {
            f"{name}_point_absolute_delta_mae": _puzzle_macro(
                losses[f"{name}_point_absolute"]
            )
            for name in ("parent", "aligned", "shift17")
        }
    )
    finite = n_qualified > 0 and all(np.isfinite(value) for value in metrics.values())
    return {
        **metrics,
        **integrity,
        "n_qualified_positions": n_qualified,
        "score_integrity_pass": bool(finite),
    }


def score_complete(merged: dict[str, Any], m2_csv: Path) -> dict[str, Any]:
    if (
        set(merged) != EXPECTED_MERGED_FIELDS
        or merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("phase") != "B5RP1"
        or merged.get("status") != MERGED_STATUS
        or merged.get("expected_folds") != EXPECTED_FOLDS
        or int(merged.get("expected_seed", -1)) != EXPECTED_SEED
        or not merged_integrity_pass(merged.get("merge_integrity", {}))
    ):
        raise ValueError("branch5 scorer requires one valid complete unscored merge")
    fold_rows = {int(row["outer_fold"]): row for row in merged.get("folds", [])}
    if sorted(fold_rows) != EXPECTED_FOLDS or len(fold_rows) != 20:
        raise ValueError("branch5 scorer requires unique folds0-19")

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if (
        identity.get("n_canonical_mutant_full_profiles") != 13976
        or identity.get("canonical_mutant_full_profile_identity")
        != "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("branch5 scorer requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(row.outer_fold): row for row in split["folds"]}
    rows = []
    integrity_errors = []
    for fold_id in EXPECTED_FOLDS:
        fold = folds[fold_id]
        source = fold_rows[fold_id]
        if str(source.get("held_puzzle")) != str(fold.held_puzzle):
            integrity_errors.append(f"fold{fold_id}:held_puzzle")
            continue
        held_records = [
            record for record in records if record.puzzle == fold.held_puzzle
        ]
        try:
            result = score_fold(
                univ,
                held_records,
                _load_prediction(Path(source["prediction_artifact"]), fold_id),
            )
        except (ValueError, KeyError, OSError, FloatingPointError) as error:
            integrity_errors.append(
                f"fold{fold_id}:prediction_or_metric_integrity:{error}"
            )
            continue
        result.update({"outer_fold": fold_id, "held_puzzle": str(fold.held_puzzle)})
        if result.get("score_integrity_pass") is not True:
            integrity_errors.append(f"fold{fold_id}:prediction_or_metric_integrity")
        rows.append(result)
    complete = len(rows) == 20 and not integrity_errors
    return {
        "schema_version": SCHEMA,
        "phase": SCORE_PHASE,
        "status": COMPLETE_STATUS if complete else INDETERMINATE_STATUS,
        "scores": rows if complete else [],
        "integrity_errors": integrity_errors,
        "complete_valid_score": complete,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES",
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_provenance_complete": complete,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    merged_json = args.merged_json.resolve()
    m2_csv = args.m2_csv.resolve()
    out_json = args.out_json.resolve()
    assert_score_authority(
        args.repo_root.resolve(),
        merged_json=merged_json,
        m2_csv=m2_csv,
        out_json=out_json,
    )
    if out_json.exists():
        raise FileExistsError("branch5 refuses to overwrite its one complete score")
    result = score_complete(json.loads(merged_json.read_text(encoding="utf-8")), m2_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_json.with_name(f"{out_json.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, out_json)
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0 if result["complete_valid_score"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
