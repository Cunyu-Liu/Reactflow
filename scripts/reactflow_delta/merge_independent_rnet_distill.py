#!/usr/bin/env python3
"""Build one complete target-free merge for independent RNet2 downstream folds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EVIDENCE_STATUS,
    EXPECTED_FOLDS,
    EXPECTED_PREDICTION_FIELDS,
    EXPECTED_SCHEDULE,
    EXPECTED_SEED,
    FOLD_SCHEMA,
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
    PRETRAIN_FILENAMES,
)
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    assert_run_authority,
)


SCHEMA = "reactflow_delta.independent_rnet_distill_complete_unscored_merge.v1"
STATUS = {
    "RND2": "RND2_COMPLETE_UNSCORED_ENGINEERING_SMOKE_MERGE_PASS",
    "RND3": "RND3_COMPLETE_UNSCORED_PREDICTION_MERGE_PASS",
}
EXPECTED_MERGED_FIELDS = frozenset(
    {"schema_version", "phase", "status", "folds", "merge_integrity"}
)
EXPECTED_FOLD_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "phase",
        "evidence_status",
        "metric_eligibility",
        "started_at_utc",
        "finished_at_utc",
        "git_commit",
        "command",
        "outer_fold",
        "held_puzzle",
        "seed",
        "point_epochs",
        "calibration_epochs",
        "training_device",
        "gpu_name",
        "contract_paths",
        "split",
        "pretraining_checkpoints",
        "point_checkpoints",
        "residual_checkpoints",
        "prediction_artifact",
        "training_histories",
        "n_train_cells",
        "n_registered_prediction_rows",
        "feature41_replay_max_abs_difference",
        "point_parameter_counts",
        "residual_parameter_counts",
        "invariants",
        "exit_code",
    }
)
EXPECTED_INVARIANTS = {
    "target_profile_identity_exact": True,
    "pretrained_source_pair_bound": True,
    "residual_heads_identical_before_downstream_step_one": True,
    "pretrained_encoders_different_before_downstream": True,
    "same_downstream_training_order_and_dropout_stream": True,
    "distillation_head_excluded_and_frozen_downstream": True,
    "point_frozen_during_calibration": True,
    "v10_residual_family_reused": True,
    "feature41_replay_at_1e_7": True,
    "authoritative_feature41_seed0_comparator_or_smoke_not_applicable": True,
    "median_constraint_all_held_rows": True,
    "cuda_only_training": True,
    "held_target_read": False,
    "held_score_computed": False,
    "partial_score_inspected": False,
    "prediction_contains_target_fields": False,
    "external_outcome_accessed": False,
}
MERGE_INTEGRITY = {
    "complete_fold_seed_universe": True,
    "unique_fold_seed_pairs": True,
    "prediction_only_schema": True,
    "target_free_all_runs": True,
    "target_identity_exact": True,
    "pretrained_source_pair_bound_all_runs": True,
    "residual_heads_equal_before_downstream_all_runs": True,
    "pretrained_encoders_different_before_downstream_all_runs": True,
    "same_downstream_training_order_and_dropout_stream_all_runs": True,
    "point_frozen_during_calibration_all_runs": True,
    "v10_residual_family_all_runs": True,
    "feature41_replay_all_runs": True,
    "authoritative_feature41_seed0_comparator_provenance_all_runs": True,
    "median_constraint_all_runs": True,
    "cuda_only_all_runs": True,
    "partial_scores_inspected": False,
    "external_outcome_accessed": False,
}
PRETRAIN_DIR = Path(
    "/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd1_pretrain"
)
_RESULT_RE = re.compile(r"^rnet_distill_fold_result_fold(\d+)_seed(\d+)\.json$")
_PREDICTION_RE = re.compile(r"^rnet_distill_predictions_fold(\d+)_seed(\d+)\.npz$")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _expected_paths(input_dir: Path, fold: int, seed: int) -> dict[str, Path]:
    stem = f"fold{fold}_seed{seed}"
    return {
        "result": input_dir / f"rnet_distill_fold_result_{stem}.json",
        "prediction": input_dir / f"rnet_distill_predictions_{stem}.npz",
        "candidate_point": input_dir / f"rnet_distill_candidate_point_{stem}.pt",
        "null_point": input_dir / f"rnet_distill_null_point_{stem}.pt",
        "feature41_residual": input_dir / f"rnet_distill_feature41_asymmetric_{stem}.pt",
        "candidate_residual": input_dir / f"rnet_distill_candidate_asymmetric_{stem}.pt",
        "null_residual": input_dir / f"rnet_distill_null_asymmetric_{stem}.pt",
    }


def _require_exact_path(observed: object, expected: Path, *, label: str) -> None:
    if not isinstance(observed, str) or Path(observed).resolve() != expected.resolve():
        raise RuntimeError(f"{label} path differs: observed={observed!r} expected={expected}")
    if not expected.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {expected}")


def _reject_unexpected_basenames(input_dir: Path, expected: set[str]) -> None:
    observed = {
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and path.name.startswith("rnet_distill_")
    }
    unexpected = sorted(observed - expected)
    if unexpected:
        raise RuntimeError(f"unexpected independent RNet artifacts: {unexpected}")


def prediction_checks(
    path: Path, *, fold: int, seed: int, expected_rows: int
) -> tuple[dict[str, bool], tuple[str, ...]]:
    with np.load(path, allow_pickle=True) as handle:
        names = frozenset(handle.files)
        keys = tuple(map(str, handle["keys"])) if "keys" in names else ()
        checks = {
            "exact_fields": names == EXPECTED_PREDICTION_FIELDS,
            "forbidden_fields_absent": not bool(names & FORBIDDEN_PREDICTION_FIELDS),
            "schema": "schema_version" in names
            and str(handle["schema_version"].item()) == PREDICTION_SCHEMA,
            "expected_rows": len(keys) == expected_rows and expected_rows > 0,
            "unique_keys": len(keys) == len(set(keys)),
            "biological_key_match": "biological_scoring_key" in names
            and keys == tuple(map(str, handle["biological_scoring_key"])),
            "fold": "outer_fold" in names
            and set(map(int, handle["outer_fold"])) == {fold},
            "seed": "seed" in names
            and set(map(int, handle["seed"])) == {seed},
            "covered": "registered_status" in names
            and set(map(str, handle["registered_status"])) == {"covered"},
            "row_alignment": all(
                name == "schema_version" or handle[name].shape[0] == expected_rows
                for name in names
            ),
            "finite": all(
                bool(np.isfinite(handle[name]).all())
                for name in names
                if handle[name].dtype.kind in "fiu"
            ),
        }
    return checks, keys


def _finite_history(value: object, expected_length: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_length:
        return False
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(array).all())


def _validate_row(
    row: dict[str, Any], *, input_dir: Path, phase: str, fold: int, seed: int
) -> tuple[str, ...]:
    if frozenset(row) != EXPECTED_FOLD_FIELDS:
        missing = sorted(EXPECTED_FOLD_FIELDS - frozenset(row))
        unexpected = sorted(frozenset(row) - EXPECTED_FOLD_FIELDS)
        raise RuntimeError(
            f"fold {fold} JSON schema differs: missing={missing} unexpected={unexpected}"
        )
    if row["schema_version"] != FOLD_SCHEMA or row["phase"] != phase:
        raise RuntimeError(f"fold {fold} schema or phase differs")
    if row["evidence_status"] != EVIDENCE_STATUS[phase] or row[
        "metric_eligibility"
    ] != EVIDENCE_STATUS[phase]:
        raise RuntimeError(f"fold {fold} evidence status differs")
    if int(row["outer_fold"]) != fold or int(row["seed"]) != seed:
        raise RuntimeError(f"fold {fold} identity differs")
    point_epochs, calibration_epochs = EXPECTED_SCHEDULE[phase]
    if int(row["point_epochs"]) != point_epochs or int(
        row["calibration_epochs"]
    ) != calibration_epochs:
        raise RuntimeError(f"fold {fold} epoch schedule differs")
    if row["training_device"] != "cuda:0" or not str(row["gpu_name"]):
        raise RuntimeError(f"fold {fold} lacks exact CUDA execution evidence")
    if int(row["exit_code"]) != 0:
        raise RuntimeError(f"fold {fold} did not exit successfully")
    if not isinstance(row["held_puzzle"], str) or not row["held_puzzle"]:
        raise RuntimeError(f"fold {fold} lacks held puzzle identity")
    split = row["split"]
    if split != {
        "name": "split_v4_lopo_puzzle",
        "seed": 20260813,
        "fold_universe": list(range(20)),
    }:
        raise RuntimeError(f"fold {fold} split binding differs")
    if row["invariants"] != EXPECTED_INVARIANTS:
        raise RuntimeError(f"fold {fold} invariants differ")
    if int(row["n_train_cells"]) < 1 or int(row["n_registered_prediction_rows"]) < 1:
        raise RuntimeError(f"fold {fold} has an empty train or prediction universe")
    replay = float(row["feature41_replay_max_abs_difference"])
    if not np.isfinite(replay) or replay > 1e-7:
        raise RuntimeError(f"fold {fold} feature41 replay differs")

    pretrain = row["pretraining_checkpoints"]
    if set(pretrain) != {"candidate", "null", "audit"}:
        raise RuntimeError(f"fold {fold} pretraining checkpoint fields differ")
    for name in ("candidate", "null", "audit"):
        _require_exact_path(
            pretrain[name],
            PRETRAIN_DIR / PRETRAIN_FILENAMES[name],
            label=f"fold {fold} pretraining {name}",
        )

    paths = _expected_paths(input_dir, fold, seed)
    point_checkpoints = row["point_checkpoints"]
    residual_checkpoints = row["residual_checkpoints"]
    if set(point_checkpoints) != {"candidate", "null"}:
        raise RuntimeError(f"fold {fold} point checkpoint fields differ")
    if set(residual_checkpoints) != {"feature41", "candidate", "null"}:
        raise RuntimeError(f"fold {fold} residual checkpoint fields differ")
    _require_exact_path(
        point_checkpoints["candidate"], paths["candidate_point"], label=f"fold {fold} candidate point"
    )
    _require_exact_path(
        point_checkpoints["null"], paths["null_point"], label=f"fold {fold} null point"
    )
    for name, path_key in (
        ("feature41", "feature41_residual"),
        ("candidate", "candidate_residual"),
        ("null", "null_residual"),
    ):
        _require_exact_path(
            residual_checkpoints[name], paths[path_key], label=f"fold {fold} {name} residual"
        )
    _require_exact_path(
        row["prediction_artifact"], paths["prediction"], label=f"fold {fold} prediction"
    )

    counts = row["point_parameter_counts"]
    if set(counts) != {"candidate", "null"} or len(set(map(int, counts.values()))) != 1:
        raise RuntimeError(f"fold {fold} point parameter match differs")
    if int(counts["candidate"]) < 1:
        raise RuntimeError(f"fold {fold} point parameter universe is empty")
    residual_counts = row["residual_parameter_counts"]
    if set(residual_counts) != {"feature41", "candidate", "null"} or set(
        map(int, residual_counts.values())
    ) != {63748}:
        raise RuntimeError(f"fold {fold} V10 residual family differs")
    histories = row["training_histories"]
    expected_histories = {
        "candidate_point",
        "null_point",
        "feature41_residual",
        "candidate_residual",
        "null_residual",
    }
    if set(histories) != expected_histories or not all(
        _finite_history(histories[name], point_epochs if name.endswith("point") else calibration_epochs)
        for name in expected_histories
    ):
        raise RuntimeError(f"fold {fold} training history differs")

    checks, keys = prediction_checks(
        paths["prediction"],
        fold=fold,
        seed=seed,
        expected_rows=int(row["n_registered_prediction_rows"]),
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"fold {fold} prediction checks failed: {failed}")
    return keys


def merge_folds(input_dir: Path, phase: str) -> dict[str, Any]:
    if phase not in EXPECTED_FOLDS:
        raise ValueError(f"unsupported independent RNet merge phase: {phase}")
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"independent RNet input directory is missing: {input_dir}")
    expected_folds = EXPECTED_FOLDS[phase]
    expected_seed = EXPECTED_SEED
    expected_names: set[str] = set()
    for fold in expected_folds:
        expected_names.update(
            path.name for path in _expected_paths(input_dir, fold, expected_seed).values()
        )
    _reject_unexpected_basenames(input_dir, expected_names)

    rows: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    held_puzzles: set[str] = set()
    point_parameter_count: int | None = None
    for fold in expected_folds:
        paths = _expected_paths(input_dir, fold, expected_seed)
        match = _RESULT_RE.fullmatch(paths["result"].name)
        if match is None or tuple(map(int, match.groups())) != (fold, expected_seed):
            raise RuntimeError(f"fold {fold} result basename differs")
        if not paths["result"].is_file():
            raise FileNotFoundError(f"missing fold result: {paths['result']}")
        row = _read_json(paths["result"])
        keys = _validate_row(
            row,
            input_dir=input_dir,
            phase=phase,
            fold=fold,
            seed=expected_seed,
        )
        duplicate_keys = all_keys & set(keys)
        if duplicate_keys:
            raise RuntimeError(f"biological keys repeat across folds: {len(duplicate_keys)}")
        all_keys.update(keys)
        held_puzzle = str(row["held_puzzle"])
        if held_puzzle in held_puzzles:
            raise RuntimeError(f"held puzzle repeats across folds: {held_puzzle}")
        held_puzzles.add(held_puzzle)
        current_count = int(row["point_parameter_counts"]["candidate"])
        if point_parameter_count is None:
            point_parameter_count = current_count
        elif current_count != point_parameter_count:
            raise RuntimeError("point parameter count differs across folds")
        rows.append(row)

    if len(rows) != len(expected_folds):
        raise RuntimeError("independent RNet fold universe is incomplete")
    result = {
        "schema_version": SCHEMA,
        "phase": phase,
        "status": STATUS[phase],
        "folds": rows,
        "merge_integrity": dict(MERGE_INTEGRITY),
    }
    if frozenset(result) != EXPECTED_MERGED_FIELDS:
        raise AssertionError("independent RNet merged field universe changed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(EXPECTED_FOLDS), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_run_authority(args.repo_root.resolve(), args.phase)
    if args.out_json.exists():
        raise FileExistsError(
            f"refusing to overwrite independent RNet merge: {args.out_json}"
        )
    result = merge_folds(args.input_dir, args.phase)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
