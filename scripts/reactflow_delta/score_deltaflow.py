#!/usr/bin/env python3
"""DeltaFlow score-once (Task 9.1): four primary metrics + joint metrics.

This is the single permitted held-outcome access of the DeltaFlow phase.
It runs only after every seed of the frozen grid has a complete, published
fold universe and a per-seed canonical merge; it reads the merged
prediction files plus the per-fold joint-sample sidecars.

Scoring semantics follow the historical V10/V11 scorer family verbatim:
  - qualified positions: WT-observed AND finite mutant target;
  - signed delta = target - WT reactivity at the qualified positions;
  - signed MAE  = |signed - point| (point = mixture mean for assemblies);
  - point absolute MAE = |absolute - |point||;
  - distribution absolute MAE = |absolute - E|X||;
  - CRPS via the exact weighted-Gaussian-mixture CRPS;
  - per-fold aggregation: position -> mutant -> method -> puzzle macro.

Assembly (freeze F1): the five seeds mix equally (0.2 per seed) as a
5-component mixture per position; per-seed metrics are also reported for
the >=4/5-seeds direction condition.  Flow arms (candidate/null) are the
primary F2 comparison; feature41 / V8 / historical-V10 are references.

Joint-structure metrics (F3) use deltaflow_joint_metrics on the per-fold
joint sample exports, computed per seed and summarized as the median
across seeds.

This script computes and writes the score artifact only.  Judgement
(thresholds, t-test, MDE, verdicts) lives in qualify_deltaflow.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reactflow_delta.deltaflow_joint_metrics import (
    DEFAULT_B_PERM,
    DISTAL_K,
    compare_arms,
    score_arm,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import weighted_gaussian_mixture_crps
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v6_probe import _puzzle_macro
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.deltaflow_score.v1"
MERGED_SCHEMA = "reactflow_delta.deltaflow_merged_prediction.v1"
SPLIT_SEED = 20260813
SEED_UNIVERSE = (0, 1, 2, 3, 4)
ASSEMBLY_WEIGHT = 0.2
FLOW_ARMS = ("flow_candidate", "flow_null")
REFERENCE_ARMS = ("feature41", "v8", "historical_v10")
METRIC_SUFFIXES = (
    "signed_delta_mae",
    "point_absolute_delta_mae",
    "distribution_absolute_delta_mae",
    "crps",
)


def parse_seed_dir(raw: str) -> tuple[int, Path]:
    seed_text, _, path_text = raw.partition("=")
    return int(seed_text), Path(path_text)


def load_merged(experiment_dir: Path, seed: int) -> dict[str, np.ndarray]:
    path = experiment_dir / f"deltaflow_merged_predictions_seed{seed}.npz"
    with np.load(path, allow_pickle=True) as handle:
        data = {name: handle[name] for name in handle.files}
    if str(data["schema_version"].item()) != MERGED_SCHEMA:
        raise ValueError(f"merged prediction schema mismatch in {path}")
    if set(map(int, data["seed"])) != {seed}:
        raise ValueError(f"merged seed mismatch in {path}")
    keys = list(map(str, data["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate merged keys in {path}")
    return data


def joint_dir_for(experiment_dir: Path, fold: int, seed: int) -> Path:
    return (
        experiment_dir
        / f"deltaflow_predictions_fold{fold}_seed{seed}_joint_samples"
    )


def _position_table(univ: M2Universe, held_records: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        qualified = construct.wt_observed & np.isfinite(target)
        positions = np.flatnonzero(qualified)
        if not len(positions):
            continue
        signed = target[positions] - construct.wt_reactivity[positions]
        rows.append(
            {
                "record": record,
                "construct_id": record.construct_id,
                "positions": positions,
                "keys": [
                    _bio_key(univ, record, int(position)) for position in positions
                ],
                "signed": signed,
                "absolute": np.abs(signed),
                "edit_position": int(record.full_pos),
                "length": len(construct.sequence),
            }
        )
    return rows


def _mixtures(
    merged_by_seed: dict[int, dict[str, np.ndarray]],
    flow_key_index: dict[str, int],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Assembly mixtures and per-seed single-component mixtures per arm.

    The assembly stacks the five seed distributions as a 5-component
    equally-weighted mixture; the point estimate of an assembly is its
    mixture mean and its E|X| the component-weighted mean of E|X|.
    """

    n_rows = len(flow_key_index)
    assembly: dict[str, dict[str, np.ndarray]] = {}
    for arm in FLOW_ARMS:
        locations = np.stack(
            [
                merged_by_seed[seed][f"{arm}_locations"].astype(np.float64)
                for seed in SEED_UNIVERSE
            ],
            axis=1,
        )
        scales = np.stack(
            [
                merged_by_seed[seed][f"{arm}_scales"].astype(np.float64)
                for seed in SEED_UNIVERSE
            ],
            axis=1,
        )
        expected = sum(
            ASSEMBLY_WEIGHT
            * merged_by_seed[seed][f"{arm}_expected_absolute_delta"].astype(np.float64)
            for seed in SEED_UNIVERSE
        )
        assembly[arm] = {
            "weights": np.full((n_rows, len(SEED_UNIVERSE)), ASSEMBLY_WEIGHT),
            "locations": locations,
            "scales": scales,
            "point": np.sum(ASSEMBLY_WEIGHT * locations, axis=1),
            "expected": expected,
        }
    per_seed: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for seed in SEED_UNIVERSE:
        merged = merged_by_seed[seed]
        for arm in FLOW_ARMS:
            per_seed[f"{arm}_seed{seed}"] = {
                "weights": np.ones((n_rows, 1), dtype=np.float64),
                "locations": merged[f"{arm}_locations"].astype(np.float64)[:, None],
                "scales": merged[f"{arm}_scales"].astype(np.float64)[:, None],
                "point": merged[f"{arm}_locations"].astype(np.float64),
                "expected": merged[f"{arm}_expected_absolute_delta"].astype(
                    np.float64
                ),
            }
    return assembly, per_seed


def _mixture_metric_values(
    rows: list[dict[str, Any]],
    flow_key_index: dict[str, int],
    mixture: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    values = {suffix: {} for suffix in METRIC_SUFFIXES}
    for row in rows:
        row_index = np.asarray(
            [flow_key_index[key] for key in row["keys"]], dtype=np.int64
        )
        point = mixture["point"][row_index]
        values["signed_delta_mae"].update(
            {
                key: float(value)
                for key, value in zip(row["keys"], np.abs(row["signed"] - point))
            }
        )
        values["point_absolute_delta_mae"].update(
            {
                key: float(value)
                for key, value in zip(row["keys"], np.abs(row["absolute"] - np.abs(point)))
            }
        )
        values["distribution_absolute_delta_mae"].update(
            {
                key: float(value)
                for key, value in zip(
                    row["keys"],
                    np.abs(row["absolute"] - mixture["expected"][row_index]),
                )
            }
        )
        values["crps"].update(
            {
                key: float(value)
                for key, value in zip(
                    row["keys"],
                    weighted_gaussian_mixture_crps(
                        mixture["locations"][row_index],
                        mixture["scales"][row_index],
                        mixture["weights"][row_index],
                        row["signed"],
                    ),
                )
            }
        )
    return values


def _reference_values(
    rows: list[dict[str, Any]],
    merged: dict[str, np.ndarray],
    name: str,
) -> dict[str, dict[str, float]]:
    index = {
        key: row_number for row_number, key in enumerate(map(str, merged["keys"]))
    }
    values = {suffix: {} for suffix in METRIC_SUFFIXES}
    for row in rows:
        row_index = np.asarray([index[key] for key in row["keys"]], dtype=np.int64)
        point = merged[f"{name}_point"][row_index].astype(np.float64)
        values["signed_delta_mae"].update(
            {
                key: float(value)
                for key, value in zip(row["keys"], np.abs(row["signed"] - point))
            }
        )
        values["point_absolute_delta_mae"].update(
            {
                key: float(value)
                for key, value in zip(row["keys"], np.abs(row["absolute"] - np.abs(point)))
            }
        )
        if f"{name}_expected_absolute_delta" in merged:
            expected = merged[f"{name}_expected_absolute_delta"][row_index].astype(
                np.float64
            )
            values["distribution_absolute_delta_mae"].update(
                {
                    key: float(value)
                    for key, value in zip(row["keys"], np.abs(row["absolute"] - expected))
                }
            )
            if (
                f"{name}_weights" in merged
                and f"{name}_locations" in merged
                and f"{name}_scales" in merged
            ):
                values["crps"].update(
                    {
                        key: float(value)
                        for key, value in zip(
                            row["keys"],
                            weighted_gaussian_mixture_crps(
                                merged[f"{name}_locations"][row_index].astype(
                                    np.float64
                                ),
                                merged[f"{name}_scales"][row_index].astype(np.float64),
                                merged[f"{name}_weights"][row_index].astype(np.float64),
                                row["signed"],
                            ),
                        )
                    }
                )
    return values


def _joint_mutants(
    univ: M2Universe,
    rows: list[dict[str, Any]],
    held_records: list[Any],
    experiment_dir: Path,
    fold_id: int,
    seed: int,
    arm_field: str,
) -> list[dict[str, Any]]:
    joint_dir = joint_dir_for(experiment_dir, fold_id, seed)
    if not joint_dir.is_dir():
        raise FileNotFoundError(f"joint samples missing for fold {fold_id} seed {seed}")
    by_construct: dict[str, list[Any]] = {}
    for record in held_records:
        by_construct.setdefault(record.construct_id, []).append(record)
    mutants: list[dict[str, Any]] = []
    for row in rows:
        record = row["record"]
        samples_path = joint_dir / f"{record.construct_id}.npz"
        if not samples_path.is_file():
            raise FileNotFoundError(f"joint sample file missing: {samples_path}")
        with np.load(samples_path, allow_pickle=False) as handle:
            samples = np.asarray(handle[arm_field], dtype=np.float64)
        record_index = by_construct[record.construct_id].index(record)
        observed = np.full(row["length"], np.nan)
        observed[row["positions"]] = row["signed"]
        positions = np.arange(row["length"])
        distal = (np.abs(positions - row["edit_position"]) > DISTAL_K) & np.isfinite(
            observed
        )
        mutants.append(
            {
                "construct_id": record.construct_id,
                "edit_position": row["edit_position"],
                "length": row["length"],
                "observed_delta": observed,
                "samples": samples[:, record_index, :],
                "distal_mask": distal,
            }
        )
    return mutants


def score_fold(
    univ: M2Universe,
    fold: Any,
    merged_by_seed: dict[int, dict[str, np.ndarray]],
    experiment_dirs: dict[int, Path],
    flow_key_index: dict[str, int],
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    held_records = [
        record
        for record in univ.get_records()
        if record.puzzle == fold.held_puzzle
    ]
    rows = _position_table(univ, held_records)
    expected_keys = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(
            len(univ.get_construct(record.construct_id).sequence)
        )
    }
    for seed in SEED_UNIVERSE:
        merged_keys = set(map(str, merged_by_seed[seed]["keys"]))
        if merged_keys != expected_keys:
            raise ValueError(f"fold {fold_id} seed {seed} key universe mismatch")

    assembly, per_seed = _mixtures(merged_by_seed, flow_key_index)
    result: dict[str, Any] = {}
    for label, mixture in {**assembly, **per_seed}.items():
        values = _mixture_metric_values(rows, flow_key_index, mixture)
        for suffix in METRIC_SUFFIXES:
            result[f"{label}_{suffix}"] = _puzzle_macro(values[suffix])
    for name in REFERENCE_ARMS:
        values = _reference_values(rows, merged_by_seed[SEED_UNIVERSE[0]], name)
        for suffix in METRIC_SUFFIXES:
            if values[suffix]:
                result[f"{name}_{suffix}"] = _puzzle_macro(values[suffix])
    result["n_qualified_mutants"] = len(rows)
    result["n_qualified_positions"] = int(sum(len(row["keys"]) for row in rows))
    result["outer_fold"] = fold_id
    result["held_puzzle"] = str(fold.held_puzzle)

    joint_summary: dict[str, Any] = {}
    for seed in SEED_UNIVERSE:
        candidate_mutants = _joint_mutants(
            univ, rows, held_records, experiment_dirs[seed], fold_id, seed,
            "candidate_samples",
        )
        null_mutants = _joint_mutants(
            univ, rows, held_records, experiment_dirs[seed], fold_id, seed,
            "null_samples",
        )
        candidate_score = score_arm(
            arm="candidate", mutants=candidate_mutants, b_perm=DEFAULT_B_PERM,
            seed=seed,
        )
        null_score = score_arm(
            arm="null", mutants=null_mutants, b_perm=DEFAULT_B_PERM, seed=seed
        )
        joint_summary[f"seed{seed}"] = compare_arms(candidate_score, null_score)
    result["joint_metrics_by_seed"] = joint_summary
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        action="append",
        required=True,
        help="seed=DIR (merged npz + fold artifacts for that seed)",
    )
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    experiment_dirs = {seed: path for seed, path in map(parse_seed_dir, args.experiment_dir)}
    if tuple(sorted(experiment_dirs)) != SEED_UNIVERSE:
        raise ValueError(f"score-once requires exactly seeds 0-4, got {sorted(experiment_dirs)}")
    merged_by_seed = {
        seed: load_merged(experiment_dirs[seed], seed) for seed in SEED_UNIVERSE
    }
    flow_key_index = {
        key: index
        for index, key in enumerate(map(str, merged_by_seed[0]["flow_keys"]))
    }
    for seed in SEED_UNIVERSE[1:]:
        if {
            key: index
            for index, key in enumerate(map(str, merged_by_seed[seed]["flow_keys"]))
        } != flow_key_index:
            raise ValueError(f"flow key universe differs at seed {seed}")

    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("score-once requires exact canonical target identity")
    univ.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in univ.get_records()}), seed=SPLIT_SEED
    )
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    if sorted(folds) != list(range(20)):
        raise RuntimeError("score-once requires folds 0-19")

    scores = []
    for fold_id in range(20):
        scores.append(
            score_fold(univ, folds[fold_id], merged_by_seed, experiment_dirs, flow_key_index)
        )
    result = {
        "schema_version": SCHEMA,
        "status": "DELTAFLOW_COMPLETE_SCORE_PASS",
        "scores": scores,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "seeds": list(SEED_UNIVERSE),
        "assembly_weight": ASSEMBLY_WEIGHT,
        "held_score_computed_once": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
