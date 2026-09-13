#!/usr/bin/env python3
"""DeltaFlow score-once (Task 9.1): four primary metrics + joint metrics.

This is the single permitted held-outcome access of the DeltaFlow phase.
It runs only after every seed of the frozen grid has a complete, published
fold universe and a per-seed canonical merge; it reads the merged
prediction files plus the per-fold joint-sample sidecars.

Fold slicing uses the merge audit's fold_rows / fold_flow_rows boundaries
(the merged arrays are concatenations in fold order with an outer_fold
column that is cross-checked against the audit).

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
joint sample exports, computed per seed.

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
FOLD_UNIVERSE = tuple(range(20))
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


def load_merge_audit(experiment_dir: Path, seed: int) -> dict[str, Any]:
    path = experiment_dir / f"deltaflow_merge_audit_seed{seed}.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != MERGED_SCHEMA or int(audit["seed"]) != seed:
        raise ValueError(f"merge audit mismatch in {path}")
    if sorted(int(f) for f in audit["folds"]) != list(FOLD_UNIVERSE):
        raise ValueError(f"merge audit fold universe mismatch in {path}")
    return audit


def fold_slices(audit: dict[str, Any]) -> dict[int, tuple[int, int, int, int]]:
    """Per-fold (key_start, key_stop, flow_start, flow_stop) boundaries."""

    boundaries: dict[int, tuple[int, int, int, int]] = {}
    key_offset = 0
    flow_offset = 0
    for fold in FOLD_UNIVERSE:
        key_count = int(audit["fold_rows"][str(fold)])
        flow_count = int(audit["fold_flow_rows"][str(fold)])
        boundaries[fold] = (key_offset, key_offset + key_count, flow_offset, flow_offset + flow_count)
        key_offset += key_count
        flow_offset += flow_count
    return boundaries


def fold_view(
    merged: dict[str, np.ndarray], fold: int, slices: dict[int, tuple[int, int, int, int]]
) -> dict[str, Any]:
    key_start, key_stop, flow_start, flow_stop = slices[fold]
    outer_fold = np.asarray(merged["outer_fold"])
    if not (outer_fold[key_start:key_stop] == fold).all():
        raise ValueError(f"merged outer_fold column disagrees with audit at fold {fold}")
    keys = list(map(str, merged["keys"][key_start:key_stop]))
    view: dict[str, Any] = {
        "keys": keys,
        "key_index": {key: index for index, key in enumerate(keys)},
        "flow_keys": list(map(str, merged["flow_keys"][flow_start:flow_stop])),
        "flow_index": {
            key: index
            for index, key in enumerate(map(str, merged["flow_keys"][flow_start:flow_stop]))
        },
    }
    for name, value in merged.items():
        if name in ("keys", "flow_keys", "outer_fold", "seed", "schema_version",
                    "biological_scoring_key"):
            continue
        if value.shape[0] == outer_fold.shape[0]:
            view[name] = np.asarray(value[key_start:key_stop])
        elif value.shape[0] == len(merged["flow_keys"]):
            sliced = np.asarray(value[flow_start:flow_stop])
            if sliced.ndim == 2 and sliced.shape[1] == 1:
                sliced = sliced.reshape(-1)
            view[name] = sliced
    return view


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


def _assembly_mixtures(views: dict[int, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    assembly: dict[str, dict[str, np.ndarray]] = {}
    for arm in FLOW_ARMS:
        locations = np.stack(
            [views[seed][f"{arm}_locations"].astype(np.float64) for seed in SEED_UNIVERSE],
            axis=1,
        )
        scales = np.stack(
            [views[seed][f"{arm}_scales"].astype(np.float64) for seed in SEED_UNIVERSE],
            axis=1,
        )
        expected = np.sum(
            np.stack(
                [
                    views[seed][f"{arm}_expected_absolute_delta"].astype(np.float64)
                    for seed in SEED_UNIVERSE
                ],
                axis=0,
            )
            * ASSEMBLY_WEIGHT,
            axis=0,
        )
        assembly[arm] = {
            "weights": np.full((locations.shape[0], len(SEED_UNIVERSE)), ASSEMBLY_WEIGHT),
            "locations": locations,
            "scales": scales,
            "point": np.sum(ASSEMBLY_WEIGHT * locations, axis=1),
            "expected": expected,
        }
    return assembly


def _seed_mixtures(view: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    mixtures: dict[str, dict[str, np.ndarray]] = {}
    for arm in FLOW_ARMS:
        locations = view[f"{arm}_locations"].astype(np.float64)
        mixtures[f"{arm}_seed"] = {
            "weights": np.ones((locations.shape[0], 1), dtype=np.float64),
            "locations": locations[:, None],
            "scales": view[f"{arm}_scales"].astype(np.float64)[:, None],
            "point": locations,
            "expected": view[f"{arm}_expected_absolute_delta"].astype(np.float64),
        }
    return mixtures


def _mixture_metric_values(
    rows: list[dict[str, Any]],
    flow_index: dict[str, int],
    mixture: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    values = {suffix: {} for suffix in METRIC_SUFFIXES}
    for row in rows:
        row_index = np.asarray(
            [flow_index[key] for key in row["keys"]], dtype=np.int64
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
    view: dict[str, Any],
    name: str,
) -> dict[str, dict[str, float]]:
    index = view["key_index"]
    values = {suffix: {} for suffix in METRIC_SUFFIXES}
    has_point = f"{name}_point" in view
    has_distribution = (
        f"{name}_expected_absolute_delta" in view
        and f"{name}_weights" in view
        and f"{name}_locations" in view
        and f"{name}_scales" in view
    )
    if not has_point and not has_distribution:
        return values
    for row in rows:
        row_index = np.asarray([index[key] for key in row["keys"]], dtype=np.int64)
        if has_point:
            point = np.asarray(view[f"{name}_point"])[row_index].astype(np.float64)
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
        if has_distribution:
            expected = np.asarray(view[f"{name}_expected_absolute_delta"])[
                row_index
            ].astype(np.float64)
            values["distribution_absolute_delta_mae"].update(
                {
                    key: float(value)
                    for key, value in zip(row["keys"], np.abs(row["absolute"] - expected))
                }
            )
            values["crps"].update(
                {
                    key: float(value)
                    for key, value in zip(
                        row["keys"],
                        weighted_gaussian_mixture_crps(
                            np.asarray(view[f"{name}_locations"])[row_index].astype(
                                np.float64
                            ),
                            np.asarray(view[f"{name}_scales"])[row_index].astype(
                                np.float64
                            ),
                            np.asarray(view[f"{name}_weights"])[row_index].astype(
                                np.float64
                            ),
                            row["signed"],
                        ),
                    )
                }
            )
    return values


def _joint_mutants(
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
    views: dict[int, dict[str, Any]],
    experiment_dirs: dict[int, Path],
    joint_seed_override: int | None = None,
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    held_records = [
        record for record in univ.get_records() if record.puzzle == fold.held_puzzle
    ]
    rows = _position_table(univ, held_records)
    expected_keys = set(views[SEED_UNIVERSE[0]]["keys"])
    for seed in SEED_UNIVERSE:
        if set(views[seed]["keys"]) != expected_keys:
            raise ValueError(f"fold {fold_id} seed {seed} key universe mismatch")
    flow_index = views[SEED_UNIVERSE[0]]["flow_index"]
    for seed in SEED_UNIVERSE[1:]:
        if views[seed]["flow_index"] != flow_index:
            raise ValueError(f"fold {fold_id} seed {seed} flow key mismatch")

    assembly = _assembly_mixtures(views)
    seed_mixtures = {}
    for seed in SEED_UNIVERSE:
        for label, mixture in _seed_mixtures(views[seed]).items():
            seed_mixtures[f"{label}{seed}"] = mixture
    result: dict[str, Any] = {}
    for label, mixture in {**assembly, **seed_mixtures}.items():
        values = _mixture_metric_values(rows, flow_index, mixture)
        for suffix in METRIC_SUFFIXES:
            result[f"{label}_{suffix}"] = _puzzle_macro(values[suffix])
    for name in REFERENCE_ARMS:
        values = _reference_values(rows, views[SEED_UNIVERSE[0]], name)
        for suffix in METRIC_SUFFIXES:
            if values[suffix]:
                result[f"{name}_{suffix}"] = _puzzle_macro(values[suffix])
    result["n_qualified_mutants"] = len(rows)
    result["n_qualified_positions"] = int(sum(len(row["keys"]) for row in rows))
    result["outer_fold"] = fold_id
    result["held_puzzle"] = str(fold.held_puzzle)

    joint_summary: dict[str, Any] = {}
    for seed in SEED_UNIVERSE:
        joint_seed = joint_seed_override if joint_seed_override is not None else seed
        joint_dir = experiment_dirs[joint_seed]
        candidate_mutants = _joint_mutants(
            rows, held_records, joint_dir, fold_id, joint_seed,
            "candidate_samples",
        )
        null_mutants = _joint_mutants(
            rows, held_records, joint_dir, fold_id, joint_seed,
            "null_samples",
        )
        candidate_score = score_arm(
            arm="candidate", mutants=candidate_mutants, b_perm=DEFAULT_B_PERM, seed=seed
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

    experiment_dirs = {
        seed: path for seed, path in map(parse_seed_dir, args.experiment_dir)
    }
    if tuple(sorted(experiment_dirs)) != SEED_UNIVERSE:
        raise ValueError(
            f"score-once requires exactly seeds 0-4, got {sorted(experiment_dirs)}"
        )
    merged_by_seed = {
        seed: load_merged(experiment_dirs[seed], seed) for seed in SEED_UNIVERSE
    }
    audits = {
        seed: load_merge_audit(experiment_dirs[seed], seed) for seed in SEED_UNIVERSE
    }
    boundaries = fold_slices(audits[SEED_UNIVERSE[0]])
    for seed in SEED_UNIVERSE[1:]:
        if fold_slices(audits[seed]) != boundaries:
            raise ValueError(f"fold boundaries differ across seeds at seed {seed}")

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
    if sorted(folds) != list(FOLD_UNIVERSE):
        raise RuntimeError("score-once requires folds 0-19")

    scores = []
    for fold_id in FOLD_UNIVERSE:
        views = {
            seed: fold_view(merged_by_seed[seed], fold_id, boundaries)
            for seed in SEED_UNIVERSE
        }
        scores.append(
            score_fold(univ, folds[fold_id], views, experiment_dirs)
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
