#!/usr/bin/env python3
"""Assemble fixed seeds 0--4 into the original R2M4 formal mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v3 import CANDIDATE
from scripts.reactflow_delta.run_model_rescue_m2_v1 import score_predictions
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v3_formal_run.v1"
FORMAL_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v3_formal_prediction.v1"
SEEDS = list(range(5))


def assert_r3m4_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "R3M4":
        raise RuntimeError("formal assembler is closed unless active phase is R3M4")
    if active["gate_state"]["R3M3"] != "PASS":
        raise RuntimeError("formal assembler requires R3M3 PASS")
    if active["gate_state"]["R3M4"] != "IN_PROGRESS":
        raise RuntimeError("formal assembler requires R3M4 IN_PROGRESS")
    if active["new_external_outcome_access_allowed"] is not False:
        raise RuntimeError("formal assembler requires external outcomes locked")


def combine_five_seed_predictions(
    predictions: list[dict[str, np.ndarray]], candidate_id: str
) -> dict[str, np.ndarray]:
    if len(predictions) != 5:
        raise ValueError("formal prediction requires exactly five seeds")
    keys = predictions[0]["keys"]
    expected_components = 1 if candidate_id == BASELINE else 2
    for prediction in predictions:
        if not np.array_equal(prediction["keys"], keys):
            raise ValueError("seed prediction key universes differ")
        if prediction["locations"].shape[1] != expected_components:
            raise ValueError(f"{candidate_id} seed prediction has wrong components")
    if len(set(map(str, keys))) != len(keys):
        raise ValueError("formal key universe contains duplicates")
    locations = np.concatenate([prediction["locations"] for prediction in predictions], axis=1)
    scales = np.concatenate([prediction["scales"] for prediction in predictions], axis=1)
    weights = np.concatenate(
        [prediction["weights"] / 5.0 for prediction in predictions], axis=1
    )
    weights /= weights.sum(axis=1, keepdims=True)
    seed_point_means = np.stack(
        [
            np.sum(prediction["weights"] * prediction["locations"], axis=1)
            / prediction["weights"].sum(axis=1)
            for prediction in predictions
        ],
        axis=1,
    )
    point_mean = seed_point_means.mean(axis=1)
    if not np.allclose(
        np.sum(weights * locations, axis=1), point_mean, atol=1e-7, rtol=0
    ):
        raise RuntimeError("formal mixture mean differs from five-seed point mean")
    n = len(keys)
    return {
        "schema_version": np.asarray(FORMAL_PREDICTION_SCHEMA),
        "keys": keys,
        "candidate_id": np.full(n, candidate_id, dtype=object),
        "seed_universe": np.asarray(SEEDS, dtype=np.int64),
        "seed_point_means": seed_point_means.astype(np.float64),
        "point_mean": point_mean.astype(np.float64),
        "locations": locations.astype(np.float64),
        "scales": scales.astype(np.float64),
        "weights": weights.astype(np.float64),
        "registered_status": np.full(n, "covered", dtype=object),
    }


def _load_prediction(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        return {name: handle[name] for name in handle.files}


def assemble_formal_fold(
    *,
    univ: M2Universe,
    fold: Any,
    all_records: list[Any],
    seed_result_dirs: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    if len(seed_result_dirs) != 5:
        raise ValueError("formal assembly requires five seed result directories")
    baseline_predictions = []
    candidate_predictions = []
    seed_artifacts = []
    for seed, result_dir in enumerate(seed_result_dirs):
        path = result_dir / f"v3_fold_result_fold{fold.outer_fold}_seed{seed}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("seed", -1)) != seed or int(row.get("outer_fold", -1)) != int(
            fold.outer_fold
        ):
            raise ValueError(f"wrong fold/seed identity in {path}")
        if row.get("baseline", {}).get("model_id") != BASELINE:
            raise ValueError(f"wrong baseline in {path}")
        if row.get("candidate", {}).get("candidate_id") != CANDIDATE:
            raise ValueError(f"wrong candidate in {path}")
        invariants = row.get("invariants", {})
        if not (
            invariants.get("held_target_error_mask_invariance") is True
            and invariants.get("inner_crossfit_complete") is True
            and invariants.get("method_used_as_gate_input") is False
            and invariants.get("residual_changed_point_mean") is False
        ):
            raise ValueError(f"seed {seed} fold {fold.outer_fold} invariants failed")
        baseline_predictions.append(
            _load_prediction(row["baseline"]["prediction_artifact"])
        )
        candidate_predictions.append(
            _load_prediction(row["candidate"]["prediction_artifact"])
        )
        seed_artifacts.append(
            {
                "seed": seed,
                "fold_result": str(path),
                "baseline_prediction": row["baseline"]["prediction_artifact"],
                "candidate_prediction": row["candidate"]["prediction_artifact"],
                "inner_crossfit_ledger": row["candidate"]["inner_crossfit_ledger"],
                "target_error_mask_invariance": True,
                "inner_crossfit_complete": True,
            }
        )
    baseline_formal = combine_five_seed_predictions(baseline_predictions, BASELINE)
    candidate_formal = combine_five_seed_predictions(candidate_predictions, CANDIDATE)
    if not np.array_equal(baseline_formal["keys"], candidate_formal["keys"]):
        raise ValueError("formal baseline and candidate key universes differ")
    baseline_path = out_dir / f"r3m4_formal_b1_fold{fold.outer_fold}.npz"
    candidate_path = out_dir / f"r3m4_formal_candidate_fold{fold.outer_fold}.npz"
    np.savez_compressed(baseline_path, **baseline_formal)
    np.savez_compressed(candidate_path, **candidate_formal)
    held_records = [record for record in all_records if record.puzzle == fold.held_puzzle]
    return {
        "schema_version": SCHEMA,
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "all_seed_target_error_mask_invariance": True,
        "all_seed_inner_crossfit_complete": True,
        "baseline": {
            "model_id": BASELINE,
            "seed_universe": SEEDS,
            "prediction_artifact": str(baseline_path),
            "score": score_predictions(baseline_formal, univ, held_records),
        },
        "candidate": {
            "model_id": CANDIDATE,
            "seed_universe": SEEDS,
            "prediction_artifact": str(candidate_path),
            "score": score_predictions(candidate_formal, univ, held_records),
        },
        "seed_artifacts": seed_artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-result-dirs", type=Path, nargs=5, required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    assert_r3m4_authority(args.repo_root.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    universe = M2Universe(args.m2_csv)
    universe.build()
    records = universe.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = {int(value) for value in args.folds.split(",") if value}
    folds = [fold for fold in split["folds"] if fold.outer_fold in selected]
    if not folds:
        raise ValueError("no requested outer folds")
    for fold in folds:
        result = assemble_formal_fold(
            univ=universe,
            fold=fold,
            all_records=records,
            seed_result_dirs=args.seed_result_dirs,
            out_dir=args.out_dir,
        )
        path = args.out_dir / f"v3_formal_fold_result_fold{fold.outer_fold}.json"
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[R3M4] fold={fold.outer_fold} artifact={path} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
