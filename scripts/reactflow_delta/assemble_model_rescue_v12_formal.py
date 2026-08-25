#!/usr/bin/env python3
"""Assemble the frozen V12 five-seed equal mixture without target access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.merge_model_rescue_v12 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v9 import expected_absolute_delta
from scripts.reactflow_delta.model_rescue_v12 import PREDICTION_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v12_formal_assembly.v1"
FORMAL_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v12_formal_prediction.v1"


def _load_prediction(path: Path, fold: int, seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: np.asarray(handle[name]) for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V12 formal source schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold} or set(
        map(int, prediction["seed"])
    ) != {seed}:
        raise ValueError(f"V12 formal source fold or seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)) or not np.array_equal(
        prediction["keys"], prediction["biological_scoring_key"]
    ):
        raise ValueError(f"V12 formal source keys are invalid in {path}")
    return prediction


def _expected_absolute(
    weights: np.ndarray, locations: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        values = expected_absolute_delta(
            torch.tensor(weights, dtype=torch.float64),
            torch.tensor(locations, dtype=torch.float64),
            torch.tensor(scales, dtype=torch.float64),
        )
    return values.numpy()


def assemble_fold(
    rows: list[dict[str, Any]], *, fold: int, out_dir: Path
) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in rows}
    if sorted(by_seed) != list(range(5)):
        raise ValueError(f"V12 formal fold {fold} requires seeds0-4")
    source = [
        _load_prediction(Path(by_seed[seed]["prediction_artifact"]), fold, seed)
        for seed in range(5)
    ]
    reference_keys = list(map(str, source[0]["keys"]))
    deterministic = (
        "feature41_point",
        "v11_parent_point",
        "feature41_weights",
        "feature41_locations",
        "feature41_scales",
        "feature41_expected_absolute_delta",
        "parent_weights",
        "parent_locations",
        "parent_scales",
        "parent_expected_absolute_delta",
        "historical_v10_weights",
        "historical_v10_locations",
        "historical_v10_scales",
        "historical_v10_expected_absolute_delta",
    )
    for seed, prediction in enumerate(source[1:], start=1):
        if list(map(str, prediction["keys"])) != reference_keys:
            raise ValueError(f"V12 formal fold {fold} seed{seed} key order differs")
        for field in deterministic:
            if not np.allclose(
                prediction[field], source[0][field], atol=1e-7, rtol=0.0
            ):
                raise ValueError(
                    f"V12 formal fold {fold} deterministic {field} differs by seed"
                )
    n_rows = len(reference_keys)
    candidate_weights = np.concatenate(
        [np.asarray(item["candidate_weights"], dtype=np.float64) / 5.0 for item in source],
        axis=1,
    )
    candidate_locations = np.concatenate(
        [np.asarray(item["candidate_locations"], dtype=np.float64) for item in source],
        axis=1,
    )
    candidate_scales = np.concatenate(
        [np.asarray(item["candidate_scales"], dtype=np.float64) for item in source],
        axis=1,
    )
    if any(
        value.shape != (n_rows, 10)
        for value in (candidate_weights, candidate_locations, candidate_scales)
    ):
        raise ValueError("V12 formal candidate must contain ten Gaussian components")
    if not np.allclose(candidate_weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("V12 formal candidate seed weights do not sum to one")
    if not np.isfinite(candidate_locations).all() or not (
        candidate_scales > 0.0
    ).all():
        raise ValueError("V12 formal candidate contains an invalid component")
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FORMAL_PREDICTION_SCHEMA),
        "keys": np.asarray(reference_keys, dtype=object),
        "biological_scoring_key": np.asarray(reference_keys, dtype=object),
        "outer_fold": np.full(n_rows, fold, dtype=np.int64),
        "seed": np.full(n_rows, -1, dtype=np.int64),
        "assembled_seed_count": np.full(n_rows, 5, dtype=np.int64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
        "feature41_point": np.asarray(source[0]["feature41_point"], dtype=np.float64),
        "v11_parent_point": np.asarray(source[0]["v11_parent_point"], dtype=np.float64),
        "candidate_point": np.mean(
            [np.asarray(item["candidate_point"], dtype=np.float64) for item in source],
            axis=0,
        ),
        "candidate_weights": candidate_weights,
        "candidate_locations": candidate_locations,
        "candidate_scales": candidate_scales,
        "candidate_expected_absolute_delta": _expected_absolute(
            candidate_weights, candidate_locations, candidate_scales
        ),
    }
    for prefix in ("feature41", "parent", "historical_v10"):
        for suffix in ("weights", "locations", "scales", "expected_absolute_delta"):
            output[f"{prefix}_{suffix}"] = np.asarray(
                source[0][f"{prefix}_{suffix}"], dtype=np.float64
            )
    path = out_dir / f"v12_formal_predictions_fold{fold}_seeds0_4.npz"
    np.savez_compressed(path, **output)
    return {
        "outer_fold": fold,
        "seeds": [0, 1, 2, 3, 4],
        "prediction_artifact": str(path),
        "n_registered_prediction_rows": n_rows,
        "candidate_components_per_distribution": 10,
        "equal_seed_weight": 0.2,
    }


def assemble(merged: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V12M4_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V12 formal assembler requires complete V12M4 merge")
    rows_by_fold: dict[int, list[dict[str, Any]]] = {fold: [] for fold in range(20)}
    for row in merged.get("folds", []):
        rows_by_fold[int(row["outer_fold"])].append(row)
    if any(len(rows) != 5 for rows in rows_by_fold.values()):
        raise ValueError("V12 formal assembler requires five runs per fold")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = [
        assemble_fold(rows_by_fold[fold], fold=fold, out_dir=out_dir)
        for fold in range(20)
    ]
    return {
        "schema_version": SCHEMA,
        "phase": "V12M4",
        "status": "V12M4_FIVE_SEED_PREDICTION_ONLY_ASSEMBLY_PASS",
        "folds": folds,
        "equal_seed_mixture": True,
        "best_seed_selection_performed": False,
        "score_computed": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_json.exists():
        raise FileExistsError(f"V12 formal assembler refuses to overwrite {args.out_json}")
    result = assemble(json.loads(args.merged_json.read_text()), args.out_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
