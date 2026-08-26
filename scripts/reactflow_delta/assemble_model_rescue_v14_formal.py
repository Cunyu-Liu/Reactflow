#!/usr/bin/env python3
"""Assemble V14 five-seed equal mixtures without scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.merge_model_rescue_v14 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v9 import expected_absolute_delta
from scripts.reactflow_delta.model_rescue_v14 import PREDICTION_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v14_formal_assembly.v1"
FORMAL_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v14_formal_prediction.v1"
POINT_NAMES = ("feature41", "candidate", "null")


def _load_prediction(path: Path, fold: int, seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: np.asarray(handle[name]) for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V14 source schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold} or set(
        map(int, prediction["seed"])
    ) != {seed}:
        raise ValueError(f"V14 formal source fold/seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate V14 formal source keys in {path}")
    return prediction


def _expected_absolute(
    weights: np.ndarray, locations: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        value = expected_absolute_delta(
            torch.tensor(weights, dtype=torch.float64),
            torch.tensor(locations, dtype=torch.float64),
            torch.tensor(scales, dtype=torch.float64),
        )
    return value.numpy()


def assemble_fold(
    rows: list[dict[str, Any]], *, fold: int, out_dir: Path
) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in rows}
    if sorted(by_seed) != list(range(5)):
        raise ValueError(f"V14 formal fold {fold} requires seeds0-4")
    sources = [
        _load_prediction(Path(by_seed[seed]["prediction_artifact"]), fold, seed)
        for seed in range(5)
    ]
    keys = list(map(str, sources[0]["keys"]))
    for seed, prediction in enumerate(sources[1:], start=1):
        if list(map(str, prediction["keys"])) != keys:
            raise ValueError(f"V14 fold {fold} seed{seed} key order differs")
        if not np.allclose(
            prediction["feature41_point"],
            sources[0]["feature41_point"],
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError(f"V14 fold {fold} feature41 differs by seed")
    n_rows = len(keys)
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FORMAL_PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(n_rows, fold, dtype=np.int64),
        "seed": np.full(n_rows, -1, dtype=np.int64),
        "assembled_seed_count": np.full(n_rows, 5, dtype=np.int64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
        "feature41_point": np.asarray(sources[0]["feature41_point"], dtype=np.float64),
        "candidate_point": np.mean(
            [np.asarray(item["candidate_point"], dtype=np.float64) for item in sources],
            axis=0,
        ),
        "null_point": np.mean(
            [np.asarray(item["null_point"], dtype=np.float64) for item in sources],
            axis=0,
        ),
    }
    for name in POINT_NAMES:
        weights = np.concatenate(
            [np.asarray(item[f"{name}_weights"], dtype=np.float64) / 5.0 for item in sources],
            axis=1,
        )
        locations = np.concatenate(
            [np.asarray(item[f"{name}_locations"], dtype=np.float64) for item in sources],
            axis=1,
        )
        scales = np.concatenate(
            [np.asarray(item[f"{name}_scales"], dtype=np.float64) for item in sources],
            axis=1,
        )
        if weights.shape != (n_rows, 10) or locations.shape != (n_rows, 10) or scales.shape != (n_rows, 10):
            raise ValueError("V14 formal distributions must contain ten components")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
            raise ValueError("V14 formal seed weights do not sum to one")
        if not np.isfinite(locations).all() or not (scales > 0.0).all():
            raise ValueError("V14 formal mixture contains an invalid component")
        output[f"{name}_weights"] = weights
        output[f"{name}_locations"] = locations
        output[f"{name}_scales"] = scales
        output[f"{name}_expected_absolute_delta"] = _expected_absolute(
            weights, locations, scales
        )
    path = out_dir / f"v14_formal_predictions_fold{fold}_seeds0_4.npz"
    np.savez_compressed(path, **output)
    return {
        "outer_fold": fold,
        "seeds": [0, 1, 2, 3, 4],
        "prediction_artifact": str(path),
        "n_registered_prediction_rows": n_rows,
        "components_per_distribution": 10,
        "equal_seed_weight": 0.2,
    }


def assemble(merged: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V14M4_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V14 formal assembler requires complete V14M4 merge")
    rows_by_fold: dict[int, list[dict[str, Any]]] = {fold: [] for fold in range(20)}
    for row in merged.get("folds", []):
        rows_by_fold[int(row["outer_fold"])].append(row)
    if any(len(rows) != 5 for rows in rows_by_fold.values()):
        raise ValueError("V14 formal assembler requires five runs per fold")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = [
        assemble_fold(rows_by_fold[fold], fold=fold, out_dir=out_dir)
        for fold in range(20)
    ]
    return {
        "schema_version": SCHEMA,
        "phase": "V14M4",
        "status": "V14M4_FIVE_SEED_PREDICTION_ONLY_ASSEMBLY_PASS",
        "folds": folds,
        "equal_seed_mixture": True,
        "best_seed_selection_performed": False,
        "score_computed": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble(json.loads(args.merged_json.read_text(encoding="utf-8")), args.out_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
