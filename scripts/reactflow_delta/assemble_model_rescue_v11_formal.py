#!/usr/bin/env python3
"""Assemble the frozen V11 five-seed equal mixtures without scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.merge_model_rescue_v11 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v9 import expected_absolute_delta
from scripts.reactflow_delta.model_rescue_v11 import PREDICTION_SCHEMA


SCHEMA = "reactflow_delta.model_rescue_v11_formal_assembly.v1"
FORMAL_PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v11_formal_prediction.v1"
POINT_NAMES = ("feature41", "anchored", "unanchored")


def _load_prediction(path: Path, fold: int, seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"invalid V11 prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold}:
        raise ValueError(f"V11 formal source fold mismatch in {path}")
    if set(map(int, prediction["seed"])) != {seed}:
        raise ValueError(f"V11 formal source seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate V11 source keys in {path}")
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
        raise ValueError(f"V11 formal fold {fold} requires seeds0-4")
    source = [
        _load_prediction(Path(by_seed[seed]["prediction_artifact"]), fold, seed)
        for seed in range(5)
    ]
    reference_keys = list(map(str, source[0]["keys"]))
    for seed, prediction in enumerate(source[1:], start=1):
        if list(map(str, prediction["keys"])) != reference_keys:
            raise ValueError(f"V11 formal fold {fold} seed{seed} key order differs")
        for field in ("feature41_point", "v8_point"):
            if not np.allclose(
                prediction[field], source[0][field], atol=1e-7, rtol=0.0
            ):
                raise ValueError(
                    f"V11 formal fold {fold} deterministic {field} differs by seed"
                )
    n_rows = len(reference_keys)
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FORMAL_PREDICTION_SCHEMA),
        "keys": np.asarray(reference_keys, dtype=object),
        "biological_scoring_key": np.asarray(reference_keys, dtype=object),
        "outer_fold": np.full(n_rows, fold, dtype=np.int64),
        "seed": np.full(n_rows, -1, dtype=np.int64),
        "assembled_seed_count": np.full(n_rows, 5, dtype=np.int64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
        "feature41_point": np.asarray(source[0]["feature41_point"], dtype=np.float64),
        "v8_point": np.asarray(source[0]["v8_point"], dtype=np.float64),
        "anchored_point": np.mean(
            [np.asarray(item["anchored_point"], dtype=np.float64) for item in source],
            axis=0,
        ),
        "unanchored_point": np.mean(
            [np.asarray(item["unanchored_point"], dtype=np.float64) for item in source],
            axis=0,
        ),
    }
    for name in POINT_NAMES:
        weights = np.concatenate(
            [np.asarray(item[f"{name}_weights"], dtype=np.float64) / 5.0 for item in source],
            axis=1,
        )
        locations = np.concatenate(
            [np.asarray(item[f"{name}_locations"], dtype=np.float64) for item in source],
            axis=1,
        )
        scales = np.concatenate(
            [np.asarray(item[f"{name}_scales"], dtype=np.float64) for item in source],
            axis=1,
        )
        if (
            weights.shape != (n_rows, 10)
            or locations.shape != (n_rows, 10)
            or scales.shape != (n_rows, 10)
        ):
            raise ValueError("V11 formal mixture must contain ten components")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
            raise ValueError("V11 formal mixture seed weights do not sum to one")
        if not np.isfinite(locations).all() or not (scales > 0.0).all():
            raise ValueError("V11 formal mixture contains an invalid component")
        output[f"{name}_weights"] = weights
        output[f"{name}_locations"] = locations
        output[f"{name}_scales"] = scales
        output[f"{name}_expected_absolute_delta"] = _expected_absolute(
            weights, locations, scales
        )

    # Terminal V10 remains descriptive context and is carried once from seed0;
    # it is never promoted to a five-seed formal comparator.
    for suffix in ("weights", "locations", "scales", "expected_absolute_delta"):
        output[f"historical_v10_{suffix}"] = np.asarray(
            source[0][f"historical_v10_{suffix}"], dtype=np.float64
        )
    path = out_dir / f"v11_formal_predictions_fold{fold}_seeds0_4.npz"
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
        "V11M4_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V11 formal assembler requires complete V11M4 merge")
    rows_by_fold: dict[int, list[dict[str, Any]]] = {fold: [] for fold in range(20)}
    for row in merged.get("folds", []):
        rows_by_fold[int(row["outer_fold"])].append(row)
    if any(len(rows) != 5 for rows in rows_by_fold.values()):
        raise ValueError("V11 formal assembler requires five runs per fold")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = [
        assemble_fold(rows_by_fold[fold], fold=fold, out_dir=out_dir)
        for fold in range(20)
    ]
    return {
        "schema_version": SCHEMA,
        "phase": "V11M4",
        "status": "V11M4_FIVE_SEED_PREDICTION_ONLY_ASSEMBLY_PASS",
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
    result = assemble(
        json.loads(args.merged_json.read_text(encoding="utf-8")), args.out_dir
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
