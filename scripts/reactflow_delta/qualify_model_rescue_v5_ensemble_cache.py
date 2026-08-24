#!/usr/bin/env python3
"""Mechanically qualify the complete outcome-blind v5 ensemble cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from scripts.reactflow_delta.build_model_rescue_v5_ensemble_cache import (
    FEATURE_NAMES,
    SCHEMA,
    SOURCE_COLUMNS,
    assert_cache_authority,
    build_construct_groups,
)


FORBIDDEN_DATASETS = {
    "reactivity",
    "reactivity_error",
    "target",
    "target_error",
    "target_mask",
    "qualified_target_mask",
    "score",
}


def _strings(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ]


def qualify_cache(
    cache_path: Path,
    manifest_path: Path,
    m2_csv: Path,
    *,
    expected_constructs: int,
    expected_mutants: int,
    expected_length: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(m2_csv, usecols=list(SOURCE_COLUMNS))
    frame = frame.loc[:, list(SOURCE_COLUMNS)]
    groups = build_construct_groups(frame)
    expected_rows = {
        mutant["row_id"]: (mutant["design_pos"], mutant["full_pos"])
        for group in groups[:expected_constructs]
        for mutant in group["mutants"]
    }
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == SCHEMA,
        "manifest_status": manifest.get("status")
        == "OUTCOME_BLIND_ENSEMBLE_CACHE_COMPLETE",
        "source_columns_exact": tuple(manifest.get("source_columns", []))
        == SOURCE_COLUMNS,
        "feature_names_exact": tuple(manifest.get("feature_names", []))
        == FEATURE_NAMES,
        "outcome_columns_not_read": manifest.get("outcome_columns_read") is False,
        "external_outcome_not_accessed": manifest.get("external_outcome_accessed")
        is False,
        "manifest_construct_count": manifest.get("n_constructs")
        == expected_constructs,
        "manifest_mutant_count": manifest.get("n_registered_mutants")
        == expected_mutants,
        "manifest_sequence_length": manifest.get("sequence_length")
        == expected_length,
    }
    with h5py.File(cache_path, "r") as handle:
        row_ids = _strings(handle["row_id"])
        design_pos = np.asarray(handle["design_pos"][:], dtype=np.int64)
        full_pos = np.asarray(handle["full_pos"][:], dtype=np.int64)
        features = handle["features"]
        checks.update(
            {
                "cache_schema": handle.attrs.get("schema_version") == SCHEMA,
                "forbidden_datasets_absent": FORBIDDEN_DATASETS.isdisjoint(handle.keys()),
                "row_count": len(row_ids) == expected_mutants,
                "row_ids_unique": len(set(row_ids)) == len(row_ids),
                "registered_row_universe_exact": set(row_ids) == set(expected_rows),
                "feature_shape": features.shape
                == (expected_mutants, expected_length, len(FEATURE_NAMES)),
                "feature_dtype_float32": features.dtype == np.dtype(np.float32),
            }
        )
        coordinate_ok = True
        for row_id, design, full in zip(row_ids, design_pos, full_pos):
            if row_id not in expected_rows or expected_rows[row_id] != (int(design), int(full)):
                coordinate_ok = False
                break
        checks["coordinate_identity"] = coordinate_ok
        finite = True
        nonnegative_magnitudes = True
        for start in range(0, features.shape[0], 128):
            block = np.asarray(features[start : start + 128])
            if not np.isfinite(block).all():
                finite = False
                break
            if np.any(block[..., 5] < 0) or np.any(block[..., 6] < 0):
                nonnegative_magnitudes = False
                break
        checks["features_finite"] = finite
        checks["change_magnitudes_nonnegative"] = nonnegative_magnitudes
    status = (
        "V5M1_OUTCOME_BLIND_ENSEMBLE_CACHE_PASS"
        if all(checks.values())
        else "V5M1_OUTCOME_BLIND_ENSEMBLE_CACHE_FAIL"
    )
    return {
        "schema_version": "reactflow_delta.model_rescue_v5_ensemble_cache_qualification.v1",
        "status": status,
        "checks": checks,
        "cache": str(cache_path),
        "manifest": str(manifest_path),
        "outcome_columns_read": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--expected-constructs", type=int, default=160)
    parser.add_argument("--expected-mutants", type=int, default=13976)
    parser.add_argument("--expected-length", type=int, default=177)
    args = parser.parse_args(argv)
    assert_cache_authority(args.repo_root.resolve())
    result = qualify_cache(
        args.cache,
        args.manifest,
        args.m2_csv,
        expected_constructs=args.expected_constructs,
        expected_mutants=args.expected_mutants,
        expected_length=args.expected_length,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}, indent=2))
    return 0 if result["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
