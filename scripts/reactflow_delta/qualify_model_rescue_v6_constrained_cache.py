#!/usr/bin/env python3
"""Mechanically qualify the complete outcome-blind v6 constrained cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reactflow_delta.build_model_rescue_v6_constrained_cache import (
    assert_cache_authority,
    attach_wt_constraints,
    load_outcome_blind_inputs,
)
from scripts.reactflow_delta.model_rescue_v6_schema import (
    CACHE_SCHEMA,
    DEIGAN_INTERCEPT,
    DEIGAN_SLOPE,
    FEATURE_NAMES,
    METADATA_COLUMNS,
    QUALIFICATION_SCHEMA,
)
from scripts.reactflow_delta.model_rescue_v5_schema import CACHE_SCHEMA as V5_CACHE_SCHEMA


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


def _compare_with_unconstrained(
    constrained_cache: Path,
    unconstrained_cache: Path,
) -> dict[str, bool | int]:
    with h5py.File(constrained_cache, "r") as constrained, h5py.File(
        unconstrained_cache, "r"
    ) as unconstrained:
        constrained_ids = _strings(constrained["row_id"])
        unconstrained_ids = _strings(unconstrained["row_id"])
        universe_equal = constrained_ids == unconstrained_ids
        if not universe_equal:
            return {
                "unconstrained_schema": unconstrained.attrs.get("schema_version")
                == V5_CACHE_SCHEMA,
                "unconstrained_row_universe_order_equal": False,
                "all_missing_exact_unconstrained_fallback": False,
                "every_observed_construct_changes_features": False,
                "observed_constructs_compared": 0,
                "all_missing_rows_compared": 0,
            }
        constrained_features = constrained["features"]
        unconstrained_features = unconstrained["features"]
        all_missing = np.asarray(constrained["constraint_all_missing"][:], dtype=bool)
        construct_ids = np.asarray(_strings(constrained["construct_id"]), dtype=object)
        fallback_equal = True
        observed_changed: dict[str, bool] = {}
        for start in range(0, len(constrained_ids), 128):
            stop = min(start + 128, len(constrained_ids))
            current = np.asarray(constrained_features[start:stop])
            base = np.asarray(unconstrained_features[start:stop])
            flags = all_missing[start:stop]
            if flags.any() and not np.array_equal(current[flags], base[flags]):
                fallback_equal = False
            for local_index, construct_id in enumerate(construct_ids[start:stop]):
                if flags[local_index]:
                    continue
                changed = not np.array_equal(current[local_index], base[local_index])
                observed_changed[str(construct_id)] = (
                    observed_changed.get(str(construct_id), False) or changed
                )
        return {
            "unconstrained_schema": unconstrained.attrs.get("schema_version")
            == V5_CACHE_SCHEMA,
            "unconstrained_row_universe_order_equal": True,
            "all_missing_exact_unconstrained_fallback": fallback_equal,
            "every_observed_construct_changes_features": bool(observed_changed)
            and all(observed_changed.values()),
            "observed_constructs_compared": len(observed_changed),
            "all_missing_rows_compared": int(all_missing.sum()),
        }


def qualify_cache(
    cache_path: Path,
    manifest_path: Path,
    unconstrained_cache_path: Path,
    m2_csv: Path,
    *,
    expected_constructs: int,
    expected_mutants: int,
    expected_length: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata, reactivity_columns, wt_reactivity = load_outcome_blind_inputs(m2_csv)
    groups = attach_wt_constraints(metadata, wt_reactivity)[:expected_constructs]
    expected_rows = {
        mutant["row_id"]: {
            "coordinate": (mutant["design_pos"], mutant["full_pos"]),
            "construct_id": group["construct_id"],
            "all_missing": bool(group["constraint_policy"]["all_missing"]),
        }
        for group in groups
        for mutant in group["mutants"]
    }
    expected_all_missing = sorted(
        group["construct_id"]
        for group in groups
        if bool(group["constraint_policy"]["all_missing"])
    )
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == CACHE_SCHEMA,
        "manifest_status": manifest.get("status")
        == "OUTCOME_BLIND_WT_2A3_CONSTRAINED_CACHE_COMPLETE",
        "metadata_columns_exact": tuple(manifest.get("metadata_columns", []))
        == METADATA_COLUMNS,
        "reactivity_columns_exact": tuple(manifest.get("reactivity_columns", []))
        == reactivity_columns,
        "feature_names_exact": tuple(manifest.get("feature_names", [])) == FEATURE_NAMES,
        "mutant_outcome_columns_not_used": manifest.get("mutant_outcome_columns_used")
        is False,
        "mutant_reactivity_rows_not_read": manifest.get("mutant_reactivity_rows_read")
        == 0,
        "wt_reactivity_row_count": manifest.get("wt_reactivity_rows_read")
        == len(wt_reactivity),
        "external_outcome_not_accessed": manifest.get("external_outcome_accessed")
        is False,
        "manifest_construct_count": manifest.get("n_constructs")
        == expected_constructs,
        "manifest_mutant_count": manifest.get("n_registered_mutants")
        == expected_mutants,
        "manifest_sequence_length": manifest.get("sequence_length") == expected_length,
        "viennarna_version": manifest.get("viennarna_version") == "2.7.2",
        "deigan_slope": manifest.get("deigan_slope") == DEIGAN_SLOPE,
        "deigan_intercept": manifest.get("deigan_intercept") == DEIGAN_INTERCEPT,
        "same_constraint_for_wt_and_mutant": manifest.get(
            "wt_and_mutant_share_identical_constraint_vector"
        )
        is True,
        "mutation_site_constraint_not_removed": manifest.get(
            "remove_mutation_site_constraint"
        )
        is False,
        "all_missing_construct_universe": sorted(
            manifest.get("all_missing_construct_ids", [])
        )
        == expected_all_missing,
    }
    with h5py.File(cache_path, "r") as handle:
        row_ids = _strings(handle["row_id"])
        construct_ids = _strings(handle["construct_id"])
        design_pos = np.asarray(handle["design_pos"][:], dtype=np.int64)
        full_pos = np.asarray(handle["full_pos"][:], dtype=np.int64)
        all_missing = np.asarray(handle["constraint_all_missing"][:], dtype=bool)
        features = handle["features"]
        checks.update(
            {
                "cache_schema": handle.attrs.get("schema_version") == CACHE_SCHEMA,
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
        all_missing_identity_ok = True
        for row_id, construct_id, design, full, missing in zip(
            row_ids, construct_ids, design_pos, full_pos, all_missing
        ):
            expected = expected_rows.get(row_id)
            if expected is None or expected["coordinate"] != (int(design), int(full)):
                coordinate_ok = False
                break
            if expected["construct_id"] != construct_id or expected["all_missing"] != bool(
                missing
            ):
                all_missing_identity_ok = False
                break
        checks["coordinate_identity"] = coordinate_ok
        checks["constraint_identity"] = all_missing_identity_ok
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

    comparison = _compare_with_unconstrained(cache_path, unconstrained_cache_path)
    for name in (
        "unconstrained_schema",
        "unconstrained_row_universe_order_equal",
        "all_missing_exact_unconstrained_fallback",
        "every_observed_construct_changes_features",
    ):
        checks[name] = bool(comparison[name])
    status = (
        "V6M1_OUTCOME_BLIND_CONSTRAINED_CACHE_PASS"
        if all(checks.values())
        else "V6M1_OUTCOME_BLIND_CONSTRAINED_CACHE_FAIL"
    )
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": status,
        "checks": checks,
        "comparison_counts": {
            "observed_constructs": comparison["observed_constructs_compared"],
            "all_missing_rows": comparison["all_missing_rows_compared"],
        },
        "cache": str(cache_path),
        "manifest": str(manifest_path),
        "unconstrained_cache": str(unconstrained_cache_path),
        "mutant_outcome_columns_used": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
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
        args.unconstrained_cache,
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
