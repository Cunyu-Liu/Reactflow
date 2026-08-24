#!/usr/bin/env python3
"""Mechanically qualify the complete outcome-blind v7 dependency cache."""

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

from scripts.reactflow_delta.build_model_rescue_v7_dependency_cache import (
    assert_cache_authority,
    load_outcome_blind_groups,
)
from scripts.reactflow_delta.model_rescue_v7_schema import (
    CACHE_SCHEMA,
    FEATURE_NAMES,
    FORBIDDEN_CACHE_DATASETS,
    QUALIFICATION_SCHEMA,
    RINALMO_CODE_COMMIT,
    RINALMO_MODEL_NAME,
    RINALMO_PARAMETER_COUNT,
)


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
    groups = load_outcome_blind_groups(m2_csv, max_constructs=expected_constructs)
    expected_rows = {
        str(mutant["row_id"]): {
            "construct_id": str(group["construct_id"]),
            "puzzle": str(group["puzzle"]),
            "method": str(group["method"]),
            "wt_sequence": str(group["wt_sequence"]),
            "mutant_sequence": str(mutant["sequence"]),
            "design_pos": int(mutant["design_pos"]),
            "full_pos": int(mutant["full_pos"]),
            "ref": str(mutant["ref"]),
            "alt": str(mutant["alt"]),
        }
        for group in groups
        for mutant in group["mutants"]
    }
    expected_inference_sequences = {
        sequence
        for row in expected_rows.values()
        for sequence in (row["wt_sequence"], row["mutant_sequence"])
    }
    expected_dependency_edges = {
        (row["wt_sequence"], row["mutant_sequence"], row["full_pos"])
        for row in expected_rows.values()
    }
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == CACHE_SCHEMA,
        "manifest_status": manifest.get("status")
        == "OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_COMPLETE",
        "manifest_model": manifest.get("model_name") == RINALMO_MODEL_NAME,
        "manifest_parameter_count": manifest.get("model_parameter_count")
        == RINALMO_PARAMETER_COUNT,
        "manifest_code_commit": manifest.get("model_code_commit")
        == RINALMO_CODE_COMMIT,
        "manifest_feature_names": tuple(manifest.get("feature_names", []))
        == FEATURE_NAMES,
        "manifest_parameter_dtype": manifest.get("parameter_dtype")
        == "FLOAT32_OFFICIAL_CHECKPOINT",
        "manifest_forward_autocast_dtype": manifest.get("forward_autocast_dtype")
        == "FLOAT16_OFFICIAL_CUDA_AUTOCAST_DEFAULT",
        "manifest_output_dtype": manifest.get("output_logit_and_log_odds_dtype")
        == "FLOAT32",
        "manifest_construct_count": manifest.get("n_constructs")
        == expected_constructs,
        "manifest_mutant_count": manifest.get("n_registered_mutants")
        == expected_mutants,
        "manifest_unique_wt_sequence_count": manifest.get("n_unique_wt_sequences")
        == len({row["wt_sequence"] for row in expected_rows.values()}),
        "manifest_unique_mutant_sequence_count": manifest.get(
            "n_unique_mutant_sequences"
        )
        == len({row["mutant_sequence"] for row in expected_rows.values()}),
        "manifest_unique_inference_sequence_count": manifest.get(
            "n_unique_inference_sequences"
        )
        == len(expected_inference_sequences),
        "manifest_unique_dependency_edge_count": manifest.get(
            "n_unique_dependency_edges"
        )
        == len(expected_dependency_edges),
        "manifest_dependency_edge_reuse_count": manifest.get(
            "n_dependency_edge_reuse_rows"
        )
        == expected_mutants - len(expected_dependency_edges),
        "manifest_sequence_length": manifest.get("sequence_length")
        == expected_length,
        "manifest_feature_width": manifest.get("feature_width")
        == len(FEATURE_NAMES),
        "full_unmasked_exact_intervention": manifest.get(
            "full_unmasked_wt_and_exact_mutant"
        )
        is True,
        "registered_mutations_only": manifest.get("registered_mutations_only")
        is True,
        "mutant_outcome_rows_zero": manifest.get("mutant_reactivity_rows_read") == 0,
        "target_error_or_mask_absent": manifest.get("target_error_or_mask_read")
        is False,
        "external_outcome_locked": manifest.get("external_outcome_accessed") is False,
    }

    with h5py.File(cache_path, "r") as handle:
        row_ids = _strings(handle["row_id"])
        construct_ids = _strings(handle["construct_id"])
        puzzles = _strings(handle["puzzle"])
        methods = _strings(handle["method"])
        wt_sequences = _strings(handle["wt_sequence"])
        mutant_sequences = _strings(handle["mutant_sequence"])
        refs = _strings(handle["ref"])
        alts = _strings(handle["alt"])
        design_pos = np.asarray(handle["design_pos"][:], dtype=np.int64)
        full_pos = np.asarray(handle["full_pos"][:], dtype=np.int64)
        features = handle["features"]
        checks.update(
            {
                "cache_schema": handle.attrs.get("schema_version") == CACHE_SCHEMA,
                "cache_feature_names": tuple(
                    json.loads(handle.attrs.get("feature_names", "[]"))
                )
                == FEATURE_NAMES,
                "forbidden_datasets_absent": FORBIDDEN_CACHE_DATASETS.isdisjoint(
                    handle.keys()
                ),
                "row_count": len(row_ids) == expected_mutants,
                "row_ids_unique": len(set(row_ids)) == len(row_ids),
                "registered_row_universe_exact": set(row_ids) == set(expected_rows),
                "feature_shape": features.shape
                == (expected_mutants, expected_length, len(FEATURE_NAMES)),
                "feature_dtype_float32": features.dtype == np.dtype(np.float32),
            }
        )

        coordinate_identity = True
        sequence_identity = True
        finite = True
        self_zero = True
        max_absolute_identity = True
        any_off_diagonal_signal = False
        exact_sequence_reuse_identical = True
        first_by_dependency: dict[tuple[str, str, int], np.ndarray] = {}
        for index, row_id in enumerate(row_ids):
            expected = expected_rows.get(row_id)
            if expected is None:
                coordinate_identity = False
                sequence_identity = False
                continue
            observed_identity = {
                "construct_id": construct_ids[index],
                "puzzle": puzzles[index],
                "method": methods[index],
                "design_pos": int(design_pos[index]),
                "full_pos": int(full_pos[index]),
                "ref": refs[index],
                "alt": alts[index],
            }
            if any(observed_identity[name] != expected[name] for name in observed_identity):
                coordinate_identity = False
            if (
                wt_sequences[index] != expected["wt_sequence"]
                or mutant_sequences[index] != expected["mutant_sequence"]
            ):
                sequence_identity = False
            block = np.asarray(features[index], dtype=np.float32)
            if not np.isfinite(block).all():
                finite = False
            source = int(full_pos[index])
            if not np.array_equal(block[source], np.zeros(len(FEATURE_NAMES), np.float32)):
                self_zero = False
            off_diagonal = np.delete(block, source, axis=0)
            any_off_diagonal_signal = any_off_diagonal_signal or bool(
                np.any(off_diagonal != 0.0)
            )
            if not np.allclose(
                block[:, 5], np.max(np.abs(block[:, :4]), axis=1), atol=1e-6, rtol=0.0
            ):
                max_absolute_identity = False
            dependency_key = (
                wt_sequences[index],
                mutant_sequences[index],
                source,
            )
            reference = first_by_dependency.setdefault(dependency_key, block)
            if not np.array_equal(reference, block):
                exact_sequence_reuse_identical = False

        checks.update(
            {
                "coordinate_identity": coordinate_identity,
                "sequence_identity": sequence_identity,
                "features_finite": finite,
                "self_dependency_exact_zero": self_zero,
                "max_absolute_channel_identity": max_absolute_identity,
                "off_diagonal_signal_nonzero": any_off_diagonal_signal,
                "duplicated_exact_sequence_dependency_identical": (
                    exact_sequence_reuse_identical
                ),
                "unique_dependency_count_matches_manifest": len(first_by_dependency)
                == manifest.get("n_unique_dependency_edges"),
            }
        )

    status = (
        "V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_PASS"
        if all(checks.values())
        else "V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_FAIL"
    )
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": status,
        "checks": checks,
        "cache": str(cache_path),
        "manifest": str(manifest_path),
        "mutant_outcome_columns_used": False,
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
