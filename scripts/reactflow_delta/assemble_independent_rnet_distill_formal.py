#!/usr/bin/env python3
"""Assemble the complete RND6P five-seed prediction-only mixture."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.merge_independent_rnet_distill import (
    EXPECTED_MERGED_FIELDS,
    MERGE_FILENAME,
    MERGE_INTEGRITY,
    SCHEMA as MERGED_SCHEMA,
    STATUS as MERGED_STATUS,
)
from scripts.reactflow_delta.model_rescue_v9 import expected_absolute_delta
from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EXPECTED_FOLDS,
    EXPECTED_PREDICTION_FIELDS,
    EXPECTED_SEEDS,
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)


SCHEMA = "reactflow_delta.independent_rnet_distill_formal_assembly.v1"
FORMAL_PREDICTION_SCHEMA = (
    "reactflow_delta.independent_rnet_distill_formal_prediction.v1"
)
ASSEMBLY_STATUS = "RND6P_EQUAL_SEED_PREDICTION_ONLY_ASSEMBLY_PASS"
FORMAL_DIR = Path(
    "/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/"
    "rnd6_formal_seeds0_4"
)
MERGED_PATH = FORMAL_DIR / MERGE_FILENAME
ASSEMBLY_DIR = FORMAL_DIR / "assembled"
ASSEMBLY_PATH = ASSEMBLY_DIR / (
    "rnet_distill_five_seed_prediction_only_assembly.json"
)

EXPECTED_FORMAL_PREDICTION_FIELDS = frozenset(
    {*EXPECTED_PREDICTION_FIELDS, "assembled_seed_count"}
)
EXPECTED_ASSEMBLY_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "status",
        "folds",
        "equal_seed_mixture",
        "equal_seed_weight",
        "best_seed_selection_performed",
        "score_computed",
        "target_accessed",
        "external_outcome_accessed",
    }
)
EXPECTED_FOLD_MANIFEST_FIELDS = frozenset(
    {
        "outer_fold",
        "seeds",
        "prediction_artifact",
        "n_registered_prediction_rows",
        "candidate_components_per_distribution",
        "null_components_per_distribution",
        "feature41_components_per_distribution",
        "historical_v10_components_per_distribution",
    }
)

PHASE = "RND6P"
SEEDS = tuple(range(5))
FOLDS = tuple(range(20))
POINT_FIELDS = ("feature41_point", "v8_point", "candidate_point", "null_point")
DISTRIBUTION_NAMES = ("feature41", "candidate", "null", "historical_v10")
DISTRIBUTION_SUFFIXES = (
    "weights",
    "locations",
    "scales",
    "expected_absolute_delta",
)
FIXED_COMPARATOR_FIELDS = (
    "feature41_point",
    "v8_point",
    *(f"feature41_{suffix}" for suffix in DISTRIBUTION_SUFFIXES),
    *(f"historical_v10_{suffix}" for suffix in DISTRIBUTION_SUFFIXES),
)


def _expected_source_basename(fold: int, seed: int) -> str:
    return f"rnet_distill_predictions_fold{fold}_seed{seed}.npz"


def _expected_formal_basename(fold: int) -> str:
    return f"rnet_distill_formal_predictions_fold{fold}_seeds0_4.npz"


def _load_source_prediction(
    path: Path, *, fold: int, seed: int, expected_rows: int
) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if path.name != _expected_source_basename(fold, seed):
        raise ValueError(f"RND6P fold {fold} seed {seed} prediction basename differs")
    try:
        with np.load(path, allow_pickle=True) as handle:
            names = frozenset(handle.files)
            if names != EXPECTED_PREDICTION_FIELDS or names & FORBIDDEN_PREDICTION_FIELDS:
                raise ValueError(
                    f"RND6P fold {fold} seed {seed} prediction field universe differs"
                )
            prediction = {name: np.asarray(handle[name]) for name in names}
    except (OSError, KeyError) as error:
        raise ValueError(
            f"RND6P fold {fold} seed {seed} prediction cannot be read"
        ) from error

    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError(f"RND6P fold {fold} seed {seed} prediction schema differs")
    keys = tuple(map(str, prediction["keys"]))
    n_rows = len(keys)
    if (
        n_rows != expected_rows
        or n_rows < 1
        or len(keys) != len(set(keys))
        or prediction["biological_scoring_key"].shape != (n_rows,)
        or keys != tuple(map(str, prediction["biological_scoring_key"]))
        or prediction["outer_fold"].shape != (n_rows,)
        or set(map(int, prediction["outer_fold"])) != {fold}
        or prediction["seed"].shape != (n_rows,)
        or set(map(int, prediction["seed"])) != {seed}
        or prediction["registered_status"].shape != (n_rows,)
        or set(map(str, prediction["registered_status"])) != {"covered"}
    ):
        raise ValueError(f"RND6P fold {fold} seed {seed} identity differs")

    for field in POINT_FIELDS:
        values = prediction[field]
        if values.shape != (n_rows,) or not np.isfinite(values).all():
            raise ValueError(
                f"RND6P fold {fold} seed {seed} {field} is invalid"
            )
    for name in DISTRIBUTION_NAMES:
        weights = prediction[f"{name}_weights"]
        locations = prediction[f"{name}_locations"]
        scales = prediction[f"{name}_scales"]
        expected_absolute = prediction[f"{name}_expected_absolute_delta"]
        if (
            weights.shape != (n_rows, 2)
            or locations.shape != (n_rows, 2)
            or scales.shape != (n_rows, 2)
            or expected_absolute.shape != (n_rows,)
            or not np.isfinite(weights).all()
            or not np.isfinite(locations).all()
            or not np.isfinite(scales).all()
            or not np.isfinite(expected_absolute).all()
            or np.any(weights < 0.0)
            or np.any(scales <= 0.0)
            or np.any(expected_absolute < 0.0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0)
        ):
            raise ValueError(
                f"RND6P fold {fold} seed {seed} {name} distribution is invalid"
            )
    return prediction


def _validated_sources(
    merged: dict[str, Any],
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    if (
        frozenset(merged) != EXPECTED_MERGED_FIELDS
        or merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("phase") != PHASE
        or merged.get("status") != MERGED_STATUS[PHASE]
        or merged.get("merge_integrity") != MERGE_INTEGRITY
    ):
        raise ValueError("RND6P assembler requires the exact complete target-free merge")
    rows = merged.get("folds")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("RND6P assembler requires exactly 100 fold-seed rows")

    row_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("RND6P merged fold-seed row is not a mapping")
        try:
            pair = (int(row["outer_fold"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("RND6P merged fold-seed identity is malformed") from error
        if pair in row_by_pair:
            raise ValueError(f"RND6P duplicate fold-seed row: {pair}")
        row_by_pair[pair] = row
    expected_pairs = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if set(row_by_pair) != expected_pairs:
        raise ValueError("RND6P merged fold-seed universe is incomplete or unexpected")

    sources: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    keys_by_seed = {seed: set() for seed in SEEDS}
    reference_by_fold: dict[int, tuple[str, tuple[str, ...], int]] = {}
    for seed in SEEDS:
        for fold in FOLDS:
            row = row_by_pair[(fold, seed)]
            try:
                expected_rows = int(row["n_registered_prediction_rows"])
                held_puzzle = str(row["held_puzzle"])
                prediction_path = Path(str(row["prediction_artifact"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"RND6P fold {fold} seed {seed} source row is malformed"
                ) from error
            prediction = _load_source_prediction(
                prediction_path,
                fold=fold,
                seed=seed,
                expected_rows=expected_rows,
            )
            keys = tuple(map(str, prediction["keys"]))
            duplicate_keys = keys_by_seed[seed] & set(keys)
            if duplicate_keys:
                raise ValueError(
                    f"RND6P seed {seed} biological keys repeat across folds"
                )
            keys_by_seed[seed].update(keys)
            reference = (held_puzzle, keys, expected_rows)
            if fold not in reference_by_fold:
                reference_by_fold[fold] = reference
            elif reference_by_fold[fold] != reference:
                raise ValueError(
                    f"RND6P fold {fold} held puzzle, key order, or row count differs across seeds"
                )
            sources[(fold, seed)] = prediction
    return sources


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


def _assemble_fold_payload(
    sources: dict[tuple[int, int], dict[str, np.ndarray]], *, fold: int
) -> dict[str, np.ndarray]:
    predictions = [sources[(fold, seed)] for seed in SEEDS]
    reference = predictions[0]
    for seed, prediction in enumerate(predictions[1:], start=1):
        for field in FIXED_COMPARATOR_FIELDS:
            if not np.array_equal(prediction[field], reference[field]):
                raise ValueError(
                    f"RND6P fold {fold} fixed comparator {field} differs at seed {seed}"
                )

    keys = np.asarray(reference["keys"], dtype=object)
    n_rows = len(keys)
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FORMAL_PREDICTION_SCHEMA),
        "keys": keys.copy(),
        "biological_scoring_key": np.asarray(
            reference["biological_scoring_key"], dtype=object
        ).copy(),
        "outer_fold": np.full(n_rows, fold, dtype=np.int64),
        "seed": np.full(n_rows, -1, dtype=np.int64),
        "assembled_seed_count": np.full(n_rows, 5, dtype=np.int64),
        "registered_status": np.asarray(
            reference["registered_status"], dtype=object
        ).copy(),
        "feature41_point": np.asarray(
            reference["feature41_point"], dtype=np.float64
        ).copy(),
        "v8_point": np.asarray(reference["v8_point"], dtype=np.float64).copy(),
        "candidate_point": np.mean(
            [np.asarray(item["candidate_point"], dtype=np.float64) for item in predictions],
            axis=0,
        ),
        "null_point": np.mean(
            [np.asarray(item["null_point"], dtype=np.float64) for item in predictions],
            axis=0,
        ),
    }

    for name in ("feature41", "historical_v10"):
        for suffix in DISTRIBUTION_SUFFIXES:
            output[f"{name}_{suffix}"] = np.asarray(
                reference[f"{name}_{suffix}"], dtype=np.float64
            ).copy()

    for name in ("candidate", "null"):
        weights = np.concatenate(
            [
                np.asarray(item[f"{name}_weights"], dtype=np.float64) / 5.0
                for item in predictions
            ],
            axis=1,
        )
        locations = np.concatenate(
            [
                np.asarray(item[f"{name}_locations"], dtype=np.float64)
                for item in predictions
            ],
            axis=1,
        )
        scales = np.concatenate(
            [
                np.asarray(item[f"{name}_scales"], dtype=np.float64)
                for item in predictions
            ],
            axis=1,
        )
        if (
            weights.shape != (n_rows, 10)
            or locations.shape != (n_rows, 10)
            or scales.shape != (n_rows, 10)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0)
            or not np.isfinite(locations).all()
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
        ):
            raise ValueError(f"RND6P fold {fold} assembled {name} mixture is invalid")
        output[f"{name}_weights"] = weights
        output[f"{name}_locations"] = locations
        output[f"{name}_scales"] = scales
        output[f"{name}_expected_absolute_delta"] = _expected_absolute(
            weights, locations, scales
        )

    if frozenset(output) != EXPECTED_FORMAL_PREDICTION_FIELDS:
        missing = sorted(EXPECTED_FORMAL_PREDICTION_FIELDS - frozenset(output))
        unexpected = sorted(frozenset(output) - EXPECTED_FORMAL_PREDICTION_FIELDS)
        raise AssertionError(
            f"RND6P formal prediction fields differ: missing={missing} unexpected={unexpected}"
        )
    return output


def assemble(
    merged: dict[str, Any],
    out_dir: Path,
    *,
    published_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate all 100 sources, then write twenty target-free fold mixtures."""

    sources = _validated_sources(merged)
    payloads = {
        fold: _assemble_fold_payload(sources, fold=fold) for fold in FOLDS
    }
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"RND6P assembly output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    published = (published_dir or out_dir).expanduser().resolve()

    fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        basename = _expected_formal_basename(fold)
        path = out_dir / basename
        np.savez_compressed(path, **payloads[fold])
        row = {
            "outer_fold": fold,
            "seeds": list(SEEDS),
            "prediction_artifact": str(published / basename),
            "n_registered_prediction_rows": int(len(payloads[fold]["keys"])),
            "candidate_components_per_distribution": 10,
            "null_components_per_distribution": 10,
            "feature41_components_per_distribution": 2,
            "historical_v10_components_per_distribution": 2,
        }
        if frozenset(row) != EXPECTED_FOLD_MANIFEST_FIELDS:
            raise AssertionError("RND6P fold assembly manifest fields changed")
        fold_rows.append(row)

    result = {
        "schema_version": SCHEMA,
        "phase": PHASE,
        "status": ASSEMBLY_STATUS,
        "folds": fold_rows,
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "score_computed": False,
        "target_accessed": False,
        "external_outcome_accessed": False,
    }
    if frozenset(result) != EXPECTED_ASSEMBLY_FIELDS:
        raise AssertionError("RND6P assembly manifest fields changed")
    return result


def validate_cli_binding(
    merged_json: Path, out_dir: Path, out_json: Path
) -> dict[str, str]:
    observed = {
        "merged_json": merged_json.expanduser().resolve(),
        "out_dir": out_dir.expanduser().resolve(),
        "out_json": out_json.expanduser().resolve(),
    }
    expected = {
        "merged_json": MERGED_PATH.resolve(),
        "out_dir": ASSEMBLY_DIR.resolve(),
        "out_json": ASSEMBLY_PATH.resolve(),
    }
    for name, expected_path in expected.items():
        if observed[name] != expected_path:
            raise RuntimeError(
                f"RND6P assembler {name} differs: "
                f"observed={observed[name]} expected={expected_path}"
            )
    return {name: str(path) for name, path in expected.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli_binding(args.merged_json, args.out_dir, args.out_json)
    if args.out_dir.exists() or args.out_json.exists():
        raise FileExistsError("RND6P refuses to overwrite its canonical assembly")
    merged = json.loads(args.merged_json.read_text(encoding="utf-8"))
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.out_dir.name}.", dir=args.out_dir.parent
    ) as staging_name:
        staging_dir = Path(staging_name)
        result = assemble(
            merged,
            staging_dir,
            published_dir=args.out_dir,
        )
        staging_manifest = staging_dir / args.out_json.name
        staging_manifest.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(args.out_dir)
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
