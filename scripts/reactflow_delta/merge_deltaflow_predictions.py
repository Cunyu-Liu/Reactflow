#!/usr/bin/env python3
"""Canonical unscored merge for DeltaFlow fold predictions (Task 6.6/8.3).

Merges the per-fold prediction npz artifacts of one experiment directory
into a single canonical unscored assembly file.  The merge is exactly
once per (experiment, seed): a published merge marker makes reruns refuse
to overwrite; fold files are verified for key universes, row counts, and
frozen constants before any output is written; the output contains no
target/observed fields.

The merged layout keeps per-fold fields stacked in fold order with
aligned keys; flow marginal fields stack over the flow key rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MERGE_SCHEMA = "reactflow_delta.deltaflow_merged_prediction.v1"
REQUIRED_FILES = (
    "deltaflow_fold_result_fold{fold}_seed{seed}.json",
    "deltaflow_predictions_fold{fold}_seed{seed}.npz",
)
STACK_FIELDS = (
    "feature41_point",
    "v8_point",
    "candidate_point",
    "null_point",
    "feature41_weights",
    "feature41_locations",
    "feature41_scales",
    "feature41_expected_absolute_delta",
    "candidate_weights",
    "candidate_locations",
    "candidate_scales",
    "candidate_expected_absolute_delta",
    "null_weights",
    "null_locations",
    "null_scales",
    "null_expected_absolute_delta",
    "historical_v10_weights",
    "historical_v10_locations",
    "historical_v10_scales",
    "historical_v10_expected_absolute_delta",
)
FLOW_FIELDS = (
    "flow_candidate_weights",
    "flow_candidate_locations",
    "flow_candidate_scales",
    "flow_candidate_expected_absolute_delta",
    "flow_null_weights",
    "flow_null_locations",
    "flow_null_scales",
    "flow_null_expected_absolute_delta",
)


def _load_fold(prediction_path: Path) -> dict:
    with np.load(prediction_path, allow_pickle=True) as handle:
        return {name: handle[name] for name in handle.files}


def merge_experiment(out_dir: Path, *, seed: int, folds: list[int]) -> dict:
    marker = out_dir / f"deltaflow_merged_predictions_seed{seed}.npz"
    audit = out_dir / f"deltaflow_merge_audit_seed{seed}.json"
    if marker.exists() or audit.exists():
        raise FileExistsError(f"merge already published for seed {seed}; refusing overwrite")
    keys: list[str] = []
    flow_keys: list[str] = []
    stacked: dict[str, list[np.ndarray]] = {name: [] for name in STACK_FIELDS}
    flow_stacked: dict[str, list[np.ndarray]] = {name: [] for name in FLOW_FIELDS}
    fold_index: list[int] = []
    fold_field_rows: list[int] = []
    flow_rows: list[int] = []
    for fold in folds:
        result_path = out_dir / REQUIRED_FILES[0].format(fold=fold, seed=seed)
        prediction_path = out_dir / REQUIRED_FILES[1].format(fold=fold, seed=seed)
        if not result_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"fold {fold} seed {seed} is incomplete")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result["seed"]) != seed or int(result["outer_fold"]) != fold:
            raise ValueError(f"fold result identity mismatch at fold {fold}")
        if result.get("invariants", {}).get("held_score_computed") is not False:
            raise ValueError(f"fold {fold} reports held scores; merge refuses")
        data = _load_fold(prediction_path)
        fold_keys = list(map(str, data["keys"]))
        if len(fold_keys) != int(result["n_registered_prediction_rows"]):
            raise ValueError(f"fold {fold} row count mismatch")
        if len(set(fold_keys)) != len(fold_keys):
            raise ValueError(f"fold {fold} contains duplicate keys")
        fold_flow_keys = list(map(str, data["flow_keys"]))
        for name in STACK_FIELDS:
            if data[name].shape[0] != len(fold_keys):
                raise ValueError(f"fold {fold} field {name} misaligned with keys")
            stacked[name].append(np.asarray(data[name]))
        for name in FLOW_FIELDS:
            if data[name].shape[0] != len(fold_flow_keys):
                raise ValueError(f"fold {fold} field {name} misaligned with flow keys")
            flow_stacked[name].append(np.asarray(data[name]))
        keys.extend(fold_keys)
        flow_keys.extend(fold_flow_keys)
        fold_index.extend([fold] * len(fold_keys))
        fold_field_rows.append(len(fold_keys))
        flow_rows.append(len(fold_flow_keys))
    if len(set(keys)) != len(keys):
        raise ValueError("merged key universe contains duplicates across folds")
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(MERGE_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.asarray(fold_index, dtype=np.int64),
        "seed": np.full(len(keys), seed, dtype=np.int64),
        "flow_keys": np.asarray(flow_keys, dtype=object),
    }
    for name in STACK_FIELDS:
        output[name] = np.concatenate(stacked[name], axis=0)
    for name in FLOW_FIELDS:
        output[name] = np.concatenate(flow_stacked[name], axis=0)
    np.savez_compressed(marker, **output)
    audit_payload = {
        "schema_version": MERGE_SCHEMA,
        "seed": seed,
        "folds": list(folds),
        "n_rows": len(keys),
        "n_flow_rows": len(flow_keys),
        "fold_rows": dict(zip(folds, fold_field_rows)),
        "fold_flow_rows": dict(zip(folds, flow_rows)),
        "held_score_computed": False,
        "target_fields_present": False,
        "merge_once_marker": str(marker),
    }
    audit.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", default=",".join(str(f) for f in range(20)))
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if not str(out_dir).startswith("/mnt/cunyuliu/"):
        raise RuntimeError("merged artifacts must live under /mnt/cunyuliu")
    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    payload = merge_experiment(out_dir, seed=args.seed, folds=folds)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
