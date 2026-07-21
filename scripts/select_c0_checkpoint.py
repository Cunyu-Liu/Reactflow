#!/usr/bin/env python3
"""Select a C0 checkpoint on validation only and freeze its hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from reactflow.c0_evaluate import aggregate_structure_records, sha256_path, structure_record_metrics
from reactflow.checkpoint import read_training_checkpoint
from reactflow.cli import _load_c0_samples
from reactflow.features import load_frozen_features
from reactflow.inference import DecoderConfig, InferenceConfig, InferenceMode, predict_structure


def parameter_count(checkpoint) -> int:
    params = checkpoint.result.parameters
    count = sum(len(row) for row in params.input_weight) + len(params.input_bias)
    count += sum(len(row) for row in params.pair_matrix)
    count += len(params.unpaired_weight) + 2
    adapter = checkpoint.result.adapter_parameters
    if adapter is not None:
        count += sum(len(row) for row in adapter.weight) + len(adapter.bias)
    return count


def select_checkpoint(
    paths: list[Path],
    validation_json: Path,
    *,
    validation_count: int,
    frozen_dir: Optional[Path],
) -> dict:
    samples = _load_c0_samples(str(validation_json), limit=validation_count)
    if not samples:
        raise ValueError("validation set is empty")
    frozen = load_frozen_features(frozen_dir) if frozen_dir is not None else None
    rows = []
    for path in sorted(set(paths)):
        checkpoint = read_training_checkpoint(path)
        records = []
        for sample in samples:
            result = predict_structure(
                checkpoint,
                sample.sequence,
                frozen,
                InferenceConfig(mode=InferenceMode.LEGACY_DIRECT),
                DecoderConfig(min_loop=checkpoint.config.min_loop),
            )
            records.append(
                {
                    "metrics": structure_record_metrics(result.structure, sample.pair_matrix),
                    "legal": result.validation.valid,
                    "runtime_seconds": result.runtime_seconds,
                }
            )
        rows.append(
            {
                "checkpoint_path": str(path.resolve()),
                "checkpoint_sha256": sha256_path(path),
                "parameter_count": parameter_count(checkpoint),
                **aggregate_structure_records(records),
            }
        )
    if not rows:
        raise ValueError("no completed checkpoints supplied")
    selected = max(
        rows,
        key=lambda row: (
            row.get("mean_exact_f1", 0.0),
            row.get("mean_shifted_f1", 0.0),
            -row["parameter_count"],
        ),
    )
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "selection_mode": "legacy_direct_regression_control",
        "test_metrics_used": False,
        "validation_path": str(validation_json.resolve()),
        "validation_sha256": sha256_path(validation_json),
        "validation_sample_ids": [sample.source_id for sample in samples],
        "candidates": rows,
        "selected": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = select_checkpoint(
        args.checkpoint,
        args.validation_json,
        validation_count=args.validation_count,
        frozen_dir=args.frozen_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": payload["selected"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
