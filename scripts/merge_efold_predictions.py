#!/usr/bin/env python3
"""Merge eFold worker prediction JSONL files into a single file per tier.

Workers process partitions of the gold split. This script concatenates all
worker outputs, deduplicates by record key (source_id/id/sequence), and writes
a single prediction file compatible with evaluate_external_baseline_predictions.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


def _record_key(obj: dict) -> str:
    for key in ("source_id", "id", "record_id", "reference"):
        value = obj.get(key)
        if value not in (None, ""):
            return str(value)
    sequence = obj.get("sequence")
    if sequence not in (None, ""):
        return f"sequence:{sequence}"
    return f"ordinal:{id(obj)}"


def merge_worker_outputs(worker_dirs: list[Path], pattern: str, output_path: Path) -> dict:
    """Merge worker output JSONL files into a single file.

    Returns stats dict with counts.
    """
    records: Dict[str, dict] = {}
    duplicate_count = 0
    files_processed = 0

    for wdir in worker_dirs:
        if not wdir.exists():
            print(f"[merge] Skipping non-existent dir: {wdir}", file=sys.stderr)
            continue
        for pred_file in sorted(wdir.glob(pattern)):
            files_processed += 1
            print(f"[merge] Processing {pred_file}", file=sys.stderr)
            with open(pred_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = _record_key(obj)
                    if key in records:
                        duplicate_count += 1
                        # Keep the first occurrence
                        continue
                    records[key] = obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for obj in records.values():
            f.write(json.dumps(obj) + "\n")

    stats = {
        "files_processed": files_processed,
        "unique_records": len(records),
        "duplicates_skipped": duplicate_count,
        "output_path": str(output_path),
    }
    print(f"[merge] Written {len(records)} records to {output_path}", file=sys.stderr)
    print(f"[merge] Stats: {stats}", file=sys.stderr)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge eFold worker predictions")
    parser.add_argument("--tier", required=True, choices=["in_clan", "novel_clan"],
                        help="Which tier to merge")
    parser.add_argument("--worker-dirs", nargs="+", required=True,
                        help="Worker output directories to merge")
    parser.add_argument("--pattern", default="*.efold.predictions.jsonl",
                        help="Glob pattern for prediction files")
    parser.add_argument("--output", required=True, help="Output prediction JSONL path")
    args = parser.parse_args()

    worker_dirs = [Path(d) for d in args.worker_dirs]
    output_path = Path(args.output)
    stats = merge_worker_outputs(worker_dirs, args.pattern, output_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
