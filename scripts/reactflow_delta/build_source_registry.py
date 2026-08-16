#!/usr/bin/env python3
"""Create D0 RMDB metadata registry and raw manifest without downloading RDATs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import (
    build_rmdb_metadata_registry,
    write_json_document,
    write_jsonl_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True, help="ISO-8601 timestamp with timezone")
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, raw_manifest = build_rmdb_metadata_registry(
        args.metadata_dir,
        retrieved_at=args.retrieved_at,
    )
    write_jsonl_records(args.source_registry, records)
    write_json_document(args.raw_manifest, raw_manifest)
    print(
        json.dumps(
            {
                "source_registry": str(args.source_registry),
                "raw_manifest": str(args.raw_manifest),
                "record_count": len(records),
                "release_count": len(raw_manifest["release_summary"]),
                "scope": raw_manifest["scope"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
