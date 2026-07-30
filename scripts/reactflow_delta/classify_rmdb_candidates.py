#!/usr/bin/env python3
"""Build a D0 filename-only RMDB candidate manifest; do not infer experiment class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import build_rmdb_filename_candidate_manifest, write_json_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_rmdb_filename_candidate_manifest(args.release_index)
    write_json_document(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fixture_count": len(manifest["fixture_selection"]),
                "categories": {item["candidate_category"]: item["candidate_count"] for item in manifest["categories"]},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
