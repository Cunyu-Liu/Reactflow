#!/usr/bin/env python3
"""Verify frozen RDAT fixture bytes; do not parse RDAT content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import build_rdat_fixture_manifest, write_json_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_rdat_fixture_manifest(args.candidate_manifest, args.fixture_dir)
    write_json_document(args.output, manifest)
    print(json.dumps({"output": str(args.output), "fixture_count": len(manifest["fixtures"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
