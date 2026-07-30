#!/usr/bin/env python3
"""Audit parsed RMDB constructs before any candidate-pair construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import write_json_document
from reactflow.delta.pairing import build_rmdb_construct_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--construct-parse-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_rmdb_construct_audit(args.fixture_manifest, args.construct_parse_manifest)
    write_json_document(args.output, document)
    print(json.dumps({"output": str(args.output), "summary": document["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
