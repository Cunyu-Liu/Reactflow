#!/usr/bin/env python3
"""Build D0 construct/pair candidate registries without final normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import write_json_document
from reactflow.delta.registry import (
    CONSTRUCT_CANDIDATE_COLUMNS,
    PAIR_CANDIDATE_COLUMNS,
    build_candidate_pair_registry,
    construct_candidate_rows,
    write_parquet_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct-audit", type=Path, required=True)
    parser.add_argument("--ribonanza-availability", type=Path, required=True)
    parser.add_argument("--pair-json-output", type=Path, required=True)
    parser.add_argument("--construct-parquet-output", type=Path, required=True)
    parser.add_argument("--pair-parquet-output", type=Path, required=True)
    args = parser.parse_args()
    registry = build_candidate_pair_registry(args.construct_audit, args.ribonanza_availability)
    construct_audit = json.loads(args.construct_audit.read_text())
    construct_rows = construct_candidate_rows(construct_audit)
    write_json_document(args.pair_json_output, registry)
    write_parquet_rows(args.construct_parquet_output, construct_rows, CONSTRUCT_CANDIDATE_COLUMNS)
    write_parquet_rows(args.pair_parquet_output, registry["candidate_pairs"], PAIR_CANDIDATE_COLUMNS)
    print(json.dumps({"construct_row_count": len(construct_rows), "pair_row_count": len(registry["candidate_pairs"]), "pair_json_output": str(args.pair_json_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
