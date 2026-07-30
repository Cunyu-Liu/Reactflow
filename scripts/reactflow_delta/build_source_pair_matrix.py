#!/usr/bin/env python3
"""Write the D0 source/study/parent/probe/condition candidate matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import write_json_document
from reactflow.delta.matrix import build_source_pair_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct-audit", type=Path, required=True)
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--ribonanza-availability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_source_pair_matrix(args.construct_audit, args.candidate_registry, args.ribonanza_availability)
    write_json_document(args.output, document)
    print(json.dumps(document["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
