#!/usr/bin/env python3
"""Audit D0 pair relation categories without creating pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import write_json_document
from reactflow.delta.relations import classify_pair_relations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--construct-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = classify_pair_relations(args.candidate_registry, args.construct_audit)
    write_json_document(args.output, document)
    print(json.dumps(document["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
