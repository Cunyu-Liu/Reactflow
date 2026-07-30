#!/usr/bin/env python3
"""Parse frozen RDAT fixtures into a construct-level D0 audit artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reactflow.delta.data import write_json_document
from reactflow.delta.rdat import build_rdat_construct_parse_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_rdat_construct_parse_manifest(args.fixture_manifest)
    write_json_document(args.output, document)
    print(json.dumps({"fixture_count": document["fixture_count"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
