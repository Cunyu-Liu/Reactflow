#!/usr/bin/env python3
"""Record Ribonanza acquisition availability without downloading raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from reactflow.delta.availability import (
    RIBONANZA_COMPETITION_DATA_URL,
    RIBONANZA_TRAIN_DOWNLOAD_URL,
    build_ribonanza_availability_report,
    probe_http_head,
)
from reactflow.delta.data import write_json_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument("--kaggle-config", type=Path, default=Path("/home/cunyuliu/.kaggle/kaggle.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror-output", type=Path)
    args = parser.parse_args()
    probes = [probe_http_head(RIBONANZA_COMPETITION_DATA_URL), probe_http_head(RIBONANZA_TRAIN_DOWNLOAD_URL)]
    document = build_ribonanza_availability_report(
        retrieved_at=args.retrieved_at,
        kaggle_cli_path=shutil.which("kaggle"),
        kaggle_config_present=args.kaggle_config.is_file(),
        endpoint_probes=probes,
    )
    write_json_document(args.output, document)
    if args.mirror_output is not None:
        write_json_document(args.mirror_output, document)
    print(json.dumps({"data_access_outcome": document["data_access_outcome"], "output": str(args.output), "same_condition_single_edit_pair_count": document["same_condition_single_edit_pair_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
