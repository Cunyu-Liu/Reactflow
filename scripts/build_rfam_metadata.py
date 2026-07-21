#!/usr/bin/env python3
"""Build Rfam/MMseqs metadata TSV for ReactFlow eFold cache splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reactflow.rfam_metadata import build_rfam_metadata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="+", help="prepared eFold cache JSONL file(s)")
    parser.add_argument("--output", required=True, help="metadata TSV output path")
    parser.add_argument("--manifest", default="", help="JSON manifest path; defaults beside --output")
    parser.add_argument("--clan-membership", default="", help="optional Rfam clan_membership.txt(.gz)")
    parser.add_argument("--rfam-download-dir", default="", help="directory for downloaded Rfam database files")
    parser.add_argument("--no-download-rfam", action="store_true", help="do not download Rfam clan_membership.txt.gz")
    parser.add_argument("--cluster-method", choices=["auto", "exact", "mmseqs", "python-identity"], default="auto")
    parser.add_argument("--mmseqs", default="mmseqs", help="MMseqs2 executable")
    parser.add_argument("--mmseqs-min-seq-id", type=float, default=0.9)
    parser.add_argument("--mmseqs-coverage", type=float, default=0.8)
    parser.add_argument("--mmseqs-cov-mode", type=int, default=1)
    parser.add_argument("--python-identity-max-records", type=int, default=20000, help="safety limit for quadratic python-identity clustering")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--work-dir", default="", help="optional persistent MMseqs work directory")
    args = parser.parse_args(argv)

    summary = build_rfam_metadata(
        [Path(path) for path in args.cache],
        Path(args.output),
        manifest_path=Path(args.manifest) if args.manifest else None,
        clan_membership_path=Path(args.clan_membership) if args.clan_membership else None,
        rfam_download_dir=Path(args.rfam_download_dir) if args.rfam_download_dir else None,
        download_rfam=not args.no_download_rfam,
        cluster_method=args.cluster_method,
        mmseqs_bin=args.mmseqs,
        mmseqs_min_seq_id=args.mmseqs_min_seq_id,
        mmseqs_coverage=args.mmseqs_coverage,
        mmseqs_cov_mode=args.mmseqs_cov_mode,
        threads=args.threads,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        python_identity_max_records=args.python_identity_max_records,
    )
    print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
