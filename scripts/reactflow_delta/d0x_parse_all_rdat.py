#!/usr/bin/env python3
"""D0-X: parse every requalified RDAT asset with the strict D0-X parser.

Streams per-file candidate inventory to JSONL (one record per file, each with
its profile records) and maintains only aggregate counters in memory.  This
avoids holding the entire corpus in RAM.  Outputs land in the D0-X artifact
root.  Candidate inventory only; no exact-pair, eligibility, Tier, split,
training, or scientific claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from reactflow.delta.d0x import audit_rdat_candidate_profiles, D0XContractError

# asset_name is the 2nd top-level key (after "accounting") in the sort_keys
# JSONL records.  Regex extraction is far cheaper than json.loads on the huge
# per-file lines and is used only for resume bookkeeping.
_ASSET_NAME_RE = re.compile(rb'"asset_name": "([^"]+)"')


def _parse_one(asset: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    accession = asset["source_accession"]
    name = asset["asset_name"]
    path = raw_dir / name
    if not path.is_file():
        return {
            "source_accession": accession,
            "asset_name": name,
            "source_group": asset["source_group"],
            "status": "MISSING_FILE",
            "error": None,
            "profile_count": 0,
        }
    try:
        audit = audit_rdat_candidate_profiles(path, source_accession=accession)
        return {
            "source_accession": accession,
            "asset_name": name,
            "source_group": asset["source_group"],
            "status": "PARSED",
            "error": None,
            "file_sha256": audit["source_sha256"],
            "profile_count": len(audit["profile_records"]),
            "records": audit["profile_records"],
            "accounting": audit["profile_accounting"],
        }
    except D0XContractError as exc:
        return {
            "source_accession": accession,
            "asset_name": name,
            "source_group": asset["source_group"],
            "status": "PARSE_FAILED",
            "error": f"D0XContractError: {exc}",
            "profile_count": 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "source_accession": accession,
            "asset_name": name,
            "source_group": asset["source_group"],
            "status": "PARSE_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "profile_count": 0,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset-manifest", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True,
                    help="streamed per-file candidate inventory JSONL")
    ap.add_argument("--output-summary", type=Path, required=True,
                    help="aggregate summary JSON")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--resume-jsonl",
        type=Path,
        default=None,
        help="existing inventory JSONL; files already present are skipped",
    )
    args = ap.parse_args()

    assets: list[dict[str, Any]] = []
    with args.asset_manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                assets.append(json.loads(line))

    # Collect already-parsed asset names if resuming, so we never re-parse or
    # duplicate them.  Memory-bounded: only asset_name is kept.  Uses raw-byte
    # regex extraction instead of json.loads to avoid parsing the huge lines.
    already_parsed: set[str] = set()
    if args.resume_jsonl is not None and args.resume_jsonl.exists():
        with args.resume_jsonl.open("rb") as fh:
            for line in fh:
                m = _ASSET_NAME_RE.search(line)
                if m:
                    already_parsed.add(m.group(1).decode("utf-8"))
        pending = [a for a in assets if a["asset_name"] not in already_parsed]
    else:
        pending = list(assets)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    parsed_count = 0
    failed_count = 0
    missing_count = 0
    skipped_count = len(already_parsed)
    total_profile_count = 0
    role_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    missing_files: list[dict[str, Any]] = []

    # multiprocessing.Pool.imap_unordered streams results as they complete and
    # does NOT retain all completed results in the parent process, bounding
    # memory for the full 1024-file corpus.  chunksize=1 keeps per-task memory
    # low.  functools.partial keeps the worker callable picklable.
    mode = "a" if args.resume_jsonl is not None and args.resume_jsonl.exists() else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        with Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(
                partial(_parse_one, raw_dir=args.raw_dir), pending, chunksize=1
            ):
                status = result["status"]
                if status == "PARSED":
                    parsed_count += 1
                    total_profile_count += result["profile_count"]
                    for rec in result["records"]:
                        role_counter[rec.get("provisional_data_role")] += 1
                        status_counter[rec.get("exact_mutation_evidence_status")] += 1
                    out.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                elif status == "PARSE_FAILED":
                    failed_count += 1
                    failures.append(result)
                else:
                    missing_count += 1
                    missing_files.append(result)

    summary = {
        "schema_version": "reactflow_delta.d0x_candidate_inventory.v1",
        "run_id": f"d0x_parse_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S+0000')}",
        "phase_id": "D0-X",
        "asset_manifest_path": str(args.asset_manifest),
        "raw_dir": str(args.raw_dir),
        "frozen_asset_count": len(assets),
        "resume_skipped_already_parsed_count": skipped_count,
        "parsed_file_count": parsed_count,
        "parse_failed_file_count": failed_count,
        "missing_file_count": missing_count,
        "total_profile_count": total_profile_count,
        "profile_role_counts": dict(role_counter),
        "profile_exact_status_counts": dict(status_counter),
        "parse_failures": failures,
        "missing_files": missing_files,
        "scientific_boundary": (
            "D0-X candidate inventory only; no exact pair count, eligibility, "
            "Tier, split, training, or scientific claim."
        ),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "frozen_asset_count": len(assets),
                "resume_skipped_already_parsed_count": skipped_count,
                "parsed_file_count": parsed_count,
                "parse_failed_file_count": failed_count,
                "missing_file_count": missing_count,
                "total_profile_count": total_profile_count,
                "profile_role_counts": dict(role_counter),
            },
            sort_keys=True,
        )
    )
    return 0 if failed_count == 0 and missing_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())