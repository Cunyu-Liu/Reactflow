#!/usr/bin/env python3
"""D0-X: requalify all frozen RMDB release assets against their frozen SHA-256.

Reads the frozen D0-X release-asset manifest and verifies, for every asset, that
the file already present in the raw root (or downloaded) matches the frozen
upstream SHA-256 and byte count.  Writes a run-linked requalification ledger
inside the D0-X artifact root.  This is a data-integrity gate only: it makes no
scientific claim about pair counts, eligibility, or learnability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reactflow.delta.d0x import sha256_file


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _requalify_one(asset: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    name = asset["asset_name"]
    path = raw_dir / name
    record = {
        "source_accession": asset["source_accession"],
        "source_group": asset["source_group"],
        "asset_name": name,
        "expected_sha256": asset["expected_sha256"],
        "expected_bytes": asset["expected_bytes"],
        "file_present": path.is_file(),
        "actual_bytes": path.stat().st_size if path.is_file() else None,
        "actual_sha256": None,
        "hash_match": False,
        "size_match": False,
        "disposition": "MISSING_FILE" if not path.is_file() else "HASH_PENDING",
        "verified_at": None,
    }
    if path.is_file():
        actual = _sha256(path)
        record["actual_sha256"] = actual
        record["size_match"] = record["actual_bytes"] == asset["expected_bytes"]
        record["hash_match"] = actual == asset["expected_sha256"]
        record["disposition"] = (
            "VERIFIED" if record["hash_match"] and record["size_match"] else "HASH_MISMATCH"
        )
        record["verified_at"] = datetime.now(timezone.utc).isoformat()
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset-manifest", type=Path, required=True,
                    help="frozen D0-X release asset JSONL")
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="raw RDAT root that must contain all frozen assets")
    ap.add_argument("--output", type=Path, required=True,
                    help="output requalification ledger JSON")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    assets: list[dict[str, Any]] = []
    with args.asset_manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                assets.append(json.loads(line))
    if not assets:
        print("no frozen assets", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_requalify_one, a, args.raw_dir): a for a in assets}
        for fut in as_completed(futures):
            records.append(fut.result())

    records.sort(key=lambda r: r["source_accession"])
    verified = [r for r in records if r["disposition"] == "VERIFIED"]
    mismatch = [r for r in records if r["disposition"] == "HASH_MISMATCH"]
    missing = [r for r in records if r["disposition"] == "MISSING_FILE"]

    ledger = {
        "schema_version": "reactflow_delta.d0x_requalification_ledger.v1",
        "run_id": f"d0x_requalify_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S+0000')}",
        "phase_id": "D0-X",
        "asset_manifest_path": str(args.asset_manifest),
        "raw_dir": str(args.raw_dir),
        "frozen_asset_count": len(assets),
        "verified_count": len(verified),
        "hash_mismatch_count": len(mismatch),
        "missing_count": len(missing),
        "all_verified": len(records) == len(verified),
        "records": records,
        "scientific_boundary": (
            "D0-X data-integrity requalification only; no exact pair count, "
            "eligibility, Tier, split, training, or scientific claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "frozen_asset_count": len(assets),
                "verified_count": len(verified),
                "hash_mismatch_count": len(mismatch),
                "missing_count": len(missing),
                "all_verified": ledger["all_verified"],
            },
            sort_keys=True,
        )
    )
    return 0 if ledger["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())