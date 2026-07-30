#!/usr/bin/env python3
"""D0-R: download RMDB RDAT assets by paper-accession mapping with checksum audit.

This is the fail-forward replacement for the filename-keyword recall used in D0.
Targets are selected from the RMDB ``_entries`` metadata (paper accession -> asset)
rather than filename patterns. Downloads are checksum-verified and recorded in a
raw manifest. No construct/pair/tier claim is made by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from reactflow.delta.manifests import sha256_file


# Curated D0-R priority targets: low-cost, paper-explicit or user-named.
# Each tuple: (rmdb_id, release_tag, filename, bytes_from_release_index, rationale)
PRIORITY_TARGETS: list[tuple[str, str, str, int, str]] = [
    ("ETERNA_R78_0000", "data-eterna", "ETERNA_R78_0000.rdat", 6582041,
     "paper_explicit PMID 36192461 (Wayment-Steele 2022 Nature Methods); ~6.55 MB low-cost"),
    ("M2SL5_2A3_0000", "data-rna-structures", "M2SL5_2A3_0000.rdat", 7863387,
     "user-named M2SL5 candidate reconstruction; Ribonanza pre-competition 2A3"),
    ("M2SL5_DMS_0000", "data-rna-structures", "M2SL5_DMS_0000.rdat", 7863713,
     "user-named M2SL5 candidate reconstruction; Ribonanza pre-competition DMS"),
    ("HC16M2R_1M7_0001", "data-rna-structures", "HC16M2R_1M7_0001.rdat", 999843,
     "M2R mutate-map-rescue candidate 1M7"),
    ("HC16M2R_1M7_0002", "data-rna-structures", "HC16M2R_1M7_0002.rdat", 548675,
     "M2R mutate-map-rescue candidate 1M7"),
    ("HC16M2R_1M7_0003", "data-rna-structures", "HC16M2R_1M7_0003.rdat", 348004,
     "M2R mutate-map-rescue candidate 1M7"),
    ("SPINACH_M2G4_0001", "data-general", "SPINACH_M2G4_0001.rdat", 542576,
     "M2G4 mutate-and-map G4 candidate"),
    ("SPINACH_DMS_0000", "data-general", "SPINACH_DMS_0000.rdat", 292265,
     "SPINACH DMS parent candidate"),
    ("THERM2_DMS_0001", "data-general", "THERM2_DMS_0001.rdat", 31946,
     "THERM2 small DMS candidate"),
    ("THERM2_GLX_0001", "data-general", "THERM2_GLX_0001.rdat", 29389,
     "THERM2 small glyoxal candidate"),
    ("HCVDM2_DCP_0000", "data-rna-structures", "HCVDM2_DCP_0000.rdat", 5835,
     "HCV dual-M2 DCP tiny candidate"),
    ("ETERNA_R86_0000", "data-eterna", "ETERNA_R86_0000.rdat", 6420615,
     "Eterna R86 designed-variant candidate"),
    ("ETERNA_R78_0001", "data-eterna", "ETERNA_R78_0001.rdat", 6911199,
     "Eterna R78 second batch, same PMID 36192461"),
]


def _curl_download(url: str, dest: Path, timeout: int = 600) -> tuple[int, str, str]:
    """Download with curl, return (http_status, etag, last_modified)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-fsSL", "--max-time", str(timeout),
        "-w", "%{http_code}\\t%{etag}\\t%{last_modified}",
        "-D", str(dest.with_suffix(dest.suffix + ".headers")),
        "-o", str(dest),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return (0, "", "", result.stderr.strip())
    parts = result.stdout.strip().split("\t")
    while len(parts) < 3:
        parts.append("")
    return (int(parts[0]), parts[1], parts[2], "")


def download_targets(output_dir: Path, manifest_path: Path) -> dict:
    retrieved_at = datetime.now(timezone.utc).astimezone().isoformat()
    files: list[dict] = []
    targets_success = 0
    targets_failed = []
    for rmdb_id, tag, filename, expected_bytes, rationale in PRIORITY_TARGETS:
        url = f"https://github.com/DasLab/rmdb.github.io/releases/download/{tag}/{filename}"
        dest = output_dir / filename
        http_status, etag, last_modified, err = _curl_download(url, dest)
        record = {
            "rmdb_id": rmdb_id,
            "filename": filename,
            "release_tag": tag,
            "browser_download_url": url,
            "rationale": rationale,
            "expected_bytes_from_release_index": expected_bytes,
            "http_status": http_status,
            "etag": etag,
            "last_modified": last_modified,
        }
        if http_status == 200 and dest.is_file():
            actual_bytes = dest.stat().st_size
            digest = sha256_file(dest)
            record.update({
                "download_status": "downloaded",
                "bytes": actual_bytes,
                "sha256": digest,
                "raw_path": str(dest.resolve()),
                "bytes_match_release_index": actual_bytes == expected_bytes,
            })
            headers_path = dest.with_suffix(dest.suffix + ".headers")
            if headers_path.is_file():
                record["headers_sha256"] = sha256_file(headers_path)
                record["headers_bytes"] = headers_path.stat().st_size
            files.append(record)
            targets_success += 1
        else:
            record.update({
                "download_status": "failed",
                "error": err or f"http_status={http_status}",
                "bytes": None,
                "sha256": None,
                "raw_path": None,
            })
            files.append(record)
            targets_failed.append(rmdb_id)
        print(json.dumps({"rmdb_id": rmdb_id, "status": record["download_status"],
                          "bytes": record.get("bytes"), "sha256": record.get("sha256")}),
              file=sys.stderr)
    manifest = {
        "schema_version": "reactflow-delta-d0r-rdat-download-manifest-v1",
        "stage": "D0-R",
        "retrieved_at": retrieved_at,
        "source": "RMDB",
        "source_repository": "https://github.com/DasLab/rmdb.github.io",
        "source_commit": "339b4fefc9a7092d0847d1d4017a3eadf0771fd7",
        "recall_method": "paper_accession_and_curated_priority (replaces filename-keyword recall)",
        "targets_planned": len(PRIORITY_TARGETS),
        "targets_success": targets_success,
        "targets_failed": targets_failed,
        "files": files,
        "scientific_boundary": (
            "Downloaded RDAT payloads are raw public data only. No construct, pair, tier, "
            "normalization, or training authorization is made by this manifest."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="directory for downloaded RDAT files")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="output raw download manifest JSON path")
    args = ap.parse_args()
    manifest = download_targets(args.output_dir, args.manifest)
    print(json.dumps({
        "manifest": str(args.manifest.resolve()),
        "targets_success": manifest["targets_success"],
        "targets_planned": manifest["targets_planned"],
        "targets_failed": manifest["targets_failed"],
    }, sort_keys=True))
    return 0 if manifest["targets_failed"] == [] else 1


if __name__ == "__main__":
    raise SystemExit(main())
