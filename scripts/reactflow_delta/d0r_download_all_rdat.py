#!/usr/bin/env python3
"""D0-R: batch download ALL RDAT files listed in the accession registry.

Reads data_registry/d0r_accession_registry.jsonl, downloads every rdat_url
that is not already present in --output-dir, and writes a per-file manifest.
Uses concurrent downloads with retry.  Data files land in /mnt (large storage);
this script itself lives in the GitHub repo at /home/cunyuliu/reactflow_delta_goal_20260729.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from reactflow.delta.manifests import sha256_file
except Exception:  # fallback if package path is unavailable
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


def _curl_download(url: str, dest: Path, timeout: int = 600, retries: int = 3) -> tuple[int, str]:
    """Download with curl + retry.  Returns (http_status, error_str)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for attempt in range(retries):
        cmd = [
            "curl", "-fsSL", "--max-time", str(timeout),
            "-o", str(dest),
            "-w", "%{http_code}",
            url,
        ]
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                code = int(result.stdout.strip().split("\n")[-1])
            except Exception:
                code = 200
            if code == 200 and dest.is_file() and dest.stat().st_size > 0:
                return (200, "")
            last_err = f"http={code} size={dest.stat().st_size if dest.is_file() else 0}"
        else:
            last_err = result.stderr.strip()[:200]
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return (0, last_err)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path,
                    default=Path("data_registry/d0r_accession_registry.jsonl"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="download only first N missing (0 = all)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    # Load registry
    entries: list[dict] = []
    with args.registry.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"[registry] {len(entries)} entries", file=sys.stderr)

    # Determine what is missing
    existing = {f.name for f in args.output_dir.glob("*.rdat")}
    todo: list[dict] = []
    skipped = 0
    for e in entries:
        rid = e.get("rmdb_id", "")
        fname = f"{rid}.rdat"
        if fname in existing:
            skipped += 1
            continue
        url = e.get("rdat_url", "")
        if not url:
            continue
        todo.append({"rmdb_id": rid, "filename": fname, "url": url,
                     "release_tag": url.split("/releases/download/")[-1].split("/")[0]
                     if "/releases/download/" in url else "unknown"})
    print(f"[skip] {skipped} already present", file=sys.stderr)
    print(f"[todo] {len(todo)} to download", file=sys.stderr)
    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"[limit] downloading first {len(todo)} only", file=sys.stderr)

    # Download with thread pool
    results: list[dict] = []
    success = 0
    failed: list[str] = []
    retrieved_at = datetime.now(timezone.utc).astimezone().isoformat()

    def _do_one(item: dict) -> dict:
        dest = args.output_dir / item["filename"]
        code, err = _curl_download(item["url"], dest, timeout=args.timeout)
        rec = {"rmdb_id": item["rmdb_id"], "filename": item["filename"],
               "url": item["url"], "release_tag": item["release_tag"],
               "http_status": code}
        if code == 200 and dest.is_file():
            rec.update({"download_status": "downloaded",
                        "bytes": dest.stat().st_size,
                        "sha256": sha256_file(dest),
                        "raw_path": str(dest.resolve())})
        else:
            rec.update({"download_status": "failed", "error": err,
                        "bytes": None, "sha256": None, "raw_path": None})
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_do_one, it): it for it in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            results.append(rec)
            if rec["download_status"] == "downloaded":
                success += 1
            else:
                failed.append(rec["rmdb_id"])
            if i % 25 == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} done, success={success}, failed={len(failed)}",
                      file=sys.stderr)

    manifest = {
        "schema_version": "reactflow-delta-d0r-rdat-batch-download-manifest-v1",
        "stage": "D0-R",
        "retrieved_at": retrieved_at,
        "source": "RMDB",
        "source_repository": "https://github.com/DasLab/rmdb.github.io",
        "registry_path": str(args.registry.resolve()),
        "registry_entries": len(entries),
        "already_present": skipped,
        "planned": len(todo),
        "success": success,
        "failed": failed,
        "workers": args.workers,
        "files": sorted(results, key=lambda r: r["rmdb_id"]),
        "scientific_boundary": (
            "Downloaded RDAT payloads are raw public data only. No construct, pair, tier, "
            "normalization, or training authorization is made by this manifest."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(json.dumps({"manifest": str(args.manifest.resolve()),
                      "success": success, "failed": len(failed),
                      "failed_ids": failed[:20]}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
