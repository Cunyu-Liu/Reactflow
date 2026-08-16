#!/usr/bin/env python3
"""Phase 3 closure -> benchmark/resource route (deliverable 4).

Write an auditable manifest (SHA256SUMS + manifest.json) over all committed
benchmark/resource artifacts (scripts, reports, diagnostic table, draft), so the
negative-result evidence chain is hash-pinned and reproducible.

CPU-only.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
from datetime import datetime, timezone


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"[manifest] WARN missing: {f}", file=sys.stderr)
            continue
        entries[str(p)] = sha256(p)

    manifest = {
        "schema": "reactflow_delta.phase3.benchmark_resource.manifest.v1",
        "run_id": Path(args.out_dir).name,
        "authority_epoch": 18,
        "phase": "PHASE3-BENCHMARK-RESOURCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as fh:
        for path, h in sorted(entries.items()):
            fh.write(f"{h}  {path}\n")
    print(json.dumps(manifest, indent=2))
    print(f"\n[manifest] wrote -> {out/'manifest.json'} and SHA256SUMS ({len(entries)} files)")


if __name__ == "__main__":
    raise SystemExit(main())
