#!/usr/bin/env python3
"""Rebuild a ReactFlow sharded frozen-feature manifest from child shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def _read_provenance(shard_dir: Path) -> dict:
    return json.loads((shard_dir / "provenance.json").read_text(encoding="utf-8"))


def rebuild_manifest(directory: Path, *, shard_size: int) -> dict:
    shards = []
    model_name = None
    model_version = None
    weights_sha256 = None
    record_count = 0
    for shard_dir in sorted(p for p in directory.iterdir() if p.is_dir() and p.name.startswith("shard_")):
        required = [shard_dir / name for name in ("features.npz", "index.jsonl", "provenance.json")]
        if any(not path.exists() or path.stat().st_size <= 0 for path in required):
            continue
        provenance = _read_provenance(shard_dir)
        if model_name is None:
            model_name = provenance["model_name"]
            model_version = provenance["model_version"]
            weights_sha256 = provenance["weights_sha256"]
        elif (
            provenance["model_name"] != model_name
            or provenance["model_version"] != model_version
            or provenance["weights_sha256"] != weights_sha256
        ):
            raise ValueError(f"inconsistent provenance in {shard_dir}")
        count = int(provenance.get("record_count", 0))
        shards.append(
            {
                "path": shard_dir.name,
                "record_count": count,
                "content_sha256": str(provenance.get("content_sha256", "")),
                "weights_sha256": str(provenance["weights_sha256"]),
            }
        )
        record_count += count
    if not shards:
        raise ValueError(f"no complete child shards found in {directory}")
    manifest = {
        "layout": "reactflow-sharded-frozen-v1",
        "model_name": model_name,
        "model_version": model_version,
        "weights_sha256": weights_sha256,
        "record_count": record_count,
        "shard_count": len(shards),
        "shard_size": shard_size,
        "shards": shards,
    }
    (directory / "sharded_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild sharded frozen manifest from provenance files")
    parser.add_argument("--directory", required=True, type=Path, help="parent sharded frozen directory")
    parser.add_argument("--shard-size", required=True, type=int, help="records per full shard")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = rebuild_manifest(args.directory, shard_size=args.shard_size)
    print(json.dumps({k: manifest[k] for k in ("record_count", "shard_count", "shard_size", "weights_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
