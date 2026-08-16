#!/usr/bin/env python3
"""C1-1 Task 3: Build the global RNA data registry.

This script loads all data sources, unifies them into :class:`DataRecord`
instances, annotates them with Rfam clan and MMseqs cluster metadata, builds
global contamination groups, and emits:

1. ``artifacts/c1_1/global_registry_manifest.json`` — per-source stats, total
   record count, deduplication info, build provenance.
2. ``artifacts/c1_1/contamination_groups.jsonl`` — one line per contamination
   group with ``group_id``, ``members``, ``size``.
3. ``artifacts/c1_1/global_registry_records.jsonl`` — one line per
   :class:`DataRecord` (the unified registry).

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 240-263.

Usage::

    python scripts/build_global_registry.py \
        --cache-dir artifacts/full_runs/full_ablation_20260709_003012/cache \
        --split-dir artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0 \
        --output-dir artifacts/c1_1 \
        [--source-version 2026-07-09] \
        [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for in-place execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reactflow.data_registry import (
    DataRecord,
    DataSourceSpec,
    KNOWN_SOURCES,
    RegistryStats,
    default_cache_dir,
    default_split_dir,
    iter_jsonl,
    load_cache_file,
)
from reactflow.contamination import (
    ContaminationGrouper,
    annotate_records_from_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the global RNA data registry (C1-1 Task 3)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="Directory containing the cache JSONL files.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=default_split_dir(),
        help="Directory containing the split_manifest.json (for cluster/clan annotation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/c1_1"),
        help="Output directory for artifacts.",
    )
    parser.add_argument(
        "--source-version",
        type=str,
        default="2026-07-09",
        help="Version string stamped on every record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit records per source (for smoke tests).",
    )
    parser.add_argument(
        "--emit-records",
        action="store_true",
        help="Emit global_registry_records.jsonl (large; off by default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_global_registry] cache_dir={args.cache_dir}")
    print(f"[build_global_registry] split_dir={args.split_dir}")
    print(f"[build_global_registry] output_dir={output_dir}")

    stats = RegistryStats()
    records: Dict[str, DataRecord] = {}
    seen_record_ids: Dict[str, int] = {}
    source_stats: Dict[str, Dict[str, int]] = {}

    for spec in KNOWN_SOURCES:
        cache_path = args.cache_dir / spec.cache_filename
        if not cache_path.exists():
            reason = "not downloaded (registered for provenance)" if not spec.downloaded else "cache file not found"
            print(f"[build_global_registry] SKIP {spec.name}: {cache_path} ({reason})")
            source_stats[spec.name] = {
                "loaded": 0,
                "duplicates_skipped": 0,
                "downloaded": spec.downloaded,
                "skipped_reason": reason,
            }
            continue
        print(f"[build_global_registry] loading {spec.name} from {cache_path}")
        source_count = 0
        source_duplicates = 0
        for record in load_cache_file(
            cache_path,
            source=spec.name,
            source_version=args.source_version,
            limit=args.limit,
        ):
            if record.record_id in records:
                source_duplicates += 1
                seen_record_ids[record.record_id] = seen_record_ids.get(record.record_id, 0) + 1
                # Keep first occurrence; this is a data-quality signal, not an error.
                continue
            records[record.record_id] = record
            stats.add(record)
            source_count += 1
        source_stats[spec.name] = {
            "loaded": source_count,
            "duplicates_skipped": source_duplicates,
            "downloaded": True,
        }
        print(f"[build_global_registry]   {spec.name}: {source_count} records "
              f"({source_duplicates} duplicates skipped)")

    # Compute unique checksums
    checksums = {r.checksum for r in records.values()}
    stats.unique_checksums = len(checksums)
    stats.duplicate_record_ids = sum(seen_record_ids.values())

    print(f"[build_global_registry] total records: {stats.total_records}")
    print(f"[build_global_registry] unique checksums: {stats.unique_checksums}")
    print(f"[build_global_registry] duplicate record IDs: {stats.duplicate_record_ids}")

    # Annotate with MMseqs cluster and Rfam clan from the split manifest
    split_manifest_path = args.split_dir / "split_manifest.json"
    if split_manifest_path.exists():
        print(f"[build_global_registry] annotating from {split_manifest_path}")
        n_annotated = annotate_records_from_split_manifest(records, split_manifest_path)
        print(f"[build_global_registry]   annotated {n_annotated} records with cluster/clan")
    else:
        print(f"[build_global_registry] WARNING: split_manifest.json not found at {split_manifest_path}")

    # Build contamination groups
    print("[build_global_registry] building contamination groups")
    grouper = ContaminationGrouper()
    grouper.add_records(records.values())
    grouper.merge_all()
    group_stats = grouper.stats_dict()
    print(f"[build_global_registry]   total groups: {group_stats['total_groups']}")
    print(f"[build_global_registry]   singleton groups: {group_stats['singleton_groups']}")
    print(f"[build_global_registry]   multi-record groups: {group_stats['multi_record_groups']}")
    print(f"[build_global_registry]   largest group size: {group_stats['largest_group_size']}")

    # Write contamination_groups.jsonl
    groups_path = output_dir / "contamination_groups.jsonl"
    n_groups_written = grouper.to_jsonl(groups_path)
    print(f"[build_global_registry] wrote {n_groups_written} groups to {groups_path}")

    # Optionally write full records JSONL
    records_path = output_dir / "global_registry_records.jsonl"
    if args.emit_records:
        print(f"[build_global_registry] writing records to {records_path}")
        with open(records_path, "w", encoding="utf-8") as f:
            for rid in sorted(records):
                f.write(json.dumps(records[rid].to_dict()) + "\n")
        print(f"[build_global_registry]   wrote {len(records)} records")

    # Write global_registry_manifest.json
    manifest = {
        "schema_version": "1.0",
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_version": args.source_version,
        "cache_dir": str(args.cache_dir),
        "split_dir": str(args.split_dir),
        "sources": [
            {
                "name": s.name,
                "cache_filename": s.cache_filename,
                "description": s.description,
                "has_real_profiles": s.has_real_profiles,
                "is_windowed": s.is_windowed,
                "downloaded": s.downloaded,
                "upstream_url": s.upstream_url,
                "upstream_license": s.upstream_license,
                "loaded": source_stats.get(s.name, {}).get("loaded", 0),
                "duplicates_skipped": source_stats.get(s.name, {}).get("duplicates_skipped", 0),
                "skipped_reason": source_stats.get(s.name, {}).get("skipped_reason"),
            }
            for s in KNOWN_SOURCES
        ],
        "registry_stats": stats.to_dict(),
        "contamination_stats": group_stats,
        "artifacts": {
            "contamination_groups": str(groups_path.relative_to(ROOT) if paths_are_relative(groups_path, ROOT) else groups_path),
            "global_registry_records": str(records_path.relative_to(ROOT) if paths_are_relative(records_path, ROOT) else records_path) if args.emit_records else None,
        },
        "record_count": len(records),
        "group_count": n_groups_written,
    }
    manifest_path = output_dir / "global_registry_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[build_global_registry] wrote manifest to {manifest_path}")

    return 0


def paths_are_relative(p1: Path, p2: Path) -> bool:
    """Return True if p1 is relative to p2."""
    try:
        p1.relative_to(p2)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())
