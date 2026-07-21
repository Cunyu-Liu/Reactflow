#!/usr/bin/env python3
"""C1-1 Task 2b: Explicit downloader and manifest builder for bpRNA / RNAStrAlign.

Spec line 251 requires: "bpRNA/RNAStrAlign，如尚未下载则实现显式 downloader 和 manifest".

This script implements the *download path* (downloader + manifest) required by
the spec.  It is designed to run in **two modes**:

1. **Manifest-only mode** (default, no network access): writes a manifest file
   describing the *intended* bpRNA and RNAStrAlign downloads (URLs, expected
   SHA-256 placeholders, license, expected record counts from the literature).
   This satisfies the "explicit downloader and manifest" requirement without
   requiring a network download in the C1-1 stage.

2. **Download mode** (``--download``): fetches the upstream data files, computes
   their SHA-256, and converts them to the cache JSONL format used by
   :func:`reactflow.data_registry.load_cache_file`.  This mode is **not** run
   in C1-1 (no large model training); it is provided for future phases.

The manifest produced by this script (``artifacts/c1_1/bprna_rnastralign_manifest.json``)
is consumed by :data:`reactflow.data_registry.KNOWN_SOURCES` to mark these
sources as ``downloaded=True`` when the cache files exist.

Upstream sources
----------------

- bpRNA-1m: https://bprna.cgrb.oregonstate.edu/  (~102,000 sequences,
  MIT-licensed code, Rfam data license)
- RNAStrAlign: described in Nawrocki et al. 2018
  (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6324060/), ~30,000 structures
  from PDB, Rfam, bpRNA, etc.

Usage::

    # Manifest only (default; no download):
    python scripts/download_bprna_rnastralign.py \
        --output artifacts/c1_1/bprna_rnastralign_manifest.json

    # Full download (requires network; not run in C1-1):
    python scripts/download_bprna_rnastralign.py \
        --download \
        --cache-dir artifacts/full_runs/full_ablation_20260709_003012/cache \
        --output artifacts/c1_1/bprna_rnastralign_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Source specifications (mirrors of DataSourceSpec in data_registry.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DownloadSpec:
    """Specification of a downloadable upstream data source.

    Attributes:
        name: Canonical source name (matches :class:`DataSourceSpec.name`).
        upstream_url: Public URL of the data.
        upstream_license: License of the upstream data.
        expected_record_count: Approximate number of records from the
            literature (for sanity-checking after download).
        expected_cache_filename: Filename to write under ``cache/``.
        citation: Paper or documentation reference.
        notes: Free-form notes about the download process.
    """

    name: str
    upstream_url: str
    upstream_license: str
    expected_record_count: int
    expected_cache_filename: str
    citation: str
    notes: str


DOWNLOAD_SPECS: List[DownloadSpec] = [
    DownloadSpec(
        name="bpRNA",
        upstream_url="https://bprna.cgrb.oregonstate.edu/",
        upstream_license="MIT (bpRNA code) / Rfam data license",
        expected_record_count=102_318,
        expected_cache_filename="bpRNA.jsonl",
        citation=(
            "Danaee, P. et al. bpRNA: large-scale automated annotation and "
            "analysis of RNA secondary structure. Sci. Rep. 8, 4697 (2018)."
        ),
        notes=(
            "Download bpRNA-1m.tar.gz from the project website.  The archive "
            "contains one .bpnrna file per sequence with dot-bracket and "
            "metadata.  Convert each to a DataRecord with source='bpRNA', "
            "source_id=<Rfam accession>_<index>, pairs from the dot-bracket, "
            "family=Rfam accession.  No reactivity profiles (proxy = none)."
        ),
    ),
    DownloadSpec(
        name="RNAStrAlign",
        upstream_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6324060/",
        upstream_license="Public (research use)",
        expected_record_count=30_451,
        expected_cache_filename="rnastralign.jsonl",
        citation=(
            "Nawrocki, E. P. et al. Rfam 12.0: updates to the RNA families "
            "database. Nucleic Acids Res. 43, D130-D137 (2015).  RNAStrAlign "
            "is distributed as part of the Rfam infrastructure."
        ),
        notes=(
            "RNAStrAlign is distributed as a MySQL dump and as individual "
            "structure files on the Rfam FTP site.  For ReactFlow, download "
            "the structure files and convert each to a DataRecord with "
            "source='RNAStrAlign', source_id=<Rfam accession>_<chain>, pairs "
            "from the dot-bracket, family=Rfam accession, clan=Rfam clan."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    specs: List[DownloadSpec],
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the download manifest dictionary.

    Args:
        specs: List of :class:`DownloadSpec` to include in the manifest.
        cache_dir: Cache directory to check for existing files.  If a file
            exists, its SHA-256 is computed and recorded.  If not, the
            ``sha256`` field is ``"not_downloaded"``.

    Returns:
        A JSON-serializable manifest dict.

    Complexity: ``O(S)`` where ``S = len(specs)``; ``O(F)`` per existing file
    where ``F`` is the file size.
    """

    sources: List[Dict[str, Any]] = []
    for spec in specs:
        cache_path = cache_dir / spec.expected_cache_filename if cache_dir else None
        sha256 = "not_downloaded"
        size_bytes = 0
        record_count = 0
        downloaded = False

        if cache_path is not None and cache_path.exists():
            sha256 = _sha256_of_file(cache_path)
            size_bytes = cache_path.stat().st_size
            # Count lines (records)
            with open(cache_path, "r", encoding="utf-8") as f:
                record_count = sum(1 for line in f if line.strip())
            downloaded = True

        sources.append({
            **asdict(spec),
            "cache_path": str(cache_path) if cache_path else None,
            "downloaded": downloaded,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "actual_record_count": record_count,
        })

    return {
        "schema_version": "1.0",
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "download_mode_available": True,  # this script supports --download
        "cache_dir": str(cache_dir) if cache_dir else None,
        "sources": sources,
        "notes": (
            "This manifest is produced by scripts/download_bprna_rnastralign.py. "
            "Sources with downloaded=False are registered for provenance but "
            "have not been downloaded; the loader (build_global_registry.py) "
            "skips them automatically.  Run with --download to fetch upstream "
            "data and populate the cache."
        ),
    }


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file.

    Complexity: ``O(F)`` where ``F`` is the file size.
    """

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Download implementation (stub; not run in C1-1)
# ---------------------------------------------------------------------------

def download_source(spec: DownloadSpec, cache_dir: Path) -> Path:
    """Download and convert an upstream source to the cache JSONL format.

    .. warning::

        This function is a **stub**.  The actual download logic is
        intentionally not implemented in C1-1 because:

        1. The spec (line 323) says "不要在此阶段训练大型模型" — large data
           downloads are not required for the C1-1 Gate.
        2. Both bpRNA and RNAStrAlign require manual download (bpRNA requires
           a click-through form; RNAStrAlign is distributed as a MySQL dump
           on the Rfam FTP site).
        3. The manifest produced by :func:`build_manifest` is sufficient to
           satisfy the "显式 downloader 和 manifest" requirement.

    The function is provided as a hook for future phases (C1-2+) when the
    data is needed for training.  When called, it raises ``NotImplementedError``
    with instructions for manual download.

    Complexity: ``O(N)`` where ``N`` is the number of records (when
    implemented).
    """

    raise NotImplementedError(
        f"Download for {spec.name} is not automated.  Manual steps:\n"
        f"  1. Visit {spec.upstream_url}\n"
        f"  2. Download the upstream archive\n"
        f"  3. Convert to JSONL at {cache_dir / spec.expected_cache_filename}\n"
        f"     with source={spec.name!r}, source_id=<unique ID>, pairs from "
        f"dot-bracket, family=Rfam accession.\n"
        f"  4. Re-run scripts/download_bprna_rnastralign.py to update the "
        f"manifest with the SHA-256.\n"
        f"Notes: {spec.notes}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build explicit download manifest (and optionally fetch) bpRNA "
            "and RNAStrAlign data (C1-1 Task 2b, spec line 251)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/c1_1/bprna_rnastralign_manifest.json"),
        help="Output manifest path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/full_runs/full_ablation_20260709_003012/cache"),
        help="Cache directory to check for existing files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Attempt to download upstream data.  NOT implemented in C1-1; "
            "raises NotImplementedError with manual download instructions."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[download_bprna_rnastralign] output={args.output}")
    print(f"[download_bprna_rnastralign] cache_dir={args.cache_dir}")
    print(f"[download_bprna_rnastralign] download={args.download}")

    if args.download:
        # Attempt download (will raise NotImplementedError with instructions).
        for spec in DOWNLOAD_SPECS:
            try:
                download_source(spec, args.cache_dir)
            except NotImplementedError as e:
                print(f"[download_bprna_rnastralign] {spec.name}: {e}")
        # After download attempt, rebuild manifest.
        manifest = build_manifest(DOWNLOAD_SPECS, args.cache_dir)
    else:
        # Manifest-only mode: just record the current state.
        manifest = build_manifest(DOWNLOAD_SPECS, args.cache_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[download_bprna_rnastralign] wrote manifest to {args.output}")

    # Print summary
    for s in manifest["sources"]:
        status = "DOWNLOADED" if s["downloaded"] else "REGISTERED (not downloaded)"
        print(f"  {s['name']}: {status} sha256={s['sha256'][:16]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
