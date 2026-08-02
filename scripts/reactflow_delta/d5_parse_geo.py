#!/usr/bin/env python3
"""D5: parse GEO mRNA functional datasets (GSE114002, GSE145046).

These datasets form the D5 "mRNA functional data" tier in the ReactFlow-Δ
data plan (sota_catchup_goals_and_todo.md §6.7). They provide UTR sequences
with translation/stability measurements, enabling future structure-function
relationship analysis and edit-effect benchmarking.

This script does NOT re-download raw files. It reuses the raw GEO data
already downloaded by the UTR Editflow project at --raw-dir (default
/mnt/cunyuliu/mrna_editflow_p0/) and parses them into ReactFlow-Δ-compatible
records with full provenance.

Outputs (to --out-dir, default
/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/d5/):
  - d5_geo_manifest.json       : per-file provenance (source URL, SHA-256,
                                  record counts) read from existing manifests
  - d5_gse114002_records.jsonl : one record per (utr, condition, replicate)
  - d5_gse145046_records.jsonl : one record per sequence with multi-condition
                                  read counts and derived labels
  - d5_geo_summary.json        : aggregate stats per dataset/condition

Forward-only: records are candidate D5 entries. No structure features are
computed here (deferred to a separate thermo build step).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# GSE114002 (Sample et al. 2019) — CSV.gz with UTR + ribosome load
# ---------------------------------------------------------------------------
# File naming convention: GSM{accession}_{library}_{condition}_{replicate}.csv.gz
# e.g. GSM3130435_egfp_unmod_1.csv.gz
# Columns: ,utr,0,1,...,13,total_reads,total,r0,...,r13,rl
# The first column is an index, 'utr' is the 50bp UTR sequence, 'rl' is the
# mean ribosome load (functional label). Columns 0-13 are bin proportions,
# r0-r13 are normalized bin proportions.

GSE114002_FILES = [
    # (filename, library, rna_chemistry, replicate, cds)
    ("GSM3130435_egfp_unmod_1.csv.gz",       "random_50mer", "unmodified",  1, "eGFP"),
    ("GSM3130436_egfp_unmod_2.csv.gz",       "random_50mer", "unmodified",  2, "eGFP"),
    ("GSM3130437_egfp_pseudo_1.csv.gz",      "random_50mer", "pseudouridine", 1, "eGFP"),
    ("GSM3130438_egfp_pseudo_2.csv.gz",      "random_50mer", "pseudouridine", 2, "eGFP"),
    ("GSM3130439_egfp_m1pseudo_1.csv.gz",    "random_50mer", "m1psi",       1, "eGFP"),
    ("GSM3130440_egfp_m1pseudo_2.csv.gz",    "random_50mer", "m1psi",       2, "eGFP"),
    ("GSM3130441_mcherry_1.csv.gz",          "random_50mer", "unmodified",  1, "mCherry"),
    ("GSM3130442_mcherry_2.csv.gz",          "random_50mer", "unmodified",  2, "mCherry"),
    ("GSM3130443_designed_library.csv.gz",   "designed",     "unmodified",  None, "eGFP"),
    ("GSM4084997_varying_length_25to100.csv.gz", "varying_length", "unmodified", None, "eGFP"),
]

# ---------------------------------------------------------------------------
# GSE145046 (PERSIST-seq) — TXT.gz with seq + read_count + norm
# ---------------------------------------------------------------------------
# Each file = one condition. Merge across files by sequence.
# Filename convention: GSM{accession}_{index}_read_count_{condition}.txt.gz

# Condition categories for derived label computation
GSE145046_CONDITION_MAP = {
    # translation-related (ribosome association)
    "In_vivo_Monosome":            ("translation", "monosome"),
    "In_vivo_Polysome":            ("translation", "polysome"),
    "In_vivo_Ribosome_free":       ("translation", "ribosome_free"),
    "Non_functional_cap_Ribosome_free": ("translation", "nfc_ribosome_free"),
    "Non_functional_cap_Ribosome_bound":("translation", "nfc_ribosome_bound"),
    # gating (sorted by fluorescence)
    "In_vivo_Gating_25D_low":      ("gating", "25d_low"),
    "In_vivo_Gating_25D_high":     ("gating", "25d_high"),
    "In_vivo_Gating_GFP_low":      ("gating", "gfp_low"),
    "In_vivo_Gating_GFP_high":     ("gating", "gfp_high"),
    # stability (half-life measurements)
    "In_vivo_Half_life_2h":        ("stability", "half_life_2h"),
    "In_vivo_Half_life_5h":        ("stability", "half_life_5h"),
    "In_vitro_Half_life_0min":     ("stability", "half_life_0min"),
    "In_vitro_Half_life_10min":    ("stability", "half_life_10min"),
    "In_vitro_Half_life_30min":    ("stability", "half_life_30min"),
    "In_vitro_Half_life_60min":    ("stability", "half_life_60min"),
    "Non_functional_cap_in_vivo_half_life_2h": ("stability", "nfc_half_life_2h"),
    "Non_functional_cap_in_vivo_half_life_5h": ("stability", "nfc_half_life_5h"),
    # input library
    "Randomly_synthesized_oligos": ("input", "random_oligos"),
}

# Regex to extract condition name from GSE145046 filename
# e.g. GSM4305123_2_read_count_In_vivo_Monosome_rep1.txt.gz -> In_vivo_Monosome
GSE145046_CONDITION_RE = re.compile(
    r"read_count_(.+?)(?:_rep\d+)?\.txt\.gz$"
)


def parse_gse114002_file(path: Path, library: str, chemistry: str,
                         replicate: int | None, cds: str) -> Iterator[dict]:
    """Parse one GSE114002 CSV.gz file, yielding record dicts.

    Each record: {utr, rl, library, rna_chemistry, replicate, cds,
                  total_reads, source_file}

    Note: some GSE114002 files have trailing gzip corruption (incomplete
    download). We read line-by-line and stop gracefully when decompression
    fails, recovering all valid records before the corruption point.
    """
    import zlib
    fh = gzip.open(path, "rt", newline="", encoding="utf-8")
    reader = csv.DictReader(fh)
    try:
        for row in reader:
            utr = (row.get("utr") or "").strip().upper()
            if not utr:
                continue
            # Skip non-ACGT characters (e.g., N, designed sequences with gaps)
            if not set("ACGT").issuperset(utr):
                continue
            rl_raw = row.get("rl")
            if rl_raw is None or rl_raw == "":
                continue
            try:
                rl = float(rl_raw)
            except (TypeError, ValueError):
                continue
            total_reads_raw = row.get("total_reads")
            try:
                total_reads = float(total_reads_raw) if total_reads_raw else None
            except (TypeError, ValueError):
                total_reads = None
            yield {
                "utr": utr,
                "rl": rl,
                "library": library,
                "rna_chemistry": chemistry,
                "replicate": replicate,
                "cds": cds,
                "total_reads": total_reads,
                "source_file": path.name,
            }
    except (zlib.error, EOFError, OSError):
        # Graceful stop on trailing corruption — records before this point are valid
        pass
    finally:
        fh.close()


def parse_gse145046_file(path: Path) -> tuple[str, str | None, int | None, list[dict]]:
    """Parse one GSE145046 TXT.gz file.

    Returns (condition_key, condition_category, replicate, records).
    Each record: {seq, read_count, norm_value}
    """
    name = path.name
    # Extract condition from filename
    m = GSE145046_CONDITION_RE.search(name)
    if not m:
        return ("unknown", None, None, [])
    condition_key = m.group(1)
    # Determine replicate from filename suffix
    rep_match = re.search(r"_rep(\d+)\.txt\.gz$", name)
    replicate = int(rep_match.group(1)) if rep_match else None
    # Map to category
    # Try exact match first, then prefix match
    category = None
    sub = None
    for key, (cat, s) in GSE145046_CONDITION_MAP.items():
        if condition_key == key:
            category = cat
            sub = s
            break
    if category is None:
        # Try prefix matching for conditions like "In_vivo_Monosome_rep1"
        # (already handled by regex, but fallback)
        for key, (cat, s) in GSE145046_CONDITION_MAP.items():
            if condition_key.startswith(key):
                category = cat
                sub = s
                break
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            seq = parts[0].strip().upper()
            if not seq:
                continue
            try:
                read_count = int(parts[1])
            except (TypeError, ValueError):
                try:
                    read_count = float(parts[1])
                except (TypeError, ValueError):
                    continue
            norm_value = None
            if len(parts) >= 3:
                try:
                    norm_value = float(parts[2])
                except (TypeError, ValueError):
                    pass
            records.append({
                "seq": seq,
                "read_count": read_count,
                "norm_value": norm_value,
            })
    return (condition_key, category, replicate, records)


def merge_gse145046_conditions(
    file_records: list[tuple[str, str | None, int | None, list[dict]]],
) -> dict[str, dict]:
    """Merge per-file records by sequence into per-sequence multi-condition dict.

    Returns {seq: {condition_key: [(replicate, read_count, norm_value), ...]}}
    """
    merged: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for condition_key, _category, replicate, records in file_records:
        for rec in records:
            merged[rec["seq"]][condition_key].append(
                (replicate, rec["read_count"], rec["norm_value"])
            )
    return merged


def compute_derived_labels(seq: str, conditions: dict) -> dict:
    """Compute derived functional labels from multi-condition read counts.

    Returns dict with:
      - te_estimate: translation efficiency = polysome / (monosome + polysome)
      - stability_estimate: half-life ratio (5h / (2h + 5h)) for in_vivo
      - total_reads_sum: sum of all read counts across conditions
    """
    labels = {}

    # Helper: average read counts across replicates for a condition prefix
    def avg_for_prefix(prefix: str) -> float | None:
        counts = []
        for ck, vals in conditions.items():
            if ck.startswith(prefix):
                for _rep, rc, _norm in vals:
                    if rc is not None:
                        counts.append(float(rc))
        return sum(counts) / len(counts) if counts else None

    mono = avg_for_prefix("In_vivo_Monosome")
    poly = avg_for_prefix("In_vivo_Polysome")
    if mono is not None and poly is not None and (mono + poly) > 0:
        labels["te_estimate"] = poly / (mono + poly)
    else:
        labels["te_estimate"] = None

    hl2 = avg_for_prefix("In_vivo_Half_life_2h")
    hl5 = avg_for_prefix("In_vivo_Half_life_5h")
    if hl2 is not None and hl5 is not None and (hl2 + hl5) > 0:
        labels["stability_estimate"] = hl5 / (hl2 + hl5)
    else:
        labels["stability_estimate"] = None

    total = 0.0
    for vals in conditions.values():
        for _rep, rc, _norm in vals:
            if rc is not None:
                total += float(rc)
    labels["total_reads_sum"] = total

    return labels


def load_source_manifest(raw_dir: Path, accession: str) -> dict:
    """Load the existing download manifest for provenance."""
    manifest_path = raw_dir / accession / "manifest.json"
    if not manifest_path.exists():
        return {"accession": accession, "manifest_found": False}
    with manifest_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("manifest_found", True)
    return data


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_gse114002(raw_dir: Path, out_dir: Path) -> dict:
    """Parse all GSE114002 files and write records + return stats."""
    data_dir = raw_dir / "GSE114002"
    records_path = out_dir / "d5_gse114002_records.jsonl"
    record_count = 0
    per_file_counts = {}
    utr_set: set[str] = set()
    chemistry_counts: dict[str, int] = defaultdict(int)

    with records_path.open("w", encoding="utf-8") as out_fh:
        for filename, library, chemistry, replicate, cds in GSE114002_FILES:
            path = data_dir / filename
            if not path.exists():
                per_file_counts[filename] = {"found": False, "records": 0}
                continue
            count = 0
            for rec in parse_gse114002_file(path, library, chemistry, replicate, cds):
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                utr_set.add(rec["utr"])
                chemistry_counts[chemistry] += 1
                count += 1
                record_count += 1
            per_file_counts[filename] = {"found": True, "records": count}

    stats = {
        "accession": "GSE114002",
        "total_records": record_count,
        "distinct_utrs": len(utr_set),
        "per_file": per_file_counts,
        "chemistry_distribution": dict(chemistry_counts),
    }
    return stats


def parse_gse145046(raw_dir: Path, out_dir: Path) -> dict:
    """Parse all GSE145046 files, merge by sequence, write records + stats."""
    data_dir = raw_dir / "GSE145046"
    records_path = out_dir / "d5_gse145046_records.jsonl"
    per_file_counts = {}
    file_records_list = []

    # Parse each file
    for path in sorted(data_dir.glob("GSM*_read_count_*.txt.gz")):
        condition_key, category, replicate, records = parse_gse145046_file(path)
        per_file_counts[path.name] = {
            "found": True,
            "condition": condition_key,
            "category": category,
            "replicate": replicate,
            "records": len(records),
        }
        file_records_list.append((condition_key, category, replicate, records))

    # Merge by sequence
    merged = merge_gse145046_conditions(file_records_list)

    # Filter to ACGT-only sequences and write records
    record_count = 0
    condition_set: set[str] = set()
    with records_path.open("w", encoding="utf-8") as out_fh:
        for seq in sorted(merged.keys()):
            if not set("ACGT").issuperset(seq):
                continue
            conditions = merged[seq]
            condition_set.update(conditions.keys())
            derived = compute_derived_labels(seq, conditions)
            # Build condition read-count summary
            cond_summary = {}
            for ck, vals in conditions.items():
                rcs = [float(rc) for _r, rc, _n in vals if rc is not None]
                cond_summary[ck] = {
                    "replicates": len(vals),
                    "read_count_sum": sum(rcs),
                    "read_count_mean": sum(rcs) / len(rcs) if rcs else None,
                }
            rec = {
                "seq": seq,
                "seq_length": len(seq),
                "conditions": cond_summary,
                "derived_labels": derived,
                "n_conditions": len(conditions),
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            record_count += 1

    stats = {
        "accession": "GSE145046",
        "total_records": record_count,
        "total_files_parsed": len(file_records_list),
        "distinct_conditions": sorted(condition_set),
        "per_file": per_file_counts,
    }
    return stats


def build_manifest(raw_dir: Path, out_dir: Path) -> dict:
    """Build D5 manifest with provenance from existing download manifests."""
    manifest = {
        "schema_version": "reactflow-delta-d5-geo-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "datasets": {},
    }
    for accession in ("GSE114002", "GSE145046"):
        src_manifest = load_source_manifest(raw_dir, accession)
        manifest["datasets"][accession] = {
            "source_url": src_manifest.get("source_url"),
            "retrieved_at_utc": src_manifest.get("retrieved_at_utc"),
            "manifest_found": src_manifest.get("manifest_found", True),
            "files": src_manifest.get("files", []),
            "skipped": src_manifest.get("skipped", []),
        }
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse GEO mRNA functional datasets (GSE114002, GSE145046) for D5 tier"
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("/mnt/cunyuliu/mrna_editflow_p0"),
        help="Root directory of raw GEO data (default: /mnt/cunyuliu/mrna_editflow_p0)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/d5"),
        help="Output directory for parsed records and manifest",
    )
    parser.add_argument(
        "--accession", choices=["GSE114002", "GSE145046", "all"], default="all",
        help="Which dataset to parse (default: all)",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[d5_parse_geo] raw_dir={args.raw_dir}")
    print(f"[d5_parse_geo] out_dir={args.out_dir}")
    print(f"[d5_parse_geo] accession={args.accession}")

    summary = {
        "schema_version": "reactflow-delta-d5-geo-summary-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(args.raw_dir),
        "out_dir": str(args.out_dir),
        "datasets": {},
    }

    if args.accession in ("GSE114002", "all"):
        print("[d5_parse_geo] parsing GSE114002...")
        stats = parse_gse114002(args.raw_dir, args.out_dir)
        summary["datasets"]["GSE114002"] = stats
        print(f"[d5_parse_geo] GSE114002: {stats['total_records']} records, "
              f"{stats['distinct_utrs']} distinct UTRs")

    if args.accession in ("GSE145046", "all"):
        print("[d5_parse_geo] parsing GSE145046...")
        stats = parse_gse145046(args.raw_dir, args.out_dir)
        summary["datasets"]["GSE145046"] = stats
        print(f"[d5_parse_geo] GSE145046: {stats['total_records']} records, "
              f"{stats['total_files_parsed']} files parsed")

    # Write manifest
    manifest = build_manifest(args.raw_dir, args.out_dir)
    manifest_path = args.out_dir / "d5_geo_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[d5_parse_geo] manifest -> {manifest_path}")

    # Write summary
    summary_path = args.out_dir / "d5_geo_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[d5_parse_geo] summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
