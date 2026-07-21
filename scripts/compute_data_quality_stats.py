#!/usr/bin/env python3
"""C1-1 Task 6: Compute data quality statistics for the global registry.

Reads ``artifacts/c1_1/global_registry_records.jsonl`` and emits
``artifacts/c1_1/data_quality_stats.json`` with the statistics required by the
spec (lines 289-301):

- source distribution
- family/clan long-tail
- length distribution
- pair count
- pair distance
- pseudoknot ratio
- canonical/noncanonical
- profile missingness
- reads/SNR (documented as not-available in current cache; raw reads not stored)
- window lost-pair ratio
- fallback pseudo-clan fraction

Usage::

    python scripts/compute_data_quality_stats.py \
        --records artifacts/c1_1/global_registry_records.jsonl \
        --output artifacts/c1_1/data_quality_stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reactflow.data_registry import DataRecord, iter_jsonl


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute data quality statistics for the global registry (C1-1 Task 6)."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("artifacts/c1_1/global_registry_records.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/c1_1/data_quality_stats.json"),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit records (for smoke tests).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.records.exists():
        print(f"ERROR: records file not found: {args.records}")
        return 1

    print(f"[data_quality] reading {args.records}")

    # Accumulators
    lengths: List[int] = []
    pair_counts: List[int] = []
    pair_distances: List[int] = []
    pseudoknot_counts: List[int] = []
    canonical_counts: List[int] = []
    wobble_counts: List[int] = []
    noncanonical_counts: List[int] = []
    profile_missingness: List[float] = []  # fraction of None/0 positions
    source_counter: Counter = Counter()
    family_counter: Counter = Counter()
    clan_counter: Counter = Counter()
    probe_counter: Counter = Counter()
    length_bucket_counter: Counter = Counter()
    windowed_records = 0
    records_with_pseudoknot = 0
    records_with_family = 0
    records_with_clan = 0
    records_with_real_profile = 0
    records_with_proxy_profile = 0
    records_with_no_profile = 0
    total_records = 0

    # Window lost-pair ratio accumulators.
    # For each windowed record (parent_id is not None and parent_coordinates
    # is not None), we compute:
    #   window_length = parent_coordinates[1] - parent_coordinates[0]
    #   parent_length = record.parent_length (if available)
    #   coverage = window_length / parent_length
    #   lost_pair_ratio = 1.0 - coverage  (proxy; assumes uniform pair density)
    # We also group by parent_id to compute the total pair count observed
    # across all windows of the same parent, which is an upper bound on the
    # number of pairs that survived windowing.
    window_lost_pair_ratios: List[float] = []
    window_coverages: List[float] = []
    per_parent_window_counts: Dict[str, int] = {}
    per_parent_pair_counts: Dict[str, int] = {}

    # Reads/SNR accumulators.
    # The current cache JSONL format does not store raw read counts or SNR
    # (only the final reactivity profile after aggregation).  We track the
    # presence/absence of these fields so the report can document this
    # explicitly.
    records_with_reads_field = 0
    records_with_snr_field = 0
    reads_values: List[float] = []
    snr_values: List[float] = []

    for i, row in enumerate(iter_jsonl(args.records)):
        if args.max_records is not None and i >= args.max_records:
            break
        rec = DataRecord.from_dict(row)
        total_records += 1
        lengths.append(rec.length())
        pair_counts.append(len(rec.pairs))
        pseudoknot_counts.append(len(rec.pseudoknot_pairs))
        canonical_counts.append(rec.canonical_pair_count())
        wobble_counts.append(rec.wobble_pair_count())
        noncanonical_counts.append(rec.noncanonical_pair_count())
        for i_idx, j_idx in rec.pairs:
            pair_distances.append(j_idx - i_idx)
        source_counter[rec.source] += 1
        if rec.family:
            family_counter[rec.family] += 1
        if rec.clan:
            clan_counter[rec.clan] += 1
        else:
            # pseudo-clan = unannotated
            clan_counter["__unannotated__"] += 1
        probe_counter[rec.probe] += 1
        length_bucket_counter[rec.length_bucket] += 1
        if rec.window_index is not None:
            windowed_records += 1
        if rec.has_pseudoknot():
            records_with_pseudoknot += 1
        if rec.family:
            records_with_family += 1
        if rec.clan:
            records_with_clan += 1
        if rec.has_real_profile():
            records_with_real_profile += 1
        elif rec.reactivity_source == "structure_forward_proxy":
            records_with_proxy_profile += 1
        else:
            records_with_no_profile += 1
        # Profile missingness: fraction of positions with 0.0 or None
        if rec.reactivity is not None and len(rec.reactivity) > 0:
            missing = sum(1 for x in rec.reactivity if x is None or x == 0.0)
            profile_missingness.append(missing / len(rec.reactivity))

        # Window lost-pair ratio (spec line 300).
        # For windowed records, compute coverage = window_length / parent_length
        # and lost_pair_ratio = 1 - coverage (proxy).
        if rec.parent_id is not None and rec.parent_coordinates is not None:
            window_length = rec.parent_coordinates[1] - rec.parent_coordinates[0]
            if rec.parent_length is not None and rec.parent_length > 0:
                coverage = window_length / rec.parent_length
                window_coverages.append(coverage)
                window_lost_pair_ratios.append(1.0 - coverage)
            # Per-parent aggregation
            per_parent_window_counts[rec.parent_id] = (
                per_parent_window_counts.get(rec.parent_id, 0) + 1
            )
            per_parent_pair_counts[rec.parent_id] = (
                per_parent_pair_counts.get(rec.parent_id, 0) + len(rec.pairs)
            )

        # Reads/SNR fields (spec line 299).
        # The cache format does not currently store these; check for them
        # defensively in case future records include them.
        raw_row = row
        if "reads" in raw_row and raw_row["reads"] is not None:
            records_with_reads_field += 1
            try:
                reads_values.append(float(raw_row["reads"]))
            except (TypeError, ValueError):
                pass
        if "snr" in raw_row and raw_row["snr"] is not None:
            records_with_snr_field += 1
            try:
                snr_values.append(float(raw_row["snr"]))
            except (TypeError, ValueError):
                pass

    print(f"[data_quality] processed {total_records} records")

    # Compute summary stats
    total_pairs = sum(pair_counts)
    total_canonical = sum(canonical_counts)
    total_wobble = sum(wobble_counts)
    total_noncanonical = sum(noncanonical_counts)
    total_pseudoknot = sum(pseudoknot_counts)

    stats = {
        "schema_version": "1.0",
        "total_records": total_records,
        "source_distribution": dict(source_counter.most_common()),
        "probe_distribution": dict(probe_counter.most_common()),
        "length_bucket_distribution": dict(length_bucket_counter.most_common()),
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": statistics.mean(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "p25": _percentile([float(x) for x in lengths], 25),
            "p75": _percentile([float(x) for x in lengths], 75),
            "p95": _percentile([float(x) for x in lengths], 95),
        },
        "pair_count_stats": {
            "total_pairs": total_pairs,
            "mean_per_record": statistics.mean(pair_counts) if pair_counts else 0,
            "median_per_record": statistics.median(pair_counts) if pair_counts else 0,
            "max": max(pair_counts) if pair_counts else 0,
        },
        "pair_distance_stats": {
            "mean": statistics.mean(pair_distances) if pair_distances else 0,
            "median": statistics.median(pair_distances) if pair_distances else 0,
            "min": min(pair_distances) if pair_distances else 0,
            "max": max(pair_distances) if pair_distances else 0,
            "p25": _percentile([float(x) for x in pair_distances], 25),
            "p75": _percentile([float(x) for x in pair_distances], 75),
        },
        "pseudoknot_stats": {
            "records_with_pseudoknot": records_with_pseudoknot,
            "pseudoknot_ratio": records_with_pseudoknot / total_records if total_records else 0,
            "total_pseudoknot_pairs": total_pseudoknot,
        },
        "pair_chemistry": {
            "canonical": total_canonical,
            "wobble": total_wobble,
            "noncanonical": total_noncanonical,
            "canonical_fraction": total_canonical / total_pairs if total_pairs else 0,
            "wobble_fraction": total_wobble / total_pairs if total_pairs else 0,
            "noncanonical_fraction": total_noncanonical / total_pairs if total_pairs else 0,
        },
        "profile_stats": {
            "records_with_real_profile": records_with_real_profile,
            "records_with_proxy_profile": records_with_proxy_profile,
            "records_with_no_profile": records_with_no_profile,
            "mean_missingness": statistics.mean(profile_missingness) if profile_missingness else 0,
            "median_missingness": statistics.median(profile_missingness) if profile_missingness else 0,
        },
        "family_stats": {
            "records_with_family": records_with_family,
            "unique_families": len([k for k in family_counter if k != "__unannotated__"]),
            "top10_families": dict(family_counter.most_common(10)),
            "singleton_families": sum(1 for k, v in family_counter.items() if v == 1 and k != "__unannotated__"),
        },
        "clan_stats": {
            "records_with_clan": records_with_clan,
            "unique_clans": len([k for k in clan_counter if k != "__unannotated__"]),
            "top10_clans": dict(clan_counter.most_common(10)),
            "fallback_pseudo_clan_fraction": (
                clan_counter.get("__unannotated__", 0) / total_records if total_records else 0
            ),
        },
        "window_stats": {
            "windowed_records": windowed_records,
            "window_fraction": windowed_records / total_records if total_records else 0,
        },
        "window_lost_pair_ratio_stats": {
            "definition": (
                "Proxy: 1.0 - (window_length / parent_length) for windowed "
                "records.  Assumes uniform pair density across the parent.  "
                "True lost-pair ratio would require parent_pairs, which are "
                "not stored in the current cache format."
            ),
            "windowed_records_with_parent_length": len(window_lost_pair_ratios),
            "mean": statistics.mean(window_lost_pair_ratios) if window_lost_pair_ratios else 0,
            "median": statistics.median(window_lost_pair_ratios) if window_lost_pair_ratios else 0,
            "min": min(window_lost_pair_ratios) if window_lost_pair_ratios else 0,
            "max": max(window_lost_pair_ratios) if window_lost_pair_ratios else 0,
            "p25": _percentile(window_lost_pair_ratios, 25),
            "p75": _percentile(window_lost_pair_ratios, 75),
            "p95": _percentile(window_lost_pair_ratios, 95),
            "mean_window_coverage": (
                statistics.mean(window_coverages) if window_coverages else 0
            ),
            "median_window_coverage": (
                statistics.median(window_coverages) if window_coverages else 0
            ),
            "per_parent_window_count_distribution": {
                "unique_parents": len(per_parent_window_counts),
                "mean_windows_per_parent": (
                    statistics.mean(list(per_parent_window_counts.values()))
                    if per_parent_window_counts else 0
                ),
                "median_windows_per_parent": (
                    statistics.median(list(per_parent_window_counts.values()))
                    if per_parent_window_counts else 0
                ),
                "max_windows_per_parent": (
                    max(per_parent_window_counts.values()) if per_parent_window_counts else 0
                ),
            },
            "per_parent_pair_count_stats": {
                "mean_pairs_per_parent": (
                    statistics.mean(list(per_parent_pair_counts.values()))
                    if per_parent_pair_counts else 0
                ),
                "median_pairs_per_parent": (
                    statistics.median(list(per_parent_pair_counts.values()))
                    if per_parent_pair_counts else 0
                ),
                "max_pairs_per_parent": (
                    max(per_parent_pair_counts.values()) if per_parent_pair_counts else 0
                ),
            },
        },
        "reads_snr_stats": {
            "definition": (
                "Reads (sequencing depth) and SNR (signal-to-noise ratio) "
                "are not stored in the current cache JSONL format.  The "
                "cache stores only the final aggregated reactivity profile.  "
                "Raw reads/SNR would need to be re-parsed from upstream "
                "fastq/bam files (DMS/SHAPE/2A3 mapping experiments).  This "
                "field is included for spec compliance (line 299) and to "
                "document the gap."
            ),
            "records_with_reads_field": records_with_reads_field,
            "records_with_snr_field": records_with_snr_field,
            "reads_values_count": len(reads_values),
            "snr_values_count": len(snr_values),
            "reads_stats": {
                "mean": statistics.mean(reads_values) if reads_values else None,
                "median": statistics.median(reads_values) if reads_values else None,
                "min": min(reads_values) if reads_values else None,
                "max": max(reads_values) if reads_values else None,
            },
            "snr_stats": {
                "mean": statistics.mean(snr_values) if snr_values else None,
                "median": statistics.median(snr_values) if snr_values else None,
                "min": min(snr_values) if snr_values else None,
                "max": max(snr_values) if snr_values else None,
            },
            "status": (
                "not_available_in_cache" if not reads_values and not snr_values
                else "partial"
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"[data_quality] wrote stats to {args.output}")

    # Print summary
    print(f"[data_quality] total_records={total_records}")
    print(f"[data_quality] total_pairs={total_pairs} "
          f"(canonical={total_canonical}, wobble={total_wobble}, noncanonical={total_noncanonical})")
    print(f"[data_quality] pseudoknot_records={records_with_pseudoknot} "
          f"({records_with_pseudoknot/total_records*100:.1f}%)" if total_records else "")
    print(f"[data_quality] real_profiles={records_with_real_profile}, "
          f"proxy_profiles={records_with_proxy_profile}, "
          f"no_profiles={records_with_no_profile}")
    print(f"[data_quality] unique_families={stats['family_stats']['unique_families']}, "
          f"unique_clans={stats['clan_stats']['unique_clans']}")
    print(f"[data_quality] fallback_pseudo_clan_fraction="
          f"{stats['clan_stats']['fallback_pseudo_clan_fraction']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
