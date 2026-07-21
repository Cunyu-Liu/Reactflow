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
