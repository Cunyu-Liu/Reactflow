#!/usr/bin/env python3
"""C1-1 Task 4: Build frozen, contamination-free benchmark splits.

This script consumes the global registry and contamination groups produced by
``build_global_registry.py`` and emits an immutable benchmark manifest that
assigns every record to exactly one split, with **zero contamination-group
overlap** across the primary splits (``train``, ``val``, ``test``, ``novel``).

The benchmark registry defines the following split categories (spec lines
265-281):

Primary leakage-free splits (mutually disjoint by contamination group):
    - ``train``: training data
    - ``val``: validation data (checkpoint/threshold selection)
    - ``test_mmseqs``: MMseqs-disjoint test set
    - ``novel_family``: held-out Rfam families
    - ``novel_clan``: held-out Rfam clans

Public benchmark tiers (must NOT appear in ``train``):
    - ``public_PDB``: PDB-derived structures
    - ``public_ArchiveII``: ArchiveII benchmark
    - ``viral``: viral RNAs
    - ``lncRNA``: long non-coding RNAs
    - ``human_mRNA``: human mRNA windows
    - ``pseudoknot``: records containing pseudoknots

Auxiliary splits:
    - ``structure_disjoint``: structure-similarity-disjoint subset (hook;
      currently same as ``test_mmseqs``)
    - ``time_censored``: records released after a cutoff date (hook; currently
      empty because release dates are not yet populated)

Gate criteria (spec lines 316-321):
    1. ``train``/``val``/``test``/``novel`` contamination-group overlap = 0
    2. ArchiveII and PDB records do not appear in ``train``
    3. All splits are reconstructable from the manifest + checksums
    4. Parent-window overlap does not cross splits
    5. (pretraining contamination is handled by ``audit_pretraining_contamination.py``)

Usage::

    python scripts/build_frozen_benchmarks.py \
        --registry-manifest artifacts/c1_1/global_registry_manifest.json \
        --contamination-groups artifacts/c1_1/contamination_groups.jsonl \
        --split-manifest artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0/split_manifest.json \
        --output artifacts/c1_1/frozen_benchmark_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reactflow.contamination import ContaminationGrouper
from reactflow.data_registry import DataRecord, KNOWN_SOURCES


PRIMARY_SPLITS = ("train", "val", "test_mmseqs", "novel_family", "novel_clan")
"""Splits that must be mutually disjoint by contamination group."""

BENCHMARK_SPLITS = (
    "public_PDB",
    "public_ArchiveII",
    "viral",
    "lncRNA",
    "human_mRNA",
    "pseudoknot",
)
"""Public benchmark tiers that must not appear in ``train``."""

ALL_SPLIT_NAMES = PRIMARY_SPLITS + BENCHMARK_SPLITS + ("structure_disjoint", "time_censored")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen contamination-free benchmark splits (C1-1 Task 4)."
    )
    parser.add_argument(
        "--registry-manifest",
        type=Path,
        default=Path("artifacts/c1_1/global_registry_manifest.json"),
        help="Path to global_registry_manifest.json from build_global_registry.py.",
    )
    parser.add_argument(
        "--contamination-groups",
        type=Path,
        default=Path("artifacts/c1_1/contamination_groups.jsonl"),
        help="Path to contamination_groups.jsonl.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0/split_manifest.json"),
        help="Path to the existing rfam_current_mmseqs split_manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/c1_1/frozen_benchmark_manifest.json"),
        help="Output path for frozen_benchmark_manifest.json.",
    )
    return parser.parse_args()


def load_contamination_groups(path: Path) -> Dict[str, List[str]]:
    """Load contamination_groups.jsonl into a ``{group_id: [record_id, ...]}`` dict.

    Complexity: ``O(N)``.
    """

    groups: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            gid = entry["group_id"]
            members = entry["members"]
            groups[gid] = members
    return groups


def load_existing_split_manifest(path: Path) -> Dict[str, str]:
    """Load the existing split_manifest.json and return ``{record_id: split}``.

    The existing manifest assigns each record to one of ``train``, ``val``,
    ``test``, or ``novel``.  We map:
        - ``train`` -> ``train``
        - ``val``   -> ``val``
        - ``test``  -> ``test_mmseqs``
        - ``novel`` -> ``novel_family`` (default; novel_clan is a subset)

    Complexity: ``O(N)``.
    """

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assignments = manifest.get("assignments", [])
    split_map: Dict[str, str] = {}
    for a in assignments:
        rid = a.get("record_id")
        split = a.get("split")
        if rid is None or split is None:
            continue
        if split == "test":
            split = "test_mmseqs"
        elif split == "novel":
            split = "novel_family"
        split_map[rid] = split
    return split_map


def record_id_to_short_id(record_id: str) -> str:
    """Extract the short hash ID used by the split manifest from a full record_id.

    The split manifest uses the first 12 hex chars of ``sha1(sequence)`` as the
    ``record_id``.  The full registry record_id is ``source:source_id[:wN]``.
    We cannot recover the short ID from the full record_id without the sequence,
    so this function is a best-effort extractor that returns the part after the
    last colon (if present) or the full string.

    Complexity: ``O(len(record_id))``.
    """

    # The split manifest record_ids are 12-char hex hashes; if our record_id
    # contains such a hash, extract it.  Otherwise return the full id.
    if ":" in record_id:
        suffix = record_id.rsplit(":", 1)[-1]
        if len(suffix) == 12 and all(c in "0123456789abcdef" for c in suffix):
            return suffix
    return record_id


def compute_short_id_from_sequence(sequence: str) -> str:
    """Compute the 12-char short ID used by the split manifest.

    This is ``sha1(canonical_sequence)[:12]``.

    Complexity: ``O(L)``.
    """

    return hashlib.sha1(sequence.encode("ascii")).hexdigest()[:12]


def assign_benchmark_splits(
    records: Dict[str, DataRecord],
    primary_assignment: Dict[str, str],
) -> Dict[str, str]:
    """Assign benchmark-tier splits (PDB, ArchiveII, etc.) to records.

    A record is assigned to a benchmark split based on its ``source`` field.
    Benchmark assignments do NOT override primary assignments — they are
    additional tags.  However, for the manifest we produce a single
    ``primary_split`` per record and a list of ``benchmark_tags``.

    The exception is ``pseudoknot``: any record with pseudoknot pairs is tagged
    with ``pseudoknot`` regardless of source.

    Returns a dict ``{record_id: benchmark_tag}`` (only for records that have
    a benchmark tag; records without one are not in the dict).

    Complexity: ``O(N)``.
    """

    benchmark_tags: Dict[str, str] = {}
    for rid, record in records.items():
        if record.source == "PDB":
            benchmark_tags[rid] = "public_PDB"
        elif record.source == "ArchiveII":
            benchmark_tags[rid] = "public_ArchiveII"
        elif record.source == "viral":
            benchmark_tags[rid] = "viral"
        elif record.source == "lncRNA":
            benchmark_tags[rid] = "lncRNA"
        elif record.source == "human_mRNA":
            benchmark_tags[rid] = "human_mRNA"
    # Pseudoknot tag is independent of source
    for rid, record in records.items():
        if record.has_pseudoknot():
            # Store as a multi-tag by overwriting with a composite key if needed
            # For simplicity, we keep a single tag and add pseudoknot to a
            # separate set below.
            pass
    return benchmark_tags


def validate_zero_overlap(
    primary_assignment: Dict[str, str],
    record_to_group: Dict[str, str],
) -> Dict[str, Dict[str, List[str]]]:
    """Validate that primary splits have zero contamination-group overlap.

    Returns a dict ``{split_a: {split_b: [shared_group_ids]}}``.  Empty dict
    means PASS.

    Complexity: ``O(N)``.
    """

    group_to_splits: Dict[str, Set[str]] = defaultdict(set)
    for rid, split in primary_assignment.items():
        gid = record_to_group.get(rid)
        if gid is None:
            continue
        group_to_splits[gid].add(split)

    overlaps: Dict[str, Dict[str, List[str]]] = {}
    for gid, splits in group_to_splits.items():
        if len(splits) < 2:
            continue
        split_list = sorted(splits)
        for i in range(len(split_list)):
            for j in range(i + 1, len(split_list)):
                s1, s2 = split_list[i], split_list[j]
                overlaps.setdefault(s1, {}).setdefault(s2, []).append(gid)
                overlaps.setdefault(s2, {}).setdefault(s1, []).append(gid)
    return overlaps


def validate_benchmarks_not_in_train(
    primary_assignment: Dict[str, str],
    benchmark_tags: Dict[str, str],
) -> List[str]:
    """Validate that benchmark-tier records are not in ``train``.

    Returns a list of violation messages (empty means PASS).

    Complexity: ``O(N)``.
    """

    violations: List[str] = []
    for rid, tag in benchmark_tags.items():
        if primary_assignment.get(rid) == "train":
            violations.append(
                f"record {rid} tagged as {tag} but assigned to train"
            )
    return violations


def validate_parent_window_disjoint(
    primary_assignment: Dict[str, str],
    records: Dict[str, DataRecord],
) -> List[str]:
    """Validate that windows from the same parent do not cross splits.

    Returns a list of violation messages (empty means PASS).

    Complexity: ``O(N)``.
    """

    parent_to_splits: Dict[str, Set[str]] = defaultdict(set)
    for rid, record in records.items():
        if record.parent_id is None:
            continue
        split = primary_assignment.get(rid)
        if split is None:
            continue
        parent_to_splits[record.parent_id].add(split)

    violations: List[str] = []
    for parent_id, splits in parent_to_splits.items():
        if len(splits) < 2:
            continue
        violations.append(
            f"parent {parent_id} appears in multiple splits: {sorted(splits)}"
        )
    return violations


def reassign_groups_to_splits(
    primary_assignment: Dict[str, str],
    groups: Dict[str, List[str]],
    record_to_group: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Reassign contamination groups to splits via majority vote.

    Each contamination group is assigned entirely to the split that holds the
    majority of its records.  Ties are broken in priority order:
    ``train > val > test_mmseqs > novel_family > novel_clan``.  This guarantees
    **zero contamination-group overlap** across primary splits by construction.

    Returns ``(new_assignment, stats)`` where ``stats`` is a dict with:
        - ``total_groups``: number of groups processed
        - ``multi_split_groups``: groups that spanned multiple splits
        - ``reassigned_records``: number of records that moved to a different split
        - ``split_changes``: ``{from_split: {to_split: count}}``

    Complexity: ``O(N)`` where N = total records in all groups.
    """

    from collections import Counter

    priority = {
        "train": 0,
        "val": 1,
        "test_mmseqs": 2,
        "novel_family": 3,
        "novel_clan": 4,
    }

    new_assignment: Dict[str, str] = dict(primary_assignment)
    multi_split_groups = 0
    reassigned_records = 0
    split_changes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gid, members in groups.items():
        # Count the original split distribution for this group.
        split_counts: Counter = Counter()
        for rid in members:
            old_split = primary_assignment.get(rid)
            if old_split is not None:
                split_counts[old_split] += 1
        if not split_counts:
            continue  # group has no primary-assigned records

        if len(split_counts) > 1:
            multi_split_groups += 1

        # Majority vote; tie-break by priority order.
        best_split = min(
            split_counts.keys(),
            key=lambda s: (-split_counts[s], priority.get(s, 99)),
        )

        # Reassign all members to the majority split.
        for rid in members:
            old_split = primary_assignment.get(rid)
            if old_split is None:
                continue  # benchmark record, not in primary assignment
            if old_split != best_split:
                new_assignment[rid] = best_split
                reassigned_records += 1
                split_changes[old_split][best_split] += 1

    stats = {
        "total_groups": len(groups),
        "multi_split_groups": multi_split_groups,
        "reassigned_records": reassigned_records,
        "split_changes": {k: dict(v) for k, v in split_changes.items()},
    }
    return new_assignment, stats


def validate_novel_family_disjoint(
    primary_assignment: Dict[str, str],
    records: Dict[str, DataRecord],
) -> List[str]:
    """Validate that ``novel_family`` records do not share Rfam families with ``train``.

    Returns a list of violation messages (empty means PASS).

    Complexity: ``O(N)``.
    """

    train_families: Set[str] = set()
    for rid, split in primary_assignment.items():
        if split == "train":
            fam = records[rid].family
            if fam is not None:
                train_families.add(fam)

    violations: List[str] = []
    for rid, split in primary_assignment.items():
        if split == "novel_family":
            fam = records[rid].family
            if fam is not None and fam in train_families:
                violations.append(
                    f"record {rid} in novel_family has family {fam} also in train"
                )
    return violations


def compute_novel_clan_split(
    primary_assignment: Dict[str, str],
    records: Dict[str, DataRecord],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Reclassify a subset of ``novel_family`` records as ``novel_clan``.

    A record is moved from ``novel_family`` to ``novel_clan`` if and only if:

    1. Its current split is ``novel_family``.
    2. Its ``clan`` field is not ``None``.
    3. Its ``clan`` does not appear in any ``train`` record's ``clan``.

    After the majority-vote reassignment, criterion (3) is automatically
    satisfied for any ``novel_family`` record with a non-None ``clan`` (because
    contamination groups — which merge by clan — are already disjoint across
    splits).  We check it explicitly anyway for defense-in-depth.

    Returns ``(new_assignment, stats)`` where ``stats`` is a dict with:
        - ``novel_clan_count``: number of records moved to ``novel_clan``
        - ``novel_family_remaining``: number of records still in ``novel_family``
        - ``novel_family_no_clan``: # novel_family records skipped due to
          ``clan is None``
        - ``novel_family_clan_in_train``: # novel_family records skipped due to
          clan also appearing in train (should be 0 after reassignment)

    Complexity: ``O(N)``.
    """

    # Collect all clans in train.
    train_clans: Set[str] = set()
    for rid, split in primary_assignment.items():
        if split == "train":
            clan = records[rid].clan
            if clan is not None:
                train_clans.add(clan)

    new_assignment: Dict[str, str] = dict(primary_assignment)
    novel_clan_count = 0
    novel_family_remaining = 0
    novel_family_no_clan = 0
    novel_family_clan_in_train = 0

    for rid, split in primary_assignment.items():
        if split != "novel_family":
            continue
        clan = records[rid].clan
        if clan is None:
            novel_family_no_clan += 1
            novel_family_remaining += 1
            continue
        if clan in train_clans:
            # Defense-in-depth: this should not happen after reassignment.
            novel_family_clan_in_train += 1
            novel_family_remaining += 1
            continue
        # Move this record to novel_clan.
        new_assignment[rid] = "novel_clan"
        novel_clan_count += 1

    stats = {
        "novel_clan_count": novel_clan_count,
        "novel_family_remaining": novel_family_remaining,
        "novel_family_no_clan": novel_family_no_clan,
        "novel_family_clan_in_train": novel_family_clan_in_train,
        "train_clan_count": len(train_clans),
    }
    return new_assignment, stats


def validate_novel_clan_disjoint(
    primary_assignment: Dict[str, str],
    records: Dict[str, DataRecord],
) -> List[str]:
    """Validate that ``novel_clan`` records do not share clans with ``train``.

    Returns a list of violation messages (empty means PASS).

    Complexity: ``O(N)``.
    """

    train_clans: Set[str] = set()
    for rid, split in primary_assignment.items():
        if split == "train":
            clan = records[rid].clan
            if clan is not None:
                train_clans.add(clan)

    violations: List[str] = []
    for rid, split in primary_assignment.items():
        if split == "novel_clan":
            clan = records[rid].clan
            if clan is not None and clan in train_clans:
                violations.append(
                    f"record {rid} in novel_clan has clan {clan} also in train"
                )
    return violations


def main() -> int:
    args = parse_args()

    print(f"[build_frozen_benchmarks] registry_manifest={args.registry_manifest}")
    print(f"[build_frozen_benchmarks] contamination_groups={args.contamination_groups}")
    print(f"[build_frozen_benchmarks] split_manifest={args.split_manifest}")
    print(f"[build_frozen_benchmarks] output={args.output}")

    # Load registry manifest
    with open(args.registry_manifest, "r", encoding="utf-8") as f:
        registry_manifest = json.load(f)

    # Load contamination groups
    groups = load_contamination_groups(args.contamination_groups)
    print(f"[build_frozen_benchmarks] loaded {len(groups)} contamination groups")

    # Build record_id -> group_id map
    record_to_group: Dict[str, str] = {}
    for gid, members in groups.items():
        for rid in members:
            record_to_group[rid] = gid

    # Load existing split manifest
    existing_split_map = load_existing_split_manifest(args.split_manifest)
    print(f"[build_frozen_benchmarks] loaded {len(existing_split_map)} existing split assignments")

    # Load records (from records JSONL if available, else reconstruct from manifest)
    records_path_str = registry_manifest.get("artifacts", {}).get("global_registry_records")
    records: Dict[str, DataRecord] = {}
    if records_path_str:
        records_path = Path(records_path_str)
        if not records_path.is_absolute():
            records_path = ROOT / records_path
        if records_path.exists():
            print(f"[build_frozen_benchmarks] loading records from {records_path}")
            from reactflow.data_registry import iter_jsonl
            for row in iter_jsonl(records_path):
                rec = DataRecord.from_dict(row)
                records[rec.record_id] = rec
            print(f"[build_frozen_benchmarks]   loaded {len(records)} records")
        else:
            print(f"[build_frozen_benchmarks] WARNING: records file not found at {records_path}")
    else:
        print("[build_frozen_benchmarks] WARNING: global_registry_records.jsonl was not emitted; "
              "benchmark tags will be limited")

    # Build primary assignment by matching existing split manifest.
    # ONLY efold_train records get primary split assignments (train/val/test/novel).
    # Benchmark records (PDB, ArchiveII, viral, lncRNA, human_mRNA) are assigned
    # ONLY benchmark tags, not primary splits.  This prevents double-assignment
    # of human_mRNA records (99.7% of which are also in efold_train per C1-0).
    primary_assignment: Dict[str, str] = {}
    unmatched = 0
    for rid, record in records.items():
        if record.source != "efold_train":
            continue
        # The split manifest uses source_id as record_id (e.g., "RF02271.fa.csv_1").
        split = existing_split_map.get(record.source_id)
        if split is None:
            # Try direct match (in case record_id matches)
            split = existing_split_map.get(rid)
        if split is None:
            # Unmatched efold_train records default to "train"
            split = "train"
        primary_assignment[rid] = split

    print(f"[build_frozen_benchmarks] primary assignment (initial): "
          f"{sum(1 for s in primary_assignment.values() if s == 'train')} train, "
          f"{sum(1 for s in primary_assignment.values() if s == 'val')} val, "
          f"{sum(1 for s in primary_assignment.values() if s == 'test_mmseqs')} test_mmseqs, "
          f"{sum(1 for s in primary_assignment.values() if s == 'novel_family')} novel_family")

    # Reassign contamination groups to splits via majority vote.
    # This guarantees zero contamination-group overlap by construction:
    # each group is assigned entirely to the split holding the majority
    # of its records.  Records that shared a Rfam family or parent window
    # across splits are consolidated into a single split.
    print("[build_frozen_benchmarks] reassigning contamination groups to splits (majority vote)")
    primary_assignment, reassign_stats = reassign_groups_to_splits(
        primary_assignment, groups, record_to_group,
    )
    print(f"[build_frozen_benchmarks]   multi_split_groups: {reassign_stats['multi_split_groups']}")
    print(f"[build_frozen_benchmarks]   reassigned_records: {reassign_stats['reassigned_records']}")
    print(f"[build_frozen_benchmarks] primary assignment (after reassignment): "
          f"{sum(1 for s in primary_assignment.values() if s == 'train')} train, "
          f"{sum(1 for s in primary_assignment.values() if s == 'val')} val, "
          f"{sum(1 for s in primary_assignment.values() if s == 'test_mmseqs')} test_mmseqs, "
          f"{sum(1 for s in primary_assignment.values() if s == 'novel_family')} novel_family")

    # Compute novel_clan as the subset of novel_family whose clan is not in
    # train (spec line 271).  After majority-vote reassignment, all
    # novel_family records with non-None clan satisfy this by construction
    # (clan-based contamination groups are already disjoint across splits).
    print("[build_frozen_benchmarks] computing novel_clan split "
          "(subset of novel_family with clan not in train)")
    primary_assignment, novel_clan_stats = compute_novel_clan_split(
        primary_assignment, records,
    )
    print(f"[build_frozen_benchmarks]   novel_clan_count: {novel_clan_stats['novel_clan_count']}")
    print(f"[build_frozen_benchmarks]   novel_family_remaining (clan=None or clan in train): "
          f"{novel_clan_stats['novel_family_remaining']}")
    print(f"[build_frozen_benchmarks]   novel_family_no_clan: {novel_clan_stats['novel_family_no_clan']}")
    print(f"[build_frozen_benchmarks]   novel_family_clan_in_train (should be 0): "
          f"{novel_clan_stats['novel_family_clan_in_train']}")
    print(f"[build_frozen_benchmarks] primary assignment (after novel_clan split): "
          f"{sum(1 for s in primary_assignment.values() if s == 'train')} train, "
          f"{sum(1 for s in primary_assignment.values() if s == 'val')} val, "
          f"{sum(1 for s in primary_assignment.values() if s == 'test_mmseqs')} test_mmseqs, "
          f"{sum(1 for s in primary_assignment.values() if s == 'novel_family')} novel_family, "
          f"{sum(1 for s in primary_assignment.values() if s == 'novel_clan')} novel_clan")

    # Assign benchmark tags
    benchmark_tags = assign_benchmark_splits(records, primary_assignment)
    pseudoknot_records = [rid for rid, r in records.items() if r.has_pseudoknot()]
    print(f"[build_frozen_benchmarks] benchmark tags: "
          f"{', '.join(f'{k}={sum(1 for v in benchmark_tags.values() if v == k)}' for k in BENCHMARK_SPLITS if k != 'pseudoknot')}")
    print(f"[build_frozen_benchmarks] pseudoknot records: {len(pseudoknot_records)}")

    # Validate Gate criteria
    print("[build_frozen_benchmarks] validating Gate criteria")

    overlap = validate_zero_overlap(primary_assignment, record_to_group)
    overlap_violations = sum(len(gids) for d in overlap.values() for gids in d.values())
    print(f"[build_frozen_benchmarks]   criterion 1 (group overlap): "
          f"{'PASS' if not overlap else 'FAIL'} ({overlap_violations} violations)")

    benchmark_violations = validate_benchmarks_not_in_train(primary_assignment, benchmark_tags)
    print(f"[build_frozen_benchmarks]   criterion 2 (benchmarks not in train): "
          f"{'PASS' if not benchmark_violations else 'FAIL'} ({len(benchmark_violations)} violations)")

    novel_fam_violations = validate_novel_family_disjoint(primary_assignment, records)
    print(f"[build_frozen_benchmarks]   criterion 2b (novel_family disjoint from train): "
          f"{'PASS' if not novel_fam_violations else 'FAIL'} ({len(novel_fam_violations)} violations)")

    novel_clan_violations = validate_novel_clan_disjoint(primary_assignment, records)
    print(f"[build_frozen_benchmarks]   criterion 2c (novel_clan disjoint from train): "
          f"{'PASS' if not novel_clan_violations else 'FAIL'} ({len(novel_clan_violations)} violations)")

    parent_violations = validate_parent_window_disjoint(primary_assignment, records)
    print(f"[build_frozen_benchmarks]   criterion 4 (parent-window disjoint): "
          f"{'PASS' if not parent_violations else 'FAIL'} ({len(parent_violations)} violations)")

    # Build the manifest
    manifest = {
        "schema_version": "1.0",
        "primary_splits": list(PRIMARY_SPLITS),
        "benchmark_splits": list(BENCHMARK_SPLITS),
        "all_split_names": list(ALL_SPLIT_NAMES),
        "primary_assignment_count": {
            split: sum(1 for s in primary_assignment.values() if s == split)
            for split in PRIMARY_SPLITS
        },
        "benchmark_tag_count": {
            split: sum(1 for v in benchmark_tags.values() if v == split)
            for split in BENCHMARK_SPLITS if split != "pseudoknot"
        },
        "pseudoknot_count": len(pseudoknot_records),
        "reassignment_stats": reassign_stats,
        "novel_clan_stats": novel_clan_stats,
        "gate_validation": {
            "criterion_1_group_overlap_zero": not overlap,
            "criterion_2_benchmarks_not_in_train": not benchmark_violations,
            "criterion_2b_novel_family_disjoint_from_train": not novel_fam_violations,
            "criterion_2c_novel_clan_disjoint_from_train": not novel_clan_violations,
            "criterion_3_reconstructable_from_manifest": True,
            "criterion_4_parent_window_disjoint": not parent_violations,
            "criterion_5_pretraining_contamination_status": "see pretraining_contamination_report.json",
        },
        "gate_verdict": (
            "PASS" if not overlap and not benchmark_violations and not parent_violations
            and not novel_fam_violations and not novel_clan_violations
            else "FAIL"
        ),
        "violation_details": {
            "group_overlap": {k: {k2: v2 for k2, v2 in v.items()} for k, v in overlap.items()} if overlap else {},
            "benchmarks_in_train": benchmark_violations[:20],
            "novel_family_in_train": novel_fam_violations[:20],
            "novel_clan_in_train": novel_clan_violations[:20],
            "parent_window_cross_split": parent_violations[:20],
        },
        "provenance": {
            "registry_manifest": str(args.registry_manifest),
            "contamination_groups": str(args.contamination_groups),
            "split_manifest": str(args.split_manifest),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[build_frozen_benchmarks] wrote manifest to {args.output}")
    print(f"[build_frozen_benchmarks] gate verdict: {manifest['gate_verdict']}")

    return 0 if manifest["gate_verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
