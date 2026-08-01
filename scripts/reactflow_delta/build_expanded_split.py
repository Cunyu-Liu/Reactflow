#!/usr/bin/env python3
"""Build an expanded split that includes annotation-only pairs.

Keeps the original 1509 true pairs and adds 6151 safe annotation-only pairs
(only exclusion = annotation_only_alt_not_verifiable, all safety criteria met).
Split is source-disjoint by rmdb_id (parent) to prevent leakage.
"""
import json
import argparse
import os
import random
from collections import defaultdict
from pathlib import Path


def _make_pair_id(entry):
    """Construct pair_id matching _pair_id_from_entry in evaluate.py.

    Format: "{rdat_basename}:{wt_idx}:{mut_idx}:{edit_pos_1idx}".
    """
    rdat_name = os.path.basename(entry["rdat_path"])
    mut = entry["matched_mutation"]
    return "{}:{}:{}:{}".format(
        rdat_name,
        entry["wt_profile_index"],
        entry["mutant_profile_index"],
        mut["encoded_position_1indexed"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True, help="d1_true_pair_registry.json")
    parser.add_argument("--output", type=Path, required=True, help="output split_members.json")
    parser.add_argument("--val-parents", type=int, default=5)
    parser.add_argument("--test-parents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.registry) as f:
        reg = json.load(f)

    items = reg["registry"]

    # True pairs (always include)
    true_pairs = [item for item in items if item.get("true_pair")]
    # Safe annotation-only pairs
    safe_anno = [
        item for item in items
        if not item.get("true_pair")
        and item.get("exclusion_reasons") == ["annotation_only_alt_not_verifiable"]
        and item.get("parent_lineage_verified") is True
        and item.get("has_wt_anchor") is True
        and item.get("comparable_fraction", 0) >= 0.6
        and item.get("normalization_domain_compatible") is True
        and item.get("condition_match_status") == "match"
        and item.get("in_vivo_in_vitro_mixed") is False
        and item.get("edit_count") == 1
        and item.get("edit_type") == "substitution"
    ]

    print(f"true_pairs: {len(true_pairs)}, safe_anno: {len(safe_anno)}")

    # Group all pairs by rmdb_id (parent)
    by_parent = defaultdict(list)
    for item in true_pairs + safe_anno:
        parent = item.get("rmdb_id", "unknown")
        pid = _make_pair_id(item)
        by_parent[parent].append((pid, item.get("true_pair", False)))

    # Sort parents by total pair count (descending) for stable allocation
    parents_sorted = sorted(by_parent.keys(), key=lambda p: -len(by_parent[p]))

    # Assign parents to splits: test first (smallest parents), then val, rest train
    random.seed(args.seed)
    random.shuffle(parents_sorted)

    test_parents = parents_sorted[:args.test_parents]
    val_parents = parents_sorted[args.test_parents:args.test_parents + args.val_parents]
    train_parents = parents_sorted[args.test_parents + args.val_parents:]

    train_pids = []
    val_pids = []
    test_pids = []
    for parent in train_parents:
        train_pids.extend(pid for pid, _ in by_parent[parent])
    for parent in val_parents:
        val_pids.extend(pid for pid, _ in by_parent[parent])
    for parent in test_parents:
        test_pids.extend(pid for pid, _ in by_parent[parent])

    split = {
        "train": {"pair_ids": sorted(train_pids), "parents": sorted(train_parents)},
        "validation": {"pair_ids": sorted(val_pids), "parents": sorted(val_parents)},
        "test": {"pair_ids": sorted(test_pids), "parents": sorted(test_parents)},
        "split_method": "source_disjoint_by_rmdb_id_expanded",
        "includes_annotation_only_pairs": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)

    print(f"train: {len(train_pids)} pairs, {len(train_parents)} parents")
    print(f"validation: {len(val_pids)} pairs, {len(val_parents)} parents")
    print(f"test: {len(test_pids)} pairs, {len(test_parents)} parents")
    print(f"Total: {len(train_pids) + len(val_pids) + len(test_pids)} pairs")


if __name__ == "__main__":
    main()
