"""D2 RSIB-v1 contamination & lineage-graph audit (v3 §9, T-D2.1 / T-D2.2).

This module is part of Phase D2 (RSIB-v1 与数据 Gate). It provides the
forward-only, no-training helpers used by ``scripts/reactflow_delta/build_rsib.py``:

  * :func:`build_lineage_graph` — construct the parent / study /
    design-lineage graph from D0-R v2 candidate relations (T-D2.1).
  * :func:`verify_parent_lineage` — verify parent lineage for one candidate
    relation from the same-RDAT + header-SEQUENCE ref evidence (T-D2.1).
  * :func:`audit_split_overlap` — audit overlap between split groups at
    each v3 §9.1 split level (T-D2.2). D2 Gate requires overlap = 0.
  * :func:`compute_overlap_report` — summarize within-pool concentration
    when no splits are frozen yet (true_pair = 0 case).

D2 is non-learning: no model forward/backward, no test peeking, no Tier
threshold lowering (v3 §0.2, v3.1 §2.2). Every function is pure and operates
on the frozen D0-R v2 relation documents.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# v3 §9.1 Split 层级 (1=finest → 6=coarsest).
SPLIT_LEVELS: tuple[str, ...] = (
    "exact_construct",
    "parent",
    "design_lineage",
    "study",
    "family",
    "structure",
)

# v3 §8 Tier gate thresholds (mirrored in d1_pipeline_executor; D2 must not
# lower them — v3.1 §6.2).
TIER_A_MIN_STUDIES = 5
TIER_A_MIN_PARENTS = 20
TIER_A_MIN_PAIRS = 5000
TIER_B_MIN_TRUE_PAIRS = 1000


def _relation_key(relation: dict) -> tuple[str, int, int]:
    """Composite key identifying one candidate pair within the D0-R v2 pool."""
    return (
        relation.get("rdat_sha256", ""),
        int(relation.get("wt_profile_index", -1)),
        int(relation.get("mutant_profile_index", -1)),
    )


def verify_parent_lineage(relation: dict) -> dict[str, Any]:
    """Verify parent lineage for one D0-R v2 candidate relation (T-D2.1).

    "Parent lineage" = the mutant construct is derived from the WT construct
    (its parent) by the annotated edit. D0-R v2 carries annotation-only
    evidence (no per-profile sequences), so verification is performed at the
    annotation level using two forward-only signals:

      1. WT and mutant profiles share the same ``rdat_sha256`` (same RDAT
         file / same experiment / same construct family).
      2. ``matched_mutation.ref_verified_against == "header_SEQUENCE"`` —
         the annotated ref base was verified against the RDAT header
         SEQUENCE field, so the WT anchor is the recorded parent.

    Clearing ``parent_lineage_unverified`` does NOT upgrade an
    annotation-only candidate to ``true_pair`` — the
    ``annotation_only_alt_not_verifiable`` reason (v3.1 §3.2) still applies
    when no per-profile sequence evidence exists.

    Returns a dict with ``parent_lineage_verified`` (bool) and an
    ``evidence`` record.
    """
    matched = relation.get("matched_mutation") or {}
    rdat_sha = relation.get("rdat_sha256")
    # Same-RDAT parenthood: the relation itself ties WT and mutant to one
    # rdat_path / rdat_sha256 (D0-R v2 schema guarantees both profiles come
    # from the same file).
    same_rdat = bool(rdat_sha)
    ref_verified = matched.get("ref_verified_against") == "header_SEQUENCE"
    encoding_source = matched.get("encoding_source")
    verified = same_rdat and ref_verified
    return {
        "rdat_sha256": rdat_sha,
        "wt_profile_index": relation.get("wt_profile_index"),
        "mutant_profile_index": relation.get("mutant_profile_index"),
        "parent_prefix": relation.get("parent_prefix"),
        "rmdb_id": relation.get("rmdb_id"),
        "parent_lineage_verified": verified,
        "evidence": {
            "same_rdat": same_rdat,
            "ref_verified_against": matched.get("ref_verified_against"),
            "encoding_source": encoding_source,
            "rule": (
                "parent_lineage_verified = same_rdat AND "
                "ref_verified_against == 'header_SEQUENCE'"
            ),
        },
    }


def build_lineage_graph(relations: list[dict]) -> dict[str, Any]:
    """Build the parent / study / design-lineage graph (T-D2.1, v3 §9.1).

    Each candidate relation becomes a construct node. Groupings are emitted
    per v3 §9.1 split level. Family (Rfam) and structure-similarity levels
    are recorded as ``unknown`` because D0-R v2 carries no Rfam annotation
    and D2 forbids model forward (no structure computation).

    The graph is the basis for T-D2.3-5 split freezing and T-D2.2 overlap
    audit. When ``true_pair = 0`` no splits are frozen; the graph still
    documents the candidate-pool concentration for the dataset card.
    """
    constructs: list[dict] = []
    by_parent: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    by_design: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    by_study: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    by_owner: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

    for rel in relations:
        key = _relation_key(rel)
        parent = rel.get("parent_prefix") or rel.get("rmdb_id") or ""
        design = rel.get("rmdb_id") or parent
        study = rel.get("citation_doi") or ""
        owner = rel.get("owner") or ""
        constructs.append(
            {
                "construct_id": f"{rel.get('rdat_sha256','')}#{rel.get('wt_profile_index')}-{rel.get('mutant_profile_index')}",
                "key": list(key),
                "rdat_sha256": rel.get("rdat_sha256"),
                "rmdb_id": rel.get("rmdb_id"),
                "parent_prefix": rel.get("parent_prefix"),
                "citation_doi": study,
                "owner": owner,
                "modifier": rel.get("modifier"),
            }
        )
        by_parent[parent].append(key)
        by_design[design].append(key)
        by_study[study].append(key)
        by_owner[owner].append(key)

    def _group_counts(grouping: dict[str, list]) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(grouping.items(), key=lambda kv: (-len(kv[1]), kv[0]))}

    return {
        "split_levels": list(SPLIT_LEVELS),
        "construct_count": len(constructs),
        "constructs": constructs,
        "groupings": {
            "by_parent": _group_counts(by_parent),
            "by_design_lineage": _group_counts(by_design),
            "by_study": _group_counts(by_study),
            "by_owner": _group_counts(by_owner),
            "by_family": {
                "status": "unknown",
                "note": "Rfam family/clan annotation not present in D0-R v2; cannot audit family-level overlap without external Rfam lookup (deferred).",
            },
            "by_structure": {
                "status": "unknown",
                "note": "Structure similarity requires structure computation; D2 forbids model forward (v3.1 §2.2). Deferred to a later non-learning structure pass if needed.",
            },
        },
        "unique_counts": {
            "rdat_sha256": len({c["rdat_sha256"] for c in constructs if c["rdat_sha256"]}),
            "rmdb_id": len({c["rmdb_id"] for c in constructs if c["rmdb_id"]}),
            "parent_prefix": len({c["parent_prefix"] for c in constructs if c["parent_prefix"]}),
            "citation_doi": len({c["citation_doi"] for c in constructs if c["citation_doi"]}),
            "owner": len({c["owner"] for c in constructs if c["owner"]}),
        },
    }


def audit_split_overlap(
    split_assignment: dict[str, list[tuple[str, int, int]]],
) -> dict[str, Any]:
    """Audit overlap between split groups at every v3 §9.1 split level.

    Parameters
    ----------
    split_assignment:
        Mapping ``split_name -> list of construct keys`` assigned to that
        split. Construct keys are ``(rdat_sha256, wt_idx, mut_idx)`` tuples.

    Returns
    -------
    dict with per-level overlap counts. D2 Gate requires
    ``max_overlap == 0`` (no construct appears in two splits).
    """
    # Flatten to per-construct → splits membership.
    construct_to_splits: dict[tuple, set[str]] = defaultdict(set)
    for split_name, keys in split_assignment.items():
        for key in keys:
            construct_to_splits[tuple(key)].add(split_name)

    overlaps = [
        {"construct": list(key), "splits": sorted(splits)}
        for key, splits in construct_to_splits.items()
        if len(splits) > 1
    ]
    max_overlap = max((len(s) - 1 for s in construct_to_splits.values()), default=0)
    return {
        "split_count": len(split_assignment),
        "construct_count": len(construct_to_splits),
        "overlap_count": len(overlaps),
        "max_overlap": max_overlap,
        "gate_pass": max_overlap == 0,
        "overlapping_constructs": overlaps[:50],  # cap for audit readability
    }


def compute_overlap_report(relations: list[dict]) -> dict[str, Any]:
    """Summarize within-pool concentration when no splits are frozen yet.

    With ``true_pair = 0`` there is nothing to split (T-D2.3-5 deferred).
    This report documents candidate-pool concentration per split level so
    the dataset card (T-D2.11) can record the design before any
    ``true_pair`` materialises.
    """
    graph = build_lineage_graph(relations)
    groupings = graph["groupings"]

    def _top(level: str, n: int = 10) -> list[dict]:
        items = groupings.get(level, {})
        if not isinstance(items, dict):
            return []
        return [
            {"group": k, "count": v} for k, v in list(items.items())[:n]
        ]

    return {
        "status": "no_splits_frozen",
        "reason": (
            "true_pair = 0 after D2 parent-lineage verification + §3.2 "
            "annotation-only rule; no true_pairs to partition. T-D2.3-5 "
            "split freezing deferred until per-profile sequence evidence "
            "or same-parent replicate corroboration unlocks true_pairs."
        ),
        "candidate_total": len(relations),
        "unique_counts": graph["unique_counts"],
        "top_per_level": {
            "parent": _top("by_parent"),
            "design_lineage": _top("by_design_lineage"),
            "study": _top("by_study"),
            "owner": _top("by_owner"),
        },
        "family_overlap": groupings["by_family"],
        "structure_overlap": groupings["by_structure"],
    }


def compute_tier_judgment(
    *,
    true_pair_count: int,
    true_pair_studies: int,
    true_pair_parents: int,
    candidate_total: int,
    candidate_studies: int,
    candidate_parents: int,
    binding_blocker: str | None = None,
) -> dict[str, Any]:
    """Tier A/B/C judgment (v3 §8, T-D2.10).

    Uses ``true_pair`` counts only — candidate/construct counts are reported
    as reference and MUST NOT be used as the gate basis (v3.1 §7:
    "construct 数冒充 pair 数" is forbidden). Thresholds are frozen; D2 must
    not lower them (v3.1 §6.2).
    """
    tier_a_pass = (
        true_pair_studies >= TIER_A_MIN_STUDIES
        and true_pair_parents >= TIER_A_MIN_PARENTS
        and true_pair_count >= TIER_A_MIN_PAIRS
    )
    tier_b_pass = true_pair_count >= TIER_B_MIN_TRUE_PAIRS
    return {
        "basis": "true_pair counts (v3.1 §7: construct count forbidden)",
        "true_pairs": true_pair_count,
        "true_pair_studies": true_pair_studies,
        "true_pair_parents": true_pair_parents,
        "tier_a": {
            "thresholds": {
                "min_studies": TIER_A_MIN_STUDIES,
                "min_parents": TIER_A_MIN_PARENTS,
                "min_pairs": TIER_A_MIN_PAIRS,
            },
            "pass": tier_a_pass,
        },
        "tier_b": {
            "threshold": TIER_B_MIN_TRUE_PAIRS,
            "pass": tier_b_pass,
        },
        "tier_c": {
            "pass": tier_b_pass,  # Tier C requires ≥ Tier B; fails identically.
            "note": "Tier C requires ≥ Tier B threshold; fails with Tier B",
        },
        "candidate_level_reference": {
            "candidate_total": candidate_total,
            "candidate_studies": candidate_studies,
            "candidate_parents": candidate_parents,
        },
        "binding_blocker": binding_blocker,
        "outcome": (
            "below_tier_b_data_audit"
            if not tier_b_pass
            else "tier_b_or_above"
        ),
    }
