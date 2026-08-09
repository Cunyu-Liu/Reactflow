#!/usr/bin/env python3
"""Sequence / parent / lineage leakage audit v1 (benchmark_v3 / Task 1C).

Computes, over the eligible exact pairs:
  * exact-sequence duplicate counts;
  * shared WT parent counts;
  * existing lineage / parent / family counts;
  * prespecified homology clusters at the audit's 70/70, 80/80, 90/90
    identity/coverage thresholds (reported together; sensitivity is monotonic
    in the threshold and we never pick the most favorable threshold afterward).

Writes data_registry/reactflow_delta/sequence_lineage_overlap_v1.tsv and
docs/reactflow_delta/split_policy_v3.md (homology thresholds are written into
the policy BEFORE any model results exist).

The core logic is pure and testable with an in-memory list of pair rows.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

NOT_RUN = "NOT_RUN"
UNKNOWN_NOT_ASSERTED = "UNKNOWN_NOT_ASSERTED"

# Prespecified identity/coverage thresholds (identity% AND coverage%).
HOMOLOGY_THRESHOLDS = (70, 80, 90)

SEQUENCE_KEYS = [
    "pair_id", "source_accession", "study_id", "publication_id_normalized",
    "sequence_sha256", "lineage_id", "parent_id", "rna_family",
    "exact_sequence_duplicate", "shared_wt_parent",
    "homology_flag_70", "homology_flag_80", "homology_flag_90",
]


def seq_identity_coverage(seq_a: str, seq_b: str):
    """Pairwise identity and coverage (ungapped, position-wise match).

    identity = matches / max length; coverage = min/max length.
    """
    if not seq_a or not seq_b:
        return 0.0, 0.0
    n = min(len(seq_a), len(seq_b))
    matches = sum(1 for i in range(n) if seq_a[i] == seq_b[i])
    m = max(len(seq_a), len(seq_b))
    return matches / m, n / m


def homology_components(sequences: Sequence[str], threshold: int) -> List[List[str]]:
    """Connected components of distinct sequences at (identity>=T and coverage>=T)."""
    nodes = list(sequences)
    adj: Dict[str, set] = {s: set() for s in nodes}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ident, cov = seq_identity_coverage(nodes[i], nodes[j])
            if ident >= threshold / 100.0 and cov >= threshold / 100.0:
                adj[nodes[i]].add(nodes[j])
                adj[nodes[j]].add(nodes[i])
    seen: set = set()
    components: List[List[str]] = []
    for node in nodes:
        if node in seen:
            continue
        comp = []
        stack = [node]
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(comp)
    return components


def flagged_pairs_at_threshold(pair_rows: List[dict], threshold: int) -> int:
    """Pairs whose sequence's homology component spans >1 distinct publication."""
    seq_to_publications: Dict[str, set] = defaultdict(set)
    for r in pair_rows:
        seq_to_publications[r["sequence"]].add(r["publication_id_normalized"])
    components = homology_components(list(seq_to_publications.keys()), threshold)
    flagged_seqs: set = set()
    for comp in components:
        pubs = set()
        for s in comp:
            pubs |= seq_to_publications[s]
        if len(pubs) > 1:
            flagged_seqs.update(comp)
    return sum(1 for r in pair_rows if r["sequence"] in flagged_seqs)


def compute_sequence_lineage_metrics(
    pair_rows: Iterable[dict],
    thresholds: Sequence[int] = HOMOLOGY_THRESHOLDS,
):
    """Compute sequence/lineage leakage metrics over eligible exact pairs.

    Each pair_row needs: pair_id, source_accession, sequence, lineage_id,
    parent_id, rna_family, publication_id_normalized.

    Returns (rows, summary).  rows carry per-pair leakage flags; summary
    carries exact-dup / shared-parent / lineage counts and homology sensitivity
    at every prespecified threshold (reported together, monotonic).
    """
    rows = list(pair_rows)
    total = len(rows)

    # exact-sequence duplicates
    seq_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        seq_counts[r["sequence"]] += 1
    exact_dup_sequences = {s for s, c in seq_counts.items() if c > 1}
    exact_dup_pairs = sum(1 for r in rows if r["sequence"] in exact_dup_sequences)

    # shared WT parent (design_group) across distinct publications
    parent_to_pubs: Dict[str, set] = defaultdict(set)
    for r in rows:
        parent_to_pubs[r["parent_id"]].add(r["publication_id_normalized"])
    shared_parent_pubs = {p for p, pubs in parent_to_pubs.items() if len(pubs) > 1}
    shared_parent_pairs = sum(1 for r in rows if r["parent_id"] in shared_parent_pubs)

    existing_lineage = len({r["lineage_id"] for r in rows})
    existing_parent = len({r["parent_id"] for r in rows})
    existing_family = len({r["rna_family"] for r in rows})

    homology = []
    for t in thresholds:
        flagged = flagged_pairs_at_threshold(rows, t)
        sensitivity = flagged / total if total else 0.0
        homology.append({
            "identity_coverage_threshold": t,
            "flagged_pairs": flagged,
            "sensitivity": round(sensitivity, 6),
        })

    summary = {
        "total_pairs": total,
        "exact_sequence_duplicate_sequences": len(exact_dup_sequences),
        "exact_sequence_duplicate_pairs": exact_dup_pairs,
        "shared_wt_parent_groups": len(shared_parent_pubs),
        "shared_wt_parent_pairs": shared_parent_pairs,
        "existing_lineage_count": existing_lineage,
        "existing_parent_count": existing_parent,
        "existing_family_count": existing_family,
        "homology": homology,
    }

    per_pair = []
    for r in rows:
        per_pair.append({
            "pair_id": r["pair_id"],
            "source_accession": r["source_accession"],
            "study_id": r["study_id"],
            "publication_id_normalized": r["publication_id_normalized"],
            "sequence_sha256": r.get("sequence_sha256") or NOT_RUN,
            "lineage_id": r.get("lineage_id") or NOT_RUN,
            "parent_id": r.get("parent_id") or NOT_RUN,
            "rna_family": r.get("rna_family") or UNKNOWN_NOT_ASSERTED,
            "exact_sequence_duplicate": int(r["sequence"] in exact_dup_sequences),
            "shared_wt_parent": int(r["parent_id"] in shared_parent_pubs),
            **{f"homology_flag_{t}": int(r["sequence"] in {
                s for s in _flagged_sequences(rows, t)})
               for t in thresholds},
        })
    return per_pair, summary


def _flagged_sequences(rows: List[dict], threshold: int) -> set:
    seq_to_publications: Dict[str, set] = defaultdict(set)
    for r in rows:
        seq_to_publications[r["sequence"]].add(r["publication_id_normalized"])
    flagged: set = set()
    for comp in homology_components(list(seq_to_publications.keys()), threshold):
        pubs = set()
        for s in comp:
            pubs |= seq_to_publications[s]
        if len(pubs) > 1:
            flagged.update(comp)
    return flagged


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def write_tsv(rows: List[dict], out_path: Path) -> None:
    def _fmt(v):
        return "" if v is None else str(v)
    with out_path.open("w") as fh:
        fh.write("\t".join(SEQUENCE_KEYS) + "\n")
        for r in rows:
            fh.write("\t".join(_fmt(r.get(k)) for k in SEQUENCE_KEYS) + "\n")


SPLIT_POLICY_TEMPLATE = """# Split Policy v3 — publication-disjoint + homology sensitivity

authority_scope: benchmark_v3
schema_version: "reactflow_delta.split_policy_v3.v1"
generated_by: "scripts/reactflow_delta/sequence_lineage_overlap_v1.py"
status: PRE-MODEL (homology thresholds fixed before any model results exist)

## 1. Publication-disjoint primary split
- The highest-level exchangeable unit is the **publication** (resolved PMID;
  falls back to DOI; otherwise UNRESOLVED_PUBLICATION:<study>).
- Development roles:
  - `16SFWJ` = DEVELOPMENT_CONSUMED / INVALID_FOR_CONFIRMATORY_USE
  - existing Phase-3 pool = DEVELOPMENT_USED
  - SL5CV2 / SL5HKU / SL5MER = merged into a single publication domain
    `pmid_38427602` (publication N = 1, NOT sufficient for confirmatory).
- New confirmatory candidates MUST come from unexposed, provenance-confirmed
  publications (never from an exposed/development publication domain).

## 2. Stricter publication + homology sensitivity split
- Homology thresholds are PRE-SPECIFIED and fixed (never tuned after seeing
  model results): {identity_coverage_thresholds} (identity% AND coverage%).
- Sensitivity is reported at every threshold jointly; we do NOT pick the most
  favorable threshold afterward.
- A pair is homology-flagged if its sequence falls in a connected component
  that spans more than one distinct publication at the given threshold.

## 3. Conservative leakage guard
- Confirmatory test pairs must be publication-disjoint AND not homology-flagged
  at the strictest prespecified threshold (identity>=90 AND coverage>=90)
  relative to any development pair.

## 4. Homology sensitivity snapshot (computed over eligible exact pairs)
{homology_snapshot}

## 5. Exact-sequence / parent / lineage counts
{lineage_counts}
"""


def write_split_policy_md(summary: dict, out_path: Path) -> None:
    thresh = "/".join(str(t) for t in HOMOLOGY_THRESHOLDS)
    if summary.get("homology"):
        snapshot = "\n".join(
            f"- identity/coverage {h['identity_coverage_threshold']}/"
            f"{h['identity_coverage_threshold']}: {h['flagged_pairs']}/{summary['total_pairs']} "
            f"flagged (sensitivity {h['sensitivity']})"
            for h in summary["homology"])
        lineage_counts = (
            f"- exact-sequence duplicate sequences: {summary['exact_sequence_duplicate_sequences']}\n"
            f"- exact-sequence duplicate pairs: {summary['exact_sequence_duplicate_pairs']}\n"
            f"- shared WT parent groups: {summary['shared_wt_parent_groups']}\n"
            f"- shared WT parent pairs: {summary['shared_wt_parent_pairs']}\n"
            f"- existing lineage count: {summary['existing_lineage_count']}\n"
            f"- existing parent count: {summary['existing_parent_count']}\n"
            f"- existing family count: {summary['existing_family_count']}")
    else:
        snapshot = f"- homology counts NOT_RUN (39GB canonical pass not yet produced)"
        lineage_counts = f"- lineage counts NOT_RUN"
    text = SPLIT_POLICY_TEMPLATE.format(
        identity_coverage_thresholds=thresh,
        homology_snapshot=snapshot,
        lineage_counts=lineage_counts,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")


# ---------------------------------------------------------------------------
# file extraction (heavy 39GB pass)
# ---------------------------------------------------------------------------
def extract_pair_sequence_lineage(
    primary_pairs_path: Path,
    canonical_path: Path,
    progress_every: int = 1_000_000,
):
    """Single streaming pass over canonical_records to build pair sequence+lineage.

    Returns (pair_rows_with_sequence, record_counts_by_accession).
    """
    # load pairs -> needed WT profile keys
    pairs = [json.loads(l) for l in primary_pairs_path.open() if l.strip()]
    # key: (source_accession, wt_profile_index)
    pair_by_wt: Dict[tuple, List[dict]] = defaultdict(list)
    for p in pairs:
        pair_by_wt[(p.get("source_accession"), p.get("wt_profile_index"))].append(p)

    record_counts: Dict[str, dict] = {}
    wt_info: Dict[tuple, dict] = {}
    nlines = 0
    with canonical_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            acc = rec.get("source_accession")
            if acc is None:
                continue
            entry = record_counts.setdefault(acc, {"n_records": 0, "n_profiles": 0, "_p": set()})
            entry["n_records"] += 1
            entry["_p"].add(rec.get("source_profile_index"))
            key = (acc, rec.get("source_profile_index"))
            if key in pair_by_wt and key not in wt_info:
                ple = rec.get("parent_lineage_evidence") or {}
                wt_info[key] = {
                    "sequence": rec.get("canonical_sequence") or "",
                    "lineage_id": ple.get("parent_sequence_sha256") or UNKNOWN_NOT_ASSERTED,
                    "parent_id": ple.get("design_group") or UNKNOWN_NOT_ASSERTED,
                }
            nlines += 1
            if nlines % progress_every == 0:
                print(f"  streamed {nlines} canonical records", flush=True)

    counts_out = {acc: {"n_records": e["n_records"], "n_profiles": len(e["_p"])}
                  for acc, e in record_counts.items()}

    out_rows = []
    for p in pairs:
        key = (p.get("source_accession"), p.get("wt_profile_index"))
        info = wt_info.get(key, {})
        out_rows.append({
            "pair_id": NOT_RUN,  # filled by join with publication registry
            "source_accession": p.get("source_accession"),
            "study_id": p.get("source_accession", "").split("_")[0],
            "sequence": info.get("sequence", ""),
            "lineage_id": info.get("lineage_id", NOT_RUN),
            "parent_id": info.get("parent_id", NOT_RUN),
            "rna_family": UNKNOWN_NOT_ASSERTED,
        })
    return out_rows, counts_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_registry_join(
    registry_path: Path,
) -> Dict[str, List[dict]]:
    """Load the pair publication registry v1 keyed by source_accession (in TSV order).

    Returns {source_accession: [registry_row, ...]} preserving file order so the
    per-accession order matches the pair-sequence cache (both derive from the same
    primary_pairs_v2.jsonl). Each registry_row carries pair_id and
    publication_id_normalized.
    """
    import csv
    by_acc: Dict[str, List[dict]] = defaultdict(list)
    with registry_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            acc = (row.get("source_accession") or "").strip()
            if not acc:
                continue
            by_acc[acc].append({
                "pair_id": row.get("pair_id") or NOT_RUN,
                "publication_id_normalized": (
                    row.get("publication_id_normalized") or UNKNOWN_NOT_ASSERTED),
            })
    return by_acc


def join_registry_into_cache(
    rows: List[dict], registry_by_acc: Dict[str, List[dict]],
) -> List[dict]:
    """Enrich pair-sequence cache rows with pair_id + publication by accession.

    Consumes registry rows per accession in order (cache and registry both derive
    from the same primary_pairs_v2.jsonl, so per-accession order matches). Rows
    with no matching registry entry keep NOT_RUN / UNKNOWN_NOT_ASSERTED.
    """
    cursor: Dict[str, int] = defaultdict(int)
    out = []
    for r in rows:
        acc = r.get("source_accession") or ""
        reg_rows = registry_by_acc.get(acc) or []
        idx = cursor[acc]
        cursor[acc] += 1
        if idx < len(reg_rows):
            r["pair_id"] = reg_rows[idx]["pair_id"]
            r["publication_id_normalized"] = reg_rows[idx]["publication_id_normalized"]
        else:
            r.setdefault("pair_id", NOT_RUN)
            r.setdefault("publication_id_normalized", UNKNOWN_NOT_ASSERTED)
        out.append(r)
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    worktree_root = here.parent.parent
    default_out = worktree_root / "data_registry" / "reactflow_delta"
    default_docs = worktree_root / "docs" / "reactflow_delta"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-pairs", type=Path, default=Path(
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/primary_pairs_v2.jsonl"))
    parser.add_argument("--canonical-records", type=Path, default=Path(
        "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/"
        "d1x_v2/d1x_v2_canonicalization_20260807T1830+0800/canonical_records_v2.jsonl"))
    parser.add_argument("--pair-sequence-cache", type=Path)
    parser.add_argument("--registry", type=Path, default=Path(
        "/home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/"
        "data_registry/reactflow_delta/pair_publication_registry_v1.tsv"))
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--docs-dir", type=Path, default=default_docs)
    parser.add_argument("--refresh", action="store_true",
                        help="Stream 39GB canonical to build the pair-sequence cache (background).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.pair_sequence_cache or (args.output_dir / "pair_sequence_lineage.jsonl")

    if args.refresh:
        print("streaming canonical_records_v2 (39GB) for pair sequence+lineage ...", flush=True)
        rows, counts = extract_pair_sequence_lineage(args.primary_pairs, args.canonical_records)
        with cache_path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        # count primary pairs per accession (small file)
        pair_counts: Dict[str, int] = {}
        for r in rows:
            acc = r["source_accession"]
            pair_counts[acc] = pair_counts.get(acc, 0) + 1

        # also write the shared asset count cache used by Task 1A (same format)
        (args.output_dir / "asset_counts_by_accession.json").write_text(
            json.dumps({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "canonical_counts": counts,
                "pair_counts": pair_counts,
            }) + "\n")
        print(f"wrote {len(rows)} pair-sequence rows to {cache_path}", flush=True)

    if cache_path.exists():
        rows = [json.loads(l) for l in cache_path.open() if l.strip()]
        for r in rows:
            r.setdefault("pair_id", NOT_RUN)
            r.setdefault("publication_id_normalized", UNKNOWN_NOT_ASSERTED)
        if args.registry and args.registry.exists():
            registry_by_acc = load_registry_join(args.registry)
            rows = join_registry_into_cache(rows, registry_by_acc)
        per_pair, summary = compute_sequence_lineage_metrics(rows)
    else:
        per_pair = []
        summary = {"total_pairs": 0, "homology": []}

    tsv_path = args.output_dir / "sequence_lineage_overlap_v1.tsv"
    md_path = args.docs_dir / "split_policy_v3.md"
    write_tsv(per_pair, tsv_path)
    write_split_policy_md(summary, md_path)

    print(json.dumps({"per_pair_rows": len(per_pair), "summary": summary,
                      "tsv": str(tsv_path), "md": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())