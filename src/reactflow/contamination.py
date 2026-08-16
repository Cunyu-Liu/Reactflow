"""Global contamination grouping for ReactFlow Phase C1-1.

This module implements union-find contamination groups that merge records by
multiple criteria so that no two records in different splits can leak
information to each other.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 253-263 (contamination
group criteria), 316-321 (Gate: zero cross-split overlap).

Merge criteria (any one triggers a union)
-----------------------------------------
1. **Exact sequence**: identical canonical (U-normalized) sequence.
2. **T/U normalization**: subsumed by criterion 1 (canonicalization handles it).
3. **Parent transcript windows**: records sharing the same ``parent_id`` (e.g.,
   windows of the same mRNA).
4. **MMseqs clusters**: records in the same sequence-identity cluster.
5. **Rfam family**: records sharing the same ``RFxxxxx`` family accession.
6. **Rfam clan**: records sharing the same ``CLxxxxx`` clan accession.
7. **PDB chain**: records derived from the same PDB chain (e.g., ``2N7X-2D``).
8. **Probing constructs**: same parent + same probe (replicate-level leakage).
9. **Structure similarity**: deferred to a future phase (requires external
   structure-alignment tooling); the hook is provided but is a no-op here.

The grouper is deterministic: running it twice on the same input produces
byte-identical group IDs (lexicographic root selection).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .data_registry import DataRecord, canonicalize_sequence


# ---------------------------------------------------------------------------
# UnionFind (deterministic, lexicographic root)
# ---------------------------------------------------------------------------

class UnionFind:
    """Deterministic union-find with lexicographic root selection.

    Two instances that consume the same ``union`` calls in any order produce
    identical ``find`` results, because the root is always the
    lexicographically smallest member of each component.

    Complexity: ``O(alpha(N))`` amortized per ``find``/``union`` where
    ``alpha`` is the inverse Ackermann function.
    """

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        """Return the canonical root of ``item``.

        Complexity: amortized ``O(alpha(N))``.
        """

        if item not in self.parent:
            self.parent[item] = item
            return item
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        """Merge the components containing ``a`` and ``b``.

        The lexicographically smaller root becomes the parent, so the root is
        stable across different insertion orders.

        Complexity: amortized ``O(alpha(N))``.
        """

        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a

    def union_many(self, items: Iterable[str]) -> None:
        """Merge all items in ``items`` into a single component.

        Complexity: ``O(k * alpha(N))`` for ``k`` items.
        """

        items = list(items)
        if len(items) < 2:
            for item in items:
                self.find(item)
            return
        root = self.find(items[0])
        for item in items[1:]:
            self.union(root, item)

    def components(self) -> Dict[str, List[str]]:
        """Return component members grouped by root.

        Complexity: ``O(N * alpha(N))``.
        """

        result: Dict[str, List[str]] = {}
        for item in list(self.parent):
            result.setdefault(self.find(item), []).append(item)
        return result

    def group_of(self, item: str) -> str:
        """Return the group ID (root) for ``item``.

        Complexity: amortized ``O(alpha(N))``.
        """

        return self.find(item)

    def num_groups(self) -> int:
        """Return the number of distinct components.

        Complexity: ``O(N * alpha(N))``.
        """

        return len(self.components())


# ---------------------------------------------------------------------------
# PDB chain extraction
# ---------------------------------------------------------------------------

_PDB_CHAIN_RE = re.compile(r"^(\d[A-Z0-9]{3})[-_]?([A-Za-z0-9]+)?")
"""Regex to extract a PDB ID and chain from a source_id like ``2N7X-2D``."""


def extract_pdb_chain(source_id: str) -> Optional[str]:
    """Extract a ``PDB_ID-chain`` key from a source identifier.

    Returns ``None`` if the source_id does not look like a PDB identifier.
    Example: ``"2N7X-2D"`` -> ``"2N7X-2D"``; ``"1EHZ_A"`` -> ``"1EHZ-A"``.

    Complexity: ``O(len(source_id))``.
    """

    if not source_id:
        return None
    m = _PDB_CHAIN_RE.match(source_id)
    if not m:
        return None
    pdb_id = m.group(1)
    chain = m.group(2) or ""
    if chain:
        return f"{pdb_id}-{chain}"
    return pdb_id


# ---------------------------------------------------------------------------
# ContaminationGrouper
# ---------------------------------------------------------------------------

@dataclass
class ContaminationMergeStats:
    """Statistics about how many merges each criterion triggered.

    Each counter is the number of ``union`` calls made by that criterion
    (not the number of distinct components merged — a union that hits an
    already-merged pair still counts as a call).
    """

    exact_sequence: int = 0
    parent_window: int = 0
    mmseqs_cluster: int = 0
    rfam_family: int = 0
    rfam_clan: int = 0
    pdb_chain: int = 0
    probing_construct: int = 0
    structure_similarity: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "exact_sequence": self.exact_sequence,
            "parent_window": self.parent_window,
            "mmseqs_cluster": self.mmseqs_cluster,
            "rfam_family": self.rfam_family,
            "rfam_clan": self.rfam_clan,
            "pdb_chain": self.pdb_chain,
            "probing_construct": self.probing_construct,
            "structure_similarity": self.structure_similarity,
        }


@dataclass
class ContaminationGrouper:
    """Build global contamination groups over a set of :class:`DataRecord`.

    Usage::

        grouper = ContaminationGrouper()
        grouper.add_records(records)
        grouper.merge_exact_sequences()
        grouper.merge_parent_windows()
        grouper.merge_mmseqs_clusters()
        grouper.merge_rfam_family()
        grouper.merge_rfam_clan()
        grouper.merge_pdb_chains()
        grouper.merge_probing_constructs()
        groups = grouper.groups()  # Dict[str, List[str]]

    All merge methods are idempotent and can be called in any order.

    Attributes:
        uf: The underlying :class:`UnionFind`.
        stats: Per-criterion merge counters.
        records: Mapping from ``record_id`` to :class:`DataRecord` (filled by
            :meth:`add_records`).
        _by_checksum: Mapping from sequence checksum to record IDs (filled by
            :meth:`merge_exact_sequences`).
        _by_parent: Mapping from ``parent_id`` to record IDs.
        _by_cluster: Mapping from MMseqs cluster ID to record IDs.
        _by_family: Mapping from Rfam family to record IDs.
        _by_clan: Mapping from Rfam clan to record IDs.
        _by_pdb_chain: Mapping from PDB chain key to record IDs.
        _by_construct: Mapping from ``(parent_id, probe)`` to record IDs.
    """

    uf: UnionFind = field(default_factory=UnionFind)
    stats: ContaminationMergeStats = field(default_factory=ContaminationMergeStats)
    records: Dict[str, DataRecord] = field(default_factory=dict)
    _by_checksum: Dict[str, List[str]] = field(default_factory=dict)
    _by_parent: Dict[str, List[str]] = field(default_factory=dict)
    _by_cluster: Dict[str, List[str]] = field(default_factory=dict)
    _by_family: Dict[str, List[str]] = field(default_factory=dict)
    _by_clan: Dict[str, List[str]] = field(default_factory=dict)
    _by_pdb_chain: Dict[str, List[str]] = field(default_factory=dict)
    _by_construct: Dict[str, List[str]] = field(default_factory=dict)

    # --- Population ---

    def add_record(self, record: DataRecord) -> None:
        """Register a single record.

        Complexity: ``O(1)`` amortized.
        """

        self.records[record.record_id] = record
        self.uf.find(record.record_id)  # ensure singleton

    def add_records(self, records: Iterable[DataRecord]) -> None:
        """Register many records.

        Complexity: ``O(N)`` amortized for ``N`` records.
        """

        for record in records:
            self.add_record(record)

    # --- Merge criteria ---

    def merge_exact_sequences(self) -> None:
        """Merge records sharing the same canonical sequence checksum.

        Complexity: ``O(N)``.
        """

        self._by_checksum.clear()
        for record in self.records.values():
            self._by_checksum.setdefault(record.checksum, []).append(record.record_id)
        for group in self._by_checksum.values():
            if len(group) > 1:
                self.uf.union_many(group)
                self.stats.exact_sequence += len(group) - 1

    def merge_parent_windows(self) -> None:
        """Merge records sharing the same ``parent_id``.

        Records without a ``parent_id`` are skipped (they are full-length and
        cannot leak via windowing).

        Complexity: ``O(N)``.
        """

        self._by_parent.clear()
        for record in self.records.values():
            if record.parent_id is None:
                continue
            self._by_parent.setdefault(record.parent_id, []).append(record.record_id)
        for group in self._by_parent.values():
            if len(group) > 1:
                self.uf.union_many(group)
                self.stats.parent_window += len(group) - 1

    def merge_mmseqs_clusters(self) -> None:
        """Merge records sharing the same MMseqs ``sequence_cluster``.

        Records without a ``sequence_cluster`` are skipped.

        Complexity: ``O(N)``.
        """

        self._by_cluster.clear()
        for record in self.records.values():
            if record.sequence_cluster is None:
                continue
            self._by_cluster.setdefault(record.sequence_cluster, []).append(record.record_id)
        for group in self._by_cluster.values():
            if len(group) > 1:
                self.uf.union_many(group)
                self.stats.mmseqs_cluster += len(group) - 1

    def merge_rfam_family(self) -> None:
        """Merge records sharing the same Rfam ``family``.

        Records without a ``family`` are skipped (they are treated as
        singletons to avoid merging all unannotated records together).

        Complexity: ``O(N)``.
        """

        self._by_family.clear()
        for record in self.records.values():
            if record.family is None:
                continue
            self._by_family.setdefault(record.family, []).append(record.record_id)
        for group in self._by_family.values():
            if len(group) > 1:
                self.uf.union_many(group)
                self.stats.rfam_family += len(group) - 1

    def merge_rfam_clan(self) -> None:
        """Merge records sharing the same Rfam ``clan``.

        Records without a ``clan`` are skipped.

        Complexity: ``O(N)``.
        """

        self._by_clan.clear()
        for record in self.records.values():
            if record.clan is None:
                continue
            self._by_clan.setdefault(record.clan, []).append(record.record_id)
        for group in self._by_clan.values():
            if len(group) > 1:
                self.uf.union_many(group)
                self.stats.rfam_clan += len(group) - 1

    def merge_pdb_chains(self) -> None:
        """Merge records derived from the same PDB chain.

        Uses :func:`extract_pdb_chain` to parse ``source_id``.  Records whose
        ``source_id`` does not match a PDB pattern are skipped.

        Complexity: ``O(N * L)`` where ``L`` is the max ``source_id`` length.
        """

        self._by_pdb_chain.clear()
        for record in self.records.values():
            chain_key = extract_pdb_chain(record.source_id)
            if chain_key is None:
                continue
            self._by_pdb_chain.setdefault(chain_key, []).append(record.record_id)
        for group in self._by_pdb_chain.values():
            if len(group) > 1:
                self.uf.union_many(group)
                self.stats.pdb_chain += len(group) - 1

    def merge_probing_constructs(self) -> None:
        """Merge records sharing the same ``(parent_id, probe)`` construct.

        This catches replicate-level leakage where the same construct was
        measured multiple times with the same probe.  Records without a
        ``parent_id`` are skipped.

        Complexity: ``O(N)``.
        """

        self._by_construct.clear()
        for record in self.records.values():
            if record.parent_id is None:
                continue
            key = (record.parent_id, record.probe)
            self._by_construct.setdefault(key, []).append(record.record_id)
        for group in self._by_construct.values():
            if len(group) > 1:
                self.uf.union_many(group)
                self.stats.probing_construct += len(group) - 1

    def merge_structure_similarity(self, *_args: Any, **_kwargs: Any) -> None:
        """Hook for structure-similarity merging.

        Currently a no-op: structure-similarity clustering requires external
        tooling (e.g., RNAforester or ARTS) and is deferred to a future phase.
        The method signature is kept for API stability.

        Complexity: ``O(1)``.
        """

        # Intentionally empty: see module docstring criterion 9.
        return

    def merge_all(self) -> None:
        """Run all merge criteria in the canonical order.

        Order: exact sequence -> parent windows -> MMseqs clusters ->
        Rfam family -> Rfam clan -> PDB chain -> probing constructs ->
        structure similarity (no-op).

        Complexity: ``O(N * alpha(N))``.
        """

        self.merge_exact_sequences()
        self.merge_parent_windows()
        self.merge_mmseqs_clusters()
        self.merge_rfam_family()
        self.merge_rfam_clan()
        self.merge_pdb_chains()
        self.merge_probing_constructs()
        self.merge_structure_similarity()

    # --- Queries ---

    def group_of(self, record_id: str) -> str:
        """Return the contamination group ID for ``record_id``.

        Complexity: amortized ``O(alpha(N))``.
        """

        return self.uf.group_of(record_id)

    def groups(self) -> Dict[str, List[str]]:
        """Return all contamination groups as ``{group_id: [record_id, ...]}``.

        Complexity: ``O(N * alpha(N))``.
        """

        return self.uf.components()

    def num_groups(self) -> int:
        """Return the number of distinct contamination groups.

        Complexity: ``O(N * alpha(N))``.
        """

        return self.uf.num_groups()

    def same_group(self, a: str, b: str) -> bool:
        """Return ``True`` if ``a`` and ``b`` are in the same group.

        Complexity: amortized ``O(alpha(N))``.
        """

        return self.group_of(a) == self.group_of(b)

    # --- Split validation ---

    def split_overlap(
        self,
        split_assignments: Mapping[str, Iterable[str]],
    ) -> Dict[str, Dict[str, List[str]]]:
        """Check for contamination-group overlap across splits.

        Args:
            split_assignments: Mapping from split name (e.g., ``"train"``) to
                an iterable of ``record_id`` strings in that split.

        Returns:
            A dict ``{split_name: {other_split: [shared_group_id, ...]}}``.
            An empty dict at the top level means zero overlap (Gate PASS).

        Complexity: ``O(N * alpha(N))``.
        """

        # Map group_id -> set of splits that contain it
        group_to_splits: Dict[str, Set[str]] = {}
        for split_name, record_ids in split_assignments.items():
            for rid in record_ids:
                if rid not in self.records:
                    continue
                gid = self.group_of(rid)
                group_to_splits.setdefault(gid, set()).add(split_name)

        overlaps: Dict[str, Dict[str, List[str]]] = {}
        for gid, splits in group_to_splits.items():
            if len(splits) < 2:
                continue
            for s1 in splits:
                for s2 in splits:
                    if s1 >= s2:
                        continue
                    overlaps.setdefault(s1, {}).setdefault(s2, []).append(gid)
                    overlaps.setdefault(s2, {}).setdefault(s1, []).append(gid)
        return overlaps

    # --- Serialization ---

    def to_jsonl(self, path: Path) -> int:
        """Write contamination groups to a JSONL file.

        Each line is ``{"group_id": str, "members": [str, ...], "size": int}``.
        Groups are sorted by ``group_id`` for deterministic output.

        Returns the number of groups written.

        Complexity: ``O(N log N)`` due to sorting.
        """

        groups = self.groups()
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for gid in sorted(groups):
                members = sorted(groups[gid])
                f.write(json.dumps({
                    "group_id": gid,
                    "members": members,
                    "size": len(members),
                }) + "\n")
                count += 1
        return count

    def stats_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the grouper state."""

        groups = self.groups()
        sizes = [len(v) for v in groups.values()]
        size_distribution: Dict[int, int] = {}
        for s in sizes:
            size_distribution[s] = size_distribution.get(s, 0) + 1
        return {
            "total_records": len(self.records),
            "total_groups": len(groups),
            "singleton_groups": size_distribution.get(1, 0),
            "multi_record_groups": sum(1 for s in sizes if s > 1),
            "largest_group_size": max(sizes) if sizes else 0,
            "size_distribution_top10": dict(sorted(size_distribution.items())[:10]),
            "merge_stats": self.stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# Annotate records with sequence_cluster / clan from external metadata
# ---------------------------------------------------------------------------

def annotate_records_from_split_manifest(
    records: Dict[str, DataRecord],
    split_manifest_path: Path,
) -> int:
    """Annotate records with ``sequence_cluster`` and ``clan`` from an existing
    split manifest.

    The split manifest (produced by ``reactflow.splits``) has assignments with
    ``record_id``, ``cluster``, and ``clan`` fields.  This function matches by
    ``record_id`` and mutates the records dict in place by replacing each
    :class:`DataRecord` with an annotated copy.

    Returns the number of records annotated.

    Complexity: ``O(N)``.
    """

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assignments = manifest.get("assignments", [])
    annotation_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for a in assignments:
        rid = a.get("record_id")
        if rid is None:
            continue
        annotation_map[rid] = (a.get("cluster"), a.get("clan"))

    # The split manifest uses source_id as record_id (e.g., "RF02271.fa.csv_1"),
    # NOT a hash.  We match by source_id first, then by the full record_id.
    count = 0
    for rid, record in list(records.items()):
        cluster, clan = annotation_map.get(record.source_id, (None, None))
        if cluster is None:
            # Try direct record_id match
            cluster, clan = annotation_map.get(rid, (None, None))
        if cluster is not None or clan is not None:
            # frozen dataclass -> replace
            from dataclasses import replace
            records[rid] = replace(
                record,
                sequence_cluster=cluster if cluster is not None else record.sequence_cluster,
                clan=clan if clan is not None else record.clan,
            )
            count += 1
    return count


def annotate_records_from_rfam_clan(
    records: Dict[str, DataRecord],
    family_to_clan: Mapping[str, str],
) -> int:
    """Annotate records with ``clan`` from a Rfam family->clan mapping.

    Returns the number of records annotated.

    Complexity: ``O(N)``.
    """

    from dataclasses import replace
    count = 0
    for rid, record in list(records.items()):
        if record.clan is not None:
            continue
        if record.family is None:
            continue
        clan = family_to_clan.get(record.family)
        if clan is None:
            continue
        records[rid] = replace(record, clan=clan)
        count += 1
    return count
