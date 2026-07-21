"""Cross-family (Rfam clan) data splitting with automatic leakage validation.

Why clan-disjoint splitting
---------------------------
ReactFlow's headline claim is *cross-family generalization*.  A random train/test
split of RNA structures silently leaks information: near-homologous sequences from
the same family land in both partitions, so a model can score well by memorizing
family motifs rather than generalizing.  eFold (de Lajarte et al., *Sci Adv* 2026,
DOI ``10.1126/sciadv.adz4967``) quantifies exactly this "generalization gap".  To
measure it honestly we split at the granularity of an **Rfam clan** (a group of
related families) and guarantee that the clan sets of ``train`` / ``val`` /
``test`` / ``novel`` are pairwise disjoint.

Split roles
-----------
* ``train`` / ``val`` / ``test`` -- clan-disjoint partitions used for the standard
  in-distribution vs cross-clan comparison.
* ``novel`` -- clans held out *entirely* from every other split.  This is the
  strictest out-of-distribution probe and the one used to report the
  generalization gap ``F1(in-clan) - F1(novel-clan)``.

The split *unit* is the clan, never the individual sequence, so no clan can span
two splits by construction.  Sequence-identity de-duplication (MMseqs2 / CD-HIT)
is performed offline; this module only *consumes* a cluster label per record and
additionally verifies that no de-dup cluster straddles two splits, catching the
case where two splits contain near-identical sequences that were assigned
different clan labels upstream.

Determinism
-----------
Clan-to-split assignment uses a seeded ``random.Random`` (Mersenne Twister),
which is byte-for-byte reproducible across platforms for a given seed.  Given the
same records, seed, fractions and bucket boundaries, the manifest is identical on
Linux and Windows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test", "novel")


@dataclass(frozen=True)
class SplitRecord:
    """One record to be assigned to a split.

    Attributes:
        record_id: Unique identifier of the sequence/structure record.
        length: Sequence length ``L`` (used for length-bucket stratification).
        clan: Rfam clan label.  ``None`` records are treated as their own
            singleton clan ``singleton:<record_id>`` so they can never leak.
        cluster: Optional sequence-identity cluster label from offline MMseqs2 /
            CD-HIT de-duplication.  Used for the cross-split cluster check.

    Complexity: O(1) metadata storage.
    """

    record_id: str
    length: int
    clan: Optional[str] = None
    cluster: Optional[str] = None

    def effective_clan(self) -> str:
        """Return the clan used for splitting, materializing singletons.

        Complexity: O(1).
        """

        return self.clan if self.clan is not None else f"singleton:{self.record_id}"


@dataclass(frozen=True)
class SplitAssignment:
    """Resolved split assignment for one record.

    Complexity: O(1) metadata storage.
    """

    record_id: str
    clan: str
    cluster: Optional[str]
    length: int
    length_bucket: str
    split: str


@dataclass(frozen=True)
class SplitManifest:
    """Full manifest plus the parameters that produced it (for provenance).

    Complexity: O(N) assignment storage.
    """

    assignments: Tuple[SplitAssignment, ...]
    fractions: Mapping[str, float]
    novel_clan_fraction: float
    length_bucket_boundaries: Tuple[int, ...]
    seed: int

    def clans_by_split(self) -> Dict[str, set]:
        """Return the set of clans present in each split.

        Complexity: O(N) over assignments.
        """

        result: Dict[str, set] = {name: set() for name in SPLIT_NAMES}
        for assignment in self.assignments:
            result[assignment.split].add(assignment.clan)
        return result

    def counts_by_split(self) -> Dict[str, int]:
        """Return the record count in each split.

        Complexity: O(N).
        """

        counts = {name: 0 for name in SPLIT_NAMES}
        for assignment in self.assignments:
            counts[assignment.split] += 1
        return counts

    def counts_by_bucket(self) -> Dict[str, Dict[str, int]]:
        """Return per-split record counts broken down by length bucket.

        Complexity: O(N).
        """

        table: Dict[str, Dict[str, int]] = {name: {} for name in SPLIT_NAMES}
        for assignment in self.assignments:
            bucket_counts = table[assignment.split]
            bucket_counts[assignment.length_bucket] = bucket_counts.get(assignment.length_bucket, 0) + 1
        return table


@dataclass(frozen=True)
class SplitCacheSummary:
    """Summary for materialized cache files split by a leakage-safe manifest.

    Complexity: O(SB) storage for S splits and B length buckets.
    """

    output_dir: str
    manifest_path: str
    split_paths: Mapping[str, str]
    counts_by_split: Mapping[str, int]
    counts_by_bucket: Mapping[str, Mapping[str, int]]
    input_records: int
    metadata_records: int


def length_bucket_label(length: int, boundaries: Sequence[int]) -> str:
    """Map a sequence length to a human-readable bucket label.

    ``boundaries`` must be strictly increasing.  For boundaries ``(b_1, ..., b_m)``
    the buckets are the half-open intervals

        (-inf, b_1], (b_1, b_2], ..., (b_{m-1}, b_m], (b_m, +inf).

    The bucket index is the count of boundaries strictly below ``length``; this is
    a simple monotone step function, so longer sequences never map to an earlier
    bucket.

    Complexity: O(m) for ``m`` boundaries (m is tiny in practice).
    """

    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1]:
            raise ValueError("length_bucket_boundaries must be strictly increasing")
    exceeded = sum(1 for boundary in boundaries if length > boundary)
    if not boundaries:
        return "all"
    if exceeded == 0:
        return f"len_le_{boundaries[0]}"
    if exceeded == len(boundaries):
        return f"len_gt_{boundaries[-1]}"
    return f"len_{boundaries[exceeded - 1] + 1}_{boundaries[exceeded]}"


def _group_by_clan(records: Sequence[SplitRecord]) -> Dict[str, List[int]]:
    """Group record indices by effective clan.

    Complexity: O(N).
    """

    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.effective_clan(), []).append(index)
    return groups


def build_split_manifest(
    records: Sequence[SplitRecord],
    *,
    fractions: Optional[Mapping[str, float]] = None,
    novel_clan_fraction: float = 0.15,
    length_bucket_boundaries: Sequence[int] = (50, 200),
    seed: int = 0,
) -> SplitManifest:
    """Assign records to clan-disjoint ``train``/``val``/``test`` + ``novel`` splits.

    Algorithm
    ---------
    1. Group records by effective clan (``None`` -> per-record singleton).
    2. Order clans deterministically by ``(clan_id)`` then apply a seeded
       Fisher-Yates shuffle so the assignment does not depend on input order.
    3. **Novel-clan holdout**: walking the shuffled clan list, move whole clans
       into ``novel`` until the cumulative record count first reaches
       ``novel_clan_fraction * N``.  Whole clans keep the holdout family-disjoint.
    4. **train/val/test**: process the remaining clans largest-first (ties broken
       by clan id for determinism) and assign each whole clan to the split with
       the largest *record deficit* ``target_count - current_count``.  Assigning
       the biggest clans first, each to the currently most-underfilled split, is a
       standard greedy longest-processing-time heuristic that keeps whole clans
       together while tracking the requested record fractions closely.
    5. Emit one :class:`SplitAssignment` per record and validate leakage.

    ``fractions`` defaults to ``{"train": 0.8, "val": 0.1, "test": 0.1}`` and is
    normalized to sum to one over the non-novel record budget.

    Complexity: O(N + C log C) for ``N`` records and ``C`` clans (the ``log C`` is
    the clan sort); memory O(N + C).
    """

    if fractions is None:
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    if set(fractions) != {"train", "val", "test"}:
        raise ValueError("fractions must define exactly train/val/test")
    total_fraction = sum(fractions.values())
    if total_fraction <= 0:
        raise ValueError("fractions must sum to a positive value")
    if not 0.0 <= novel_clan_fraction < 1.0:
        raise ValueError("novel_clan_fraction must be in [0, 1)")
    normalized = {name: value / total_fraction for name, value in fractions.items()}

    total_records = len(records)
    if total_records == 0:
        return SplitManifest(
            assignments=tuple(),
            fractions=dict(normalized),
            novel_clan_fraction=novel_clan_fraction,
            length_bucket_boundaries=tuple(length_bucket_boundaries),
            seed=seed,
        )

    clan_groups = _group_by_clan(records)
    ordered_clans = sorted(clan_groups)
    rng = random.Random(seed)
    rng.shuffle(ordered_clans)

    clan_split: Dict[str, str] = {}

    # Step 3: novel-clan holdout by cumulative record count.
    novel_budget = novel_clan_fraction * total_records
    novel_count = 0
    remaining_clans: List[str] = []
    for clan in ordered_clans:
        if novel_count < novel_budget:
            clan_split[clan] = "novel"
            novel_count += len(clan_groups[clan])
        else:
            remaining_clans.append(clan)

    # Step 4: greedy longest-processing-time over train/val/test.
    non_novel_records = total_records - novel_count
    targets = {name: normalized[name] * non_novel_records for name in ("train", "val", "test")}
    current = {name: 0 for name in ("train", "val", "test")}
    remaining_clans.sort(key=lambda clan: (-len(clan_groups[clan]), clan))
    for clan in remaining_clans:
        deficits = [(targets[name] - current[name], name) for name in ("train", "val", "test")]
        # Largest deficit wins; ties broken by fixed split order for determinism.
        _, chosen = max(deficits, key=lambda item: (item[0], -("train", "val", "test").index(item[1])))
        clan_split[clan] = chosen
        current[chosen] += len(clan_groups[clan])

    assignments: List[SplitAssignment] = []
    for record in records:
        clan = record.effective_clan()
        assignments.append(
            SplitAssignment(
                record_id=record.record_id,
                clan=clan,
                cluster=record.cluster,
                length=record.length,
                length_bucket=length_bucket_label(record.length, length_bucket_boundaries),
                split=clan_split[clan],
            )
        )

    manifest = SplitManifest(
        assignments=tuple(assignments),
        fractions=dict(normalized),
        novel_clan_fraction=novel_clan_fraction,
        length_bucket_boundaries=tuple(length_bucket_boundaries),
        seed=seed,
    )
    validate_split_leakage(manifest)
    return manifest


def validate_split_leakage(manifest: SplitManifest) -> None:
    """Assert clan-disjointness and cluster-disjointness across splits.

    Guarantees enforced (raises :class:`ValueError` on any violation):

    * **Clan disjointness**: the clan set of ``train`` shares no clan with
      ``val``, ``test`` or ``novel``, and ``novel`` shares no clan with any other
      split.  This is the family-leakage guard the split exists for.
    * **Cluster disjointness**: no sequence-identity cluster label appears in more
      than one split, catching near-duplicate sequences that slipped across the
      clan boundary upstream.  Records without a cluster label are ignored here.

    Complexity: O(N + S^2 C) where ``S`` is the number of splits (4) and ``C`` the
    clan count -- dominated by the O(N) scan in practice.
    """

    clans = manifest.clans_by_split()
    for i in range(len(SPLIT_NAMES)):
        for j in range(i + 1, len(SPLIT_NAMES)):
            a, b = SPLIT_NAMES[i], SPLIT_NAMES[j]
            overlap = clans[a] & clans[b]
            if overlap:
                raise ValueError(
                    f"clan leakage between '{a}' and '{b}': {sorted(overlap)[:5]}"
                )

    cluster_split: Dict[str, str] = {}
    for assignment in manifest.assignments:
        if assignment.cluster is None:
            continue
        seen = cluster_split.get(assignment.cluster)
        if seen is None:
            cluster_split[assignment.cluster] = assignment.split
        elif seen != assignment.split:
            raise ValueError(
                f"cluster '{assignment.cluster}' spans splits '{seen}' and '{assignment.split}'"
            )


def manifest_to_json(manifest: SplitManifest, path: Path) -> None:
    """Write a split manifest to ``path`` as deterministic, sorted JSON.

    Records are sorted by ``record_id`` so the artifact is byte-stable and
    diff-friendly.  Complexity: O(N log N) for the sort.
    """

    payload = {
        "fractions": dict(manifest.fractions),
        "novel_clan_fraction": manifest.novel_clan_fraction,
        "length_bucket_boundaries": list(manifest.length_bucket_boundaries),
        "seed": manifest.seed,
        "assignments": [asdict(a) for a in sorted(manifest.assignments, key=lambda a: a.record_id)],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def manifest_from_json(path: Path) -> SplitManifest:
    """Load a split manifest previously written by :func:`manifest_to_json`.

    The loaded manifest is re-validated for leakage so a hand-edited or corrupted
    file cannot reintroduce family contamination.  Complexity: O(N).
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments = tuple(
        SplitAssignment(
            record_id=str(item["record_id"]),
            clan=str(item["clan"]),
            cluster=item["cluster"],
            length=int(item["length"]),
            length_bucket=str(item["length_bucket"]),
            split=str(item["split"]),
        )
        for item in payload["assignments"]
    )
    manifest = SplitManifest(
        assignments=assignments,
        fractions=dict(payload["fractions"]),
        novel_clan_fraction=float(payload["novel_clan_fraction"]),
        length_bucket_boundaries=tuple(int(b) for b in payload["length_bucket_boundaries"]),
        seed=int(payload["seed"]),
    )
    validate_split_leakage(manifest)
    return manifest


def read_split_metadata_tsv(path: Path) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Read optional record metadata used to assign Rfam clans/clusters.

    The accepted TSV schema is either headered with columns ``record_id``,
    ``clan`` and optional ``cluster``, or headerless with columns
    ``record_id<TAB>clan[<TAB>cluster]``.  Blank lines and ``#`` comments are
    ignored.  Empty clan/cluster cells become ``None``.

    Complexity: O(N) over metadata rows.
    """

    metadata: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    if not rows:
        return metadata

    header = [cell.strip().lower() for cell in rows[0]]
    has_header = "record_id" in header and ("clan" in header or "family" in header)
    if has_header:
        id_index = header.index("record_id")
        clan_index = header.index("clan") if "clan" in header else header.index("family")
        cluster_index = header.index("cluster") if "cluster" in header else None
        data_rows = rows[1:]
    else:
        id_index = 0
        clan_index = 1
        cluster_index = 2
        data_rows = rows

    for row in data_rows:
        if len(row) <= max(id_index, clan_index):
            raise ValueError("metadata TSV rows must include record_id and clan")
        record_id = row[id_index].strip()
        if not record_id:
            raise ValueError("metadata TSV record_id must be non-empty")
        clan = row[clan_index].strip() or None
        cluster = None
        if cluster_index is not None and len(row) > cluster_index:
            cluster = row[cluster_index].strip() or None
        metadata[record_id] = (clan, cluster)
    return metadata


def _cache_row_id(path: Path, line_number: int, row: Mapping[str, object], seen: Dict[str, int]) -> str:
    """Return a unique manifest id for one cache row."""

    base = str(row.get("source_id") or row.get("record_id") or f"{Path(path).name}:{line_number}")
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}#{count + 1}"


def split_efold_cache_by_clan(
    cache_paths: Sequence[Path],
    output_dir: Path,
    *,
    metadata_tsv: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    fractions: Optional[Mapping[str, float]] = None,
    novel_clan_fraction: float = 0.15,
    length_bucket_boundaries: Sequence[int] = (64, 128, 256),
    seed: int = 0,
) -> SplitCacheSummary:
    """Write train/val/test/novel JSONL caches from eFold cache rows.

    This is the user-facing bridge from ``prepare-efold-cache`` artifacts to a
    paper-grade cross-family protocol.  Each JSONL row becomes one
    :class:`SplitRecord`; its clan comes from, in order, metadata TSV override,
    cache ``family`` field, cache ``clan`` field, or singleton fallback.  The
    generated manifest is validated for clan and cluster leakage before any split
    cache is written.

    Complexity: O(N + C log C) for N cache rows and C clans.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path) if manifest_path is not None else output_dir / "split_manifest.json"
    metadata = read_split_metadata_tsv(Path(metadata_tsv)) if metadata_tsv is not None else {}

    rows: List[Tuple[str, str]] = []
    records: List[SplitRecord] = []
    seen_ids: Dict[str, int] = {}
    for cache_path in cache_paths:
        cache_path = Path(cache_path)
        with cache_path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"{cache_path}:{line_number} is not a JSON object")
                sequence = str(row.get("sequence") or "")
                if not sequence:
                    raise ValueError(f"{cache_path}:{line_number} missing sequence")
                row_id = _cache_row_id(cache_path, line_number, row, seen_ids)
                source_id = str(row.get("source_id") or row_id)
                clan, cluster = metadata.get(row_id, metadata.get(source_id, (None, None)))
                if clan is None:
                    raw_clan = row.get("family")
                    if raw_clan is None:
                        raw_clan = row.get("clan")
                    clan = str(raw_clan) if raw_clan not in (None, "") else None
                if cluster is None and row.get("cluster") not in (None, ""):
                    cluster = str(row.get("cluster"))
                rows.append((row_id, dict(row)))
                records.append(SplitRecord(record_id=row_id, length=len(sequence), clan=clan, cluster=cluster))

    manifest = build_split_manifest(
        records,
        fractions=fractions,
        novel_clan_fraction=novel_clan_fraction,
        length_bucket_boundaries=length_bucket_boundaries,
        seed=seed,
    )
    manifest_to_json(manifest, manifest_path)
    assignment_by_id = {assignment.record_id: assignment for assignment in manifest.assignments}

    split_paths = {name: output_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    handles = {name: split_paths[name].open("w", encoding="utf-8") for name in SPLIT_NAMES}
    try:
        for row_id, row in rows:
            assignment = assignment_by_id[row_id]
            enriched = dict(row)
            if assignment.clan is not None:
                enriched["clan"] = assignment.clan
            if assignment.cluster is not None:
                enriched["cluster"] = assignment.cluster
            handles[assignment.split].write(json.dumps(enriched, sort_keys=True, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    return SplitCacheSummary(
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        split_paths={name: str(path) for name, path in split_paths.items()},
        counts_by_split=manifest.counts_by_split(),
        counts_by_bucket=manifest.counts_by_bucket(),
        input_records=len(records),
        metadata_records=len(metadata),
    )
