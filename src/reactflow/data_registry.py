"""Unified RNA data registry for ReactFlow Phase C1-1.

This module provides the canonical ``DataRecord`` schema that unifies records
from all RNA data sources (eFold/Dryad, ArchiveII, PDB, viral, lncRNA,
human_mRNA, Rfam, MMseqs splits, Ribonanza, bpRNA/RNAStrAlign) into a single
format with comprehensive provenance, quality, and contamination metadata.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 217-238 (unified schema
fields), 240-251 (data source onboarding).

Design goals
------------
1. **Backward compatibility**: every existing cache JSONL row at
   ``artifacts/full_runs/full_ablation_*/cache/*.jsonl`` can be loaded
   losslessly via :meth:`DataRecord.from_cache_row`.
2. **Forward compatibility**: all spec-required fields are present, even if
   some are ``None`` for older data sources.
3. **Immutability**: ``DataRecord`` is a frozen dataclass so it can be safely
   shared across threads and used as a dict key.
4. **Deterministic IDs**: ``record_id`` is built from ``source:source_id``
   plus an optional window suffix, making records dedup-able across builds.
5. **Cheap checksums**: ``checksum`` is SHA-256 of the canonicalized sequence
   (U-normalized, uppercase, gaps stripped), matching the convention already
   used by ``reactflow.rfam_metadata.sequence_sha1``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_PAIRS = frozenset({"AU", "UA", "GC", "CG"})
"""Canonical Watson-Crick base pairs (chemistry-aware)."""

WOBBLE_PAIRS = frozenset({"GU", "UG"})
"""Wobble (non-canonical but geometrically allowed) base pairs."""

VALID_NUCLEOTIDES = frozenset("ACGU")
"""The four standard RNA nucleotides after T->U normalization."""

_PROBE_ALIASES: Dict[str, str] = {
    "2a3": "2A3",
    "dms": "DMS",
    "shape": "SHAPE",
    "cmct": "CMCT",
    "oh": "OH",
    "none": "none",
    "": "none",
    "shapecmct": "SHAPE+CMCT",
    "shape_cmct": "SHAPE+CMCT",
    "shape+cmct": "SHAPE+CMCT",
}
"""Case-insensitive alias map for probe names."""

_SOURCE_ALIASES: Dict[str, str] = {
    "efold_train": "efold_train",
    "efold": "efold_train",
    "pdb": "PDB",
    "archiveii": "ArchiveII",
    "archive_ii": "ArchiveII",
    "viral": "viral",
    "lncrna": "lncRNA",
    "human_mrna": "human_mRNA",
    "humanmrna": "human_mRNA",
    "ribonanza": "Ribonanza",
    "ribonanza2": "Ribonanza2",
    "bprna": "bpRNA",
    "rnastralign": "RNAStrAlign",
    "rfam": "Rfam",
}
"""Case-insensitive alias map for source names."""

_RFAM_ACCESSION_RE = re.compile(r"(RF\d{5})")
"""Regex to extract an ``RFxxxxx`` family accession from a free-form string."""


# ---------------------------------------------------------------------------
# Sequence / pair helpers
# ---------------------------------------------------------------------------

def canonicalize_sequence(sequence: str) -> str:
    """Normalize an RNA sequence.

    The canonical form is uppercase, T->U converted, gap characters (``.`` and
    ``-``) removed, and any non-``ACGU`` character replaced with ``N`` so that
    the length is preserved (important for reactivity-profile alignment).

    Complexity: ``O(L)`` where ``L = len(sequence)``.
    """

    s = sequence.upper().replace("T", "U")
    s = re.sub(r"[.-]", "", s)
    return "".join(c if c in VALID_NUCLEOTIDES else "N" for c in s)


def sequence_checksum(sequence: str) -> str:
    """SHA-256 hex digest of the canonical sequence.

    Complexity: ``O(L)``.
    """

    return hashlib.sha256(canonicalize_sequence(sequence).encode("ascii")).hexdigest()


def normalize_probe_name(probe: Any) -> str:
    """Normalize a probe name to canonical form.

    Complexity: ``O(1)``.
    """

    if probe is None:
        return "none"
    key = str(probe).strip().lower().replace(" ", "")
    return _PROBE_ALIASES.get(key, str(probe))


def normalize_source_name(source: Any) -> str:
    """Normalize a source name to canonical form.

    Complexity: ``O(1)``.
    """

    if source is None:
        return "unknown"
    key = str(source).strip().lower()
    return _SOURCE_ALIASES.get(key, str(source))


def classify_pair(nuc_i: str, nuc_j: str) -> str:
    """Classify a base pair by chemistry.

    Returns a 2-char string: ``"AU"``, ``"UA"``, ``"GC"``, ``"CG"`` (canonical),
    ``"GU"``, ``"UG"`` (wobble), or ``"XX"`` (non-canonical or contains ``N``).

    Complexity: ``O(1)``.
    """

    pair = (nuc_i.upper() + nuc_j.upper()).replace("T", "U")
    if pair in CANONICAL_PAIRS:
        return pair
    if pair in WOBBLE_PAIRS:
        return pair
    return "XX"


def detect_pseudoknots(
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    """Return the subset of pairs that participate in at least one crossing.

    A pair ``(i, j)`` crosses ``(k, l)`` (with ``i < j`` and ``k < l``) when
    ``i < k < j < l`` or ``k < i < l < j``.  Self-pairs and duplicates are
    ignored.

    Complexity: ``O(P^2)`` where ``P = len(pairs)``.  Acceptable for typical
    RNA structures (``P < 500``); for very long RNAs the caller should pre-filter.
    """

    # Normalize: drop self-pairs, swap (j, i) -> (i, j) when j < i, dedupe.
    seen: set = set()
    pair_list: List[Tuple[int, int]] = []
    for raw_i, raw_j in pairs:
        i, j = int(raw_i), int(raw_j)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) not in seen:
            seen.add((i, j))
            pair_list.append((i, j))
    crossing: set = set()
    for a_idx in range(len(pair_list)):
        i, j = pair_list[a_idx]
        for b_idx in range(a_idx + 1, len(pair_list)):
            k, l = pair_list[b_idx]
            if (i < k < j < l) or (k < i < l < j):
                crossing.add(a_idx)
                crossing.add(b_idx)
    return tuple(pair_list[idx] for idx in sorted(crossing))


def _normalize_pair(raw: Any, length: int) -> Optional[Tuple[int, int]]:
    """Normalize a raw pair entry to a 0-based ``(i, j)`` tuple with ``i < j``.

    Accepts ``[i, j]`` lists, ``(i, j)`` tuples, or ``{"i": i, "j": j}`` dicts.
    Returns ``None`` for self-pairs, out-of-range entries, or malformed input.

    Heuristic for 1-based vs 0-based: if either index is ``0``, the input is
    assumed 0-based; otherwise, if either index exceeds ``length``, the input
    is assumed 1-based and decremented.

    Complexity: ``O(1)``.
    """

    if raw is None:
        return None
    try:
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            i, j = int(raw[0]), int(raw[1])
        elif isinstance(raw, Mapping) and "i" in raw and "j" in raw:
            i, j = int(raw["i"]), int(raw["j"])
        else:
            return None
    except (ValueError, TypeError, KeyError):
        return None

    if i == 0 or j == 0:
        pass  # 0-based
    elif i > length or j > length:
        i -= 1
        j -= 1

    if i == j:
        return None
    if i < 0 or j < 0 or i >= length or j >= length:
        return None
    if i > j:
        i, j = j, i
    return (i, j)


# ---------------------------------------------------------------------------
# DataRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataRecord:
    """Unified RNA data record.

    All fields are required by the C1-1 spec (lines 217-238).  Optional fields
    default to ``None`` or empty tuples to accommodate heterogeneous sources.

    Attributes:
        record_id: Globally unique record identifier
            (``source:source_id[:w<window_index>]``).
        sequence: Canonical RNA sequence (U-normalized, uppercase, gaps
            stripped, non-ACGU replaced with ``N``).
        checksum: SHA-256 of the canonical sequence.
        source: Canonical source name (e.g., ``"efold_train"``, ``"PDB"``).
        source_version: Version string (e.g., cache build date ``"2026-07-09"``).
        source_id: Original identifier from the source database.
        parent_id: Parent RNA identifier (``None`` for full-length records).
        parent_coordinates: ``(start, end)`` 0-based half-open coordinates in
            the parent.
        parent_length: Full length of the parent RNA.
        window_index: Window index within the parent (0-based) for windowed
            records.
        pairs: Tuple of 0-based ``(i, j)`` base pairs with ``i < j``.
        pair_types: Tuple of pair-type strings parallel to :attr:`pairs`
            (``"AU"`` | ``"UA"`` | ``"GC"`` | ``"CG"`` | ``"GU"`` | ``"UG"`` |
            ``"XX"``).
        pseudoknot_pairs: Subset of :attr:`pairs` that participate in
            crossings.
        reactivity: Tuple of per-position reactivity values (``None`` if
            absent).
        reactivity_source: ``"real_profile"`` | ``"structure_forward_proxy"`` |
            ``"none"``.
        probe: ``"DMS"`` | ``"2A3"`` | ``"SHAPE"`` | ``"CMCT"`` | ``"OH"`` |
            ``"SHAPE+CMCT"`` | ``"none"``.
        replicate: Replicate identifier (``None`` if not applicable).
        experimental_condition: Free-text experimental condition.
        family: Rfam family accession (``RFxxxxx``) or ``None``.
        clan: Rfam clan accession (``CLxxxxx``) or ``None``.
        sequence_cluster: MMseqs/CD-HIT cluster identifier.
        structure_cluster: Structure-similarity cluster identifier.
        release_date: ISO 8601 release date (e.g., ``"2024-01-15"``) or
            ``None``.
        quality_flags: Tuple of flag strings (e.g., ``"windowed"``,
            ``"proxy_reactivity"``).
        length_bucket: Length-bucket label (e.g., ``"len_129_256"``).

    Complexity: ``O(L + P)`` storage for sequence length ``L`` and pair count
    ``P``.
    """

    record_id: str
    sequence: str
    checksum: str
    source: str
    source_version: str
    source_id: str
    parent_id: Optional[str] = None
    parent_coordinates: Optional[Tuple[int, int]] = None
    parent_length: Optional[int] = None
    window_index: Optional[int] = None
    pairs: Tuple[Tuple[int, int], ...] = ()
    pair_types: Tuple[str, ...] = ()
    pseudoknot_pairs: Tuple[Tuple[int, int], ...] = ()
    reactivity: Optional[Tuple[float, ...]] = None
    reactivity_source: str = "none"
    probe: str = "none"
    replicate: Optional[str] = None
    experimental_condition: Optional[str] = None
    family: Optional[str] = None
    clan: Optional[str] = None
    sequence_cluster: Optional[str] = None
    structure_cluster: Optional[str] = None
    release_date: Optional[str] = None
    quality_flags: Tuple[str, ...] = ()
    length_bucket: str = ""

    # --- Derived views ---

    def length(self) -> int:
        """Return the sequence length.

        Complexity: ``O(1)``.
        """

        return len(self.sequence)

    def parent_display_id(self) -> str:
        """Return a human-readable parent identifier.

        For full-length records, returns :attr:`source_id`.  For windows,
        returns :attr:`parent_id` (or :attr:`source_id` if ``parent_id`` is
        ``None``).

        Complexity: ``O(1)``.
        """

        if self.parent_id is not None:
            return self.parent_id
        return self.source_id

    def has_reactivity(self) -> bool:
        """Return ``True`` if this record has any reactivity profile.

        Complexity: ``O(1)``.
        """

        return (
            self.reactivity is not None
            and len(self.reactivity) > 0
            and self.reactivity_source != "none"
        )

    def has_real_profile(self) -> bool:
        """Return ``True`` if the reactivity profile is from a real experiment.

        Complexity: ``O(1)``.
        """

        return (
            self.reactivity is not None
            and len(self.reactivity) > 0
            and self.reactivity_source == "real_profile"
        )

    def has_pseudoknot(self) -> bool:
        """Return ``True`` if this record contains crossing (pseudoknot) pairs.

        Complexity: ``O(1)``.
        """

        return len(self.pseudoknot_pairs) > 0

    def canonical_pair_count(self) -> int:
        """Return the number of canonical (Watson-Crick) pairs.

        Complexity: ``O(P)``.
        """

        return sum(1 for pt in self.pair_types if pt in CANONICAL_PAIRS)

    def wobble_pair_count(self) -> int:
        """Return the number of wobble (GU/UG) pairs.

        Complexity: ``O(P)``.
        """

        return sum(1 for pt in self.pair_types if pt in WOBBLE_PAIRS)

    def noncanonical_pair_count(self) -> int:
        """Return the number of non-canonical (non-WC, non-wobble) pairs.

        Complexity: ``O(P)``.
        """

        return sum(1 for pt in self.pair_types if pt == "XX")

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Complexity: ``O(L + P)``.
        """

        d = asdict(self)
        d["pairs"] = [list(p) for p in self.pairs]
        d["pair_types"] = list(self.pair_types)
        d["pseudoknot_pairs"] = [list(p) for p in self.pseudoknot_pairs]
        d["quality_flags"] = list(self.quality_flags)
        if self.reactivity is not None:
            d["reactivity"] = list(self.reactivity)
        if self.parent_coordinates is not None:
            d["parent_coordinates"] = list(self.parent_coordinates)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DataRecord":
        """Deserialize from a dict.

        Complexity: ``O(L + P)``.
        """

        def _tuple_of_pairs(v: Any) -> Tuple[Tuple[int, int], ...]:
            if v is None:
                return ()
            return tuple(tuple(int(x) for x in p) for p in v)

        def _tuple_of_str(v: Any) -> Tuple[str, ...]:
            if v is None:
                return ()
            return tuple(str(x) for x in v)

        def _tuple_of_float(v: Any) -> Optional[Tuple[float, ...]]:
            if v is None:
                return None
            return tuple(float(x) for x in v)

        def _coords(v: Any) -> Optional[Tuple[int, int]]:
            if v is None:
                return None
            return (int(v[0]), int(v[1]))

        return cls(
            record_id=str(d["record_id"]),
            sequence=str(d["sequence"]),
            checksum=str(d["checksum"]),
            source=str(d["source"]),
            source_version=str(d.get("source_version", "")),
            source_id=str(d["source_id"]),
            parent_id=d.get("parent_id"),
            parent_coordinates=_coords(d.get("parent_coordinates")),
            parent_length=d.get("parent_length"),
            window_index=d.get("window_index"),
            pairs=_tuple_of_pairs(d.get("pairs", [])),
            pair_types=_tuple_of_str(d.get("pair_types", [])),
            pseudoknot_pairs=_tuple_of_pairs(d.get("pseudoknot_pairs", [])),
            reactivity=_tuple_of_float(d.get("reactivity")),
            reactivity_source=str(d.get("reactivity_source", "none")),
            probe=str(d.get("probe", "none")),
            replicate=d.get("replicate"),
            experimental_condition=d.get("experimental_condition"),
            family=d.get("family"),
            clan=d.get("clan"),
            sequence_cluster=d.get("sequence_cluster"),
            structure_cluster=d.get("structure_cluster"),
            release_date=d.get("release_date"),
            quality_flags=_tuple_of_str(d.get("quality_flags", [])),
            length_bucket=str(d.get("length_bucket", "")),
        )

    @classmethod
    def from_cache_row(
        cls,
        row: Mapping[str, Any],
        *,
        source: str,
        source_version: str = "",
        line_index: int = 0,
    ) -> "DataRecord":
        """Build a :class:`DataRecord` from an existing cache JSONL row.

        The existing cache schema (used by all files in
        ``artifacts/full_runs/full_ablation_*/cache/*.jsonl``) has the fields:
        ``family``, ``length_bucket``, ``pairs``, ``probe``, ``reactivity``,
        ``reactivity_source``, ``sequence``, ``source_id``, and optionally
        ``window`` (with ``start``/``end``/``index``/``parent_length``).

        Complexity: ``O(L + P^2)`` (the ``P^2`` is from pseudoknot detection).
        """

        sequence = canonicalize_sequence(str(row["sequence"]))
        length = len(sequence)
        source_id = str(row.get("source_id", ""))
        source_canonical = normalize_source_name(source)

        window = row.get("window")
        if window and isinstance(window, Mapping) and "index" in window:
            window_index = int(window["index"])
            record_id = f"{source_canonical}:{source_id}:w{window_index}"
            # For windowed records, the parent is the source_id prefix before
            # the coordinates (e.g., "ENSG00000004399.13" from
            # "ENSG00000004399.13:0-256").
            parent_id = source_id.split(":")[0] if ":" in source_id else source_id
            parent_start = int(window.get("start", 0))
            parent_end = int(window.get("end", length))
            parent_length_val = int(window.get("parent_length", parent_end))
            parent_coordinates = (parent_start, parent_end)
        else:
            window_index = None
            record_id = f"{source_canonical}:{source_id}"
            parent_id = None
            parent_coordinates = None
            parent_length_val = None

        raw_pairs = row.get("pairs", [])
        pairs_list: List[Tuple[int, int]] = []
        for raw in raw_pairs:
            p = _normalize_pair(raw, length)
            if p is not None:
                pairs_list.append(p)
        pairs_tuple = tuple(pairs_list)

        pair_types = tuple(
            classify_pair(sequence[i], sequence[j]) for i, j in pairs_tuple
        )
        pseudoknots = detect_pseudoknots(pairs_tuple)

        raw_reactivity = row.get("reactivity")
        if raw_reactivity is not None and len(raw_reactivity) > 0:
            # Coerce None -> 0.0 (missing measurements become 0 reactivity).
            reactivity = tuple(float(x) if x is not None else 0.0 for x in raw_reactivity)
        else:
            reactivity = None
        reactivity_source = str(row.get("reactivity_source", "none"))

        probe = normalize_probe_name(row.get("probe"))

        family = row.get("family")
        if family is not None:
            family = str(family) if str(family).strip() else None
        if family is None:
            m = _RFAM_ACCESSION_RE.search(source_id)
            if m:
                family = m.group(1)

        flags: List[str] = []
        if window is not None:
            flags.append("windowed")
        if reactivity_source == "structure_forward_proxy":
            flags.append("proxy_reactivity")
        if family is None:
            flags.append("no_family")
        if length == 0:
            flags.append("empty_sequence")
        if "N" in sequence:
            flags.append("contains_N")

        return cls(
            record_id=record_id,
            sequence=sequence,
            checksum=sequence_checksum(sequence),
            source=source_canonical,
            source_version=source_version,
            source_id=source_id,
            parent_id=parent_id,
            parent_coordinates=parent_coordinates,
            parent_length=parent_length_val,
            window_index=window_index,
            pairs=pairs_tuple,
            pair_types=pair_types,
            pseudoknot_pairs=pseudoknots,
            reactivity=reactivity,
            reactivity_source=reactivity_source,
            probe=probe,
            replicate=None,
            experimental_condition=None,
            family=family,
            clan=None,
            sequence_cluster=None,
            structure_cluster=None,
            release_date=None,
            quality_flags=tuple(flags),
            length_bucket=str(row.get("length_bucket", "")),
        )


# ---------------------------------------------------------------------------
# Streaming loaders
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream JSONL records from a file without loading everything into memory.

    Complexity: ``O(N)`` total, ``O(1)`` peak memory.
    """

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_cache_file(
    path: Path,
    *,
    source: str,
    source_version: str = "",
    limit: Optional[int] = None,
) -> Iterator[DataRecord]:
    """Load a cache JSONL file and yield :class:`DataRecord` instances.

    Complexity: ``O(N * (L + P^2))`` for ``N`` records.
    """

    count = 0
    for row in iter_jsonl(path):
        yield DataRecord.from_cache_row(
            row, source=source, source_version=source_version, line_index=count
        )
        count += 1
        if limit is not None and count >= limit:
            break


# ---------------------------------------------------------------------------
# Registry statistics
# ---------------------------------------------------------------------------

@dataclass
class RegistryStats:
    """Summary statistics accumulated while building a registry."""

    total_records: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    by_probe: Dict[str, int] = field(default_factory=dict)
    by_length_bucket: Dict[str, int] = field(default_factory=dict)
    with_reactivity: int = 0
    with_real_profile: int = 0
    with_pseudoknot: int = 0
    with_family: int = 0
    with_window: int = 0
    with_parent: int = 0
    total_pairs: int = 0
    total_canonical_pairs: int = 0
    total_wobble_pairs: int = 0
    total_noncanonical_pairs: int = 0
    total_pseudoknot_pairs: int = 0
    total_length: int = 0
    unique_checksums: int = 0
    duplicate_record_ids: int = 0

    def add(self, record: DataRecord) -> None:
        """Accumulate stats from a single record.

        Complexity: ``O(P)``.
        """

        self.total_records += 1
        self.by_source[record.source] = self.by_source.get(record.source, 0) + 1
        self.by_probe[record.probe] = self.by_probe.get(record.probe, 0) + 1
        self.by_length_bucket[record.length_bucket] = (
            self.by_length_bucket.get(record.length_bucket, 0) + 1
        )
        if record.has_reactivity():
            self.with_reactivity += 1
        if record.has_real_profile():
            self.with_real_profile += 1
        if record.has_pseudoknot():
            self.with_pseudoknot += 1
        if record.family is not None:
            self.with_family += 1
        if record.parent_id is not None:
            self.with_parent += 1
        if record.window_index is not None:
            self.with_window += 1
        self.total_pairs += len(record.pairs)
        self.total_canonical_pairs += record.canonical_pair_count()
        self.total_wobble_pairs += record.wobble_pair_count()
        self.total_noncanonical_pairs += record.noncanonical_pair_count()
        self.total_pseudoknot_pairs += len(record.pseudoknot_pairs)
        self.total_length += record.length()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        return asdict(self)


# ---------------------------------------------------------------------------
# Known data source registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSourceSpec:
    """Specification of a known data source.

    Attributes:
        name: Canonical source name (matches :attr:`DataRecord.source`).
        cache_filename: Default cache filename under ``cache/``.  Empty string
            if the source is not yet downloaded (the loader will skip it).
        description: Human-readable description.
        has_real_profiles: Whether the source contains real probing profiles
            (vs. structure-derived proxies).
        is_windowed: Whether records are typically windowed from longer parents.
        downloaded: Whether the source has been downloaded and is available
            in the cache directory.  When ``False``, ``cache_filename`` may
            still name the *intended* location, but the loader will skip it.
        upstream_url: URL of the upstream data source (for downloader scripts).
        upstream_license: License of the upstream data (if known).
    """

    name: str
    cache_filename: str
    description: str
    has_real_profiles: bool
    is_windowed: bool
    downloaded: bool = True
    upstream_url: Optional[str] = None
    upstream_license: Optional[str] = None


KNOWN_SOURCES: Tuple[DataSourceSpec, ...] = (
    # --- Cached sources (present in artifacts/full_runs/.../cache/) ---
    DataSourceSpec(
        name="efold_train",
        cache_filename="efold_train.jsonl",
        description="eFold/RNAndria Dryad training set (proxy reactivity).",
        has_real_profiles=False,
        is_windowed=False,
        downloaded=True,
        upstream_url="https://datadryad.org/stash/dataset/doi:10.5061/dryad.8kot6hj8",
        upstream_license="CC-BY-4.0 (Dryad)",
    ),
    DataSourceSpec(
        name="PDB",
        cache_filename="PDB.jsonl",
        description="PDB-derived RNA structures (proxy reactivity).",
        has_real_profiles=False,
        is_windowed=False,
        downloaded=True,
        upstream_url="https://www.rcsb.org/",
        upstream_license="PDB Data Usage Statement (public domain)",
    ),
    DataSourceSpec(
        name="ArchiveII",
        cache_filename="archiveII.jsonl",
        description="ArchiveII benchmark RNA structures (proxy reactivity).",
        has_real_profiles=False,
        is_windowed=False,
        downloaded=True,
        upstream_url="https://rna.urmc.rochester.edu/pub/archiveII/",
        upstream_license="Academic use only (RPI)",
    ),
    DataSourceSpec(
        name="viral",
        cache_filename="viral.jsonl",
        description="Viral RNA windows (mixed real and proxy profiles).",
        has_real_profiles=True,
        is_windowed=False,
        downloaded=True,
    ),
    DataSourceSpec(
        name="lncRNA",
        cache_filename="lncRNA.jsonl",
        description="Long non-coding RNA windows (proxy reactivity).",
        has_real_profiles=False,
        is_windowed=True,
        downloaded=True,
    ),
    DataSourceSpec(
        name="human_mRNA",
        cache_filename="human_mRNA.jsonl",
        description="Human mRNA 5'UTR/CDS/3'UTR windows (real DMS profiles).",
        has_real_profiles=True,
        is_windowed=True,
        downloaded=True,
    ),
    # --- Registered but not-yet-downloaded sources (spec lines 248-251) ---
    DataSourceSpec(
        name="Rfam",
        cache_filename="rfam.jsonl",
        description=(
            "Rfam family/clan annotations.  Used as metadata annotation on "
            "other sources (via rfam_metadata.py) rather than as a standalone "
            "sequence source.  Registered for completeness; not loaded as a "
            "standalone cache file."
        ),
        has_real_profiles=False,
        is_windowed=False,
        downloaded=False,
        upstream_url="https://rfam.org/",
        upstream_license="CC-BY-4.0 (Rfam)",
    ),
    DataSourceSpec(
        name="Ribonanza",
        cache_filename="ribonanza.jsonl",
        description=(
            "Ribonanza RNA mapping dataset (Kaggle 2023).  Chemical mapping "
            "data (DMS, 2A3, SHAPE) for ~2 million RNA sequences.  Metadata "
            "registered; raw data not yet downloaded."
        ),
        has_real_profiles=True,
        is_windowed=False,
        downloaded=False,
        upstream_url="https://www.kaggle.com/competitions/ribonanza-rna-folding",
        upstream_license="Kaggle competition data (research use)",
    ),
    DataSourceSpec(
        name="Ribonanza2",
        cache_filename="ribonanza2.jsonl",
        description=(
            "Ribonanza2 RNA mapping dataset (Kaggle 2024).  Extends "
            "Ribonanza with additional sequences and probing conditions.  "
            "Used by RibonanzaNet2 pretraining.  Metadata registered; raw "
            "data not yet downloaded."
        ),
        has_real_profiles=True,
        is_windowed=False,
        downloaded=False,
        upstream_url="https://www.kaggle.com/competitions/ribonanza-rna-folding",
        upstream_license="Kaggle competition data (research use)",
    ),
    DataSourceSpec(
        name="bpRNA",
        cache_filename="bpRNA.jsonl",
        description=(
            "bpRNA-1m: ~102,000 RNA sequences with secondary structure "
            "annotations from Rfam and bpRNA-1m.  Used by RibonanzaNet2 "
            "pretraining.  Run scripts/download_bprna_rnastralign.py to "
            "fetch and build the manifest."
        ),
        has_real_profiles=False,
        is_windowed=False,
        downloaded=False,
        upstream_url="https://bprna.cgrb.oregonstate.edu/",
        upstream_license="MIT (bpRNA code) / Rfam data license",
    ),
    DataSourceSpec(
        name="RNAStrAlign",
        cache_filename="rnastralign.jsonl",
        description=(
            "RNAStrAlign: ~30,000 RNA secondary structures from multiple "
            "databases (PDB, Rfam, bpRNA, etc.).  Used by RibonanzaNet2 "
            "pretraining.  Run scripts/download_bprna_rnastralign.py to "
            "fetch and build the manifest."
        ),
        has_real_profiles=False,
        is_windowed=False,
        downloaded=False,
        upstream_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6324060/",
        upstream_license="Public (research use)",
    ),
)
"""Specifications for all known RNA data sources (spec lines 240-251).

Sources with ``downloaded=False`` are registered for provenance and audit
purposes; the loader (``build_global_registry.py``) skips them when the
cache file does not exist.  Use ``scripts/download_bprna_rnastralign.py``
or future downloaders to populate the cache for these sources.
"""


def default_cache_dir() -> Path:
    """Return the default cache directory used by the full ablation runs."""

    return Path(
        "artifacts/full_runs/full_ablation_20260709_003012/cache"
    )


def default_split_dir() -> Path:
    """Return the default split directory used by the full ablation runs."""

    return Path(
        "artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0"
    )
