"""Build Rfam/MMseqs metadata for leakage-safe eFold cache splits.

The generated TSV is consumed by :func:`reactflow.splits.split_efold_cache_by_clan`.
It resolves each cache row to a split group derived from the Rfam clan/family and
sequence-identity cluster.  When a sequence cluster connects multiple Rfam
groups, all connected rows are assigned to the same split component so the
downstream split is both family-disjoint and cluster-disjoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.request import urlretrieve


RFAM_ACCESSION_RE = re.compile(r"(RF\d{5})")
RFAM_DATABASE_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files"
RFAM_CLAN_MEMBERSHIP_URL = f"{RFAM_DATABASE_BASE_URL}/clan_membership.txt.gz"
RFAM_DB_VERSION_URL = f"{RFAM_DATABASE_BASE_URL}/db_version.txt.gz"


@dataclass(frozen=True)
class CacheMetadataRecord:
    """A prepared eFold cache row reduced to the fields needed for splitting.

    Complexity: O(L) storage because the RNA sequence is retained for clustering.
    """

    record_id: str
    source_id: str
    sequence: str
    rfam_acc: Optional[str]


@dataclass(frozen=True)
class RfamMetadataRow:
    """One emitted metadata TSV row.

    Complexity: O(1) metadata storage plus identifier string lengths.
    """

    record_id: str
    clan: Optional[str]
    cluster: Optional[str]
    rfam_acc: Optional[str]
    source_id: str
    sequence_sha1: str
    rfam_group: Optional[str]
    rfam_clan_acc: Optional[str]


@dataclass(frozen=True)
class RfamMetadataSummary:
    """Provenance and counts for a generated Rfam/MMseqs metadata TSV.

    Complexity: O(C) storage for optional command tuple length C.
    """

    output_path: str
    manifest_path: str
    input_records: int
    metadata_records: int
    records_with_rfam_accession: int
    records_with_rfam_clan: int
    records_with_family_fallback: int
    records_without_rfam_accession: int
    unique_rfam_accessions: int
    cluster_method: str
    cluster_count: int
    split_group_count: int
    rfam_clan_membership_path: Optional[str]
    rfam_clan_membership_url: str
    rfam_db_version: Optional[str]
    mmseqs_command: Optional[Tuple[str, ...]]
    mmseqs_error: Optional[str]


class UnionFind:
    """Tiny deterministic union-find used to merge Rfam groups and clusters.

    Complexity: amortized near-O(1) ``find`` with path compression; storage O(N).
    """

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        """Return the canonical root for ``item``.

        Complexity: amortized inverse-Ackermann time.
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

        Complexity: amortized inverse-Ackermann time.
        """

        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        # Lexicographic parent choice keeps labels byte-stable.
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a

    def components(self) -> Dict[str, List[str]]:
        """Return component members grouped by root.

        Complexity: O(N alpha(N)) over tracked nodes.
        """

        result: Dict[str, List[str]] = {}
        for item in list(self.parent):
            result.setdefault(self.find(item), []).append(item)
        return result


def extract_rfam_accession(value: object) -> Optional[str]:
    """Extract an ``RFxxxxx`` accession from a cache/source identifier.

    Complexity: O(len(value)).
    """

    if value in (None, ""):
        return None
    match = RFAM_ACCESSION_RE.search(str(value))
    return match.group(1) if match else None


def sequence_sha1(sequence: str) -> str:
    """Return a stable SHA1 for an RNA sequence.

    Complexity: O(L) for sequence length L.
    """

    return hashlib.sha1(sequence.upper().encode("ascii", errors="ignore")).hexdigest()


def _open_text(path: Path):
    """Open plain text or gzip text by suffix."""

    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return Path(path).open(encoding="utf-8")


def read_rfam_clan_membership(path: Path) -> Dict[str, str]:
    """Read Rfam ``clan_membership.txt(.gz)`` into ``rfam_acc -> clan_acc``.

    The official dump has two tab-separated columns in this order:
    ``clan_acc`` and ``rfam_acc``.

    Complexity: O(R) over R membership rows.
    """

    mapping: Dict[str, str] = {}
    with _open_text(Path(path)) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if parts[0].lower() == "clan_acc":
                continue
            clan_acc, rfam_acc = parts[0].strip(), parts[1].strip()
            if not RFAM_ACCESSION_RE.fullmatch(rfam_acc):
                continue
            if not re.fullmatch(r"CL\d{5}", clan_acc):
                continue
            mapping[rfam_acc] = clan_acc
    return mapping


def _download(url: str, destination: Path) -> Path:
    """Download ``url`` unless ``destination`` already exists."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    urlretrieve(url, destination)
    return destination


def download_rfam_clan_membership(download_dir: Path) -> Path:
    """Download the small official Rfam clan-membership dump.

    Complexity: O(file bytes) when a download is needed, O(1) if cached.
    """

    return _download(RFAM_CLAN_MEMBERSHIP_URL, Path(download_dir) / "clan_membership.txt.gz")


def download_rfam_db_version(download_dir: Path) -> Optional[str]:
    """Download and read Rfam ``db_version.txt.gz`` when available.

    Complexity: O(file bytes) when a download is needed, O(1) if cached.
    """

    try:
        path = _download(RFAM_DB_VERSION_URL, Path(download_dir) / "db_version.txt.gz")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _cache_row_id(path: Path, line_number: int, row: Mapping[str, object], seen: Dict[str, int]) -> str:
    """Mirror ``reactflow.splits`` row-id materialization for metadata joins."""

    base = str(row.get("source_id") or row.get("record_id") or f"{Path(path).name}:{line_number}")
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}#{count + 1}"


def read_cache_metadata_records(cache_paths: Sequence[Path]) -> Tuple[CacheMetadataRecord, ...]:
    """Read prepared eFold cache JSONL rows needed for metadata construction.

    Complexity: O(NL) for N cache rows and average sequence length L.
    """

    records: List[CacheMetadataRecord] = []
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
                sequence = str(row.get("sequence") or "").upper()
                if not sequence:
                    raise ValueError(f"{cache_path}:{line_number} missing sequence")
                record_id = _cache_row_id(cache_path, line_number, row, seen_ids)
                source_id = str(row.get("source_id") or row.get("record_id") or record_id)
                rfam_acc = extract_rfam_accession(source_id) or extract_rfam_accession(record_id)
                records.append(
                    CacheMetadataRecord(
                        record_id=record_id,
                        source_id=source_id,
                        sequence=sequence,
                        rfam_acc=rfam_acc,
                    )
                )
    return tuple(records)


def _exact_clusters(records: Sequence[CacheMetadataRecord]) -> Dict[str, str]:
    """Cluster records by exact sequence identity."""

    return {record.record_id: f"exact:{sequence_sha1(record.sequence)[:16]}" for record in records}


def _ungapped_identity_and_coverage(sequence_a: str, sequence_b: str) -> Tuple[float, float]:
    """Return ungapped global identity and length coverage for two sequences.

    The identity is measured on the aligned prefix of length
    ``min(len(a), len(b))``:

        identity(a,b) = #{i < min_len : a_i = b_i} / min_len.

    The coverage term prevents a short sequence from clustering with an
    unrelated long sequence merely because its prefix matches:

        coverage(a,b) = min(len(a), len(b)) / max(len(a), len(b)).

    This is not a replacement for MMseqs2 local alignment.  It is a deterministic
    standard-library fallback for small sensitivity analyses and CI fixtures
    where installing MMseqs2 is impossible.

    Complexity: O(min(len(a), len(b))).
    """

    seq_a = sequence_a.upper()
    seq_b = sequence_b.upper()
    min_len = min(len(seq_a), len(seq_b))
    max_len = max(len(seq_a), len(seq_b))
    if min_len == 0 or max_len == 0:
        return 0.0, 0.0
    matches = sum(1 for index in range(min_len) if seq_a[index] == seq_b[index])
    return matches / float(min_len), min_len / float(max_len)


def _python_identity_clusters(
    records: Sequence[CacheMetadataRecord],
    *,
    min_seq_id: float,
    coverage: float,
    max_records: int,
) -> Dict[str, str]:
    """Cluster records by deterministic ungapped sequence identity.

    Two records are connected when both
    ``identity >= min_seq_id`` and ``coverage >= coverage`` hold.  Clusters are
    the connected components of that graph, implemented with
    :class:`UnionFind`.  The exhaustive pairwise pass is exact for this ungapped
    criterion, but it is quadratic, so a protective ``max_records`` guard keeps
    users from accidentally running it on the full 300k-window corpus.

    Complexity: O(N^2 * L) time and O(N) memory for N records and average
    compared prefix length L.
    """

    if max_records <= 0:
        raise ValueError("python_identity_max_records must be positive")
    if len(records) > max_records:
        raise ValueError(
            "python-identity clustering is quadratic and was asked to cluster "
            f"{len(records)} records (limit {max_records}); install MMseqs2 and "
            "use --cluster-method mmseqs for full-scale data"
        )
    union_find = UnionFind()
    for record in records:
        union_find.find(record.record_id)

    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            observed_coverage = min(len(left.sequence), len(right.sequence)) / float(max(len(left.sequence), len(right.sequence)))
            if observed_coverage < coverage:
                continue
            observed_identity, observed_coverage = _ungapped_identity_and_coverage(left.sequence, right.sequence)
            if observed_identity >= min_seq_id and observed_coverage >= coverage:
                union_find.union(left.record_id, right.record_id)

    return {
        record.record_id: f"python-identity:{union_find.find(record.record_id)}"
        for record in records
    }


def _write_fasta(records: Sequence[CacheMetadataRecord], path: Path) -> Dict[str, str]:
    """Write records to FASTA with safe synthetic identifiers."""

    id_by_record: Dict[str, str] = {}
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            seq_id = f"s{index}"
            id_by_record[record.record_id] = seq_id
            handle.write(f">{seq_id}\n{record.sequence}\n")
    return id_by_record


def _run_mmseqs_clusters(
    records: Sequence[CacheMetadataRecord],
    *,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    work_dir: Path,
) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """Run ``mmseqs easy-cluster`` and return ``record_id -> cluster``."""

    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "cache_sequences.fasta"
    prefix = work_dir / "mmseqs_cluster"
    tmp = work_dir / "mmseqs_tmp"
    id_by_record = _write_fasta(records, fasta)
    record_by_id = {seq_id: record_id for record_id, seq_id in id_by_record.items()}
    cmd = (
        mmseqs_bin,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(tmp),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    )
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    if not cluster_tsv.exists():
        raise RuntimeError(f"MMseqs did not write {cluster_tsv}")

    representative_by_member: Dict[str, str] = {}
    with cluster_tsv.open(encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split("\t")
            if len(parts) < 2:
                continue
            representative_by_member[parts[1]] = parts[0]

    clusters: Dict[str, str] = {}
    for record_id, seq_id in id_by_record.items():
        representative = representative_by_member.get(seq_id, seq_id)
        clusters[record_id] = f"mmseqs:{record_by_id.get(representative, representative)}"
    return clusters, cmd


def _called_process_error_tail(exc: subprocess.CalledProcessError, *, limit: int = 4000) -> str:
    """Return a compact stdout/stderr tail for failed external commands.

    Complexity: O(limit).
    """

    stdout = (exc.stdout or "")[-limit:]
    stderr = (exc.stderr or "")[-limit:]
    parts = [f"returncode={exc.returncode}"]
    if stdout:
        parts.append(f"stdout_tail={stdout!r}")
    if stderr:
        parts.append(f"stderr_tail={stderr!r}")
    return "; ".join(parts)


def resolve_sequence_clusters(
    records: Sequence[CacheMetadataRecord],
    *,
    method: str = "auto",
    mmseqs_bin: str = "mmseqs",
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    cov_mode: int = 1,
    threads: int = 1,
    work_dir: Optional[Path] = None,
    python_identity_max_records: int = 20000,
) -> Tuple[Dict[str, str], str, Optional[Tuple[str, ...]], Optional[str]]:
    """Resolve sequence clusters using MMseqs2 when requested/available.

    Formula: the returned map is ``record_id -> cluster_id``.  In strict
    ``mmseqs`` mode, cluster IDs come from MMseqs2 local-alignment clustering
    with threshold ``min_seq_id`` and coverage ``coverage``; in fallback modes
    they come from exact sequence hashes or deterministic Python identity
    components.  Complexity is dominated by MMseqs2 externally; exact fallback is
    O(NL) and Python identity fallback is O(N^2 L).
    """

    if method not in {"auto", "exact", "mmseqs", "python-identity"}:
        raise ValueError("method must be one of: auto, exact, mmseqs, python-identity")
    if method == "python-identity":
        return (
            _python_identity_clusters(
                records,
                min_seq_id=min_seq_id,
                coverage=coverage,
                max_records=python_identity_max_records,
            ),
            "python-identity",
            None,
            None,
        )
    mmseqs_path = shutil.which(mmseqs_bin) or mmseqs_bin
    use_mmseqs = method == "mmseqs" or (method == "auto" and shutil.which(mmseqs_bin) is not None)
    if not use_mmseqs:
        return _exact_clusters(records), "exact", None, None

    cleanup_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if work_dir is None:
            cleanup_dir = tempfile.TemporaryDirectory(prefix="reactflow_mmseqs_")
            resolved_work_dir = Path(cleanup_dir.name)
        else:
            resolved_work_dir = Path(work_dir)
        clusters, cmd = _run_mmseqs_clusters(
            records,
            mmseqs_bin=mmseqs_path,
            min_seq_id=min_seq_id,
            coverage=coverage,
            cov_mode=cov_mode,
            threads=threads,
            work_dir=resolved_work_dir,
        )
        return clusters, "mmseqs", cmd, None
    except subprocess.CalledProcessError as exc:
        error = _called_process_error_tail(exc)
        if method == "mmseqs":
            raise RuntimeError(f"MMseqs clustering failed: {error}") from exc
        return _exact_clusters(records), "exact", None, f"MMseqs fallback to exact clusters: {error}"
    except (OSError, RuntimeError) as exc:
        if method == "mmseqs":
            raise RuntimeError(f"MMseqs clustering failed: {exc}") from exc
        return _exact_clusters(records), "exact", None, f"MMseqs fallback to exact clusters: {exc}"
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()


def _component_label(members: Iterable[str]) -> str:
    """Create a stable split-group label for a union-find component."""

    member_list = sorted(members)
    groups = sorted(item[2:] for item in member_list if item.startswith("g:"))
    if len(groups) == 1:
        return groups[0]
    seed = "|".join(groups if groups else member_list)
    return "component:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def build_metadata_rows(
    records: Sequence[CacheMetadataRecord],
    *,
    rfam_to_clan: Mapping[str, str],
    clusters: Mapping[str, str],
) -> Tuple[RfamMetadataRow, ...]:
    """Build TSV rows, merging Rfam groups and sequence clusters into components.

    Complexity: O(N alpha(N)) over N records/components.
    """

    rfam_group_by_record: Dict[str, Optional[str]] = {}
    rfam_clan_by_record: Dict[str, Optional[str]] = {}
    union_find = UnionFind()

    for record in records:
        rfam_clan = rfam_to_clan.get(record.rfam_acc or "")
        rfam_group = rfam_clan or record.rfam_acc
        rfam_group_by_record[record.record_id] = rfam_group
        rfam_clan_by_record[record.record_id] = rfam_clan

        cluster = clusters.get(record.record_id)
        group_node = f"g:{rfam_group}" if rfam_group else f"g:unannotated:{record.record_id}"
        union_find.find(group_node)
        if cluster:
            cluster_node = f"c:{cluster}"
            union_find.union(group_node, cluster_node)

    label_by_root = {root: _component_label(members) for root, members in union_find.components().items()}

    rows: List[RfamMetadataRow] = []
    for record in records:
        rfam_group = rfam_group_by_record[record.record_id]
        cluster = clusters.get(record.record_id)
        group_node = f"g:{rfam_group}" if rfam_group else f"g:unannotated:{record.record_id}"
        clan = label_by_root[union_find.find(group_node)]
        rows.append(
            RfamMetadataRow(
                record_id=record.record_id,
                clan=clan,
                cluster=cluster,
                rfam_acc=record.rfam_acc,
                source_id=record.source_id,
                sequence_sha1=sequence_sha1(record.sequence),
                rfam_group=rfam_group,
                rfam_clan_acc=rfam_clan_by_record[record.record_id],
            )
        )
    return tuple(rows)


def write_metadata_tsv(rows: Sequence[RfamMetadataRow], path: Path) -> None:
    """Write metadata rows as a headered TSV accepted by the split CLI.

    Complexity: O(N) rows plus string output size.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "record_id",
        "clan",
        "cluster",
        "rfam_acc",
        "source_id",
        "sequence_sha1",
        "rfam_group",
        "rfam_clan_acc",
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            values = [getattr(row, field) or "" for field in fields]
            handle.write("\t".join(values) + "\n")


def build_rfam_metadata(
    cache_paths: Sequence[Path],
    output_tsv: Path,
    *,
    manifest_path: Optional[Path] = None,
    clan_membership_path: Optional[Path] = None,
    rfam_download_dir: Optional[Path] = None,
    download_rfam: bool = True,
    cluster_method: str = "auto",
    mmseqs_bin: str = "mmseqs",
    mmseqs_min_seq_id: float = 0.9,
    mmseqs_coverage: float = 0.8,
    mmseqs_cov_mode: int = 1,
    threads: int = 1,
    work_dir: Optional[Path] = None,
    python_identity_max_records: int = 20000,
) -> RfamMetadataSummary:
    """Generate metadata TSV and a JSON manifest for true Rfam/MMseqs splits.

    Complexity: O(NL + clustering) for N cache rows and average length L; the
    clustering term is external MMseqs2, O(NL) exact fallback, or O(N^2 L) Python
    identity fallback.
    """

    output_tsv = Path(output_tsv)
    manifest_path = Path(manifest_path) if manifest_path is not None else output_tsv.with_suffix(".manifest.json")
    download_dir = Path(rfam_download_dir) if rfam_download_dir is not None else output_tsv.parent / "rfam_database_files"

    resolved_clan_membership: Optional[Path] = Path(clan_membership_path) if clan_membership_path else None
    rfam_db_version: Optional[str] = None
    if resolved_clan_membership is None and download_rfam:
        resolved_clan_membership = download_rfam_clan_membership(download_dir)
        rfam_db_version = download_rfam_db_version(download_dir)
    elif download_rfam:
        rfam_db_version = download_rfam_db_version(download_dir)

    rfam_to_clan = read_rfam_clan_membership(resolved_clan_membership) if resolved_clan_membership else {}
    records = read_cache_metadata_records(tuple(Path(path) for path in cache_paths))
    clusters, resolved_cluster_method, mmseqs_command, mmseqs_error = resolve_sequence_clusters(
        records,
        method=cluster_method,
        mmseqs_bin=mmseqs_bin,
        min_seq_id=mmseqs_min_seq_id,
        coverage=mmseqs_coverage,
        cov_mode=mmseqs_cov_mode,
        threads=threads,
        work_dir=work_dir,
        python_identity_max_records=python_identity_max_records,
    )
    rows = build_metadata_rows(records, rfam_to_clan=rfam_to_clan, clusters=clusters)
    write_metadata_tsv(rows, output_tsv)

    records_with_rfam_accession = sum(1 for record in records if record.rfam_acc is not None)
    records_with_rfam_clan = sum(1 for record in records if record.rfam_acc in rfam_to_clan)
    records_with_family_fallback = sum(1 for record in records if record.rfam_acc is not None and record.rfam_acc not in rfam_to_clan)
    records_without_rfam_accession = len(records) - records_with_rfam_accession
    summary = RfamMetadataSummary(
        output_path=str(output_tsv),
        manifest_path=str(manifest_path),
        input_records=len(records),
        metadata_records=len(rows),
        records_with_rfam_accession=records_with_rfam_accession,
        records_with_rfam_clan=records_with_rfam_clan,
        records_with_family_fallback=records_with_family_fallback,
        records_without_rfam_accession=records_without_rfam_accession,
        unique_rfam_accessions=len({record.rfam_acc for record in records if record.rfam_acc is not None}),
        cluster_method=resolved_cluster_method,
        cluster_count=len(set(clusters.values())),
        split_group_count=len({row.clan for row in rows if row.clan is not None}),
        rfam_clan_membership_path=str(resolved_clan_membership) if resolved_clan_membership else None,
        rfam_clan_membership_url=RFAM_CLAN_MEMBERSHIP_URL,
        rfam_db_version=rfam_db_version,
        mmseqs_command=mmseqs_command,
        mmseqs_error=mmseqs_error,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
