"""Frozen-feature shard format and provenance for Stage-A warm-start.

Cycle C5 runs a 100M-parameter encoder (RibonanzaNet2 or eFold) *once, offline*
and freezes its per-nucleotide and pairwise representations to disk.  The pure
standard-library ``reactflow`` training core then reads those tensors as fixed
inputs (cycle C5.3).  This module defines the on-disk contract shared by the
offline exporter (``scripts/export_frozen_features.py``) and the reader; it
depends only on the standard library and :mod:`reactflow.npio`.

Shard layout
------------
A *shard* is a directory holding three files::

    <shard>/provenance.json   # model provenance + data-file content hash
    <shard>/features.npz       # numeric arrays, one group of members per record
    <shard>/index.jsonl        # one JSON object per record (order = write order)

Each record contributes NPZ members named ``"<row:06d>.<array>"`` where
``<row>`` is its zero-based line number in ``index.jsonl`` and ``<array>`` is one
of the declared array names.  Encoding the row rather than the raw record id
keeps member names ASCII and collision-free regardless of source identifiers.

Array contract
--------------
============  =========  ==================  ==========================
name          required   shape               meaning
============  =========  ==================  ==========================
single        yes        ``(L, d_single)``   per-nucleotide sequence rep
pair          no         ``(L, L, d_pair)``  pairwise representation
react_logits  no         ``(L, n_probe)``    reactivity head logits
============  =========  ==================  ==========================

``single`` is mandatory because the linear adapter of cycle C5.3 projects it
into the denoiser input; ``pair`` and ``react_logits`` are optional warm signals
that a shard may omit (the pairwise tensor is ``O(L^2 d)`` and is the first thing
dropped when disk is tight).

Integrity
---------
``provenance.json`` stores ``content_sha256`` -- the SHA-256 of
``features.npz`` concatenated with ``index.jsonl`` -- so a reader can detect a
truncated or swapped data file.  It also carries ``weights_sha256`` (hash of the
source model weights, computed by the exporter) so a run is traceable to the
exact checkpoint; the reader carries that value through without recomputing it,
since the weights are not present on the training host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from reactflow.npio import NdArray, load_npz, load_npz_member, load_npz_members, save_npz

ARRAY_SINGLE = "single"
ARRAY_PAIR = "pair"
ARRAY_REACT = "react_logits"
REQUIRED_ARRAYS: Tuple[str, ...] = (ARRAY_SINGLE,)
OPTIONAL_ARRAYS: Tuple[str, ...] = (ARRAY_PAIR, ARRAY_REACT)
KNOWN_ARRAYS: Tuple[str, ...] = REQUIRED_ARRAYS + OPTIONAL_ARRAYS

_PROVENANCE_NAME = "provenance.json"
_FEATURES_NAME = "features.npz"
_INDEX_NAME = "index.jsonl"


def default_schema(*, d_single: int, d_pair: Optional[int] = None, n_probe: Optional[int] = None) -> Dict[str, Dict[str, object]]:
    """Return a schema mapping describing the arrays a shard declares.

    The schema documents, per array, its NumPy dtype and the symbolic axis
    labels so a reader can validate shapes without hard-coding dimensions.  Only
    arrays whose dimension is supplied are included, so a features-only shard and
    a full shard produce different but self-describing schemas.

    Complexity: O(1).
    """

    schema: Dict[str, Dict[str, object]] = {
        ARRAY_SINGLE: {"dtype": "<f4", "axes": ["L", int(d_single)]},
    }
    if d_pair is not None:
        schema[ARRAY_PAIR] = {"dtype": "<f4", "axes": ["L", "L", int(d_pair)]}
    if n_probe is not None:
        schema[ARRAY_REACT] = {"dtype": "<f4", "axes": ["L", int(n_probe)]}
    return schema


@dataclass(frozen=True)
class FrozenFeatureProvenance:
    """Traceability header for one frozen-feature shard.

    Formula: provenance is a deterministic JSON object whose integrity field is
    ``sha256(features.npz || index.jsonl)``.  Complexity: O(1) metadata storage;
    content hashing is performed by shard read/write helpers.

    Attributes:
        model_name: e.g. ``"RibonanzaNet2"`` or ``"eFold"``.
        model_version: checkpoint tag, e.g. ``"alpha-v1"``.
        weights_sha256: SHA-256 of the source weight file(s); ``""`` when the
            exporter ran without real weights (a dry run is still auditable but
            must be labelled as such).
        produced_by: free-form producer string (script + host).
        date: ISO-8601 date the shard was produced.
        schema: per-array dtype/axis contract (see :func:`default_schema`).
        content_sha256: SHA-256 of ``features.npz`` + ``index.jsonl``; set by
            :func:`write_frozen_shard` and re-checked on read.
        record_count: number of records in the shard.
        notes: optional human notes, e.g. "dry-run: random weights".
    """

    model_name: str
    model_version: str
    weights_sha256: str
    produced_by: str
    date: str
    schema: Mapping[str, Mapping[str, object]]
    content_sha256: str = ""
    record_count: int = 0
    notes: str = ""

    def to_json_obj(self) -> Dict[str, object]:
        """Return a JSON-serializable dict with deterministic key order.

        Formula: ``schema`` is shallow-copied as ``{name: dict(spec)}`` so JSON
        serialization cannot mutate the source mapping.  Complexity: O(K), where
        K is the number of schema entries.
        """

        return {
            "content_sha256": self.content_sha256,
            "date": self.date,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "notes": self.notes,
            "produced_by": self.produced_by,
            "record_count": self.record_count,
            "schema": {name: dict(spec) for name, spec in self.schema.items()},
            "weights_sha256": self.weights_sha256,
        }

    @staticmethod
    def from_json_obj(obj: Mapping[str, object]) -> "FrozenFeatureProvenance":
        """Rebuild provenance from a parsed ``provenance.json`` object.

        Formula: required keys are validated before coercing scalar fields and
        copying ``schema``.  Complexity: O(K), where K is schema entry count.
        """

        required = ("model_name", "model_version", "weights_sha256", "produced_by", "date", "schema")
        missing = [key for key in required if key not in obj]
        if missing:
            raise ValueError(f"provenance is missing required keys: {missing}")
        raw_schema = obj["schema"]
        if not isinstance(raw_schema, Mapping):
            raise ValueError("provenance 'schema' must be a mapping")
        schema = {str(name): dict(spec) for name, spec in raw_schema.items()}  # type: ignore[union-attr]
        return FrozenFeatureProvenance(
            model_name=str(obj["model_name"]),
            model_version=str(obj["model_version"]),
            weights_sha256=str(obj["weights_sha256"]),
            produced_by=str(obj["produced_by"]),
            date=str(obj["date"]),
            schema=schema,
            content_sha256=str(obj.get("content_sha256", "")),
            record_count=int(obj.get("record_count", 0)),
            notes=str(obj.get("notes", "")),
        )


@dataclass
class FrozenFeatureRecord:
    """Per-sequence frozen representations plus identity metadata.

    Formula: mandatory ``single`` has shape ``(L, d_single)`` and optional arrays
    have leading dimension ``L`` (or ``L x L`` for pair features).  Complexity:
    O(1) metadata plus array storage.

    Attributes:
        record_id: source identifier (carried from the eFold/Ribonanza record).
        sequence: RNA sequence over A/C/G/U; ``len(sequence) == L``.
        arrays: mapping of array name to :class:`~reactflow.npio.NdArray`.
        family: optional Rfam family/clan label for split assignment.
    """

    record_id: str
    sequence: str
    arrays: Dict[str, NdArray]
    family: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the mandatory ``single`` array and its leading dimension."""

        for name in REQUIRED_ARRAYS:
            if name not in self.arrays:
                raise ValueError(f"frozen record {self.record_id!r} is missing required array {name!r}")
        length = len(self.sequence)
        single = self.arrays[ARRAY_SINGLE]
        if len(single.shape) != 2 or single.shape[0] != length:
            raise ValueError(
                f"'single' shape {single.shape} incompatible with sequence length {length}"
            )
        if ARRAY_PAIR in self.arrays:
            pair = self.arrays[ARRAY_PAIR]
            if len(pair.shape) != 3 or pair.shape[0] != length or pair.shape[1] != length:
                raise ValueError(f"'pair' shape {pair.shape} incompatible with length {length}")
        if ARRAY_REACT in self.arrays:
            react = self.arrays[ARRAY_REACT]
            if len(react.shape) != 2 or react.shape[0] != length:
                raise ValueError(f"'react_logits' shape {react.shape} incompatible with length {length}")

    @property
    def length(self) -> int:
        """Sequence length ``L``.

        Formula: ``L = len(sequence)``.  Complexity: O(1).
        """

        return len(self.sequence)

    @property
    def d_single(self) -> int:
        """Per-nucleotide representation dimension ``d_single``.

        Formula: ``d_single = single.shape[1]``.  Complexity: O(1).
        """

        return self.arrays[ARRAY_SINGLE].shape[1]

    def single(self) -> NdArray:
        """Return the per-nucleotide sequence representation array.

        Formula: returns mandatory array ``H in R^{L x d_single}``.  Complexity:
        O(1) reference lookup.
        """

        return self.arrays[ARRAY_SINGLE]

    def pair(self) -> Optional[NdArray]:
        """Return the pairwise representation array if present.

        Formula: optional pair tensor has shape ``(L, L, d_pair)``.  Complexity:
        O(1) reference lookup.
        """

        return self.arrays.get(ARRAY_PAIR)

    def react_logits(self) -> Optional[NdArray]:
        """Return the reactivity head logits array if present.

        Formula: optional logits tensor has shape ``(L, n_probe)``.  Complexity:
        O(1) reference lookup.
        """

        return self.arrays.get(ARRAY_REACT)


@dataclass
class FrozenShard:
    """A loaded shard: provenance plus its records in write order.

    Complexity: O(N) record references plus array storage held by each record.
    """

    provenance: FrozenFeatureProvenance
    records: List[FrozenFeatureRecord] = field(default_factory=list)

    def by_id(self) -> Dict[str, FrozenFeatureRecord]:
        """Return a mapping from record id to record.

        Raises if two records share an id, since downstream alignment keys on it.
        Complexity: O(N).
        """

        mapping: Dict[str, FrozenFeatureRecord] = {}
        for record in self.records:
            if record.record_id in mapping:
                raise ValueError(f"duplicate record id in shard: {record.record_id!r}")
            mapping[record.record_id] = record
        return mapping


def _member_name(row: int, array_name: str) -> str:
    """Return the NPZ member base name for a record row and array.

    Complexity: O(1).
    """

    return f"{row:06d}.{array_name}"


def _hash_bytes(*chunks: bytes) -> str:
    """Return the hex SHA-256 of the concatenation of ``chunks``.

    Complexity: O(total bytes).
    """

    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _verify_shard_content(directory: Path, provenance: FrozenFeatureProvenance) -> None:
    """Verify ``features.npz`` + ``index.jsonl`` against shard provenance.

    Complexity: O(total shard bytes).  Callers that repeatedly access the same
    shard should cache successful verification and skip this check on later
    lookups.
    """

    if not provenance.content_sha256:
        return
    features_path = directory / _FEATURES_NAME
    index_path = directory / _INDEX_NAME
    recomputed = _hash_bytes(features_path.read_bytes(), index_path.read_bytes())
    if recomputed != provenance.content_sha256:
        raise ValueError(
            "frozen shard content hash mismatch: "
            f"expected {provenance.content_sha256}, computed {recomputed}"
        )


def _validate_record_against_schema(record: FrozenFeatureRecord, schema: Mapping[str, Mapping[str, object]]) -> None:
    """Assert that a record's arrays are all declared in ``schema``.

    Complexity: O(number of arrays).
    """

    for name in record.arrays:
        if name not in KNOWN_ARRAYS:
            raise ValueError(f"unknown array name {name!r} in record {record.record_id!r}")
        if name not in schema:
            raise ValueError(
                f"record {record.record_id!r} carries array {name!r} not declared in schema"
            )


def write_frozen_shard(
    directory: Union[str, Path],
    records: Sequence[FrozenFeatureRecord],
    provenance: FrozenFeatureProvenance,
    *,
    compress: bool = False,
) -> FrozenFeatureProvenance:
    """Write ``records`` and ``provenance`` to a shard directory.

    The NPZ and JSONL data files are written first; their combined SHA-256 is
    then stored as ``content_sha256`` in ``provenance.json``.  The returned
    provenance is the finalized copy (with ``content_sha256`` and
    ``record_count`` filled in).

    Complexity: O(total array bytes).
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, NdArray] = {}
    index_lines: List[str] = []
    for row, record in enumerate(records):
        _validate_record_against_schema(record, provenance.schema)
        entry: Dict[str, object] = {
            "row": row,
            "record_id": record.record_id,
            "sequence": record.sequence,
            "length": record.length,
            "family": record.family,
            "arrays": {},
        }
        for name in sorted(record.arrays):
            nd = record.arrays[name]
            arrays[_member_name(row, name)] = nd
            entry["arrays"][name] = {"dtype": nd.descr, "shape": list(nd.shape)}  # type: ignore[index]
        index_lines.append(json.dumps(entry, sort_keys=True, ensure_ascii=False))

    features_path = directory / _FEATURES_NAME
    index_path = directory / _INDEX_NAME
    save_npz(features_path, arrays, compress=compress)
    index_text = "".join(line + "\n" for line in index_lines)
    index_path.write_text(index_text, encoding="utf-8")

    content_hash = _hash_bytes(features_path.read_bytes(), index_text.encode("utf-8"))
    finalized = FrozenFeatureProvenance(
        model_name=provenance.model_name,
        model_version=provenance.model_version,
        weights_sha256=provenance.weights_sha256,
        produced_by=provenance.produced_by,
        date=provenance.date,
        schema=provenance.schema,
        content_sha256=content_hash,
        record_count=len(records),
        notes=provenance.notes,
    )
    (directory / _PROVENANCE_NAME).write_text(
        json.dumps(finalized.to_json_obj(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return finalized


def read_frozen_shard(directory: Union[str, Path], *, verify: bool = True) -> FrozenShard:
    """Load a shard directory into a :class:`FrozenShard`.

    When ``verify`` is true the data-file content hash is recomputed and checked
    against ``provenance.json``; a mismatch raises :class:`ValueError` so a
    corrupted or tampered shard cannot silently enter training.  Each record's
    arrays are re-validated against the sequence length by
    :class:`FrozenFeatureRecord`.

    Complexity: O(total array bytes).
    """

    directory = Path(directory)
    provenance_obj = json.loads((directory / _PROVENANCE_NAME).read_text(encoding="utf-8"))
    provenance = FrozenFeatureProvenance.from_json_obj(provenance_obj)

    features_path = directory / _FEATURES_NAME
    index_path = directory / _INDEX_NAME
    index_bytes = index_path.read_bytes()

    if verify:
        _verify_shard_content(directory, provenance)

    members = load_npz(features_path)
    records: List[FrozenFeatureRecord] = []
    for line_no, raw_line in enumerate(index_bytes.decode("utf-8").splitlines()):
        if not raw_line.strip():
            continue
        entry = json.loads(raw_line)
        row = int(entry["row"])
        declared = entry.get("arrays", {})
        arrays: Dict[str, NdArray] = {}
        for name in declared:
            key = _member_name(row, name)
            if key not in members:
                raise ValueError(f"index references missing NPZ member {key!r}")
            arrays[name] = members[key]
        record = FrozenFeatureRecord(
            record_id=str(entry["record_id"]),
            sequence=str(entry["sequence"]),
            arrays=arrays,
            family=entry.get("family"),
        )
        _validate_record_against_schema(record, provenance.schema)
        records.append(record)

    if provenance.record_count and provenance.record_count != len(records):
        raise ValueError(
            f"record_count {provenance.record_count} disagrees with {len(records)} index lines"
        )
    return FrozenShard(provenance=provenance, records=records)


def read_frozen_single_array(
    directory: Union[str, Path],
    row: int,
    *,
    verify: bool = True,
) -> NdArray:
    """Read only one record's mandatory ``single`` array from a frozen shard.

    Full-scale sharded training commonly needs a single sequence at a time.  A
    complete :func:`read_frozen_shard` call materializes every record in the
    child shard, which is unnecessary when the training order jumps among
    shards.  This helper verifies the shard content when requested, then reads
    exactly member ``"<row:06d>.single"`` from ``features.npz``.

    Complexity: O(total shard bytes + L*d_single) on the first verified access
    to a shard, and O(L*d_single) when the caller has already verified that
    shard and passes ``verify=False``.
    """

    if row < 0:
        raise ValueError("row must be non-negative")
    directory = Path(directory)
    provenance_obj = json.loads((directory / _PROVENANCE_NAME).read_text(encoding="utf-8"))
    provenance = FrozenFeatureProvenance.from_json_obj(provenance_obj)
    if verify:
        _verify_shard_content(directory, provenance)
    return load_npz_member(directory / _FEATURES_NAME, _member_name(row, ARRAY_SINGLE))


def read_frozen_single_arrays(
    directory: Union[str, Path],
    rows: Sequence[int],
    *,
    verify: bool = True,
) -> Dict[int, NdArray]:
    """Read several mandatory ``single`` arrays from one frozen shard.

    Formula: for row set ``R = {r_1, ..., r_m}``, this loads NPZ members
    ``"<r_j:06d>.single"`` in one ZIP session and returns ``{r_j: H_j}``, where
    ``H_j in R^{L_j x d_single}``.  It is mathematically identical to calling
    :func:`read_frozen_single_array` once per row; the only change is that the
    ZIP central directory and optional content hash are paid once for the group.

    Complexity: O(total shard bytes + selected L*d_single) on verified access,
    and O(selected L*d_single + |R|) when ``verify=False``.
    """

    unique_rows = list(dict.fromkeys(int(row) for row in rows))
    for row in unique_rows:
        if row < 0:
            raise ValueError("rows must be non-negative")
    directory = Path(directory)
    provenance_obj = json.loads((directory / _PROVENANCE_NAME).read_text(encoding="utf-8"))
    provenance = FrozenFeatureProvenance.from_json_obj(provenance_obj)
    if verify:
        _verify_shard_content(directory, provenance)
    member_by_row = {row: _member_name(row, ARRAY_SINGLE) for row in unique_rows}
    arrays = load_npz_members(directory / _FEATURES_NAME, member_by_row.values())
    return {row: arrays[member_name] for row, member_name in member_by_row.items()}
