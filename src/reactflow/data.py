"""Data manifests and preprocessing utilities for Ribonanza-style profiles.

Mathematical contract
---------------------
For a sequence of length ``L`` and a chemical probe ``k`` we represent measured
reactivity as a vector

    r^(k) = (r_1, ..., r_L),  r_i in R or NaN.

Only positions with valid experimental signal should contribute to the loss.
This module turns raw values into a validated, normalized profile and a mask
``m_i``.  Missing values are preserved as masked positions rather than silently
imputed, because chemical mapping data contain primer regions and low-coverage
positions where a numeric imputation would fabricate signal.

The main operations are linear scans over sequence positions, so their time
complexity is O(L) and memory complexity is O(L).
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


VALID_BASES = frozenset({"A", "C", "G", "U"})
PROBES = frozenset({"DMS", "2A3"})


@dataclass(frozen=True)
class PublicDatasetSource:
    """Verifiable public source descriptor.

    The URLs are meant to be auditable by humans and scripts.  They are not
    downloaded implicitly by the library because Kaggle datasets generally
    require authenticated API tokens and explicit user consent.

    Complexity: O(1) metadata per source descriptor.
    """

    name: str
    url: str
    license: str
    description: str
    expected_schema: Tuple[str, ...]


PUBLIC_DATASETS: Tuple[PublicDatasetSource, ...] = (
    PublicDatasetSource(
        name="Ribonanza2 Training Data",
        url="https://www.kaggle.com/datasets/rhijudas/ribonanza2-training-data",
        license="CC BY 4.0",
        description=(
            "64M RNA chemical probing profiles across DMS and 2A3 H5 files; "
            "official data card reports reads, SNR, reactivity, error, norm, heatmap datasets."
        ),
        expected_schema=("reads", "SNR", "reactivity", "error", "norm", "heatmap"),
    ),
    PublicDatasetSource(
        name="Stanford Ribonanza RNA Folding",
        url="https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding",
        license="competition terms",
        description=(
            "Original Ribonanza competition CSV profiles with sequence, experiment_type, "
            "reads, signal_to_noise, SN_filter, reactivity_* and reactivity_error_* columns."
        ),
        expected_schema=(
            "sequence_id",
            "sequence",
            "experiment_type",
            "reads",
            "signal_to_noise",
            "SN_filter",
        ),
    ),
    PublicDatasetSource(
        name="RNAndria / eFold Dryad Dataset",
        url="https://doi.org/10.5061/dryad.79cnp5j95",
        license="Dryad public dataset; see dataset page",
        description=(
            "Dryad release for eFold/RNAndria cross-family RNA secondary-structure "
            "prediction, including efold_train, ArchiveII, PDB, viral, lncRNA, "
            "pri_miRNA and human mRNA JSON files."
        ),
        expected_schema=("sequence", "structure", "shape"),
    ),
    PublicDatasetSource(
        name="RibonanzaNet2 Kaggle Model",
        url="https://www.kaggle.com/models/shujun717/ribonanzanet2/PyTorch/alpha/1",
        license="MIT",
        description=(
            "Kaggle alpha checkpoint used only as a frozen encoder source; model "
            "card reports roughly 100M parameters and DMS/2A3 pretraining on "
            "RNA 100mer chemical-mapping profiles."
        ),
        expected_schema=("Network.py", "pairwise.yaml", "pytorch_model_fsdp.bin"),
    ),
)


@dataclass(frozen=True)
class ProfileValidationReport:
    """Result of profile-level quality control.

    Complexity: O(1) summary storage plus O(M) validation messages.
    """

    sequence_length: int
    valid_positions: int
    missing_positions: int
    negative_positions: int
    high_outlier_positions: int
    snr_pass: bool
    reads_pass: bool
    messages: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether basic completeness and validity gates are satisfied.

        Complexity: O(1).
        """

        return self.valid_positions > 0 and self.snr_pass and self.reads_pass and not self.messages


@dataclass(frozen=True)
class RibonanzaProfile:
    """One sequence/probe chemical mapping profile.

    Attributes:
        sequence: RNA sequence over A/C/G/U.
        probe: ``DMS`` or ``2A3``.
        reactivity: Position-wise signal, length must match the sequence.
        error: Optional position-wise uncertainty, length must match the sequence.
        reads: Total reads assigned to the profile.
        snr: Signal-to-noise ratio.

    Missing values use ``math.nan``.  Downstream masks are derived from finite
    values plus probe/base validity.

    Complexity: O(L) storage for a sequence/profile of length L.
    """

    sequence: str
    probe: str
    reactivity: Tuple[float, ...]
    error: Optional[Tuple[float, ...]] = None
    reads: Optional[float] = None
    snr: Optional[float] = None
    sequence_id: Optional[str] = None


def _is_finite(value: float) -> bool:
    """Return whether a value is a usable finite scalar.

    This is the atomic completeness predicate used by masks and normalization.
    Complexity: O(1).
    """

    return value is not None and math.isfinite(float(value))


def coerce_float(value: object) -> float:
    """Convert CSV/H5 scalar values into floats, preserving blanks as NaN.

    Complexity: O(1) time and memory.
    """

    if value is None:
        return math.nan
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return math.nan
    return float(text)


def probe_base_mask(sequence: str, probe: str) -> Tuple[bool, ...]:
    """Return chemically meaningful positions for a probe.

    DMS predominantly reports accessibility of A/C Watson-Crick faces, while
    2A3/SHAPE-like reagents report backbone flexibility for all four bases.

    Complexity: O(L) time and O(L) memory for sequence length L.
    """

    probe = normalize_probe_name(probe)
    if probe == "DMS":
        return tuple(base in {"A", "C"} for base in sequence)
    return tuple(base in VALID_BASES for base in sequence)


def normalize_probe_name(probe: str) -> str:
    """Normalize probe names from Ribonanza CSV/H5 conventions.

    Formula: ``DMS_MaP -> DMS`` and ``SHAPE -> 2A3``; the normalized symbol must
    belong to the supported probe set.  Complexity: O(len(probe)).
    """

    normalized = probe.strip().upper().replace("_MAP", "")
    if normalized == "SHAPE":
        normalized = "2A3"
    if normalized not in PROBES:
        raise ValueError(f"Unsupported probe '{probe}'. Expected one of {sorted(PROBES)}.")
    return normalized


def validate_profile(
    profile: RibonanzaProfile,
    *,
    min_reads: float = 100.0,
    min_snr: float = 1.0,
    outlier_z: float = 8.0,
) -> ProfileValidationReport:
    """Validate sequence/reactivity completeness and experimental plausibility.

    The outlier rule uses robust median absolute deviation:

        z_i = 0.6745 (r_i - median(r)) / MAD.

    Positions with ``z_i > outlier_z`` are flagged rather than deleted.  Negative
    values are also flagged but not rejected automatically because Ribonanza
    documentation notes that measurement errors may make normalized reactivity
    slightly negative.

    Complexity: O(L log L) time due to median sorting and O(L) memory.
    """

    sequence = profile.sequence.upper()
    messages: List[str] = []
    if not sequence:
        messages.append("sequence is empty")
    invalid_bases = sorted(set(sequence) - VALID_BASES)
    if invalid_bases:
        messages.append(f"sequence contains invalid RNA bases: {''.join(invalid_bases)}")

    if len(profile.reactivity) != len(sequence):
        messages.append(
            f"reactivity length {len(profile.reactivity)} does not match sequence length {len(sequence)}"
        )
    if profile.error is not None and len(profile.error) != len(sequence):
        messages.append(f"error length {len(profile.error)} does not match sequence length {len(sequence)}")

    finite_values = [float(v) for v in profile.reactivity if _is_finite(v)]
    valid_positions = len(finite_values)
    missing_positions = len(profile.reactivity) - valid_positions
    negative_positions = sum(1 for v in finite_values if v < 0)
    high_outlier_positions = _count_high_outliers(finite_values, outlier_z=outlier_z)

    reads_pass = profile.reads is None or profile.reads > min_reads
    snr_pass = profile.snr is None or profile.snr > min_snr
    if not reads_pass:
        messages.append(f"reads={profile.reads} <= min_reads={min_reads}")
    if not snr_pass:
        messages.append(f"snr={profile.snr} <= min_snr={min_snr}")
    if valid_positions == 0 and len(sequence) > 0:
        messages.append("no finite reactivity values")

    return ProfileValidationReport(
        sequence_length=len(sequence),
        valid_positions=valid_positions,
        missing_positions=missing_positions,
        negative_positions=negative_positions,
        high_outlier_positions=high_outlier_positions,
        snr_pass=snr_pass,
        reads_pass=reads_pass,
        messages=tuple(messages),
    )


def _count_high_outliers(values: Sequence[float], *, outlier_z: float) -> int:
    """Count robust high outliers by median absolute deviation.

    Formula: ``z_i = 0.6745 * (x_i - median(x)) / MAD``.  Only high positive
    outliers are counted because negative values are tracked separately.
    Complexity: O(L log L) due to median sorting.
    """

    if len(values) < 3:
        return 0
    median = _median(values)
    deviations = [abs(v - median) for v in values]
    mad = _median(deviations)
    if mad == 0:
        return 0
    return sum(1 for v in values if 0.6745 * (v - median) / mad > outlier_z)


def _median(values: Sequence[float]) -> float:
    """Return the exact median of a finite sequence.

    Complexity: O(L log L) time and O(L) memory.
    """

    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median is undefined for empty values")
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def normalize_profile(
    values: Sequence[float],
    *,
    method: str = "p90",
    clip_negative: bool = True,
) -> Tuple[float, ...]:
    """Normalize reactivity values while preserving missing positions.

    Supported methods:

    * ``p90``: divide by the 90th percentile of finite values, matching the
      normalization described by Ribonanza CSV documentation.
    * ``zscore``: subtract mean and divide by population standard deviation.
    * ``minmax``: map finite values to [0, 1].

    Missing values remain NaN.  If ``clip_negative=True``, finite negatives are
    clipped to zero after normalization; this is useful for loss targets because
    reactivity is non-negative in the underlying chemistry, while raw estimates
    may be slightly negative from noise.

    Complexity: O(L log L) for percentile sorting, O(L) memory.
    """

    finite = [float(v) for v in values if _is_finite(v)]
    if not finite:
        raise ValueError("cannot normalize profile with no finite values")

    method = method.lower()
    if method == "p90":
        scale = _percentile(finite, 90.0)
        shift = 0.0
        if scale <= 0:
            scale = max(max(finite), 1.0)
    elif method == "zscore":
        shift = sum(finite) / len(finite)
        variance = sum((v - shift) ** 2 for v in finite) / len(finite)
        scale = math.sqrt(variance) if variance > 0 else 1.0
    elif method == "minmax":
        lo, hi = min(finite), max(finite)
        shift = lo
        scale = hi - lo if hi > lo else 1.0
    else:
        raise ValueError("method must be one of: p90, zscore, minmax")

    normalized: List[float] = []
    for value in values:
        if not _is_finite(value):
            normalized.append(math.nan)
            continue
        result = (float(value) - shift) / scale
        if clip_negative and result < 0:
            result = 0.0
        normalized.append(result)
    return tuple(normalized)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile.

    The rank is ``p/100 * (n-1)``; neighboring sorted values are linearly
    interpolated.  Complexity: O(L log L).
    """

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def effective_mask(profile: RibonanzaProfile) -> Tuple[bool, ...]:
    """Return positions contributing to supervised losses.

    ``m_i=1`` iff the value is finite and the probe is chemically meaningful for
    the nucleotide.  Complexity: O(L).
    """

    base_mask = probe_base_mask(profile.sequence.upper(), profile.probe)
    return tuple(base_ok and _is_finite(value) for base_ok, value in zip(base_mask, profile.reactivity))


def inverse_error_weights(
    error: Optional[Sequence[float]],
    mask: Sequence[bool],
    *,
    min_error: float = 1e-3,
) -> Tuple[float, ...]:
    """Compute reliability weights ``rho_i = 1 / max(error_i, eps)^2``.

    Masked positions receive zero weight.  If no error vector is provided, valid
    positions receive unit weight.  Complexity: O(L).
    """

    if error is None:
        return tuple(1.0 if keep else 0.0 for keep in mask)
    if len(error) != len(mask):
        raise ValueError("error and mask lengths differ")
    weights: List[float] = []
    for keep, err in zip(mask, error):
        if not keep or not _is_finite(err):
            weights.append(0.0)
        else:
            sigma = max(abs(float(err)), min_error)
            weights.append(1.0 / (sigma * sigma))
    return tuple(weights)


def read_ribonanza_csv(path: Path, *, limit: Optional[int] = None) -> Iterator[RibonanzaProfile]:
    """Read Stanford Ribonanza CSV profiles.

    The reader supports the competition's wide format with columns named
    ``reactivity_0001`` ... and optional ``reactivity_error_0001`` ....

    Complexity: O(NL) time for N rows and max length L; streaming memory O(L).
    """

    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        reactivity_cols = sorted(name for name in fieldnames if name.startswith("reactivity_"))
        error_cols = sorted(name for name in fieldnames if name.startswith("reactivity_error_"))
        if not reactivity_cols:
            raise ValueError("CSV does not contain reactivity_* columns")
        for row_index, row in enumerate(reader):
            if limit is not None and row_index >= limit:
                break
            sequence = (row.get("sequence") or "").upper()
            length = len(sequence)
            reactivity = tuple(coerce_float(row.get(col)) for col in reactivity_cols[:length])
            error = tuple(coerce_float(row.get(col)) for col in error_cols[:length]) if error_cols else None
            probe = normalize_probe_name(row.get("experiment_type") or row.get("probe") or "")
            reads = coerce_float(row.get("reads")) if "reads" in row else None
            snr = coerce_float(row.get("signal_to_noise") or row.get("SNR")) if (
                "signal_to_noise" in row or "SNR" in row
            ) else None
            yield RibonanzaProfile(
                sequence=sequence,
                probe=probe,
                reactivity=reactivity,
                error=error,
                reads=reads,
                snr=snr,
                sequence_id=row.get("sequence_id"),
            )


@dataclass(frozen=True)
class EfoldRecord:
    """One eFold/RNAndria structure record.

    Attributes:
        sequence: RNA sequence over A/C/G/U (parsed uppercase).
        pairs: Zero-based base pairs ``(i, j)`` with ``i < j``.
        shape: Optional per-position normalized SHAPE/DMS reactivity, length ``L``
            with ``math.nan`` for missing positions.
        reactivity_probe: Probe label for ``shape`` when present.  Dryad uses
            ``shape`` for SHAPE-like profiles (mapped to ``2A3`` by ReactFlow)
            and ``dms`` for DMS profiles; records without probing data set this
            to ``None``.
        family: Optional family/clan label carried through for split assignment.
        record_id: Optional source identifier.

    Complexity: O(L + P) storage for sequence length L and P base pairs.
    """

    sequence: str
    pairs: Tuple[Tuple[int, int], ...]
    shape: Optional[Tuple[float, ...]] = None
    reactivity_probe: Optional[str] = None
    family: Optional[str] = None
    record_id: Optional[str] = None


def _normalize_pair(raw_pair: Sequence[object], length: int, one_based: bool) -> Tuple[int, int]:
    """Coerce one raw ``[i, j]`` entry into a validated zero-based ``(i, j)``.

    The eFold JSON stores base pairs as two-element lists.  Some upstream dumps
    are one-based (following Connectivity Table / bpseq conventions), so an
    explicit ``one_based`` flag shifts indices by one rather than guessing.  The
    returned pair is ordered ``i < j`` and range-checked; a self-pair or an
    out-of-range index raises instead of being silently dropped.

    Complexity: O(1).
    """

    if len(raw_pair) != 2:
        raise ValueError(f"base pair must have exactly two indices, got {raw_pair!r}")
    i, j = int(raw_pair[0]), int(raw_pair[1])
    if one_based:
        i -= 1
        j -= 1
    if i == j:
        raise ValueError(f"base pair ({i},{j}) is a self-pair")
    if not (0 <= i < length and 0 <= j < length):
        raise ValueError(f"base pair ({i},{j}) out of range for length {length}")
    return (i, j) if i < j else (j, i)


def parse_efold_record(entry: Mapping[str, object], *, one_based: bool = False) -> EfoldRecord:
    """Parse a single eFold JSON object into a validated :class:`EfoldRecord`.

    The eFold/RNAndria schema (Dryad DOI ``10.5061/dryad.79cnp5j95``) stores each
    record as ``{"sequence": "ACGU...", "structure": [[i, j], ...], "shape":
    [floats]}``.  Field names vary slightly across the released files, so the
    common aliases ``structure``/``pairs`` and ``shape``/``dms``/``reactivity``
    are accepted.  Parsing is strict about *structure*: malformed pairs raise so
    corrupt external data cannot enter the pipeline unnoticed.

    The ``shape`` vector, when present, is coerced with :func:`coerce_float`
    (blanks and ``null`` become ``math.nan``) and its length is required to match
    the sequence, because a mismatched reactivity vector would silently
    misalign supervision.

    Complexity: O(L + P) for sequence length ``L`` and ``P`` base pairs.
    """

    sequence = str(entry.get("sequence", "")).upper()
    if not sequence:
        raise ValueError("eFold record is missing a non-empty 'sequence'")
    invalid = sorted(set(sequence) - VALID_BASES)
    if invalid:
        raise ValueError(f"eFold sequence contains invalid RNA bases: {''.join(invalid)}")

    raw_structure = entry.get("structure")
    if raw_structure is None:
        raw_structure = entry.get("pairs")
    if raw_structure is None:
        raise ValueError("eFold record is missing 'structure'/'pairs'")
    if not isinstance(raw_structure, (list, tuple)):
        raise ValueError("eFold 'structure' must be a list of [i, j] pairs")
    pairs = tuple(
        _normalize_pair(raw_pair, len(sequence), one_based)  # type: ignore[arg-type]
        for raw_pair in raw_structure
    )

    raw_shape = entry.get("shape")
    reactivity_probe: Optional[str] = "2A3" if raw_shape is not None else None
    if raw_shape is None:
        raw_shape = entry.get("dms")
        if raw_shape is not None:
            reactivity_probe = "DMS"
    if raw_shape is None:
        raw_shape = entry.get("reactivity")
        if raw_shape is not None:
            reactivity_probe = str(entry.get("probe", "2A3"))
    shape: Optional[Tuple[float, ...]] = None
    if raw_shape is not None:
        if not isinstance(raw_shape, (list, tuple)):
            raise ValueError("eFold 'shape' must be a list of floats")
        if len(raw_shape) != len(sequence):
            raise ValueError(
                f"shape length {len(raw_shape)} does not match sequence length {len(sequence)}"
            )
        values = []
        for value in raw_shape:
            coerced = coerce_float(value)
            # The eFold Dryad README documents DMS files with -1000 sentinels
            # at unavailable positions.  Treat them as missing observations so
            # they are masked out of calibration and reactivity losses.
            values.append(math.nan if math.isfinite(coerced) and coerced <= -999.0 else coerced)
        shape = tuple(values)
        if reactivity_probe is not None:
            reactivity_probe = normalize_probe_name(reactivity_probe)

    family = entry.get("family")
    if family is None:
        family = entry.get("clan")
    record_id = entry.get("id")
    if record_id is None:
        record_id = entry.get("reference")
    return EfoldRecord(
        sequence=sequence,
        pairs=pairs,
        shape=shape,
        reactivity_probe=reactivity_probe,
        family=str(family) if family is not None else None,
        record_id=str(record_id) if record_id is not None else None,
    )


def read_efold_json(path: Path, *, limit: Optional[int] = None, one_based: bool = False) -> Iterator[EfoldRecord]:
    """Stream eFold/RNAndria JSON structure records from ``path``.

    The released files are JSON documents that are either a top-level list of
    records or a mapping ``{id: record}``.  Both layouts are supported; the
    mapping form injects the key as ``record_id`` when the record itself lacks
    one.  Records are yielded lazily after per-record validation by
    :func:`parse_efold_record`.

    Note that this reader uses :func:`json.load`, so it materializes one file in
    memory.  The individual eFold files (e.g. ``archiveII.json``) are small; the
    349 MB ``efold_train.json`` should be pre-split offline before use, matching
    the "download stays out of the import graph" contract of cycle C5.

    Complexity: O(N (L + P)) for ``N`` records; memory O(file size) for the load
    plus O(L + P) per yielded record.
    """

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, Mapping):
        items: Iterable[Tuple[Optional[str], Mapping[str, object]]] = (
            (str(key), value) for key, value in payload.items()  # type: ignore[misc]
        )
    elif isinstance(payload, (list, tuple)):
        items = ((None, value) for value in payload)  # type: ignore[misc]
    else:
        raise ValueError("eFold JSON must be a list of records or a mapping of records")

    for index, (key, raw_entry) in enumerate(items):
        if limit is not None and index >= limit:
            break
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"eFold record #{index} is not a JSON object")
        record = parse_efold_record(raw_entry, one_based=one_based)
        if record.record_id is None and key is not None:
            record = EfoldRecord(
                sequence=record.sequence,
                pairs=record.pairs,
                shape=record.shape,
                reactivity_probe=record.reactivity_probe,
                family=record.family,
                record_id=key,
            )
        yield record


def efold_pair_matrix(record: EfoldRecord) -> Tuple[Tuple[int, ...], ...]:
    """Return the symmetric binary pair matrix for an :class:`EfoldRecord`.

    Thin bridge to :func:`reactflow.constraints.pairs_to_matrix` so downstream
    training/evaluation code consumes eFold structures in the same ``P`` matrix
    representation used everywhere else.  Imported lazily to avoid a module-level
    import cycle between :mod:`reactflow.data` and :mod:`reactflow.constraints`.

    Complexity: O(L^2 + P).
    """

    from reactflow.constraints import pairs_to_matrix

    return pairs_to_matrix(record.pairs, len(record.sequence))


def inspect_ribonanza2_h5(path: Path) -> Mapping[str, Tuple[int, ...]]:
    """Inspect a Ribonanza2 H5 file schema.

    This function fully implements the H5 schema check but imports ``h5py``
    lazily so the core package remains usable without optional data dependencies.

    Complexity: O(K) over the number of top-level H5 datasets; it does not load
    the 174GB arrays into memory.
    """

    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("inspect_ribonanza2_h5 requires optional dependency h5py") from exc

    expected = set(PUBLIC_DATASETS[0].expected_schema)
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
        missing = expected - keys
        if missing:
            raise ValueError(f"H5 file is missing required datasets: {sorted(missing)}")
        return {key: tuple(handle[key].shape) for key in sorted(keys)}


def feature_engineering_report(profile: RibonanzaProfile) -> Dict[str, object]:
    """Return deterministic feature-engineering metadata for documentation.

    Features:
    * base composition;
    * GC fraction;
    * probe-effective mask count;
    * normalized profile summary.

    Complexity: O(L).
    """

    sequence = profile.sequence.upper()
    length = len(sequence)
    counts = {base: sequence.count(base) for base in sorted(VALID_BASES)}
    gc_fraction = (counts["G"] + counts["C"]) / length if length else math.nan
    mask = effective_mask(profile)
    finite = [value for value in profile.reactivity if _is_finite(value)]
    return {
        "sequence_id": profile.sequence_id,
        "length": length,
        "base_counts": counts,
        "gc_fraction": gc_fraction,
        "probe": normalize_probe_name(profile.probe),
        "effective_positions": sum(mask),
        "finite_reactivity_min": min(finite) if finite else math.nan,
        "finite_reactivity_max": max(finite) if finite else math.nan,
    }
