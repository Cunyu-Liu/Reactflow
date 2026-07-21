"""Frozen-encoder warm-start features for cycle C5.3.

Cycle C5 freezes a 100M-parameter encoder (RibonanzaNet2 / eFold) *once,
offline* and stores its per-nucleotide representation ``h_i in R^{d_single}`` to
disk (see :mod:`reactflow.frozen`).  This module is the pure standard-library
bridge that lets the hand-written denoiser consume those frozen vectors without
ever importing PyTorch: a small **linear adapter** projects each frozen vector
into a low-dimensional slot that is concatenated onto the C3 hand-written
features.

Adapter
-------
For each position ``i`` the adapter computes

    a_i = W h_i + b,        W in R^{d_adapter x d_single},  b in R^{d_adapter}.

There is no non-linearity: the adapter is a plain affine map, so its Jacobian is
constant and its gradients are exact by inspection.  Because the encoder is
*frozen*, ``h_i`` is a constant input and the adapter only accumulates gradients
into ``W`` and ``b``:

    dL/dW[p][f] = sum_i g_i[p] h_i[f],      dL/db[p] = sum_i g_i[p],

where ``g_i = dL/da_i`` is the upstream gradient of the adapter output.  When the
adapter output occupies the trailing ``d_adapter`` slots of the denoiser feature
vector, ``g_i`` is exactly the trailing block of the denoiser's
``grad_features`` (see :meth:`reactflow.model.PairwiseDenoiser.backward`).  This
closes the chain from the DFM loss all the way back to the adapter weights.

Fallback
--------
The repository must still train on a machine with no external weights.  When no
frozen shard is supplied, :func:`build_augmented_features` returns the plain C3
hand-written features unchanged (``FEATURE_SIZE`` dimensions), so the pilot is
bit-for-bit identical to C3.  When a shard *is* supplied but a particular
sequence has no matching frozen record, the adapter input falls back to a zero
vector, so its contribution degrades gracefully to the bias ``b`` while keeping
the feature dimensionality constant across the dataset.

Determinism
-----------
Adapter parameters are seeded and the frozen vectors are constants, so the whole
pipeline is bit-for-bit reproducible across platforms.

Complexity
----------
Forward and backward are ``O(L * d_adapter * d_single)`` per sequence.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from reactflow.frozen import (
    FrozenFeatureProvenance,
    FrozenShard,
    read_frozen_shard,
    read_frozen_single_array,
    read_frozen_single_arrays,
)


@dataclass
class AdapterParameters:
    """Learnable parameters of the linear frozen-feature adapter.

    Formula: each frozen encoder row ``h_i in R^{d_single}`` is projected as
    ``a_i = W h_i + b`` with ``W in R^{d_adapter x d_single}``.
    Complexity: O(1) storage metadata plus O(d_adapter * d_single) parameters.

    Attributes:
        weight: ``W`` as a ``d_adapter x d_single`` nested list.
        bias: ``b`` as a length ``d_adapter`` list.
    """

    weight: List[List[float]]  # d_adapter x d_single
    bias: List[float]  # d_adapter

    @property
    def d_adapter(self) -> int:
        """Output dimension ``d_adapter``.

        Formula: ``d_adapter = len(b)``.  Complexity: O(1).
        """

        return len(self.bias)

    @property
    def d_single(self) -> int:
        """Input frozen-representation dimension ``d_single``.

        Formula: ``d_single = W.shape[1]`` when ``W`` is non-empty.  Complexity:
        O(1).
        """

        return len(self.weight[0]) if self.weight else 0

    @staticmethod
    def random_init(
        d_single: int,
        d_adapter: int,
        *,
        seed: int = 0,
        scale: Optional[float] = None,
    ) -> "AdapterParameters":
        """Deterministically initialize the adapter with small uniform weights.

        The default ``scale`` is ``1 / sqrt(d_single)`` (a Xavier-style bound) so
        the projected activation ``a_i = W h_i + b`` has ``O(1)`` magnitude even
        when ``d_single`` is large (e.g. 384 for RibonanzaNet2).  A fixed ``seed``
        guarantees identical parameters across platforms.

        Complexity: O(d_adapter * d_single).
        """

        if d_single <= 0 or d_adapter <= 0:
            raise ValueError("d_single and d_adapter must be positive")
        if scale is None:
            scale = 1.0 / math.sqrt(d_single)
        rng = random.Random(seed)

        def uniform() -> float:
            """Sample one symmetric initialization coefficient.

            Formula: ``u = (2 * U[0,1) - 1) * scale``.  Complexity: O(1).
            """

            return (rng.random() * 2.0 - 1.0) * scale

        weight = [[uniform() for _ in range(d_single)] for _ in range(d_adapter)]
        bias = [0.0 for _ in range(d_adapter)]
        return AdapterParameters(weight=weight, bias=bias)


@dataclass
class AdapterGradients:
    """Gradient container mirroring :class:`AdapterParameters`.

    Complexity: O(d_adapter * d_single) storage.
    """

    weight: List[List[float]]
    bias: List[float]


class FeatureAdapter:
    """A linear projection ``a_i = W h_i + b`` with hand-written backprop.

    Formula: the backward pass accumulates ``dL/dW = sum_i g_i h_i^T`` and
    ``dL/db = sum_i g_i``.  Complexity per sequence is
    O(L * d_adapter * d_single).
    """

    def __init__(self, parameters: AdapterParameters) -> None:
        """Store the adapter parameters."""

        self.parameters = parameters

    def forward(self, single_rows: Sequence[Sequence[float]]) -> List[List[float]]:
        """Project each frozen row ``h_i`` to ``a_i = W h_i + b``.

        ``single_rows`` is a length-``L`` sequence of ``d_single`` vectors.  The
        returned matrix is ``L x d_adapter``.

        Complexity: O(L * d_adapter * d_single).
        """

        params = self.parameters
        d_single = params.d_single
        outputs: List[List[float]] = []
        for row in single_rows:
            if len(row) != d_single:
                raise ValueError(
                    f"frozen row has wrong dimension {len(row)} (expected {d_single})"
                )
            projected = []
            for weight_row, bias in zip(params.weight, params.bias):
                total = bias
                for weight, value in zip(weight_row, row):
                    total += weight * float(value)
                projected.append(total)
            outputs.append(projected)
        return outputs

    def backward(
        self,
        single_rows: Sequence[Sequence[float]],
        grad_output: Sequence[Sequence[float]],
    ) -> AdapterGradients:
        """Backpropagate ``g_i = dL/da_i`` into ``dL/dW`` and ``dL/db``.

        ``grad_output[i]`` is the length ``d_adapter`` upstream gradient of the
        adapter output at position ``i``.  Because the frozen input ``h_i`` is a
        constant, the adapter has no input gradient; only the affine parameters
        accumulate:

            dL/dW[p][f] = sum_i g_i[p] h_i[f],   dL/db[p] = sum_i g_i[p].

        Complexity: O(L * d_adapter * d_single).
        """

        params = self.parameters
        d_single = params.d_single
        d_adapter = params.d_adapter
        if len(grad_output) != len(single_rows):
            raise ValueError("grad_output length must match single_rows length")

        grad_weight = [[0.0 for _ in range(d_single)] for _ in range(d_adapter)]
        grad_bias = [0.0 for _ in range(d_adapter)]
        for row, g_row in zip(single_rows, grad_output):
            if len(g_row) != d_adapter:
                raise ValueError(
                    f"grad_output row has wrong dimension {len(g_row)} (expected {d_adapter})"
                )
            if len(row) != d_single:
                raise ValueError(
                    f"frozen row has wrong dimension {len(row)} (expected {d_single})"
                )
            for p in range(d_adapter):
                g = float(g_row[p])
                grad_bias[p] += g
                grad_row = grad_weight[p]
                for f_index in range(d_single):
                    grad_row[f_index] += g * float(row[f_index])
        return AdapterGradients(weight=grad_weight, bias=grad_bias)


def adapter_sgd_update(
    parameters: AdapterParameters,
    gradients: AdapterGradients,
    learning_rate: float,
) -> None:
    """Apply an in-place SGD step to the adapter parameters.

    Complexity: O(d_adapter * d_single).
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    for row, grad_row in zip(parameters.weight, gradients.weight):
        for index in range(len(row)):
            row[index] -= learning_rate * grad_row[index]
    for index in range(len(parameters.bias)):
        parameters.bias[index] -= learning_rate * gradients.bias[index]


@dataclass
class FrozenFeatureLookup:
    """Sequence-keyed view over a loaded frozen shard.

    :class:`~reactflow.synthetic.SyntheticSample` carries no record id, so warm
    signals are aligned to the training data by exact RNA sequence.  A duplicate
    sequence within the shard raises, since the alignment must be unambiguous.

    Attributes:
        d_single: per-nucleotide representation dimension.
        by_sequence: eagerly loaded mapping from RNA sequence to its ``L x d_single`` rows.
        shard_by_sequence: optional lazy index mapping sequence to the shard
            directory that contains it.  This is used for full-scale sharded
            exports where loading every frozen row into Python tuples at once
            would be prohibitively large.
        row_by_sequence: optional lazy index mapping sequence to its zero-based
            row within the child shard.  When present, lookup reads only that
            NPZ member instead of materializing the whole shard.
        max_loaded_shards: maximum number of numeric child shards kept in the
            in-process LRU cache.  ``1`` reproduces the original single-active
            shard behaviour; values such as ``4`` reduce repeated NPZ reads when
            length-bucketed training jumps among nearby shards while keeping
            memory bounded.

    Complexity: O(N) index storage for N records plus O(K * shard_rows) bounded
    row-cache storage for ``max_loaded_shards = K``.
    """

    d_single: int
    by_sequence: Dict[str, Tuple[Tuple[float, ...], ...]]
    shard_by_sequence: Optional[Dict[str, Path]] = None
    row_by_sequence: Optional[Dict[str, int]] = None
    verify: bool = True
    record_count: int = 0
    max_loaded_shards: int = 4
    _loaded_shards: OrderedDict[Path, Dict[str, Tuple[Tuple[float, ...], ...]]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _verified_shards: Set[Path] = field(default_factory=set, init=False, repr=False)

    def __len__(self) -> int:
        """Return the number of indexed sequences/records."""

        return self.record_count or len(self.by_sequence)

    def single_rows(self, sequence: str) -> Optional[Tuple[Tuple[float, ...], ...]]:
        """Return the frozen per-nucleotide rows for ``sequence`` if present.

        Complexity: O(1) for eager records, or O(shard bytes) on the first lazy
        verified lookup into a new shard and O(L * d_single) for later targeted
        record loads from already verified shards.  The LRU cache bounds memory
        to recently used rows from ``max_loaded_shards`` child shards instead of
        the full export.
        """

        key = sequence.upper()
        rows = self.by_sequence.get(key)
        if rows is not None:
            return rows
        if self.shard_by_sequence is None:
            return None
        shard_path = self.shard_by_sequence.get(key)
        if shard_path is None:
            return None
        loaded = self._loaded_shards.get(shard_path)
        if loaded is not None:
            self._loaded_shards.move_to_end(shard_path)
            cached = loaded.get(key)
            if cached is not None:
                return cached
        else:
            loaded = {}
            self._loaded_shards[shard_path] = loaded
            self._loaded_shards.move_to_end(shard_path)
            limit = max(1, int(self.max_loaded_shards))
            while len(self._loaded_shards) > limit:
                self._loaded_shards.popitem(last=False)

        row = self.row_by_sequence.get(key) if self.row_by_sequence is not None else None
        if row is not None:
            should_verify = self.verify and shard_path not in self._verified_shards
            single = read_frozen_single_array(shard_path, row, verify=should_verify)
            if should_verify:
                self._verified_shards.add(shard_path)
            if len(single.shape) != 2 or single.shape[1] != self.d_single:
                raise ValueError(
                    f"'single' shape {single.shape} incompatible with d_single {self.d_single}"
                )
            rows_for_record = tuple(single.row(i) for i in range(single.shape[0]))
            loaded[key] = rows_for_record
            return rows_for_record

        # Backward-compatible fallback for lookups constructed without row
        # metadata: materialize the child shard once and keep rows in the LRU.
        shard = read_frozen_shard(
            shard_path,
            verify=self.verify and shard_path not in self._verified_shards,
        )
        self._verified_shards.add(shard_path)
        for record in shard.records:
            record_key = record.sequence.upper()
            if record_key in loaded:
                continue
            single = record.single()
            loaded[record_key] = tuple(single.row(i) for i in range(record.length))
        return loaded.get(key)

    def prefetch(self, sequences: Iterable[str]) -> int:
        """Load uncached frozen rows for upcoming sequences into the LRU cache.

        Formula: requested sequences are upper-cased, filtered to lazy indexed
        rows, grouped by child shard in first-use order, then the first
        ``max_loaded_shards`` missing shard groups read selected members
        ``H_s in R^{L_s x d_single}`` with one :func:`read_frozen_single_arrays`
        call per shard.  Limiting prefetch to the LRU capacity avoids a batch
        spanning many shards from evicting its own earliest prefetched rows
        before the trainer consumes them.  The returned integer is the number of
        newly cached sequences.  Training outputs are unchanged because
        :meth:`single_rows` later returns the same row tensors it would have read
        one at a time.

        Complexity: O(B + min(S,K) selected shard bytes + selected L*d_single),
        where B is requested sequence count, S is missing shard count and
        ``K=max_loaded_shards``.
        """

        if self.shard_by_sequence is None:
            return 0
        grouped: Dict[Path, List[Tuple[str, int]]] = {}
        seen_keys: Set[str] = set()
        for sequence in sequences:
            key = str(sequence).upper()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if key in self.by_sequence:
                continue
            shard_path = self.shard_by_sequence.get(key)
            if shard_path is None:
                continue
            loaded = self._loaded_shards.get(shard_path)
            if loaded is not None and key in loaded:
                continue
            row = self.row_by_sequence.get(key) if self.row_by_sequence is not None else None
            if row is None:
                continue
            grouped.setdefault(shard_path, []).append((key, row))

        cached = 0
        limit = max(1, int(self.max_loaded_shards))
        for shard_path, items in list(grouped.items())[:limit]:
            loaded = self._loaded_shards.get(shard_path)
            if loaded is None:
                loaded = {}
                self._loaded_shards[shard_path] = loaded
            self._loaded_shards.move_to_end(shard_path)
            while len(self._loaded_shards) > limit:
                self._loaded_shards.popitem(last=False)
            should_verify = self.verify and shard_path not in self._verified_shards
            arrays = read_frozen_single_arrays(
                shard_path,
                [row for _, row in items],
                verify=should_verify,
            )
            if should_verify:
                self._verified_shards.add(shard_path)
            for key, row in items:
                single = arrays[row]
                if len(single.shape) != 2 or single.shape[1] != self.d_single:
                    raise ValueError(
                        f"'single' shape {single.shape} incompatible with d_single {self.d_single}"
                    )
                loaded[key] = tuple(single.row(i) for i in range(single.shape[0]))
                cached += 1
        return cached

    def has(self, sequence: str) -> bool:
        """Return whether a frozen record exists for ``sequence``.

        Complexity: O(1).
        """

        key = sequence.upper()
        return key in self.by_sequence or (
            self.shard_by_sequence is not None and key in self.shard_by_sequence
        )


def _shard_dirs(directory: Path) -> List[Path]:
    """Return direct shard directories under ``directory``.

    ``directory`` itself is returned for the legacy single-shard layout.
    """

    if (directory / "provenance.json").exists():
        return [directory]
    return sorted(
        child
        for child in directory.iterdir()
        if child.is_dir() and (child / "provenance.json").exists() and (child / "index.jsonl").exists()
    )


def _d_single_from_provenance(path: Path) -> int:
    """Read ``d_single`` from a shard provenance file without loading arrays."""

    obj = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    provenance = FrozenFeatureProvenance.from_json_obj(obj)
    axes = provenance.schema["single"]["axes"]
    return int(axes[1])  # type: ignore[index]


def load_frozen_features(
    directory: Union[str, Path],
    *,
    verify: bool = True,
    max_loaded_shards: int = 4,
) -> FrozenFeatureLookup:
    """Load a frozen shard and index its ``single`` arrays by sequence.

    The shard's content hash is verified by :func:`reactflow.frozen.read_frozen_shard`
    (unless ``verify`` is false).  All records must share the same ``d_single`` so
    a single adapter can consume the whole shard.

    For a directory containing multiple child shard directories, only
    ``index.jsonl`` + provenance files are read up front.  The numeric arrays are
    loaded lazily into a bounded LRU cache when
    :meth:`FrozenFeatureLookup.single_rows` is called.  This is the full-scale
    warm-start path.

    Complexity: O(total single-array elements) for a legacy single shard, or
    O(total index lines) upfront plus lazy O(shard bytes) for each shard's first
    verified lookup and O(L * d_single) for later targeted record loads.
    """

    directory = Path(directory)
    shards = _shard_dirs(directory)
    if not shards:
        raise ValueError(f"no frozen shard found in {directory}")
    if len(shards) > 1:
        d_single: Optional[int] = None
        shard_by_sequence: Dict[str, Path] = {}
        row_by_sequence: Dict[str, int] = {}
        record_count = 0
        for shard_dir in shards:
            shard_d_single = _d_single_from_provenance(shard_dir)
            if d_single is None:
                d_single = shard_d_single
            elif shard_d_single != d_single:
                raise ValueError(f"inconsistent d_single in sharded frozen directory: {shard_d_single} vs {d_single}")
            with (shard_dir / "index.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record_count += 1
                    entry = json.loads(line)
                    key = str(entry["sequence"]).upper()
                    # RibonanzaNet2 frozen features are sequence-only, so exact
                    # duplicate windows are intentionally de-duplicated in the
                    # lazy index instead of making alignment ambiguous.
                    if key not in shard_by_sequence:
                        shard_by_sequence[key] = shard_dir
                        row_by_sequence[key] = int(entry["row"])
        if d_single is None or record_count == 0:
            raise ValueError("frozen shard contains no records")
        return FrozenFeatureLookup(
            d_single=d_single,
            by_sequence={},
            shard_by_sequence=shard_by_sequence,
            row_by_sequence=row_by_sequence,
            verify=verify,
            record_count=record_count,
            max_loaded_shards=max_loaded_shards,
        )

    shard: FrozenShard = read_frozen_shard(shards[0], verify=verify)
    by_sequence: Dict[str, Tuple[Tuple[float, ...], ...]] = {}
    d_single: Optional[int] = None
    for record in shard.records:
        if d_single is None:
            d_single = record.d_single
        elif record.d_single != d_single:
            raise ValueError(
                f"inconsistent d_single in shard: {record.d_single} vs {d_single}"
            )
        key = record.sequence.upper()
        if key in by_sequence:
            raise ValueError(f"duplicate sequence in frozen shard: {key!r}")
        single = record.single()
        rows = tuple(single.row(i) for i in range(record.length))
        by_sequence[key] = rows
    if d_single is None:
        raise ValueError("frozen shard contains no records")
    return FrozenFeatureLookup(d_single=d_single, by_sequence=by_sequence)


def zero_single_rows(length: int, d_single: int) -> Tuple[Tuple[float, ...], ...]:
    """Return ``length`` rows of ``d_single`` zeros (missing-record fallback).

    Complexity: O(length * d_single).
    """

    zero = tuple(0.0 for _ in range(d_single))
    return tuple(zero for _ in range(length))


def build_augmented_features(
    base_features: Sequence[Sequence[float]],
    adapter: Optional[FeatureAdapter],
    single_rows: Optional[Sequence[Sequence[float]]],
) -> Tuple[Tuple[Tuple[float, ...], ...], Optional[Tuple[Tuple[float, ...], ...]]]:
    """Concatenate the adapter output onto the C3 hand-written features.

    Behaviour:

    * ``adapter is None``  -> return ``base_features`` unchanged and ``None`` for
      the adapter input (the C3 pilot path; bit-for-bit unchanged).
    * ``adapter`` given, ``single_rows`` given -> append ``a_i = W h_i + b`` to
      each base row, giving ``len(base) + d_adapter`` dimensions.
    * ``adapter`` given, ``single_rows is None`` -> fall back to a zero frozen
      vector per position, so the augmented dimensionality is unchanged and the
      adapter contributes only its bias.

    Returns ``(augmented_features, used_single_rows)`` where ``used_single_rows``
    is the (possibly zero-filled) frozen input the adapter consumed -- the caller
    needs it to run :meth:`FeatureAdapter.backward`.

    Complexity: O(L * d_adapter * d_single).
    """

    if adapter is None:
        return tuple(tuple(float(v) for v in row) for row in base_features), None

    length = len(base_features)
    if single_rows is None:
        used = zero_single_rows(length, adapter.parameters.d_single)
    else:
        if len(single_rows) != length:
            raise ValueError("single_rows length must match base_features length")
        used = tuple(tuple(float(v) for v in row) for row in single_rows)

    projected = adapter.forward(used)
    augmented = tuple(
        tuple(float(v) for v in base_features[i]) + tuple(projected[i])
        for i in range(length)
    )
    return augmented, used


def split_feature_gradient(
    grad_features: Sequence[Sequence[float]],
    base_size: int,
) -> List[List[float]]:
    """Slice the trailing adapter block out of the denoiser feature gradient.

    The denoiser returns ``dL/dfeat_i`` of length ``base_size + d_adapter``; the
    adapter owns the trailing ``d_adapter`` components, which are exactly its
    upstream gradient ``g_i = dL/da_i``.

    Complexity: O(L * d_adapter).
    """

    return [list(row[base_size:]) for row in grad_features]
