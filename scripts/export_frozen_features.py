#!/usr/bin/env python3
"""Stage-A frozen-feature exporter for ReactFlow cycle C5.2.

This is a *one-time, offline* utility.  It loads a large pretrained RNA encoder
(RibonanzaNet2 or eFold), runs a forward pass over a set of sequences, and
freezes the per-nucleotide and pairwise representations to a ReactFlow shard
(``reactflow.frozen``).  The pure standard-library training core then consumes
that shard as fixed input in cycle C5.3.

Design contract
---------------
* **PyTorch is an optional dependency.**  It is imported lazily inside the
  ``torch`` backend only.  Running ``--backend dry-run`` requires nothing beyond
  the standard library plus the ``reactflow`` package, so the shard *format* and
  the whole pipeline are exercisable on a machine without a GPU or torch.
* **This file is never imported by the ``reactflow`` package.**  It lives under
  ``scripts/`` and puts ``src/`` on ``sys.path`` itself, so PyTorch can never
  leak into the audited import graph of the library.
* **Honesty red line.**  The ``dry-run`` backend writes *deterministic random*
  features and stamps ``weights_sha256=""`` plus an explicit
  ``notes="DRY RUN: deterministic random features, NOT real model weights"``.
  It exists only to validate the plumbing; its output must never be presented as
  a real warm-start signal.  Real features require ``--backend torch`` with a
  genuine checkpoint, whose SHA-256 is recorded.

RibonanzaNet2 interface (verified against Shujun-He/RibonanzaNet ``Network.py``)
-------------------------------------------------------------------------------
``RibonanzaNet(config).forward(src, src_mask, return_aw)`` takes integer token
ids ``src`` of shape ``[B, L]`` (embedding ``nn.Embedding(ntoken, ninp,
padding_idx=4)`` with the vocabulary ``{A:0, C:1, G:2, U:3}`` and pad id 4),
builds the pairwise tensor via ``outer_product_mean(src) + pos_encoder(src)``,
threads both through the ``transformer_encoder`` ModuleList, and finally applies
``decoder`` to the sequence representation to produce reactivity logits.  For
RibonanzaNet2 the dimensions are ``ninp = 384`` (per-nucleotide) and
``pairwise_dimension = 128`` with ``nlayers = 48``.  The exporter captures the
last-layer sequence representation as ``single``, the last-layer pairwise tensor
as ``pair``, and the decoder output as ``react_logits``.

Usage
-----
    # dry run -- no torch, validates the shard format end to end
    python scripts/export_frozen_features.py \
        --sequences data/mini.jsonl --out data/frozen/mini_dry \
        --backend dry-run --d-single 384 --d-pair 128 --n-probe 2

    # real weights -- requires torch and a checkpoint
    python scripts/export_frozen_features.py \
        --sequences data/efold_val.jsonl --out data/frozen/efold_val \
        --backend torch --model RibonanzaNet2 \
        --config path/to/config.yaml --weights path/to/RibonanzaNet2.pt \
        --network-dir path/to/ribonanzanet2d-final
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reactflow.frozen import (  # noqa: E402
    ARRAY_PAIR,
    ARRAY_REACT,
    ARRAY_SINGLE,
    FrozenFeatureProvenance,
    FrozenFeatureRecord,
    default_schema,
    write_frozen_shard,
)
from reactflow.npio import NdArray  # noqa: E402

# The RibonanzaNet vocabulary; pad id 4 matches ``padding_idx=4`` in Network.py.
TOKEN_MAP: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}
PAD_ID = 4


class _ImportOnlyMatplotlibPyplot(types.ModuleType):
    """Import stub for official checkpoints that import pyplot but do not use it.

    Some public RibonanzaNet2 snapshots import ``matplotlib.pyplot`` at module
    import time even though the inference path does not call it.  Installing a
    large plotting stack just to import ``Network.py`` would violate the exporter
    contract, so this stub keeps the forward-only path lightweight while failing
    loudly if plotting is actually attempted.
    """

    def __getattr__(self, name: str) -> object:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            "matplotlib is unavailable; the ReactFlow exporter only provides an "
            f"import-only matplotlib.pyplot stub, but pyplot.{name} was used"
        )


def _install_import_only_matplotlib_stub() -> None:
    """Allow ``import matplotlib.pyplot`` when matplotlib is not installed.

    Complexity: O(1).
    """

    if importlib.util.find_spec("matplotlib") is not None:
        return
    matplotlib = sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    setattr(matplotlib, "__path__", [])
    pyplot = sys.modules.setdefault(
        "matplotlib.pyplot",
        _ImportOnlyMatplotlibPyplot("matplotlib.pyplot"),
    )
    setattr(matplotlib, "pyplot", pyplot)


def tokenize(sequence: str) -> List[int]:
    """Map an RNA sequence to RibonanzaNet token ids.

    ``T`` is folded to ``U`` so DNA-style inputs do not silently become pad
    tokens.  Any other character raises, since a wrong token id would corrupt
    the frozen representation without any downstream signal.

    Complexity: O(L).
    """

    ids: List[int] = []
    for base in sequence.upper():
        base = "U" if base == "T" else base
        if base not in TOKEN_MAP:
            raise ValueError(f"sequence contains non-ACGU base {base!r}")
        ids.append(TOKEN_MAP[base])
    return ids


def read_sequences(path: Path, *, limit: Optional[int] = None) -> List[Tuple[str, str, Optional[str]]]:
    """Read ``(record_id, sequence, family)`` triples from a JSONL file.

    Each line is a JSON object with a ``sequence`` field and optional
    ``id``/``record_id`` and ``family``/``clan`` fields.  A JSONL input keeps the
    exporter decoupled from any single dataset schema; callers pre-convert eFold
    or Ribonanza records into this minimal form.

    Complexity: O(N) lines.
    """

    records: List[Tuple[str, str, Optional[str]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle):
            if not raw.strip():
                continue
            if limit is not None and len(records) >= limit:
                break
            obj = json.loads(raw)
            sequence = str(obj["sequence"]).upper()
            record_id = str(obj.get("id") or obj.get("record_id") or f"seq{line_no:06d}")
            family = obj.get("family")
            if family is None:
                family = obj.get("clan")
            records.append((record_id, sequence, str(family) if family is not None else None))
    return records


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed in chunks.

    Complexity: O(file bytes), O(1) memory.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_dir(path: Path) -> str:
    """Return a SHA-256 over all files in a directory, sorted by relative path.

    Used to fingerprint a multi-file checkpoint directory (weights + config).
    Complexity: O(total bytes).
    """

    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(_sha256_file(file).encode("ascii"))
    return digest.hexdigest()


class DryRunBackend:
    """Deterministic random feature generator for pipeline validation.

    This backend produces reproducible pseudo-features so the shard format and
    downstream reader can be tested without torch, a GPU or 174 GB of data.  It
    is explicitly *not* a model: every shard it writes is stamped as a dry run so
    it can never be mistaken for a real warm-start signal.
    """

    weights_sha256 = ""
    is_real = False

    def __init__(self, *, d_single: int, d_pair: Optional[int], n_probe: Optional[int], seed: int = 0) -> None:
        """Store output dimensions and the master seed."""

        self.d_single = d_single
        self.d_pair = d_pair
        self.n_probe = n_probe
        self.seed = seed

    def _rng_for(self, record_id: str) -> random.Random:
        """Return a per-record RNG seeded from the master seed and id.

        Seeding per record makes each sequence's features independent of batch
        order, so a shard is bit-for-bit identical regardless of how records are
        grouped.  Complexity: O(1).
        """

        mixed = f"{self.seed}:{record_id}".encode("utf-8")
        seed_int = int.from_bytes(hashlib.sha256(mixed).digest()[:8], "big")
        return random.Random(seed_int)

    def encode(self, record_id: str, sequence: str) -> Dict[str, NdArray]:
        """Produce dry-run arrays for one sequence.

        Complexity: O(L * d_single + L^2 * d_pair + L * n_probe).
        """

        length = len(sequence)
        rng = self._rng_for(record_id)
        single_rows = [[rng.gauss(0.0, 1.0) for _ in range(self.d_single)] for _ in range(length)]
        arrays: Dict[str, NdArray] = {
            ARRAY_SINGLE: NdArray.from_nested(single_rows, kind="float32"),
        }
        if self.d_pair is not None:
            pair = [
                [[rng.gauss(0.0, 1.0) for _ in range(self.d_pair)] for _ in range(length)]
                for _ in range(length)
            ]
            arrays[ARRAY_PAIR] = NdArray.from_nested(pair, kind="float32")
        if self.n_probe is not None:
            react = [[rng.gauss(0.0, 1.0) for _ in range(self.n_probe)] for _ in range(length)]
            arrays[ARRAY_REACT] = NdArray.from_nested(react, kind="float32")
        return arrays


class TorchRibonanzaBackend:
    """Real RibonanzaNet/RibonanzaNet2 forward pass (requires torch).

    The backend imports torch lazily and loads the official architecture from a
    ``Network.py`` checkpoint directory (Kaggle ``ribonanzanet2d-final`` style).
    It reproduces the verified forward path: token embedding -> outer-product
    pairwise init + relative-position encoding -> ``transformer_encoder`` stack,
    capturing the last-layer sequence and pairwise representations and the
    decoder reactivity logits.
    """

    is_real = True

    def __init__(
        self,
        *,
        network_dir: Path,
        config_path: Path,
        weights_path: Path,
        device: str = "cpu",
        include_pair: bool = True,
        include_react: bool = True,
    ) -> None:
        """Load the model onto ``device`` and switch it to eval mode.

        Complexity: O(model parameters) for the checkpoint load.
        """

        import torch  # noqa: F401  (lazy, optional dependency)

        self._torch = torch
        self.network_dir = network_dir
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.include_pair = include_pair
        self.include_react = include_react
        self.weights_sha256 = _sha256_file(weights_path)

        # The official repo ships the architecture as ``Network.py`` alongside a
        # ``dropout.py`` helper; add the checkpoint dir to the path so its
        # relative imports resolve, then import the class by name.
        _install_import_only_matplotlib_stub()
        sys.path.insert(0, str(network_dir))
        import importlib

        network_module = importlib.import_module("Network")
        config = self._load_config(config_path)
        self.model = network_module.RibonanzaNet(config)
        state = torch.load(str(weights_path), map_location=device)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(state, strict=False)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _load_config(config_path: Path) -> object:
        """Load a RibonanzaNet config object from a YAML file.

        The official config is a small YAML consumed as attribute access, so it
        is wrapped in a namespace.  YAML is parsed with :mod:`yaml` when present
        or a minimal ``key: value`` fallback otherwise, keeping the hard torch
        path free of extra required deps.

        Complexity: O(config size).
        """

        from types import SimpleNamespace

        text = config_path.read_text(encoding="utf-8")
        data: Dict[str, object]
        try:
            import yaml  # type: ignore

            data = dict(yaml.safe_load(text))
        except Exception:
            data = {}
            for line in text.splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = _coerce_scalar(value.strip())
        return SimpleNamespace(**data)

    def encode(self, record_id: str, sequence: str) -> Dict[str, NdArray]:
        """Run the real forward pass and return frozen arrays for one sequence.

        Complexity: O(nlayers * L^2 * ninp) on the model's device.
        """

        torch = self._torch
        token_ids = tokenize(sequence)
        src = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            single, pair, react = self._forward_capture(src)
        arrays: Dict[str, NdArray] = {
            ARRAY_SINGLE: _tensor_to_ndarray(single[0]),
        }
        if self.include_pair:
            arrays[ARRAY_PAIR] = _tensor_to_ndarray(pair[0])
        if self.include_react and react is not None:
            arrays[ARRAY_REACT] = _tensor_to_ndarray(react[0])
        return arrays

    def encode_many(
        self,
        records: Sequence[Tuple[str, str, Optional[str]]],
        *,
        batch_size: int,
    ) -> List[Dict[str, NdArray]]:
        """Encode records in same-length mini-batches.

        The public alpha checkpoint supports batched tensors.  To preserve exact
        single-sequence semantics, this method only batches records with
        identical sequence length, so no padding or mask convention changes are
        introduced.  Results are returned in the caller's original order.

        Complexity: O(sum over batches of nlayers * B * L^2 * ninp).
        """

        if batch_size <= 1:
            return [self.encode(record_id, sequence) for record_id, sequence, _ in records]
        buckets: Dict[int, List[Tuple[int, str, str]]] = {}
        for index, (record_id, sequence, _) in enumerate(records):
            buckets.setdefault(len(sequence), []).append((index, record_id, sequence))
        encoded: List[Optional[Dict[str, NdArray]]] = [None] * len(records)
        torch = self._torch
        for same_length in buckets.values():
            for start in range(0, len(same_length), batch_size):
                batch_records = same_length[start : start + batch_size]
                token_rows = [tokenize(sequence) for _, _, sequence in batch_records]
                src = torch.tensor(token_rows, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    single, pair, react = self._forward_capture(src)
                for batch_row, (original_index, _, _) in enumerate(batch_records):
                    arrays: Dict[str, NdArray] = {
                        ARRAY_SINGLE: _tensor_to_ndarray(single[batch_row]),
                    }
                    if self.include_pair:
                        arrays[ARRAY_PAIR] = _tensor_to_ndarray(pair[batch_row])
                    if self.include_react and react is not None:
                        arrays[ARRAY_REACT] = _tensor_to_ndarray(react[batch_row])
                    encoded[original_index] = arrays
        if any(arrays is None for arrays in encoded):
            raise RuntimeError("batched encoder returned incomplete outputs")
        return [arrays for arrays in encoded if arrays is not None]

    def _forward_capture(self, src: object) -> Tuple[object, object, Optional[object]]:
        """Reproduce ``RibonanzaNet.forward`` capturing intermediate reps.

        Mirrors the verified ``Network.py`` path so the captured tensors match
        what a fine-tuning head would consume: ``single`` and ``pair`` are the
        representations after the final encoder layer, and ``react`` is the
        decoder output.  Complexity: O(nlayers * L^2 * ninp).
        """

        torch = self._torch
        model = self.model
        batch, length = src.shape
        seq = model.encoder(src).reshape(batch, length, -1)
        pairwise = model.outer_product_mean(seq)
        pairwise = pairwise + model.pos_encoder(seq)
        # Kaggle RibonanzaNet2 alpha packs each layer input as
        # ``[seq, pairwise, src_mask, return_aw]``; older verified snapshots used
        # the simpler ``layer(seq, pairwise)`` call.  Keep both paths so the
        # exporter remains tied to the checkpoint's own ``Network.py``.
        src_mask = torch.ones((batch, length), dtype=torch.long, device=src.device)
        packed_layer_api = False
        try:
            import inspect

            first_layer = model.transformer_encoder[0]
            packed_layer_api = len(inspect.signature(first_layer.forward).parameters) == 1
        except Exception:
            packed_layer_api = False
        for layer in model.transformer_encoder:
            if packed_layer_api:
                seq, pairwise = layer([seq, pairwise, src_mask, False])
            else:
                seq, pairwise = layer(seq, pairwise)
        react = None
        decoder = getattr(model, "decoder", None)
        if decoder is not None:
            react = decoder(seq)
        return seq, pairwise, react


def _coerce_scalar(text: str) -> object:
    """Coerce a YAML scalar string to int/float/bool/str.

    Complexity: O(len(text)).
    """

    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")


def _tensor_to_ndarray(tensor: object) -> NdArray:
    """Convert a detached float torch tensor to a float32 :class:`NdArray`.

    Complexity: O(number of elements).
    """

    nested = tensor.detach().to("cpu").float().tolist()
    return NdArray.from_nested(nested, kind="float32")


def build_provenance(
    *,
    model_name: str,
    model_version: str,
    weights_sha256: str,
    d_single: int,
    d_pair: Optional[int],
    n_probe: Optional[int],
    notes: str,
) -> FrozenFeatureProvenance:
    """Assemble the shard provenance header.

    Complexity: O(1).
    """

    return FrozenFeatureProvenance(
        model_name=model_name,
        model_version=model_version,
        weights_sha256=weights_sha256,
        produced_by=f"export_frozen_features.py@{Path(__file__).name}",
        date=_dt.date.today().isoformat(),
        schema=default_schema(d_single=d_single, d_pair=d_pair, n_probe=n_probe),
        notes=notes,
    )


def export(
    sequences: Sequence[Tuple[str, str, Optional[str]]],
    backend: object,
    provenance: FrozenFeatureProvenance,
    out_dir: Path,
    *,
    compress: bool = False,
) -> FrozenFeatureProvenance:
    """Encode every sequence with ``backend`` and write the shard.

    Complexity: O(sum over records of per-record encode cost).
    """

    records: List[FrozenFeatureRecord] = []
    for record_id, sequence, family in sequences:
        arrays = backend.encode(record_id, sequence)  # type: ignore[attr-defined]
        records.append(
            FrozenFeatureRecord(record_id=record_id, sequence=sequence, arrays=arrays, family=family)
        )
    return write_frozen_shard(out_dir, records, provenance, compress=compress)


def _hash_shard_payload(features_path: Path, index_path: Path) -> str:
    """Return SHA-256 over ``features.npz`` followed by ``index.jsonl``.

    Complexity: O(total shard bytes), O(1) memory.
    """

    digest = hashlib.sha256()
    for path in (features_path, index_path):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _schema_json_obj(provenance: FrozenFeatureProvenance) -> Dict[str, Dict[str, object]]:
    """Return a comparable JSON-style copy of a provenance schema.

    Complexity: O(number of arrays).
    """

    return {name: dict(spec) for name, spec in provenance.schema.items()}


def _existing_shard_summary(
    shard_dir: Path,
    *,
    expected_record_count: int,
    expected_provenance: FrozenFeatureProvenance,
) -> Optional[dict]:
    """Return manifest entry for a complete matching shard, else ``None``.

    This is intentionally lighter than :func:`reactflow.frozen.read_frozen_shard`:
    resume only needs to prove that the on-disk shard is complete and matches
    the requested model/schema contract before skipping it.

    Complexity: O(total shard bytes) for hash verification.
    """

    provenance_path = shard_dir / "provenance.json"
    features_path = shard_dir / "features.npz"
    index_path = shard_dir / "index.jsonl"
    if not (provenance_path.exists() and features_path.exists() and index_path.exists()):
        return None
    if features_path.stat().st_size <= 0 or index_path.stat().st_size <= 0:
        return None
    try:
        provenance = FrozenFeatureProvenance.from_json_obj(
            json.loads(provenance_path.read_text(encoding="utf-8"))
        )
    except Exception:
        return None
    if provenance.record_count != expected_record_count:
        return None
    if provenance.model_name != expected_provenance.model_name:
        return None
    if provenance.model_version != expected_provenance.model_version:
        return None
    if provenance.weights_sha256 != expected_provenance.weights_sha256:
        return None
    if _schema_json_obj(provenance) != _schema_json_obj(expected_provenance):
        return None
    if not provenance.content_sha256:
        return None
    try:
        line_count = 0
        with index_path.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                json.loads(raw)
                line_count += 1
        if line_count != expected_record_count:
            return None
        if _hash_shard_payload(features_path, index_path) != provenance.content_sha256:
            return None
    except Exception:
        return None
    return {
        "path": shard_dir.name,
        "record_count": provenance.record_count,
        "content_sha256": provenance.content_sha256,
        "weights_sha256": provenance.weights_sha256,
    }


def _encode_chunk(
    backend: object,
    chunk: Sequence[Tuple[str, str, Optional[str]]],
    *,
    batch_size: int,
) -> List[Dict[str, NdArray]]:
    """Encode one shard chunk, using a backend batch path when available.

    Complexity: delegated to the backend.
    """

    encode_many = getattr(backend, "encode_many", None)
    if batch_size > 1 and callable(encode_many):
        arrays = encode_many(chunk, batch_size=batch_size)
        if len(arrays) != len(chunk):
            raise RuntimeError(f"backend returned {len(arrays)} arrays for {len(chunk)} records")
        return list(arrays)
    return [backend.encode(record_id, sequence) for record_id, sequence, _ in chunk]  # type: ignore[attr-defined]


def export_sharded(
    sequences: Sequence[Tuple[str, str, Optional[str]]],
    backend: object,
    provenance: FrozenFeatureProvenance,
    out_dir: Path,
    *,
    shard_size: int,
    compress: bool = False,
    resume: bool = False,
    batch_size: int = 1,
) -> dict:
    """Encode sequences into multiple bounded-size shard directories.

    Each child directory is a normal ReactFlow frozen shard, so existing readers
    can validate and load it independently.  The parent directory receives a
    lightweight ``sharded_manifest.json`` with per-shard provenance hashes.

    Complexity: O(sum over records of per-record encode cost); peak memory is
    bounded by one shard's records rather than the whole input.
    """

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    total = 0
    for shard_index, start in enumerate(range(0, len(sequences), shard_size)):
        chunk = sequences[start : start + shard_size]
        shard_dir = out_dir / f"shard_{shard_index:05d}"
        if resume:
            existing = _existing_shard_summary(
                shard_dir,
                expected_record_count=len(chunk),
                expected_provenance=provenance,
            )
            if existing is not None:
                print(f"resume: skipping complete {shard_dir.name}", file=sys.stderr, flush=True)
                shards.append(existing)
                total += int(existing["record_count"])
                continue
        print(
            f"export: writing {shard_dir.name} records={len(chunk)} batch_size={batch_size}",
            file=sys.stderr,
            flush=True,
        )
        encoded_arrays = _encode_chunk(backend, chunk, batch_size=batch_size)
        records: List[FrozenFeatureRecord] = []
        for (record_id, sequence, family), arrays in zip(chunk, encoded_arrays):
            records.append(
                FrozenFeatureRecord(record_id=record_id, sequence=sequence, arrays=arrays, family=family)
            )
        finalized = write_frozen_shard(shard_dir, records, provenance, compress=compress)
        print(
            f"export: wrote {shard_dir.name} records={finalized.record_count}",
            file=sys.stderr,
            flush=True,
        )
        shards.append(
            {
                "path": shard_dir.name,
                "record_count": finalized.record_count,
                "content_sha256": finalized.content_sha256,
                "weights_sha256": finalized.weights_sha256,
            }
        )
        total += finalized.record_count
    manifest = {
        "layout": "reactflow-sharded-frozen-v1",
        "model_name": provenance.model_name,
        "model_version": provenance.model_version,
        "weights_sha256": provenance.weights_sha256,
        "record_count": total,
        "shard_count": len(shards),
        "shard_size": shard_size,
        "shards": shards,
    }
    (out_dir / "sharded_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser.

    Complexity: O(1).
    """

    parser = argparse.ArgumentParser(description="Stage-A frozen-feature exporter (ReactFlow C5.2)")
    parser.add_argument("--sequences", required=True, type=Path, help="JSONL of {sequence,id,family}")
    parser.add_argument("--out", required=True, type=Path, help="output shard directory")
    parser.add_argument("--backend", choices=("dry-run", "torch"), default="dry-run")
    parser.add_argument("--model", default="RibonanzaNet2", help="model name for provenance")
    parser.add_argument("--model-version", default="alpha-v1", help="model version tag")
    parser.add_argument("--limit", type=int, default=None, help="cap number of sequences")
    parser.add_argument("--d-single", type=int, default=384, help="per-nucleotide dim (dry-run)")
    parser.add_argument("--d-pair", type=int, default=128, help="pairwise dim (dry-run; 0 to omit)")
    parser.add_argument("--n-probe", type=int, default=2, help="reactivity probes (dry-run; 0 to omit)")
    parser.add_argument("--seed", type=int, default=0, help="dry-run master seed")
    parser.add_argument("--compress", action="store_true", help="DEFLATE the NPZ archive")
    parser.add_argument("--shard-size", type=int, default=0, help="write child shards of at most this many records (0 disables)")
    parser.add_argument("--resume", action="store_true", help="skip complete matching shards when --shard-size is set")
    parser.add_argument("--batch-size", type=int, default=1, help="torch same-length mini-batch size for sharded export")
    # torch backend
    parser.add_argument("--network-dir", type=Path, default=None, help="dir with Network.py + weights")
    parser.add_argument("--config", type=Path, default=None, help="model config YAML")
    parser.add_argument("--weights", type=Path, default=None, help="model weights .pt")
    parser.add_argument("--device", default="cpu", help="torch device")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns a process exit code.

    Complexity: dominated by :func:`export`.
    """

    args = build_arg_parser().parse_args(argv)
    d_pair = None if args.d_pair in (0, None) else args.d_pair
    n_probe = None if args.n_probe in (0, None) else args.n_probe
    sequences = read_sequences(args.sequences, limit=args.limit)

    if args.backend == "dry-run":
        backend: object = DryRunBackend(d_single=args.d_single, d_pair=d_pair, n_probe=n_probe, seed=args.seed)
        provenance = build_provenance(
            model_name=args.model,
            model_version=args.model_version,
            weights_sha256="",
            d_single=args.d_single,
            d_pair=d_pair,
            n_probe=n_probe,
            notes="DRY RUN: deterministic random features, NOT real model weights",
        )
    else:
        missing = [name for name, value in (("--network-dir", args.network_dir), ("--config", args.config), ("--weights", args.weights)) if value is None]
        if missing:
            raise SystemExit(f"--backend torch requires {', '.join(missing)}")
        backend = TorchRibonanzaBackend(
            network_dir=args.network_dir,
            config_path=args.config,
            weights_path=args.weights,
            device=args.device,
            include_pair=d_pair is not None,
            include_react=n_probe is not None,
        )
        d_single_real = getattr(backend, "d_single", args.d_single)
        provenance = build_provenance(
            model_name=args.model,
            model_version=args.model_version,
            weights_sha256=backend.weights_sha256,  # type: ignore[attr-defined]
            d_single=d_single_real,
            d_pair=d_pair,
            n_probe=n_probe,
            notes=f"real weights sha256={backend.weights_sha256[:12]}...",  # type: ignore[attr-defined]
        )

    if args.shard_size:
        manifest = export_sharded(
            sequences,
            backend,
            provenance,
            args.out,
            shard_size=args.shard_size,
            compress=args.compress,
            resume=args.resume,
            batch_size=args.batch_size,
        )
        summary = {
            "backend": args.backend,
            "records": manifest["record_count"],
            "shard_count": manifest["shard_count"],
            "shard_size": manifest["shard_size"],
            "weights_sha256": manifest["weights_sha256"],
            "out": str(args.out),
            "notes": provenance.notes,
        }
    else:
        finalized = export(sequences, backend, provenance, args.out, compress=args.compress)
        summary = {
            "backend": args.backend,
            "records": finalized.record_count,
            "content_sha256": finalized.content_sha256,
            "weights_sha256": finalized.weights_sha256,
            "out": str(args.out),
            "notes": finalized.notes,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
